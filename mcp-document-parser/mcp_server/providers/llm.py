from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import httpx
from pydantic import BaseModel

from ..config import ModelProfile, get_settings


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _strip_trailing_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _normalize_azure_endpoint(endpoint: str) -> tuple[str, bool]:
    """Normalize Azure OpenAI endpoint and detect OpenAI v1-style routing.

    Accepts endpoints like:
      - https://{resource}.openai.azure.com
      - https://{resource}.openai.azure.com/openai
      - https://{resource}.openai.azure.com/openai/v1

    Returns:
      (normalized_endpoint, is_v1)

    Notes:
      - For v1 endpoints, Azure still requires `api-version` as a query parameter.
    """

    ep = _strip_trailing_slash(endpoint)
    if not ep:
        return "", False

    # Detect /openai/v1 style base URLs.
    if re.search(r"/openai/v1$", ep, flags=re.IGNORECASE):
        return ep, True

    # Strip a trailing /openai if someone included it.
    ep = re.sub(r"/openai$", "", ep, flags=re.IGNORECASE)
    ep = _strip_trailing_slash(ep)
    return ep, False


def _header_request_ids(headers: httpx.Headers) -> Dict[str, str]:
    """Extract common request/correlation IDs for easier debugging."""
    keys = [
        "x-ms-request-id",
        "x-request-id",
        "apim-request-id",
        "x-correlation-id",
        "traceparent",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = headers.get(k)
        if v:
            out[k] = v
    return out


def _parse_retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        # Most common: integer seconds
        return float(str(ra).strip())
    except Exception:
        return None


async def _sleep_backoff(attempt: int, *, base_s: float, max_s: float, retry_after_s: Optional[float] = None) -> None:
    if retry_after_s is not None and retry_after_s > 0:
        await asyncio.sleep(min(float(retry_after_s), float(max_s)))
        return

    # Exponential backoff with jitter.
    exp = float(base_s) * (2.0 ** max(0, int(attempt)))
    jitter = 0.5 + random.random()  # 0.5..1.5
    await asyncio.sleep(min(exp * jitter, float(max_s)))


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


def _to_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class LLMClient:
    """Profile-based LLM client.

    Supports:
      - openai_compatible endpoints (Ollama / vLLM / gateways): base_url + model
      - azure_openai deployments: azure_endpoint + api_version + deployment

    Embeddings:
      - If profile.embeddings_url is set (e.g., Ollama native /api/embed), it is preferred.
      - Otherwise use OpenAI-style /embeddings endpoint.

    Ollama native embeddings endpoint shape:
      POST /api/embed {"model": "nomic-embed-text:latest", "input": "..."}
      -> {"embeddings": [[...], ...]}
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        timeout_s: float = 60.0,
    ) -> None:
        self.profile = profile

        # Allow future config-driven tuning without breaking older profiles.
        self.timeout_s = float(getattr(profile, "timeout_s", None) or getattr(profile, "timeout_seconds", None) or timeout_s)
        self.max_retries = int(getattr(profile, "max_retries", None) or 3)
        self.retry_backoff_base_s = float(getattr(profile, "retry_backoff_base_s", None) or 0.5)
        self.retry_backoff_max_s = float(getattr(profile, "retry_backoff_max_s", None) or 8.0)

        # Azure endpoint normalization + v1 detection (OpenAI v1-style routing).
        self._azure_base: str = ""
        self._azure_is_v1: bool = False
        if self.profile.provider == "azure_openai":
            self._azure_base, self._azure_is_v1 = _normalize_azure_endpoint(self.profile.azure_endpoint or "")

        max_connections = int(getattr(profile, "max_connections", None) or 20)
        max_keepalive = int(getattr(profile, "max_keepalive_connections", None) or 20)
        keepalive_expiry = float(getattr(profile, "keepalive_expiry_s", None) or 30.0)

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )

        # TLS / proxy behavior
        # - Some enterprise environments use TLS interception; allow disabling verification or providing a CA bundle.
        # - Keep this opt-in via env vars so local dev remains secure by default.
        verify: bool | str = True
        follow_redirects = True
        trust_env = True  # respect HTTPS_PROXY/HTTP_PROXY/NO_PROXY if set

        if self.profile.provider == "azure_openai":
            # Prefer a CA bundle over disabling verification.
            ca_bundle = (
                os.getenv("AZURE_OPENAI_CA_BUNDLE", "").strip()
                or os.getenv("AZURE_CA_BUNDLE", "").strip()
                or os.getenv("REQUESTS_CA_BUNDLE", "").strip()
                or os.getenv("SSL_CERT_FILE", "").strip()
            )
            if ca_bundle:
                verify = ca_bundle
            else:
                # Explicit opt-out switch (use only when you cannot install the corp CA bundle).
                v = (
                    os.getenv("AZURE_OPENAI_SSL_VERIFY", "").strip()
                    or os.getenv("AZURE_SSL_VERIFY", "").strip()
                    or os.getenv("SSL_VERIFY", "").strip()
                )
                if v.lower() in ("0", "false", "no", "off"):
                    verify = False

        self._client = httpx.AsyncClient(
            timeout=self.timeout_s,
            headers=self._base_headers(),
            limits=limits,
            verify=verify,
            follow_redirects=follow_redirects,
            trust_env=trust_env,
        )

    def _base_headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}

        # Merge profile headers
        for k, v in (self.profile.headers or {}).items():
            if v is None:
                continue
            h[str(k)] = str(v)

        # Auth
        if self.profile.provider == "openai_compatible":
            # Standard OpenAI: Authorization bearer. Ollama ignores it but OK.
            if self.profile.api_key:
                h.setdefault("Authorization", f"Bearer {self.profile.api_key}")
        else:
            # Azure OpenAI uses api-key header
            if self.profile.api_key:
                h.setdefault("api-key", self.profile.api_key)

        return h

    async def aclose(self) -> None:
        await self._client.aclose()

    # -------------------------
    # Endpoints
    # -------------------------

    def _chat_url(self) -> str:
        if self.profile.provider == "openai_compatible":
            base = _strip_trailing_slash(self.profile.base_url or "")
            return f"{base}/chat/completions"

        base = self._azure_base or _strip_trailing_slash(self.profile.azure_endpoint or "")
        dep = self.profile.azure_deployment
        ver = self.profile.azure_api_version

        # Azure OpenAI v1 routing uses OpenAI-style paths under /openai/v1, but still requires api-version.
        if self._azure_is_v1:
            return f"{base}/chat/completions?api-version={ver}"

        # Legacy Azure OpenAI routing uses the deployment in the URL.
        return f"{base}/openai/deployments/{dep}/chat/completions?api-version={ver}"

    def _embeddings_url(self) -> str:
        # Prefer explicit embeddings URL (e.g., Ollama native /api/embed)
        if getattr(self.profile, "embeddings_url", None):
            return _strip_trailing_slash(self.profile.embeddings_url)  # type: ignore[arg-type]

        if self.profile.provider == "openai_compatible":
            base = _strip_trailing_slash(self.profile.base_url or "")
            return f"{base}/embeddings"

        base = self._azure_base or _strip_trailing_slash(self.profile.azure_endpoint or "")

        dep = getattr(self.profile, "azure_embeddings_deployment", None)
        if not dep or not str(dep).strip():
            raise LLMError(
                "Azure embeddings deployment is not configured. Set azure_embeddings_deployment (recommended: an Ada-002 deployment)."
            )

        ver = getattr(self.profile, "azure_embeddings_api_version", None) or self.profile.azure_api_version

        # Azure OpenAI v1 routing uses OpenAI-style paths under /openai/v1, but still requires api-version.
        if self._azure_is_v1:
            return f"{base}/embeddings?api-version={ver}"

        # Legacy Azure OpenAI routing uses the deployment in the URL.
        return f"{base}/openai/deployments/{dep}/embeddings?api-version={ver}"

    def _default_chat_model(self) -> Optional[str]:
        # For Azure v1 routing, Azure expects the *deployment name* in the payload `model` field.
        if self.profile.provider == "azure_openai" and self._azure_is_v1:
            return (self.profile.model or self.profile.azure_deployment)
        return self.profile.model

    def _default_embedding_model(self) -> Optional[str]:
        # Prefer dedicated embedding model if provided.
        emb_model = getattr(self.profile, "embedding_model", None) or self.profile.model
        # For Azure v1 routing, default to using the embeddings deployment name.
        if self.profile.provider == "azure_openai" and self._azure_is_v1:
            return emb_model or getattr(self.profile, "azure_embeddings_deployment", None)
        return emb_model

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None

        for attempt in range(0, max(0, int(self.max_retries)) + 1):
            try:
                resp = await self._client.post(url, json=payload)
            except Exception as e:
                last_exc = e
                # Retry network-ish errors.
                if attempt < int(self.max_retries):
                    await _sleep_backoff(
                        attempt,
                        base_s=self.retry_backoff_base_s,
                        max_s=self.retry_backoff_max_s,
                    )
                    continue
                raise LLMError(f"LLM request failed after retries: {e}") from e

            # Retry on transient status codes.
            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < int(self.max_retries):
                retry_after = _parse_retry_after_seconds(resp.headers)
                await _sleep_backoff(
                    attempt,
                    base_s=self.retry_backoff_base_s,
                    max_s=self.retry_backoff_max_s,
                    retry_after_s=retry_after,
                )
                continue

            if resp.status_code >= 400:
                model_used = payload.get("model")

                # Best-effort parse of common error envelope.
                err_code: Optional[str] = None
                err_msg: Optional[str] = None
                body_text = resp.text or ""
                try:
                    j = resp.json()
                    if isinstance(j, dict):
                        e = j.get("error")
                        if isinstance(e, dict):
                            if isinstance(e.get("code"), str):
                                err_code = e.get("code")
                            if isinstance(e.get("message"), str):
                                err_msg = e.get("message")
                        # Some gateways use {"message": "..."}
                        if err_msg is None and isinstance(j.get("message"), str):
                            err_msg = j.get("message")
                except Exception:
                    pass

                req_ids = _header_request_ids(resp.headers)
                req_ids_str = " ".join([f"{k}={v}" for k, v in req_ids.items()])

                snippet = body_text[:1200] + ("…" if len(body_text) > 1200 else "")
                core = err_msg or snippet

                raise LLMError(
                    f"LLM error {resp.status_code} url={url} provider={self.profile.provider} model={model_used!r} "
                    f"code={err_code!r} {req_ids_str}: {core}"
                )

            try:
                return resp.json()
            except Exception as e:
                ct = (resp.headers.get("content-type") or "").strip()
                loc = (resp.headers.get("location") or "").strip()
                req_ids = _header_request_ids(resp.headers)
                req_ids_str = " ".join([f"{k}={v}" for k, v in req_ids.items()])
                snippet = (resp.text or "")[:800]
                raise LLMError(
                    f"LLM returned non-JSON response: {e}; status={resp.status_code} content_type={ct!r} "
                    f"location={loc!r} {req_ids_str}; body={snippet}"
                ) from e

        # Should not happen, but be defensive.
        raise LLMError(f"LLM request failed after retries: {last_exc}")

    # -------------------------
    # Chat
    # -------------------------

    async def chat_text(
        self,
        *,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 12000,
        model: Optional[str] = None,
    ) -> str:
        url = self._chat_url()
        model_id = model or self._default_chat_model()
        if isinstance(model_id, str):
            model_id = model_id.strip() or None

        payload: Dict[str, Any] = {
            "messages": _to_openai_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if model_id:
            payload["model"] = model_id

        data = await self._post_json(url, payload)

        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            raise LLMError(f"Unexpected chat response shape: {e}; data_keys={list(data.keys())}") from e

    async def chat_object(
        self,
        *,
        messages: List[ChatMessage],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 15000,
        model: Optional[str] = None,
    ) -> T:
        # Force JSON-only output.
        constraint = ChatMessage(
            role="system",
            content=(
                "Return ONLY a single JSON object that matches the required schema. "
                "Do not include markdown fences, explanations, or extra keys unless required."
            ),
        )

        url = self._chat_url()
        model_id = model or self._default_chat_model()
        if isinstance(model_id, str):
            model_id = model_id.strip() or None

        payload: Dict[str, Any] = {
            "messages": _to_openai_messages([constraint, *messages]),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            # Prefer JSON mode when supported; we fall back automatically if a provider rejects it.
            "response_format": {"type": "json_object"},
        }
        if model_id:
            payload["model"] = model_id

        try:
            data = await self._post_json(url, payload)
        except LLMError as e:
            # Some providers (or older API versions) reject response_format.
            msg = str(e).lower()
            if "response_format" in msg or "unsupported" in msg or "unrecognized" in msg:
                payload.pop("response_format", None)
                data = await self._post_json(url, payload)
            else:
                raise

        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            raise LLMError(f"Unexpected chat response shape: {e}; data_keys={list(data.keys())}") from e

        try:
            obj = _json_extract(text)
        except Exception as e:
            raise LLMError(f"Failed to parse JSON from model output: {e}; raw={text[:800]}") from e

        try:
            return schema.model_validate(obj)
        except Exception as e:
            raise LLMError(f"Model output failed schema validation: {e}; obj={str(obj)[:800]}") from e

    # -------------------------
    # Embeddings
    # -------------------------

    async def embed_texts(
        self,
        texts: Union[str, List[str]],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        url = self._embeddings_url()
        model_id = model or self._default_embedding_model()
        if isinstance(model_id, str):
            model_id = model_id.strip() or None

        # Ollama native endpoint
        if "/api/embed" in url:
            if not model_id:
                raise LLMError("No embedding model configured for Ollama /api/embed")
            payload: Dict[str, Any] = {
                "model": model_id,
                "input": texts,
            }
            data = await self._post_json(url, payload)
            embs = data.get("embeddings")
            if not isinstance(embs, list):
                raise LLMError(f"Unexpected Ollama embeddings response: keys={list(data.keys())}")
            return [[float(v) for v in vec] for vec in embs]

        # OpenAI-style embeddings endpoint
        payload: Dict[str, Any] = {
            "input": texts,
        }

        # For Azure legacy embeddings endpoints (deployment in URL), do NOT send model unless explicitly provided.
        if model_id and not (self.profile.provider == "azure_openai" and not self._azure_is_v1):
            payload["model"] = model_id

        data = await self._post_json(url, payload)

        try:
            items = data["data"]
            items = sorted(items, key=lambda x: x.get("index", 0))
            return [[float(v) for v in it["embedding"]] for it in items]
        except Exception as e:
            raise LLMError(f"Unexpected embeddings response shape: {e}; data_keys={list(data.keys())}") from e


# -------------------------
# Profile resolution + singleton cache
# -------------------------


_CLIENTS: Dict[str, LLMClient] = {}


def resolve_profile_name(requested: str) -> str:
    """Map logical names (chat/assessment) to configured profile keys."""
    settings = get_settings()
    req = (requested or "").strip()
    if not req:
        return settings.default_chat_profile
    if req == "chat":
        return settings.default_chat_profile
    if req == "assessment":
        return settings.default_assessment_profile
    return req


def get_llm(profile: str, *, override: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Get an LLM client for a given profile."""

    settings = get_settings()
    key = resolve_profile_name(profile)

    base_profile = settings.model_profiles.get(key)
    if base_profile is None:
        raise LLMError(f"Unknown model profile '{key}'. Available: {list(settings.model_profiles.keys())}")

    prof = base_profile
    if override:
        if not settings.allow_model_override:
            raise LLMError("Model override is disabled by server configuration")

        allowed = {
            "provider",
            "base_url",
            "api_key",
            "model",
            "embeddings_url",
            "embedding_model",
            "azure_endpoint",
            "azure_api_version",
            "azure_deployment",
            "azure_embeddings_deployment",
            "azure_embeddings_api_version",
            "headers",
        }
        upd = {k: v for k, v in override.items() if k in allowed}
        prof = base_profile.model_copy(update=upd)

    cache_key = key
    if override:
        cache_key = key + ":override:" + str(hash(json.dumps(override, sort_keys=True)))

    client = _CLIENTS.get(cache_key)
    if client is not None:
        return client

    client = LLMClient(prof)
    _CLIENTS[cache_key] = client
    return client


async def aclose_all_llm_clients() -> None:
    """Close all cached clients (useful for tests/shutdown)."""
    for c in list(_CLIENTS.values()):
        try:
            await c.aclose()
        except Exception:
            pass
    _CLIENTS.clear()