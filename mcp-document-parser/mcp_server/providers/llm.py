from __future__ import annotations

import json
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


def _json_extract(text: str) -> Any:
    """Best-effort JSON extractor.

    Models sometimes return fenced code blocks or extra prose. We try:
      1) direct json.loads
      2) fenced ```json ...```
      3) first {...} block
    """

    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty response")

    # 1) direct
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) first object block
    m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output")


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
        self.timeout_s = float(timeout_s)
        self._client = httpx.AsyncClient(timeout=self.timeout_s, headers=self._base_headers())

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

        base = _strip_trailing_slash(self.profile.azure_endpoint or "")
        dep = self.profile.azure_deployment
        ver = self.profile.azure_api_version
        return f"{base}/openai/deployments/{dep}/chat/completions?api-version={ver}"

    def _embeddings_url(self) -> str:
        # Prefer explicit embeddings URL (e.g., Ollama native /api/embed)
        if getattr(self.profile, "embeddings_url", None):
            return _strip_trailing_slash(self.profile.embeddings_url)  # type: ignore[arg-type]

        if self.profile.provider == "openai_compatible":
            base = _strip_trailing_slash(self.profile.base_url or "")
            return f"{base}/embeddings"

        base = _strip_trailing_slash(self.profile.azure_endpoint or "")
        dep = getattr(self.profile, "azure_embeddings_deployment", None) or self.profile.azure_deployment
        ver = getattr(self.profile, "azure_embeddings_api_version", None) or self.profile.azure_api_version
        return f"{base}/openai/deployments/{dep}/embeddings?api-version={ver}"

    def _default_chat_model(self) -> Optional[str]:
        return self.profile.model

    def _default_embedding_model(self) -> Optional[str]:
        # Prefer dedicated embedding model if provided.
        return getattr(self.profile, "embedding_model", None) or self.profile.model

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(url, json=payload)
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e

        if resp.status_code >= 400:
            body = resp.text
            snippet = body[:1000] + ("…" if len(body) > 1000 else "")
            model_used = payload.get("model")
            raise LLMError(
                f"LLM error {resp.status_code} url={url} model={model_used!r}: {snippet}"
            )

        try:
            return resp.json()
        except Exception as e:
            raise LLMError(f"LLM returned non-JSON response: {e}; body={resp.text[:500]}") from e

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
        }
        if model_id:
            payload["model"] = model_id

        data = await self._post_json(url, payload)

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
        if model_id:
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