from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _s(v: Optional[str]) -> Optional[str]:
    """Strip whitespace from optional strings."""
    if v is None:
        return None
    vv = str(v).strip()
    return vv if vv else None


def _normalize_prompt_text(text: str) -> str:
    """Normalize prompt text loaded from env/file.

    - Strips surrounding whitespace.
    - Converts literal '\\n' sequences (common in env files) into real newlines.
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Support env-file friendly newline encoding.
    t = t.replace("\\n", "\n")
    return t


def _read_text_file(path_str: str) -> str:
    p = Path(path_str).expanduser()
    return p.read_text(encoding="utf-8")


def _resolve_prompt(inline_value: Optional[str], file_path: Optional[str]) -> tuple[Optional[str], str]:
    """Resolve a prompt from either a file path or an inline env value.

    Precedence: file_path > inline_value > None
    Returns: (prompt_text_or_None, source) where source is one of: 'file' | 'env' | 'default'
    """
    fp = _s(file_path)
    iv = _s(inline_value)

    if fp:
        try:
            return _normalize_prompt_text(_read_text_file(fp)), "file"
        except Exception as e:
            raise ValueError(f"Failed to read prompt file '{fp}': {e}")

    if iv:
        return _normalize_prompt_text(iv), "env"

    return None, "default"


class ModelProfile(BaseModel):
    """A named model configuration profile.

    Supports:
      - openai_compatible endpoints (Ollama / vLLM / gateways)
      - azure_openai deployments

    Also supports a dedicated embeddings endpoint + model (e.g., Ollama /api/embed)
    and optional separate Azure embeddings deployment/version.
    """

    provider: Literal["openai_compatible", "azure_openai"] = "openai_compatible"

    # openai_compatible (chat/completions)
    base_url: Optional[str] = None  # e.g. http://localhost:11434/v1
    api_key: Optional[str] = None
    model: Optional[str] = None

    # openai_compatible (embeddings)
    embeddings_url: Optional[str] = None  # e.g. http://localhost:11434/api/embed
    embedding_model: Optional[str] = None  # e.g. nomic-embed-text:latest

    # azure_openai (chat/completions)
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_deployment: Optional[str] = None

    # azure_openai (embeddings) - optionally separate deployment/version
    azure_embeddings_deployment: Optional[str] = None
    azure_embeddings_api_version: Optional[str] = None

    # -----------------
    # Optional runtime tuning (used by providers/llm.py)
    # -----------------
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    retry_backoff_base_s: Optional[float] = None
    retry_backoff_max_s: Optional[float] = None

    max_connections: Optional[int] = None
    max_keepalive_connections: Optional[int] = None
    keepalive_expiry_s: Optional[float] = None

    # Optional headers passthrough (enterprise proxies, etc.)
    headers: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_provider(self) -> "ModelProfile":
        if self.provider == "openai_compatible":
            if not self.base_url:
                raise ValueError("openai_compatible profile requires base_url")
        if self.provider == "azure_openai":
            if not self.azure_endpoint or not self.azure_api_version or not self.azure_deployment:
                raise ValueError(
                    "azure_openai profile requires azure_endpoint, azure_api_version, and azure_deployment"
                )

            # Fail fast if no auth is configured. The provider layer will send either:
            #   - the `api-key` header (preferred), or
            #   - whatever the caller provided via `headers`.
            has_key = bool((self.api_key or "").strip())
            has_hdr_key = False
            try:
                has_hdr_key = bool((self.headers.get("api-key") or "").strip()) or bool((self.headers.get("Authorization") or "").strip())
            except Exception:
                has_hdr_key = False
            if not has_key and not has_hdr_key:
                raise ValueError("azure_openai profile requires api_key or headers['api-key']/headers['Authorization']")
        return self


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore", env_file=".env", env_file_encoding="utf-8")
    # -----------------
    # Server
    # -----------------
    host: str = Field(default="0.0.0.0", alias="MCP_SERVER_HOST")
    port: int = Field(default=8765, alias="MCP_SERVER_PORT")

    # Root/application log level. Use DEBUG to enable deep RAG/pgvector diagnostics.
    # Allowed: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Uvicorn log level (optional). If unset/blank, defaults to LOG_LEVEL (lower-cased).
    uvicorn_log_level: Optional[str] = Field(default=None, alias="UVICORN_LOG_LEVEL")

    # -----------------
    # Policy DB / RAG
    # -----------------
    policy_db_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/pgvector_mcp_document",
        alias="POLICY_DB_URL",
    )
    policy_default_collection: str = Field(default="default", alias="POLICY_DEFAULT_COLLECTION")
    policy_top_k_default: int = Field(default=3, alias="POLICY_TOP_K_DEFAULT")

    # When true, if we cannot extract a clause_number from the clause label/title,
    # the risk-assessment workflow will derive a numeric clause_number from the internal
    # clause_id token (e.g., clause_0004_xxx -> "4"). This helps avoid termset-only fallback.
    rag_use_clause_id_fallback: bool = Field(default=True, alias="RAG_USE_CLAUSE_ID_FALLBACK")

    # -----------------
    # Risk assessment prompts
    # -----------------
    # These allow you to override the default system prompts used by the risk assessment workflow.
    # Prefer *_FILE for long prompts; it is easier to manage and avoids shell/env escaping issues.

    # Clause-by-clause system prompt
    risk_assessment_system_prompt: Optional[str] = Field(default=None, alias="RISK_ASSESSMENT_SYSTEM_PROMPT")
    risk_assessment_system_prompt_file: Optional[str] = Field(default=None, alias="RISK_ASSESSMENT_SYSTEM_PROMPT_FILE")
    # Where the prompt came from: file | env | default
    risk_assessment_system_prompt_source: Literal["file", "env", "default"] = "default"

    # Final summary system prompt
    risk_summary_system_prompt: Optional[str] = Field(default=None, alias="RISK_SUMMARY_SYSTEM_PROMPT")
    risk_summary_system_prompt_file: Optional[str] = Field(default=None, alias="RISK_SUMMARY_SYSTEM_PROMPT_FILE")
    # Where the prompt came from: file | env | default
    risk_summary_system_prompt_source: Literal["file", "env", "default"] = "default"

    # -----------------
    # Model profiles
    # -----------------
    model_profiles_json: Optional[str] = Field(default=None, alias="MODEL_PROFILES_JSON")

    default_chat_profile: str = Field(default="chat", alias="DEFAULT_CHAT_MODEL_PROFILE")
    default_assessment_profile: str = Field(default="assessment", alias="DEFAULT_ASSESSMENT_MODEL_PROFILE")

    allow_model_override: bool = Field(default=False, alias="ALLOW_MODEL_OVERRIDE")

    # -----------------
    # Local Ollama defaults
    # -----------------
    # If set to Ollama native API (e.g., http://localhost:11434/api), it will be converted to /v1.
    ollama_base_url: str = Field(default="http://localhost:11434/api", alias="OLLAMA_BASE_URL")
    ollama_openai_base_url: Optional[str] = Field(default=None, alias="OLLAMA_OPENAI_BASE_URL")
    # Note: whitespace is stripped at build time to avoid hidden spaces causing 404 errors
    ollama_model: str = Field(default="gpt-oss:20b", alias="OLLAMA_MODEL")

    # -----------------
    # Embeddings configuration
    # -----------------
    embeddings_model_profile: str = Field(default="", alias="EMBEDDINGS_MODEL_PROFILE")
    embeddings_dim: int = Field(default=768, alias="EMBEDDINGS_DIM")

    # Ollama native embeddings endpoint + model
    ollama_embeddings_url: str = Field(default="http://localhost:11434/api/embed", alias="OLLAMA_EMBEDDINGS_URL")
    ollama_embeddings_model: str = Field(default="nomic-embed-text:latest", alias="OLLAMA_EMBEDDINGS_MODEL")

    # -----------------
    # Azure OpenAI (optional)
    # -----------------
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")

    # Optional separate Azure embeddings deployment/version
    azure_openai_embeddings_deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    azure_openai_embeddings_api_version: Optional[str] = Field(default=None, alias="AZURE_OPENAI_EMBEDDINGS_API_VERSION")

    # -----------------
    # Azure TLS / cert behavior
    # -----------------
    # These are used by providers/llm.py to decide whether to verify TLS certificates
    # and/or which CA bundle to trust when calling Azure endpoints.
    azure_openai_ssl_verify: bool = Field(default=False, alias="AZURE_OPENAI_SSL_VERIFY")
    azure_openai_ca_bundle: Optional[str] = Field(default=None, alias="AZURE_OPENAI_CA_BUNDLE")

    # Parsed profiles
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _build_profiles(self) -> "Settings":
        # -----------------
        # Normalize/validate logging settings early
        # -----------------
        lvl = (self.log_level or "INFO").strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if lvl not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)} (got {self.log_level!r})")
        self.log_level = lvl

        # If UVICORN_LOG_LEVEL is not set, mirror LOG_LEVEL (uvicorn expects lower-case names).
        if self.uvicorn_log_level and str(self.uvicorn_log_level).strip():
            self.uvicorn_log_level = str(self.uvicorn_log_level).strip().lower()
        else:
            self.uvicorn_log_level = lvl.lower()
        profiles: Dict[str, Any] = {}

        # 1) Explicit JSON profiles win.
        if self.model_profiles_json and self.model_profiles_json.strip():
            try:
                profiles = json.loads(self.model_profiles_json)
            except Exception as e:
                raise ValueError(f"MODEL_PROFILES_JSON is not valid JSON: {e}")

        # 2) Defaults for local dev.
        if not profiles:
            base = (self.ollama_openai_base_url or self.ollama_base_url).strip().rstrip("/")
            if base.endswith("/api"):
                base = base[: -len("/api")] + "/v1"
            elif not base.endswith("/v1"):
                base = base + "/v1"

            profiles = {
                "chat": {
                    "provider": "openai_compatible",
                    "base_url": _s(base) or base,
                    "api_key": "ollama",
                    "model": (_s(self.ollama_model) or self.ollama_model),
                    "embeddings_url": (_s(self.ollama_embeddings_url) or self.ollama_embeddings_url),
                    "embedding_model": (_s(self.ollama_embeddings_model) or self.ollama_embeddings_model),
                }
            }

            # If Azure envs are present, use Azure for assessment; otherwise default to chat.
            if self.azure_openai_endpoint and self.azure_openai_api_version and self.azure_openai_deployment:
                profiles["assessment"] = {
                    "provider": "azure_openai",
                    "azure_endpoint": _s(self.azure_openai_endpoint) or self.azure_openai_endpoint,
                    "azure_api_version": _s(self.azure_openai_api_version) or self.azure_openai_api_version,
                    "azure_deployment": _s(self.azure_openai_deployment) or self.azure_openai_deployment,
                    "api_key": _s(self.azure_openai_api_key) or self.azure_openai_api_key,

                    # Do NOT silently fall back to the chat deployment for embeddings.
                    # Use a dedicated embeddings profile (created below) for retrieval.
                    "azure_embeddings_deployment": _s(self.azure_openai_embeddings_deployment) or None,
                    "azure_embeddings_api_version": _s(self.azure_openai_embeddings_api_version)
                    or _s(self.azure_openai_api_version)
                    or self.azure_openai_api_version,
                }

                # Dedicated embeddings profile (recommended: Ada-002 deployment).
                # This ensures we never accidentally embed with the chat deployment.
                if self.azure_openai_embeddings_deployment and str(self.azure_openai_embeddings_deployment).strip():
                    profiles["embeddings"] = {
                        "provider": "azure_openai",
                        "azure_endpoint": _s(self.azure_openai_endpoint) or self.azure_openai_endpoint,
                        "azure_api_version": _s(self.azure_openai_api_version) or self.azure_openai_api_version,

                        # Keep azure_deployment populated for validation/back-compat.
                        "azure_deployment": _s(self.azure_openai_deployment) or self.azure_openai_deployment,
                        "api_key": _s(self.azure_openai_api_key) or self.azure_openai_api_key,

                        "azure_embeddings_deployment": _s(self.azure_openai_embeddings_deployment)
                        or self.azure_openai_embeddings_deployment,
                        "azure_embeddings_api_version": _s(self.azure_openai_embeddings_api_version)
                        or _s(self.azure_openai_api_version)
                        or self.azure_openai_api_version,
                    }
            else:
                profiles["assessment"] = profiles["chat"]

        # 3) Validate/instantiate profiles.
        built: Dict[str, ModelProfile] = {}
        for name, cfg in profiles.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Model profile '{name}' must be an object")

            # Normalize whitespace on common string fields (covers MODEL_PROFILES_JSON too)
            cfg = dict(cfg)
            for k in (
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
            ):
                if k in cfg and isinstance(cfg[k], str):
                    cfg[k] = cfg[k].strip()

            built[name] = ModelProfile(**cfg)

        # -----------------
        # Auto defaults (only when user did NOT explicitly set env values)
        # -----------------
        fields_set = getattr(self, "model_fields_set", set()) or set()

        # Prefer a dedicated embeddings profile when present.
        if "embeddings_model_profile" not in fields_set:
            if "embeddings" in built:
                self.embeddings_model_profile = "embeddings"
            elif not (self.embeddings_model_profile or "").strip():
                self.embeddings_model_profile = "chat"
        else:
            # If user explicitly set it to empty, normalize to chat.
            if not (self.embeddings_model_profile or "").strip():
                self.embeddings_model_profile = "chat"

        # If user didn't set EMBEDDINGS_DIM and an Azure embeddings profile is present,
        # default to 1536 (text-embedding-ada-002).
        if "embeddings_dim" not in fields_set:
            if "embeddings" in built:
                self.embeddings_dim = 1536

        self.model_profiles = built

        # -----------------
        # Resolve risk assessment prompts from env/file
        # -----------------
        prompt, src = _resolve_prompt(self.risk_assessment_system_prompt, self.risk_assessment_system_prompt_file)
        self.risk_assessment_system_prompt = prompt
        self.risk_assessment_system_prompt_source = src  # type: ignore[assignment]

        sprompt, ssrc = _resolve_prompt(self.risk_summary_system_prompt, self.risk_summary_system_prompt_file)
        self.risk_summary_system_prompt = sprompt
        self.risk_summary_system_prompt_source = ssrc  # type: ignore[assignment]

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
