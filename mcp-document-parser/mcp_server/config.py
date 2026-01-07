from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _s(v: Optional[str]) -> Optional[str]:
    """Strip whitespace from optional strings."""
    if v is None:
        return None
    vv = str(v).strip()
    return vv if vv else None


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
        return self


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # -----------------
    # Server
    # -----------------
    host: str = Field(default="0.0.0.0", alias="MCP_SERVER_HOST")
    port: int = Field(default=8765, alias="MCP_SERVER_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # -----------------
    # Policy DB / RAG
    # -----------------
    policy_db_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/pgvector_mcp_document",
        alias="POLICY_DB_URL",
    )
    policy_default_collection: str = Field(default="default", alias="POLICY_DEFAULT_COLLECTION")
    policy_top_k_default: int = Field(default=3, alias="POLICY_TOP_K_DEFAULT")

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
    embeddings_model_profile: str = Field(default="chat", alias="EMBEDDINGS_MODEL_PROFILE")
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

    # Parsed profiles
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _build_profiles(self) -> "Settings":
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
                    "azure_embeddings_deployment": _s(self.azure_openai_embeddings_deployment) or _s(self.azure_openai_deployment) or self.azure_openai_deployment,
                    "azure_embeddings_api_version": _s(self.azure_openai_embeddings_api_version) or _s(self.azure_openai_api_version) or self.azure_openai_api_version,
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

        self.model_profiles = built
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
