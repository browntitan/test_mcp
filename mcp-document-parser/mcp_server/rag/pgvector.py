from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..config import get_settings
from ..schemas import PolicyCitation


# -----------------------------
# Types
# -----------------------------

Embedder = Union[
    Callable[[str], Sequence[float]],
    Callable[[str], Awaitable[Sequence[float]]],
]


@dataclass(frozen=True)
class PgVectorSearchConfig:
    table: str = "policy_chunks"
    embedding_column: str = "embedding"
    text_column: str = "text"
    metadata_column: str = "metadata"
    policy_id_column: str = "policy_id"
    chunk_id_column: str = "chunk_id"
    collection_column: str = "collection"

    # Distance operator: '<=>' is cosine distance in pgvector.
    # We'll return score = 1 - distance.
    distance_operator: str = "<=>"



# -----------------------------
# Utilities
# -----------------------------


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _safe_ident(name: str, *, kind: str) -> str:
    """Validate a SQL identifier used in f-string SQL composition.

    We only allow schema-qualified identifiers like `public.table`.
    This prevents SQL injection if a caller overrides PgVectorSearchConfig.
    """
    n = (name or "").strip()
    if not n or not _IDENT_RE.match(n):
        raise ValueError(f"Invalid SQL identifier for {kind}: {name!r}")
    return n


def _coerce_embedding(vec: Any, *, expected_dim: Optional[int] = None) -> List[float]:
    """Coerce an embedder return into a 1D list[float] and optionally validate dimension.

    Supported:
      - Sequence[float]
      - Sequence[Sequence[float]] with length==1 (we flatten)

    Raises:
      - ValueError with actionable guidance on mismatch.
    """
    if vec is None:
        raise ValueError("Embedder returned None")

    # If someone accidentally passes an embedder that returns a batch (e.g., [[...]]), flatten when size==1.
    if isinstance(vec, (list, tuple)) and vec and isinstance(vec[0], (list, tuple)):
        if len(vec) == 1:
            vec = vec[0]
        else:
            raise ValueError(
                "Embedder returned a batch of embeddings, but pgvector search expects a single vector for a single query string."
            )

    if not isinstance(vec, (list, tuple)):
        raise ValueError(f"Embedder returned non-sequence embedding: {type(vec).__name__}")

    out: List[float] = []
    for x in vec:
        try:
            out.append(float(x))
        except Exception:
            raise ValueError("Embedder returned a vector containing non-numeric values")

    if expected_dim is not None and expected_dim > 0 and len(out) != int(expected_dim):
        raise ValueError(
            f"Embedding dimension mismatch: got {len(out)}, expected {int(expected_dim)}. "
            "This usually means your embeddings model/deployment changed (e.g., Ada-002 is 1536) or your database vector column was created with a different dimension. "
            "Fix by aligning EMBEDDINGS_DIM, the embeddings deployment, and the pgvector column dimension; then re-embed policy chunks."
        )

    return out


def _normalize_db_url(url: str) -> str:
    """Normalize SQLAlchemy-style URLs to psycopg conninfo URLs.

    Examples:
      - postgresql+psycopg://... -> postgresql://...
    """
    url = (url or "").strip()
    url = re.sub(r"^postgresql\+psycopg://", "postgresql://", url)
    url = re.sub(r"^postgresql\+psycopg2://", "postgresql://", url)
    return url


def _vector_literal(vec: Sequence[float]) -> str:
    # pgvector accepts literals like '[0.1,0.2,0.3]'
    # Ensure floats + compact formatting.
    parts: List[str] = []
    for x in vec:
        try:
            fx = float(x)
        except Exception:
            fx = 0.0
        parts.append(f"{fx:.8f}".rstrip("0").rstrip(".") if fx != 0.0 else "0")
    return "[" + ",".join(parts) + "]"


async def _maybe_await(v: Union[Any, Awaitable[Any]]) -> Any:
    # `typing.Awaitable` is not always safe for isinstance checks; use inspect.
    if inspect.isawaitable(v):
        return await v  # type: ignore[misc]
    return v


def _build_filter_sql(filters: Dict[str, Any], cfg: PgVectorSearchConfig) -> Tuple[str, Dict[str, Any]]:
    """Convert a filters dict to SQL.

    Supported patterns:
      - {"policy_id": "..."} matches policy_id column
      - {"collection": "..."} matches collection column (usually passed separately)
      - {"metadata": {"k":"v"}} exact match on metadata jsonb keys
      - {"k": "v"} (any other key) treated as metadata key equality

    Note: This is intentionally conservative (equality only) for safety.
    """

    if not filters:
        return "", {}

    clauses: List[str] = []
    params: Dict[str, Any] = {}

    # Explicit column filters
    if "policy_id" in filters and filters["policy_id"] is not None:
        clauses.append(f"{cfg.policy_id_column} = %(policy_id)s")
        params["policy_id"] = str(filters["policy_id"])

    # Metadata filters via nested dict
    md = filters.get("metadata")
    if isinstance(md, dict):
        for k, v in md.items():
            if v is None:
                continue
            key = f"md_{k}"
            clauses.append(f"{cfg.metadata_column} ->> %({key}_k)s = %({key})s")
            params[f"{key}_k"] = str(k)
            params[key] = str(v)

    # Any remaining keys treated as metadata key equality
    for k, v in filters.items():
        if k in ("policy_id", "collection", "metadata"):
            continue
        if v is None:
            continue
        key = f"md_{k}"
        clauses.append(f"{cfg.metadata_column} ->> %({key}_k)s = %({key})s")
        params[f"{key}_k"] = str(k)
        params[key] = str(v)

    if not clauses:
        return "", {}

    return " AND " + " AND ".join(clauses), params


# -----------------------------
# Retriever
# -----------------------------


class PgVectorPolicyRetriever:
    """Policy retriever backed by Postgres + pgvector.

    This module is intentionally dependency-light and works with psycopg3.

    Expected table schema (minimum):
      policy_chunks(
        policy_id text,
        chunk_id text,
        collection text,
        text text,
        metadata jsonb,
        embedding vector
      )

    You can rename columns via PgVectorSearchConfig.
    """

    def __init__(
        self,
        db_url: str,
        *,
        cfg: Optional[PgVectorSearchConfig] = None,
        embedder: Optional[Embedder] = None,
        connect_timeout_s: int = 10,
    ) -> None:
        self._db_url = _normalize_db_url(db_url)
        self._cfg = cfg or PgVectorSearchConfig()
        self._embedder = embedder
        self._connect_timeout_s = int(connect_timeout_s)

        # Optional async pool (if psycopg_pool is installed)
        self._pool = None
        try:
            from psycopg_pool import AsyncConnectionPool  # type: ignore

            self._pool = AsyncConnectionPool(
                conninfo=self._db_url,
                open=False,
                kwargs={"connect_timeout": self._connect_timeout_s},
            )
        except Exception:
            self._pool = None

    async def _ensure_pool(self) -> None:
        if self._pool is None:
            return
        if getattr(self._pool, "_opened", False):
            return
        await self._pool.open()

    def set_embedder(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def search(
        self,
        *,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 6,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        embedder: Optional[Embedder] = None,
    ) -> List[PolicyCitation]:
        """Vector similarity search over policy chunks.

        Returns PolicyCitation items including a `score` (higher is better).
        """

        settings = get_settings()
        cfg = self._cfg

        # Validate identifiers to avoid SQL injection if cfg is overridden.
        table = _safe_ident(cfg.table, kind="table")
        embedding_col = _safe_ident(cfg.embedding_column, kind="embedding_column")
        text_col = _safe_ident(cfg.text_column, kind="text_column")
        metadata_col = _safe_ident(cfg.metadata_column, kind="metadata_column")
        policy_id_col = _safe_ident(cfg.policy_id_column, kind="policy_id_column")
        chunk_id_col = _safe_ident(cfg.chunk_id_column, kind="chunk_id_column")
        collection_col = _safe_ident(cfg.collection_column, kind="collection_column")

        collection = (collection or settings.policy_default_collection).strip() or "default"
        top_k = int(top_k or settings.policy_top_k_default)
        top_k = max(1, min(50, top_k))

        use_embedder = embedder or self._embedder
        if use_embedder is None:
            raise RuntimeError(
                "No embedder configured for pgvector retrieval. Provide an embedder to PgVectorPolicyRetriever "
                "or pass embedder=... to search(). In the risk workflow, this should come from your embeddings model profile "
                "(e.g., Azure Ada-002 deployment or Ollama embeddings endpoint)."
            )

        raw_vec = await _maybe_await(use_embedder(query))
        # Validate and normalize embeddings against configured dimension (e.g., Ada-002 is 1536).
        vec_list = _coerce_embedding(raw_vec, expected_dim=getattr(settings, "embeddings_dim", None))
        vec_lit = _vector_literal(vec_list)

        filter_sql, filter_params = _build_filter_sql(filters or {}, cfg)

        # Score definition: 1 - cosine_distance
        # Distance operator '<=>' is cosine distance in pgvector.
        distance_expr = f"{embedding_col} {cfg.distance_operator} (%(qvec)s)::vector"
        score_expr = f"(1 - ({distance_expr}))"

        where_clause = f"{collection_col} = %(collection)s{filter_sql}"
        if min_score is not None:
            where_clause += " AND " + score_expr + " >= %(min_score)s"

        sql = f"""
            SELECT
              {policy_id_col} AS policy_id,
              {chunk_id_col} AS chunk_id,
              {text_col} AS text,
              {metadata_col} AS metadata,
              {score_expr} AS score
            FROM {table}
            WHERE {where_clause}
            ORDER BY {distance_expr} ASC
            LIMIT %(top_k)s
        """.strip()

        params: Dict[str, Any] = {
            "qvec": vec_lit,
            "collection": collection,
            "top_k": top_k,
        }
        params.update(filter_params)
        if min_score is not None:
            params["min_score"] = float(min_score)

        rows: List[Tuple[Any, ...]] = []

        # Use pool if available
        if self._pool is not None:
            await self._ensure_pool()
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    rows = await cur.fetchall()
        else:
            import psycopg

            async with await psycopg.AsyncConnection.connect(
                self._db_url,
                connect_timeout=self._connect_timeout_s,
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    rows = await cur.fetchall()

        out: List[PolicyCitation] = []
        for policy_id, chunk_id, text, metadata, score in rows:
            md: Dict[str, Any] = {}
            if metadata is None:
                md = {}
            elif isinstance(metadata, dict):
                md = metadata
            else:
                # psycopg may return json as str
                try:
                    md = json.loads(metadata)
                except Exception:
                    md = {"raw": str(metadata)}

            out.append(
                PolicyCitation(
                    policy_id=str(policy_id),
                    chunk_id=str(chunk_id),
                    score=float(score),
                    text=str(text),
                    metadata=md,
                )
            )

        return out


# -----------------------------
# Singleton retriever
# -----------------------------


_RETRIEVER: Optional[PgVectorPolicyRetriever] = None


def get_retriever() -> PgVectorPolicyRetriever:
    """Get a singleton retriever using settings.POLICY_DB_URL."""
    global _RETRIEVER
    if _RETRIEVER is None:
        settings = get_settings()
        _RETRIEVER = PgVectorPolicyRetriever(settings.policy_db_url)
    return _RETRIEVER


async def search_policies(
    query: str,
    *,
    collection: Optional[str] = None,
    top_k: int = 6,
    min_score: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    embedder: Optional[Embedder] = None,
) -> List[PolicyCitation]:
    """Convenience function to run a search using the singleton retriever."""
    retriever = get_retriever()
    return await retriever.search(
        query=query,
        collection=collection,
        top_k=top_k,
        min_score=min_score,
        filters=filters,
        embedder=embedder,
    )
