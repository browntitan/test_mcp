#!/usr/bin/env python3
"""
Seed pgvector policy chunks from a folder of .txt/.md files.

Where to put this file (recommended):
  mcp-document-parser/scripts/seed_policies.py

Run example (from repo root):
  python scripts/seed_policies.py --input-dir ./test_policies --collection default --upsert

Environment (reads MCP server settings):
  POLICY_DB_URL=postgresql+psycopg://postgres:postgres@localhost:5432/pgvector_mcp_document
  POLICY_DEFAULT_COLLECTION=default
  EMBEDDINGS_MODEL_PROFILE=chat
  EMBEDDINGS_DIM=768
  OLLAMA_EMBEDDINGS_URL=http://localhost:11434/api/embed
  OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text:latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Make repo root importable when script is placed in ./scripts
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional .env loading (safe if python-dotenv is not installed)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

import psycopg  # psycopg3

from mcp_server.config import get_settings
from mcp_server.providers.llm import LLMError, get_llm


def _normalize_db_url(url: str) -> str:
    url = (url or "").strip()
    url = re.sub(r"^postgresql\+psycopg://", "postgresql://", url)
    url = re.sub(r"^postgresql\+psycopg2://", "postgresql://", url)
    return url


def _vector_literal(vec: Sequence[float]) -> str:
    # pgvector literal: '[0.1,0.2,...]'
    parts: List[str] = []
    for x in vec:
        try:
            fx = float(x)
        except Exception:
            fx = 0.0
        parts.append(f"{fx:.8f}".rstrip("0").rstrip(".") if fx != 0.0 else "0")
    return "[" + ",".join(parts) + "]"


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    """Simple character-based chunker with overlap, preserving paragraph boundaries when possible."""
    t = (text or "").replace("\r", "")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_len = 0

    for p in paras:
        # if a single paragraph is huge, split it hard
        if len(p) > chunk_size:
            flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_size)
                chunks.append(p[start:end].strip())
                start = max(end - overlap, end)
            continue

        if buf_len + len(p) + (2 if buf else 0) <= chunk_size:
            buf.append(p)
            buf_len += len(p) + (2 if buf_len else 0)
        else:
            flush()
            buf.append(p)
            buf_len = len(p)

    flush()

    # Add overlap by repeating tail text from previous chunk (lightweight)
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        prev_tail = ""
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
                prev_tail = c[-overlap:]
                continue
            overlapped.append((prev_tail + "\n" + c).strip())
            prev_tail = c[-overlap:]
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def discover_files(input_dir: Path) -> List[Path]:
    exts = {".txt", ".md"}
    out: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def policy_id_from_path(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").upper()
    return f"POLICY-{stem}"


def first_heading(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()[:200] or None
        return s[:200]
    return None


async def embed_batch(profile: str, texts: List[str], expected_dim: Optional[int]) -> List[List[float]]:
    llm = get_llm(profile)
    embs = await llm.embed_texts(texts)
    if not embs or len(embs) != len(texts):
        raise LLMError(f"Embedder returned {len(embs) if embs else 0} vectors for {len(texts)} texts")
    if expected_dim is not None:
        for i, v in enumerate(embs):
            if len(v) != expected_dim:
                raise LLMError(f"Embedding dim mismatch at index {i}: got {len(v)}, expected {expected_dim}")
    return embs


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS policy_chunks (
              policy_id   text NOT NULL,
              chunk_id    text NOT NULL,
              collection  text NOT NULL DEFAULT 'default',
              text        text NOT NULL,
              metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
              embedding   vector({dim}) NOT NULL,
              created_at  timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (policy_id, chunk_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS policy_chunks_collection_idx ON policy_chunks (collection);")
        # HNSW cosine index (requires vector(dim))
        cur.execute(
            "CREATE INDEX IF NOT EXISTS policy_chunks_embedding_hnsw "
            "ON policy_chunks USING hnsw (embedding vector_cosine_ops);"
        )
    conn.commit()


def maybe_clear_collection(conn: psycopg.Connection, collection: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM policy_chunks WHERE collection = %s;", (collection,))
        deleted = cur.rowcount or 0
    conn.commit()
    return deleted


def upsert_chunk(
    conn: psycopg.Connection,
    *,
    policy_id: str,
    chunk_id: str,
    collection: str,
    text: str,
    metadata: Dict[str, Any],
    embedding: Sequence[float],
    dim: int,
) -> None:
    emb_lit = _vector_literal(embedding)
    md_json = json.dumps(metadata)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO policy_chunks (policy_id, chunk_id, collection, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, (%s)::vector({dim}))
            ON CONFLICT (policy_id, chunk_id) DO UPDATE SET
              collection = EXCLUDED.collection,
              text = EXCLUDED.text,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding;
            """,
            (policy_id, chunk_id, collection, text, md_json, emb_lit),
        )
    conn.commit()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed policy documents into pgvector.")
    parser.add_argument("--input-dir", required=True, help="Folder containing .txt/.md policy files")
    parser.add_argument("--collection", default=None, help="Collection name (default from env/config)")
    parser.add_argument("--profile", default=None, help="Model profile to use for embeddings (default EMBEDDINGS_MODEL_PROFILE)")
    parser.add_argument("--dim", type=int, default=None, help="Embedding dim (default EMBEDDINGS_DIM)")
    parser.add_argument("--chunk-size", type=int, default=1400, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap size in characters")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--clear-collection", action="store_true", help="Delete existing rows for this collection before seeding")
    parser.add_argument("--upsert", action="store_true", help="Upsert rows (recommended). If false, still upserts (kept for backward compat).")
    args = parser.parse_args()

    settings = get_settings()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    collection = (args.collection or settings.policy_default_collection or "default").strip() or "default"
    profile = (args.profile or getattr(settings, "embeddings_model_profile", None) or "chat").strip()
    dim = int(args.dim or getattr(settings, "embeddings_dim", 768))

    db_url = _normalize_db_url(settings.policy_db_url)
    print(f"[seed] DB: {db_url}")
    print(f"[seed] Collection: {collection}")
    print(f"[seed] Embedding profile: {profile}")
    print(f"[seed] Embedding dim: {dim}")
    print(f"[seed] Input dir: {input_dir}")

    files = discover_files(input_dir)
    if not files:
        raise SystemExit(f"No .txt/.md files found under: {input_dir}")

    # Connect + ensure schema (including index)
    with psycopg.connect(db_url) as conn:
        ensure_schema(conn, dim)

        if args.clear_collection:
            deleted = maybe_clear_collection(conn, collection)
            print(f"[seed] Cleared collection '{collection}': deleted {deleted} rows")

        total_chunks = 0
        total_docs = 0

        for f in files:
            text = read_text(f)
            title = first_heading(text) or f.stem
            policy_id = policy_id_from_path(f)

            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            if not chunks:
                print(f"[seed] Skipping empty file: {f}")
                continue

            # Embed in batches
            embeddings: List[List[float]] = []
            for i in range(0, len(chunks), args.batch_size):
                batch = chunks[i : i + args.batch_size]
                embs = await embed_batch(profile, batch, expected_dim=dim)
                embeddings.extend(embs)

            if len(embeddings) != len(chunks):
                raise SystemExit(f"Embedding count mismatch for {f}: {len(embeddings)} vs {len(chunks)}")

            # Upsert chunks
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
                chunk_id = f"{policy_id}:CH-{idx:04d}"
                md = {
                    "source": "seed",
                    "filename": f.name,
                    "title": title,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                }
                upsert_chunk(
                    conn,
                    policy_id=policy_id,
                    chunk_id=chunk_id,
                    collection=collection,
                    text=chunk,
                    metadata=md,
                    embedding=emb,
                    dim=dim,
                )

            total_docs += 1
            total_chunks += len(chunks)
            print(f"[seed] Upserted {len(chunks):3d} chunks from {f.name} -> {policy_id}")

        print(f"[seed] Done. Docs={total_docs}  Chunks={total_chunks}  Collection='{collection}'")


if __name__ == "__main__":
    asyncio.run(main())
