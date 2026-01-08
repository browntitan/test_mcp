* **pgvector startup (docker compose)** step-by-step
* **DB schema overview**
* **Seeding policies end-to-end** using your **scripts/seed_policies.py**
* **Integration points** (how your MCP server uses pgvector + embeddings)
* A clear **repo layout** diagram (with the important paths you shared)

---

# **pgvector + Policy RAG: Startup, Schema, and Integration**

This project uses **Postgres + pgvector** as a policy knowledge base for RAG. The MCP server embeds queries (clauses) using a configured embeddings provider (Ollama native **/api/embed** by default), then performs a vector similarity search over **policy_chunks**.

This document explains:

1. How to start pgvector with Docker Compose
2. What tables and indexes exist (schema)
3. How to seed policies from files
4. How the MCP server uses it during **risk_assessment** runs
5. Repo layout (“repo schema”)

---

## **1) Start pgvector (Docker Compose)**

### **1.1 Files you already have**

```
docker-compose.yml
initdb/
  01_extensions.sql
  02_schema.sql
```

**docker-compose.yml**

```
services:
  pgvector_mcp_document:
    image: pgvector/pgvector:0.8.1-pg16-trixie
    container_name: pgvector_mcp_document
    environment:
      POSTGRES_DB: pgvector_mcp_document
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"   # change to "5433:5432" if you already have Postgres on 5432
    volumes:
      - pgvector_mcp_document_data:/var/lib/postgresql/data
      - ./initdb:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d pgvector_mcp_document"]
      interval: 5s
      timeout: 3s
      retries: 20

volumes:
  pgvector_mcp_document_data:
```

**initdb/01_extensions.sql**

```
CREATE EXTENSION IF NOT EXISTS vector;
```

**initdb/02_schema.sql**

```
CREATE TABLE IF NOT EXISTS policy_chunks (
  policy_id   text NOT NULL,
  chunk_id    text NOT NULL,
  collection  text NOT NULL DEFAULT 'default',
  text        text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding   vector NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (policy_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS policy_chunks_collection_idx
  ON policy_chunks (collection);
```

> Note: your seeding script also creates an **HNSW cosine index** and enforces the embedding dimension as **vector(dim)** (recommended). If you want the DB init scripts to match exactly, see the “Schema alignment” note below.

---

### **1.2 Pull and start the container**

From the repo root (same folder as **docker-compose.yml**):

```
docker compose pull
docker compose up -d
```

Check container status:

```
docker ps
docker logs -f pgvector_mcp_document
```

Wait for healthcheck:

```
docker inspect --format='{{json .State.Health}}' pgvector_mcp_document | jq
```

Or just run:

```
docker exec -it pgvector_mcp_document pg_isready -U postgres -d pgvector_mcp_document
```

Expected:

* **pg_isready** returns “accepting connections”
* Compose healthcheck becomes “healthy”

---

### **1.3 Connect and verify schema**

Connect via psql (host machine):

```
psql "postgresql://postgres:postgres@localhost:5432/pgvector_mcp_document"
```

Then verify:

```
\dx
-- should show "vector" extension installed

\dt
-- should show policy_chunks table

\d policy_chunks
```

---

## **2) Database schema (table + indexes)**

### **2.1 Core table:** ****

### **policy_chunks**

This table stores **chunked text** + metadata + embeddings:

| **Column** | **Type** | **Purpose**                                                                         |
| ---------------- | -------------- | ----------------------------------------------------------------------------------------- |
| policy_id        | text           | Stable identifier for the source doc (e.g.,**POLICY-DFARS-...**)                    |
| chunk_id         | text           | Stable identifier per chunk (**POLICY-X:CH-0001**)                                  |
| collection       | text           | Namespace for grouping documents (e.g.,**default**,**itars**,**dfars**) |
| text             | text           | The chunk’s raw text                                                                     |
| metadata         | jsonb          | Arbitrary attributes (filename, title, etc.)                                              |
| embedding        | vector         | pgvector embedding                                                                        |
| created_at       | timestamptz    | Insert timestamp                                                                          |

**Primary key: **(policy_id, chunk_id)**.**

### **2.2 Indexes**

You currently have:

* **policy_chunks_collection_idx** on **(collection)** for filtering by collection quickly.

Your **seed_policies.py** also creates:

* an **HNSW** index for cosine similarity:

```
CREATE INDEX IF NOT EXISTS policy_chunks_embedding_hnsw
ON policy_chunks USING hnsw (embedding vector_cosine_ops);
```

This is strongly recommended for fast similarity search at scale.

---

### **2.3 Schema alignment recommendation (important)**

**Your **initdb/02_schema.sql** uses:**

```
embedding vector NOT NULL
```

…but your seeding script and best practices use:

```
embedding vector(768) NOT NULL
```

The dimensional form is better because:

* pgvector can enforce dimension correctness
* HNSW indexes are defined consistently
* your MCP workflow checks embedding dims against **EMBEDDINGS_DIM**

**Recommended update**: change initdb schema to:

```
embedding vector(768) NOT NULL
```

**…and keep **EMBEDDINGS_DIM=768** in **.env**.**

If you later switch embed models (dim changes), update **both** DB schema + **.env**.

---

## **3) Environment config (MCP server → pgvector)**

Your MCP server reads:

```
POLICY_DB_URL=postgresql+psycopg://postgres:postgres@localhost:5432/pgvector_mcp_document
POLICY_DEFAULT_COLLECTION=default
POLICY_TOP_K_DEFAULT=3

EMBEDDINGS_MODEL_PROFILE=chat
OLLAMA_EMBEDDINGS_URL=http://localhost:11434/api/embed
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text:latest
EMBEDDINGS_DIM=768
```

### **What these do**

* **POLICY_DB_URL** → DB connection (SQLAlchemy-style URL in server config)
* POLICY_DEFAULT_COLLECTION** → default namespace**
* **POLICY_TOP_K_DEFAULT** → how many chunks per query to retrieve (kept low to control prompt size)
* **EMBEDDINGS_MODEL_PROFILE** → which model profile to use for embeddings (**chat** usually)
* OLLAMA_EMBEDDINGS_URL** / **OLLAMA_EMBEDDINGS_MODEL** → embedding source**
* **EMBEDDINGS_DIM** → enforced dimension (also used by your risk workflow’s health check)

---

## **4) Seed policies into pgvector (end-to-end)**

You already have:

```
mcp-document-parser/scripts/seed_policies.py
```

### **4.1 Create a policies folder**

Example:

```
test_policies/
  dfars.txt
  itar.md
  corporate_policy.txt
```

### **4.2 Ensure Ollama is running (for embeddings)**

```
ollama serve
ollama pull nomic-embed-text:latest
```

### **4.3 Seed the database**

From repo root:

```
python scripts/seed_policies.py \
  --input-dir ./test_policies \
  --collection default \
  --upsert \
  --clear-collection
```

What this script does:

1. Reads **.txt/.md** files recursively
2. Splits each file into overlapping chunks (default chunk_size=1400, overlap=200)
3. Calls embeddings via your configured **EMBEDDINGS_MODEL_PROFILE**
4. Ensures schema exists:
   * creates extension
   * creates table (with **vector(dim)**)
   * creates index on collection
   * creates HNSW cosine index
5. Upserts each chunk by **(policy_id, chunk_id)**

### **4.4 Verify rows loaded**

Connect and run:

```
SELECT collection, count(*) FROM policy_chunks GROUP BY 1;
SELECT policy_id, count(*) FROM policy_chunks GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
SELECT * FROM policy_chunks WHERE collection='default' LIMIT 1;
```

---

## **5) How the MCP server uses pgvector at runtime**

### **5.1 Where policy search happens**

In your server, the flow is:

1. **In **risk_assessment/runner.py**:**
   * you build **clause_text** (bounded)
   * **call **search_policies(...)
2. **In **rag/pgvector.py**:**
   * embed the query text (via **embed_llm**)
   * run cosine distance search against **policy_chunks**
   * return **PolicyCitation[]** (policy_id, chunk_id, score, text, metadata)
3. Back in the runner:
   * you build a bounded “Relevant internal policy excerpts” block
   * pass it into the **ClauseAssessment** structured output prompt

### **5.2 Why this architecture is stable**

* Embeddings are **decoupled** from your reasoning model:
  * you can use local embeddings + Azure chat (or vice versa)
* Prompt size is bounded:
  * top_k is controlled
  * per-snippet and total policy block are capped
* Results are stored in an ordered job store:
  * the report always returns results in clause plan order

---

## **6) Repo schema (server-side)**

Here’s a practical “repo schema” diagram based on your implementation:

```
mcp-document-parser/
├── docker-compose.yml
├── initdb/
│   ├── 01_extensions.sql
│   └── 02_schema.sql
├── .env
├── scripts/
│   └── seed_policies.py
└── mcp_server/
    ├── __init__.py
    ├── __main__.py              # python -m mcp_server
    ├── server.py                # MCP host: /sse + /messages JSON-RPC
    ├── config.py                # env settings + model profiles
    ├── schemas.py               # Pydantic models + MCP tool registry
    ├── providers/
    │   └── llm.py                # chat_text / chat_object + embeddings client wrapper
    ├── rag/
    │   └── pgvector.py           # vector search over policy_chunks
    ├── tools/
    │   ├── parse_docx.py         # DOCX parser (tracked changes + comments)
    │   ├── parse_docx2.py        # New parser for TOC + heading-based docs (if added)
    │   ├── parse_pdf.py          # PDF parser (PyMuPDF)
    │   ├── normalize_clauses.py  # clause normalization & boundary application
    │   └── risk_assessment.py    # MCP tool handlers for assessment workflow
    └── workflows/
        └── risk_assessment/
            ├── runner.py         # deterministic assessment engine (RAG + LLM JSON)
            └── store.py          # in-memory job store (status/report/results)
```

---

## **7) “pgvector integration script” (optional: one-command bootstrap)**

If you want a single script that:

* starts docker compose
* waits for DB ready
* seeds policies
  **…you can create **scripts/bootstrap_pgvector.sh**:**

```
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] Starting pgvector..."
cd "$ROOT"
docker compose pull
docker compose up -d

echo "[bootstrap] Waiting for DB to be ready..."
for i in {1..60}; do
  if docker exec -it pgvector_mcp_document pg_isready -U postgres -d pgvector_mcp_document >/dev/null 2>&1; then
    echo "[bootstrap] DB is ready."
    break
  fi
  sleep 2
done

echo "[bootstrap] Seeding policies..."
python "$ROOT/scripts/seed_policies.py" \
  --input-dir "$ROOT/test_policies" \
  --collection default \
  --upsert \
  --clear-collection

echo "[bootstrap] Done."
```

Run:

```
chmod +x scripts/bootstrap_pgvector.sh
./scripts/bootstrap_pgvector.sh
```

---

## **8) Common troubleshooting**

### **“Connection refused” on localhost:5432**

* Another Postgres is already using 5432
* Fix: change compose port mapping:
  * "5433:5432"** and update **POLICY_DB_URL** accordingly**

### **“Embedding dim mismatch”**

* Your **EMBEDDINGS_DIM** doesn’t match the embedding model output
* Fix: change **EMBEDDINGS_DIM** or use a different embed model
* If you already created **vector(768)** but model outputs different dim, you must re-migrate.

### **“No policies retrieved”**

* You didn’t seed anything into the **collection** being queried
* Ensure **--collection** matches **policy_collection** used in risk assessment (default: **default**)

---

If you paste your **mcp_server/rag/pgvector.py** file and tell me whether you want cosine similarity via **<=>** or **vector_cosine_ops** explicitly, I can also provide a **fully consistent SQL query + index strategy** that exactly matches your runtime retrieval logic.
