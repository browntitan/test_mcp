Below is a combined **README + User Guide** you can drop into your repo (e.g., **README.md**). It’s written so a brand-new user can set up the whole stack (OpenWebUI → Pipelines → MCP server → pgvector) and understand the data flow, tool surface, and where to modify behavior.

---

# **MCP Document Parser + Supplier Risk Assessment Server**

A FastAPI-based **MCP server** that:

1. Parses supplier contract documents (**DOCX**, **PDF**) into structured **clause objects** (including tracked changes/comments when present).
2. Retrieves relevant internal policy guidance from **Postgres + pgvector** (metadata-filtered RAG).
3. Runs a deterministic, clause-by-clause **risk assessment** using an LLM with strict **JSON schema output**.
4. Integrates with **OpenWebUI** (via Pipelines and/or Actions) to let end users upload a document and receive a risk report + optional CSV export.

---

## **Table of Contents**

* What this server does
* Architecture overview
* Supported inputs and outputs
* End-to-end data flow
* LLM “what it receives” (message shape)
* RAG filtering (clause_number + termset)
* Setup from scratch (Docker + pgvector + seeding policies)
* Running locally without Docker (optional)
* OpenWebUI integration
  * Pipelines (recommended)
  * Actions (optional)
* Configuration
  * mcp-server.env** reference**
  * Model providers (Ollama vs Azure OpenAI)
  * Configurable system prompts (env/file)
* Common operations
  * Truncate/reseed policy chunks
  * Health checks and verification
* Troubleshooting
* Where to modify things (code map)

---

# **1) What this server does**

### **Document parsing**

* Accepts a DOCX/PDF and returns:
  * **DocumentMetadata** (filename, media type, etc.)
  * A list of **Clause** objects:
    * clause_id**, **label**, **title**, **level**, **parent_clause_id
    * text
    * tracked changes/comments (when enabled)
    * optional raw spans (for downstream rendering/debug)

### **Termset extraction (DOCX)**

* The parser extracts a **termset id** from the DOCX **footer** of the form:
  * **CTM-P-ST-002** → stores  **numeric suffix only** **: **002
* This termset is used to narrow policy retrieval in pgvector.

### **Risk assessment workflow**

* For each clause:
  1. Build clause text (optionally includes “Changes” section)
  2. Query pgvector for policy excerpts using:
     * **clause_number** (derived from clause label like **2**, **2.1**, etc.)
     * **termset** (derived from footer or user-supplied termset_id)
  3. Send clause + retrieved policy excerpts to the LLM
  4. Require one bounded JSON output (**ClauseAssessment**) per clause
* Produces a report:
  * JSON output: structured results
  * Markdown output: human-readable report

---

# **2) Architecture Overview**

**Core components**

* **FastAPI server** (mcp_server.server**)**
  * REST endpoints (simple HTTP)
  * MCP-over-SSE transport (JSON-RPC over **/sse** + **/messages**)
* **Parsers**
  * DOCX parser (zip/XML): extracts clauses + tracked changes + comments
  * PDF parser: extracts text + annotations (as supported)
* **RAG**
  * Postgres **policy_chunks** table with pgvector embeddings
  * Metadata filtering (collection, clause_number, termset, etc.)
* **Risk Assessment workflow**
  * Clause-by-clause LLM calls with schema-validated JSON output
  * In-memory store tracking job status + per-clause results + report

**External dependencies**

* Postgres with pgvector extension
* LLM provider:
  * Local: Ollama (OpenAI-compatible **/v1** + native embeddings **/api/embed**)
  * Cloud: Azure OpenAI (optional)

---

# **3) Supported Inputs and Outputs**

## **Supported upload file types**

* **DOCX**: application/vnd.openxmlformats-officedocument.wordprocessingml.document
* **PDF**: application/pdf

> Note: the **/risk_assess** demo REST wrapper endpoint only accepts **.docx**. The main risk assessment workflow supports docx + pdf.

## **Primary outputs**

* **DocumentParseResult**
  * **document**: metadata (including optional **termset_id** if found)
  * **clauses**: list of clauses
  * **warnings**: parser warnings
* **Risk assessment**
  * **Start response: **assessment_id**, **status**, **document**, **clause_count**, **warnings**, **termset_id
  * Status: progress counters and current clause id
  * Clause result: per clause assessment JSON
  * Report: JSON or Markdown summary + totals

---

# **4) End-to-End Data Flow**

This is the “happy path” for OpenWebUI Pipelines:

1. User uploads DOCX/PDF in OpenWebUI chat
2. Pipeline captures file metadata (**inlet**)
3. Pipeline downloads the file bytes from OpenWebUI files API
4. Pipeline calls MCP server:
   * POST /tools/risk_assessment/start
   * sends base64 file + filename + file_type + policy settings
5. MCP server:
   * parses document into clauses (**parse_docx** / **parse_pdf**)
   * extracts termset from DOCX footer (if present)
   * determines effective **termset_id** for RAG filtering
   * creates an assessment record (in-memory store)
   * runs clause loop:
     * retrieves policy chunks from pgvector
     * calls LLM for JSON assessment per clause
     * stores results
   * creates summary (LLM or deterministic fallback)
6. Pipeline polls:
   * POST /tools/risk_assessment/status** until completed**
7. Pipeline fetches report:
   * POST /tools/risk_assessment/report** (**markdown** for the UI; **json** for CSV export)**
8. Pipeline uploads CSV to OpenWebUI for download

---

# **5) What the LLM Receives (Risk Assessment)**

The clause-by-clause assessment call uses **two messages**:

### **1)** ****

### **system**

A configurable system prompt (env/file) that enforces:

* “return ONLY valid JSON”
* strict enums (**low|medium|high**)
* bounded lengths (issues <= 5, citations <= 5, etc.)
* citations must be verbatim from provided policy excerpts only

### **2)** ****

### **user**

A structured request containing:

* Instructions + hard rules
* The clause text (optionally including a “Changes” section)
* A “Relevant internal policy excerpts (RAG)” block containing retrieved policy chunks:
  * **policy_id**, **chunk_id**, similarity **score**, and snippet text

**Important behavior**: The assessment LLM is called via a schema-enforcing method (**chat_object**), so if the response doesn’t validate, the workflow retries once with smaller inputs.

---

# **6) RAG Filtering (clause_number + termset)**

## **Metadata fields used for filtering**

Policy chunks are stored with **metadata** JSONB fields such as:

* clause_number** (string like **"2"**, **"2.1"**)**
* **termsets** (array of strings like **["001","002"]**)
* applies_to_all_termsets** (bool)**

## **How termset filtering works**

In pgvector search:

* filter key is **termset** (not **termset_id**) at query time
* the retriever matches:
  * **rows where **applies_to_all_termsets = true**, OR**
  * **metadata.termsets** contains that termset

## **Retrieval fallback strategy**

If termset-filtered retrieval yields no results:

1. retry without termset filter (clause_number only)
2. if clause_number filtering yields nothing, retry broader filters (keeping termset if provided)

This gives “best available” guidance while still prioritizing termset-specific policy where possible.

---

# **7) Setup from Scratch (Docker + pgvector + Seeding Policies)**

## **7.1 Prerequisites**

* Docker Desktop (or Podman)
* Postgres (via Docker) with pgvector extension available
* An LLM provider:
  * Ollama running locally, OR
  * Azure OpenAI credentials/deployments

---

## **7.2 Start Postgres + pgvector**

### **Option A (recommended): use a pgvector-enabled image**

Example (adjust image tag as desired):

```
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=pgvector_mcp_document \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### **Verify Postgres is up**

```
docker logs -f pgvector
```

---

## **7.3 Create** ****

## **mcp-server.env**

Put this next to your Docker build context (repo root). At minimum:

```
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765
LOG_LEVEL=INFO

POLICY_DB_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5432/pgvector_mcp_document
POLICY_DEFAULT_COLLECTION=default
POLICY_TOP_K_DEFAULT=3

# Ollama (local dev defaults)
OLLAMA_BASE_URL=http://host.docker.internal:11434/api
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_EMBEDDINGS_URL=http://host.docker.internal:11434/api/embed
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text:latest
EMBEDDINGS_DIM=768

# Prompts (recommended: file-based)
RISK_ASSESSMENT_SYSTEM_PROMPT_FILE=/app/prompts/risk_assessment_system.txt
RISK_SUMMARY_SYSTEM_PROMPT_FILE=/app/prompts/risk_summary_system.txt
```

> If you’re running Docker on Linux (not Docker Desktop), **host.docker.internal** may not exist; use a bridge IP or run everything in one compose network.

---

## **7.4 Seed policies into pgvector**

### **Policy file format + naming convention**

Policy guidance lives as **.txt** or **.md** files. Filenames determine metadata:

Examples:

* 2. Indemnity [001,002].md
* 6. Insurance [001-004].txt
* **10. General Terms.md** (no termset bracket = applies to all)

Filename parsing extracts:

* clause_number = "2"
* termsets = ["001","002"]
* applies_to_all_termsets = false** if termsets present; else true**

### **Run seeding script**

From repo root (or wherever the script lives):

```
python scripts/seed_policies.py --input-dir ./policies --collection default --upsert
```

This script:

* **creates **CREATE EXTENSION IF NOT EXISTS vector;
* creates/validates **policy_chunks** table
* embeds policy chunks using your configured embeddings model/profile
* inserts/upserts rows into **policy_chunks**

> Make sure **EMBEDDINGS_DIM** matches your embeddings model and the **vector(dim)** column dimension in Postgres.

---

## **7.5 Build and run the MCP server container**

### **Build**

```
docker build -t mcp-server-document:4 .
```

### **Run (mount prompts folder)**

```
docker run -d --name mcp-server-document-4 \
  --env-file ./mcp-server.env \
  -p 8765:8765 \
  -v "$PWD/prompts:/app/prompts:ro" \
  mcp-server-document:4
```

### **Health checks**

```
curl http://localhost:8765/health
curl http://localhost:8765/health/llm
```

---

# **8) Running Locally Without Docker (Optional)**

If you want to run directly on your machine:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export $(cat mcp-server.env | xargs)   # on mac/linux; use PowerShell equivalent on Windows

uvicorn mcp_server.server:app --host 0.0.0.0 --port 8765
```

---

# **9) OpenWebUI Integration**

## **9.1 Pipelines (recommended)**

Use the provided pipeline (e.g., **pipelines/supplier_risk_pipe_csv.py**).

**What it does**

* Captures the uploaded file ID from OpenWebUI **files[]** in **inlet()**
* Downloads bytes from OpenWebUI
* Calls MCP server risk assessment start
* Polls until complete
* Fetches Markdown report for UI + JSON report for CSV
* Uploads CSV back to OpenWebUI and prints a download link
* Displays detected termset id to the user

**Configure pipeline valves**

Key values (example):

* MCP_BASE_URL=http://host.docker.internal:8765
* OPENWEBUI_BASE_URL=http://host.docker.internal:3000
* OPENWEBUI_API_KEY=...** (if required)**

## **9.2 Actions (optional)**

There’s a minimal REST wrapper endpoint:

* **POST /risk_assess** (multipart upload)
* returns parse output + placeholder risk stats
* useful for quick UI demos, but not the full clause-by-clause workflow

---

# **10) Configuration Reference**

## **Server + DB**

* MCP_SERVER_HOST**, **MCP_SERVER_PORT**, **LOG_LEVEL
* POLICY_DB_URL
* POLICY_DEFAULT_COLLECTION**, **POLICY_TOP_K_DEFAULT

## **LLM providers**

### **Ollama (local)**

* **OLLAMA_BASE_URL** (native **/api** is accepted; server converts to **/v1**)
* OLLAMA_MODEL
* OLLAMA_EMBEDDINGS_URL**, **OLLAMA_EMBEDDINGS_MODEL
* EMBEDDINGS_DIM

### **Azure OpenAI (optional)**

* AZURE_OPENAI_ENDPOINT
* AZURE_OPENAI_API_KEY
* AZURE_OPENAI_API_VERSION
* AZURE_OPENAI_DEPLOYMENT
* AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
* AZURE_OPENAI_EMBEDDINGS_API_VERSION
* TLS:
  * AZURE_OPENAI_SSL_VERIFY
  * AZURE_OPENAI_CA_BUNDLE

## **Configurable system prompts**

Prefer file-based prompts:

* RISK_ASSESSMENT_SYSTEM_PROMPT_FILE
* RISK_SUMMARY_SYSTEM_PROMPT_FILE

Inline also supported:

* RISK_ASSESSMENT_SYSTEM_PROMPT
* RISK_SUMMARY_SYSTEM_PROMPT

---

# **11) Common Operations**

## **Truncate and reseed policy chunks**

```
TRUNCATE TABLE policy_chunks;
```

Then reseed:

```
python scripts/seed_policies.py --input-dir ./policies --collection default --upsert
```

## **Verify termset prompt is being used**

Your logs include a line like:

* risk_assessment prompts (... clause_prompt_source=file clause_prompt_sha=... clause_prompt_len=...)

Check logs:

```
docker logs -f mcp-server-document-4 | grep -E "risk_assessment prompts|termset_id"
```

## **Verify termset extraction works**

Run a docx assessment and confirm:

* start response includes **termset_id**
* **pipeline prints **Detected termset id: **002**

---

# **12) Troubleshooting**

## **Embedding dimension mismatch**

Symptoms:

* errors like “Embedding dimension mismatch” or pgvector column mismatch.

Fix:

* **confirm **EMBEDDINGS_DIM
* confirm embeddings deployment/model
* confirm **policy_chunks.embedding** vector dimension matches
* reseed policy chunks after changing embeddings model/dim

## **No policy results**

Check:

* **POLICY_DB_URL** reachable from container
* **POLICY_DEFAULT_COLLECTION** matches seeded collection
* your policy filenames include correct clause numbers and termset brackets
* verify fallback logs (drop termset / drop clause_number)

## **Prompt changes don’t take effect**

* If using env vars: you must recreate the container (env doesn’t hot-reload)
* If using prompt files mounted into container: edits should apply immediately, but some environments cache; easiest: restart container

---

# **13) Where to Modify Things (Code Map)**

### **Server / API surface**

* mcp_server/server.py
  * **REST endpoints (**/tools/***, **/health**, **/health/llm**)**
  * MCP over SSE (**/sse**, **/messages**, JSON-RPC tool calls)

### **Schemas (payload shapes)**

* mcp_server/schemas.py
  * Clause**, **DocumentParseResult
  * RiskAssessmentStartInput/Output
  * **ClauseAssessment** schema (the LLM must produce this)

### **DOCX parser + termset extraction**

* mcp_server/tools/parse_docx.py
  * clause extraction
  * tracked changes/comments
  * footer termset extraction (**CTM-P-ST-xxx** → **002**)

### **Risk assessment workflow**

* mcp_server/workflows/risk_assessment/runner.py
  * clause loop
  * RAG retrieval calls
  * LLM calls with schema validation
  * logging (including termset + prompt fingerprint)

### **RAG retriever**

* mcp_server/rag/pgvector.py
  * metadata filter handling (**termset** key)
  * SQL safety and vector search

### **Seeding policies**

* scripts/seed_policies.py
  * creates table and extension
  * parses filenames for clause_number + termsets
  * embeds and upserts chunks

### **OpenWebUI pipeline**

* pipelines/supplier_risk_pipe_csv.py
  * handles OpenWebUI file download/upload
  * **calls MCP **/tools/risk_assessment/*
  * displays termset to user, exports CSV

---

## **Quickstart Checklist (TL;DR)**

1. Start pgvector Postgres
2. **Create **mcp-server.env
3. Seed policies (**seed_policies.py**)
4. Build + run MCP server container
5. **Verify **/health** and **/health/llm
6. Install/configure OpenWebUI pipeline to call MCP server
7. Upload doc → get risk report + CSV
