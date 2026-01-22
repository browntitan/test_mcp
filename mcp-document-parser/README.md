
Below are **full markdown** drafts for both documents. I made a few reasonable assumptions and marked them as **(unknown)** where your repo content wasn’t provided (e.g., exact **parse_pdf.py** behavior).

---

# **README.md**

```
# MCP Document Parser + Supplier Risk Assessment

A Python **MCP (Model Context Protocol)** server that:
1) parses supplier/legal **DOCX** and **PDF** documents into structured clauses (including tracked changes + comments), and  
2) runs a deterministic, clause-by-clause **risk assessment** using:
- **Azure OpenAI Gov** *or* any **OpenAI-compatible** endpoint (including **Ollama** and **vLLM/NIM**)
- **Postgres + pgvector** for policy retrieval (RAG)
- an async **job store** with **status polling**
- both **REST endpoints** and **MCP JSON-RPC over SSE**

It integrates with **OpenWebUI** via an **OpenWebUI Pipelines** “model” that captures file uploads in `inlet()` and orchestrates calls into the MCP server.

---

## Why this exists

Most “upload + summarize” workflows fail in enterprise settings because they:
- don’t preserve tracked changes and review comments
- aren’t deterministic (hard to explain/verify)
- can’t enforce a consistent JSON schema per clause
- don’t separate chat vs embeddings model configuration (critical for RAG correctness)
- don’t integrate cleanly with OpenWebUI file storage + Pipelines

This project solves that with:
- robust DOCX parsing (TOC skip, headings, redlines/comments)
- a deterministic assessment loop with structured outputs
- a dedicated embeddings profile + embedding dimension checks
- first-class OpenWebUI Pipelines integration

---

## Architecture diagram (Mermaid)

```mermaid
flowchart LR
  U[User in OpenWebUI] -->|upload docx/pdf| OW[OpenWebUI]
  OW -->|/chat/completions| P[Pipelines Model]
  P -->|GET /api/v1/files/{id}/content| OW
  P -->|POST /tools/risk_assessment/start| MCP[MCP Server (FastAPI)]
  P -->|poll /tools/risk_assessment/status| MCP
  P -->|POST /tools/risk_assessment/report| MCP

  MCP -->|embed query| EMB[Embeddings Provider<br/>Azure / vLLM / Ollama]
  MCP -->|vector search| PG[(Postgres + pgvector)]
  MCP -->|chat scoring| LLM[Chat Provider<br/>Azure / vLLM / Ollama]
```

---

## **Features**

### **Parsing**

* DOCX parsing with:
  * TOC skipping
  * heading/outline-level detection
  * grouping subclauses into “one card per main clause” for UI
  * **tracked changes (**ins/del/moveTo/moveFrom**)**
  * comments from **comments.xml** + anchored snippets
* PDF parsing (**tools/parse_pdf.py**) *(details depend on file; see Technical Docs)*

### **Risk assessment workflow**

* Deterministic clause-by-clause loop
* Async job execution with polling
* Dedicated embeddings profile + dimension validation
* RAG policy retrieval from pgvector
* Structured per-clause JSON result + markdown report rendering

### **Interfaces**

* REST endpoints for pipelines and non-MCP clients
* MCP JSON-RPC over SSE (**/sse** + **/messages**) for MCP-native clients
* /health** + **/health/llm** diagnostics**

### **OpenWebUI integration**

* Pipeline captures **body["files"]** in **inlet()**
* Downloads file bytes from OpenWebUI file API
* Calls MCP server REST endpoints
* Streams progress updates with event emitters
* Supports chunked final output to avoid OpenWebUI “Chunk too big”

---

## **Requirements**

### **Runtime**

* Python **3.11+**
* Postgres + **pgvector** (required for policy retrieval)
* One of:
  * Azure OpenAI Gov endpoint (optional)
  * OpenAI-compatible endpoint (Ollama, vLLM, NIM, etc.) (optional)

### **Python dependencies**

Installed from **requirements.txt** (exact packages depend on your file), typically includes:

* fastapi**, **uvicorn
* httpx
* pydantic**, **pydantic-settings
* **lxml** (DOCX parsing)
* **psycopg** / **psycopg_pool** (pgvector retrieval)

### **Docker (optional)**

* Docker Desktop (macOS/Windows) or Docker Engine (Linux)
* Docker is recommended for Postgres+pgvector and for containerizing the MCP server

---

## **Repo layout (high level)**

```
mcp-document-parser/
  mcp_server/
    server.py              # FastAPI + REST + MCP SSE JSON-RPC + /health/llm
    config.py              # Pydantic settings + model profile system
    schemas.py             # Pydantic models + MCP tool schemas
    providers/llm.py       # Azure + OpenAI-compatible clients (chat + embeddings)
    rag/pgvector.py        # pgvector retriever + filters + dimension validation
    tools/
      parse_docx.py        # DOCX parsing (current)
      parse_pdf.py         # PDF parsing
      normalize_clauses.py # boundary-based normalization + dedupe + hierarchy
      risk_assessment.py   # tool handlers + markdown report rendering
    workflows/risk_assessment/
      runner.py            # deterministic assessment loop + embeddings health check
      store.py             # async in-memory job store
  scripts/seed_policies.py # seed policy chunks into policy_chunks
  pipelines/supplier_risk_pipe.py # OpenWebUI Pipelines model
```

Note: you have **parse_docx2.py**, **parse_docx3.py** (variants). Current active one depends on your **tools/__init__.py** wiring (unknown). Prefer a single canonical parser file for production.

---

## **Quickstart (local run)**

### **1) Create and activate venv**

```
cd mcp-doc-parser/mcp-document-parser
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows PowerShell
```

### **2) Install deps**

```
pip install -r requirements.txt
```

### **3) Start Postgres + pgvector (Docker recommended)**

```
docker run -d --name pgvector \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=pgvector_mcp_document \
  -v pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

### **4) Configure** ****

### **.env**

See “Configuration” below. At minimum:

* POLICY_DB_URL=...
* model profiles (Ollama or Azure)
* **EMBEDDINGS_DIM** matching your embeddings model

### **5) Run the server**

```
python -m mcp_server
# or:
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8765
```

### **6) Smoke tests**

```
curl http://localhost:8765/health
curl http://localhost:8765/health/llm
```

---

## **Quickstart (Docker run)**

### **1) Build**

```
docker build -t mcp-server:latest -f mcp_server/Dockerfile .
```

### **2) Create env file for container**

Create **mcp-server.env** (docker env-file format: **KEY=VALUE** per line):

```
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765
LOG_LEVEL=INFO

POLICY_DB_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5432/pgvector_mcp_document
POLICY_DEFAULT_COLLECTION=default
POLICY_TOP_K_DEFAULT=3

EMBEDDINGS_MODEL_PROFILE=assessment
EMBEDDINGS_DIM=1536

AZURE_OPENAI_SSL_VERIFY=false
MODEL_PROFILES_JSON={...one-line-json...}
DEFAULT_CHAT_MODEL_PROFILE=chat
DEFAULT_ASSESSMENT_MODEL_PROFILE=assessment
ALLOW_MODEL_OVERRIDE=false
```

### **3) Run**

macOS/Windows:

```
docker run -d --name mcp-server \
  -p 8765:8765 \
  --env-file ./mcp-server.env \
  mcp-server:latest
```

Linux requires:

```
docker run -d --name mcp-server \
  --add-host=host.docker.internal:host-gateway \
  -p 8765:8765 \
  --env-file ./mcp-server.env \
  mcp-server:latest
```

### **4) Verify**

```
curl http://localhost:8765/health
curl http://localhost:8765/health/llm
```

---

## **Configuration**

### **Model profiles system (Ollama vs Azure vs vLLM/NIM)**

The server resolves logical model usage via **profiles**:

* **chat** profile: general chat usage (OpenWebUI interactive, summaries, etc.)
* **assessment** profile: deterministic scoring of clauses
* optional **embeddings** profile: embeddings used for RAG retrieval

Profiles can be configured two ways:

#### **A) Explicit profiles via** ****

#### **MODEL_PROFILES_JSON**

#### ** (recommended)**

Example: Azure OpenAI Gov

```
MODEL_PROFILES_JSON={"chat":{"provider":"azure_openai","azure_endpoint":"https://...","azure_api_version":"2024-02-01","azure_deployment":"chat-deploy","api_key":"...","azure_embeddings_deployment":"embed-deploy","azure_embeddings_api_version":"2024-10-21"},"assessment":{"provider":"azure_openai","azure_endpoint":"https://...","azure_api_version":"2024-02-01","azure_deployment":"chat-deploy","api_key":"...","azure_embeddings_deployment":"embed-deploy","azure_embeddings_api_version":"2024-10-21"}}
DEFAULT_CHAT_MODEL_PROFILE=chat
DEFAULT_ASSESSMENT_MODEL_PROFILE=assessment
EMBEDDINGS_MODEL_PROFILE=assessment
```

Example: vLLM/NIM (OpenAI-compatible)

```
MODEL_PROFILES_JSON={"chat":{"provider":"openai_compatible","base_url":"http://vllm-host:8000/v1","api_key":"token","model":"meta/llama-3.1-8b-instruct"},"assessment":{"provider":"openai_compatible","base_url":"http://vllm-host:8000/v1","api_key":"token","model":"meta/llama-3.1-8b-instruct"},"embeddings":{"provider":"openai_compatible","base_url":"http://embed-host:8001/v1","api_key":"token","model":"BAAI/bge-base-en-v1.5"}}
DEFAULT_CHAT_MODEL_PROFILE=chat
DEFAULT_ASSESSMENT_MODEL_PROFILE=assessment
EMBEDDINGS_MODEL_PROFILE=embeddings
EMBEDDINGS_DIM=768
```

#### **B) Implicit profiles from env vars (fallback)**

**If **MODEL_PROFILES_JSON** is empty, **config.py** builds:**

* **chat** from Ollama defaults
* **assessment** from Azure vars if present, else same as chat
* optional **embeddings** profile if Azure embeddings deployment is provided

---

## **Embeddings configuration + dimension check**

RAG retrieval requires embeddings. The system enforces:

* **EMBEDDINGS_MODEL_PROFILE** selects which profile performs embeddings
* EMBEDDINGS_DIM** must match:**
  * the embeddings model output dimension
  * the pgvector column dimension in the DB

The server checks embedding dimension during:

* /health/llm** (probe: **embed_texts("dimension check")**)**
* risk workflow start (fails fast if mismatched)

Common dimensions:

* text-embedding-ada-002**: ****1536**
* many local embedding models: **768** (varies)

---

## **Postgres schema expectations (policy_chunks)**

Minimum required table (default name **policy_chunks**):

```
CREATE TABLE IF NOT EXISTS policy_chunks (
  policy_id   text NOT NULL,
  chunk_id    text NOT NULL,
  collection  text NOT NULL DEFAULT 'default',
  text        text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding   vector(1536) NOT NULL,
  PRIMARY KEY (policy_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_collection
  ON policy_chunks (collection);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding
  ON policy_chunks USING ivfflat (embedding vector_cosine_ops);
```

Adjust the vector dimension to match **EMBEDDINGS_DIM**.

Seeding: see **scripts/seed_policies.py** (usage depends on file; unknown).

---

## **API endpoints reference**

### **Health**

* GET /health** → **{"status":"healthy","version":"..."}
* **GET /health/llm** → checks chat + embeddings connectivity + dimension match

### **Parsing (REST)**

* POST /tools/parse_docx
* POST /tools/parse_pdf
* POST /tools/normalize_clauses

### **Risk workflow (REST)**

* POST /tools/risk_assessment/start
* POST /tools/risk_assessment/status
* POST /tools/risk_assessment/get_clause_result
* POST /tools/risk_assessment/report
* POST /tools/risk_assessment/cancel
* POST /tools/risk_assessment/start_upload** (multipart helper)**

### **MCP (JSON-RPC over SSE)**

* **GET /sse** (server-sent events)
* POST /messages?session_id=...** (JSON-RPC messages)**
* Supported methods:
  * initialize
  * tools/list
  * tools/call

---

## **Example curl commands**

### **/health**

```
curl http://localhost:8765/health
```

### **/health/llm**

```
curl http://localhost:8765/health/llm
```

### **parse_docx**

```
curl -s http://localhost:8765/tools/parse_docx \
  -H "Content-Type: application/json" \
  -d '{
    "file_base64": "<BASE64>",
    "options": {
      "extract_tracked_changes": true,
      "extract_comments": true,
      "include_raw_spans": false
    }
  }'
```

### **parse_pdf**

```
curl -s http://localhost:8765/tools/parse_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "file_base64": "<BASE64>",
    "options": {
      "extract_annotations": true,
      "include_raw_spans": false
    }
  }'
```

### **risk_assessment/start**

```
curl -s http://localhost:8765/tools/risk_assessment/start \
  -H "Content-Type: application/json" \
  -d '{
    "file_base64": "<BASE64>",
    "filename": "doc.docx",
    "file_type": "docx",
    "policy_collection": "default",
    "top_k": 3,
    "model_profile": "assessment",
    "include_text_with_changes": true,
    "mode": "async"
  }'
```

### **risk_assessment/status**

```
curl -s http://localhost:8765/tools/risk_assessment/status \
  -H "Content-Type: application/json" \
  -d '{"assessment_id":"<ID>"}'
```

### **risk_assessment/report (markdown)**

```
curl -s http://localhost:8765/tools/risk_assessment/report \
  -H "Content-Type: application/json" \
  -d '{"assessment_id":"<ID>","format":"markdown"}'
```

---

## **OpenWebUI + Pipelines integration (quick guide)**

1. Install/configure OpenWebUI + Pipelines
2. **Add **pipelines/supplier_risk_pipe.py** as a Pipelines model**
3. Ensure Pipelines can reach:

* **OpenWebUI: **http://host.docker.internal:3000
* **MCP server: **http://host.docker.internal:8765

Important: OpenWebUI doesn’t always forward files to the final completion call, but the Pipelines **inlet()** receives **body["files"]**. The pipeline captures file metadata in **inlet()** and stores it into:

* body["_supplier_risk_file"]

Then **pipe()** downloads bytes from:

* GET /api/v1/files/{id}/content
  and calls MCP endpoints.

If the UI shows “Chunk too big”, the pipeline should chunk final output (this repo already includes a chunking valve **MAX_UI_CHUNK_CHARS** in the pipeline).

---

## **Troubleshooting**

### **“LLM returned non-JSON response (HTML redirect)”**

Symptoms:

* LLM returned non-JSON response ... content_type='text/html' location=...
  Common causes:
* wrong Azure endpoint path (redirect)
* proxy intercepting
* missing headers/auth
  Fix:
* **run **GET /health/llm
* verify endpoint URL and deployment names
* check request-id headers in error message

### **“CERTIFICATE_VERIFY_FAILED”**

Cause:

* TLS interception / missing corporate CA in trust store
  Fix options:
* **preferred: set **AZURE_OPENAI_CA_BUNDLE=<path-to-ca.pem>** and **AZURE_OPENAI_SSL_VERIFY=true
* **fallback: **AZURE_OPENAI_SSL_VERIFY=false** (temporary)**

### **“Proxy 407”**

Cause:

* outbound request forced through a proxy requiring auth
  Fix:
* **set **NO_PROXY=localhost,127.0.0.1,host.docker.internal
* ensure local services aren’t routed through proxy

### **“Embedding dimension mismatch”**

Cause:

* **EMBEDDINGS_DIM** doesn’t match embeddings model output or DB vector column
  Fix:
* **set correct **EMBEDDINGS_DIM
* **recreate **policy_chunks.embedding vector(`<dim>`)** and re-embed policies**

### **“No files found in pipeline body”**

Cause:

* OpenWebUI didn’t include files in the final request; must capture in inlet
  Fix:
* ensure pipeline captures in **inlet()**
* verify OpenWebUI calls **/filter/inlet** with files populated

---

## **Security notes**

* Do **not** commit API keys or **.env** files
* Prefer CA bundles over **verify=false**
* Consider adding RBAC and request authentication for production (unknown / not implemented)
* The in-memory job store is not durable; restart wipes assessments

---

## **License**

(unknown)

```
---

# TECHNICAL_DOCUMENTATION.md

```md
# TECHNICAL_DOCUMENTATION — MCP Document Parser + Risk Assessment

This document explains the internal architecture, code modules, schemas, data flow, configuration system, and OpenWebUI Pipelines integration for the MCP Document Parser project.

Assumptions:
- `tools/parse_docx.py` is the primary DOCX parser currently used by `mcp_server.tools.parse_docx` import.
- `tools/parse_pdf.py` exists but its internal details are (unknown) unless reviewed.
- OpenWebUI Pipelines module is `pipelines/supplier_risk_pipe.py`.

Where details are unknown, they are labeled **(unknown)** along with what would confirm them.

---

## 1) System overview

The system has four major responsibilities:

1) **Parse** legal/supplier documents (DOCX/PDF) into structured `Clause` objects with:
   - clause text
   - hierarchical metadata (label/title/level/parent)
   - raw spans (optional)
   - tracked changes and comments (DOCX)

2) **Retrieve policy context** for each clause using:
   - embeddings (configured via model profiles)
   - Postgres + pgvector vector similarity search

3) **Assess risk deterministically**:
   - iterate clauses in a stable order
   - prompt LLM to output a bounded JSON schema (`ClauseAssessment`)
   - store progress and results in an async job store
   - render final report (JSON or markdown)

4) **Expose interfaces**:
   - REST endpoints for Pipelines and simple clients
   - MCP JSON-RPC over SSE for MCP-native clients
   - health endpoints including `/health/llm` diagnostics

---

## 2) Data flow diagrams

### 2.1 DOCX/PDF parsing flow

```mermaid
flowchart TD
  A[Input: file_path or file_base64] --> B{file_type?}
  B -->|docx| C[tools/parse_docx.py]
  B -->|pdf| D[tools/parse_pdf.py]
  C --> E[DocumentParseResult]
  D --> E[DocumentParseResult]
  E --> F[clauses: List[Clause]]
  E --> G[warnings: List[str]]
```

### **2.2 Risk assessment flow (async job lifecycle)**

```
flowchart TD
  S[POST /tools/risk_assessment/start] --> P[load_or_parse]
  P --> J[store.create assessment_id]
  J -->|mode=async| T[asyncio task: _run_assessment]
  J -->|mode=sync| T

  T --> H[embeddings probe + dim check]
  H -->|ok| L[for each clause in deterministic order]
  L --> R[pgvector retrieval (search_policies)]
  R --> M[LLM chat_object -> ClauseAssessment JSON]
  M --> W[store.put_clause_result / put_clause_risk_result]
  W --> L

  L -->|done| SUM[LLM executive summary]
  SUM --> REP[store.set_report]
  REP --> DONE[store.set_status completed]
```

### **2.3 OpenWebUI → Pipelines → MCP integration flow**

```
flowchart LR
  U[User Uploads File] --> OW[OpenWebUI]
  OW --> INLET[Pipelines inlet() receives body.files]
  INLET -->|capture| MEM[body._supplier_risk_file]
  OW --> PIPE[Pipelines pipe()]
  PIPE -->|GET file bytes| OWF[OpenWebUI /api/v1/files/{id}/content]
  PIPE -->|POST start| MCP[ /tools/risk_assessment/start ]
  PIPE -->|poll status| MCP2[ /tools/risk_assessment/status ]
  PIPE -->|fetch report| MCP3[ /tools/risk_assessment/report ]
  PIPE -->|stream progress + final report| OW
```

---

## **3) Module-by-module walkthrough**

### **3.1** ****

### **mcp_server/server.py**

Responsibilities:

* Creates a FastAPI application with CORS enabled.
* Exposes:
  * GET /health
  * **GET /health/llm** (diagnostic chat + embeddings probe)
  * REST tool endpoints under **/tools/...**
  * MCP protocol endpoints:
    * GET /sse
    * POST /messages?session_id=...

Key components:

#### **Windows asyncio compatibility**

A Windows-only event-loop policy override is applied to avoid psycopg async incompatibility with ProactorEventLoop:

```
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

This prevents warnings like:

* “Psycopg cannot use the ProactorEventLoop … in async mode”

#### **REST endpoints**

**/tools/*** endpoints validate payloads using Pydantic schemas in **schemas.py** and dispatch to corresponding tool functions (parse/normalize/risk).

#### **MCP over SSE**

Implements MCP JSON-RPC:

* **tools/list** returns **MCP_TOOLS** schema list
* **tools/call** validates tool name + arguments and calls **_call_tool**

Notes:

* The server supports both REST and MCP transport and reuses the same underlying tool logic.

#### **/health/llm**

This endpoint:

* **selects **assessment_profile** and **embeddings_profile
* probes chat with a simple ping prompt
* **probes embeddings with **embed_texts("dimension check")
* checks embedding vector length matches **Settings.embeddings_dim**

This endpoint is the fastest way to debug:

* TLS errors
* proxy errors
* wrong deployment names
* embedding dimension mismatches

---

### **3.2** ****

### **mcp_server/schemas.py**

Defines all data models:

* **parsing outputs: **DocumentParseResult**, **Clause**, **ClauseChanges**, **Redlines**, **Comment
* **normalization outputs: **ClauseListNormalized**, **NormalizedClause
* **risk workflow inputs/outputs: **RiskAssessmentStartInput**, **RiskAssessmentStatusOutput**, **RiskAssessmentReportOutput**, etc.**
* scoring models:
  * **ClauseAssessment** (strict JSON output schema)
  * **ClauseRiskResult** (wraps the assessed clause text + **ClauseAssessment**)

Important patterns:

* Validators normalize enums like **risk_level** and **issues[].severity** to strict values: **low|medium|high**.
* RiskAssessmentStartInput** enforces:**
  * **either **file_path/file_base64** OR **parse_result
  * requires **file_type** if base64 is provided without filename/path
* **stable_clause_id(index, text)** generates stable IDs per clause content.

Tool registration:

* **MCP_TOOLS** defines the MCP tool catalog returned by **tools/list**.

---

### **3.3** ****

### **mcp_server/tools/parse_docx.py**

### ** (and variants)**

DOCX parsing capabilities (as implemented in your provided parser variant):

* Reads **.docx** zip parts:
  * word/document.xml
  * **word/styles.xml** (outline levels)
  * **word/numbering.xml** (auto numbering best-effort)
  * **word/comments.xml** (comments + metadata)
* Skips Table of Contents via:
  * TOC styles (**TOC1**, **TOCHeading**, etc.)
  * TOC field markers (**fldChar**, **instrText**)
  * literal “Table of Contents” paragraph
* Heading detection:
  * primary heuristic: choose *shallowest outline level* as boundary
  * fallback: ARTICLE/SECTION/numeric patterns
  * avoids splitting on list-numbered paragraphs (**w:numPr**)
* Tracked changes extraction:
  * w:ins**, **w:del**, **w:moveTo**, **w:moveFrom
* Comment extraction:
  * parses comment metadata
  * anchors comment text to relevant paragraph spans
* Produces:
  * **Clause.text** containing boundary + merged subclauses
  * Clause.redlines** and **Clause.changes** structured fields**

Design rationale:

* **TOC skipping** prevents false clause boundaries.
* “one card per main clause” UX: boundaries only at top-level headings; subclauses merged.

Variants:

* **parse_docx2.py**, **parse_docx3.py** exist (unknown purpose). For production, prefer a single canonical parser and remove/archvive older variants or gate them behind flags.

---

### **3.4** ****

### **mcp_server/tools/parse_pdf.py**

### ** (present, details unknown)**

Expected responsibilities:

* load PDF bytes
* extract text and (optionally) annotations (comments/strikethrough)
* **produce a **DocumentParseResult** with **Clause** list**
  To confirm: inspect **parse_pdf.py** parsing strategy (page chunking, heading detection, annotation extraction).

---

### **3.5** ****

### **mcp_server/tools/normalize_clauses.py**

Takes:

* NormalizeClausesInput(parse_result, boundaries?)

If **boundaries** provided:

* builds a global text string
* maps boundary spans back to original clause ranges
* merges metadata from contributing clauses
* **sets **was_merged** / **was_split** flags**

Always:

* deduplicates clauses by normalized text hash
* reassigns stable clause IDs
* fixes hierarchy based on **level**

Design rationale:

* Supports future LLM-assisted boundary detection while keeping normalization deterministic.
* Deduping prevents duplicate UI cards and repeated assessments.

---

### **3.6** ****

### **mcp_server/workflows/risk_assessment/runner.py**

This is the core deterministic workflow.

Key steps:

1. **Load or parse**

* If **start.parse_result** provided, trust it.
* Else infer file type and call:
  * parse_docx** or **parse_pdf

2. **Create assessment record**

* store.create(document, clause_ids, warnings, status)

3. **Resolve model profiles**

* assessment profile:
  * **from **start.model_profile** with fallback to **Settings.default_assessment_profile
* embeddings profile:
  * from **Settings.embeddings_model_profile** with fallback to chat/assessment

Design rationale:

* Embeddings profile must be decoupled from chat/assessment to prevent accidental embedding calls to wrong endpoint.

4. **Embeddings health check**

* **one-time probe **embed_texts("dimension check")
* validate returned vector length == **Settings.embeddings_dim**
* fail fast if mismatch

5. **Deterministic clause iteration**

* iterate **clause_order** in stable parsed order
* for each clause:
  * format clause text (optional changes/comments)
  * retrieve policy citations with **search_policies()**
  * **call LLM with **chat_object(... schema=ClauseAssessment ...)
  * store:
    * ClauseAssessment (back-compat)
    * ClauseRiskResult (rich, includes clause text)

6. **Executive summary**

* Builds a compact summary seed from top medium/high risk clauses
* Calls **chat_text()** for the final summary
* Falls back to deterministic summary if LLM fails

7. **Guard: fail if no results**

* If all clause assessments fail and **results** is empty, the workflow raises:
  * "All clause assessments failed (no clause results produced) ..."
    This prevents misleading “completed” status.

Observability:

* Warnings accumulated in store explain:
  * retrieval failures
  * JSON/schema validation issues
  * truncation events

---

### **3.7** ****

### **mcp_server/workflows/risk_assessment/store.py**

In-memory async job store (**RiskAssessmentStore**):

* create()** returns **assessment_id
* set_status()**, **set_progress()**, **add_warning()
* stores results in two shapes:
  * clause_results**: **clause_id -> ClauseAssessment** (legacy)**
  * clause_risk_results**: **clause_id -> ClauseRiskResult** (preferred)**
  * **clause_text_by_id**: exact assessed clause text (optional)

Report assembly:

* **report_output()** preserves clause ordering based on plan (**clause_ids**)
* prefers rich results, wraps assessments with text if available

Durability:

* In-memory only. Restart wipes all job state.
* Scaling to multiple workers requires an external store (Redis/Postgres) (not implemented).

---

### **3.8** ****

### **mcp_server/tools/risk_assessment.py**

Tool wrapper layer for the workflow:

* start/status/report/get_clause_result/cancel
* **_render_report_markdown()** builds the final markdown report:
  * summary + totals
  * per-clause sections including clause text, risk level, issues, citations, redlines

Back-compat:

* report rendering can handle:
  * ClauseAssessment only
  * ClauseRiskResult with embedded **assessment**

---

### **3.9** ****

### **mcp_server/rag/pgvector.py**

Implements vector search against Postgres+pgvector.

Highlights:

* accepts an **embedder** function (sync or async)
* normalizes embeddings to **list[float]**
* validates embedding length matches **Settings.embeddings_dim**
* constructs safe SQL:
  * validates identifiers with **_safe_ident**
* cosine distance search via **<=>**
  * score computed as **1 - distance**
* supports filters:
  * policy_id
  * **metadata** key equality
  * other keys treated as metadata equality
* **uses **psycopg_pool.AsyncConnectionPool** if installed, else uses **psycopg.AsyncConnection

Design rationale:

* dimension validation catches config drift early
* conservative filter support reduces SQL injection risk
* embedder passed in keeps retrieval decoupled from LLM client internals

---

### **3.10** ****

### **mcp_server/providers/llm.py**

Unified LLM client system driven by model profiles:

* Supports:
  * **openai_compatible** (Ollama, vLLM, NIM, gateways)
  * **azure_openai** (Azure Gov endpoints)

Key features:

* profile-driven URL construction:
  * **chat: **/chat/completions
  * embeddings: **/embeddings** or explicit **embeddings_url** (Ollama native **/api/embed**)
* retry logic for transient status codes (429, 5xx)
* rich error messages:
  * includes request/correlation IDs when present
  * shows content-type and redirect location for non-JSON responses

TLS / CA behavior:

* Azure TLS options come from Pydantic **Settings** (important: **.env** loaded by pydantic does not always populate **os.environ**)
  * AZURE_OPENAI_SSL_VERIFY
  * AZURE_OPENAI_CA_BUNDLE

JSON extraction:

* **chat_object()** forces JSON mode where supported and validates output schema.
* **_json_extract()** was added to handle:
  * fenced JSON
  * minor leading/trailing text while extracting first balanced object

Design rationale:

* strict schema validation prevents downstream ambiguity
* retries + request ids improve enterprise debugging
* separate embeddings deployment avoids “embeddings accidentally hit chat model”

---

## **4) Environment variables reference**

Grouped by concern. Defaults come from **config.py**. If an env var is “required,” it means required for that feature.

### **4.1 Server**

* MCP_SERVER_HOST** (default **0.0.0.0**)**
* MCP_SERVER_PORT** (default **8765**)**
* LOG_LEVEL** (default **INFO**)**

### **4.2 Policy DB / RAG**

* **POLICY_DB_URL** (required for RAG; default points to localhost)
* POLICY_DEFAULT_COLLECTION** (default **default**)**
* POLICY_TOP_K_DEFAULT** (default **3**)**

### **4.3 Model profiles (recommended)**

* **MODEL_PROFILES_JSON** (recommended for deterministic behavior)
* DEFAULT_CHAT_MODEL_PROFILE** (default **chat**)**
* DEFAULT_ASSESSMENT_MODEL_PROFILE** (default **assessment**)**
* ALLOW_MODEL_OVERRIDE** (default **false**)**

### **4.4 Ollama (implicit profiles)**

* OLLAMA_BASE_URL** (default **http://localhost:11434/api**)**
* OLLAMA_OPENAI_BASE_URL** (optional)**
* OLLAMA_MODEL** (default **gpt-oss:20b**)**
* OLLAMA_EMBEDDINGS_URL** (default **http://localhost:11434/api/embed**)**
* OLLAMA_EMBEDDINGS_MODEL** (default **nomic-embed-text:latest**)**

### **4.5 Azure OpenAI**

* AZURE_OPENAI_ENDPOINT
* AZURE_OPENAI_API_KEY
* AZURE_OPENAI_API_VERSION
* AZURE_OPENAI_DEPLOYMENT
* AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT** (recommended)**
* **AZURE_OPENAI_EMBEDDINGS_API_VERSION** (optional; defaults to api version)

### **4.6 Embeddings selection**

* EMBEDDINGS_MODEL_PROFILE
  * selects which profile is used for embeddings in RAG and dimension checks
* EMBEDDINGS_DIM
  * must match model output dimension and DB **vector(dim)** column

### **4.7 TLS / CA**

* AZURE_OPENAI_SSL_VERIFY** (default **true**)**
* **AZURE_OPENAI_CA_BUNDLE** (optional; preferred over verify=false)

### **4.8 Proxy**

Because **httpx** uses **trust_env=True**, it respects standard proxy env vars:

* HTTPS_PROXY
* HTTP_PROXY
* NO_PROXY

Typical enterprise config:

* include local addresses in NO_PROXY:
  * NO_PROXY=localhost,127.0.0.1,host.docker.internal

---

## **5) Typical configurations**

### **5.1 Local dev: Ollama + local pgvector**

* Run Postgres+pgvector locally (docker)
* Run Ollama locally
* Use implicit profiles or explicit JSON for clarity
* Set **EMBEDDINGS_DIM** to match your embedding model (commonly 768)

### **5.2 Enterprise: Azure Gov + CA bundle + pgvector**

* Use **MODEL_PROFILES_JSON** with azure profiles
* Prefer CA bundle:
  * AZURE_OPENAI_CA_BUNDLE=/path/to/corp-ca.pem
  * AZURE_OPENAI_SSL_VERIFY=true
* Ensure embeddings deployment is set and dimension is correct (often 1536)

### **5.3 Docker networking: localhost vs host.docker.internal**

* Inside a container:
  * **localhost** refers to the container itself
* If Postgres runs on host and MCP runs in container:
  * **use **host.docker.internal:5432** (macOS/Windows)**
  * **on Linux, add **--add-host=host.docker.internal:host-gateway
* If everything is in one compose network:
  * prefer service discovery (**postgres:5432**) rather than host bridging

---

## **6) Model profiles: resolution and behavior**

### **6.1 Profile names**

Common profiles:

* chat
* assessment
* **embeddings** (optional)

### **6.2 Resolution rules (simplified)**

* **DEFAULT_CHAT_MODEL_PROFILE** selects the default “chat” logical profile
* **DEFAULT_ASSESSMENT_MODEL_PROFILE** selects the default “assessment” logical profile
* **EMBEDDINGS_MODEL_PROFILE** selects which profile is used for embeddings

In the workflow:

* **assessment profile: **start.model_profile** → fallback **DEFAULT_ASSESSMENT_MODEL_PROFILE
* **embeddings profile: **EMBEDDINGS_MODEL_PROFILE** → fallback to **DEFAULT_CHAT_MODEL_PROFILE** or assessment**

### **6.3 MODEL_PROFILES_JSON overrides implicit profiles**

**If **MODEL_PROFILES_JSON** is present:**

* it is parsed and validated
* it becomes the source of truth for all profile configuration
* implicit Ollama/Azure fallback is skipped

Design rationale:

* explicit profiles remove ambiguity and reduce “it worked yesterday” drift.

---

## **7) Error handling and observability**

### **7.1 Request ID headers**

**providers/llm.py** extracts common request ids/correlation ids:

* x-ms-request-id
* x-request-id
* apim-request-id
* x-correlation-id
* traceparent

These are included in raised **LLMError** messages to help track issues in enterprise gateways.

### **7.2 Where errors surface**

* /health/llm** returns structured **{chat:{ok}, embeddings:{ok}}** status**
* risk workflow:
  * appends warnings for clause-level failures
  * fails fast for embedding dimension mismatch
  * fails the entire job if no clause results are produced

### **7.3 Debug strategy**

Recommended sequence:

1. GET /health
2. GET /health/llm
3. Run parse-only endpoint on a document
4. Run assessment with **mode=sync** for simplest debugging
5. Check **warnings[]** in **status** output

---

## **8) OpenWebUI Pipelines integration details**

The pipeline’s key design decision:

* file uploads appear in **body["files"]** during **inlet()**, but may be missing later.
* pipeline stores chosen file metadata into:
  * body["_supplier_risk_file"]

Then:

* downloads bytes from OpenWebUI:
  * GET /api/v1/files/{id}/content
* starts risk assessment:
  * POST /tools/risk_assessment/start
* polls status:
  * POST /tools/risk_assessment/status
* fetches report:
  * POST /tools/risk_assessment/report

UI streaming:

* OpenWebUI can show “Chunk too big” if final report is streamed as one huge chunk.
* pipeline includes chunking (**MAX_UI_CHUNK_CHARS**) to stream final report in smaller pieces.

---

## **9) “Why this design” rationale**

### **Deterministic workflow**

* Clause order is stable and predictable.
* Prompts constrain output schema and size.
* Enables reproducibility and auditability (critical for contract risk workflows).

### **Separate embeddings profile**

* Prevents accidental calls to chat endpoints for embeddings.
* Allows using different providers for:
  * chat reasoning (Azure/vLLM)
  * embeddings (Azure/local embedding model)
* Embedding dimension checks fail fast and prevent silent RAG corruption.

### **TOC skipping and heading-based clause boundaries**

* TOCs create false positives for clause boundaries.
* Using shallowest outline heading level matches “main clause” UX.
* Avoid splitting on list numbering keeps subclauses bundled for UI.

---

## **10) What would improve production readiness (not implemented)**

* Persistent job store (Redis/Postgres)
* Authn/authz for REST endpoints
* Rate limiting / quotas
* Structured logging correlation IDs across components
* Storage of generated reports as downloadable files in OpenWebUI
