# **MCP Document Parser Server — Setup & Integration Guide**

**This README is written for your ****Python MCP server** that parses **DOCX/PDF** (tracked changes + comments) and runs a **deterministic clause-by-clause risk assessment workflow** using  **pgvector RAG + LLM structured outputs** **.**

It covers:

1. **How to set up the MCP server from scratch**
2. **A drill-down of each component and tool**
3. **What to know when integrating into another system** (OpenWebUI, AI SDK apps, etc.)

---

## **1) Setup from scratch**

### **1.1 Prereqs**

* **Python**: 3.10+ recommended
* **Postgres + pgvector**: required if you want RAG retrieval (policy DB)
* **LLM endpoint** **:**
* Local **Ollama** (OpenAI-compatible **/v1** for chat, native **/api/embed** for embeddings), or
* Azure OpenAI (optional, used for “assessment” profile)

### **1.2 Clone and install**

```
git clone <your-repo-url> mcp-document-parser
cd mcp-document-parser
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you don’t have a **requirements.txt**, the core runtime requirements implied by your code are typically:

* fastapi**, **uvicorn
* pydantic**, **pydantic-settings
* httpx
* lxml
* sqlalchemy**, **psycopg** (or **psycopg2**)**
* **PyMuPDF** (for PDF parsing)
* **pgvector** (db side; python uses SQL + embedding)

### **1.3 Configure environment**

Create a **.env** in the project root (or wherever your server reads it) similar to yours:

```
# -----------------------
# MCP Server
# -----------------------
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765
LOG_LEVEL=INFO

# -----------------------
# Policy DB / pgvector (RAG)
# -----------------------
POLICY_DB_URL=postgresql+psycopg://postgres:postgres@localhost:5432/pgvector_mcp_document
POLICY_DEFAULT_COLLECTION=default
POLICY_TOP_K_DEFAULT=3

# -----------------------
# Model profiles
# -----------------------
DEFAULT_CHAT_MODEL_PROFILE=chat
DEFAULT_ASSESSMENT_MODEL_PROFILE=assessment
ALLOW_MODEL_OVERRIDE=false

# -----------------------
# Local Ollama defaults
# -----------------------
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gpt-oss:20b

# --- Embeddings selection ---
EMBEDDINGS_MODEL_PROFILE=chat

# --- Ollama embeddings (native API) ---
OLLAMA_EMBEDDINGS_URL=http://localhost:11434/api/embed
OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text:latest
EMBEDDINGS_DIM=768

# -----------------------
# Azure OpenAI (optional)
# -----------------------
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=
AZURE_OPENAI_EMBEDDINGS_API_VERSION=
```

**Important:** Keep **POLICY_TOP_K_DEFAULT** aligned with what your client and schemas use. You’ve moved toward **3** to reduce prompt bloat and output truncation risk.

### **1.4 Start Postgres + pgvector (docker)**

Example:

```
docker run --name pgvector_mcp \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

Then create your DB:

```
createdb -h localhost -p 5432 -U postgres pgvector_mcp_document
```

If you have a SQL setup script, run it. At minimum you need:

* a table storing policy chunks
* a vector column matching **EMBEDDINGS_DIM**
* an index on the vector column
* a **collection** column if you support multiple policy sets

*(If you want, I can provide a canonical **policy_chunks** schema + ingestion script in a separate response.)*

### **1.5 Start Ollama (if using local)**

```
ollama serve
```

Pull models:

```
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

Verify endpoints:

* **Chat (OpenAI compatible): **http://localhost:11434/v1/chat/completions
* **Embeddings (native): **http://localhost:11434/api/embed

### **1.6 Run the MCP server**

Your entry point is:

```
python -m mcp_server
```

You should see logs like:

* “Uvicorn running on http://0.0.0.0:8765”
* “GET /sse 200 OK” when a client connects
* “POST /messages?session_id=… 200 OK” when tools are invoked

---

## **2) MCP server drill-down: components and tools**

### **2.1 High-level structure**

Core responsibilities:

1. **MCP host layer**
   * Handles MCP sessions via SSE
   * Receives JSON-RPC messages for **tools/list** and **tools/call**
   * Returns tool results *in the JSON-RPC response body* (critical for clients)
2. **Parsing layer**
   * parse_docx** / **parse_docx2**: DOCX → **DocumentParseResult
   * parse_pdf**: PDF → **DocumentParseResult
   * **normalize_clauses**: optional transformation of parsed clauses
3. **Workflow layer**
   * **risk_assessment.start**: deterministic job runner
   * status**, **report**, **get_clause_result**, **cancel
4. **LLM + Embeddings provider layer**
   * OpenAI-compatible chat calls to Ollama/vLLM
   * Optional Azure OpenAI support
   * Separate embeddings endpoint/model support (Ollama **/api/embed**)
5. **RAG layer**
   * pgvector search: embed query → cosine distance search → top_k citations
6. **Store**
   * In-memory job store for assessment runs + outputs + warnings + progress

---

### **2.2** ****

### **config.py**

### ** — settings + model profiles**

**What it does**

* Loads env vars via Pydantic Settings
* **Builds ** **model profiles** **:**
  * **chat** profile (usually local Ollama, OpenAI-compatible **/v1**)
  * **assessment** profile (Azure if configured, else falls back to **chat**)
* **Supports ** **separate embeddings configuration** **:**
  * embeddings profile can be **chat** or **assessment**
  * embeddings can use native Ollama **/api/embed**

**Important behaviors**

* Strips whitespace from config strings (prevents invisible “bad URL” issues)
* Validates that profiles have required fields

---

### **2.3** ****

### **schemas.py**

### ** — contracts and tool schemas**

**What it defines**

* Parse output models: **Clause**, **DocumentParseResult**, metadata, redlines/comments/changes
* Workflow output models:
  * **ClauseAssessment** (strict JSON validated)
  * ClauseRiskResult** (adds **text_with_changes** for UI)**
  * RiskAssessmentStartOutput**, **Status**, **Report**, etc.**
* MCP tool registration list **MCP_TOOLS** (for **tools/list**)

**Important behaviors**

* Validators normalize:
  * **risk_level** to low|medium|high
  * **issues[].severity** to low|medium|high
* Tool schema defaults:
  * **top_k** defaults to 3 (lower prompt bloat)

---

### **2.4** ****

### **tools/parse_docx.py**

### ** — DOCX parsing (tracked changes + comments)**

**What it does**

* **Reads **word/document.xml** and (optionally) **word/comments.xml
* Produces a list of **Clause** objects with:
  * text
  * raw_spans** (**normal/inserted/deleted/comment_ref/...**)**
  * **redlines** (insertions/deletions)
  * **comments** (margin comments)
  * **changes** (structured change items + comment anchor snippets)
* Uses block-level revision containers:
  * handles paragraphs wrapped by **<w:ins>**, **<w:del>**, etc.

**Clause boundary logic**

* If document contains ARTICLE/SECTION headings:
  * split by those headings
  * do NOT split by numeric subheadings like 1.1
* Else:
  * split by numeric headings at shallowest numeric depth

---

### **2.5** ****

### **tools/parse_docx2.py**

### ** — “TOC + heading header” format parser (new)**

**Why it exists**

* Your new document format includes:
  * a **TOC** at the start that must be ignored
  * headings that appear as navigation pane items
  * auto-numbering that may not show in **w:t**

**What it adds**

* Skips TOC in multiple ways:
  * **TOC styles (**TOC1/TOC2/...**, **TOCHeading**)**
  * paragraphs generated by a TOC field (**fldChar** / **instrText**)
  * literal “TABLE OF CONTENTS”
* Prefers Word-native boundary detection:
  * paragraph outline levels (**w:outlineLvl**)
  * style-derived outline levels (**styles.xml**)
* Best-effort auto-number label reconstruction:
  * **reads **word/numbering.xml
  * uses **w:numPr** (**numId**, **ilvl**) to build labels when the number isn’t in text

**What stays the same**

* tracked changes extraction (ins/del/move)
* comment extraction + anchor capture
* structured change summary creation (**ClauseChanges**)

---

### **2.6** ****

### **tools/parse_pdf.py**

### ** — PDF parsing**

*(You didn’t paste it in this chat, but based on your system summary)*

**Expected behavior**

* Uses PyMuPDF to extract page text
* Tries to detect:
  * clause boundaries (typically via regex heuristics)
  * PDF annotations (comments) where possible
* Returns **DocumentParseResult** consistent with DOCX

---

### **2.7** ****

### **tools/normalize_clauses.py**

### ** — post-processing clause list**

**What it does**

* **Takes a **DocumentParseResult
* Optionally applies boundaries (e.g., from an LLM boundary detector)
* Dedupes by normalized text hash
* Reassigns stable clause IDs
* Recomputes hierarchy from **level**

---

### **2.8** ****

### **workflows/risk_assessment/runner.py**

### ** — deterministic assessment engine**

**Entry**

* start_risk_assessment(start: RiskAssessmentStartInput)

**Workflow**

1. Parse (if needed), otherwise use provided **parse_result**
2. Create assessment job record (store)
3. For each clause in order:
   * build **clause_text_full** including “Changes:” block (comments/redlines)
   * store the full text for UI/reporting
   * truncate clause text for prompt safety
   * retrieve policies via pgvector RAG
   * build bounded policy block
   * call LLM with strict JSON rules → validate into **ClauseAssessment**
   * store results (assessment + rich result)
4. Compute totals + summary
5. Mark status completed/failed

**Reliability features**

* Embedding dimension health check at start
* Prompt size caps to reduce truncation
* Retry once on JSON validation failure with smaller prompt

---

### **2.9** ****

### **workflows/risk_assessment/store.py**

### ** — in-memory job store**

**What it stores**

* Job status + progress
* warnings and error string
* **results in ****original clause order**
* back-compat and rich results:
  * **clause_results** (ClauseAssessment)
  * **clause_risk_results** (ClauseRiskResult)
  * **clause_text_by_id** (to render text even if only assessment exists)

**Important behavior**

* **report_output()** returns results in the **planned clause order**
* Prefers rich results where available
* Wraps assessments with stored text when needed

---

### **2.10** ****

### **tools/risk_assessment.py**

### ** — MCP tool handlers**

Implements MCP-facing methods:

* risk_assessment.start
* risk_assessment.status
* risk_assessment.get_clause_result
* risk_assessment.report** (json or markdown)**
* risk_assessment.cancel

Also contains markdown rendering that:

* avoids duplicate clause headers
* supports either ClauseAssessment or ClauseRiskResult payloads

---

## **3) Integration notes: plugging into OpenWebUI or another AI SDK system**

### **3.1 What an external system needs to know**

This server provides an **MCP tool interface**. Any client that can:

* connect to **GET /sse**
* **send JSON-RPC messages to **POST /messages?session_id=...

…can call tools, list tools, and receive structured outputs.

Most “agent frameworks” want:

* tool listing
* tool invocation
* JSON results

Your key fix (already done) was ensuring **/messages** returns the **actual JSON-RPC response** including tool results — not just **{status:"ok"}**.

---

### **3.2 Common integration patterns**

#### **Pattern A: AI SDK / Next.js backend (your current approach)**

* The backend exposes an **/api/chat** route that:
  * passes tools to the model
  * forwards tool calls to MCP server
  * returns tool results to UI

Best practices:

* Use **upload-only** flow to store base64 and filename
* Let risk assessment parse internally if parsing wasn’t done yet
* **Always poll **risk_assessment.status** and snapshot **risk_assessment.report** for UI rendering**

#### **Pattern B: OpenWebUI tool server**

OpenWebUI can integrate tools and MCP servers via “tool calling” plumbing. Practical notes:

* You typically need a **bridge** that converts OpenWebUI tool invocations into MCP JSON-RPC calls.
* If OpenWebUI expects “OpenAI tool schema,” you map MCP’s **tools/list** output into an OpenAI-style tool list.
* Keep tool output compact:
  * your report objects can get large
  * prefer returning a report summary + clause_items
  * fetch full clause details on-demand via **get_clause_result**

---

### **3.3 Key behaviors to preserve when integrating elsewhere**

#### **A) Deterministic output + job semantics**

Your workflow is intentionally deterministic:

* ordered clause processing
* consistent output model validation
* store-driven status/progress

When integrating:

* treat **risk_assessment.start** as “enqueue job”
* UI/agent should poll status and fetch report snapshots
* do not assume the report is complete until status is **completed**

#### **B) Expect and handle warnings**

Missing clause outputs are almost always explained by:

* LLM output invalid JSON → clause skipped, warning recorded
* policy DB retrieval error → warning recorded

Integrations should expose **warnings[]** somewhere (even if hidden behind a “debug” panel).

#### **C) Token/size management**

You already learned the big one:

* **max output tokens matters** (JSON truncation = validation failure = dropped clauses)

Integration tips:

* keep prompt bounded (you do)
* keep tool results bounded (especially returning clause lists + full text)
* stream report markdown only when requested

---

### **3.4 Deployment + reliability notes**

#### **In-memory stores**

Two important ephemeral stores exist:

* MCP server store for assessments/results
* (If applicable) client-side store for uploads/parse results

If you deploy in containers / scale out:

* assessments will vanish on restart
* multiple replicas will not share job state

For production:

* move store to Redis/Postgres
* store the parse result and assessment outputs durably
* store raw uploads in blob storage (S3/Azure Blob)

#### **Concurrency**

Your runner currently processes clauses sequentially (deterministic). The schema includes **concurrency**, but implementation is sequential.

If you add concurrency later:

* ensure store updates remain ordered
* cap concurrency to avoid saturating LLM endpoint
* preserve deterministic ordering in the final report (store already does this)

---

### **3.5 Security notes (especially for enterprise)**

* Treat **file_base64** as sensitive: contracts often contain regulated data
* Avoid logging file content or clause text in INFO logs
* Use TLS if crossing networks
* Restrict MCP server access to trusted clients (network ACL, auth token, mTLS, etc.)

---

## **Quick “operator” checklist**

* **✅ **python -m mcp_server** runs**
* ✅ **GET /sse** returns 200
* ✅ client can call **tools/list**
* ✅ DOCX parsing returns clauses **excluding TOC**
* **✅ **risk_assessment.start(mode=async)** returns assessment_id**
* ✅ **risk_assessment.status** progresses to completed
* ✅ **risk_assessment.report(format=json)** returns ordered clause_results + totals + summary
* ✅ warnings are surfaced somewhere in the client

---

If you want, I can also produce:

* a **minimal ****requirements.txt** matched to your code imports
* a **pgvector schema + ingestion script** (policies -> chunking -> embeddings -> insert)
* **a ****“How to add parse_docx2 as an MCP tool + route it”** checklist (server dispatch + MCP_TOOLS entry)
