# **MCP Policy Seeding & Risk Assessment Pipeline**

**This repository implements a ****deterministic, auditable policy‑aware risk assessment pipeline** built around:

* **pgvector** for policy embeddings and metadata‑driven retrieval
* **MCP (Model Context Protocol)** for tool‑based orchestration
* **LLM‑based clause analysis** with structured JSON output
* **Explicit policy citations** and traceability

The system is designed for **regulated / enterprise environments** (defense, supply chain, procurement, legal review) where:

* You *cannot* rely on web search
* You *must* justify conclusions against internal policy
* You need reproducibility, inspectability, and deterministic filters

---

## **High‑Level Architecture**

```
Documents (DOCX/PDF/TXT)
        │
        ▼
[ Parsing & Normalization ]
        │
        ▼
[ Policy Seeding → pgvector ]
        │   ├─ embeddings
        │   ├─ clause_number
        │   ├─ clause_title
        │   └─ termsets / metadata
        ▼
[ Risk Assessment Workflow ]
        │
        ├─ clause‑by‑clause evaluation
        ├─ vector + metadata retrieval
        ├─ structured LLM reasoning
        └─ JSON + citations output
        ▼
[ UI / Report / API Consumers ]
```

---

## **Key Design Principles**

### **1. Deterministic Retrieval First, LLM Second**

**The LLM ** **does not decide what policy applies** **.**

Policy relevance is determined by:

* collection
* clause_number
* termset
* vector similarity (secondary)

**The LLM is used ** **only to reason over already‑retrieved policy excerpts** **.**

---

### **2. Metadata Is a First‑Class Citizen**

Every policy chunk includes explicit metadata:

```
{
  "clause_number": "58",
  "clause_title": "Federal Acquisition Regulation (FAR)",
  "termsets": ["001", "002"],
  "filename": "58. FAR Flowdown.txt"
}
```

This allows:

* precise filtering
* explainable retrieval
* audit‑ready evidence packs

---

### **3. JSON‑Only, Schema‑Validated LLM Output**

All LLM calls:

* **run in ****JSON‑only mode**
* validate against a strict Pydantic schema
* fail fast on malformed output

This guarantees:

* UI stability
* report generation safety
* no hallucinated fields

---

## **Repository Structure**

```
mcp-document-parser/
├─ scripts/
│  └─ seed_policies.py        # Seeds policy text into pgvector
├─ mcp_server/
│  ├─ rag/
│  │  └─ pgvector.py          # Vector + metadata retrieval
│  ├─ workflows/
│  │  └─ risk_assessment/
│  │     └─ runner.py         # Clause-by-clause LLM loop
│  ├─ providers/
│  │  └─ llm.py               # LLM client + JSON enforcement
│  ├─ schemas.py              # Pydantic output schemas
│  └─ config.py               # Env-driven configuration
```

---

## **Policy Seeding (**

## **seed_policies.py**

## **)**

### **What Seeding Does**

For each **.txt** or **.md** policy file:

1. **Robustly decodes text** (UTF‑8 + cp1252 repair)
2. **Normalizes filenames** to remove encoding garbage (e.g. **â€**)
3. Extracts:
   * clause_number
   * clause_title
   * termsets
4. Chunks text deterministically
5. Generates embeddings
6. Upserts rows into **policy_chunks**

### **Why Filename‑Based Metadata Matters**

Clause association is derived from **file names**, e.g.:

```
58. Federal Acquisition Regulation (FAR)
```

If filenames are malformed or encoded incorrectly, **clause_number** becomes **null**, breaking retrieval.

**This repository includes ****automatic normalization and mojibake repair** to prevent that.

---

### **Seeding Command**

```
python scripts/seed_policies.py \
  --input-dir ./test_policies \
  --collection policy_chunks \
  --clear-collection \
  --upsert
```

Key flags:

* **--clear-collection** → delete existing rows for that collection
* **--upsert** → deterministic overwrite

---

## **Database Schema (**

## **policy_chunks**

## **)**

```
policy_chunks (
  policy_id   TEXT,
  chunk_id    TEXT,
  collection  TEXT,
  text        TEXT,
  metadata    JSONB,
  embedding   VECTOR(dim),
  created_at  TIMESTAMPTZ
)
```

Indexes:

* **GIN on **metadata
* **functional index on **(metadata->>'clause_number')
* HNSW vector index for cosine similarity

---

## **Risk Assessment Workflow**

### **Retrieval Flow**

For each clause:

1. Filter by **collection**
2. **Filter by **clause_number
3. Filter by **termset** (if provided)
4. Fallback strategies (drop termset, then drop clause)
5. Vector similarity ranking

### **LLM Reasoning**

The LLM receives:

* Clause text
* Clause changes (if any)
* Retrieved internal policy excerpts

It produces:

* Risk level
* Risk score
* Justification statement
* Identified issues
* Optional citations (subset of retrieved excerpts)

**The LLM ** **cannot invent policy** **.**

---

## **System Prompts & LLM Control**

**There are ** **two system prompts** **:**

### **1. Workflow System Prompt (Configurable)**

Set via **.env**:

```
RISK_ASSESSMENT_SYSTEM_PROMPT_PATH=./prompts/risk_assessment_system.txt
```

Controls:

* reasoning role
* justification tone
* citation rules
* structure of analysis

### **2. JSON Constraint Prompt (Internal)**

Enforced in **llm.py**:

* Forces valid JSON
* Prevents markdown / prose leakage

Both apply simultaneously.

---

## **Logging & Debugging**

Key log prefixes:

* **RAG:** — retrieval logic
* **PGVECTOR:** — database queries
* **LLM:** — model calls & validation
* **STORE:** — persistence

Example:

```
RAG: primary results clause_id=... count=3
LLM: citations_retrieved=3 citations_returned=1
```

This makes it obvious where failures occur.

---

## **Metadata Inspection (CSV Export)**

PowerShell‑safe export:

```
docker exec -i pgvector_mcp_document psql -U postgres -d pgvector_mcp_document `
  -c "\copy (SELECT policy_id, chunk_id, metadata::text FROM policy_chunks) TO STDOUT WITH CSV HEADER" `
  | Out-File -Encoding utf8 .\policy_metadata.csv
```

---

## **Common Failure Modes**

| **Symptom** | **Cause**                             |
| ----------------- | ------------------------------------------- |
| Wrong citations   | **clause_number**missing or malformed |
| No citations      | DB unreachable / filter mismatch            |
| Empty metadata    | Filename encoding damage                    |
| JSON errors       | Schema mismatch or truncated output         |

---

## **Intended Use Cases**

* Contract flowdown review
* Supplier T&Cs risk scoring
* Redline justification generation
* Policy compliance audits
* Secure, offline RAG systems

---

## **Final Notes**

This system intentionally prioritizes:

* determinism over creativity
* auditability over verbosity
* structure over free‑form reasoning
