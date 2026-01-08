# **MCP Document Parser Client (Next.js + Vercel AI SDK) — README**

This repo is a **Next.js (App Router)** web client that lets you:

* Upload a **DOCX/PDF** (stored in-memory server-side as base64)
* Optionally parse it into **clauses** using your Python **MCP document parser server**
* Chat with an LLM (Ollama OpenAI-compatible **/v1**) using **Vercel AI SDK**
* Let the model call tools:
  * local tools (list/search/get clauses)
  * MCP tools dynamically (**tools/list** → generated tool wrappers)
  * deterministic **risk_assessment.*** workflow via MCP

---

## **1) Prerequisites**

### **Required**

* **Node.js 18+** (Node 20 recommended)
* **Your Python MCP server running** (default **http://localhost:8765**)
* **Ollama running** (default http://localhost:11434**)**
  * **Chat endpoint: **http://localhost:11434/v1/chat/completions
  * Model pulled (e.g. **qwen2:7b**)

### **Optional but recommended**

* pgvector DB + seeded policies (so **risk_assessment** can do RAG)

---

## **2) Install and run**

From the client repo root:

```
npm install
npm run dev
```

Then open:

* http://localhost:3000

---

## **3) Environment variables (**

## **.env.local**

## **)**

Create **.env.local** in the client repo root.

### **Example (your current config)**

```
# # OpenAI API key used by @ai-sdk/openai (server-side)
# OPENAI_API_KEY=sk-your-key-here
# # Optional: pick a model id supported by your OpenAI account
# OPENAI_MODEL=gpt-4o-mini

OLLAMA_MODEL=qwen2:7b
OLLAMA_BASE_URL=http://localhost:11434/api

# Where your Python MCP document parser is running:
MCP_PARSER_HTTP_BASE_URL=http://localhost:8765
MCP_PARSER_SSE_URL=http://localhost:8765/sse
MCP_SESSION_ID=nextjs-bridge
```

### **What each variable does**

#### **Ollama + model selection**

* OLLAMA_MODEL
  The model id you want to chat with via AI SDK (e.g. **qwen2:7b**, **gpt-oss:20b**)
* OLLAMA_BASE_URL
  Base URL for Ollama. Your code normalizes:
  * if it ends in **/api**, it converts to **/v1** for OpenAI-compatible chat.

> In **app/api/chat/route.ts**, the AI SDK provider is created with **createOpenAI({ baseURL: ... })** and uses **openai.chat(OLLAMA_MODEL)**. The **apiKey** is set to **OPENAI_API_KEY || 'ollama'** (Ollama doesn’t require a real key locally).

#### **MCP parser server connection**

* MCP_PARSER_HTTP_BASE_URL
  Used by **lib/mcp.ts** for direct HTTP fallback tool calls:
  * /tools/parse_docx
  * /tools/parse_pdf
* MCP_PARSER_SSE_URL
  Used by **@ai-sdk/mcp** client in **lib/mcp.ts** for MCP SSE sessions (tool discovery + tool execution).
* MCP_SESSION_ID
  Used by **app/api/chat/route.ts** for the JSON-RPC over HTTP “bridge” path:
  * **POST **${MCP_BASE_URL}/messages?session_id=...

**Important mismatch to fix:**

**In **app/api/chat/route.ts**, the code uses:**

```
const MCP_BASE_URL = (process.env.MCP_SERVER_URL || 'http://localhost:8765')
```

**…but your **.env.local** uses **MCP_PARSER_HTTP_BASE_URL**.**

**✅ Recommendation: set **MCP_SERVER_URL=http://localhost:8765** in **.env.local** OR update the code to use **MCP_PARSER_HTTP_BASE_URL** consistently.**

---

## **4) How the app works (user flow)**

### **4.1 Upload-only (fast path)**

When a user selects a file in the UI (**page.tsx**):

* The client immediately calls **POST /api/upload**
* upload/route.ts**:**
  * **reads **{ filename, file_base64 }
  * enforces a base64 size limit (~25MB raw)
  * generates **docId**
  * stores upload in server memory (**putUpload**)
  * **returns **{ docId }

This enables:

* chat
* risk assessment
  …even if the document hasn’t been parsed yet.

### **4.2 Parse (enables clause browsing UI)**

If the user clicks **Parse**:

* **UI calls **POST /api/parse** with **{ filename, file_base64, options }
* parse/route.ts** calls MCP **parse_docx** or **parse_pdf
  * preferring MCP SSE tool invocation (**callToolViaMCP**)
  * falling back to direct HTTP endpoints (**callToolViaHttp**)
* Stores BOTH:
  * raw upload (**putUpload**)
  * parsed doc (**putParsed**)
* **returns **{ docId, parseResult }

Now the UI can:

* render clause list
* show a selected clause and its change summary

### **4.3 Chat + tools**

The chat panel uses **useChat()** with:

* API route: **/api/chat**
* **transport: **DefaultChatTransport

On each user message, **page.tsx** sends:

```
body: {
  docId,
  focusClauseId: selectedClauseId,
}
```

---

## **5) Backend routes and what they do**

### **app/api/upload/route.ts**

**Purpose:** Store base64 upload so the chat/risk workflow can parse on demand later.

* **Validates **filename** and **file_base64
* **Enforces size limit: **MAX_BASE64_CHARS = 35_000_000
* **Stores upload in **lib/documentStore.ts** under **uploadStore

Returns: **{ docId }**

---

### **app/api/parse/route.ts**

**Purpose:** Parse immediately and store structured clause output.

* Selects tool: **parse_docx** or **parse_pdf** based on filename
* Calls the MCP server via SSE using **@ai-sdk/mcp**:
  * callToolViaMCP(toolName, args)
* Extracts text from MCP tool output:
  * extractTextFromMcpToolOutput(...)
* **Parses JSON → **DocumentParseResult
* Stores:
  * upload (base64) for future workflows
  * parsed output for clause browsing + local tools

**Returns: **{ docId, parseResult }

---

### **app/api/chat/route.ts**

**Purpose:** Main “agent” route (AI SDK streaming + tools).

Key responsibilities:

1. Load document context:
   * **getDocument(docId)** → parsed clauses (if available)
   * **getUpload(docId)** → base64 upload fallback
2. Define **local tools** for clause interaction (when parsed doc exists):
   * document_info
   * list_clauses
   * get_clause
   * search_clauses
   * focus_clause
3. Define risk workflow wrapper tools:
   * run_risk_assessment
     * **calls MCP **risk_assessment.start
     * can pass either:
       * **parse_result** (fast) if parsed doc exists
       * **OR **file_base64 + filename + file_type** if only upload exists**
     * optionally polls status until completion
     * always fetches report JSON snapshot
     * extracts **clause_items** for UI rendering
   * get_risk_assessment_status
   * get_risk_assessment_report
4. Dynamically expose **all MCP tools**
   * **Calls **tools/list
   * Wraps each in an AI SDK tool
   * Sanitizes names (dots → **__** etc.)
5. Streams response:
   * streamText({ model, tools, messages, system, maxSteps })
   * returns a UI streaming response

---

## **6) In-memory storage model (**

## **lib/documentStore.ts**

## **)**

This app stores state in-memory on the **Next.js server process**.

Two stores:

### **A) Parsed doc store**

```
Map<docId, StoredDocument> // parsed or normalized
```

### **B) Upload store (raw file base64)**

```
Map<docId, StoredUpload>
```

Both:

* persist across Next.js dev hot reload via **globalThis**
* **expire after **TTL_MS = 1 hour

**Important integration note:**

In production (serverless or multiple replicas), in-memory storage is not durable. For production, move to:

* Redis (doc + upload cache)
* S3/Azure Blob for file storage
* DB for parse results / job states

---

## **7) Tool design (how the assistant “uses” the document)**

### **Local document tools**

These operate on **parseResult.clauses** stored in memory.

* **list_clauses** returns lightweight outline info + preview
* **get_clause** returns full text + “Changes:” block (plain text)
* **search_clauses** is a simple substring search over label/title/text
* **focus_clause** uses the UI-selected clause so the model can interpret “this clause”

### **Risk assessment wrapper**

The most important tool is:

#### **run_risk_assessment**

* does NOT require docId in tool args (by design)
* always uses the doc loaded for the current chat request
* chooses payload:
  * parsed doc → **parse_result**
  * **upload-only → **file_base64**, **filename**, **file_type
* calls:
  * risk_assessment.start
  * optionally polls:
    * risk_assessment.status
  * fetches:
    * risk_assessment.report** (JSON)**
    * optional markdown too
* returns:
  * assessment_id
  * report
  * **clause_items** (UI-friendly per-clause cards)

---

## **8) UI behavior (**

## **page.tsx**

## **)**

### **Left panel: document**

* file picker
* auto-upload on selection (so chat/risk can run without parse)
* parse button for clause browsing
* parse options toggles:
  * DOCX: tracked changes, comments, raw spans
  * PDF: annotations, raw spans
* clause list (first 250 shown for performance)
* selected clause details + changes summary

### **Right panel: chat**

* uses AI SDK **useChat**
* renders:
  * assistant text
  * tool output
  * per-clause risk cards (if tool output includes **clause_items**)

---

## **9) Integration requirements (what must be running)**

### **Minimum to use the UI**

* Next.js dev server (**npm run dev**)
* Python MCP server (default **:8765**)
* Ollama (default **:11434**)

### **To run full risk assessment with RAG**

* pgvector running + seeded policies
* MCP server configured with:
  * POLICY_DB_URL
  * embedding config (**OLLAMA_EMBEDDINGS_URL**, etc.)

---

## **10) Troubleshooting**

### **“Document not found. Upload/parse again”**

* **docId** expired from in-memory TTL
* Next.js server restarted
* Fix: re-upload or parse again

### **MCP tool calls fail**

* Ensure MCP server is reachable:
  * curl http://localhost:8765/sse** (should connect)**
* If **/api/parse** fails MCP SSE, it falls back to HTTP endpoints:
  * POST http://localhost:8765/tools/parse_docx

### **Chat calls MCP using the wrong env var**

**As mentioned: **route.ts** currently reads **MCP_SERVER_URL**, but **.env.local** uses **MCP_PARSER_HTTP_BASE_URL**.**

Fix one of:

* **add **MCP_SERVER_URL=http://localhost:8765
* **or update code to use **MCP_PARSER_HTTP_BASE_URL

### **Upload too large (413)**

* **Increase **MAX_BASE64_CHARS** in **upload/route.ts
* Or implement multipart upload / blob storage for production

---

## **11) Repo layout (“client schema”)**

```
mcp-document-parser-client/
├── .env.local
├── app/
│   ├── page.tsx
│   └── api/
│       ├── upload/
│       │   └── route.ts
│       ├── parse/
│       │   └── route.ts
│       └── chat/
│           └── route.ts
└── lib/
    ├── documentStore.ts
    ├── mcp.ts
    └── types.ts
```

---

## **12) Operational notes / production hardening**

If you deploy beyond local dev:

* **Move uploads out of memory**
  * store file bytes in S3/Azure Blob
  * keep only docId pointers in memory
* **Move parsed results out of memory**
  * DB table keyed by docId
  * cache in Redis for speed
* **Handle multi-instance deployments**
  * in-memory maps won’t be shared across replicas
  * docId might route to a different instance and “disappear”
* **Secure MCP server**
  * network ACLs
  * auth tokens / mTLS if crossing hosts

---

If you share **package.json** (and optionally your **next.config**), I can tailor the “exact commands” section (Node version, scripts, port overrides, build/start for production) to match your repo precisely.
