# MCP Document Parser Client (Next.js + Vercel AI SDK)

A minimal **front-end client** for your **Python MCP Document Parser** (`mcp-document-parser`) built with:

- **Next.js (App Router)**
- **Vercel AI SDK** (`ai` + `@ai-sdk/react`)
- **OpenAI provider** (`@ai-sdk/openai`)
- **Document parsing** via your Python server (DOCX/PDF) + **tool-based clause iteration** for the LLM

This app lets you:

1. Upload a **DOCX** or **PDF**
2. Parse it via the Python server
3. Chat with an LLM that can **iterate through clauses** using tools like:
   - `list_clauses`
   - `get_clause`
   - `search_clauses`

> Note: This client stores parsed documents **in memory** (for local/dev). For production, persist `parseResult` in a database or object store.

---

## Prerequisites

- Node.js **18+**
- Your Python server running (default port **8765**)
- An OpenAI API key

---

## Quick Start

### 1) Start the Python MCP server

From your server project:

```bash
# (inside mcp-document-parser/)
python -m mcp_server
```

Verify:

```bash
curl http://localhost:8765/health
# {"status":"healthy","version":"0.1.0"}
```

### 2) Start this Next.js client

```bash
# from this project root:
cp .env.example .env.local
# edit .env.local and add OPENAI_API_KEY

npm install
npm run dev
```

Open:

- http://localhost:3000

---

## How the “LLM iterates through the parsed document”

After parsing, the document is stored server-side (in-memory) and the chat endpoint exposes **tools** to the model:

- `list_clauses(level?, limit?, offset?)` – get an outline + IDs
- `get_clause(clause_id)` – fetch full clause text + redlines/comments counts
- `search_clauses(query, limit?)` – find relevant clauses by keyword

The assistant is prompted to use these tools whenever it needs to read sections of the document, so it can “walk” the document clause-by-clause without you having to paste the entire text into the prompt.

---

## API

### `POST /api/parse`

Parses an uploaded file by forwarding it to your Python server.

**Body:**
```json
{
  "filename": "contract.docx",
  "file_base64": "....",
  "options": { "extract_tracked_changes": true }
}
```

**Returns:**
```json
{
  "docId": "uuid",
  "parseResult": { "document": { ... }, "clauses": [...], "warnings": [] }
}
```

### `POST /api/chat`

Streams chat responses using Vercel AI SDK. Requires `docId`.

**Body:**
```json
{
  "docId": "uuid",
  "messages": [{ "id":"...", "role":"user", "parts":[{ "type":"text", "text":"..." }] }]
}
```

---

## Notes / Production Hardening

- **Persistence**: replace the in-memory store (`lib/documentStore.ts`) with Redis/Postgres/S3.
- **Auth**: add auth to `/api/parse` and `/api/chat` if handling sensitive documents.
- **Large files**: consider multipart upload + streaming to object storage; base64 JSON can be large.
- **Model choice**: set `OPENAI_MODEL` in `.env.local`.

---

## Troubleshooting

- **Python server not reachable**: confirm `MCP_PARSER_HTTP_BASE_URL` is correct.
- **Chat works but tools don’t run**: ensure `/api/chat` sets `maxSteps` > 1 (it does), and you have a valid `OPENAI_API_KEY`.

---

## License

MIT (you can change this for your project).
