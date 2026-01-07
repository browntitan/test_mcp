# MCP Document Parser

A production-ready **Python MCP (Model Context Protocol) server** for parsing legal documents:

- **DOCX**: extracts text, **tracked changes** (insertions/deletions), and **comments** (OOXML annotations)
- **PDF**: extracts text and common **annotations** (sticky notes, highlights-with-content, strikeouts)
- Returns **structured clauses** with stable clause IDs, hierarchy, redlines, comments, and source locations

The server supports:

- **MCP over SSE** (primary transport): `GET /sse` + `POST /messages` (JSON-RPC)
- **Direct HTTP** endpoints for simple integration

---

## Project layout

```
mcp-document-parser/
├── mcp_server/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── schemas.py
│   ├── server.py
│   └── tools/
│       ├── __init__.py
│       ├── parse_docx.py
│       ├── parse_pdf.py
│       └── normalize_clauses.py
├── tests/
│   ├── __init__.py
│   └── test_tools.py
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```

Environment variables:

- `MCP_SERVER_HOST` (default `0.0.0.0`)
- `MCP_SERVER_PORT` (default `8765`)
- `LOG_LEVEL` (default `INFO`)

### 3) Run the server

```bash
python -m mcp_server
```

---

## Direct HTTP API

### Health

```bash
curl http://localhost:8765/health
# {"status":"healthy","version":"0.1.0"}
```

### Parse DOCX

```bash
curl -X POST http://localhost:8765/tools/parse_docx \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/contract.docx"}'
```

Or use base64 input:

```bash
DOC_B64=$(python - <<'PY'
import base64
print(base64.b64encode(open('/tmp/contract.docx','rb').read()).decode())
PY
)

curl -X POST http://localhost:8765/tools/parse_docx \
  -H "Content-Type: application/json" \
  -d "{\"file_base64\": \"$DOC_B64\"}"
```

### Parse PDF

```bash
curl -X POST http://localhost:8765/tools/parse_pdf \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/contract.pdf"}'
```

### Normalize clauses

```bash
curl -X POST http://localhost:8765/tools/normalize_clauses \
  -H "Content-Type: application/json" \
  -d '{"parse_result": {"document": {"filename":"x","media_type":"application/pdf"}, "clauses": [], "warnings": []}}'
```

> In practice you pass the full `DocumentParseResult` returned by `parse_docx` or `parse_pdf`.

---

## MCP over SSE transport

### 1) Open the SSE stream

```bash
curl -N "http://localhost:8765/sse?session_id=test"
```

The server sends:

- `event: endpoint`
- `data: /messages?session_id=test`

### 2) Send JSON-RPC messages

Initialize:

```bash
curl -X POST "http://localhost:8765/messages?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}'
```

List tools:

```bash
curl -X POST "http://localhost:8765/messages?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

Call a tool:

```bash
curl -X POST "http://localhost:8765/messages?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"parse_docx","arguments":{"file_path":"/tmp/contract.docx"}}}'
```

Tool call responses use MCP-compatible content blocks:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{...JSON stringified DocumentParseResult...}"
      }
    ]
  }
}
```

### Supported MCP methods

- `initialize`
- `initialized`
- `tools/list`
- `tools/call`
- `ping`

Protocol version: **`2024-11-05`**

---

## TypeScript client compatibility

This server is compatible with MCP clients (including TypeScript clients using the Vercel AI SDK) using either:

- direct HTTP tool endpoints, or
- MCP JSON-RPC over SSE (`/sse` + `/messages`).

Example direct tool call (HTTP):

```ts
const response = await fetch("http://localhost:8765/tools/parse_docx", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    file_path: "/path/to/document.docx",
    options: { extract_tracked_changes: true }
  })
});

const result = await response.json();
```

---

## Running tests

```bash
pytest
```

The tests programmatically generate:

- minimal DOCX files (ZIP + OOXML XML)
- minimal PDF files (PyMuPDF)

And validate tracked changes, comments, clause detection, normalization, and error handling.

---

## Notes on parsing behavior

### Clause IDs

Clause IDs are stable and follow:

```python
f"clause_{index:04d}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
```

### DOCX tracked changes

- Insertions (`<w:ins>`) appear in **clean text** and in `redlines.insertions`.
- Deletions (`<w:del><w:delText>`) are recorded in `redlines.deletions` and do **not** appear in clean text.

### PDF annotations

The server attempts to extract common annotations:

- type `0` / `1`: text-like annotations (sticky notes)
- type `8`: highlight **with content**
- type `11`: strikeout (extracts text covered by the annotation rectangle)

If no annotations are found, a warning is included because not all PDF viewers store annotations in a standard way.
