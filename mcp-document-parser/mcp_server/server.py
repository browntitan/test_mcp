from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional
import base64
from fastapi import File, Form, UploadFile
import structlog
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .config import get_settings
from .providers.llm import ChatMessage, LLMError, get_llm
from .schemas import (
    MCP_TOOLS,
    ClauseListNormalized,
    DocumentParseResult,
    NormalizeClausesInput,
    ParseDocxInput,
    ParsePdfInput,
    RiskAssessmentCancelInput,
    RiskAssessmentGetClauseResultInput,
    RiskAssessmentReportInput,
    RiskAssessmentStartInput,
    RiskAssessmentStatusInput,
    to_jsonable,
)
from .tools import MCP_TOOL_NAMES, normalize_clauses, parse_docx, parse_pdf, risk_assessment

PROTOCOL_VERSION = "2024-11-05"


def _resolve_profile_name(settings: Any, preferred: Optional[str], fallback: str) -> str:
    """Resolve a model profile name against Settings.model_profiles.

    Ensures /health/llm doesn't crash if a profile name is missing.
    """

    profiles = getattr(settings, "model_profiles", {}) or {}
    p = (preferred or "").strip()
    if p and p in profiles:
        return p

    fb = (fallback or "").strip()
    if fb and fb in profiles:
        return fb

    for k in ("assessment", "embeddings", "chat"):
        if k in profiles:
            return k

    if profiles:
        return sorted(list(profiles.keys()))[0]
    return "chat"


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


class SessionManager:
    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def get_queue(self, session_id: str) -> asyncio.Queue[Dict[str, Any]]:
        async with self._lock:
            q = self._queues.get(session_id)
            if q is None:
                q = asyncio.Queue()
                self._queues[session_id] = q
            return q

    async def get_existing_queue(self, session_id: str) -> Optional[asyncio.Queue[Dict[str, Any]]]:
        async with self._lock:
            return self._queues.get(session_id)

    async def send(self, session_id: str, message: Dict[str, Any]) -> None:
        q = await self.get_queue(session_id)
        await q.put(message)

    async def close(self, session_id: str) -> None:
        async with self._lock:
            self._queues.pop(session_id, None)


sessions = SessionManager()


def create_app() -> FastAPI:
    _configure_logging()

    app = FastAPI(title="mcp-document-parser", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------
    # REST wrapper endpoints (for OpenWebUI Actions)
    # ------------------
    # Import here (inside create_app) so module import order doesn't matter and we can
    # attach the router to the FastAPI instance we just created.
    from .rest_risk import router as risk_router

    app.include_router(risk_router)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "version": __version__}


    @app.on_event("startup")
    async def _startup_log_profiles() -> None:
        try:
            s = get_settings()
            profiles = list((getattr(s, "model_profiles", {}) or {}).keys())
            log.info(
                "server startup: model profiles configured",
                profiles=profiles,
                default_chat=getattr(s, "default_chat_profile", None),
                default_assessment=getattr(s, "default_assessment_profile", None),
                embeddings_profile=getattr(s, "embeddings_model_profile", None),
                embeddings_dim=getattr(s, "embeddings_dim", None),
            )
        except Exception as e:
            log.exception("server startup: failed to log model profiles", error=str(e))


    @app.get("/health/llm")
    async def health_llm() -> Dict[str, Any]:
        """Validate configured LLM + embeddings connectivity.

        This is especially useful for Azure OpenAI Gov debugging:
        - Confirms chat deployment works
        - Confirms embeddings deployment works
        - Confirms embedding dimension matches EMBEDDINGS_DIM
        """

        s = get_settings()

        assessment_profile = _resolve_profile_name(
            s,
            preferred=getattr(s, "default_assessment_profile", None),
            fallback="assessment",
        )

        embeddings_profile = _resolve_profile_name(
            s,
            preferred=getattr(s, "embeddings_model_profile", None),
            fallback="embeddings",
        )

        out: Dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "assessment_profile": assessment_profile,
            "embeddings_profile": embeddings_profile,
            "embeddings_dim_expected": getattr(s, "embeddings_dim", None),
            "chat": {"ok": False},
            "embeddings": {"ok": False},
        }

        # Chat check
        try:
            llm = get_llm(assessment_profile)
            txt = await llm.chat_text(
                messages=[
                    ChatMessage(role="system", content="Reply with 'ok'."),
                    ChatMessage(role="user", content="ping"),
                ],
                temperature=0.0,
                max_tokens=16,
            )
            out["chat"] = {
                "ok": True,
                "sample": (txt or "").strip()[:64],
            }
        except Exception as e:
            out["status"] = "degraded"
            out["chat"] = {"ok": False, "error": str(e)}

        # Embeddings check
        try:
            el = get_llm(embeddings_profile)
            vecs = await el.embed_texts("dimension check")
            if not vecs or not vecs[0]:
                raise LLMError("Embeddings endpoint returned no vectors")
            dim = len(vecs[0])
            exp = getattr(s, "embeddings_dim", None)
            dim_ok = True
            if exp is not None:
                try:
                    dim_ok = dim == int(exp)
                except Exception:
                    dim_ok = True

            out["embeddings"] = {
                "ok": True,
                "dim": dim,
                "dim_ok": dim_ok,
            }
            if exp is not None:
                out["embeddings"]["expected_dim"] = int(exp)
            if not dim_ok:
                out["status"] = "degraded"
        except Exception as e:
            out["status"] = "degraded"
            out["embeddings"] = {"ok": False, "error": str(e)}

        return out

    # ------------------
    # Direct HTTP endpoints
    # ------------------

    @app.post("/tools/risk_assessment/start_upload")
    async def http_risk_assessment_start_upload(
        file: UploadFile = File(...),
        extract_comments: bool = Form(True),
        extract_tracked_changes: bool = Form(True),
        include_raw_spans: bool = Form(False),
    ) -> Dict[str, Any]:
        try:
            blob = await file.read()
            if not blob:
                raise HTTPException(status_code=400, detail="Empty file upload")

            b64 = base64.b64encode(blob).decode("utf-8")
            payload = {
                "file_path": None,
                "file_base64": b64,
                "filename": file.filename,
                "options": {
                    "extract_comments": extract_comments,
                    "extract_tracked_changes": extract_tracked_changes,
                    "include_raw_spans": include_raw_spans,
                },
            }

            # This will tell you immediately if your RiskAssessmentStartInput expects different fields.
            inp = RiskAssessmentStartInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_start(inp)
            return result.model_dump(mode="json")

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.start_upload failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/parse_docx")
    async def http_parse_docx(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = ParseDocxInput.model_validate(payload)
            result = parse_docx(inp)
            return result.model_dump(mode="json")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("parse_docx failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/parse_pdf")
    async def http_parse_pdf(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = ParsePdfInput.model_validate(payload)
            result = parse_pdf(inp)
            return result.model_dump(mode="json")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("parse_pdf failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/normalize_clauses")
    async def http_normalize_clauses(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = NormalizeClausesInput.model_validate(payload)
            result = normalize_clauses(inp)
            return result.model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("normalize_clauses failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    # ------------------
    # Risk assessment workflow endpoints
    # ------------------

    @app.post("/tools/risk_assessment/start")
    async def http_risk_assessment_start(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = RiskAssessmentStartInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_start(inp)
            return result.model_dump(mode="json")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.start failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/risk_assessment/status")
    async def http_risk_assessment_status(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = RiskAssessmentStatusInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_status(inp)
            return result.model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.status failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/risk_assessment/get_clause_result")
    async def http_risk_assessment_get_clause_result(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = RiskAssessmentGetClauseResultInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_get_clause_result(inp)
            return result.model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.get_clause_result failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/risk_assessment/report")
    async def http_risk_assessment_report(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = RiskAssessmentReportInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_report(inp)
            return result.model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.report failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/risk_assessment/cancel")
    async def http_risk_assessment_cancel(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            inp = RiskAssessmentCancelInput.model_validate(payload)
            result = await risk_assessment.risk_assessment_cancel(inp)
            return result.model_dump(mode="json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("risk_assessment.cancel failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    # ------------------
    # MCP over SSE transport
    # ------------------

    @app.get("/sse")
    async def sse(session_id: Optional[str] = Query(default=None)):
        sid = session_id or str(uuid.uuid4())
        queue = await sessions.get_queue(sid)

        async def event_generator():
            try:
                # Tell the client where to POST messages.
                yield {"event": "endpoint", "data": f"/messages?session_id={sid}"}

                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield {"event": "message", "data": json.dumps(msg, ensure_ascii=False)}
                    except asyncio.TimeoutError:
                        # keep-alive
                        yield {"event": "ping", "data": "keepalive"}
            finally:
                # Client disconnected; drop the queue to avoid unbounded growth.
                await sessions.close(sid)

        return EventSourceResponse(event_generator())

    @app.post("/messages")
    async def messages(request: Request, session_id: str = Query(...)):
        try:
            payload = await request.json()
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            # JSON-RPC parse error equivalent
            return JSONResponse({"status": "ignored"}, status_code=200)

        response = await _build_jsonrpc_response(payload)

        # Notifications (no id) do not get JSON-RPC responses.
        if response is None:
            return JSONResponse({"status": "ok"}, status_code=200)

        # If an SSE client is connected for this session, also publish the response over SSE.
        q = await sessions.get_existing_queue(session_id)
        if q is not None:
            await sessions.send(session_id, response)

        # IMPORTANT: Return the JSON-RPC response to HTTP callers.
        return JSONResponse(response, status_code=200)

    return app


app = create_app()


async def _build_jsonrpc_response(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a JSON-RPC response for an incoming message.

    Returns:
      - dict: JSON-RPC response object when the request has an `id`
      - None: for notifications (no `id`) or methods like `initialized`
    """

    method = msg.get("method")
    rpc_id = msg.get("id", None)

    # Notifications (no id) do not get responses.
    expects_response = rpc_id is not None

    # Basic request validation
    if not isinstance(method, str) or not method.strip():
        if not expects_response:
            return None
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32600, "message": "Invalid Request: missing method"},
        }

    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-document-parser", "version": __version__},
            }
            response: Dict[str, Any] = {"jsonrpc": "2.0", "id": rpc_id, "result": result}

        elif method == "initialized":
            # Notification from client.
            return None

        elif method == "ping":
            response = {"jsonrpc": "2.0", "id": rpc_id, "result": {}}

        elif method == "tools/list":
            response = {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": MCP_TOOLS}}

        elif method == "tools/call":
            params = msg.get("params") or {}
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}

            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tools/call missing params.name")

            if tool_name not in set(MCP_TOOL_NAMES):
                raise ValueError(f"Unknown tool: {tool_name}. Available: {list(MCP_TOOL_NAMES)}")

            if not isinstance(tool_args, dict):
                raise ValueError("tools/call params.arguments must be an object")

            tool_result_text = await _call_tool(tool_name, tool_args)

            response = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": tool_result_text,
                        }
                    ]
                },
            }

        else:
            response = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    except Exception as e:
        log.exception("jsonrpc handler error", method=method, error=str(e))
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32603, "message": str(e)},
        }

    if not expects_response:
        return None
    return response


async def _call_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Call a registered tool and return JSON string of the tool output."""

    if name == "parse_docx":
        inp = ParseDocxInput.model_validate(arguments)
        res: DocumentParseResult = parse_docx(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "parse_pdf":
        inp = ParsePdfInput.model_validate(arguments)
        res: DocumentParseResult = parse_pdf(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "normalize_clauses":
        inp = NormalizeClausesInput.model_validate(arguments)
        res: ClauseListNormalized = normalize_clauses(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "risk_assessment.start":
        inp = RiskAssessmentStartInput.model_validate(arguments)
        res = await risk_assessment.risk_assessment_start(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "risk_assessment.status":
        inp = RiskAssessmentStatusInput.model_validate(arguments)
        res = await risk_assessment.risk_assessment_status(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "risk_assessment.get_clause_result":
        inp = RiskAssessmentGetClauseResultInput.model_validate(arguments)
        res = await risk_assessment.risk_assessment_get_clause_result(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "risk_assessment.report":
        inp = RiskAssessmentReportInput.model_validate(arguments)
        res = await risk_assessment.risk_assessment_report(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    if name == "risk_assessment.cancel":
        inp = RiskAssessmentCancelInput.model_validate(arguments)
        res = await risk_assessment.risk_assessment_cancel(inp)
        return json.dumps(res.model_dump(mode="json"), ensure_ascii=False)

    raise ValueError(f"Unknown tool: {name}")
