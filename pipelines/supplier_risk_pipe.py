"""
title: Supplier Risk Assessment (MCP)
author: Shiv + ChatGPT
version: 0.2.0
requirements: requests,pydantic
"""

import base64
import os
import time
import asyncio
from typing import Dict, Generator, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field


class Pipeline:
    class Valves(BaseModel):
        # From Pipelines container -> host port 3000 -> OpenWebUI container 8080
        OPENWEBUI_BASE_URL: str = Field(default="http://host.docker.internal:3000")
        OPENWEBUI_API_KEY: str = Field(default="")

        # MCP server reachable from Pipelines container
        MCP_BASE_URL: str = Field(default="http://host.docker.internal:8765")

        # Assessment knobs
        POLICY_COLLECTION: str = Field(default="default")
        TOP_K: int = Field(default=3, ge=1, le=50)
        MIN_SCORE: Optional[float] = Field(default=None)
        MODEL_PROFILE: str = Field(default="assessment")
        CONCURRENCY: int = Field(default=2, ge=1, le=16)
        INCLUDE_TEXT_WITH_CHANGES: bool = Field(default=True)
        MODE: str = Field(default="async")  # async recommended
        REPORT_FORMAT: str = Field(default="markdown")  # json|markdown

        POLL_INTERVAL_S: float = Field(default=1.0, ge=0.5, le=10.0)
        HTTP_TIMEOUT_S: float = Field(default=180.0, ge=10.0, le=600.0)

        DEBUG: bool = Field(default=True)

    def __init__(self):
        self.name = "Supplier Risk Assessment (MCP)"
        # load valves from env if present
        vals = {}
        for k in self.Valves.model_fields.keys():
            v = os.getenv(k)
            if v is not None:
                vals[k] = v
        self.valves = self.Valves(**vals)

    # -------------------------
    # Inlet: capture file info
    # -------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        This runs during /<module>/filter/inlet. This is where your screenshot showed
        body["files"] is actually populated.

        We extract the uploaded file id + filename + content_type and store them into:
            body["_supplier_risk_file"] = {...}

        Then pipe() can reliably access it even if body["files"] disappears later.
        """
        try:
            files = body.get("files") or body.get("Files") or []
            if not isinstance(files, list):
                files = []

            if self.valves.DEBUG:
                print("[supplier_risk_pipe] inlet body keys:", list(body.keys()))
                print("[supplier_risk_pipe] inlet files count:", len(files))

            chosen = self._pick_best_file_from_inlet(files)

            if chosen:
                body["_supplier_risk_file"] = chosen
                if self.valves.DEBUG:
                    print("[supplier_risk_pipe] captured file:", chosen)
            else:
                if self.valves.DEBUG:
                    print("[supplier_risk_pipe] inlet: no docx/pdf file found in files[]")

        except Exception as e:
            if self.valves.DEBUG:
                print("[supplier_risk_pipe] inlet error:", str(e))

        return body

    def _pick_best_file_from_inlet(self, files: List[dict]) -> Optional[dict]:
        """
        Your screenshot shows each entry shaped like:
          entry["file"]["id"]
          entry["file"]["filename"]
          entry["file"]["meta"]["content_type"]

        We support that + a few variants.
        """
        candidates: List[dict] = []

        for entry in files:
            if not isinstance(entry, dict):
                continue

            f = entry.get("file") if isinstance(entry.get("file"), dict) else entry

            file_id = f.get("id") or f.get("file_id") or entry.get("id") or entry.get("file_id")
            filename = f.get("filename") or f.get("name") or entry.get("filename") or entry.get("name") or ""
            meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
            content_type = meta.get("content_type") or f.get("content_type") or entry.get("content_type") or ""

            if not file_id:
                continue

            ft = self._guess_file_type(filename, content_type)
            # keep only docx/pdf
            if ft in ("docx", "pdf"):
                candidates.append(
                    {
                        "id": str(file_id),
                        "filename": str(filename) if filename else "uploaded",
                        "content_type": str(content_type) if content_type else "",
                        "file_type": ft,
                    }
                )

        if not candidates:
            return None

        # prefer docx over pdf
        candidates.sort(key=lambda x: 0 if x["file_type"] == "docx" else 1)
        return candidates[0]

    def _guess_file_type(self, filename: str, content_type: str) -> Optional[str]:
        fn = (filename or "").lower().strip()
        ct = (content_type or "").lower().strip()

        if fn.endswith(".docx") or "wordprocessingml.document" in ct:
            return "docx"
        if fn.endswith(".pdf") or ct == "application/pdf" or "pdf" in ct:
            return "pdf"
        return None

    # -------------------------
    # OpenWebUI + MCP helpers
    # -------------------------
    def _owui_headers(self) -> Dict[str, str]:
        key = (self.valves.OPENWEBUI_API_KEY or "").strip()
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}"}

    def _download_file_bytes(self, file_id: str) -> bytes:
        url = self.valves.OPENWEBUI_BASE_URL.rstrip("/") + f"/api/v1/files/{file_id}/content"
        r = requests.get(url, headers=self._owui_headers(), timeout=float(self.valves.HTTP_TIMEOUT_S))
        r.raise_for_status()
        if not r.content:
            raise ValueError("Downloaded file content was empty.")
        return r.content

    def _mcp_post(self, path: str, payload: dict) -> dict:
        url = self.valves.MCP_BASE_URL.rstrip("/") + path
        r = requests.post(url, json=payload, timeout=float(self.valves.HTTP_TIMEOUT_S))
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"MCP response for {path} was not an object")
        return data

    def _status_details(self, title: str, body: str = "", done: bool = False) -> str:
        d = "true" if done else "false"
        body = (body or "").strip()
        if body:
            body = "\n\n" + body + "\n"
        return f'<details type="status" done="{d}">\n<summary>{title}</summary>{body}</details>\n'

    # -------------------------
    # Main pipeline
    # -------------------------
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        streaming = bool(body.get("stream", False))

        def gen() -> Generator[str, None, None]:
            try:
                yield self._status_details("Supplier Risk Assessment", "Starting…", done=False)
                yield "Starting supplier risk assessment…\n\n"

                # ✅ file info captured by inlet()
                finfo = body.get("_supplier_risk_file")
                if not isinstance(finfo, dict):
                    raise ValueError(
                        "No file captured from inlet(). "
                        "This usually means OpenWebUI didn't include files[] in the inlet request."
                    )

                file_id = finfo["id"]
                filename = finfo.get("filename") or "uploaded"
                file_type = finfo.get("file_type") or self._guess_file_type(filename, finfo.get("content_type", ""))

                yield self._status_details("File captured", f"{filename} (id={file_id})", done=False)
                yield "Downloading file bytes from OpenWebUI…\n"
                blob = self._download_file_bytes(file_id)

                if file_type not in ("docx", "pdf"):
                    raise ValueError(f"Unsupported file_type={file_type} for filename={filename}")

                yield "Starting MCP risk assessment…\n"
                start_payload = {
                    "file_base64": base64.b64encode(blob).decode("utf-8"),
                    "filename": filename,
                    "file_type": file_type,
                    "policy_collection": self.valves.POLICY_COLLECTION,
                    "top_k": int(self.valves.TOP_K),
                    "min_score": self.valves.MIN_SCORE,
                    "model_profile": self.valves.MODEL_PROFILE,
                    "concurrency": int(self.valves.CONCURRENCY),
                    "include_text_with_changes": bool(self.valves.INCLUDE_TEXT_WITH_CHANGES),
                    "mode": self.valves.MODE,
                }
                start_data = self._mcp_post("/tools/risk_assessment/start", start_payload)
                assessment_id = start_data.get("assessment_id")
                if not assessment_id:
                    raise ValueError(f"Missing assessment_id from MCP start response: {start_data}")

                yield self._status_details("Assessment started", f"id={assessment_id}", done=False)

                last_line = ""
                while True:
                    st = self._mcp_post("/tools/risk_assessment/status", {"assessment_id": assessment_id})
                    status = st.get("status")
                    done = int(st.get("completed_clauses") or 0)
                    total = int(st.get("total_clauses") or 0)
                    cur = st.get("current_clause_id") or ""
                    err = st.get("error") or ""

                    line = f"Progress: {done}/{total} status={status}" if total else f"Progress: {done} status={status}"
                    if cur:
                        line += f" current={cur}"

                    if line != last_line:
                        yield line + "\n"
                        yield self._status_details("Assessing clauses…", line, done=False)
                        last_line = line

                    if status in ("completed", "failed", "canceled"):
                        if status == "failed":
                            yield f"\n❌ Assessment failed: {err}\n"
                        break

                    time.sleep(float(self.valves.POLL_INTERVAL_S))

                yield "\nFetching report…\n\n"
                rep = self._mcp_post(
                    "/tools/risk_assessment/report",
                    {"assessment_id": assessment_id, "format": self.valves.REPORT_FORMAT},
                )
                content = (rep.get("summary") or "").strip()
                if not content:
                    content = f"Assessment {assessment_id} finished, but report text was empty."

                yield self._status_details("Done", f"id={assessment_id}", done=True)
                yield content + "\n"

            except Exception as e:
                yield self._status_details("Error", str(e), done=True)
                yield f"\n❌ Pipeline error: {e}\n"

        if streaming:
            return gen()
        return "".join(list(gen()))


# Pydantic v2 safety
Pipeline.Valves.model_rebuild()