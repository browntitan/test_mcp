"""
title: Supplier Risk Assessment (MCP)
author: Shiv Balodi
version: 0.2.0
requirements: requests,pydantic
"""

import base64
import os
import re
import time
import asyncio
import csv
import io
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple, Any

import requests
from pydantic import BaseModel, Field


class Pipeline:
    class Valves(BaseModel):
        # From Pipelines container -> host port 3000 -> OpenWebUI container 8080
        OPENWEBUI_BASE_URL: str = Field(default="http://host.docker.internal:3000")
        OPENWEBUI_API_KEY: str = Field(default="")

        # The URL that the end-user's browser can reach (often http://localhost:3000 or your DNS name).
        # If empty, we'll derive it from OPENWEBUI_BASE_URL (replace host.docker.internal -> localhost).
        OPENWEBUI_PUBLIC_BASE_URL: str = Field(default="")

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

        # UI safeguard: avoid OpenWebUI "Chunk too big" by streaming large outputs in smaller chunks.
        MAX_UI_CHUNK_CHARS: int = Field(default=8000, ge=1024, le=200000)

        # CSV export options
        EXPORT_CSV: bool = Field(default=True)
        CSV_INCLUDE_TEXT_WITH_CHANGES: bool = Field(default=True)
        CSV_MAX_TEXT_CHARS: int = Field(default=50000, ge=0, le=500000)

        POLL_INTERVAL_S: float = Field(default=1.0, ge=0.5, le=10.0)
        HTTP_TIMEOUT_S: float = Field(default=180.0, ge=10.0, le=600.0)

        DEBUG: bool = Field(default=True)

    def __init__(self):
        self.name = "Supplier Risk Assessment (MCP) /w .csv export"
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
    # Termset helpers
    # -------------------------
    _TERMSET_RE_LIST = [
        re.compile(r"\btermset[_ ]?id\s*=?\s*([0-9]{1,6})\b", re.IGNORECASE),
        re.compile(r"\bTermset ID:\s*([0-9]{1,6})\b", re.IGNORECASE),
        re.compile(r"\bTermset:\s*([0-9]{1,6})\b", re.IGNORECASE),
        re.compile(
            r"CTM\s*[-\u2010-\u2015\u2212]\s*P\s*[-\u2010-\u2015\u2212]\s*ST\s*[-\u2010-\u2015\u2212]\s*(\d{1,6})",
            re.IGNORECASE,
        ),
    ]

    def _normalize_termset_id(self, v: Any) -> str:
        """Normalize numeric termset ids to 3 digits (e.g., 2 -> 002)."""
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            return ""
        if s.isdigit() and len(s) <= 3:
            try:
                return f"{int(s):03d}"
            except Exception:
                return s
        return s

    def _extract_termset_id_from_start(self, start_data: dict) -> str:
        """Best-effort extraction of termset id returned by MCP start.

        We try common shapes:
          - start_data['termset_id']
          - start_data['document']['termset_id']
          - start_data['parse_result']['document']['termset_id']
          - parse from warnings/messages (e.g., "Auto-detected termset_id=002 ...")
          - parse from embedded CTM-P-ST-xxx tokens
        """
        if not isinstance(start_data, dict):
            return ""

        # Direct fields
        for key in ("termset_id", "termset"):
            if key in start_data and start_data.get(key) is not None:
                return self._normalize_termset_id(start_data.get(key))

        # Nested document metadata
        doc = start_data.get("document")
        if isinstance(doc, dict) and doc.get("termset_id") is not None:
            return self._normalize_termset_id(doc.get("termset_id"))

        pr = start_data.get("parse_result")
        if isinstance(pr, dict):
            d2 = pr.get("document")
            if isinstance(d2, dict) and d2.get("termset_id") is not None:
                return self._normalize_termset_id(d2.get("termset_id"))

        # Parse from warnings/messages
        msgs: List[str] = []
        for k in ("warnings", "parse_warnings", "messages", "info"):
            v = start_data.get(k)
            if isinstance(v, list):
                msgs.extend([str(x) for x in v if x is not None])
            elif isinstance(v, str) and v.strip():
                msgs.append(v)

        blob = "\n".join(msgs)
        if blob:
            for rx in self._TERMSET_RE_LIST:
                m = rx.search(blob)
                if m:
                    return self._normalize_termset_id(m.group(1))

        return ""

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

    def _public_base_url(self) -> str:
        """Return the base URL that the end-user's browser can reach."""
        pub = (self.valves.OPENWEBUI_PUBLIC_BASE_URL or "").strip()
        if pub:
            return pub.rstrip("/")

        base = (self.valves.OPENWEBUI_BASE_URL or "").strip().rstrip("/")
        if "host.docker.internal" in base:
            return base.replace("host.docker.internal", "localhost")
        return base

    def _owui_upload_bytes(self, filename: str, blob: bytes, content_type: str) -> str:
        """Upload a generated file to OpenWebUI so the user can download it."""
        base = self.valves.OPENWEBUI_BASE_URL.rstrip("/")
        # Avoid ingesting into RAG; we just want it downloadable.
        url = base + "/api/v1/files/?process=false&process_in_background=false"
        headers = {**self._owui_headers(), "Accept": "application/json"}
        files = {"file": (filename, blob, content_type)}
        r = requests.post(url, headers=headers, files=files, timeout=float(self.valves.HTTP_TIMEOUT_S))
        r.raise_for_status()
        data = r.json()

        # Try common response shapes.
        file_id = None
        if isinstance(data, dict):
            file_id = data.get("id") or data.get("file_id")
            if not file_id and isinstance(data.get("file"), dict):
                file_id = data["file"].get("id")
            if not file_id and isinstance(data.get("data"), dict):
                file_id = data["data"].get("id") or data["data"].get("file_id")

        if not file_id:
            raise ValueError(f"OpenWebUI file upload returned unexpected response: {data}")
        return str(file_id)

    def _iter_text_chunks(self, text: str, chunk_size: int) -> Generator[str, None, None]:
        """Yield text in smaller chunks to avoid OpenWebUI streaming limits."""
        if not text:
            return
        n = max(1, int(chunk_size or 8000))
        for i in range(0, len(text), n):
            yield text[i : i + n]

    def _as_assessment(self, item: Any) -> dict:
        """Handle ClauseAssessment vs ClauseRiskResult shapes."""
        if isinstance(item, dict) and isinstance(item.get("assessment"), dict):
            return item["assessment"]
        if isinstance(item, dict):
            return item
        return {}

    def _get_text_with_changes(self, item: Any) -> str:
        if isinstance(item, dict):
            v = item.get("text_with_changes") or item.get("clause_text") or item.get("text")
            return (v or "")
        return ""

    def _build_csv_bytes(self, assessment_id: str, rep_json: dict, termset_id: str = "") -> bytes:
        """Build a CSV export from the JSON report."""
        summary = (rep_json.get("summary") or "").strip()
        totals = rep_json.get("totals") or {}
        clause_results = rep_json.get("clause_results") or []

        include_text = bool(self.valves.CSV_INCLUDE_TEXT_WITH_CHANGES)
        max_text = int(self.valves.CSV_MAX_TEXT_CHARS or 0)

        buf = io.StringIO(newline="")
        w = csv.writer(buf)

        # Metadata header rows
        w.writerow(["assessment_id", assessment_id])
        w.writerow(["generated_at", datetime.utcnow().isoformat() + "Z"])
        if termset_id:
            w.writerow(["termset_id", termset_id])
        if totals:
            w.writerow(["totals", str(totals)])
        if summary:
            w.writerow(["summary", summary])
        w.writerow([])

        headers = [
            "clause_id",
            "label",
            "title",
            "risk_level",
            "risk_score",
            "justification",
            "issues",
            "citations",
            "recommended_redline",
        ]
        if include_text:
            headers.append("text_with_changes")
        w.writerow(headers)

        for item in clause_results:
            a = self._as_assessment(item)
            clause_id = str(a.get("clause_id") or "")
            label = str(a.get("label") or "")
            title = str(a.get("title") or "")
            risk_level = str(a.get("risk_level") or "")

            risk_score = a.get("risk_score")
            try:
                risk_score = int(risk_score)
            except Exception:
                risk_score = ""

            justification = (a.get("justification") or "")

            # Flatten issues
            issues_list = a.get("issues") or []
            issues_parts: List[str] = []
            if isinstance(issues_list, list):
                for it in issues_list:
                    if not isinstance(it, dict):
                        continue
                    sev = (it.get("severity") or "").strip()
                    cat = (it.get("category") or "").strip()
                    desc = (it.get("description") or "").strip()
                    if sev or cat or desc:
                        issues_parts.append(f"{sev}|{cat}|{desc}")
            issues = " ; ".join(issues_parts)

            # Flatten citations
            cits_list = a.get("citations") or []
            cits_parts: List[str] = []
            if isinstance(cits_list, list):
                for c in cits_list:
                    if not isinstance(c, dict):
                        continue
                    pid = (c.get("policy_id") or "").strip()
                    chid = (c.get("chunk_id") or "").strip()
                    score = c.get("score")
                    try:
                        score_s = f"{float(score):.3f}"
                    except Exception:
                        score_s = ""
                    txt = (c.get("text") or "").strip().replace("\n", " ")
                    if len(txt) > 260:
                        txt = txt[:259] + "…"
                    if pid or chid or txt:
                        cits_parts.append(f"{pid}|{chid}|{score_s}|{txt}")
            citations = " ; ".join(cits_parts)

            recommended_redline = (a.get("recommended_redline") or "")

            row = [
                clause_id,
                label,
                title,
                risk_level,
                risk_score,
                justification,
                issues,
                citations,
                recommended_redline,
            ]

            if include_text:
                txt2 = self._get_text_with_changes(item)
                if max_text and len(txt2) > max_text:
                    txt2 = txt2[: max_text - 1] + "…"
                row.append(txt2)

            w.writerow(row)

        return buf.getvalue().encode("utf-8")

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

                # Show the user which termset was captured/used (if available).
                termset_id = self._extract_termset_id_from_start(start_data)
                if termset_id:
                    yield self._status_details("Termset detected", f"termset_id={termset_id}", done=False)
                    yield f"Detected termset id: **{termset_id}**\n"
                else:
                    yield self._status_details("Termset detected", "NONE (not found)", done=False)

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

                # Primary report (often markdown)
                rep = self._mcp_post(
                    "/tools/risk_assessment/report",
                    {"assessment_id": assessment_id, "format": self.valves.REPORT_FORMAT},
                )
                content = (rep.get("summary") or "").strip()
                if not content:
                    content = f"Assessment {assessment_id} finished, but report text was empty."

                # Prepend termset id for clarity in the UI.
                if termset_id:
                    content = f"**Termset ID:** {termset_id}\n\n" + content

                csv_download_url = ""
                csv_download_path = ""  # relative path is most reliable inside OpenWebUI
                csv_filename = ""
                if bool(self.valves.EXPORT_CSV):
                    yield self._status_details("Export", "Generating CSV…", done=False)

                    # Always fetch JSON for export (best structured source)
                    rep_json = self._mcp_post(
                        "/tools/risk_assessment/report",
                        {"assessment_id": assessment_id, "format": "json"},
                    )

                    csv_bytes = self._build_csv_bytes(assessment_id, rep_json, termset_id=termset_id)
                    csv_filename = f"risk_report_{assessment_id}.csv"
                    file_id2 = self._owui_upload_bytes(csv_filename, csv_bytes, "text/csv")

                    # Relative path works best for end users (same-origin in the browser)
                    csv_download_path = f"/api/v1/files/{file_id2}/content"

                    # Absolute URL is optional and depends on correct public base url
                    pub = self._public_base_url()
                    if pub:
                        csv_download_url = f"{pub}{csv_download_path}"

                    yield self._status_details("Export", f"CSV uploaded (file_id={file_id2}).", done=False)

                yield self._status_details("Done", f"id={assessment_id}", done=True)

                # Provide CSV download link first (use relative link for reliability)
                if csv_download_path:
                    # Markdown link so it's clickable in the UI
                    link_text = f"\n\n✅ CSV export generated.\n📄 Download: [{csv_filename or 'risk_report.csv'}]({csv_download_path})\n"
                    if csv_download_url:
                        # Also show the absolute URL for environments where users need it
                        link_text += f"Direct URL: {csv_download_url}\n"
                    link_text += "\n"

                    # This message is short; yield it as-is (avoid splitting the URL).
                    yield link_text

                # Stream the final report in smaller chunks to avoid OpenWebUI streaming limits.
                for part in self._iter_text_chunks(content, int(self.valves.MAX_UI_CHUNK_CHARS)):
                    yield part
                yield "\n"

            except Exception as e:
                yield self._status_details("Error", str(e), done=True)
                yield f"\n❌ Pipeline error: {e}\n"

        if streaming:
            return gen()
        return "".join(list(gen()))


# Pydantic v2 safety
Pipeline.Valves.model_rebuild()