from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from .schemas import ParseDocxInput  # adjust if your import path differs
from .tools.parse_docx import parse_docx  # uses your updated parser

router = APIRouter()

def _safe_model_dump(obj: Any) -> Any:
    """Pydantic v2: model_dump(); v1: dict(); else passthrough."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj

@router.post("/risk_assess")
async def risk_assess(
    file: UploadFile = File(...),
    extract_comments: bool = True,
    extract_tracked_changes: bool = True,
    include_raw_spans: bool = False,
) -> Dict[str, Any]:
    """
    Minimal REST wrapper for OpenWebUI Actions:
      - Accepts multipart file upload
      - Calls parse_docx (DOCX)
      - Returns JSON payload that the Action can render
    """
    filename = file.filename or "uploaded.docx"
    ext = os.path.splitext(filename.lower())[1]

    if ext != ".docx":
        raise HTTPException(status_code=400, detail="Only .docx is supported by /risk_assess in this endpoint.")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    b64 = base64.b64encode(blob).decode("utf-8")

    # Build ParseDocxInput using your existing schema
    parsed = parse_docx(
        ParseDocxInput(
            file_path=None,
            file_base64=b64,
            options={
                # if your ParseDocxInput.options is a Pydantic model, replace with that model.
                # This dict form matches many implementations; adjust to your schema.
                "extract_comments": extract_comments,
                "extract_tracked_changes": extract_tracked_changes,
                "include_raw_spans": include_raw_spans,
            },
        )
    )

    parsed_dict = _safe_model_dump(parsed)

    # --- Placeholder “risk assessment” ---
    # Replace this with your real risk scoring logic later.
    num_clauses = len(parsed_dict.get("clauses", []) or [])
    warnings = parsed_dict.get("warnings", []) or []

    risk = {
        "overall_risk": "UNKNOWN",
        "num_clauses": num_clauses,
        "notes": [
            "Risk scoring not implemented yet in /risk_assess. This endpoint currently returns parse results + basic stats."
        ],
    }

    return {
        "filename": filename,
        "parse": parsed_dict,
        "risk": risk,
        "warnings": warnings,
    }