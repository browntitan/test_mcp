from __future__ import annotations

import logging

from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..schemas import (
    ClauseAssessment,
    RiskAssessmentCancelInput,
    RiskAssessmentCancelOutput,
    RiskAssessmentGetClauseResultInput,
    RiskAssessmentGetClauseResultOutput,
    RiskAssessmentReportInput,
    RiskAssessmentReportOutput,
    RiskAssessmentStartInput,
    RiskAssessmentStartOutput,
    RiskAssessmentStatusInput,
    RiskAssessmentStatusOutput,
)
from ..workflows.risk_assessment.store import get_store
from ..workflows.risk_assessment.runner import start_risk_assessment
from ..config import get_settings


logger = logging.getLogger(__name__)


def _not_found(name: str, assessment_id: str) -> ValueError:
    return ValueError(f"{name}: assessment_id not found: {assessment_id}")



def _as_assessment(item: Any) -> ClauseAssessment:
    """Back-compat: accept either ClauseAssessment or an object/dict with an `assessment` field."""
    if item is None:
        raise ValueError("Missing clause result item")

    # dict-like
    if isinstance(item, dict) and "assessment" in item:
        return ClauseAssessment.model_validate(item["assessment"])

    # object-like
    if hasattr(item, "assessment"):
        return getattr(item, "assessment")

    # already a ClauseAssessment
    return item


def _get_text_with_changes(item: Any) -> Optional[str]:
    # dict-like
    if isinstance(item, dict):
        v = item.get("text_with_changes") or item.get("clause_text") or item.get("text")
        return (v or None)
    # object-like
    for attr in ("text_with_changes", "clause_text", "text"):
        if hasattr(item, attr):
            v = getattr(item, attr)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _get_item_label_title(item: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract label/title from a rich item if present (e.g., ClauseRiskResult), else None."""
    # dict-like
    if isinstance(item, dict):
        lbl = item.get("label")
        ttl = item.get("title")
        return (lbl if isinstance(lbl, str) else None, ttl if isinstance(ttl, str) else None)

    # object-like
    lbl = getattr(item, "label", None)
    ttl = getattr(item, "title", None)
    return (lbl if isinstance(lbl, str) else None, ttl if isinstance(ttl, str) else None)


def _make_clause_header(item: Any, r: ClauseAssessment) -> str:
    """Prefer label/title on the rich item; fall back to the assessment; then clause_id."""
    item_label, item_title = _get_item_label_title(item)

    parts: List[str] = []
    for v in (
        (item_label or "").strip(),
        (item_title or "").strip(),
        (getattr(r, "label", None) or "").strip(),
        (getattr(r, "title", None) or "").strip(),
    ):
        if v and v not in parts:
            parts.append(v)

    header = " ".join([p for p in parts if p]).strip()
    return header or (getattr(r, "clause_id", None) or "(unknown clause)")


def _compute_totals(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "low": 0, "medium": 0, "high": 0, "avg_score": 0}

    assessments: List[ClauseAssessment] = []
    for it in results:
        try:
            assessments.append(_as_assessment(it))
        except Exception:
            continue

    if not assessments:
        return {"total": 0, "low": 0, "medium": 0, "high": 0, "avg_score": 0}

    total = len(assessments)
    low = sum(1 for r in assessments if r.risk_level == "low")
    medium = sum(1 for r in assessments if r.risk_level == "medium")
    high = sum(1 for r in assessments if r.risk_level == "high")
    avg = sum(r.risk_score for r in assessments) / total
    return {"total": total, "low": low, "medium": medium, "high": high, "avg_score": avg}


def _render_report_markdown(report: RiskAssessmentReportOutput) -> str:
    totals = report.totals or _compute_totals(report.clause_results)

    lines: List[str] = []
    lines.append(f"# Risk Assessment Report ({report.assessment_id})")
    lines.append("")
    lines.append(f"**Status:** {report.status}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(report.summary.strip() if report.summary else "(no summary)")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(
        f"- Total clauses assessed: {totals.get('total', 0)}\n"
        f"- High: {totals.get('high', 0)}\n"
        f"- Medium: {totals.get('medium', 0)}\n"
        f"- Low: {totals.get('low', 0)}\n"
        f"- Avg score: {totals.get('avg_score', 0):.1f}"
    )
    lines.append("")

    lines.append("## Clause Results")
    lines.append("")

    for item in report.clause_results:
        r = _as_assessment(item)
        clause_text = _get_text_with_changes(item)

        header = _make_clause_header(item, r)

        # If the stored clause text already begins with the same header line, drop it to avoid duplication.
        if clause_text:
            first_line = clause_text.strip().splitlines()[0].strip() if clause_text.strip() else ""
            if first_line and first_line == header:
                clause_text = "\n".join(clause_text.splitlines()[1:]).lstrip("\n").strip() or None

        lines.append(f"### {header}")
        lines.append("")

        if clause_text:
            lines.append("**Clause text (including changes/comments if available):**")
            lines.append("")
            lines.append("```text")
            lines.append(clause_text.strip())
            lines.append("```")
            lines.append("")

        lines.append(f"- Risk: **{r.risk_level.upper()}** (score={r.risk_score})")
        lines.append("")

        if r.justification:
            lines.append("**Justification:**")
            lines.append("")
            lines.append(r.justification.strip())
            lines.append("")

        if r.issues:
            lines.append("**Issues:**")
            lines.append("")
            for it in r.issues:
                lines.append(f"- [{it.severity}] {it.category}: {it.description}")
            lines.append("")

        if r.citations:
            lines.append("**Policy citations:**")
            lines.append("")
            for c in r.citations:
                snippet = (c.text or "").strip().replace("\n", " ")
                if len(snippet) > 260:
                    snippet = snippet[:259] + "…"
                lines.append(
                    f"- policy_id={c.policy_id} chunk_id={c.chunk_id} score={c.score:.3f}: {snippet}"
                )
            lines.append("")

        if r.recommended_redline:
            lines.append("**Recommended redline:**")
            lines.append("")
            lines.append("```text")
            lines.append(r.recommended_redline.strip())
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# -----------------------------
# MCP Tool handlers
# -----------------------------


async def risk_assessment_start(input_data: RiskAssessmentStartInput) -> RiskAssessmentStartOutput:
    """Start the deterministic clause-by-clause risk assessment workflow."""

    # Helpful error for base64 callers who forgot file_type/filename.
    if input_data.file_base64 and not input_data.parse_result:
        has_name = bool((input_data.filename or "").strip() or (input_data.file_path or "").strip())
        if not has_name and not input_data.file_type:
            raise ValueError(
                "risk_assessment.start: when using file_base64 without file_path/filename, you must provide file_type='docx'|'pdf'"
            )

    logger.info(
        "risk_assessment.start called (mode=%s, model_profile=%s, top_k=%s, collection=%s, termset_id=%s)",
        getattr(input_data, "mode", None),
        getattr(input_data, "model_profile", None),
        getattr(input_data, "top_k", None),
        getattr(input_data, "policy_collection", None),
        getattr(input_data, "termset_id", None),
    )

    out = await start_risk_assessment(input_data)

    # Persist RAG / retrieval configuration snapshot for auditability/debugging.
    # This is safe even if the runner also sets it; store.set_rag will just overwrite.
    try:
        settings = get_settings()
        rag = {
            "policy_collection": getattr(input_data, "policy_collection", None),
            "termset_id": getattr(input_data, "termset_id", None),
            "filters": getattr(input_data, "filters", None) or {},
            "top_k": getattr(input_data, "top_k", None),
            "min_score": getattr(input_data, "min_score", None),
            # Optional diagnostics
            "embeddings_dim": getattr(settings, "embeddings_dim", None),
            "embeddings_model_profile": getattr(settings, "embeddings_model_profile", None),
        }
        store = get_store()
        await store.set_rag(out.assessment_id, rag)
    except Exception as e:
        logger.warning("Failed to persist RAG config for assessment_id=%s: %s", getattr(out, "assessment_id", None), e)

    logger.info(
        "risk_assessment.start returning (assessment_id=%s, status=%s, clause_count=%s)",
        out.assessment_id,
        out.status,
        out.clause_count,
    )
    return out


async def risk_assessment_status(input_data: RiskAssessmentStatusInput) -> RiskAssessmentStatusOutput:
    store = get_store()
    out = await store.status_output(input_data.assessment_id)
    if out is None:
        raise _not_found("risk_assessment.status", input_data.assessment_id)
    logger.info(
        "risk_assessment.status (assessment_id=%s) -> %s (%s/%s)",
        out.assessment_id,
        out.status,
        out.completed_clauses,
        out.total_clauses,
    )
    return out


async def risk_assessment_get_clause_result(
    input_data: RiskAssessmentGetClauseResultInput,
) -> RiskAssessmentGetClauseResultOutput:
    store = get_store()
    out = await store.clause_result_output(input_data.assessment_id, input_data.clause_id)
    if out is None:
        # Disambiguate: assessment missing vs clause missing
        rec = await store.get(input_data.assessment_id)
        if rec is None:
            raise _not_found("risk_assessment.get_clause_result", input_data.assessment_id)
        raise ValueError(
            f"risk_assessment.get_clause_result: clause_id not found for assessment_id={input_data.assessment_id}: {input_data.clause_id}"
        )
    return out


async def risk_assessment_report(input_data: RiskAssessmentReportInput) -> RiskAssessmentReportOutput:
    store = get_store()
    out = await store.report_output(input_data.assessment_id)
    if out is None:
        raise _not_found("risk_assessment.report", input_data.assessment_id)

    logger.info(
        "risk_assessment.report called (assessment_id=%s, format=%s, status=%s)",
        out.assessment_id,
        input_data.format,
        out.status,
    )

    # Ensure totals present even if store hasn't set them yet.
    if not out.totals:
        out = out.model_copy(update={"totals": _compute_totals(out.clause_results)})

    if input_data.format == "markdown":
        md = _render_report_markdown(out)
        out = out.model_copy(update={"summary": md})

    logger.info(
        "risk_assessment.report returning (assessment_id=%s, status=%s, results=%s)",
        out.assessment_id,
        out.status,
        len(out.clause_results or []),
    )
    return out


async def risk_assessment_cancel(input_data: RiskAssessmentCancelInput) -> RiskAssessmentCancelOutput:
    store = get_store()
    return await store.cancel(input_data.assessment_id)


# Registry used by server tool dispatchers.
TOOL_HANDLERS: Dict[str, Callable[[Any], Awaitable[Any]]] = {
    "risk_assessment.start": risk_assessment_start,
    "risk_assessment.status": risk_assessment_status,
    "risk_assessment.get_clause_result": risk_assessment_get_clause_result,
    "risk_assessment.report": risk_assessment_report,
    "risk_assessment.cancel": risk_assessment_cancel,
}


__all__ = [
    "risk_assessment_start",
    "risk_assessment_status",
    "risk_assessment_get_clause_result",
    "risk_assessment_report",
    "risk_assessment_cancel",
    "TOOL_HANDLERS",
]