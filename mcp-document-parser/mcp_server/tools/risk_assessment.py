


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


logger = logging.getLogger(__name__)


def _not_found(name: str, assessment_id: str) -> ValueError:
    return ValueError(f"{name}: assessment_id not found: {assessment_id}")


def _compute_totals(results: List[ClauseAssessment]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "low": 0, "medium": 0, "high": 0, "avg_score": 0}
    total = len(results)
    low = sum(1 for r in results if r.risk_level == "low")
    medium = sum(1 for r in results if r.risk_level == "medium")
    high = sum(1 for r in results if r.risk_level == "high")
    avg = sum(r.risk_score for r in results) / total
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

    for r in report.clause_results:
        header = " ".join([p for p in [r.label or "", r.title or ""] if p]).strip()
        header = header or r.clause_id
        lines.append(f"### {header}")
        lines.append("")
        lines.append(f"- Risk: **{r.risk_level.upper()}** (score={r.risk_score})")
        if r.issues:
            lines.append("- Issues:")
            for it in r.issues:
                lines.append(f"  - [{it.severity}] {it.category}: {it.description}")
        if r.citations:
            lines.append("- Policy citations:")
            for c in r.citations:
                snippet = (c.text or "").strip().replace("\n", " ")
                if len(snippet) > 260:
                    snippet = snippet[:259] + "…"
                lines.append(
                    f"  - policy_id={c.policy_id} chunk_id={c.chunk_id} score={c.score:.3f}: {snippet}"
                )
        if r.recommended_redline:
            lines.append("- Recommended redline:")
            lines.append("")
            lines.append("```text")
            lines.append(r.recommended_redline.strip())
            lines.append("```")
        lines.append("")
        lines.append(r.justification.strip() if r.justification else "")
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
        "risk_assessment.start called (mode=%s, model_profile=%s, top_k=%s, collection=%s)",
        getattr(input_data, "mode", None),
        getattr(input_data, "model_profile", None),
        getattr(input_data, "top_k", None),
        getattr(input_data, "policy_collection", None),
    )

    out = await start_risk_assessment(input_data)
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