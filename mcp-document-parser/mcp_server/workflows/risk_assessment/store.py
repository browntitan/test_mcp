from __future__ import annotations

import logging

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import asyncio

from ...schemas import (
    ClauseAssessment,
    RiskAssessmentCancelOutput,
    RiskAssessmentGetClauseResultOutput,
    RiskAssessmentReportOutput,
    RiskAssessmentStatusOutput,
)


logger = logging.getLogger(__name__)
_UNSET = object()

AssessmentStatus = Literal["queued", "running", "completed", "failed", "canceled"]


def _now_ts() -> float:
    return time.time()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _AssessmentRecord:
    assessment_id: str
    status: AssessmentStatus

    created_at_ts: float
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # High-level document info
    document: Dict[str, Any] = field(default_factory=dict)

    # Clause execution plan
    clause_ids: List[str] = field(default_factory=list)

    # Progress
    total_clauses: int = 0
    completed_clauses: int = 0
    current_clause_id: Optional[str] = None

    # Results
    clause_results: Dict[str, ClauseAssessment] = field(default_factory=dict)
    summary: str = ""
    totals: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class RiskAssessmentStore:
    """In-memory job store for risk assessments.

    This store is intentionally simple for local development. It is shaped so we can later
    swap to Postgres/Redis without changing the MCP tool surface.

    Thread-safety: FastAPI can run concurrently; we guard state with an asyncio.Lock.
    """

    def __init__(self, ttl_seconds: int = 60 * 60, max_assessments: int = 128) -> None:
        self._ttl_seconds = int(ttl_seconds)
        self._max_assessments = int(max_assessments)
        self._items: Dict[str, _AssessmentRecord] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup_ts = 0.0

    async def _cleanup(self) -> None:
        """Drop expired assessments and keep memory bounded."""
        now = _now_ts()

        # Throttle cleanup to at most once per minute.
        if now - self._last_cleanup_ts < 60:
            return
        self._last_cleanup_ts = now

        # Expire
        expired: List[str] = []
        for aid, rec in self._items.items():
            if now - rec.created_at_ts > self._ttl_seconds:
                expired.append(aid)
        for aid in expired:
            self._items.pop(aid, None)

        # Bound
        if len(self._items) <= self._max_assessments:
            return
        # Remove oldest
        ordered = sorted(self._items.values(), key=lambda r: r.created_at_ts)
        to_remove = len(self._items) - self._max_assessments
        for rec in ordered[:to_remove]:
            self._items.pop(rec.assessment_id, None)

    async def create(
        self,
        *,
        document: Dict[str, Any],
        clause_ids: List[str],
        warnings: Optional[List[str]] = None,
        status: AssessmentStatus = "queued",
    ) -> str:
        """Create a new assessment record."""
        assessment_id = uuid.uuid4().hex
        rec = _AssessmentRecord(
            assessment_id=assessment_id,
            status=status,
            created_at_ts=_now_ts(),
            document=document,
            clause_ids=list(clause_ids),
            total_clauses=len(clause_ids),
            warnings=list(warnings or []),
        )
        async with self._lock:
            await self._cleanup()
            self._items[assessment_id] = rec
        return assessment_id

    async def get(self, assessment_id: str) -> Optional[_AssessmentRecord]:
        async with self._lock:
            await self._cleanup()
            return self._items.get(assessment_id)

    async def set_status(
        self,
        assessment_id: str,
        status: AssessmentStatus,
        *,
        error: Optional[str] = None,
    ) -> bool:
        async with self._lock:
            await self._cleanup()
            rec = self._items.get(assessment_id)
            if rec is None:
                return False
            prev_status = rec.status

            # Terminal states set finished_at
            if status == "running" and rec.started_at is None:
                rec.started_at = _iso_now()

            rec.status = status
            if error:
                rec.error = error
            if status in ("completed", "failed", "canceled"):
                rec.finished_at = rec.finished_at or _iso_now()
            if prev_status != rec.status and rec.status in ("running", "completed", "failed", "canceled"):
                logger.info("risk_assessment status %s: %s -> %s", assessment_id, prev_status, rec.status)
            return True

    async def add_warning(self, assessment_id: str, warning: str) -> bool:
        async with self._lock:
            rec = self._items.get(assessment_id)
            if rec is None:
                return False
            rec.warnings.append(warning)
            return True

    async def set_progress(
        self,
        assessment_id: str,
        *,
        completed_clauses: Optional[int] = None,
        current_clause_id: Any = _UNSET,
    ) -> bool:
        async with self._lock:
            rec = self._items.get(assessment_id)
            if rec is None:
                return False
            if completed_clauses is not None:
                rec.completed_clauses = int(completed_clauses)
            if current_clause_id is not _UNSET:
                rec.current_clause_id = current_clause_id  # may be None to clear
            return True

    async def put_clause_result(self, assessment_id: str, clause: ClauseAssessment) -> bool:
        async with self._lock:
            rec = self._items.get(assessment_id)
            if rec is None:
                return False
            rec.clause_results[clause.clause_id] = clause
            # Keep progress monotonic
            rec.completed_clauses = max(rec.completed_clauses, len(rec.clause_results))
            return True

    async def set_report(self, assessment_id: str, *, summary: str, totals: Dict[str, Any]) -> bool:
        async with self._lock:
            rec = self._items.get(assessment_id)
            if rec is None:
                return False
            rec.summary = summary
            rec.totals = dict(totals)
            return True

    async def cancel(self, assessment_id: str) -> RiskAssessmentCancelOutput:
        async with self._lock:
            rec = self._items.get(assessment_id)
            if rec is None:
                return RiskAssessmentCancelOutput(assessment_id=assessment_id, status="not_found")

            if rec.status in ("completed", "failed", "canceled"):
                return RiskAssessmentCancelOutput(assessment_id=assessment_id, status="already_finished")

            rec.status = "canceled"
            rec.finished_at = rec.finished_at or _iso_now()
            if not rec.error:
                rec.error = "canceled by user"
            logger.info("risk_assessment canceled %s", assessment_id)
            return RiskAssessmentCancelOutput(assessment_id=assessment_id, status="canceled")

    async def status_output(self, assessment_id: str) -> Optional[RiskAssessmentStatusOutput]:
        rec = await self.get(assessment_id)
        if rec is None:
            return None
        return RiskAssessmentStatusOutput(
            assessment_id=rec.assessment_id,
            status=rec.status,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            total_clauses=rec.total_clauses,
            completed_clauses=rec.completed_clauses,
            current_clause_id=rec.current_clause_id,
            warnings=list(rec.warnings),
            error=rec.error,
        )

    async def clause_result_output(
        self, assessment_id: str, clause_id: str
    ) -> Optional[RiskAssessmentGetClauseResultOutput]:
        rec = await self.get(assessment_id)
        if rec is None:
            return None
        clause = rec.clause_results.get(clause_id)
        if clause is None:
            return None
        return RiskAssessmentGetClauseResultOutput(assessment_id=assessment_id, clause=clause)

    async def report_output(self, assessment_id: str) -> Optional[RiskAssessmentReportOutput]:
        rec = await self.get(assessment_id)
        if rec is None:
            return None

        # Preserve original clause ordering
        ordered_results: List[ClauseAssessment] = []
        for cid in rec.clause_ids:
            r = rec.clause_results.get(cid)
            if r is not None:
                ordered_results.append(r)
        # Include any results not in the plan (shouldn't happen, but be defensive)
        for cid, r in rec.clause_results.items():
            if cid not in set(rec.clause_ids):
                ordered_results.append(r)

        return RiskAssessmentReportOutput(
            assessment_id=rec.assessment_id,
            status=rec.status,
            summary=rec.summary or "",
            clause_results=ordered_results,
            totals=dict(rec.totals),
        )


# Module-level singleton store for local usage
STORE = RiskAssessmentStore()


def get_store() -> RiskAssessmentStore:
    return STORE