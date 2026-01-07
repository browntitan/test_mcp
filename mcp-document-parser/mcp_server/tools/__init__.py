
from __future__ import annotations

"""Tool package for the MCP document parser.

This module re-exports the tool callables used by the MCP server dispatcher.
Keeping these exports explicit makes it easier to:
- import tool handlers in one place
- reference tool names consistently across server/client/tests
"""

from .normalize_clauses import normalize_clauses
from .parse_docx import parse_docx
from .parse_pdf import parse_pdf
from . import risk_assessment

# Re-export risk assessment handler functions for convenience.
from .risk_assessment import (
    risk_assessment_cancel,
    risk_assessment_get_clause_result,
    risk_assessment_report,
    risk_assessment_start,
    risk_assessment_status,
)

# Canonical list of MCP tool names exposed by this server.
MCP_TOOL_NAMES = (
    "parse_docx",
    "parse_pdf",
    "normalize_clauses",
    "risk_assessment.start",
    "risk_assessment.status",
    "risk_assessment.get_clause_result",
    "risk_assessment.report",
    "risk_assessment.cancel",
)

__all__ = [
    "parse_docx",
    "parse_pdf",
    "normalize_clauses",
    "risk_assessment",
    "risk_assessment_start",
    "risk_assessment_status",
    "risk_assessment_get_clause_result",
    "risk_assessment_report",
    "risk_assessment_cancel",
    "MCP_TOOL_NAMES",
]
