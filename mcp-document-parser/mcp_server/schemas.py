from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RawSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    kind: Literal["normal", "inserted", "deleted", "strike", "comment_ref", "moved_from", "moved_to"]


class RedlineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    context: Optional[str] = None


class Redlines(BaseModel):
    model_config = ConfigDict(extra="ignore")

    insertions: List[RedlineItem] = Field(default_factory=list)
    deletions: List[RedlineItem] = Field(default_factory=list)
    strikethroughs: List[RedlineItem] = Field(default_factory=list)


class Comment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    author: Optional[str] = None
    date: Optional[str] = None
    text: str
    context: Optional[str] = None


class ClauseChangeItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # The changed text (typically a word, phrase, sentence, or entire inserted/deleted paragraph)
    text: str

    # Best-effort label near where the change occurred (e.g., "1.4"), if detected
    label: Optional[str] = None

    # Track-change metadata (if present in the DOCX revision element)
    author: Optional[str] = None
    date: Optional[str] = None
    revision_id: Optional[str] = None


class ClauseModificationItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # A modification is usually a paired delete+insert
    from_text: str
    to_text: str

    label: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    revision_id: Optional[str] = None


class ClauseCommentItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    comment_id: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    text: str

    # Best-effort snippet of the text range the comment was anchored to
    anchor_text: Optional[str] = None


class ClauseChanges(BaseModel):
    model_config = ConfigDict(extra="ignore")

    added: List[ClauseChangeItem] = Field(default_factory=list)
    deleted: List[ClauseChangeItem] = Field(default_factory=list)
    modified: List[ClauseModificationItem] = Field(default_factory=list)
    comments: List[ClauseCommentItem] = Field(default_factory=list)


class SourceLocations(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_start: Optional[int] = None
    page_end: Optional[int] = None


class Clause(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clause_id: str
    label: Optional[str] = None
    title: Optional[str] = None
    level: int
    parent_clause_id: Optional[str] = None
    text: str
    raw_spans: List[RawSpan] = Field(default_factory=list)
    redlines: Redlines = Field(default_factory=Redlines)
    comments: List[Comment] = Field(default_factory=list)
    changes: ClauseChanges = Field(default_factory=ClauseChanges)
    source_locations: SourceLocations = Field(default_factory=SourceLocations)


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str
    media_type: Literal[
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/pdf",
    ]
    pages: Optional[int] = None
    word_count: Optional[int] = None
    # Optional contract termset identifier, typically extracted from the document footer.
    # Example footer token: "CTM-P-ST-002" -> termset_id="002".
    termset_id: Optional[str] = None


class DocumentParseResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document: DocumentMetadata
    clauses: List[Clause] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class NormalizedClause(Clause):
    model_config = ConfigDict(extra="ignore")

    original_clause_id: Optional[str] = None
    was_merged: bool = False
    was_split: bool = False


class ClauseListNormalized(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document: DocumentMetadata
    clauses: List[NormalizedClause]
    normalization_warnings: List[str] = Field(default_factory=list)


class ClauseBoundary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_char: int
    end_char: int
    level: int
    label: Optional[str] = None
    title: Optional[str] = None


# ---------------------------
# Tool input schemas
# ---------------------------


class ParseDocxOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    extract_tracked_changes: bool = True
    extract_comments: bool = True
    include_raw_spans: bool = True


class ParseDocxInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: Optional[str] = None
    file_base64: Optional[str] = None
    options: ParseDocxOptions = Field(default_factory=ParseDocxOptions)

    @model_validator(mode="after")
    def _validate_file_source(self) -> "ParseDocxInput":
        if not self.file_path and not self.file_base64:
            raise ValueError("Either file_path or file_base64 must be provided")
        if self.file_path and self.file_base64:
            # allow but prefer file_path
            return self
        return self


class ParsePdfOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    extract_annotations: bool = True
    include_raw_spans: bool = True


class ParsePdfInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: Optional[str] = None
    file_base64: Optional[str] = None
    options: ParsePdfOptions = Field(default_factory=ParsePdfOptions)

    @model_validator(mode="after")
    def _validate_file_source(self) -> "ParsePdfInput":
        if not self.file_path and not self.file_base64:
            raise ValueError("Either file_path or file_base64 must be provided")
        return self


class NormalizeClausesInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parse_result: DocumentParseResult
    boundaries: Optional[List[ClauseBoundary]] = None


# ---------------------------
# Risk assessment workflow schemas
# ---------------------------


class PolicyCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    policy_id: str
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    severity: Literal["low", "medium", "high"]
    description: str

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v: Any) -> Any:
        if isinstance(v, str):
            vv = v.strip().lower().replace("_", " ").strip()
            if vv.endswith(" risk"):
                vv = vv[: -len(" risk")].strip()

            mapping = {
                "low": "low",
                "l": "low",
                "minor": "low",
                "min": "low",
                "medium": "medium",
                "med": "medium",
                "mid": "medium",
                "moderate": "medium",
                "high": "high",
                "h": "high",
                "major": "high",
                "severe": "high",
                "critical": "high",
                "crit": "high",
            }
            if vv in mapping:
                return mapping[vv]
        return v


class ClauseAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clause_id: str
    label: Optional[str] = None
    title: Optional[str] = None

    risk_score: int = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]

    @field_validator("risk_level", mode="before")
    @classmethod
    def _normalize_risk_level(cls, v: Any) -> Any:
        if isinstance(v, str):
            vv = v.strip().lower().replace("_", " ").strip()
            if vv.endswith(" risk"):
                vv = vv[: -len(" risk")].strip()

            mapping = {
                "low": "low",
                "l": "low",
                "minor": "low",
                "min": "low",
                "medium": "medium",
                "med": "medium",
                "mid": "medium",
                "moderate": "medium",
                "high": "high",
                "h": "high",
                "major": "high",
                "severe": "high",
                "critical": "high",
                "crit": "high",
            }
            if vv in mapping:
                return mapping[vv]
        return v

    justification: str
    issues: List[RiskIssue] = Field(default_factory=list)
    citations: List[PolicyCitation] = Field(default_factory=list)


    recommended_redline: Optional[str] = None


# New richer per-clause result model that includes the clause text used for assessment
class ClauseRiskResult(BaseModel):
    """Richer per-clause result that includes the clause text used for assessment.

    This supports the UI format you want (clause text + changes/comments + risk + justification)
    while keeping backwards compatibility with older clients that expect ClauseAssessment only.
    """

    model_config = ConfigDict(extra="ignore")

    clause_id: str
    label: Optional[str] = None
    title: Optional[str] = None

    # The full clause text that was assessed, optionally including a "Changes:" block.
    text_with_changes: Optional[str] = None

    # The structured assessment output.
    assessment: ClauseAssessment


class RiskAssessmentStartInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Provide ONE of: (file_path/file_base64) OR parse_result.
    # If file_* is provided, the server will parse and then assess.
    file_path: Optional[str] = None
    file_base64: Optional[str] = None
    filename: Optional[str] = None

    # Explicit file type helps when using base64
    file_type: Optional[Literal["docx", "pdf"]] = None

    # If caller already parsed, they can pass a parse result directly.
    parse_result: Optional[DocumentParseResult] = None

    # Parsing options (used only when parsing from file_*).
    parse_docx_options: Optional[ParseDocxOptions] = None
    parse_pdf_options: Optional[ParsePdfOptions] = None

    # RAG / policy retrieval
    policy_collection: str = "default"
    top_k: int = Field(default=3, ge=1, le=50)
    min_score: Optional[float] = None
    filters: Dict[str, Any] = Field(default_factory=dict)

    # Optional term set identifier used to deterministically match internal guidance.
    # Accepts values like "003" or "3"; numeric values are normalized to 3-digit strings.
    termset_id: Optional[str] = None

    @field_validator("termset_id", mode="before")
    @classmethod
    def _normalize_termset_id(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            try:
                return f"{int(v):03d}"
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            if s.isdigit():
                return f"{int(s):03d}"
            # If caller passes a non-numeric token, preserve it as-is.
            return s
        return v

    # Model selection
    # "chat" means use the chat/default model profile.
    # "assessment" means use the assessment model profile (enterprise).
    model_profile: str = "assessment"

    # Workflow options
    focus_clause_ids: Optional[List[str]] = None
    concurrency: int = Field(default=2, ge=1, le=16)
    include_text_with_changes: bool = True

    # Execution mode
    mode: Literal["sync", "async"] = "async"

    @model_validator(mode="after")
    def _validate_source(self) -> "RiskAssessmentStartInput":
        has_file = bool(self.file_path or self.file_base64)
        has_parse = self.parse_result is not None
        if not has_file and not has_parse:
            raise ValueError("Provide either file_path/file_base64 or parse_result")
        if has_file and has_parse:
            # allow but prefer parse_result if both provided
            return self
        return self


class RiskAssessmentStartOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    status: Literal["queued", "running", "completed", "failed"]
    document: DocumentMetadata
    clause_count: int

    # Effective termset id used for policy retrieval (normalized, e.g., "2" -> "002").
    # This is typically captured from the DOCX footer token like "CTM-P-ST-002".
    termset_id: Optional[str] = None

    warnings: List[str] = Field(default_factory=list)


class RiskAssessmentStatusInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str


class RiskAssessmentStatusOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    status: Literal["queued", "running", "completed", "failed", "canceled"]

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    total_clauses: int = 0
    completed_clauses: int = 0
    current_clause_id: Optional[str] = None

    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class RiskAssessmentGetClauseResultInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    clause_id: str


class RiskAssessmentGetClauseResultOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    clause: Union[ClauseAssessment, ClauseRiskResult]


class RiskAssessmentReportInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    format: Literal["json", "markdown"] = "json"


class RiskAssessmentReportOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    status: Literal["queued", "running", "completed", "failed", "canceled"]

    summary: str
    clause_results: List[Union[ClauseAssessment, ClauseRiskResult]] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)


class RiskAssessmentCancelInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str


class RiskAssessmentCancelOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessment_id: str
    status: Literal["canceled", "not_found", "already_finished"]


def stable_clause_id(index: int, text: str) -> str:
    """Generate a stable unique clause id using the required format."""

    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"clause_{index:04d}_{digest}"


def normalize_text_for_hash(text: str) -> str:
    """Normalize text for deduplication."""

    return " ".join(text.split()).strip().lower()


def md5_8(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def to_jsonable(obj: Any) -> Any:
    """Convert Pydantic models to JSON-serializable dicts."""

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


# MCP tool registration (for tools/list)
MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "parse_docx",
        "description": (
            "Parse a Microsoft Word (.docx) document with support for tracked changes "
            "(insertions/deletions) and comments. Returns structured clauses with redlines "
            "and annotations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the .docx file"},
                "file_base64": {"type": "string", "description": "Base64-encoded file contents"},
                "options": {
                    "type": "object",
                    "properties": {
                        "extract_tracked_changes": {"type": "boolean", "default": True},
                        "extract_comments": {"type": "boolean", "default": True},
                        "include_raw_spans": {"type": "boolean", "default": True},
                    },
                },
            },
            "oneOf": [{"required": ["file_path"]}, {"required": ["file_base64"]}],
        },
    },
    {
        "name": "parse_pdf",
        "description": (
            "Parse a PDF document with annotations. Returns structured clauses with detected "
            "comments and strikethroughs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "file_base64": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "extract_annotations": {"type": "boolean", "default": True},
                        "include_raw_spans": {"type": "boolean", "default": True},
                    },
                },
            },
            "oneOf": [{"required": ["file_path"]}, {"required": ["file_base64"]}],
        },
    },
    {
        "name": "normalize_clauses",
        "description": "Normalize and restructure clauses from a parse result. Can apply LLM-detected boundaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "parse_result": {"type": "object", "description": "DocumentParseResult from parse_docx/pdf"},
                "boundaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_char": {"type": "integer"},
                            "end_char": {"type": "integer"},
                            "level": {"type": "integer"},
                            "label": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["start_char", "end_char", "level"],
                    },
                },
            },
            "required": ["parse_result"],
        },
    },
    {
        "name": "risk_assessment.start",
        "description": (
            "Start a deterministic clause-by-clause risk assessment workflow. The server will parse "
            "the document (if needed), retrieve relevant policy context via RAG, score each clause, "
            "and compile a final report. Returns an assessment_id for polling and retrieval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "file_base64": {"type": "string"},
                "filename": {"type": "string"},
                "file_type": {"type": "string", "enum": ["docx", "pdf"]},
                "parse_result": {"type": "object"},
                "parse_docx_options": {"type": "object"},
                "parse_pdf_options": {"type": "object"},
                "policy_collection": {"type": "string", "default": "default"},
                "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 50},
                "min_score": {"type": "number"},
                "filters": {"type": "object"},
                "termset_id": {
                    "type": "string",
                    "description": "Optional term set id used to match internal guidance (e.g., '003' or '3'). Numeric values are normalized to 3 digits.",
                },
                "model_profile": {"type": "string", "default": "assessment"},
                "focus_clause_ids": {"type": "array", "items": {"type": "string"}},
                "concurrency": {"type": "integer", "default": 2, "minimum": 1, "maximum": 16},
                "include_text_with_changes": {"type": "boolean", "default": True},
                "mode": {"type": "string", "enum": ["sync", "async"], "default": "async"},
            },
            "anyOf": [
                {"required": ["parse_result"]},
                {"required": ["file_path"]},
                {"required": ["file_base64"]},
            ],
        },
    },
    {
        "name": "risk_assessment.status",
        "description": "Get status/progress for a previously started risk assessment.",
        "inputSchema": {
            "type": "object",
            "properties": {"assessment_id": {"type": "string"}},
            "required": ["assessment_id"],
        },
    },
    {
        "name": "risk_assessment.get_clause_result",
        "description": "Fetch the risk assessment result for a single clause.",
        "inputSchema": {
            "type": "object",
            "properties": {"assessment_id": {"type": "string"}, "clause_id": {"type": "string"}},
            "required": ["assessment_id", "clause_id"],
        },
    },
    {
        "name": "risk_assessment.report",
        "description": "Get the compiled risk assessment report (summary + clause-level results).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "assessment_id": {"type": "string"},
                "format": {"type": "string", "enum": ["json", "markdown"], "default": "json"},
            },
            "required": ["assessment_id"],
        },
    },
    {
        "name": "risk_assessment.cancel",
        "description": "Cancel a running risk assessment (best-effort).",
        "inputSchema": {
            "type": "object",
            "properties": {"assessment_id": {"type": "string"}},
            "required": ["assessment_id"],
        },
    },
]
