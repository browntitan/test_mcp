

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ...config import get_settings
from ...providers.llm import ChatMessage, LLMError, get_llm
from ...rag.pgvector import search_policies
from ...schemas import (
    Clause,
    ClauseAssessment,
    ClauseRiskResult,
    DocumentMetadata,
    DocumentParseResult,
    ParseDocxInput,
    ParseDocxOptions,
    ParsePdfInput,
    ParsePdfOptions,
    PolicyCitation,
    RiskAssessmentStartInput,
    RiskAssessmentStartOutput,
)
from ...tools.parse_docx import parse_docx
from ...tools.parse_pdf import parse_pdf
from ...workflows.risk_assessment.store import get_store



logger = logging.getLogger(__name__)

_CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def _extract_clause_number(label: Optional[str]) -> Optional[str]:
    """Extract a canonical clause number from labels like '2', '2.', '2.1', '2.1.'"""
    if not label:
        return None
    s = str(label).strip()
    if not s:
        return None
    s = s.rstrip(".")
    m = _CLAUSE_NUMBER_RE.match(s)
    return m.group(1) if m else None

# Prompt-size guards to reduce truncated/invalid JSON responses.
MAX_CLAUSE_CHARS = 14000
MAX_CLAUSE_CHARS_RETRY = 14000
MAX_POLICY_SNIPPET_CHARS = 14000
MAX_POLICY_BLOCK_CHARS = 14000
MAX_POLICY_BLOCK_CHARS_RETRY = 14000

# Output token guards (Azure-friendly). These should be enough for your bounded JSON schema.
# If you need to tune these later, we can move them into Settings/ModelProfile.
MAX_ASSESSMENT_OUTPUT_TOKENS = 14000
MAX_SUMMARY_OUTPUT_TOKENS = 14000



def _infer_file_type(file_path: Optional[str], filename: Optional[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        e = explicit.lower().strip().lstrip(".")
        if e in ("docx", "pdf"):
            return e
    name = filename or file_path or ""
    name = name.lower()
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".pdf"):
        return "pdf"
    return None



def _norm_profile_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    n = str(name).strip()
    return n or None


def _normalize_termset_id(v: Any) -> Optional[str]:
    """Normalize a numeric termset id to a 3-digit string (e.g., 2 -> "002")."""
    if v is None:
        return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    if not s:
        return None
    if s.isdigit():
        try:
            return f"{int(s):03d}"
        except Exception:
            return s
    return s


def _resolve_profile(settings: Any, preferred: Optional[str], *, fallback: Optional[str]) -> str:
    """Resolve a model profile name safely.

    - Uses `preferred` if present and exists in settings.model_profiles
    - Else uses `fallback` if present and exists
    - Else falls back to the first available profile name (deterministic order)
    """

    preferred_n = _norm_profile_name(preferred)
    fallback_n = _norm_profile_name(fallback)

    profiles = getattr(settings, "model_profiles", {}) or {}
    if preferred_n and preferred_n in profiles:
        return preferred_n
    if fallback_n and fallback_n in profiles:
        return fallback_n

    # Deterministic fallback: prefer commonly expected names.
    for k in ("assessment", "chat", "embeddings"):
        if k in profiles:
            return k

    # Last resort: first profile in sorted order.
    if profiles:
        return sorted(list(profiles.keys()))[0]

    # Should never happen; Settings always builds at least one profile.
    return preferred_n or fallback_n or "chat"


def _changes_plain_text(clause: Clause) -> str:
    """Build a plain-text Changes section for LLM consumption.

    We prefer the new structured `clause.changes` field, but fall back to redlines/comments.
    """

    lines: List[str] = []

    ch = getattr(clause, "changes", None)
    has_structured = False
    if ch is not None:
        try:
            has_structured = bool(ch.added or ch.deleted or ch.modified or ch.comments)
        except Exception:
            has_structured = False

    ins = getattr(getattr(clause, "redlines", None), "insertions", []) or []
    dele = getattr(getattr(clause, "redlines", None), "deletions", []) or []
    comms = getattr(clause, "comments", []) or []

    if has_structured:
        lines.append("Changes:")
        if ch.added:
            lines.append("Added:")
            for a in ch.added:
                label = (a.label or "").strip() if hasattr(a, "label") else ""
                txt = (a.text or "").strip()
                if not txt:
                    continue
                lines.append(f"- {(label + ' ') if label else ''}{txt}".strip())
            lines.append("")
        if ch.deleted:
            lines.append("Deleted:")
            for d in ch.deleted:
                label = (d.label or "").strip() if hasattr(d, "label") else ""
                txt = (d.text or "").strip()
                if not txt:
                    continue
                lines.append(f"- {(label + ' ') if label else ''}{txt}".strip())
            lines.append("")
        if ch.modified:
            lines.append("Modified:")
            for m in ch.modified:
                label = (m.label or "").strip() if hasattr(m, "label") else ""
                from_t = (m.from_text or "").strip()
                to_t = (m.to_text or "").strip()
                if not from_t and not to_t:
                    continue
                prefix = (label + " ") if label else ""
                lines.append(f"- {prefix}from: {from_t} | to: {to_t}".strip())
            lines.append("")
        if ch.comments:
            lines.append("Comments:")
            for c in ch.comments:
                author = (c.author or "Reviewer") if hasattr(c, "author") else "Reviewer"
                anchor = (c.anchor_text or "").strip() if hasattr(c, "anchor_text") else ""
                txt = (c.text or "").strip()
                if not txt:
                    continue
                if anchor:
                    anchor = re.sub(r"\s+", " ", anchor)
                    lines.append(f"- {author} (anchor: \"{anchor}\"): {txt}")
                else:
                    lines.append(f"- {author}: {txt}")
            lines.append("")
        return "\n".join(lines).strip()

    # Fallback for older parse results
    if not (ins or dele or comms):
        return ""

    lines.append("Changes:")
    if ins:
        lines.append("Added:")
        for a in ins:
            txt = (getattr(a, "text", "") or "").strip()
            if txt:
                lines.append(f"- {txt}")
        lines.append("")
    if dele:
        lines.append("Deleted:")
        for d in dele:
            txt = (getattr(d, "text", "") or "").strip()
            if txt:
                lines.append(f"- {txt}")
        lines.append("")
    if comms:
        lines.append("Comments:")
        for c in comms:
            author = getattr(c, "author", None) or "Reviewer"
            txt = (getattr(c, "text", "") or "").strip()
            if txt:
                lines.append(f"- {author}: {txt}")
        lines.append("")

    return "\n".join(lines).strip()


def _format_clause_for_assessment(clause: Clause, include_changes: bool) -> str:
    header_parts: List[str] = []
    if clause.label:
        header_parts.append(str(clause.label))
    if clause.title:
        header_parts.append(str(clause.title))

    header = " ".join(header_parts).strip()
    body = (clause.text or "").strip()

    parts: List[str] = []
    if header:
        parts.append(header)
    if body:
        parts.append(body)

    if include_changes:
        changes = _changes_plain_text(clause)
        if changes:
            parts.append(changes)

    return "\n\n".join(parts).strip()


async def _load_or_parse(start: RiskAssessmentStartInput) -> Tuple[DocumentParseResult, List[str]]:
    """Return (parse_result, warnings)."""

    if start.parse_result is not None:
        # Trust caller; warnings remain empty.
        return start.parse_result, []

    file_type = _infer_file_type(start.file_path, start.filename, start.file_type)
    if not file_type:
        raise ValueError("Unable to infer file_type (docx/pdf). Provide file_type or a file extension.")

    warnings: List[str] = []

    if file_type == "docx":
        options = start.parse_docx_options or ParseDocxOptions(
            extract_tracked_changes=True,
            extract_comments=True,
            include_raw_spans=True,
        )
        inp = ParseDocxInput(file_path=start.file_path, file_base64=start.file_base64, options=options)
        res = parse_docx(inp)
        warnings.extend(res.warnings)
        return res, warnings

    if file_type == "pdf":
        options = start.parse_pdf_options or ParsePdfOptions(extract_annotations=True, include_raw_spans=True)
        inp = ParsePdfInput(file_path=start.file_path, file_base64=start.file_base64, options=options)
        res = parse_pdf(inp)
        warnings.extend(res.warnings)
        return res, warnings

    raise ValueError(f"Unsupported file_type: {file_type}")


async def start_risk_assessment(start: RiskAssessmentStartInput) -> RiskAssessmentStartOutput:
    """Start a risk assessment.

    - If mode == 'async', schedules background execution and returns immediately.
    - If mode == 'sync', runs to completion before returning.

    Note: This function is intended to be called from the MCP tool handler.
    """

    settings = get_settings()
    store = get_store()

    parse_result, warnings = await _load_or_parse(start)

    # Auto-populate termset_id from parsed document metadata (DOCX footer extraction) if caller did not provide one.
    doc_ts = _normalize_termset_id(getattr(getattr(parse_result, "document", None), "termset_id", None))
    caller_ts = _normalize_termset_id(getattr(start, "termset_id", None))

    if caller_ts and doc_ts and caller_ts != doc_ts:
        warnings.append(
            f"Provided termset_id={caller_ts} differs from parsed document termset_id={doc_ts}; using provided value."
        )
        # Ensure we use the normalized caller termset id going forward.
        start = start.model_copy(update={"termset_id": caller_ts})
    elif (not caller_ts) and doc_ts:
        warnings.append(f"Auto-detected termset_id={doc_ts} from document footer.")
        start = start.model_copy(update={"termset_id": doc_ts})
    elif caller_ts:
        # Normalize caller input even if the document did not provide one.
        start = start.model_copy(update={"termset_id": caller_ts})

    logger.info(
        "risk_assessment start (file=%s mode=%s termset_id=%s policy_collection=%s)",
        getattr(parse_result.document, "filename", None),
        start.mode,
        getattr(start, "termset_id", None) or "NONE",
        getattr(start, "policy_collection", None),
    )

    clause_ids_all = [c.clause_id for c in parse_result.clauses]
    focus_clause_ids = start.focus_clause_ids
    if focus_clause_ids:
        focus_set = set(focus_clause_ids)
        clause_ids = [cid for cid in clause_ids_all if cid in focus_set]
    else:
        clause_ids = clause_ids_all

    meta = parse_result.document
    assessment_id = await store.create(
        document=meta.model_dump(mode="json"),
        clause_ids=clause_ids,
        warnings=warnings,
        status="queued" if start.mode == "async" else "running",
    )

    if start.mode == "async":
        task = asyncio.create_task(_run_assessment(assessment_id, parse_result, start))

        def _done(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception("risk_assessment task introspection failed (assessment_id=%s): %s", assessment_id, e)
                return
            if exc:
                logger.exception("risk_assessment task crashed (assessment_id=%s): %s", assessment_id, exc)

        task.add_done_callback(_done)

        return RiskAssessmentStartOutput(
            assessment_id=assessment_id,
            status="queued",
            document=meta,
            clause_count=len(clause_ids),
            warnings=warnings,
        )

    # sync
    await _run_assessment(assessment_id, parse_result, start)
    rec = await store.get(assessment_id)
    status = rec.status if rec else "failed"
    return RiskAssessmentStartOutput(
        assessment_id=assessment_id,
        status=status if status in ("completed", "failed") else "completed",
        document=meta,
        clause_count=len(clause_ids),
        warnings=warnings,
    )


async def _run_assessment(assessment_id: str, parse_result: DocumentParseResult, start: RiskAssessmentStartInput) -> None:
    """Execute the deterministic clause-by-clause risk assessment."""

    store = get_store()

    settings = get_settings()

    # Use a dedicated model profile for embeddings so we can decouple:
    # - chat/assessment LLM (for reasoning)
    # - embeddings model/provider (for pgvector retrieval)
    requested_assessment_profile = _norm_profile_name(getattr(start, "model_profile", None))
    requested_embeddings_profile = _norm_profile_name(getattr(settings, "embeddings_model_profile", None))

    assessment_profile = _resolve_profile(
        settings,
        requested_assessment_profile,
        fallback=getattr(settings, "default_assessment_profile", None),
    )

    embeddings_profile = _resolve_profile(
        settings,
        requested_embeddings_profile,
        fallback=getattr(settings, "default_chat_profile", None) or assessment_profile,
    )

    try:
        llm = get_llm(assessment_profile)
        embed_llm = get_llm(embeddings_profile)

        # One-time embedding health check to fail fast if the embedding model/config is wrong.
        probe = await embed_llm.embed_texts("dimension check")
        if not probe or not probe[0]:
            raise LLMError("Embeddings endpoint returned no vectors")
        dim = len(probe[0])
        expected_dim = getattr(settings, "embeddings_dim", None)
        if expected_dim and dim != int(expected_dim):
            raise LLMError(
                f"Embedding dimension mismatch: got {dim}, expected {int(expected_dim)} (embeddings_profile={embeddings_profile}). "
                "Update EMBEDDINGS_DIM and/or your embeddings deployment/model profile (e.g., Azure Ada-002 deployment)."
            )

        await store.set_status(assessment_id, "running")
        logger.info(
            "risk_assessment running (assessment_id=%s, clauses=%s, termset_id=%s, policy_collection=%s, assessment_profile=%s, embeddings_profile=%s)",
            assessment_id,
            len(parse_result.clauses),
            getattr(start, "termset_id", None) or "NONE",
            getattr(start, "policy_collection", None),
            assessment_profile,
            embeddings_profile,
        )

        # Resolve clauses
        clause_map: Dict[str, Clause] = {c.clause_id: c for c in parse_result.clauses}
        clause_order = [c.clause_id for c in parse_result.clauses]
        if start.focus_clause_ids:
            focus = set(start.focus_clause_ids)
            clause_order = [cid for cid in clause_order if cid in focus]

        total = len(clause_order)
        await store.set_progress(assessment_id, completed_clauses=0, current_clause_id=None)

        # Embedder for retrieval: embed a single string -> vector
        async def embed_query(q: str) -> List[float]:
            embs = await embed_llm.embed_texts(q)
            if not embs or not embs[0]:
                raise LLMError("Embeddings endpoint returned no vectors")
            vec = embs[0]
            # Debug: confirm embed length matches settings.embeddings_dim
            expected_dim2 = getattr(settings, "embeddings_dim", None)
            try:
                exp_i = int(expected_dim2) if expected_dim2 is not None else None
            except Exception:
                exp_i = None
            logger.debug(
                "RAG: embed_query vec_len=%s expected_dim=%s (embeddings_profile=%s)",
                len(vec),
                exp_i,
                embeddings_profile,
            )
            return vec

        include_changes = bool(start.include_text_with_changes)

        results: List[ClauseAssessment] = []

        # Deterministic order: we iterate in clause_order. (Concurrency can be added later.)
        for idx, cid in enumerate(clause_order, start=1):
            rec = await store.get(assessment_id)
            if rec is None:
                raise RuntimeError("Assessment record disappeared")
            if rec.status == "canceled":
                await store.set_status(assessment_id, "canceled")
                return

            clause = clause_map.get(cid)
            if clause is None:
                await store.add_warning(assessment_id, f"Clause id not found in parse_result: {cid}")
                continue

            await store.set_progress(assessment_id, completed_clauses=idx - 1, current_clause_id=cid)
            clause_text_full = _format_clause_for_assessment(clause, include_changes=include_changes)
            if not clause_text_full.strip():
                await store.add_warning(assessment_id, f"Empty clause text for clause_id={cid}")
                continue

            # Use a truncated version for embeddings/prompting to avoid model output truncation.
            clause_text = clause_text_full
            if len(clause_text) > MAX_CLAUSE_CHARS:
                clause_text = clause_text[: MAX_CLAUSE_CHARS - 1] + "…"
                await store.add_warning(
                    assessment_id,
                    f"Truncated clause_text for clause_id={cid} to {MAX_CLAUSE_CHARS} chars for assessment prompt",
                )

            # Persist the exact clause text (including changes/comments) used for assessment.
            # This is used later to render per-clause reports without LLM post-processing.
            try:
                if hasattr(store, "put_clause_text"):
                    await store.put_clause_text(assessment_id, cid, clause_text_full)  # type: ignore[attr-defined]
            except Exception as e:
                await store.add_warning(assessment_id, f"Failed to store clause text for clause_id={cid}: {e}")

            # Retrieve policy context via pgvector.
            # We deterministically narrow guidance by clause_number and (optionally) termset.
            citations: List[PolicyCitation] = []
            # Debug: start RAG retrieval for this clause
            logger.info(
                "RAG: start clause_id=%s label=%s title=%s collection=%s top_k=%s min_score=%s start_filters=%s",
                cid,
                getattr(clause, "label", None),
                getattr(clause, "title", None),
                start.policy_collection,
                start.top_k,
                start.min_score,
                (start.filters or {}),
            )
            try:
                base_filters: Dict[str, Any] = dict(start.filters or {})

                clause_number = _extract_clause_number(getattr(clause, "label", None))
                if clause_number:
                    base_filters["clause_number"] = clause_number

                # NOTE: pgvector retriever expects the special key "termset" (not "termset_id").
                termset_id = getattr(start, "termset_id", None)
                if termset_id:
                    base_filters["termset"] = termset_id

                logger.info(
                    "RAG: primary query clause_id=%s clause_number=%s termset_id=%s filters=%s",
                    cid,
                    clause_number,
                    termset_id,
                    base_filters,
                )
                citations = await search_policies(
                    clause_text,
                    collection=start.policy_collection,
                    top_k=start.top_k,
                    min_score=start.min_score,
                    filters=base_filters,
                    embedder=embed_query,
                )
                logger.info(
                    "RAG: primary results clause_id=%s count=%s top=%s",
                    cid,
                    len(citations),
                    (
                        {
                            "policy_id": citations[0].policy_id,
                            "chunk_id": citations[0].chunk_id,
                            "score": citations[0].score,
                        }
                        if citations
                        else None
                    ),
                )
                logger.debug(
                    "RAG: primary ids clause_id=%s ids=%s",
                    cid,
                    [(c.policy_id, c.chunk_id) for c in (citations or [])[:5]],
                )

                # Fallback 1: If a termset was provided but no citations matched, retry without termset.
                if (not citations) and termset_id and ("termset" in base_filters):
                    fallback_filters = dict(base_filters)
                    fallback_filters.pop("termset", None)
                    logger.info(
                        "RAG: fallback1 (drop termset) clause_id=%s filters=%s",
                        cid,
                        fallback_filters,
                    )
                    citations = await search_policies(
                        clause_text,
                        collection=start.policy_collection,
                        top_k=start.top_k,
                        min_score=start.min_score,
                        filters=fallback_filters,
                        embedder=embed_query,
                    )
                    logger.info(
                        "RAG: fallback1 results clause_id=%s count=%s top=%s",
                        cid,
                        len(citations),
                        (
                            {
                                "policy_id": citations[0].policy_id,
                                "chunk_id": citations[0].chunk_id,
                                "score": citations[0].score,
                            }
                            if citations
                            else None
                        ),
                    )
                    if citations:
                        await store.add_warning(
                            assessment_id,
                            f"No termset-specific guidance found; used clause-only guidance for clause_id={cid} (termset_id={termset_id}).",
                        )

                # Fallback 2: If clause-number narrowing yields nothing, retry with only the caller-provided filters.
                if not citations and clause_number:
                    fallback_filters2: Dict[str, Any] = dict(start.filters or {})
                    # Keep termset if provided (it can still narrow to global guidance).
                    if termset_id:
                        fallback_filters2["termset"] = termset_id
                    logger.info(
                        "RAG: fallback2 (drop clause_number) clause_id=%s filters=%s",
                        cid,
                        fallback_filters2,
                    )
                    citations = await search_policies(
                        clause_text,
                        collection=start.policy_collection,
                        top_k=start.top_k,
                        min_score=start.min_score,
                        filters=fallback_filters2,
                        embedder=embed_query,
                    )
                    logger.info(
                        "RAG: fallback2 results clause_id=%s count=%s top=%s",
                        cid,
                        len(citations),
                        (
                            {
                                "policy_id": citations[0].policy_id,
                                "chunk_id": citations[0].chunk_id,
                                "score": citations[0].score,
                            }
                            if citations
                            else None
                        ),
                    )
                    if citations:
                        await store.add_warning(
                            assessment_id,
                            f"No clause-number-specific guidance found; used broader guidance for clause_id={cid} (clause_number={clause_number}).",
                        )

            except Exception as e:
                logger.exception(
                    "RAG: retrieval exception clause_id=%s collection=%s",
                    cid,
                    start.policy_collection,
                )
                await store.add_warning(assessment_id, f"Policy retrieval failed for clause_id={cid}: {e}")
                citations = []

            policy_block_lines: List[str] = []
            if citations:
                policy_block_lines.append("Relevant internal policy excerpts (RAG):")
                for i, c in enumerate(citations, start=1):
                    snippet = (c.text or "").strip()
                    if len(snippet) > MAX_POLICY_SNIPPET_CHARS:
                        snippet = snippet[: MAX_POLICY_SNIPPET_CHARS - 1] + "…"
                    policy_block_lines.append(
                        f"[{i}] policy_id={c.policy_id} chunk_id={c.chunk_id} score={c.score:.3f}\n{snippet}"
                    )
                    # Cap total policy block size to reduce prompt bloat.
                    if len("\n\n".join(policy_block_lines)) > MAX_POLICY_BLOCK_CHARS:
                        policy_block_lines.append("…(additional policy excerpts omitted for length)…")
                        break
                policy_block = "\n\n".join(policy_block_lines)
            else:
                policy_block = "No relevant policy context was retrieved. Use general best practices and legal reasoning."

            # Ask model for a ClauseAssessment JSON (robust + bounded).
            base_sys = (
                "You are a contract risk assessment engine. "
                "Return ONLY one valid JSON object that matches the required schema EXACTLY. "
                "No markdown fences, no prose, no trailing text. "
                "Enums: risk_level must be one of low|medium|high; issues[].severity must be one of low|medium|high. "
                "Limits: issues <= 5; citations <= 5; each citations[].text <= 240 chars; keep justification concise (<= 6 sentences). "
                "If output might be too long, shorten justification/citation snippets; NEVER output incomplete JSON."
            )

            assessment: Optional[ClauseAssessment] = None
            last_err: Optional[Exception] = None

            for attempt in range(2):
                # On retry, further shrink inputs to improve JSON reliability.
                clause_for_model = clause_text
                policy_for_model = policy_block
                if attempt == 1:
                    if len(clause_for_model) > MAX_CLAUSE_CHARS_RETRY:
                        clause_for_model = clause_for_model[: MAX_CLAUSE_CHARS_RETRY - 1] + "…"
                    if len(policy_for_model) > MAX_POLICY_BLOCK_CHARS_RETRY:
                        policy_for_model = policy_for_model[: MAX_POLICY_BLOCK_CHARS_RETRY - 1] + "…"

                sys = ChatMessage(role="system", content=base_sys)

                user = ChatMessage(
                    role="user",
                    content=(
                        "Assess the following supplier Terms & Conditions clause for risk against internal policies.\n\n"
                        "Return JSON with fields: clause_id, label, title, risk_score (0-100), risk_level (low|medium|high), "
                        "justification, issues (list of {category, severity, description}), citations (list of {policy_id, chunk_id, score, text, metadata}), "
                        "recommended_redline (optional).\n\n"
                        "Hard rules:\n"
                        "- issues[].severity must be low|medium|high (no other words).\n"
                        "- risk_level must be low|medium|high (no other words).\n"
                        "- citations must ONLY quote provided policy excerpts; omit citations if none apply.\n\n"
                        f"Clause:\n{clause_for_model}\n\n"
                        f"{policy_for_model}\n"
                    ),
                )

                try:
                    # Keep max_tokens bounded to reduce truncation risk.
                    assessment = await llm.chat_object(
                        messages=[sys, user],
                        schema=ClauseAssessment,
                        temperature=0.0,
                        max_tokens=MAX_ASSESSMENT_OUTPUT_TOKENS,
                    )
                    break
                except Exception as e:
                    last_err = e
                    # On first failure, retry once with a smaller prompt.
                    continue

            if assessment is None:
                await store.add_warning(assessment_id, f"Clause assessment failed for clause_id={cid}: {last_err}")
                continue

            logger.info(
                "LLM: clause_id=%s risk_level=%s risk_score=%s citations_returned=%s citations_retrieved=%s",
                cid,
                assessment.risk_level,
                assessment.risk_score,
                len(assessment.citations or []),
                len(citations or []),
            )
            logger.debug(
                "LLM: clause_id=%s citation_ids_returned=%s",
                cid,
                [(c.policy_id, c.chunk_id) for c in (assessment.citations or [])[:5]],
            )
            logger.debug(
                "RAG: clause_id=%s citation_ids_retrieved=%s",
                cid,
                [(c.policy_id, c.chunk_id) for c in (citations or [])[:5]],
            )

            # Force stable identity fields
            if assessment.clause_id != cid:
                assessment = assessment.model_copy(update={"clause_id": cid})
            if not assessment.label and clause.label:
                assessment = assessment.model_copy(update={"label": str(clause.label)})
            if not assessment.title and clause.title:
                assessment = assessment.model_copy(update={"title": str(clause.title)})

            # If model returned citations, ensure they are bounded (avoid huge payloads)
            bounded_cits: List[PolicyCitation] = []
            for c in assessment.citations[: max(0, min(20, len(assessment.citations)))]:
                txt = (c.text or "").strip()
                if len(txt) > 300:
                    c = c.model_copy(update={"text": txt[:299] + "…"})
                bounded_cits.append(c)
            if assessment.citations != bounded_cits:
                assessment = assessment.model_copy(update={"citations": bounded_cits})

            # Backward-compatible storage (existing clients): ClauseAssessment only
            logger.info(
                "STORE: put_clause_result assessment_id=%s clause_id=%s citations_stored=%s",
                assessment_id,
                cid,
                len(assessment.citations or []),
            )
            await store.put_clause_result(assessment_id, assessment)

            # New richer storage (new clients): include the assessed clause text + structured assessment
            try:
                if hasattr(store, "put_clause_risk_result"):
                    rr = ClauseRiskResult(
                        clause_id=cid,
                        label=assessment.label,
                        title=assessment.title,
                        text_with_changes=clause_text_full,
                        assessment=assessment,
                    )
                    await store.put_clause_risk_result(assessment_id, rr)  # type: ignore[attr-defined]
            except Exception as e:
                await store.add_warning(assessment_id, f"Failed to store ClauseRiskResult for clause_id={cid}: {e}")

            results.append(assessment)

            await store.set_progress(assessment_id, completed_clauses=idx, current_clause_id=cid)

        # Clear current clause pointer now that iteration is finished.
        try:
            await store.set_progress(assessment_id, current_clause_id=None)
        except Exception:
            pass

        # If we produced no clause results, treat this as a failure rather than a misleading "completed" run.
        # The store warnings will contain the real reason (JSON/schema validation, LLM connectivity, etc.).
        if not results:
            raise RuntimeError(
                "All clause assessments failed (no clause results produced). "
                "Check assessment warnings for JSON parsing/schema validation/LLM failures."
            )

        # Totals + summary
        totals: Dict[str, Any] = {
            "total": len(results),
            "low": sum(1 for r in results if r.risk_level == "low"),
            "medium": sum(1 for r in results if r.risk_level == "medium"),
            "high": sum(1 for r in results if r.risk_level == "high"),
            "avg_score": (sum(r.risk_score for r in results) / len(results)) if results else 0,
        }

        # Prepare a compact summary prompt
        top_high = sorted([r for r in results if r.risk_level == "high"], key=lambda x: x.risk_score, reverse=True)[:8]
        top_med = sorted([r for r in results if r.risk_level == "medium"], key=lambda x: x.risk_score, reverse=True)[:6]

        lines: List[str] = []
        lines.append(f"Document: {parse_result.document.filename}")
        lines.append(f"Termset: {getattr(start, 'termset_id', None) or getattr(parse_result.document, 'termset_id', None) or 'NONE'}")
        lines.append(f"Totals: {totals}")
        if top_high:
            lines.append("Top High Risk Clauses:")
            for r in top_high:
                lines.append(f"- {r.label or r.clause_id} {r.title or ''} (score={r.risk_score}): {r.justification[:280]}")
        if top_med:
            lines.append("Top Medium Risk Clauses:")
            for r in top_med:
                lines.append(f"- {r.label or r.clause_id} {r.title or ''} (score={r.risk_score}): {r.justification[:240]}")

        summary_seed = "\n".join(lines).strip()

        summary_text = ""
        try:
            summary_text = await llm.chat_text(
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "You are summarizing a clause-by-clause contract risk assessment for a busy stakeholder. "
                            "Produce a concise executive summary, key themes, and recommended next steps."
                        ),
                    ),
                    ChatMessage(role="user", content=summary_seed),
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_OUTPUT_TOKENS,
            )
        except Exception:
            # Deterministic fallback
            summary_text = (
                f"Risk assessment complete. High={totals['high']}, Medium={totals['medium']}, Low={totals['low']}. "
                f"Average score={totals['avg_score']:.1f}."
            )

        await store.set_report(assessment_id, summary=summary_text, totals=totals)

        # If canceled during finalization, respect it
        rec2 = await store.get(assessment_id)
        if rec2 is not None and rec2.status == "canceled":
            await store.set_status(assessment_id, "canceled")
            return

        await store.set_status(assessment_id, "completed")

    except Exception as e:
        await store.set_status(assessment_id, "failed", error=str(e))
        await store.add_warning(assessment_id, f"Assessment failed: {e}")
        return