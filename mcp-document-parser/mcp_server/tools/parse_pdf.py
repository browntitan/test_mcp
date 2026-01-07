from __future__ import annotations

import base64
import os
import re
import tempfile
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from ..schemas import (
    Clause,
    Comment,
    DocumentMetadata,
    DocumentParseResult,
    ParsePdfInput,
    RawSpan,
    RedlineItem,
    Redlines,
    SourceLocations,
    stable_clause_id,
)

# Clause label patterns (shared with DOCX)
ARTICLE_RE = re.compile(r"^\s*(ARTICLE\s+[IVXLCDM]+)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(SECTION\s+\d+(?:\.\d+)*)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\s*((?:\d+\.)+\d+|\d+\.?)(?=\s)\s+(.+)$")
PAREN_ALPHA_RE = re.compile(r"^\s*(\([a-z]\))(?=\s)\s+(.+)$", re.IGNORECASE)
PAREN_ROMAN_RE = re.compile(r"^\s*(\([ivxlcdm]+\))(?=\s)\s+(.+)$", re.IGNORECASE)


def _numeric_depth(label: str) -> int:
    raw = label.rstrip(".")
    parts = [p for p in raw.split(".") if p]
    return max(1, len(parts))


def _detect_label_kind(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Return (kind, label, title, numeric_depth_if_any)

    kind in {"article","section","numeric","alpha","roman"}.
    """
    if not text or not text.strip():
        return None, None, None, None

    m = ARTICLE_RE.match(text)
    if m:
        return "article", m.group(1).strip(), (m.group(2).strip() or None), None

    m = SECTION_RE.match(text)
    if m:
        return "section", m.group(1).strip(), (m.group(2).strip() or None), None

    m = NUMERIC_RE.match(text)
    if m:
        label = m.group(1).strip()
        return "numeric", label, None, _numeric_depth(label)

    # Roman before alpha so "(i)" isn't misclassified.
    m = PAREN_ROMAN_RE.match(text)
    if m:
        return "roman", m.group(1).strip(), None, None

    m = PAREN_ALPHA_RE.match(text)
    if m:
        return "alpha", m.group(1).strip(), None, None

    return None, None, None, None


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _extract_text_from_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    """Approximate extraction of text within a rectangle."""
    words = page.get_text("words")  # list of (x0,y0,x1,y1,word,block,line,word_no)
    parts: List[Tuple[float, float, str]] = []
    for w in words:
        wrect = fitz.Rect(w[0], w[1], w[2], w[3])
        if rect.intersects(wrect):
            parts.append((w[1], w[0], w[4]))  # sort by y, then x
    parts.sort()
    return " ".join(p[2] for p in parts).strip()


def _compute_numeric_boundary_depth(lines_by_page: List[List[str]]) -> Optional[int]:
    """
    Find the *shallowest* numeric depth used anywhere in the PDF text.
    Example:
      - if doc uses 1., 2., 3. => boundary depth = 1 (split on 1.)
      - if doc uses 1.1, 1.2, 2.1 => boundary depth = 2 (split on 1.1 / 1.2 / 2.1)
    """
    depths: List[int] = []
    for lines in lines_by_page:
        for line in lines:
            if not line.strip():
                continue
            kind, label, _title, num_depth = _detect_label_kind(line)
            if kind == "numeric" and label and num_depth is not None:
                depths.append(num_depth)
    return min(depths) if depths else None


def _has_article_or_section_lines(lines_by_page: List[List[str]]) -> bool:
    """
    If the PDF contains ARTICLE/SECTION headings, treat those as the primary clause boundaries.
    In that mode we DO NOT split on numeric labels like 1.1 / 1.2; they become sub-content.
    """
    for lines in lines_by_page:
        for line in lines:
            if not line.strip():
                continue
            kind, _label, _title, _num_depth = _detect_label_kind(line)
            if kind in ("article", "section"):
                return True
    return False


def parse_pdf(input_data: ParsePdfInput) -> DocumentParseResult:
    """
    Parse PDF into DocumentParseResult.

    IMPORTANT UPDATE:
    - We do a *per-clause* breakdown only (no separate Clause objects for subclauses).
    - If the document contains ARTICLE or SECTION headings, we split clauses by those headings.
      Example:
        ARTICLE I + 1.1 + 1.2 + 1.3 => ONE clause
        ARTICLE II + 2.1 + 2.2 => ONE clause
    - If the document does NOT contain ARTICLE/SECTION headings, we fall back to splitting by
      numeric labels at the shallowest numeric depth.
    - Deeper numeric (e.g., 1.1.1) and alpha/roman ((a), (i)) are treated as sub-content and
      remain inside the parent clause text.
    """
    tmp_path: Optional[str] = None
    file_path = input_data.file_path
    if input_data.file_base64:
        raw = base64.b64decode(input_data.file_base64)
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(raw)
        file_path = tmp_path

    assert file_path is not None

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    warnings: List[str] = []
    clauses: List[Clause] = []
    current_clause: Optional[Clause] = None

    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(file_path)
        page_count = doc.page_count

        # Pre-read lines per page (so we can compute numeric boundary depth first)
        lines_by_page: List[List[str]] = []
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            lines = [ln.rstrip() for ln in text.splitlines()]
            lines_by_page.append(lines)

        # KEY CHANGE:
        # If the PDF has ARTICLE/SECTION headings, do NOT split on numeric boundaries.
        group_by_headings = _has_article_or_section_lines(lines_by_page)
        numeric_boundary_depth = None if group_by_headings else _compute_numeric_boundary_depth(lines_by_page)

        # Track whether we have article/section context to set levels.
        seen_article = False
        seen_section = False

        # First pass: build clause list
        for page_index in range(page_count):
            page_num = page_index + 1  # 1-based
            lines = lines_by_page[page_index]

            for line in lines:
                if not line.strip():
                    continue

                kind, label, title, num_depth = _detect_label_kind(line)

                # Decide whether this line starts a NEW clause boundary
                is_boundary = False
                level = 1

                if kind == "article":
                    is_boundary = True
                    level = 1
                    seen_article = True
                    seen_section = False

                elif kind == "section":
                    is_boundary = True
                    level = 2 if seen_article else 1
                    seen_section = True

                elif (
                    kind == "numeric"
                    and (not group_by_headings)
                    and numeric_boundary_depth is not None
                    and num_depth is not None
                    and num_depth == numeric_boundary_depth
                ):
                    is_boundary = True
                    if seen_section:
                        level = 3 if seen_article else 2
                    elif seen_article:
                        level = 2
                    else:
                        level = 1

                else:
                    # alpha/roman and deeper numeric treated as sub-content
                    is_boundary = False

                if is_boundary:
                    if current_clause is not None:
                        clauses.append(current_clause)

                    current_clause = Clause(
                        clause_id="",
                        label=label,
                        title=title,
                        level=max(1, int(level)),
                        parent_clause_id=None,
                        text=line.strip(),
                        raw_spans=[RawSpan(text=line.strip(), kind="normal")] if input_data.options.include_raw_spans else [],
                        redlines=Redlines(),
                        comments=[],
                        source_locations=SourceLocations(page_start=page_num, page_end=page_num),
                    )
                else:
                    if current_clause is None:
                        current_clause = Clause(
                            clause_id="",
                            label=None,
                            title=None,
                            level=1,
                            parent_clause_id=None,
                            text=line.strip(),
                            raw_spans=[RawSpan(text=line.strip(), kind="normal")] if input_data.options.include_raw_spans else [],
                            redlines=Redlines(),
                            comments=[],
                            source_locations=SourceLocations(page_start=page_num, page_end=page_num),
                        )
                    else:
                        current_clause.text = (current_clause.text + "\n" + line.strip()).strip()
                        if input_data.options.include_raw_spans:
                            current_clause.raw_spans.append(RawSpan(text=line.strip(), kind="normal"))

                        # Update page_end if clause spans pages
                        if current_clause.source_locations.page_end is None or page_num > current_clause.source_locations.page_end:
                            current_clause.source_locations.page_end = page_num

            # Ensure page_end stays correct if clause continues but page had no boundaries
            if current_clause is not None:
                if current_clause.source_locations.page_end is None or page_num > current_clause.source_locations.page_end:
                    current_clause.source_locations.page_end = page_num

        if current_clause is not None:
            clauses.append(current_clause)

        # Generate stable IDs and hierarchy
        for idx, c in enumerate(clauses, start=1):
            c.clause_id = stable_clause_id(idx, c.text)

        _assign_hierarchy(clauses)

        # Second pass: annotations.
        any_annots = False
        if input_data.options.extract_annotations:
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                page_num = page_index + 1
                annots = page.annots()
                if not annots:
                    continue

                for annot in annots:
                    any_annots = True
                    atype = annot.type[0] if isinstance(annot.type, tuple) else int(annot.type)
                    info = annot.info or {}
                    content = (info.get("content") or "").strip()
                    author = (info.get("title") or info.get("subject") or None)
                    date = info.get("creationDate") or info.get("modDate") or None

                    target_clause = _find_clause_for_page(clauses, page_num)
                    if target_clause is None:
                        continue

                    if atype in (0, 1):
                        # Text/sticky-note style
                        if content:
                            target_clause.comments.append(
                                Comment(author=author, date=date, text=content, context=target_clause.text)
                            )
                    elif atype == 8:
                        # Highlight: only treat as comment when it has content
                        if content:
                            target_clause.comments.append(
                                Comment(author=author, date=date, text=content, context=target_clause.text)
                            )
                    elif atype == 11:
                        # StrikeOut
                        struck = _extract_text_from_rect(page, annot.rect) or content
                        if struck:
                            target_clause.redlines.strikethroughs.append(
                                RedlineItem(text=struck, context=target_clause.text)
                            )
                            if input_data.options.include_raw_spans:
                                target_clause.raw_spans.append(RawSpan(text=struck, kind="strike"))

        full_text = "\n".join([c.text for c in clauses])
        meta = DocumentMetadata(
            filename=os.path.basename(file_path),
            media_type="application/pdf",
            pages=page_count,
            word_count=_word_count(full_text),
        )

        if input_data.options.extract_annotations and not any_annots:
            warnings.append("No PDF annotations found. Note: not all viewers save annotations in standard form.")

        return DocumentParseResult(document=meta, clauses=clauses, warnings=warnings)

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _find_clause_for_page(clauses: List[Clause], page_num: int) -> Optional[Clause]:
    for c in clauses:
        ps = c.source_locations.page_start
        pe = c.source_locations.page_end
        if ps is None and pe is None:
            continue
        if ps is None:
            ps = pe
        if pe is None:
            pe = ps
        if ps <= page_num <= pe:
            return c
    return clauses[-1] if clauses else None


def _assign_hierarchy(clauses: List[Clause]) -> None:
    stack: List[Clause] = []
    for clause in clauses:
        while stack and stack[-1].level >= clause.level:
            stack.pop()
        clause.parent_clause_id = stack[-1].clause_id if stack else None
        stack.append(clause)