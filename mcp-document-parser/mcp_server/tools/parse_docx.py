from __future__ import annotations

import base64
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from lxml import etree

from ..schemas import (
    Clause,
    ClauseChangeItem,
    ClauseCommentItem,
    ClauseModificationItem,
    Comment,
    DocumentMetadata,
    DocumentParseResult,
    ParseDocxInput,
    RawSpan,
    RedlineItem,
    Redlines,
    SourceLocations,
    stable_clause_id,
)

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

_W = NAMESPACES["w"]


def _w(local: str) -> str:
    return f"{{{_W}}}{local}"


# Clause label patterns
ARTICLE_RE = re.compile(r"^\s*(ARTICLE\s+[IVXLCDM]+)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(SECTION\s+\d+(?:\.\d+)*)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\s*((?:\d+\.)+\d+|\d+\.?)(?=\s)\s+(.+)$")
PAREN_ALPHA_RE = re.compile(r"^\s*(\([a-z]\))(?=\s)\s+(.+)$", re.IGNORECASE)
PAREN_ROMAN_RE = re.compile(r"^\s*(\([ivxlcdm]+\))(?=\s)\s+(.+)$", re.IGNORECASE)


def _numeric_depth(label: str) -> int:
    """
    Compute numeric depth:
      "1." -> 1
      "1.1" -> 2
      "1.1.1" -> 3
    """
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


def _compress_spans(segments: List[Tuple[str, str]]) -> List[RawSpan]:
    """Merge consecutive segments of the same kind into a smaller span list."""
    spans: List[RawSpan] = []
    buf_text: List[str] = []
    buf_kind: Optional[str] = None

    def flush() -> None:
        nonlocal buf_text, buf_kind
        if buf_kind is None:
            return
        text = "".join(buf_text)
        if text:
            spans.append(RawSpan(text=text, kind=buf_kind))
        buf_text = []
        buf_kind = None

    for text, kind in segments:
        if not text:
            continue
        if buf_kind is None:
            buf_kind = kind
            buf_text = [text]
        elif kind == buf_kind:
            buf_text.append(text)
        else:
            flush()
            buf_kind = kind
            buf_text = [text]

    flush()
    return spans


def _walk_text_segments(node: etree._Element, kind: str = "normal") -> Iterable[Tuple[str, str]]:
    """Yield (text, kind) segments from a paragraph node in document order."""

    if node.tag == _w("ins"):
        for child in node:
            yield from _walk_text_segments(child, kind="inserted")
        return

    if node.tag == _w("moveTo"):
        for child in node:
            yield from _walk_text_segments(child, kind="moved_to")
        return

    if node.tag == _w("moveFrom"):
        for child in node:
            yield from _walk_text_segments(child, kind="moved_from")
        return

    if node.tag == _w("del"):
        del_texts: List[str] = []
        for t in node.iter():
            if t.tag in (_w("delText"), _w("t")) and t.text:
                del_texts.append(t.text)
        text = "".join(del_texts)
        if text:
            yield (text, "deleted")
        return

    if node.tag == _w("r"):
        for child in node:
            if child.tag == _w("t") and child.text:
                yield (child.text, kind)
            elif child.tag == _w("delText") and child.text:
                yield (child.text, "deleted")
            elif child.tag == _w("tab"):
                yield ("\t", kind)
            elif child.tag == _w("br"):
                yield ("\n", kind)
            elif child.tag == _w("commentReference"):
                cid = child.get(_w("id")) or child.get("id")
                marker = f"[[comment:{cid}]]" if cid is not None else "[[comment]]"
                yield (marker, "comment_ref")
        return

    if node.tag in (_w("commentRangeStart"), _w("commentRangeEnd")):
        cid = node.get(_w("id")) or node.get("id")
        marker = f"[[comment:{cid}]]" if cid is not None else "[[comment]]"
        yield (marker, "comment_ref")
        return

    # Generic container
    for child in node:
        yield from _walk_text_segments(child, kind=kind)
@dataclass(frozen=True)
class _RevMeta:
    author: Optional[str] = None
    date: Optional[str] = None
    revision_id: Optional[str] = None


@dataclass(frozen=True)
class _ChangeAtom:
    kind: str  # "inserted" | "deleted"
    text: str
    label: Optional[str]
    author: Optional[str]
    date: Optional[str]
    revision_id: Optional[str]
    paragraph_index: int


def _rev_meta_from(el: etree._Element) -> _RevMeta:
    # Tracked-change elements commonly carry w:author, w:date, w:id
    author = el.get(_w("author")) or el.get("author")
    date = el.get(_w("date")) or el.get("date")
    rid = el.get(_w("id")) or el.get("id")
    return _RevMeta(author=author, date=date, revision_id=rid)


def _iter_paragraph_blocks(body: etree._Element) -> List[Tuple[etree._Element, str, _RevMeta]]:
    """Return paragraphs in document order, including paragraphs wrapped by revision containers.

    We intentionally ignore tables for now (per your request). This function is critical because
    Word may store *whole inserted/deleted paragraphs* as <w:ins><w:p>...</w:p></w:ins> or
    <w:del><w:p>...</w:p></w:del> at the body level.

    Returns tuples of (paragraph_element, block_kind, revision_meta) where block_kind is one of:
      - "normal"
      - "inserted" / "deleted"
      - "moved_to" / "moved_from" (treated like inserted/deleted in summaries)
    """

    out: List[Tuple[etree._Element, str, _RevMeta]] = []

    for child in body:
        if child.tag == _w("p"):
            out.append((child, "normal", _RevMeta()))
            continue

        if child.tag in (_w("ins"), _w("del"), _w("moveTo"), _w("moveFrom")):
            kind = {
                _w("ins"): "inserted",
                _w("del"): "deleted",
                _w("moveTo"): "moved_to",
                _w("moveFrom"): "moved_from",
            }[child.tag]
            meta = _rev_meta_from(child)
            for gc in child:
                if gc.tag == _w("p"):
                    out.append((gc, kind, meta))
            continue

        # Ignore tables and other block types for now.

    return out


def _walk_text_segments_with_meta(
    node: etree._Element,
    kind: str = "normal",
    meta: Optional[_RevMeta] = None,
) -> Iterable[Tuple[str, str, _RevMeta]]:
    """Yield (text, kind, rev_meta) segments from a paragraph node in document order."""

    if meta is None:
        meta = _RevMeta()

    if node.tag == _w("ins"):
        m = _rev_meta_from(node)
        for child in node:
            yield from _walk_text_segments_with_meta(child, kind="inserted", meta=m)
        return

    if node.tag == _w("moveTo"):
        m = _rev_meta_from(node)
        for child in node:
            yield from _walk_text_segments_with_meta(child, kind="moved_to", meta=m)
        return

    if node.tag == _w("del"):
        m = _rev_meta_from(node)
        del_texts: List[str] = []
        for t in node.iter():
            if t.tag in (_w("delText"), _w("t")) and t.text:
                del_texts.append(t.text)
        text = "".join(del_texts)
        if text:
            yield (text, "deleted", m)
        return

    if node.tag == _w("moveFrom"):
        m = _rev_meta_from(node)
        del_texts: List[str] = []
        for t in node.iter():
            if t.tag in (_w("delText"), _w("t")) and t.text:
                del_texts.append(t.text)
        text = "".join(del_texts)
        if text:
            yield (text, "moved_from", m)
        return

    if node.tag == _w("r"):
        for child in node:
            if child.tag == _w("t") and child.text:
                yield (child.text, kind, meta)
            elif child.tag == _w("delText") and child.text:
                yield (child.text, "deleted", meta)
            elif child.tag == _w("tab"):
                yield ("\t", kind, meta)
            elif child.tag == _w("br"):
                yield ("\n", kind, meta)
            elif child.tag == _w("commentReference"):
                cid = child.get(_w("id")) or child.get("id")
                marker = f"[[comment:{cid}]]" if cid is not None else "[[comment]]"
                yield (marker, "comment_ref", meta)
        return

    if node.tag in (_w("commentRangeStart"), _w("commentRangeEnd")):
        cid = node.get(_w("id")) or node.get("id")
        marker = f"[[comment:{cid}]]" if cid is not None else "[[comment]]"
        yield (marker, "comment_ref", meta)
        return

    for child in node:
        yield from _walk_text_segments_with_meta(child, kind=kind, meta=meta)


def _extract_comment_anchors_from_paragraph(p: etree._Element, default_kind: str) -> Dict[str, str]:
    """Best-effort extraction of comment anchor snippets for a single paragraph.

    We track active commentRangeStart/End IDs and capture the visible (normal/inserted/moved_to)
    text that occurs while a given comment range is active.
    """

    anchors: Dict[str, List[str]] = defaultdict(list)
    active: List[str] = []

    def walk(node: etree._Element, k: str) -> None:
        # Revision containers adjust the kind for their children
        if node.tag == _w("ins"):
            for ch in node:
                walk(ch, "inserted")
            return
        if node.tag == _w("moveTo"):
            for ch in node:
                walk(ch, "moved_to")
            return
        if node.tag == _w("del"):
            # ignore deleted text for anchor snippets
            return
        if node.tag == _w("moveFrom"):
            # ignore moved-from text for anchor snippets
            return

        if node.tag == _w("commentRangeStart"):
            cid = node.get(_w("id")) or node.get("id")
            if cid is not None:
                active.append(str(cid))
            return

        if node.tag == _w("commentRangeEnd"):
            cid = node.get(_w("id")) or node.get("id")
            if cid is not None:
                sid = str(cid)
                # remove the last occurrence (ranges can nest)
                for i in range(len(active) - 1, -1, -1):
                    if active[i] == sid:
                        active.pop(i)
                        break
            return

        if node.tag == _w("r"):
            for ch in node:
                if ch.tag == _w("t") and ch.text:
                    if k in ("normal", "inserted", "moved_to") and active:
                        for cid in active:
                            anchors[cid].append(ch.text)
                elif ch.tag == _w("tab"):
                    if k in ("normal", "inserted", "moved_to") and active:
                        for cid in active:
                            anchors[cid].append("\t")
                elif ch.tag == _w("br"):
                    if k in ("normal", "inserted", "moved_to") and active:
                        for cid in active:
                            anchors[cid].append("\n")
            return

        for ch in node:
            walk(ch, k)

    walk(p, default_kind)

    out: Dict[str, str] = {}
    for cid, parts in anchors.items():
        txt = "".join(parts).strip()
        if txt:
            out[cid] = txt
    return out


def _extract_change_atoms_from_segments(
    segments: List[Tuple[str, str, _RevMeta]],
    label_guess: Optional[str],
    paragraph_index: int,
) -> List[_ChangeAtom]:
    """Build summarized change atoms from inline tracked changes in a paragraph."""

    atoms: List[_ChangeAtom] = []

    buf_kind: Optional[str] = None
    buf_meta: Optional[_RevMeta] = None
    buf_parts: List[str] = []

    def norm_kind(k: str) -> Optional[str]:
        if k in ("inserted", "moved_to"):
            return "inserted"
        if k in ("deleted", "moved_from"):
            return "deleted"
        return None

    def flush() -> None:
        nonlocal buf_kind, buf_meta, buf_parts
        if buf_kind is None or buf_meta is None:
            buf_kind = None
            buf_meta = None
            buf_parts = []
            return
        text = "".join(buf_parts).strip()
        if text:
            atoms.append(
                _ChangeAtom(
                    kind=buf_kind,
                    text=text,
                    label=label_guess,
                    author=buf_meta.author,
                    date=buf_meta.date,
                    revision_id=buf_meta.revision_id,
                    paragraph_index=paragraph_index,
                )
            )
        buf_kind = None
        buf_meta = None
        buf_parts = []

    for text, k, m in segments:
        nk = norm_kind(k)
        if nk is None:
            flush()
            continue

        if buf_kind is None:
            buf_kind = nk
            buf_meta = m
            buf_parts = [text]
        else:
            if nk == buf_kind and m == buf_meta:
                buf_parts.append(text)
            else:
                flush()
                buf_kind = nk
                buf_meta = m
                buf_parts = [text]

    flush()
    return atoms


def _merge_comment_item(dst: Dict[str, ClauseCommentItem], item: ClauseCommentItem) -> None:
    """Merge comment items by ID, concatenating anchor_text when repeated."""
    cid = item.comment_id or ""
    if not cid:
        # If no comment_id, just append with a synthetic key
        dst[f"_anon_{len(dst)}"] = item
        return

    existing = dst.get(cid)
    if existing is None:
        dst[cid] = item
        return

    # Prefer existing metadata/text; merge anchor snippets.
    if item.anchor_text:
        if existing.anchor_text:
            if item.anchor_text not in existing.anchor_text:
                existing.anchor_text = (existing.anchor_text + "\n" + item.anchor_text).strip()
        else:
            existing.anchor_text = item.anchor_text


def _populate_clause_changes(
    clause: Clause,
    atoms: List[_ChangeAtom],
    comment_items: Dict[str, ClauseCommentItem],
) -> None:
    """Populate clause.changes.{added,deleted,modified,comments} from raw atoms + anchored comments."""

    # Deduplicate atoms while preserving order
    seen = set()
    deduped: List[_ChangeAtom] = []
    for a in atoms:
        key = (a.kind, a.text, a.label, a.author, a.date, a.revision_id, a.paragraph_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    used = set()
    n = len(deduped)

    # Pair delete->insert as a "modified" when they are near each other and look related.
    for i, a in enumerate(deduped):
        if a.kind != "deleted" or i in used:
            continue

        best_j: Optional[int] = None
        best_score = 0

        for j in range(i + 1, min(n, i + 7)):
            b = deduped[j]
            if b.kind != "inserted" or j in used:
                continue

            score = 0
            if a.paragraph_index == b.paragraph_index:
                score += 3
            if a.label and b.label and a.label == b.label:
                score += 2
            if a.revision_id and b.revision_id and a.revision_id == b.revision_id:
                score += 2
            if a.author and b.author and a.author == b.author:
                score += 1

            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None and best_score >= 3:
            b = deduped[best_j]
            used.add(i)
            used.add(best_j)
            clause.changes.modified.append(
                ClauseModificationItem(
                    from_text=a.text,
                    to_text=b.text,
                    label=a.label or b.label,
                    author=b.author or a.author,
                    date=b.date or a.date,
                    revision_id=b.revision_id or a.revision_id,
                )
            )

    # Remaining atoms become added/deleted
    for i, a in enumerate(deduped):
        if i in used:
            continue
        if a.kind == "inserted":
            clause.changes.added.append(
                ClauseChangeItem(
                    text=a.text,
                    label=a.label,
                    author=a.author,
                    date=a.date,
                    revision_id=a.revision_id,
                )
            )
        elif a.kind == "deleted":
            clause.changes.deleted.append(
                ClauseChangeItem(
                    text=a.text,
                    label=a.label,
                    author=a.author,
                    date=a.date,
                    revision_id=a.revision_id,
                )
            )

    # Comments for the changes panel
    clause.changes.comments.extend(list(comment_items.values()))


def _extract_comments_from_comments_xml(xml_bytes: bytes) -> Dict[str, Comment]:
    comments: Dict[str, Comment] = {}
    root = etree.fromstring(xml_bytes)
    for c in root.findall("w:comment", namespaces=NAMESPACES):
        cid = c.get(_w("id")) or c.get("id")
        if cid is None:
            continue
        author = c.get(_w("author")) or c.get("author")
        date = c.get(_w("date")) or c.get("date")
        parts: List[str] = []
        for t in c.iter():
            if t.tag == _w("t") and t.text:
                parts.append(t.text)
            elif t.tag == _w("br"):
                parts.append("\n")
            elif t.tag == _w("tab"):
                parts.append("\t")
        text = "".join(parts).strip()
        if not text:
            continue
        comments[str(cid)] = Comment(author=author, date=date, text=text)
    return comments


def _paragraph_comment_ids(p: etree._Element) -> List[str]:
    ids: List[str] = []
    for el in p.iter():
        if el.tag in (_w("commentRangeStart"), _w("commentReference")):
            cid = el.get(_w("id")) or el.get("id")
            if cid is not None:
                ids.append(str(cid))
    # preserve order but dedupe
    seen = set()
    out: List[str] = []
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _compute_numeric_boundary_depth(paragraphs: List[etree._Element]) -> Optional[int]:
    """
    Find the *shallowest* numeric depth used anywhere in the doc.
    Example:
      - if doc uses 1., 2., 3. => boundary depth = 1 (split on 1.)
      - if doc uses 1.1, 1.2, 2.1 => boundary depth = 2 (split on 1.1 / 1.2 / 2.1)
    """
    depths: List[int] = []
    for p in paragraphs:
        segments = list(_walk_text_segments(p, kind="normal"))
        clean_text = "".join(t for t, k in segments if k in ("normal", "inserted")).replace("\r", "")
        kind, label, _title, num_depth = _detect_label_kind(clean_text)
        if kind == "numeric" and label and num_depth is not None:
            depths.append(num_depth)
    return min(depths) if depths else None


def _has_article_or_section(paragraphs: List[etree._Element]) -> bool:
    """
    If the document contains ARTICLE/SECTION headings, treat those as the primary clause boundaries.
    In that mode we DO NOT split on numeric labels like 1.1 / 1.2; they become sub-content.
    """
    for p in paragraphs:
        segments = list(_walk_text_segments(p, kind="normal"))
        clean_text = "".join(t for t, k in segments if k in ("normal", "inserted")).replace("\r", "")
        kind, _label, _title, _num_depth = _detect_label_kind(clean_text)
        if kind in ("article", "section"):
            return True
    return False


def parse_docx(input_data: ParseDocxInput) -> DocumentParseResult:
    """
    Parse a .docx file into a structured DocumentParseResult.

    UPDATED BEHAVIOR (what you asked for):
    - If the document contains ARTICLE or SECTION headings, we split clauses by those headings.
      Example:
        ARTICLE I + 1.1 + 1.2 + 1.3 => ONE clause
        ARTICLE II + 2.1 + 2.2 => ONE clause
    - If the document does NOT contain ARTICLE/SECTION headings, we fall back to splitting by
      numeric labels at the shallowest numeric depth.
    - We never emit separate Clause objects for alpha/roman or deeper numeric patterns; those are
      kept inside the parent clause text.
    """

    tmp_path: Optional[str] = None
    file_path = input_data.file_path

    if input_data.file_base64:
        raw = base64.b64decode(input_data.file_base64)
        fd, tmp_path = tempfile.mkstemp(suffix=".docx")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(raw)
        file_path = tmp_path

    assert file_path is not None

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    warnings: List[str] = []

    try:
        with zipfile.ZipFile(file_path, "r") as z:
            try:
                document_xml = z.read("word/document.xml")
            except KeyError as e:
                raise ValueError("Invalid DOCX: missing word/document.xml") from e

            comments_map: Dict[str, Comment] = {}
            if input_data.options.extract_comments:
                try:
                    comments_xml = z.read("word/comments.xml")
                    comments_map = _extract_comments_from_comments_xml(comments_xml)
                except KeyError:
                    comments_map = {}

            root = etree.fromstring(document_xml)
            body = root.find("w:body", namespaces=NAMESPACES)
            if body is None:
                raise ValueError("Invalid DOCX: missing w:body")

            paragraph_blocks = _iter_paragraph_blocks(body)

            # IMPORTANT: boundary detection should consider only *visible* paragraphs.
            visible_paragraphs = [p for p, bk, _m in paragraph_blocks if bk not in ("deleted", "moved_from")]

            # KEY CHANGE:
            # If the doc has ARTICLE/SECTION headings, do NOT split on numeric boundaries.
            group_by_headings = _has_article_or_section(visible_paragraphs)
            numeric_boundary_depth = None if group_by_headings else _compute_numeric_boundary_depth(visible_paragraphs)

            clauses: List[Clause] = []
            current_clause: Optional[Clause] = None

            seen_article = False
            seen_section = False

            # Per-clause accumulators (keyed by object id)
            atoms_by_clause: Dict[int, List[_ChangeAtom]] = defaultdict(list)
            change_comments_by_clause: Dict[int, Dict[str, ClauseCommentItem]] = defaultdict(dict)

            for p_index, (p, block_kind, block_meta) in enumerate(paragraph_blocks):
                default_kind = "normal" if block_kind == "normal" else block_kind

                seg3 = list(_walk_text_segments_with_meta(p, kind=default_kind, meta=block_meta))
                segments_simple = [(t, k) for t, k, _m in seg3]
                raw_spans = _compress_spans(segments_simple) if input_data.options.include_raw_spans else []

                clean_text = "".join(
                    t for t, k, _m in seg3 if k in ("normal", "inserted", "moved_to")
                ).replace("\r", "")
                deleted_text = "".join(
                    t for t, k, _m in seg3 if k in ("deleted", "moved_from")
                ).replace("\r", "")

                # comments (raw)
                comment_ids = _paragraph_comment_ids(p) if input_data.options.extract_comments else []

                # Skip empty/no-signal paragraphs unless they contain changes/comments.
                has_signal = bool(clean_text.strip() or deleted_text.strip() or raw_spans or comment_ids)
                if not has_signal:
                    continue

                # For boundary detection, we only consider visible paragraphs (not deleted/moved_from).
                kind, label, title, num_depth = _detect_label_kind(clean_text)

                # For change summaries, we want a best-effort label even for deleted paragraphs.
                label_guess: Optional[str] = None
                visible_for_label = clean_text if clean_text.strip() else deleted_text
                k2, l2, _t2, _d2 = _detect_label_kind(visible_for_label)
                if k2 == "numeric" and l2:
                    label_guess = l2

                # redlines + change atoms
                redlines = Redlines()
                change_atoms: List[_ChangeAtom] = []

                if input_data.options.extract_tracked_changes:
                    if block_kind in ("inserted", "moved_to") and clean_text.strip():
                        txt = clean_text.strip()
                        redlines.insertions.append(RedlineItem(text=txt, context=None))
                        change_atoms.append(
                            _ChangeAtom(
                                kind="inserted",
                                text=txt,
                                label=label_guess,
                                author=block_meta.author,
                                date=block_meta.date,
                                revision_id=block_meta.revision_id,
                                paragraph_index=p_index,
                            )
                        )
                    elif block_kind in ("deleted", "moved_from") and deleted_text.strip():
                        txt = deleted_text.strip()
                        redlines.deletions.append(RedlineItem(text=txt, context=None))
                        change_atoms.append(
                            _ChangeAtom(
                                kind="deleted",
                                text=txt,
                                label=label_guess,
                                author=block_meta.author,
                                date=block_meta.date,
                                revision_id=block_meta.revision_id,
                                paragraph_index=p_index,
                            )
                        )
                    else:
                        change_atoms = _extract_change_atoms_from_segments(seg3, label_guess, p_index)
                        for a in change_atoms:
                            if a.kind == "inserted":
                                redlines.insertions.append(RedlineItem(text=a.text, context=None))
                            elif a.kind == "deleted":
                                redlines.deletions.append(RedlineItem(text=a.text, context=None))

                # comments + anchors for the changes panel
                anchors = _extract_comment_anchors_from_paragraph(p, default_kind) if input_data.options.extract_comments else {}
                comments: List[Comment] = []
                change_comment_items: Dict[str, ClauseCommentItem] = {}
                if input_data.options.extract_comments and comment_ids:
                    for cid in comment_ids:
                        c = comments_map.get(cid)
                        if not c:
                            continue
                        comments.append(Comment(author=c.author, date=c.date, text=c.text, context=None))
                        change_comment_items[cid] = ClauseCommentItem(
                            comment_id=cid,
                            author=c.author,
                            date=c.date,
                            text=c.text,
                            anchor_text=anchors.get(cid) or None,
                        )

                # Determine if this paragraph starts a new clause
                is_boundary = False
                level = 1

                if block_kind not in ("deleted", "moved_from"):
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

                if is_boundary:
                    if current_clause is not None:
                        clauses.append(current_clause)

                    current_clause = Clause(
                        clause_id="",  # set later
                        label=label,
                        title=title,
                        level=max(1, int(level)),
                        parent_clause_id=None,
                        text=clean_text.strip(),
                        raw_spans=raw_spans,
                        redlines=redlines,
                        comments=comments,
                        source_locations=SourceLocations(),
                    )

                    # Accumulate changes/comments for this new clause
                    atoms_by_clause[id(current_clause)].extend(change_atoms)
                    for _cid, item in change_comment_items.items():
                        _merge_comment_item(change_comments_by_clause[id(current_clause)], item)

                else:
                    if current_clause is None:
                        # Preamble (and/or leading deleted text) before first visible heading becomes a single clause
                        current_clause = Clause(
                            clause_id="",
                            label=None,
                            title=None,
                            level=1,
                            parent_clause_id=None,
                            text=clean_text.strip(),
                            raw_spans=raw_spans,
                            redlines=redlines,
                            comments=comments,
                            source_locations=SourceLocations(),
                        )
                    else:
                        if clean_text.strip():
                            current_clause.text = (current_clause.text + "\n" + clean_text.strip()).strip()
                        if input_data.options.include_raw_spans:
                            current_clause.raw_spans.extend(raw_spans)
                        current_clause.redlines.insertions.extend(redlines.insertions)
                        current_clause.redlines.deletions.extend(redlines.deletions)
                        current_clause.comments.extend(comments)

                    # Accumulate changes/comments onto the current clause
                    atoms_by_clause[id(current_clause)].extend(change_atoms)
                    for _cid, item in change_comment_items.items():
                        _merge_comment_item(change_comments_by_clause[id(current_clause)], item)

            if current_clause is not None:
                clauses.append(current_clause)

            # Populate clause.changes AFTER clause text has been finalized.
            for c in clauses:
                _populate_clause_changes(
                    c,
                    atoms_by_clause.get(id(c), []),
                    change_comments_by_clause.get(id(c), {}),
                )

            # Assign contexts now that clause text is final
            for c in clauses:
                for item in c.redlines.insertions:
                    item.context = c.text
                for item in c.redlines.deletions:
                    item.context = c.text
                for com in c.comments:
                    com.context = c.text

            # Generate stable IDs
            for idx, c in enumerate(clauses, start=1):
                c.clause_id = stable_clause_id(idx, c.text)

            _assign_hierarchy(clauses)

            full_text = "\n".join([c.text for c in clauses])
            word_count = _word_count(full_text)

            meta = DocumentMetadata(
                filename=os.path.basename(file_path),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                pages=None,
                word_count=word_count,
            )

            if input_data.options.extract_comments and not comments_map:
                warnings.append("No comments.xml part found; no comments extracted.")

            if input_data.options.extract_tracked_changes:
                any_redlines = any(
                    c.redlines.insertions or c.redlines.deletions or c.redlines.strikethroughs
                    for c in clauses
                )
                if not any_redlines:
                    warnings.append("No tracked changes found in document.")

            return DocumentParseResult(document=meta, clauses=clauses, warnings=warnings)

    except zipfile.BadZipFile as e:
        raise ValueError("Invalid DOCX: not a valid zip archive") from e
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _assign_hierarchy(clauses: List[Clause]) -> None:
    """Assign parent_clause_id based on level using a simple stack."""
    stack: List[Clause] = []

    for clause in clauses:
        while stack and stack[-1].level >= clause.level:
            stack.pop()
        clause.parent_clause_id = stack[-1].clause_id if stack else None
        stack.append(clause)