# mcp_server/tools/parse_docx2.py
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

# --------------------------------------------------------------------------------------
# parse_docx2
#
# Goals for "Word heading + TOC + tracked changes" contract format:
#   1) Robustly SKIP the Table of Contents (TOC) section:
#        - skip TOC paragraphs by style (TOC1/TOC2/.../TOCHeading)
#        - skip paragraphs that are part of a TOC field result (fldChar/instrText)
#        - skip literal "Table of Contents" heading
#   2) Prefer Word-native heading detection when available:
#        - use outline level (w:outlineLvl) and/or paragraph styles that map to outline levels
#        - fall back to ARTICLE/SECTION/numeric regex when needed
#   3) Handle auto-numbered headings (w:numPr) well:
#        - attempt to compute numbering labels from numbering.xml (best-effort)
#        - if the visible paragraph text does NOT contain the number, we can still set Clause.label
#   4) Preserve tracked changes + comments exactly like your current parser:
#        - w:ins/w:del/moveTo/moveFrom (block-level and inline)
#        - comments via comments.xml + anchored snippets
#
# Notes:
#   - Strikethroughs in your workflow come from Track Changes deletions (w:del), which we already treat
#     as deletions. (No need to parse w:strike formatting for this "parse_docx2" request.)
#   - Tables are intentionally ignored (same as your original).
# --------------------------------------------------------------------------------------

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_W = NAMESPACES["w"]


def _w(local: str) -> str:
    return f"{{{_W}}}{local}"


# -----------------------------
# Label patterns (fallback)
# -----------------------------
ARTICLE_RE = re.compile(r"^\s*(ARTICLE\s+[IVXLCDM]+)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(SECTION\s+\d+(?:\.\d+)*)\b[\s.:\-]*([^\n]*)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\s*((?:\d+\.)+\d+|\d+\.?)(?=\s)\s+(.+)$")
PAREN_ALPHA_RE = re.compile(r"^\s*(\([a-z]\))(?=\s)\s+(.+)$", re.IGNORECASE)
PAREN_ROMAN_RE = re.compile(r"^\s*(\([ivxlcdm]+\))(?=\s)\s+(.+)$", re.IGNORECASE)


TOC_STYLE_RE = re.compile(r"^TOC\d+$", re.IGNORECASE)

# -----------------------------
# Termset extraction (footer)
# -----------------------------
# Expected footer token: "CTM-P-ST-002" -> termset_id="002" (numeric suffix only)
# We normalize 1-3 digit values to 3 digits so it aligns with policy metadata (e.g., 2 -> 002).
_TERMSET_FOOTER_RE = re.compile(
    r"CTM\s*[-\u2010-\u2015\u2212]\s*P\s*[-\u2010-\u2015\u2212]\s*ST\s*[-\u2010-\u2015\u2212]\s*(\d{1,6})",
    re.IGNORECASE,
)

_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_REL_TAG = f"{{{_REL_NS}}}Relationship"



def _normalize_termset_suffix(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    if not s.isdigit():
        return s
    # normalize 1-3 digits to 3-digit strings; keep longer values as-is
    if len(s) <= 3:
        try:
            return f"{int(s):03d}"
        except Exception:
            return s
    return s


# Helper: scan a single footer XML part for CTM-P-ST-xxx, return normalized numeric suffix.
def _scan_footer_xml_for_termset(footer_xml: bytes) -> Optional[str]:
    """Scan a single footer XML part for a CTM-P-ST-xxx token and return the normalized numeric suffix."""
    try:
        footer_root = etree.fromstring(footer_xml)
    except Exception:
        return None

    texts = [t.text for t in footer_root.findall(".//w:t", namespaces=NAMESPACES) if t.text]
    if not texts:
        return None

    # Footers often split tokens across runs; try both concatenated and spaced variants.
    cand1 = "".join(texts)
    cand2 = " ".join(texts)

    for cand in (cand1, cand2):
        m = _TERMSET_FOOTER_RE.search(cand or "")
        if m:
            return _normalize_termset_suffix(m.group(1) or "")

    return None


def _extract_termset_id_from_docx(z: zipfile.ZipFile) -> Optional[str]:
    """Best-effort extraction of a termset id from DOCX footers.

    Strategy:
      1) Read `word/_rels/document.xml.rels` to find footer relationship targets.
      2) Scan each referenced `word/footer*.xml` for a token like CTM-P-ST-002.
      3) Return only the numeric suffix (e.g., "002").
    """
    try:
        rels_xml = z.read("word/_rels/document.xml.rels")
    except KeyError:
        return None

    try:
        rels_root = etree.fromstring(rels_xml)
    except Exception:
        return None

    targets: List[str] = []
    for rel in rels_root.findall(f".//{_REL_TAG}"):
        typ = (rel.get("Type") or "").strip()
        if typ.endswith("/footer"):
            tgt = (rel.get("Target") or "").strip()
            if tgt:
                targets.append(tgt)

    if not targets:
        return None

    # De-dup targets while preserving order
    seen: set[str] = set()
    ordered: List[str] = []
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)

    for tgt in ordered:
        # Targets are typically like "footer1.xml"; occasionally they can be prefixed (./ or ../).
        norm = (tgt or "").strip()
        if not norm:
            continue
        norm = norm.lstrip("/")
        while norm.startswith("./"):
            norm = norm[2:]
        while norm.startswith("../"):
            norm = norm[3:]

        if norm.startswith("word/"):
            footer_path = norm
        else:
            footer_path = "word/" + norm

        try:
            footer_xml = z.read(footer_path)
        except KeyError:
            continue

        found = _scan_footer_xml_for_termset(footer_xml)
        if found:
            return found

    # Fallback: scan any footer parts in the archive even if relationships are missing/unusual.
    try:
        for name in z.namelist():
            if not name.startswith("word/footer") or not name.lower().endswith(".xml"):
                continue
            try:
                footer_xml = z.read(name)
            except KeyError:
                continue
            found = _scan_footer_xml_for_termset(footer_xml)
            if found:
                return found
    except Exception:
        pass

    return None


def _numeric_depth(label: str) -> int:
    raw = label.rstrip(".")
    parts = [p for p in raw.split(".") if p]
    return max(1, len(parts))


def _detect_label_kind(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """Return (kind, label, title, numeric_depth_if_any)."""
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
        title = m.group(2).strip() or None
        return "numeric", label, title, _numeric_depth(label)

    m = PAREN_ROMAN_RE.match(text)
    if m:
        return "roman", m.group(1).strip(), None, None

    m = PAREN_ALPHA_RE.match(text)
    if m:
        return "alpha", m.group(1).strip(), None, None

    return None, None, None, None


# -----------------------------
# Raw spans helpers
# -----------------------------
def _compress_spans(segments: List[Tuple[str, str]]) -> List[RawSpan]:
    spans: List[RawSpan] = []
    buf_text: List[str] = []
    buf_kind: Optional[str] = None

    def flush() -> None:
        nonlocal buf_text, buf_kind
        if buf_kind is None:
            return
        txt = "".join(buf_text)
        if txt:
            spans.append(RawSpan(text=txt, kind=buf_kind))
        buf_text = []
        buf_kind = None

    for txt, k in segments:
        if not txt:
            continue
        if buf_kind is None:
            buf_kind = k
            buf_text = [txt]
        elif k == buf_kind:
            buf_text.append(txt)
        else:
            flush()
            buf_kind = k
            buf_text = [txt]

    flush()
    return spans


# -----------------------------
# Revision metadata + blocks
# -----------------------------
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
    author = el.get(_w("author")) or el.get("author")
    date = el.get(_w("date")) or el.get("date")
    rid = el.get(_w("id")) or el.get("id")
    return _RevMeta(author=author, date=date, revision_id=rid)


def _iter_paragraph_blocks(body: etree._Element) -> List[Tuple[etree._Element, str, _RevMeta]]:
    """Return paragraphs in document order including paragraphs wrapped by revision containers."""
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
        txt = "".join(del_texts)
        if txt:
            yield (txt, "deleted", m)
        return

    if node.tag == _w("moveFrom"):
        m = _rev_meta_from(node)
        del_texts: List[str] = []
        for t in node.iter():
            if t.tag in (_w("delText"), _w("t")) and t.text:
                del_texts.append(t.text)
        txt = "".join(del_texts)
        if txt:
            yield (txt, "moved_from", m)
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


# -----------------------------
# Comments + anchors
# -----------------------------
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
        txt = "".join(parts).strip()
        if not txt:
            continue
        comments[str(cid)] = Comment(author=author, date=date, text=txt)
    return comments


def _paragraph_comment_ids(p: etree._Element) -> List[str]:
    ids: List[str] = []
    for el in p.iter():
        if el.tag in (_w("commentRangeStart"), _w("commentReference")):
            cid = el.get(_w("id")) or el.get("id")
            if cid is not None:
                ids.append(str(cid))
    seen = set()
    out: List[str] = []
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _extract_comment_anchors_from_paragraph(p: etree._Element, default_kind: str) -> Dict[str, str]:
    anchors: Dict[str, List[str]] = defaultdict(list)
    active: List[str] = []

    def walk(node: etree._Element, k: str) -> None:
        if node.tag == _w("ins"):
            for ch in node:
                walk(ch, "inserted")
            return
        if node.tag == _w("moveTo"):
            for ch in node:
                walk(ch, "moved_to")
            return
        if node.tag in (_w("del"), _w("moveFrom")):
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


# -----------------------------
# Change atoms -> clause.changes
# -----------------------------
def _extract_change_atoms_from_segments(
    segments: List[Tuple[str, str, _RevMeta]],
    label_guess: Optional[str],
    paragraph_index: int,
) -> List[_ChangeAtom]:
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
        txt = "".join(buf_parts).strip()
        if txt:
            atoms.append(
                _ChangeAtom(
                    kind=buf_kind,
                    text=txt,
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

    for txt, k, m in segments:
        nk = norm_kind(k)
        if nk is None:
            flush()
            continue

        if buf_kind is None:
            buf_kind = nk
            buf_meta = m
            buf_parts = [txt]
        else:
            if nk == buf_kind and m == buf_meta:
                buf_parts.append(txt)
            else:
                flush()
                buf_kind = nk
                buf_meta = m
                buf_parts = [txt]

    flush()
    return atoms


def _merge_comment_item(dst: Dict[str, ClauseCommentItem], item: ClauseCommentItem) -> None:
    cid = item.comment_id or ""
    if not cid:
        dst[f"_anon_{len(dst)}"] = item
        return

    existing = dst.get(cid)
    if existing is None:
        dst[cid] = item
        return

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

    clause.changes.comments.extend(list(comment_items.values()))


# -----------------------------
# Word-native heading / styles
# -----------------------------
def _has_numpr(p: etree._Element) -> bool:
    """Return True if the paragraph uses Word list/outline numbering (w:numPr).

    Many contract subclauses (a)/(b)/(c), 1.1/1.2, etc. are implemented as numbered lists.
    We DO NOT want those to become separate Clause objects.
    """
    try:
        return p.find("w:pPr/w:numPr", namespaces=NAMESPACES) is not None
    except Exception:
        return False

def _pstyle_id(p: etree._Element) -> Optional[str]:
    el = p.find("w:pPr/w:pStyle", namespaces=NAMESPACES)
    if el is None:
        return None
    sid = el.get(_w("val")) or el.get("val") or ""
    sid = sid.strip()
    return sid or None


def _p_outline_lvl_direct(p: etree._Element) -> Optional[int]:
    el = p.find("w:pPr/w:outlineLvl", namespaces=NAMESPACES)
    if el is None:
        return None
    v = el.get(_w("val")) or el.get("val")
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except Exception:
        return None


def _load_style_outline_levels(z: zipfile.ZipFile) -> Dict[str, int]:
    """Return styleId -> outlineLvl for paragraph styles that define it (best-effort)."""
    out: Dict[str, int] = {}
    try:
        styles_xml = z.read("word/styles.xml")
    except KeyError:
        return out

    try:
        root = etree.fromstring(styles_xml)
    except Exception:
        return out

    for style in root.findall(".//w:style", namespaces=NAMESPACES):
        style_id = style.get(_w("styleId")) or style.get("styleId")
        if not style_id:
            continue
        # only paragraph styles
        st_type = style.get(_w("type")) or style.get("type")
        if st_type and str(st_type).strip().lower() != "paragraph":
            continue

        ol = style.find(".//w:pPr/w:outlineLvl", namespaces=NAMESPACES)
        if ol is None:
            continue
        v = ol.get(_w("val")) or ol.get("val")
        if v is None:
            continue
        try:
            out[str(style_id)] = int(str(v).strip())
        except Exception:
            continue

    return out


def _paragraph_outline_level(p: etree._Element, style_outline: Dict[str, int]) -> Optional[int]:
    direct = _p_outline_lvl_direct(p)
    if direct is not None:
        return direct
    sid = _pstyle_id(p)
    if sid and sid in style_outline:
        return style_outline[sid]
    # common heading styles without outlineLvl defined explicitly
    if sid and sid.lower().startswith("heading"):
        # Heading1 -> 0, Heading2 -> 1, ...
        m = re.match(r"^heading\s*([1-9])$", sid.strip().lower())
        if m:
            return int(m.group(1)) - 1
    return None


# -----------------------------
# TOC detection (skip it)
# -----------------------------
@dataclass
class _TocState:
    active: bool = False
    is_toc: bool = False
    in_result: bool = False

    def update_from_paragraph(self, p: etree._Element) -> None:
        """Best-effort tracking of complex field begin/separate/end for TOC fields."""
        for el in p.iter():
            if el.tag == _w("fldSimple"):
                instr = el.get(_w("instr")) or el.get("instr") or ""
                if instr.strip().upper().startswith("TOC"):
                    # Entire fldSimple is effectively a TOC field; treat paragraph as TOC-ish.
                    self.active = True
                    self.is_toc = True
                    self.in_result = True

            if el.tag == _w("fldChar"):
                t = el.get(_w("fldCharType")) or el.get("fldCharType") or ""
                t = str(t).strip().lower()
                if t == "begin":
                    self.active = True
                    self.is_toc = False
                    self.in_result = False
                elif t == "separate":
                    if self.active and self.is_toc:
                        self.in_result = True
                elif t == "end":
                    self.active = False
                    self.is_toc = False
                    self.in_result = False

            if el.tag == _w("instrText"):
                if self.active:
                    txt = (el.text or "").strip().upper()
                    if txt.startswith("TOC"):
                        self.is_toc = True


def _is_toc_style_id(style_id: Optional[str]) -> bool:
    if not style_id:
        return False
    sid = style_id.strip()
    if not sid:
        return False
    if TOC_STYLE_RE.match(sid):
        return True
    if sid.lower() in ("tocheading", "toc", "tocheading1", "tocheading"):
        return True
    return False


def _is_toc_paragraph(clean_text: str, style_id: Optional[str], toc_state: _TocState) -> bool:
    # literal heading
    if clean_text.strip().upper() in ("TABLE OF CONTENTS", "CONTENTS"):
        return True
    if _is_toc_style_id(style_id):
        return True
    # while TOC field is producing results
    if toc_state.is_toc and toc_state.in_result:
        return True
    return False


# -----------------------------
# Numbering (auto-labeled) best-effort
# -----------------------------
def _roman(n: int) -> str:
    if n <= 0:
        return ""
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    x = n
    for v, s in vals:
        while x >= v:
            out.append(s)
            x -= v
    return "".join(out)


def _alpha(n: int, upper: bool) -> str:
    if n <= 0:
        return ""
    # 1->A, 26->Z, 27->AA
    n0 = n
    chars = []
    while n0 > 0:
        n0 -= 1
        chars.append(chr((n0 % 26) + (65 if upper else 97)))
        n0 //= 26
    return "".join(reversed(chars))


@dataclass
class _NumLvlDef:
    num_fmt: str
    lvl_text: str


class _NumberingEngine:
    """
    Best-effort numbering label generator for common contract numbering (decimal, roman, alpha).
    This is NOT a full Word numbering implementation, but it's good enough for typical Heading/list numbering.
    """

    def __init__(self, z: zipfile.ZipFile) -> None:
        self.num_to_abs: Dict[str, str] = {}
        self.abs_lvls: Dict[str, Dict[int, _NumLvlDef]] = {}
        self.counters: Dict[str, List[int]] = defaultdict(list)
        self._load(z)

    def _load(self, z: zipfile.ZipFile) -> None:
        try:
            xml = z.read("word/numbering.xml")
        except KeyError:
            return
        try:
            root = etree.fromstring(xml)
        except Exception:
            return

        # abstractNum -> levels
        for absn in root.findall(".//w:abstractNum", namespaces=NAMESPACES):
            abs_id = absn.get(_w("abstractNumId")) or absn.get("abstractNumId")
            if abs_id is None:
                continue
            abs_id = str(abs_id)
            lvls: Dict[int, _NumLvlDef] = {}
            for lvl in absn.findall("w:lvl", namespaces=NAMESPACES):
                ilvl = lvl.get(_w("ilvl")) or lvl.get("ilvl")
                if ilvl is None:
                    continue
                try:
                    i = int(str(ilvl))
                except Exception:
                    continue
                numFmt_el = lvl.find("w:numFmt", namespaces=NAMESPACES)
                fmt = (numFmt_el.get(_w("val")) or numFmt_el.get("val") or "decimal") if numFmt_el is not None else "decimal"
                lvlText_el = lvl.find("w:lvlText", namespaces=NAMESPACES)
                lvl_text = (lvlText_el.get(_w("val")) or lvlText_el.get("val") or "") if lvlText_el is not None else ""
                lvls[i] = _NumLvlDef(num_fmt=str(fmt), lvl_text=str(lvl_text))
            self.abs_lvls[abs_id] = lvls

        # numId -> abstractNumId
        for num in root.findall(".//w:num", namespaces=NAMESPACES):
            num_id = num.get(_w("numId")) or num.get("numId")
            if num_id is None:
                continue
            abs_el = num.find("w:abstractNumId", namespaces=NAMESPACES)
            if abs_el is None:
                continue
            abs_id = abs_el.get(_w("val")) or abs_el.get("val")
            if abs_id is None:
                continue
            self.num_to_abs[str(num_id)] = str(abs_id)

    def _p_numpr(self, p: etree._Element) -> Tuple[Optional[str], Optional[int]]:
        numPr = p.find("w:pPr/w:numPr", namespaces=NAMESPACES)
        if numPr is None:
            return None, None
        ilvl_el = numPr.find("w:ilvl", namespaces=NAMESPACES)
        numId_el = numPr.find("w:numId", namespaces=NAMESPACES)
        if ilvl_el is None or numId_el is None:
            return None, None
        num_id = ilvl = None
        v1 = numId_el.get(_w("val")) or numId_el.get("val")
        v2 = ilvl_el.get(_w("val")) or ilvl_el.get("val")
        if v1 is not None:
            num_id = str(v1).strip()
        if v2 is not None:
            try:
                ilvl = int(str(v2).strip())
            except Exception:
                ilvl = None
        return num_id, ilvl

    def next_label(self, p: etree._Element) -> Tuple[Optional[str], Optional[int]]:
        """Advance numbering state for this paragraph; return (label, ilvl) if numbered."""
        num_id, ilvl = self._p_numpr(p)
        if not num_id or ilvl is None or ilvl < 0:
            return None, None

        # increment counters for this num_id/ilvl
        ctrs = self.counters[num_id]
        while len(ctrs) <= ilvl:
            ctrs.append(0)

        ctrs[ilvl] += 1
        # reset deeper levels
        for j in range(ilvl + 1, len(ctrs)):
            ctrs[j] = 0

        # build label using lvlText if possible
        abs_id = self.num_to_abs.get(num_id)
        lvl_defs = self.abs_lvls.get(abs_id or "", {})
        lvl_def = lvl_defs.get(ilvl)

        # If we can't find definitions, fall back to dotted decimals
        if lvl_def is None:
            nums = [c for c in ctrs[: ilvl + 1] if c > 0]
            if not nums:
                return None, ilvl
            return ".".join(str(n) for n in nums), ilvl

        fmt = (lvl_def.num_fmt or "decimal").lower()
        # ignore bullets
        if fmt == "bullet":
            return None, ilvl

        # helper to format each %k
        def fmt_k(k: int) -> str:
            # k is 1-based placeholder index
            idx = k - 1
            if idx < 0 or idx >= len(ctrs):
                return ""
            n = ctrs[idx]
            if n <= 0:
                return ""
            if fmt in ("decimal", "decimalzero"):
                return str(n)
            if fmt in ("upperroman", "roman"):
                return _roman(n)
            if fmt in ("lowerroman"):
                return _roman(n).lower()
            if fmt in ("upperletter", "alpha"):
                return _alpha(n, upper=True)
            if fmt in ("lowerletter"):
                return _alpha(n, upper=False)
            # fallback
            return str(n)

        lvl_text = lvl_def.lvl_text or ""
        if lvl_text:
            out = lvl_text
            # replace %1..%9
            for k in range(1, 10):
                out = out.replace(f"%{k}", fmt_k(k))
            out = out.strip()
            # Some lvlText already includes trailing punctuation
            return out, ilvl

        # fallback if lvlText empty
        nums = [c for c in ctrs[: ilvl + 1] if c > 0]
        if not nums:
            return None, ilvl
        return ".".join(str(n) for n in nums), ilvl


# -----------------------------
# Small utilities
# -----------------------------
def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _assign_hierarchy(clauses: List[Clause]) -> None:
    stack: List[Clause] = []
    for clause in clauses:
        while stack and stack[-1].level >= clause.level:
            stack.pop()
        clause.parent_clause_id = stack[-1].clause_id if stack else None
        stack.append(clause)


def _mk_header_line(label: Optional[str], title: Optional[str], clean_text: str) -> str:
    ct = (clean_text or "").strip()
    if label and title:
        return f"{label} {title}".strip()
    if label and ct:
        # avoid double-prefix if ct already begins with the label
        if ct.lower().startswith(label.strip().lower()):
            return ct
        return f"{label} {ct}".strip()
    return ct


# ======================================================================================
# Public tool: parse_docx2
# ======================================================================================
def parse_docx(input_data: ParseDocxInput) -> DocumentParseResult:
    """
    Parse a .docx file into a structured DocumentParseResult, optimized for:
      - Heading-driven contracts with a Table of Contents up front
      - Auto-numbered headings (best-effort)
      - Tracked changes + comments
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

            # Termset id (CTM-P-ST-xxx) is expected in the footer. Extract the numeric suffix (e.g., 002).
            termset_id = _extract_termset_id_from_docx(z)
            if not termset_id:
                warnings.append("No termset id found in document footer (expected pattern CTM-P-ST-xxx).")

            # Styles outline map (best-effort)
            style_outline = _load_style_outline_levels(z)

            # Numbering engine (best-effort)
            numbering = _NumberingEngine(z)

            # Comments map
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

            # Pre-scan headings on visible, non-TOC paragraphs so we can pick a "primary" heading level.
            # Also compute a safe numeric boundary depth fallback for docs that don't expose outline levels.
            #
            # IMPORTANT FIX:
            #   The prior heuristic chose the *most frequent* outline level, which can be wrong when
            #   subclauses are styled as Heading 2/3 and occur far more often than top-level headings.
            #   For the "one card per main clause" UX, we choose the *shallowest* outline level observed.
            toc_state_scan = _TocState()
            heading_levels: List[int] = []
            numeric_depths: List[int] = []

            for (p, bk, _m) in paragraph_blocks:
                toc_state_scan.update_from_paragraph(p)
                if bk in ("deleted", "moved_from"):
                    continue

                # We need clean text to evaluate literal TOC heading and TOC-ish styles
                seg3 = list(_walk_text_segments_with_meta(p, kind="normal", meta=_RevMeta()))
                clean_text = "".join(
                    t for t, k, _mm in seg3 if k in ("normal", "inserted", "moved_to")
                ).replace("\r", "")

                sid = _pstyle_id(p)
                if _is_toc_paragraph(clean_text, sid, toc_state_scan):
                    continue

                # Prefer Word outline levels / heading styles.
                ol = _paragraph_outline_level(p, style_outline)
                if ol is not None:
                    heading_levels.append(int(ol))

                # Numeric fallback boundaries should only consider paragraphs that are NOT list-numbered.
                # This avoids splitting subclauses that are implemented as lists (w:numPr).
                if not _has_numpr(p):
                    k3, _lbl, _ttl, nd = _detect_label_kind((clean_text or "").strip())
                    if k3 == "numeric" and nd is not None:
                        numeric_depths.append(int(nd))

            # Primary heading level heuristic:
            # For "one card per main clause", treat the SHALLOWEST outline level observed as the boundary.
            primary_heading_level: Optional[int] = min(heading_levels) if heading_levels else None

            # Numeric boundary depth fallback: choose the shallowest numeric depth observed.
            numeric_boundary_depth: Optional[int] = min(numeric_depths) if numeric_depths else None

            clauses: List[Clause] = []
            current_clause: Optional[Clause] = None

            atoms_by_clause: Dict[int, List[_ChangeAtom]] = defaultdict(list)
            change_comments_by_clause: Dict[int, Dict[str, ClauseCommentItem]] = defaultdict(dict)

            toc_state = _TocState()

            for p_index, (p, block_kind, block_meta) in enumerate(paragraph_blocks):
                toc_state.update_from_paragraph(p)

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

                comment_ids = _paragraph_comment_ids(p) if input_data.options.extract_comments else []
                has_signal = bool(clean_text.strip() or deleted_text.strip() or raw_spans or comment_ids)
                if not has_signal:
                    # still advance numbering state for consistency
                    numbering.next_label(p)
                    continue

                style_id = _pstyle_id(p)

                # Skip Table of Contents section entirely
                if _is_toc_paragraph(clean_text, style_id, toc_state):
                    # still advance numbering state for consistency
                    numbering.next_label(p)
                    continue

                # Best-effort outline level (Word-native)
                outline_level = _paragraph_outline_level(p, style_outline)

                # Best-effort numbering label (Word auto-numbering)
                num_label, num_ilvl = numbering.next_label(p)

                # Determine label/title using visible text first (most reliable)
                kind, label, title, num_depth = _detect_label_kind(clean_text.strip())

                # If visible text doesn't contain numeric label but Word numbering exists, use numbering as label.
                # This is especially helpful for auto-numbered headings where the number is not present in w:t.
                if kind is None and num_label and (num_ilvl is not None) and clean_text.strip():
                    # treat as numeric heading-like
                    kind = "numeric"
                    label = num_label.strip()
                    title = clean_text.strip()
                    num_depth = _numeric_depth(label) if label else None

                # For change summaries, we want a best-effort label even for deleted paragraphs.
                label_guess: Optional[str] = None
                visible_for_label = clean_text.strip() if clean_text.strip() else deleted_text.strip()
                k2, l2, _t2, _d2 = _detect_label_kind(visible_for_label)
                if k2 in ("numeric", "article", "section") and l2:
                    label_guess = l2
                elif label:
                    label_guess = label
                elif num_label:
                    label_guess = num_label.strip()

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
                anchors = (
                    _extract_comment_anchors_from_paragraph(p, default_kind)
                    if input_data.options.extract_comments
                    else {}
                )
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

                # Determine if this paragraph starts a new *top-level* clause.
                # Goal: one Clause object per main clause, with subclauses kept inside the clause text.
                #
                # Rules:
                #   1) If outline levels are available, ONLY paragraphs at the primary (shallowest) heading level
                #      start a new clause.
                #      - However, if a paragraph has NO outline level but clearly matches ARTICLE/SECTION,
                #        still allow it to start a clause (common in some templates).
                #   2) If outline levels are missing, fall back to ARTICLE/SECTION or numeric headings at the
                #      shallowest observed numeric depth, but DO NOT split on list-numbered paragraphs (w:numPr).
                is_boundary = False
                is_list_item = _has_numpr(p)

                if block_kind not in ("deleted", "moved_from"):
                    if primary_heading_level is not None:
                        # Heading-driven mode
                        if outline_level is not None and int(outline_level) == int(primary_heading_level):
                            is_boundary = True
                        elif outline_level is None and kind in ("article", "section") and not is_list_item:
                            is_boundary = True
                    else:
                        # Fallback mode
                        if kind in ("article", "section") and not is_list_item:
                            is_boundary = True
                        elif (
                            kind == "numeric"
                            and not is_list_item
                            and numeric_boundary_depth is not None
                            and num_depth is not None
                            and int(num_depth) == int(numeric_boundary_depth)
                        ):
                            is_boundary = True

                # Clause level:
                # Since we intentionally group subclauses into the parent clause, all emitted clauses are level 1.
                level = 1

                # Build the visible header line we store in clause.text for boundary paragraphs.
                header_line = _mk_header_line(label, title, clean_text)

                if is_boundary:
                    if current_clause is not None:
                        clauses.append(current_clause)

                    current_clause = Clause(
                        clause_id="",  # set later
                        label=label,
                        title=title,
                        level=1,
                        parent_clause_id=None,
                        text=header_line.strip(),
                        raw_spans=raw_spans,
                        redlines=redlines,
                        comments=comments,
                        source_locations=SourceLocations(),
                    )

                    atoms_by_clause[id(current_clause)].extend(change_atoms)
                    for _cid, item in change_comment_items.items():
                        _merge_comment_item(change_comments_by_clause[id(current_clause)], item)

                else:
                    if current_clause is None:
                        # Preamble before first clause becomes one clause
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

                    atoms_by_clause[id(current_clause)].extend(change_atoms)
                    for _cid, item in change_comment_items.items():
                        _merge_comment_item(change_comments_by_clause[id(current_clause)], item)

            if current_clause is not None:
                clauses.append(current_clause)

            if not clauses:
                meta = DocumentMetadata(
                    filename=os.path.basename(file_path),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    pages=None,
                    word_count=0,
                    termset_id=termset_id,
                )
                warnings.append("No clauses detected (document may be empty or fully skipped as TOC).")
                return DocumentParseResult(document=meta, clauses=[], warnings=warnings)

            # Populate clause.changes after clause text is finalized
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

            # Stable IDs
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
                termset_id=termset_id,
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

            # Helpful diagnostic: if we skipped a TOC field, mention it
            # (Not strictly necessary, but useful when debugging doc formats)
            if toc_state.is_toc or toc_state.active:
                warnings.append("Detected TOC field/styles; TOC content was skipped during parsing.")

            return DocumentParseResult(document=meta, clauses=clauses, warnings=warnings)

    except zipfile.BadZipFile as e:
        raise ValueError("Invalid DOCX: not a valid zip archive") from e
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass