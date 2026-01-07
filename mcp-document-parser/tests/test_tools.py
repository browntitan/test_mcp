from __future__ import annotations

import base64
import os
import tempfile
import zipfile

import fitz
import pytest

from mcp_server.schemas import (
    ClauseBoundary,
    DocumentParseResult,
    NormalizeClausesInput,
    ParseDocxInput,
    ParsePdfInput,
)
from mcp_server.tools.normalize_clauses import normalize_clauses
from mcp_server.tools.parse_docx import parse_docx
from mcp_server.tools.parse_pdf import parse_pdf


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _minimal_content_types_xml() -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/word/comments.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml\"/>"
        "</Types>"
    )


def _build_docx(document_xml: str, comments_xml: str | None = None) -> str:
    fd, path = tempfile.mkstemp(suffix=".docx")
    os.close(fd)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _minimal_content_types_xml())
        z.writestr("word/document.xml", document_xml)
        if comments_xml is not None:
            z.writestr("word/comments.xml", comments_xml)

    return path


def _docx_xml_with_tracked_changes_and_comment() -> tuple[str, str]:
    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:document xmlns:w=\"{_W_NS}\">
  <w:body>
    <w:p>
      <w:r><w:t>1. This is </w:t></w:r>
      <w:ins><w:r><w:t>inserted</w:t></w:r></w:ins>
      <w:r><w:t> text.</w:t></w:r>
      <w:del><w:r><w:delText>deleted</w:delText></w:r></w:del>
      <w:commentRangeStart w:id=\"0\"/>
      <w:r><w:t> Commented</w:t></w:r>
      <w:commentRangeEnd w:id=\"0\"/>
      <w:r><w:commentReference w:id=\"0\"/></w:r>
    </w:p>
  </w:body>
</w:document>
"""

    comments_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:comments xmlns:w=\"{_W_NS}\">
  <w:comment w:id=\"0\" w:author=\"Alice\" w:date=\"2026-01-01T00:00:00Z\">
    <w:p><w:r><w:t>Test comment</w:t></w:r></w:p>
  </w:comment>
</w:comments>
"""

    return document_xml, comments_xml


def test_parse_docx_tracked_changes_and_comments():
    document_xml, comments_xml = _docx_xml_with_tracked_changes_and_comment()
    path = _build_docx(document_xml, comments_xml)

    try:
        result = parse_docx(ParseDocxInput(file_path=path))
        assert isinstance(result, DocumentParseResult)
        assert result.document.media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert len(result.clauses) == 1

        clause = result.clauses[0]
        assert "inserted" in clause.text
        assert "deleted" not in clause.text  # deletions excluded from clean text

        assert any(i.text == "inserted" for i in clause.redlines.insertions)
        assert any(d.text == "deleted" for d in clause.redlines.deletions)

        assert len(clause.comments) == 1
        assert clause.comments[0].text == "Test comment"
        assert clause.comments[0].author == "Alice"

        # raw spans should include inserted and deleted kinds
        kinds = {s.kind for s in clause.raw_spans}
        assert "inserted" in kinds
        assert "deleted" in kinds
        assert "comment_ref" in kinds

    finally:
        os.remove(path)


def test_docx_clause_label_detection_and_hierarchy():
    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:document xmlns:w=\"{_W_NS}\">
  <w:body>
    <w:p><w:r><w:t>ARTICLE I General Provisions</w:t></w:r></w:p>
    <w:p><w:r><w:t>1. First Section</w:t></w:r></w:p>
    <w:p><w:r><w:t>1.1 Subsection</w:t></w:r></w:p>
    <w:p><w:r><w:t>(a) Alpha clause</w:t></w:r></w:p>
    <w:p><w:r><w:t>(i) Roman clause</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

    path = _build_docx(document_xml, comments_xml=None)
    try:
        result = parse_docx(ParseDocxInput(file_path=path))
        labels = [c.label for c in result.clauses]
        assert labels[0].upper().startswith("ARTICLE")
        assert labels[1] == "1."
        assert labels[2] == "1.1"
        assert labels[3].lower() == "(a)"
        assert labels[4].lower() == "(i)"

        levels = [c.level for c in result.clauses]
        # Article is top-level
        assert levels[0] == 1
        # Numeric and nested clauses are deeper than article
        assert levels[1] > levels[0]

        # Hierarchy checks
        article_id = result.clauses[0].clause_id
        first_id = result.clauses[1].clause_id
        alpha_id = result.clauses[3].clause_id

        assert result.clauses[1].parent_clause_id == article_id
        assert result.clauses[2].parent_clause_id == first_id
        # (i) should be under (a)
        assert result.clauses[4].parent_clause_id == alpha_id

    finally:
        os.remove(path)


def test_parse_pdf_with_annotations_and_strikeout():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "1. Clause one\nStrike me\n")

        # Strikeout 'Strike'
        strike_rect = page.search_for("Strike")[0]
        strike_annot = page.add_strikeout_annot(strike_rect)
        strike_annot.set_info(content="strikeout comment", title="Bob")
        strike_annot.update()

        # Sticky note
        page.add_text_annot((72, 120), "Sticky note")

        # Highlight with content
        hl_rect = page.search_for("Clause one")[0]
        hl = page.add_highlight_annot(hl_rect)
        hl.set_info(content="highlight comment", title="Alice")
        hl.update()

        doc.save(path)
        doc.close()

        result = parse_pdf(ParsePdfInput(file_path=path))
        assert result.document.media_type == "application/pdf"
        assert result.document.pages == 1
        assert len(result.clauses) >= 1

        c0 = result.clauses[0]
        assert c0.label == "1."

        # Comments from sticky + highlight
        comment_texts = [c.text for c in c0.comments]
        assert "Sticky note" in comment_texts
        assert "highlight comment" in comment_texts

        # Strikethrough captured
        assert len(c0.redlines.strikethroughs) >= 1
        assert any("Strike" in r.text for r in c0.redlines.strikethroughs)

    finally:
        os.remove(path)


def test_normalize_clauses_without_boundaries():
    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:document xmlns:w=\"{_W_NS}\">
  <w:body>
    <w:p><w:r><w:t>1. First clause</w:t></w:r></w:p>
    <w:p><w:r><w:t>2. Second clause</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

    path = _build_docx(document_xml, comments_xml=None)
    try:
        parse_result = parse_docx(ParseDocxInput(file_path=path))
        norm = normalize_clauses(NormalizeClausesInput(parse_result=parse_result))

        assert len(norm.clauses) == len(parse_result.clauses)
        assert all(c.original_clause_id is not None for c in norm.clauses)

    finally:
        os.remove(path)


def test_normalize_clauses_with_boundaries_merge_all():
    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:document xmlns:w=\"{_W_NS}\">
  <w:body>
    <w:p><w:r><w:t>1. First clause</w:t></w:r></w:p>
    <w:p><w:r><w:t>2. Second clause</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

    path = _build_docx(document_xml, comments_xml=None)
    try:
        parse_result = parse_docx(ParseDocxInput(file_path=path))
        full_text = "\n".join([c.text for c in parse_result.clauses])
        boundaries = [ClauseBoundary(start_char=0, end_char=len(full_text), level=1, label="1-2")]

        norm = normalize_clauses(NormalizeClausesInput(parse_result=parse_result, boundaries=boundaries))
        assert len(norm.clauses) == 1
        assert norm.clauses[0].was_merged is True
        assert "First clause" in norm.clauses[0].text
        assert "Second clause" in norm.clauses[0].text

    finally:
        os.remove(path)


def test_error_handling_file_not_found_and_invalid_docx():
    with pytest.raises(FileNotFoundError):
        parse_docx(ParseDocxInput(file_path="/no/such/file.docx"))

    # invalid DOCX (not a zip)
    fd, path = tempfile.mkstemp(suffix=".docx")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(b"not a zip")

        with pytest.raises(ValueError):
            parse_docx(ParseDocxInput(file_path=path))
    finally:
        os.remove(path)


def test_base64_input_for_docx_and_pdf():
    # DOCX base64
    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<w:document xmlns:w=\"{_W_NS}\"><w:body>
  <w:p><w:r><w:t>1. Hello</w:t></w:r></w:p>
</w:body></w:document>"""
    path = _build_docx(document_xml, comments_xml=None)
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        res = parse_docx(ParseDocxInput(file_base64=b64))
        assert len(res.clauses) == 1
        assert "Hello" in res.clauses[0].text
    finally:
        os.remove(path)

    # PDF base64
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "1. Hello PDF")
        doc.save(pdf_path)
        doc.close()

        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        res = parse_pdf(ParsePdfInput(file_base64=b64))
        assert len(res.clauses) >= 1
        assert "Hello PDF" in res.clauses[0].text
    finally:
        os.remove(pdf_path)
