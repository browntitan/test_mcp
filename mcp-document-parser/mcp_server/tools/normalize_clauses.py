from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    ClauseBoundary,
    ClauseListNormalized,
    DocumentParseResult,
    NormalizedClause,
    NormalizeClausesInput,
    Redlines,
    SourceLocations,
    normalize_text_for_hash,
    stable_clause_id,
)


@dataclass
class _ClauseRange:
    start: int
    end: int
    clause: NormalizedClause


def normalize_clauses(input_data: NormalizeClausesInput) -> ClauseListNormalized:
    """Normalize clauses by applying optional boundaries, fixing hierarchy and deduplicating."""

    parse_result: DocumentParseResult = input_data.parse_result
    normalization_warnings: List[str] = []

    # Start from existing clauses
    base_clauses: List[NormalizedClause] = [
        NormalizedClause(**c.model_dump())
        for c in parse_result.clauses
    ]

    if input_data.boundaries:
        # Build global text for mapping
        full_text, ranges = _build_global_text_and_ranges(base_clauses)

        boundaries = sorted(input_data.boundaries, key=lambda b: (b.start_char, b.end_char))
        new_clauses: List[NormalizedClause] = []

        # Count how many times each original clause contributes
        contrib_counts: Dict[str, int] = {}

        for b in boundaries:
            start = max(0, min(len(full_text), b.start_char))
            end = max(0, min(len(full_text), b.end_char))
            if end <= start:
                normalization_warnings.append(f"Ignored empty boundary range {b.start_char}-{b.end_char}.")
                continue

            snippet = full_text[start:end].strip()
            if not snippet:
                normalization_warnings.append(f"Ignored whitespace-only boundary range {b.start_char}-{b.end_char}.")
                continue

            contributors = _contributors_for_range(ranges, start, end)
            if not contributors:
                normalization_warnings.append(
                    f"Boundary range {b.start_char}-{b.end_char} did not map to any existing clauses; created clause without metadata."
                )

            merged = _merge_contributors(contributors)

            label = b.label or merged.label
            title = b.title or merged.title

            nc = NormalizedClause(
                clause_id="",  # set later
                label=label,
                title=title,
                level=max(1, int(b.level)),
                parent_clause_id=None,
                text=snippet,
                raw_spans=merged.raw_spans,
                redlines=merged.redlines,
                comments=merged.comments,
                source_locations=merged.source_locations,
                original_clause_id=merged.original_clause_id,
                was_merged=len(contributors) > 1,
                was_split=False,  # set later
            )

            new_clauses.append(nc)

            # Track split counts only when original_clause_id is meaningful
            for c in contributors:
                if c.original_clause_id:
                    contrib_counts[c.original_clause_id] = contrib_counts.get(c.original_clause_id, 0) + 1

        # Mark splits
        split_ids = {cid for cid, n in contrib_counts.items() if n > 1}
        for nc in new_clauses:
            if nc.original_clause_id in split_ids:
                nc.was_split = True

        clauses = new_clauses
    else:
        clauses = base_clauses
        for c in clauses:
            # Preserve traceability: normalized clause originates from itself initially.
            c.original_clause_id = c.clause_id

    # Deduplicate clauses by normalized text
    clauses, dedupe_count = _dedupe_clauses(clauses)
    if dedupe_count:
        normalization_warnings.append(f"Deduplicated {dedupe_count} duplicate clause(s) by normalized text hash.")

    # Re-assign stable IDs
    for idx, c in enumerate(clauses, start=1):
        c.clause_id = stable_clause_id(idx, c.text)

    # Fix hierarchy
    _assign_hierarchy(clauses)

    return ClauseListNormalized(
        document=parse_result.document,
        clauses=clauses,
        normalization_warnings=normalization_warnings,
    )


def _build_global_text_and_ranges(clauses: List[NormalizedClause]) -> Tuple[str, List[_ClauseRange]]:
    """
    Return (full_text, ranges) where ranges map global offsets to clauses.

    Update: include the newline separator in the range mapping to reduce
    spurious "did not map to any existing clauses" warnings when boundaries
    land on separators.
    """
    parts: List[str] = []
    ranges: List[_ClauseRange] = []
    pos = 0

    for c in clauses:
        text = c.text or ""
        parts.append(text)
        start = pos
        end = pos + len(text)

        # Include the separator newline in the range for more intuitive mapping.
        end_with_sep = end + 1
        ranges.append(_ClauseRange(start=start, end=end_with_sep, clause=c))

        pos = end
        parts.append("\n")
        pos += 1

    full_text = "".join(parts)
    return full_text, ranges


def _contributors_for_range(ranges: List[_ClauseRange], start: int, end: int) -> List[NormalizedClause]:
    contributors: List[NormalizedClause] = []
    for r in ranges:
        if r.end <= start:
            continue
        if r.start >= end:
            break
        # overlap
        if r.start < end and r.end > start:
            contributors.append(r.clause)
    return contributors


def _merge_contributors(contributors: List[NormalizedClause]) -> NormalizedClause:
    """
    Merge contributor metadata (raw_spans, redlines, comments, source_locations).

    Update: original_clause_id handling
    - If exactly one contributor: preserve its original_clause_id (or fallback to its clause_id).
    - If multiple contributors: set original_clause_id = None (ambiguous provenance).
    """
    if not contributors:
        return NormalizedClause(
            clause_id="",
            label=None,
            title=None,
            level=1,
            parent_clause_id=None,
            text="",
            raw_spans=[],
            redlines=Redlines(),
            comments=[],
            source_locations=SourceLocations(),
            original_clause_id=None,
            was_merged=False,
            was_split=False,
        )

    merged = NormalizedClause(**contributors[0].model_dump())
    merged.was_merged = len(contributors) > 1

    if len(contributors) == 1:
        merged.original_clause_id = contributors[0].original_clause_id or contributors[0].clause_id
    else:
        merged.original_clause_id = None

    for c in contributors[1:]:
        merged.raw_spans.extend(c.raw_spans)
        merged.redlines.insertions.extend(c.redlines.insertions)
        merged.redlines.deletions.extend(c.redlines.deletions)
        merged.redlines.strikethroughs.extend(c.redlines.strikethroughs)
        merged.comments.extend(c.comments)

        merged.source_locations.page_start = _min_opt(merged.source_locations.page_start, c.source_locations.page_start)
        merged.source_locations.page_end = _max_opt(merged.source_locations.page_end, c.source_locations.page_end)

    # Ensure source_locations coherence
    if merged.source_locations.page_start is not None and merged.source_locations.page_end is None:
        merged.source_locations.page_end = merged.source_locations.page_start
    if merged.source_locations.page_end is not None and merged.source_locations.page_start is None:
        merged.source_locations.page_start = merged.source_locations.page_end

    return merged


def _min_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _max_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _dedupe_clauses(clauses: List[NormalizedClause]) -> Tuple[List[NormalizedClause], int]:
    seen: Dict[str, NormalizedClause] = {}
    out: List[NormalizedClause] = []
    deduped = 0

    for c in clauses:
        key = normalize_text_for_hash(c.text)
        if not key:
            out.append(c)
            continue
        if key not in seen:
            seen[key] = c
            out.append(c)
            continue

        # merge into existing
        kept = seen[key]
        kept.was_merged = True
        kept.raw_spans.extend(c.raw_spans)
        kept.redlines.insertions.extend(c.redlines.insertions)
        kept.redlines.deletions.extend(c.redlines.deletions)
        kept.redlines.strikethroughs.extend(c.redlines.strikethroughs)
        kept.comments.extend(c.comments)
        kept.source_locations.page_start = _min_opt(kept.source_locations.page_start, c.source_locations.page_start)
        kept.source_locations.page_end = _max_opt(kept.source_locations.page_end, c.source_locations.page_end)
        deduped += 1

    return out, deduped


def _assign_hierarchy(clauses: List[NormalizedClause]) -> None:
    stack: List[NormalizedClause] = []
    for clause in clauses:
        while stack and stack[-1].level >= clause.level:
            stack.pop()
        clause.parent_clause_id = stack[-1].clause_id if stack else None
        stack.append(clause)