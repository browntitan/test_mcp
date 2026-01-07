export type RawSpanKind = 'normal' | 'inserted' | 'deleted' | 'strike' | 'comment_ref' | 'moved_from' | 'moved_to';

export interface RawSpan {
  text: string;
  kind: RawSpanKind;
}

export interface RedlineItem {
  text: string;
  context?: string | null;
}

export interface Redlines {
  insertions: RedlineItem[];
  deletions: RedlineItem[];
  strikethroughs: RedlineItem[];
}

export interface Comment {
  author?: string | null;
  date?: string | null;
  text: string;
  context?: string | null;
}

export interface ClauseChangeItem {
  text: string;
  label?: string | null;
  author?: string | null;
  date?: string | null;
  revision_id?: string | null;
}

export interface ClauseModificationItem {
  from_text: string;
  to_text: string;
  label?: string | null;
  author?: string | null;
  date?: string | null;
  revision_id?: string | null;
}

export interface ClauseCommentItem {
  comment_id?: string | null;
  author?: string | null;
  date?: string | null;
  text: string;
  anchor_text?: string | null;
}

export interface ClauseChanges {
  added: ClauseChangeItem[];
  deleted: ClauseChangeItem[];
  modified: ClauseModificationItem[];
  comments: ClauseCommentItem[];
}

export interface SourceLocations {
  page_start?: number | null;
  page_end?: number | null;
}

export interface Clause {
  clause_id: string;
  label?: string | null;
  title?: string | null;
  level: number;
  parent_clause_id?: string | null;
  text: string;
  raw_spans: RawSpan[];
  redlines: Redlines;
  comments: Comment[];
  changes: ClauseChanges;
  source_locations: SourceLocations;
}

export type MediaType =
  | 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  | 'application/pdf';

export interface DocumentMetadata {
  filename: string;
  media_type: MediaType;
  pages?: number | null;
  word_count?: number | null;
}

export interface DocumentParseResult {
  document: DocumentMetadata;
  clauses: Clause[];
  warnings: string[];
}

export interface ClauseBoundary {
  start_char: number;
  end_char: number;
  level: number;
  label?: string | null;
  title?: string | null;
}

export interface NormalizedClause extends Clause {
  original_clause_id?: string | null;
  was_merged: boolean;
  was_split: boolean;
}

export interface ClauseListNormalized {
  document: DocumentMetadata;
  clauses: NormalizedClause[];
  normalization_warnings: string[];
}

export type StoredDocument =
  | {
      kind: 'parsed';
      docId: string;
      createdAt: number;
      data: DocumentParseResult;
    }
  | {
      kind: 'normalized';
      docId: string;
      createdAt: number;
      data: ClauseListNormalized;
      sourceDocId?: string;
    };

export function isDocumentParseResult(x: unknown): x is DocumentParseResult {
  if (!x || typeof x !== 'object') return false;
  const obj = x as any;
  return (
    typeof obj?.document?.filename === 'string' &&
    Array.isArray(obj?.clauses) &&
    Array.isArray(obj?.warnings)
  );
}

export function isClauseListNormalized(x: unknown): x is ClauseListNormalized {
  if (!x || typeof x !== 'object') return false;
  const obj = x as any;
  return (
    typeof obj?.document?.filename === 'string' &&
    Array.isArray(obj?.clauses) &&
    Array.isArray(obj?.normalization_warnings)
  );
}
