import type { ClauseListNormalized, DocumentParseResult, StoredDocument } from './types';

const TTL_MS = 1000 * 60 * 60; // 1 hour

// In Next.js dev, route handlers can be reloaded (HMR) which would normally
// reset module-level state. Keep the store on globalThis so parsed docs are
// still available across hot reloads.
type _DocStore = Map<string, StoredDocument>;
const _g = globalThis as unknown as { __mcpDocStore__?: _DocStore };
const store: _DocStore = _g.__mcpDocStore__ ?? (_g.__mcpDocStore__ = new Map());

// Separate store for raw uploaded files (so /api/chat can start risk_assessment with file_base64
// when a parse_result is not available yet).
export type StoredUpload = {
  kind: 'upload';
  docId: string;
  createdAt: number;
  filename: string;
  mediaType?: string;
  fileType?: 'docx' | 'pdf';
  fileBase64: string;
};

type _UploadStore = Map<string, StoredUpload>;
const _g2 = globalThis as unknown as { __mcpUploadStore__?: _UploadStore };
const uploadStore: _UploadStore = _g2.__mcpUploadStore__ ?? (_g2.__mcpUploadStore__ = new Map());

let lastCleanupAt = 0;

function now() {
  return Date.now();
}

function cleanup() {
  const t = now();

  // Throttle cleanup to at most once per minute.
  if (t - lastCleanupAt < 60_000) return;
  lastCleanupAt = t;

  for (const [id, doc] of store.entries()) {
    if (t - doc.createdAt > TTL_MS) {
      store.delete(id);
    }
  }

  for (const [id, up] of uploadStore.entries()) {
    if (t - up.createdAt > TTL_MS) {
      uploadStore.delete(id);
    }
  }
}

export function putParsed(docId: string, data: DocumentParseResult): void {
  cleanup();
  store.set(docId, {
    kind: 'parsed',
    docId,
    createdAt: now(),
    data,
  });
}

export function putNormalized(
  docId: string,
  data: ClauseListNormalized,
  sourceDocId?: string,
): void {
  cleanup();
  store.set(docId, {
    kind: 'normalized',
    docId,
    createdAt: now(),
    data,
    sourceDocId,
  });
}

export function putUpload(docId: string, upload: Omit<StoredUpload, 'kind' | 'docId' | 'createdAt'>): void {
  cleanup();
  uploadStore.set(docId, {
    kind: 'upload',
    docId,
    createdAt: now(),
    filename: upload.filename,
    mediaType: upload.mediaType,
    fileType: upload.fileType,
    fileBase64: upload.fileBase64,
  });
}

export function getUpload(docId: string): StoredUpload | undefined {
  cleanup();
  return uploadStore.get(docId);
}

export function deleteUpload(docId: string): boolean {
  return uploadStore.delete(docId);
}

export function listUploadIds(): string[] {
  cleanup();
  return [...uploadStore.keys()];
}

export function getDocument(docId: string): StoredDocument | undefined {
  cleanup();
  return store.get(docId);
}

export function deleteDocument(docId: string): boolean {
  const a = store.delete(docId);
  const b = uploadStore.delete(docId);
  return a || b;
}

export function listDocumentIds(): string[] {
  cleanup();
  const ids = new Set<string>();
  for (const k of store.keys()) ids.add(k);
  for (const k of uploadStore.keys()) ids.add(k);
  return [...ids];
}
