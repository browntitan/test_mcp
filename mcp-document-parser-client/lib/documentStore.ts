import type { ClauseListNormalized, DocumentParseResult, StoredDocument } from './types';

const TTL_MS = 1000 * 60 * 60; // 1 hour

// In Next.js dev, route handlers can be reloaded (HMR) which would normally
// reset module-level state. Keep the store on globalThis so parsed docs are
// still available across hot reloads.
type _DocStore = Map<string, StoredDocument>;
const _g = globalThis as unknown as { __mcpDocStore__?: _DocStore };
const store: _DocStore = _g.__mcpDocStore__ ?? (_g.__mcpDocStore__ = new Map());

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

export function getDocument(docId: string): StoredDocument | undefined {
  cleanup();
  return store.get(docId);
}

export function deleteDocument(docId: string): boolean {
  return store.delete(docId);
}

export function listDocumentIds(): string[] {
  cleanup();
  return [...store.keys()];
}
