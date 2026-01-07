'use client';

import { useMemo, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import type { Clause, DocumentParseResult } from '../lib/types';

type ParseOptions =
  | {
      // DOCX options
      extract_tracked_changes?: boolean;
      extract_comments?: boolean;
      include_raw_spans?: boolean;
    }
  | {
      // PDF options
      extract_annotations?: boolean;
      include_raw_spans?: boolean;
    };

function inferIsPdf(filename: string, mime?: string) {
  if (mime?.toLowerCase() === 'application/pdf') return true;
  return filename.toLowerCase().endsWith('.pdf');
}

function safeJsonPreview(value: any, maxChars = 2000): string {
  try {
    const s = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    if (!s) return '';
    if (s.length <= maxChars) return s;
    return s.slice(0, maxChars - 1) + '…';
  } catch {
    return String(value);
  }
}

function renderMessageParts(m: any) {
  const parts: any[] = Array.isArray(m?.parts) ? m.parts : [];
  const out: any[] = [];

  if (parts.length === 0) {
    const content = typeof m?.content === 'string' ? m.content : '';
    if (content && content.trim()) {
      return <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>;
    }
    return <div className="subtle">(no assistant text – likely tool-only step)</div>;
  }

  for (let idx = 0; idx < parts.length; idx++) {
    const p = parts[idx];

    if (p?.type === 'text') {
      const txt = typeof p.text === 'string' ? p.text : '';
      if (!txt) continue;
      out.push(
        <div key={idx} style={{ whiteSpace: 'pre-wrap' }}>
          {txt}
        </div>,
      );
      continue;
    }
      // Hide step bookkeeping parts (AI SDK emits these during multi-step runs)
      if (p?.type === 'step-start' || p?.type === 'step-finish' || p?.type === 'step') {
        continue;
      }
  
      // Some transports emit tool parts with type names like "tool-run_risk_assessment".
      // Render them as an expandable JSON blob so we can debug tool outputs.
      if (typeof p?.type === 'string' && (p.type.startsWith('tool-') || p.type.startsWith('tool_'))) {
        const toolName = p.type.replace(/^tool[-_]/, '');
        out.push(
          <details key={idx} style={{ marginTop: 6 }} open>
            <summary className="subtle">tool: {toolName}</summary>
            <div className="pre" style={{ marginTop: 6 }}>
              {safeJsonPreview(p)}
            </div>
          </details>,
        );
        continue;
      }

    if (p?.type === 'tool-invocation') {
      const inv: any = p.toolInvocation;
      const name = inv?.toolName || inv?.name || 'tool';
      const state = inv?.state || (inv?.result ? 'result' : 'called');

      out.push(
        <div key={idx} className="toolLine mono">
          ↳ tool: {name} ({state})
        </div>,
      );

      if (inv?.args) {
        out.push(
          <details key={`${idx}-args`} style={{ marginTop: 6 }}>
            <summary className="subtle">args</summary>
            <div className="pre" style={{ marginTop: 6 }}>
              {safeJsonPreview(inv.args)}
            </div>
          </details>,
        );
      }

      if (inv?.result) {
        out.push(
          <details key={`${idx}-result`} style={{ marginTop: 6 }} open>
            <summary className="subtle">result</summary>
            <div className="pre" style={{ marginTop: 6 }}>
              {safeJsonPreview(inv.result)}
            </div>
          </details>,
        );
      }

      continue;
    }

    out.push(
      <details key={idx} style={{ marginTop: 6 }}>
        <summary className="subtle">
          (unrendered part: {String(p?.type ?? 'unknown')})
        </summary>
        <div className="pre" style={{ marginTop: 6 }}>
          {safeJsonPreview(p)}
        </div>
      </details>,
    );
  }

  if (out.length === 0) {
    const content = typeof m?.content === 'string' ? m.content : '';
    if (content && content.trim()) {
      return <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>;
    }
    return <div className="subtle">(no assistant text)</div>;
  }

  return out;
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is a data URL: data:...;base64,XXXX
      const comma = result.indexOf(',');
      if (comma === -1) return reject(new Error('Unexpected file reader result'));
      resolve(result.slice(comma + 1));
    };
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

function clauseHeading(c: Clause): string {
  const label = c.label?.trim();
  const title = c.title?.trim();
  if (label && title) return `${label} — ${title}`;
  if (label) return label;
  if (title) return title;
  return c.clause_id;
}

function preview(text: string, max = 180): string {
  const t = text.replace(/\s+/g, ' ').trim();
  if (t.length <= max) return t;
  return t.slice(0, max - 1) + '…';
}

function formatChangesPlainTextForUI(c: Clause): string {
  // Prefer the new structured `changes` field.
  const ch = (c as any).changes as any;
  const hasStructured =
    ch &&
    ((Array.isArray(ch.added) && ch.added.length > 0) ||
      (Array.isArray(ch.deleted) && ch.deleted.length > 0) ||
      (Array.isArray(ch.modified) && ch.modified.length > 0) ||
      (Array.isArray(ch.comments) && ch.comments.length > 0));

  // Back-compat fallback if `changes` isn't present.
  const ins = c.redlines?.insertions ?? [];
  const del = c.redlines?.deletions ?? [];
  const comms = c.comments ?? [];
  const hasFallback = ins.length > 0 || del.length > 0 || comms.length > 0;

  if (!hasStructured && !hasFallback) return '';

  const lines: string[] = [];
  lines.push('Changes:');

  if (hasStructured) {
    const added = Array.isArray(ch.added) ? ch.added : [];
    const deleted = Array.isArray(ch.deleted) ? ch.deleted : [];
    const modified = Array.isArray(ch.modified) ? ch.modified : [];
    const comments = Array.isArray(ch.comments) ? ch.comments : [];

    if (added.length) {
      lines.push('Added:');
      for (const a of added) {
        const label = a?.label ? String(a.label) : '';
        const txt = (a?.text ?? '').toString().trim();
        if (!txt) continue;
        lines.push(`- ${label ? label + ' ' : ''}${txt}`.trim());
      }
      lines.push('');
    }

    if (deleted.length) {
      lines.push('Deleted:');
      for (const d of deleted) {
        const label = d?.label ? String(d.label) : '';
        const txt = (d?.text ?? '').toString().trim();
        if (!txt) continue;
        lines.push(`- ${label ? label + ' ' : ''}${txt}`.trim());
      }
      lines.push('');
    }

    if (modified.length) {
      lines.push('Modified:');
      for (const m of modified) {
        const label = m?.label ? String(m.label) : '';
        const from = (m?.from_text ?? '').toString().trim();
        const to = (m?.to_text ?? '').toString().trim();
        if (!from && !to) continue;
        lines.push(`- ${label ? label + ' ' : ''}from: ${from} | to: ${to}`.trim());
      }
      lines.push('');
    }

    if (comments.length) {
      lines.push('Comments:');
      for (const cm of comments) {
        const author = cm?.author ? String(cm.author) : 'Reviewer';
        const anchor = cm?.anchor_text
          ? String(cm.anchor_text).replace(/\s+/g, ' ').trim()
          : '';
        const txt = (cm?.text ?? '').toString().trim();
        if (!txt) continue;
        if (anchor) {
          lines.push(`- ${author} (anchor: "${anchor}"): ${txt}`);
        } else {
          lines.push(`- ${author}: ${txt}`);
        }
      }
      lines.push('');
    }

    return lines.join('\n').trimEnd();
  }

  // Fallback formatting.
  if (ins.length) {
    lines.push('Added:');
    for (const a of ins) {
      const txt = (a?.text ?? '').toString().trim();
      if (!txt) continue;
      lines.push(`- ${txt}`);
    }
    lines.push('');
  }

  if (del.length) {
    lines.push('Deleted:');
    for (const d of del) {
      const txt = (d?.text ?? '').toString().trim();
      if (!txt) continue;
      lines.push(`- ${txt}`);
    }
    lines.push('');
  }

  if (comms.length) {
    lines.push('Comments:');
    for (const cm of comms) {
      const author = cm.author || 'Reviewer';
      const txt = (cm.text ?? '').toString().trim();
      if (!txt) continue;
      lines.push(`- ${author}: ${txt}`);
    }
    lines.push('');
  }

  return lines.join('\n').trimEnd();
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [docId, setDocId] = useState<string | null>(null);
  const [parseResult, setParseResult] = useState<DocumentParseResult | null>(null);
  const [selectedClauseId, setSelectedClauseId] = useState<string | null>(null);

  const [parseLoading, setParseLoading] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

  const [optTrackedChanges, setOptTrackedChanges] = useState(true);
  const [optComments, setOptComments] = useState(true);
  const [optAnnotations, setOptAnnotations] = useState(true);
  const [optRawSpans, setOptRawSpans] = useState(true);

  const { messages, sendMessage, status, error, stop, regenerate } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),
  });

  const selectedClause = useMemo(() => {
    if (!parseResult || !selectedClauseId) return null;
    return parseResult.clauses.find(c => c.clause_id === selectedClauseId) || null;
  }, [parseResult, selectedClauseId]);

  const isPdf = useMemo(() => {
    if (!file) return false;
    return inferIsPdf(file.name, file.type);
  }, [file]);

  async function onParse() {
    if (!file) return;

    setParseError(null);
    setParseLoading(true);

    try {
      const file_base64 = await readFileAsBase64(file);

      const options: ParseOptions = isPdf
        ? {
            extract_annotations: optAnnotations,
            include_raw_spans: optRawSpans,
          }
        : {
            extract_tracked_changes: optTrackedChanges,
            extract_comments: optComments,
            include_raw_spans: optRawSpans,
          };

      const res = await fetch('/api/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          file_base64,
          options,
        }),
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(txt || `Parse failed (${res.status})`);
      }

      const data = (await res.json()) as { docId: string; parseResult: DocumentParseResult };

      setDocId(data.docId);
      setParseResult(data.parseResult);
      setSelectedClauseId(data.parseResult.clauses[0]?.clause_id ?? null);
    } catch (e: any) {
      setParseError(e?.message || String(e));
      setDocId(null);
      setParseResult(null);
      setSelectedClauseId(null);
    } finally {
      setParseLoading(false);
    }
  }

  async function onSend(text: string) {
    if (!text.trim()) return;
    if (!docId) {
      alert('Parse a document first (left panel).');
      return;
    }

    await sendMessage(
      { text },
      {
        body: {
          docId,
          focusClauseId: selectedClauseId,
        },
      },
    );
  }

  return (
    <div className="grid">
      {/* LEFT: Document */}
      <section className="panel">
        <div className="panelHeader">
          <div>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Document</div>
            <div className="subtle">Upload and parse a DOCX or PDF.</div>
          </div>
          <div className="row">
            <span className="pill">
              MCP Server: <span className="mono">:8765</span>
            </span>
          </div>
        </div>

        <div className="panelBody">
          <div className="row">
            <input
              className="input"
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={e => {
                const f = e.target.files?.[0] || null;
                setFile(f);
                setDocId(null);
                setParseResult(null);
                setSelectedClauseId(null);
                setParseError(null);
              }}
            />

            <button
              className="btn btnPrimary"
              onClick={onParse}
              disabled={!file || parseLoading}
            >
              {parseLoading ? 'Parsing…' : 'Parse'}
            </button>

            {docId ? (
              <span className="pill ok">
                Parsed <span className="mono">{docId.slice(0, 8)}</span>
              </span>
            ) : (
              <span className="pill subtle">No document loaded</span>
            )}
          </div>

          <div style={{ marginTop: 10 }} className="row">
            {!isPdf ? (
              <>
                <label className="pill">
                  <input
                    type="checkbox"
                    checked={optTrackedChanges}
                    onChange={e => setOptTrackedChanges(e.target.checked)}
                  />
                  Tracked changes
                </label>
                <label className="pill">
                  <input
                    type="checkbox"
                    checked={optComments}
                    onChange={e => setOptComments(e.target.checked)}
                  />
                  Comments
                </label>
              </>
            ) : (
              <label className="pill">
                <input
                  type="checkbox"
                  checked={optAnnotations}
                  onChange={e => setOptAnnotations(e.target.checked)}
                />
                PDF annotations
              </label>
            )}

            <label className="pill">
              <input
                type="checkbox"
                checked={optRawSpans}
                onChange={e => setOptRawSpans(e.target.checked)}
              />
              Raw spans
            </label>
          </div>

          {parseError ? (
            <div style={{ marginTop: 10 }} className="subtle danger">
              {parseError}
            </div>
          ) : null}

          {parseResult ? (
            <>
              <div className="kvs">
                <div>Filename</div>
                <div className="mono">{parseResult.document.filename}</div>

                <div>Media type</div>
                <div className="mono">{parseResult.document.media_type}</div>

                <div>Pages</div>
                <div className="mono">{String(parseResult.document.pages ?? '—')}</div>

                <div>Word count</div>
                <div className="mono">{String(parseResult.document.word_count ?? '—')}</div>

                <div>Clauses</div>
                <div className="mono">{parseResult.clauses.length}</div>
              </div>

              {parseResult.warnings?.length ? (
                <div style={{ marginTop: 10 }}>
                  <div className="subtle">Warnings:</div>
                  <ul className="subtle" style={{ marginTop: 6 }}>
                    {parseResult.warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div style={{ marginTop: 12, fontWeight: 600, fontSize: 13 }}>
                Clause list
              </div>
              <div className="clauses">
                {parseResult.clauses.slice(0, 250).map(c => {
                  const active = c.clause_id === selectedClauseId;
                  const ins = c.redlines?.insertions?.length ?? 0;
                  const del = c.redlines?.deletions?.length ?? 0;
                  const com = c.comments?.length ?? 0;

                  return (
                    <div
                      key={c.clause_id}
                      className={'clauseItem ' + (active ? 'clauseItemActive' : '')}
                      onClick={() => setSelectedClauseId(c.clause_id)}
                    >
                      <div className="clauseTitle">
                        <span>{clauseHeading(c)}</span>
                        <span className="mono">L{c.level}</span>
                      </div>
                      <div className="subtle">{preview(c.text)}</div>
                      <div className="clauseMeta" style={{ marginTop: 6 }}>
                        {ins ? <span className="pill">+{ins} ins</span> : null}
                        {del ? <span className="pill">-{del} del</span> : null}
                        {com ? <span className="pill">{com} comments</span> : null}
                        <span className="pill mono">{c.clause_id.slice(0, 16)}…</span>
                      </div>
                    </div>
                  );
                })}
                {parseResult.clauses.length > 250 ? (
                  <div className="subtle">
                    Showing first 250 clauses (for UI performance). Chat tools can access all clauses.
                  </div>
                ) : null}
              </div>

              {selectedClause ? (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>Selected clause</div>
                  <div className="pill mono" style={{ marginTop: 6 }}>
                    {selectedClause.clause_id}
                  </div>
                  <div className="pre" style={{ marginTop: 8 }}>
                    {selectedClause.text}
                  </div>

                  {(() => {
                    const changesText = formatChangesPlainTextForUI(selectedClause);
                    if (!changesText) return null;
                    return (
                      <div className="pre" style={{ marginTop: 8 }}>
                        {changesText}
                      </div>
                    );
                  })()}
                </div>
              ) : null}
            </>
          ) : (
            <div style={{ marginTop: 14 }} className="subtle">
              Parse a document to view clauses.
            </div>
          )}
        </div>
      </section>

      {/* RIGHT: Chat */}
      <section className="panel">
        <div className="panelHeader">
          <div>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Chat</div>
            <div className="subtle">
              Ask questions. The assistant can use tools to list/search/fetch clauses.
            </div>
          </div>
          <div className="row">
            <button className="btn" onClick={() => stop()} disabled={!(status === 'streaming' || status === 'submitted')}>
              Stop
            </button>
            <button className="btn" onClick={() => regenerate()} disabled={status !== 'ready'}>
              Regenerate
            </button>
            <span className="pill">
              Status: <span className="mono">{status}</span>
            </span>
          </div>
        </div>

        <div className="panelBody">
          <div className="chatMessages">
            {messages.map(m => (
              <div key={m.id} className={'msg ' + (m.role === 'user' ? 'msgUser' : '')}>
                <div className="msgHeader">
                  <span>{m.role === 'user' ? 'You' : 'Assistant'}</span>
                  <span className="mono">{new Date(m.createdAt || Date.now()).toLocaleTimeString()}</span>
                </div>
                <div className="msgBody">{renderMessageParts(m)}</div>
              </div>
            ))}
          </div>

          {error ? (
            <div style={{ marginTop: 10 }} className="subtle danger">
              {error.message}
            </div>
          ) : null}

          <ChatComposer
            disabled={status !== 'ready'}
            onSend={onSend}
            docReady={!!docId}
          />
        </div>
      </section>
    </div>
  );
}

function ChatComposer({
  disabled,
  onSend,
  docReady,
}: {
  disabled: boolean;
  docReady: boolean;
  onSend: (text: string) => Promise<void>;
}) {
  const [input, setInput] = useState('');

  return (
    <form
      style={{ marginTop: 12, display: 'flex', gap: 10 }}
      onSubmit={async e => {
        e.preventDefault();
        const text = input;
        setInput('');
        await onSend(text);
      }}
    >
      <input
        className="input"
        style={{ flex: 1, minWidth: 0 }}
        value={input}
        onChange={e => setInput(e.target.value)}
        disabled={disabled}
        placeholder={docReady ? 'Ask about the document…' : 'Parse a document first…'}
      />
      <button className="btn btnPrimary" type="submit" disabled={disabled || !input.trim()}>
        Send
      </button>
    </form>
  );
}
