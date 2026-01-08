import { z } from 'zod';
import { createOpenAI } from '@ai-sdk/openai';
import { convertToModelMessages, stepCountIs, streamText, tool, type UIMessage } from 'ai';

import { getDocument, getUpload } from '../../../lib/documentStore';
import type { StoredUpload } from '../../../lib/documentStore';
import type { Clause, StoredDocument } from '../../../lib/types';
function getUploadMeta(up: StoredUpload) {
  const ft = up.fileType || (up.filename.toLowerCase().endsWith('.pdf') ? 'pdf' : 'docx');
  const media_type = up.mediaType
    ? up.mediaType
    : ft === 'pdf'
      ? 'application/pdf'
      : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';

  return {
    filename: up.filename,
    media_type,
    pages: null,
    word_count: null,
    clause_count: 0,
    warnings: [],
  };
}

function inferUploadFileType(up: StoredUpload): 'pdf' | 'docx' {
  return up.fileType || (up.filename.toLowerCase().endsWith('.pdf') ? 'pdf' : 'docx');
}

export const runtime = 'nodejs';
export const maxDuration = 60;

const ChatRequestSchema = z.object({
  docId: z.string().min(1),
  focusClauseId: z.string().optional().nullable(),
  // `useChat` sends UIMessage[]; validating fully is noisy, so keep it permissive.
  messages: z.array(z.any()),
});

// ------------------------------
// MCP server bridge (dynamic tools/list + tools/call)
// ------------------------------

const MCP_BASE_URL = (process.env.MCP_SERVER_URL || 'http://localhost:8765').replace(/\/$/, '');
const MCP_SESSION_ID = process.env.MCP_SESSION_ID || 'nextjs-bridge';

// Minimal MCP JSON-RPC over HTTP helper. We do not need SSE for this bridge.
async function mcpRpc<T>(method: string, params: any): Promise<T> {
  const body = {
    jsonrpc: '2.0',
    id: Math.floor(Math.random() * 1e9),
    method,
    params,
  };

  const res = await fetch(
    `${MCP_BASE_URL}/messages?session_id=${encodeURIComponent(MCP_SESSION_ID)}`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
  );

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`MCP RPC failed (${res.status}): ${txt || res.statusText}`);
  }

  const json = (await res.json()) as any;
  if (json.error) {
    throw new Error(`MCP RPC error: ${json.error.message || JSON.stringify(json.error)}`);
  }
  return json.result as T;
}

type McpToolDef = {
  name: string;
  description?: string;
  inputSchema?: any;
};

async function mcpListTools(): Promise<McpToolDef[]> {
  // MCP servers typically return { tools: [...] } for tools/list
  const result = await mcpRpc<any>('tools/list', {});
  const tools = Array.isArray(result?.tools) ? result.tools : Array.isArray(result) ? result : [];
  return tools as McpToolDef[];
}

async function mcpCallTool(name: string, args: any): Promise<any> {
  // MCP tools/call typically expects { name, arguments }
  const result = await mcpRpc<any>('tools/call', { name, arguments: args ?? {} });

  // MCP servers often return: { content: [{ type: 'text', text: '...' }, ...] }
  const content = result?.content;
  if (Array.isArray(content) && content.length > 0) {
    // Collect *all* text-ish blocks, not just the first.
    const texts = content
      .map((c: any) => {
        if (typeof c?.text === 'string') return c.text;
        if (typeof c?.content === 'string') return c.content;
        return '';
      })
      .filter(Boolean);

    const joined = texts.join('\n').trim();

    if (joined) {
      // Try to parse JSON; fall back to raw text.
      try {
        return JSON.parse(joined);
      } catch {
        return joined;
      }
    }

    // If there were no text blocks, return the raw content array.
    return content;
  }

  // Some servers return a structured result directly.
  if (result && typeof result === 'object') {
    if ('output' in result) return (result as any).output;
    if ('result' in result) return (result as any).result;
  }

  return result;
}

let _mcpInitialized = false;
async function ensureMcpInitialized(): Promise<void> {
  if (_mcpInitialized) return;
  _mcpInitialized = true;
  try {
    await mcpRpc('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'mcp-document-parser-client', version: '0.1.0' },
    });
    await mcpRpc('initialized', {});
  } catch {
    // Best-effort: some servers allow tools/list without initialize.
  }
}

// Cache tool list per process to avoid re-fetching on every request.
let _mcpToolsCache: { at: number; tools: McpToolDef[] } | null = null;
async function getMcpToolsCached(ttlMs = 30_000): Promise<McpToolDef[]> {
  const now = Date.now();
  if (_mcpToolsCache && now - _mcpToolsCache.at < ttlMs) return _mcpToolsCache.tools;
  await ensureMcpInitialized();
  const tools = await mcpListTools();
  _mcpToolsCache = { at: now, tools };
  return tools;
}

function sleepMs(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function sanitizeMcpToolName(name: string) {
  // AI SDK/OpenAI-style tool names are safest as identifier-ish strings.
  // Convert dots/slashes/spaces into double-underscores.
  return (
    'mcp__' +
    String(name)
      .trim()
      .replace(/[\s\/\.]+/g, '__')
      .replace(/[^a-zA-Z0-9_:-]/g, '_')
  );
}

function getDocMeta(doc: StoredDocument) {
  const data: any = doc.data;
  const base = {
    filename: data.document?.filename,
    media_type: data.document?.media_type,
    pages: data.document?.pages ?? null,
    word_count: data.document?.word_count ?? null,
    clause_count: Array.isArray(data.clauses) ? data.clauses.length : 0,
  };

  const warnings =
    doc.kind === 'parsed'
      ? (data.warnings ?? [])
      : (data.normalization_warnings ?? []);

  return { ...base, warnings };
}

function safePreview(text: string, max = 240) {
  const t = (text || '').replace(/\s+/g, ' ').trim();
  if (t.length <= max) return t;
  return t.slice(0, max - 1) + '…';
}

function formatLabelTitle(c: Clause) {
  const parts: string[] = [];
  if (c.label) parts.push(String(c.label));
  if (c.title) parts.push(String(c.title));
  return parts.join(' ').trim();
}

function formatChangesPlainText(c: Clause) {
  const changes = (c as any).changes;
  const hasAny =
    changes &&
    (Array.isArray(changes.added) && changes.added.length > 0 ||
      Array.isArray(changes.deleted) && changes.deleted.length > 0 ||
      Array.isArray(changes.modified) && changes.modified.length > 0 ||
      Array.isArray(changes.comments) && changes.comments.length > 0);

  // Back-compat fallback if `changes` is missing
  const fallbackHasAny =
    (c.redlines?.insertions?.length ?? 0) > 0 ||
    (c.redlines?.deletions?.length ?? 0) > 0 ||
    (c.redlines?.strikethroughs?.length ?? 0) > 0 ||
    (c.comments?.length ?? 0) > 0;

  if (!hasAny && !fallbackHasAny) return '';

  const lines: string[] = [];
  lines.push('');
  lines.push('Changes:');

  if (hasAny) {
    const added = Array.isArray(changes.added) ? changes.added : [];
    const deleted = Array.isArray(changes.deleted) ? changes.deleted : [];
    const modified = Array.isArray(changes.modified) ? changes.modified : [];
    const comms = Array.isArray(changes.comments) ? changes.comments : [];

    if (added.length) {
      lines.push('Added:');
      for (const a of added) {
        const label = a?.label ? String(a.label) : '';
        const txt = (a?.text ?? '').toString().trim();
        lines.push(`- ${label ? label + ' ' : ''}${txt}`.trim());
      }
      lines.push('');
    }

    if (deleted.length) {
      lines.push('Deleted:');
      for (const d of deleted) {
        const label = d?.label ? String(d.label) : '';
        const txt = (d?.text ?? '').toString().trim();
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
        lines.push(`- ${label ? label + ' ' : ''}from: ${from} | to: ${to}`.trim());
      }
      lines.push('');
    }

    if (comms.length) {
      lines.push('Comments:');
      for (const cm of comms) {
        const author = cm?.author ? String(cm.author) : 'Reviewer';
        const anchor = cm?.anchor_text ? String(cm.anchor_text).replace(/\s+/g, ' ').trim() : '';
        const txt = (cm?.text ?? '').toString().trim();
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

  // Fallback formatting if server didn't provide `changes`
  const ins = c.redlines?.insertions ?? [];
  const del = c.redlines?.deletions ?? [];
  const comms = c.comments ?? [];

  if (ins.length) {
    lines.push('Added:');
    for (const a of ins) {
      const txt = (a?.text ?? '').toString().trim();
      lines.push(`- ${txt}`);
    }
    lines.push('');
  }

  if (del.length) {
    lines.push('Deleted:');
    for (const d of del) {
      const txt = (d?.text ?? '').toString().trim();
      lines.push(`- ${txt}`);
    }
    lines.push('');
  }

  if (comms.length) {
    lines.push('Comments:');
    for (const cm of comms) {
      const author = cm.author || 'Reviewer';
      const txt = (cm.text ?? '').toString().trim();
      lines.push(`- ${author}: ${txt}`);
    }
    lines.push('');
  }

  return lines.join('\n').trimEnd();
}

function formatClauseForLLM(c: Clause) {
  const header = formatLabelTitle(c);
  const parts: string[] = [];
  if (header) parts.push(header);
  parts.push(c.text);
  const changes = formatChangesPlainText(c);
  if (changes) parts.push(changes);
  return parts.join('\n').trim();
}

type ClauseItem = {
  clause_id: string | null;
  label: string | null;
  title: string | null;
  header: string;
  text_with_changes: string | null;
  risk_level: string | null;
  risk_score: number | null;
  justification: string | null;
  issues: any[];
  citations: any[];
  recommended_redline: string | null;
};

function extractClauseItemsFromReport(report: any): ClauseItem[] {
  const results = Array.isArray(report?.clause_results) ? report.clause_results : [];

  return results.map((item: any) => {
    const assessment = item?.assessment ?? item;

    const clause_id =
      (assessment?.clause_id ?? item?.clause_id ?? null) !== undefined
        ? String(assessment?.clause_id ?? item?.clause_id)
        : null;

    const labelRaw = item?.label ?? assessment?.label ?? null;
    const titleRaw = item?.title ?? assessment?.title ?? null;

    const label = typeof labelRaw === 'string' ? labelRaw : labelRaw != null ? String(labelRaw) : null;
    const title = typeof titleRaw === 'string' ? titleRaw : titleRaw != null ? String(titleRaw) : null;

    const header = [label, title].filter(Boolean).join(' ').trim() || clause_id || '(unknown clause)';

    const text_with_changes =
      typeof item?.text_with_changes === 'string' && item.text_with_changes.trim()
        ? item.text_with_changes
        : null;

    const risk_level = typeof assessment?.risk_level === 'string' ? assessment.risk_level : null;
    const risk_score = Number.isFinite(assessment?.risk_score) ? Number(assessment.risk_score) : null;
    const justification = typeof assessment?.justification === 'string' ? assessment.justification : null;

    const issues = Array.isArray(assessment?.issues) ? assessment.issues : [];
    const citations = Array.isArray(assessment?.citations) ? assessment.citations : [];
    const recommended_redline =
      typeof assessment?.recommended_redline === 'string' && assessment.recommended_redline.trim()
        ? assessment.recommended_redline
        : null;

    return {
      clause_id,
      label,
      title,
      header,
      text_with_changes,
      risk_level,
      risk_score,
      justification,
      issues,
      citations,
      recommended_redline,
    };
  });
}

function zodFromJsonSchemaLoose(_schema: any) {
  // Full JSON Schema -> Zod conversion is overkill here.
  // We accept any args object and let the MCP server validate.
  return z.record(z.any()).default({});
}

function findClause(clauses: Clause[], clause_id: string): Clause | undefined {
  return clauses.find(c => c.clause_id === clause_id);
}

// OpenAI-compatible provider pointing at Ollama's /v1 endpoint
const openai = createOpenAI({
  baseURL: (() => {
    const explicit = process.env.OLLAMA_OPENAI_BASE_URL;
    if (explicit && explicit.trim()) return explicit.trim().replace(/\/$/, '');

    const raw = (process.env.OLLAMA_BASE_URL || 'http://localhost:11434/api').trim();
    const cleaned = raw.replace(/\/$/, '');
    if (cleaned.endsWith('/api')) return cleaned.slice(0, -4) + '/v1';
    if (cleaned.endsWith('/v1')) return cleaned;
    return cleaned + '/v1';
  })(),
  apiKey: process.env.OPENAI_API_KEY || 'ollama',
});

export async function POST(req: Request) {
  const json = await req.json().catch(() => null);
  const parsed = ChatRequestSchema.safeParse(json);

  if (!parsed.success) {
    return Response.json(
      { error: 'Invalid request', details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const { docId, focusClauseId, messages } = parsed.data;

  const doc = getDocument(docId);
  const upload = getUpload(docId);

  if (!doc && !upload) {
    return Response.json(
      { error: 'Document not found. Upload/parse again to get a new docId.' },
      { status: 404 },
    );
  }

  const data: any = doc ? (doc as any).data : null;
  const clauses: Clause[] = (data?.clauses ?? []) as Clause[];
  const meta = doc ? getDocMeta(doc) : getUploadMeta(upload as StoredUpload);

  const focusClause = focusClauseId ? findClause(clauses, focusClauseId) : undefined;

  const localTools = {
    document_info: tool({
      description:
        'Get document metadata: filename, media type, pages, word count, clause count, and warnings.',
      inputSchema: z.object({}),
      execute: async () => meta,
    }),

    list_clauses: tool({
      description:
        'List clauses (outline). Returns clause_id, label, title, level, parent_clause_id, and a short preview of the text.',
      inputSchema: z.object({
        level: z.number().int().min(1).optional(),
        limit: z.number().int().min(1).max(500).default(50),
        offset: z.number().int().min(0).default(0),
      }),
      execute: async ({ level, limit, offset }) => {
        const filtered = typeof level === 'number' ? clauses.filter(c => c.level === level) : clauses;
        const slice = filtered.slice(offset, offset + limit);

        return slice.map(c => ({
          clause_id: c.clause_id,
          label: c.label ?? null,
          title: c.title ?? null,
          level: c.level,
          parent_clause_id: c.parent_clause_id ?? null,
          preview: safePreview(c.text, 200),
        }));
      },
    }),

    get_clause: tool({
      description:
        'Fetch a single clause by clause_id. Returns the clause text and a plain-text "Changes" section (added/deleted/modified/comments) so you can assess risk with full review context.',
      inputSchema: z.object({
        clause_id: z.string().min(1),
      }),
      execute: async ({ clause_id }) => {
        const c = findClause(clauses, clause_id);
        if (!c) {
          return { error: `Clause not found: ${clause_id}` };
        }

        const insertions = c.redlines?.insertions?.length ?? 0;
        const deletions = c.redlines?.deletions?.length ?? 0;
        const strikes = c.redlines?.strikethroughs?.length ?? 0;
        const comments = c.comments?.length ?? 0;

        return {
          clause_id: c.clause_id,
          label: c.label ?? null,
          title: c.title ?? null,
          level: c.level,
          parent_clause_id: c.parent_clause_id ?? null,
          text: c.text,
          text_with_changes: formatClauseForLLM(c),
          changes: (c as any).changes ?? null,
          redlines: { insertions, deletions, strikethroughs: strikes },
          comments_count: comments,
        };
      },
    }),

    search_clauses: tool({
      description:
        'Search for text in clause bodies (case-insensitive). Returns best matches with clause_id and a snippet.',
      inputSchema: z.object({
        query: z.string().min(1),
        limit: z.number().int().min(1).max(50).default(10),
      }),
      execute: async ({ query, limit }) => {
        const q = query.toLowerCase();
        const matches: Array<{ clause: Clause; score: number }> = [];

        for (const c of clauses) {
          const hay = `${c.label ?? ''} ${c.title ?? ''} ${c.text ?? ''}`.toLowerCase();
          const idx = hay.indexOf(q);
          if (idx !== -1) matches.push({ clause: c, score: Math.max(1, 10000 - idx) });
        }

        matches.sort((a, b) => b.score - a.score);

        return matches.slice(0, limit).map(m => ({
          clause_id: m.clause.clause_id,
          label: m.clause.label ?? null,
          title: m.clause.title ?? null,
          level: m.clause.level,
          snippet: safePreview(m.clause.text, 260),
        }));
      },
    }),

    focus_clause: tool({
      description:
        'Get the currently selected clause from the UI (if any). Useful when the user says "this clause".',
      inputSchema: z.object({}),
      execute: async () => {
        if (!focusClause) return { focus: null };
        return {
          focus: {
            clause_id: focusClause.clause_id,
            label: focusClause.label ?? null,
            title: focusClause.title ?? null,
            level: focusClause.level,
            text: focusClause.text,
            text_with_changes: formatClauseForLLM(focusClause),
            changes: (focusClause as any).changes ?? null,
          },
        };
      },
    }),

    // ✅ Important: NO docId parameter. Always uses the document already loaded for this chat request.
    run_risk_assessment: tool({
      description:
        'Start the clause-by-clause risk assessment workflow for the currently loaded document (no docId needed). Returns an assessment_id immediately. Use get_risk_assessment_status to poll and get_risk_assessment_report to fetch the final report.',
      inputSchema: z.object({
        policy_collection: z.string().default('default'),
        top_k: z.number().int().min(1).max(50).default(3),
        min_score: z.number().optional(),
        model_profile: z.enum(['chat', 'assessment']).default('assessment'),
        mode: z.enum(['sync', 'async']).optional(),
        format: z.enum(['json', 'markdown']).default('markdown'),
        wait_for_completion: z.boolean().default(true),
        max_wait_ms: z.number().int().min(1000).max(55_000).default(45_000),
        poll_interval_ms: z.number().int().min(250).max(5000).default(750),
      }),
      execute: async (args: any) => {
        const parseResult: any = data;
        const uploadPayload = upload as StoredUpload | undefined;
        const hasParseResult = !!(parseResult && typeof parseResult === 'object' && Array.isArray((parseResult as any)?.clauses));

        const clauseCount = hasParseResult ? (parseResult as any).clauses.length : null;

        const waitForCompletion = args?.wait_for_completion !== false; // default true
        const maxWaitMs = Number.isFinite(args?.max_wait_ms) ? Number(args.max_wait_ms) : 45_000;
        const pollIntervalMs = Number.isFinite(args?.poll_interval_ms) ? Number(args.poll_interval_ms) : 750;

        // If the caller didn't specify a mode, pick one.
        // For small docs and when the user wants the final report now, prefer sync to finish in one request.
        let effectiveMode: 'sync' | 'async' = (args?.mode as any) || 'async';
        if (!args?.mode && waitForCompletion && typeof clauseCount === 'number' && clauseCount > 0 && clauseCount <= 12) {
          effectiveMode = 'sync';
        }

        const startPayload: any = {
          policy_collection: args?.policy_collection ?? 'default',
          top_k: args?.top_k ?? 3,
          min_score: args?.min_score,
          model_profile: args?.model_profile ?? 'assessment',
          include_text_with_changes: true,
          mode: effectiveMode,
        };

        if (hasParseResult) {
          startPayload.parse_result = parseResult;
        } else if (uploadPayload && uploadPayload.fileBase64) {
          startPayload.file_base64 = uploadPayload.fileBase64;
          startPayload.filename = uploadPayload.filename;
          startPayload.file_type = inferUploadFileType(uploadPayload);
        } else {
          return { error: 'No parsed document or upload payload available for this docId.' };
        }

        const started = await mcpCallTool('risk_assessment.start', startPayload);
        const assessment_id = (started as any)?.assessment_id;
        const status = (started as any)?.status;

        // If MCP returned an error or a non-object, surface it clearly to the model.
        if (!started || typeof started !== 'object') {
          return { error: 'risk_assessment.start returned a non-object response', started };
        }
        if ((started as any).error) {
          return { error: 'risk_assessment.start returned an error', started };
        }

        if (!assessment_id) {
          return { error: 'risk_assessment.start did not return assessment_id', started };
        }

        let status_output: any = null;
        let final_status: any = status;

        // If async, optionally poll until completion so the user gets the final report in one tool call.
        if (assessment_id && effectiveMode === 'async' && waitForCompletion && status && status !== 'completed') {
          const deadline = Date.now() + Math.min(Math.max(maxWaitMs, 1000), 55_000);
          while (Date.now() < deadline) {
            await sleepMs(pollIntervalMs);
            try {
              status_output = await mcpCallTool('risk_assessment.status', { assessment_id });
              final_status = (status_output as any)?.status ?? final_status;
              if (final_status === 'completed' || final_status === 'failed' || final_status === 'canceled') {
                break;
              }
            } catch (e) {
              // If polling fails, break and return what we have.
              break;
            }
          }
        }

        // Fetch a JSON report snapshot once we have an assessment_id.
        // Even if the job is still running, the server can return partial clause_results.
        let report: any = null; // JSON report
        let report_markdown: string | null = null;
        let report_error: string | null = null;

        if (assessment_id) {
          try {
            // Always fetch JSON so the UI can render per-clause items.
            report = await mcpCallTool('risk_assessment.report', {
              assessment_id,
              format: 'json',
            });

            // Optionally fetch markdown too (for easy display/debugging).
            const wantMarkdown = (args?.format ?? 'markdown') === 'markdown';
            if (wantMarkdown) {
              const md = await mcpCallTool('risk_assessment.report', {
                assessment_id,
                format: 'markdown',
              });
              if (md && typeof (md as any).summary === 'string') {
                report_markdown = (md as any).summary;
              }
            }
          } catch (e: any) {
            report_error = e?.message || String(e);
          }
        }

        const clause_items = report ? extractClauseItemsFromReport(report) : [];

        return {
          docId,
          assessment_id,
          status,
          effectiveMode,
          final_status,
          status_output,
          started,
          report,
          clause_items,
          report_markdown,
          report_error,
          note:
            (final_status ?? status) !== 'completed'
              ? 'Assessment started but not completed within this request. Use get_risk_assessment_status to poll and get_risk_assessment_report to fetch the final report using assessment_id.'
              : undefined,
        };
      },
    }),

    get_risk_assessment_status: tool({
      description:
        'Check status/progress for a previously started risk assessment by assessment_id (convenience wrapper).',
      inputSchema: z.object({
        assessment_id: z.string().min(1),
      }),
      execute: async ({ assessment_id }) => {
        try {
          const out = await mcpCallTool('risk_assessment.status', { assessment_id });
          if (out && typeof out === 'object' && (out as any).error) {
            return { error: 'risk_assessment.status returned an error', out };
          }
          return out;
        } catch (e: any) {
          return { error: e?.message || String(e) };
        }
      },
    }),

    get_risk_assessment_report: tool({
      description:
        'Fetch the final risk assessment report by assessment_id (convenience wrapper). Returns markdown in report_markdown when available.',
      inputSchema: z.object({
        assessment_id: z.string().min(1),
        format: z.enum(['json', 'markdown']).default('markdown'),
      }),
      execute: async ({ assessment_id, format }) => {
        try {
          // Always fetch JSON for structured clause_items
          const report = await mcpCallTool('risk_assessment.report', { assessment_id, format: 'json' });
          if (report && typeof report === 'object' && (report as any).error) {
            return { error: 'risk_assessment.report returned an error', report };
          }

          const clause_items = extractClauseItemsFromReport(report);

          let report_markdown: string | null = null;
          if ((format ?? 'markdown') === 'markdown') {
            const md = await mcpCallTool('risk_assessment.report', { assessment_id, format: 'markdown' });
            report_markdown = md && typeof (md as any).summary === 'string' ? (md as any).summary : null;
          }

          return { assessment_id, report, clause_items, report_markdown };
        } catch (e: any) {
          return { error: e?.message || String(e) };
        }
      },
    }),
  };

  // ------------------------------
  // Dynamic MCP tools
  // ------------------------------

  const mcpToolDefs = await getMcpToolsCached();

  const mcpTools: Record<string, any> = {};
  const mcpNameMap: Record<string, string> = {}; // safeName -> original MCP name

  for (const t of mcpToolDefs) {
    if (!t?.name) continue;

    const safeName = sanitizeMcpToolName(t.name);
    mcpNameMap[safeName] = t.name;

    // If a local tool already uses this safe name, skip.
    if ((localTools as any)[safeName]) continue;

    mcpTools[safeName] = tool({
      description: t.description || `MCP tool: ${t.name}`,
      inputSchema: zodFromJsonSchemaLoose(t.inputSchema),
      execute: async (args: any) => {
        try {
          return await mcpCallTool(t.name, args);
        } catch (e: any) {
          return { error: e?.message || String(e) };
        }
      },
    });
  }

  const tools = {
    ...localTools,
    ...mcpTools,
  };

  const system = [
    `You are a legal document analysis assistant.`,
    `The user is chatting about ONE document (parsed results may be available; risk assessment can parse on demand).`,
    `Use the available tools to inspect the document clause-by-clause (don't guess).`,
    `If the user asks for a risk assessment, call run_risk_assessment (no docId needed). Prefer wait_for_completion=true; it returns report (JSON) + clause_items (per-clause results) and often report_markdown.`,
    `If run_risk_assessment returns an assessment_id but final_status is not completed, use get_risk_assessment_status and then get_risk_assessment_report with that assessment_id.`,
    `After calling tools, ALWAYS write a final plain-text answer for the user.`,
    ``,
    `Document: ${meta.filename} (${meta.media_type})`,
    `Clauses: ${meta.clause_count}; Pages: ${meta.pages ?? '—'}; Word count: ${meta.word_count ?? '—'}`,
    meta.warnings?.length ? `Warnings: ${meta.warnings.join(' | ')}` : '',
    ``,
    `When you cite facts from the document, include clause references like: (Clause <label or id>).`,
    `If you need to find a topic, call search_clauses first.`,
    `If you need the full text, call get_clause.`,
  ]
    .filter(Boolean)
    .join('\n');

  const modelId = process.env.OLLAMA_MODEL || 'gpt-oss:20b';

  const result = streamText({
    model: openai.chat(modelId),
    system,
    messages: await convertToModelMessages(messages as UIMessage[]),
    tools,
    // Allow the model to do multiple tool calls if it chooses (start -> status -> report).
    maxSteps: 12,
    stopWhen: stepCountIs(12),
  });

  return result.toUIMessageStreamResponse({
    onError: (error) => {
      if (error instanceof Error) return error.message;
      return typeof error === 'string' ? error : JSON.stringify(error);
    },
  });
}