import { createMCPClient } from '@ai-sdk/mcp';

export function getParserHttpBaseUrl(): string {
  return process.env.MCP_PARSER_HTTP_BASE_URL?.trim() || 'http://localhost:8765';
}

export function getParserSseUrl(): string {
  return process.env.MCP_PARSER_SSE_URL?.trim() || 'http://localhost:8765/sse';
}

export async function withMCPClient<T>(fn: (client: Awaited<ReturnType<typeof createMCPClient>>) => Promise<T>): Promise<T> {
  const client = await createMCPClient({
    transport: {
      type: 'sse',
      url: getParserSseUrl(),
    },
  });

  try {
    return await fn(client);
  } finally {
    await client.close();
  }
}

/**
 * The Python MCP server returns `tools/call` results as `{ content: [{ type: 'text', text: '...json...' }] }`.
 * Depending on AI SDK version, the `execute()` return can be:
 * - a raw string
 * - a `{ content: [...] }` object
 * - an array of content parts
 */
export function extractTextFromMcpToolOutput(output: unknown): string {
  if (typeof output === 'string') return output;

  if (!output || typeof output !== 'object') {
    throw new Error('Unexpected MCP tool output type');
  }

  const obj: any = output;

  // Sometimes the tool output is the raw MCP result
  if (Array.isArray(obj?.content)) {
    const parts = obj.content;
    const texts = parts
      .filter((p: any) => p && typeof p === 'object' && p.type === 'text' && typeof p.text === 'string')
      .map((p: any) => p.text);
    if (texts.length) return texts.join('\n');
  }

  // Sometimes it is directly an array of parts
  if (Array.isArray(obj)) {
    const texts = (obj as any[])
      .filter((p: any) => p && typeof p === 'object' && p.type === 'text' && typeof p.text === 'string')
      .map((p: any) => p.text);
    if (texts.length) return texts.join('\n');
  }

  // Sometimes it is { text: "..." }
  if (typeof obj.text === 'string') return obj.text;

  throw new Error('Could not extract text from MCP tool output');
}

export async function callToolViaMCP(toolName: string, args: Record<string, unknown>): Promise<unknown> {
  return withMCPClient(async client => {
    const tools: any = await client.tools();

    const tool = tools?.[toolName];
    if (!tool) {
      throw new Error(`Tool not found on MCP server: ${toolName}`);
    }

    if (typeof tool.execute !== 'function') {
      throw new Error(`MCP tool is missing execute(): ${toolName}`);
    }

    return await tool.execute(args);
  });
}

export async function callToolViaHttp<T>(path: string, payload: unknown): Promise<T> {
  const base = getParserHttpBaseUrl().replace(/\/$/, '');
  const url = `${base}${path.startsWith('/') ? '' : '/'}${path}`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} from parser server: ${text || res.statusText}`);
  }

  return (await res.json()) as T;
}
