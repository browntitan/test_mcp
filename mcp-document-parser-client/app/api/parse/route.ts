import { z } from 'zod';
import { randomUUID } from 'crypto';
import { callToolViaHttp, callToolViaMCP, extractTextFromMcpToolOutput } from '../../../lib/mcp';
import { putParsed, putUpload } from '../../../lib/documentStore';
import type { DocumentParseResult } from '../../../lib/types';

export const runtime = 'nodejs';

const ParseRequestSchema = z.object({
  filename: z.string().min(1),
  file_base64: z.string().min(1),
  options: z.record(z.any()).optional(),
});

function isPdf(filename: string): boolean {
  return filename.toLowerCase().endsWith('.pdf');
}

function inferFileType(filename: string): 'pdf' | 'docx' {
  return isPdf(filename) ? 'pdf' : 'docx';
}

function inferMediaType(filename: string): string {
  return isPdf(filename)
    ? 'application/pdf'
    : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
}

export async function POST(req: Request) {
  const json = await req.json().catch(() => null);
  const parsed = ParseRequestSchema.safeParse(json);

  if (!parsed.success) {
    return Response.json(
      { error: 'Invalid request', details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const { filename, file_base64, options } = parsed.data;

  const toolName = isPdf(filename) ? 'parse_pdf' : 'parse_docx';
  const toolArgs = { file_base64, options: options ?? {} };

  let parseResult: DocumentParseResult;

  // Prefer MCP tool call; fall back to direct HTTP endpoint.
  try {
    const out = await callToolViaMCP(toolName, toolArgs);
    const text = extractTextFromMcpToolOutput(out);
    parseResult = JSON.parse(text) as DocumentParseResult;
  } catch (err: any) {
    try {
      parseResult = await callToolViaHttp<DocumentParseResult>(
        toolName === 'parse_pdf' ? '/tools/parse_pdf' : '/tools/parse_docx',
        toolArgs,
      );
    } catch (err2: any) {
      return Response.json(
        { error: err2?.message || err?.message || 'Parse failed' },
        { status: 502 },
      );
    }
  }

  // Store in memory and return docId.
  const docId = randomUUID();

  // Persist the raw upload so chat workflows can re-parse or start risk assessment from file_base64.
  putUpload(docId, {
    filename,
    mediaType: inferMediaType(filename),
    fileType: inferFileType(filename),
    fileBase64: file_base64,
  });

  putParsed(docId, parseResult);

  return Response.json({ docId, parseResult });
}
