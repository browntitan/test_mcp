import { randomUUID } from 'crypto';

import { putUpload } from '../../../lib/documentStore';

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

function asNonEmptyString(v: unknown): string | null {
  if (typeof v !== 'string') return null;
  const s = v.trim();
  return s ? s : null;
}

// Keep this conservative; base64 inflates size by ~33%.
// Increase later if you need larger documents.
const MAX_BASE64_CHARS = 35_000_000; // ~25MB raw-ish

export async function POST(req: Request) {
  const json = await req.json().catch(() => null);

  const filename = asNonEmptyString(json?.filename);
  const file_base64 = asNonEmptyString(json?.file_base64);

  if (!filename || !file_base64) {
    return Response.json(
      { error: 'Invalid request. Required fields: filename, file_base64' },
      { status: 400 },
    );
  }

  if (file_base64.length > MAX_BASE64_CHARS) {
    return Response.json(
      {
        error: `Upload too large. base64 length=${file_base64.length} exceeds limit=${MAX_BASE64_CHARS}.`,
      },
      { status: 413 },
    );
  }

  const docId = randomUUID();

  // Store the raw upload so /api/chat can start risk_assessment with file_base64 (parse on demand).
  putUpload(docId, {
    filename,
    mediaType: inferMediaType(filename),
    fileType: inferFileType(filename),
    fileBase64: file_base64,
  });

  return Response.json({ docId });
}
