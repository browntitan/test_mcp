import type { ReactNode } from 'react';
import './globals.css';

export const metadata = {
  title: 'MCP Document Parser Client',
  description: 'Next.js client for a Python MCP document parser + AI SDK chat',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <header className="header">
            <div>
              <h1>MCP Document Parser Client</h1>
              <p className="subtle">
                Upload a DOCX/PDF, parse it, and chat with an LLM that can iterate through clauses.
              </p>
            </div>
          </header>
          <main>{children}</main>
          <footer className="footer subtle">
            <span>
              Local/dev demo. For production, add auth + persistent storage.
            </span>
          </footer>
        </div>
      </body>
    </html>
  );
}
