CREATE TABLE IF NOT EXISTS policy_chunks (
  policy_id   text NOT NULL,
  chunk_id    text NOT NULL,
  collection  text NOT NULL DEFAULT 'default',
  text        text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding   vector NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (policy_id, chunk_id)
);

-- Helpful filter index
CREATE INDEX IF NOT EXISTS policy_chunks_collection_idx
  ON policy_chunks (collection);