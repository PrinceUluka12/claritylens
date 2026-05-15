-- docker/init.sql
-- Runs ONCE when the container is first created.
-- If you delete the container and recreate it, this runs again.
-- If you just restart the container, this does NOT run again.

-- Enable pgvector — this is what lets us store embeddings as a column type
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded PDF
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- One row per clause chunk extracted from a document.
-- This is the most important table — it holds the text AND its vector.
--
-- CPU NOTE: embedding dimension is 768 (DistilBERT hidden size).
-- If you later switch to your Word2Vec embedder (Phase 5),
-- you'll change this to match that model's output dimension.
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,   -- position 0, 1, 2... within the document
    text        TEXT NOT NULL,      -- raw clause text
    embedding   vector(768),        -- the semantic vector for this chunk
    risk_label  TEXT,               -- e.g. 'indemnity', 'ip_assignment'
    risk_score  FLOAT,              -- 0.0 to 1.0 confidence
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- This index is what makes Q&A search fast.
-- Without it, every search scans every row (slow).
-- With it, pgvector uses approximate nearest-neighbour search (fast).
-- ivfflat = "inverted file with flat quantization" — best for CPU use.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);