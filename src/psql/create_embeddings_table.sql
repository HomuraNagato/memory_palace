
------------------------------------------------------
-- Drop and recreate embeddings table with proper schema
DROP TABLE IF EXISTS embeddings CASCADE;

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES embedding_files(id) ON DELETE CASCADE,
    row_num INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    emb VECTOR(1536) NOT NULL,  -- 1536 for text-embedding-ada-002
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(file_id, row_num, chunk_idx)  -- Prevent duplicate chunks
);
------------------------------------------------------

-- update updated_at timestamp on row modification  --
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_embeddings_updated_at
BEFORE UPDATE ON embeddings
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
------------------------------------------------------

-- Indexes for better query performance --
CREATE INDEX idx_embeddings_file_id ON embeddings(file_id);
CREATE INDEX idx_embeddings_emb ON embeddings USING ivfflat (emb vector_cosine_ops) WITH (lists = 100);

-- Comments for documentation --
COMMENT ON TABLE embeddings IS 'Stores text chunks and their embedding vectors';
COMMENT ON COLUMN embeddings.file_id IS 'Foreign key to embedding_files table';
COMMENT ON COLUMN embeddings.row_num IS 'Row number within the source file (0-indexed)';
COMMENT ON COLUMN embeddings.chunk_idx IS 'Chunk index within the row (0-indexed)';
COMMENT ON COLUMN embeddings.emb IS 'Vector embedding (1536 dimensions for ada-002)';
------------------------------------------------------
