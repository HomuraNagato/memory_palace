
------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings_lite (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

CREATE TRIGGER update_embeddings_lite_updated_at
BEFORE UPDATE ON embeddings_lite
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
------------------------------------------------------

-- indexes for source and Vector similarity index (using pgvector ivfflat index) --
CREATE INDEX IF NOT EXISTS idx_embeddings_lite_source ON embeddings_lite(source);
------------------------------------------------------
