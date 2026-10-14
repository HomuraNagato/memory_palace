CREATE TABLE IF NOT EXISTS memory_nodes (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    parent_node_id TEXT REFERENCES memory_nodes(id) ON DELETE CASCADE,
    node_type TEXT NOT NULL,
    node_key TEXT NOT NULL,
    depth INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    metadata_text TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    embedding VECTOR(1024) NOT NULL,
    summary_embedding VECTOR(1024),
    passage_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_id, node_type, node_key)
);

ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS node_id TEXT;

INSERT INTO memory_nodes (
    id, memory_id, user_key, parent_node_id, node_type, node_key, depth,
    ordinal, title, text, metadata_text, summary, embedding,
    summary_embedding, passage_count, metadata, created_at
)
SELECT
    p.id, p.memory_id, p.user_key, NULL, p.node_type, p.node_key, 0,
    ROW_NUMBER() OVER (PARTITION BY p.memory_id ORDER BY p.created_at, p.id) - 1,
    p.title, p.text, p.metadata_text, p.summary, p.embedding,
    p.summary_embedding, p.child_count, p.metadata, p.created_at
FROM memory_parent_nodes p
ON CONFLICT DO NOTHING;

UPDATE memory_chunks c
SET node_id = c.parent_id
WHERE c.node_id IS NULL
  AND c.parent_id IS NOT NULL
  AND EXISTS (SELECT 1 FROM memory_nodes n WHERE n.id = c.parent_id);

-- Preserve any orphaned legacy chunk under a deterministic leaf node.
INSERT INTO memory_nodes (
    id, memory_id, user_key, parent_node_id, node_type, node_key, depth,
    ordinal, title, text, metadata_text, summary, embedding,
    summary_embedding, passage_count, metadata, created_at
)
SELECT
    'legacy-chunk:' || c.id, c.memory_id, c.user_key, NULL, 'passage',
    'legacy-chunk:' || c.id, 0, c.chunk_index, i.title, c.text,
    c.metadata_text, '', c.embedding, NULL, 1, c.metadata, c.created_at
FROM memory_chunks c
JOIN memory_items i ON i.id = c.memory_id
LEFT JOIN memory_nodes n ON n.id = c.node_id
WHERE c.node_id IS NULL OR n.id IS NULL
ON CONFLICT DO NOTHING;

UPDATE memory_chunks c
SET node_id = 'legacy-chunk:' || c.id
WHERE (c.node_id IS NULL OR NOT EXISTS (
    SELECT 1 FROM memory_nodes n WHERE n.id = c.node_id
)) AND EXISTS (
    SELECT 1 FROM memory_nodes n WHERE n.id = 'legacy-chunk:' || c.id
);

ALTER TABLE memory_chunks ALTER COLUMN node_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'memory_chunks'::regclass
          AND conname = 'memory_chunks_node_id_fkey'
    ) THEN
        ALTER TABLE memory_chunks
            ADD CONSTRAINT memory_chunks_node_id_fkey
            FOREIGN KEY (node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS memory_passage_windows (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    start_chunk_index INTEGER NOT NULL,
    end_chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_id, window_index)
);

CREATE INDEX IF NOT EXISTS idx_memory_items_user_updated ON memory_items(user_key, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_items_source ON memory_items(user_key, source_type, source_uri);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_user ON memory_chunks(user_key);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_node ON memory_chunks(user_key, node_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_memory_passage_windows_memory ON memory_passage_windows(user_key, memory_id, window_index);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_user ON memory_nodes(user_key, memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_parent ON memory_nodes(user_key, parent_node_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_embedding_hnsw_cosine ON memory_nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_summary_embedding_hnsw_cosine ON memory_nodes USING hnsw (summary_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw_cosine ON memory_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_passage_windows_embedding_hnsw_cosine ON memory_passage_windows USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_text_fts ON memory_chunks USING gin (to_tsvector('simple', text));
CREATE INDEX IF NOT EXISTS idx_memory_chunks_metadata_text_fts ON memory_chunks USING gin (to_tsvector('simple', metadata_text));
CREATE INDEX IF NOT EXISTS idx_memory_items_title_fts ON memory_items USING gin (to_tsvector('simple', title));
CREATE INDEX IF NOT EXISTS idx_memory_chunks_text_trgm ON memory_chunks USING gin (lower(text) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_metadata_text_trgm ON memory_chunks USING gin (lower(metadata_text) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memory_items_title_trgm ON memory_items USING gin (lower(title) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memory_items_source_uri_trgm ON memory_items USING gin (lower(source_uri) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memory_deletions_user_deleted ON memory_deletions(user_key, deleted_at DESC);
