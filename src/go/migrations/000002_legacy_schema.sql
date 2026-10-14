CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    user_key TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'note',
    source_type TEXT NOT NULL DEFAULT 'manual',
    source_uri TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]'::JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_key, source_type, source_uri)
);

CREATE TABLE IF NOT EXISTS memory_parent_nodes (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    node_type TEXT NOT NULL,
    node_key TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    metadata_text TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    embedding VECTOR(1024) NOT NULL,
    summary_embedding VECTOR(1024),
    child_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_id, node_type, node_key)
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    context_text TEXT NOT NULL DEFAULT '',
    metadata_text TEXT NOT NULL DEFAULT '',
    parent_id TEXT,
    embedding VECTOR(1024) NOT NULL,
    raw_embedding VECTOR(1024),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS memory_deletions (
    id TEXT PRIMARY KEY,
    user_key TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT '',
    source_uri TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
