ALTER TABLE memory_chunks
    ADD COLUMN IF NOT EXISTS context_text TEXT NOT NULL DEFAULT '';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'memory_chunks'
          AND column_name = 'retrieval_text'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'memory_chunks'
          AND column_name = 'metadata_text'
    ) THEN
        ALTER TABLE memory_chunks RENAME COLUMN retrieval_text TO metadata_text;
    END IF;
END $$;

ALTER TABLE memory_chunks
    ADD COLUMN IF NOT EXISTS metadata_text TEXT NOT NULL DEFAULT '';

UPDATE memory_chunks
SET metadata_text = text
WHERE metadata_text = '';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema()
          AND c.relname = 'memory_chunks'
          AND a.attname = 'embedding'
          AND a.atttypmod = -1
    ) THEN
        ALTER TABLE memory_chunks
            ALTER COLUMN embedding TYPE VECTOR(1024)
            USING embedding::VECTOR(1024);
    END IF;
END $$;
