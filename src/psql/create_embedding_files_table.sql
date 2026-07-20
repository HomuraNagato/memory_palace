-- Create table to track files that have been processed for embeddings
CREATE TABLE embedding_files (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    last_modified TIMESTAMP NOT NULL,
    hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster lookups
CREATE INDEX idx_embedding_files_path ON embedding_files(file_path);
CREATE INDEX idx_embedding_files_hash ON embedding_files(hash);
