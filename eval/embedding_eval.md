# Embedding Eval

Use the Go eval runner to compare embedding models without changing the live
`memory_chunks.embedding` column.

The runner loads chunks from Postgres, embeds the query plus candidate chunks
through an OpenAI-compatible `/v1/embeddings` endpoint, ranks chunks by cosine
similarity, and reports the best rank for chunks containing a target substring.
Use `-expected-title` when the expected result is a document rather than a
specific text substring.

For persistent comparison, use the sidecar table
`memory_chunk_embeddings_3072`. It stores alternate 3072-dimensional chunk
embeddings keyed by `chunk_id` and `model_key`, leaving the production
`memory_chunks.embedding` column unchanged. The table uses `HALFVEC(3072)`
because pgvector HNSW indexes on regular `VECTOR` columns are limited to 2000
dimensions.

## Hold-Out Query

Keep the current question and expected answer outside the repository so the
implementation cannot be tuned to the hold-out. Supply them at run time:

```sh
cd memory_palace/src/go

go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -top 8
```

## Title-Based Evaluation

```sh
cd memory_palace/src/go

go run ./cmd/embed-eval \
  -query "starbucks pistachio soy milk" \
  -expected-title "starbucks.md" \
  -scope memories \
  -collection obsidian \
  -models 'current|http://127.0.0.1:5021/v1/embeddings|Qwen3-Embedding-0.6B||'
```

## Model Specs

Model specs use:

```text
name|url|model|dimensions|key_env
```

Examples:

```sh
# Current local baseline
go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -models 'current|http://127.0.0.1:5021/v1/embeddings|Qwen3-Embedding-0.6B||'

# OpenAI, shortened to 1024 dimensions
go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -batch 128 \
  -models 'openai|https://api.openai.com/v1/embeddings|text-embedding-3-large|1024|OPENAI_API_KEY'

# OpenAI, native 3072 dimensions
go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -batch 128 \
  -models 'openai-3large-3072|https://api.openai.com/v1/embeddings|text-embedding-3-large||OPENAI_API_KEY'

# BGE-M3 served locally on port 5023
go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -batch 128 \
  -models 'bge-m3|http://127.0.0.1:5023/v1/embeddings|bge-m3||'
```

## Sidecar Table

Create the sidecar table by running the Memory Palace migration, or apply the
same statements from `api.go`:

```sql
CREATE TABLE IF NOT EXISTS memory_chunk_embeddings_3072 (
  chunk_id TEXT NOT NULL REFERENCES memory_chunks(id) ON DELETE CASCADE,
  user_key TEXT NOT NULL,
  model_key TEXT NOT NULL,
  model_name TEXT NOT NULL,
  embedding HALFVEC(3072) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (chunk_id, model_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_chunk_embeddings_3072_user_model
  ON memory_chunk_embeddings_3072(user_key, model_key);

CREATE INDEX IF NOT EXISTS idx_memory_chunk_embeddings_3072_hnsw_cosine
  ON memory_chunk_embeddings_3072 USING hnsw (embedding halfvec_cosine_ops);
```

The installed pgvector version limits HNSW indexes on the normal `vector`
type to 2,000 dimensions. `halfvec` supports 4,000 dimensions, so this
sidecar keeps the full 3,072 dimensions while allowing an HNSW cosine index.

Store OpenAI 3072d embeddings for the current comparison set:

```sh
cd memory_palace/src/go

go run ./cmd/embed-eval \
  -scope library \
  -collection naomi-novik \
  -batch 128 \
  -models 'openai-3large-3072|https://api.openai.com/v1/embeddings|text-embedding-3-large||OPENAI_API_KEY' \
  -model-key openai:text-embedding-3-large \
  -store-vector-type halfvec \
  -store
```

Store-only imports are resumable. Each completed batch commits independently,
and the runner skips rows already present for the chosen `model_key`.

Search through Postgres using the stored sidecar embeddings:

```sh
go run ./cmd/embed-eval \
  -query "$HOLDOUT_QUERY" \
  -target "$HOLDOUT_TARGET" \
  -scope library \
  -collection naomi-novik \
  -batch 128 \
  -models 'openai-3large-3072|https://api.openai.com/v1/embeddings|text-embedding-3-large||OPENAI_API_KEY' \
  -model-key openai:text-embedding-3-large \
  -store-vector-type halfvec \
  -stored-search
```

## Results: 2026-07-11

Dataset:

```text
user=local
scope=library
collection=naomi-novik
chunks=4980
target_chunks=7
```

Best target rank:

```text
text-embedding-3-large 3072d:  52
Qwen3-Embedding-0.6B:          79
text-embedding-3-large 1024d: 102
BAAI/bge-m3:                  169
```

The persisted `HALFVEC(3072)` sidecar returned the same best target rank as
the native model for the retired relationship diagnostic, with a 430 ms
database-backed search.

For that retired diagnostic, native `text-embedding-3-large` improved recall,
but the shortened 1024-dimensional OpenAI embedding and BGE-M3 were worse than
the current local Qwen embedding.

## Local Qwen 4B Check: 2026-07-11

`Qwen3-Embedding-4B-Q4_K_M` was run temporarily with llama.cpp on the RTX
3080 Ti. It produced 2,560-dimensional embeddings with all layers on the GPU.
Both Qwen 4B Q4 and the existing Qwen 0.6B placed the expected document at
rank 1 for all five questions in `search_eval.yaml`. The 4B full-corpus pass
took about 13.6 seconds for 330 chunks; 0.6B took about 1.7-2.1 seconds.
That suite does not currently justify reindexing production with the larger
model. Keep the downloaded Q4 model for broader, more discriminative testing.
