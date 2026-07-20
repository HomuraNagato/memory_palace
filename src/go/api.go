package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/lib/pq"
)

const (
	defaultDatabaseURL         = "host=100.73.65.14 port=5438 dbname=pensieve user=postgres password=postgres sslmode=disable"
	defaultEmbeddingURL        = "http://127.0.0.1:5021/v1/embeddings"
	defaultEmbeddingModel      = "Qwen3-Embedding-0.6B"
	defaultContextualizerURL   = "http://127.0.0.1:5014/v1/chat/completions"
	defaultContextualizerModel = "auto"
	defaultQueryRewriterURL    = "http://127.0.0.1:5014/v1/chat/completions"
	defaultQueryRewriterModel  = "auto"
	defaultRerankerURL         = "http://127.0.0.1:5022"
	defaultHTTPAddr            = ":8013"
	maxChunkRunes              = 700
	chunkOverlapRunes          = 120
	embeddingBatchSize         = 32
	embeddingBatchTokenBudget  = 1200
	embeddingMetadataRunes     = 500
	rawEmbeddingBackfillBatch  = 32
	libraryChunksPerDoc        = 6
	contentHashVersion         = "memory-palace-retrieval-context-v19-separate-parent-vectors"
	defaultMemoryScope         = "memories"
	searchModeSemantic         = "semantic"
	searchModeLiteral          = "literal"
	searchModeHybrid           = "hybrid"
)

type Server struct {
	db                      *sql.DB
	httpClient              *http.Client
	embeddingURL            string
	embeddingModel          string
	embeddingKey            string
	contextualizerURL       string
	contextualizerModel     string
	contextualizerBatchSize int
	queryRewriterURL        string
	queryRewriterModel      string
	queryRewriterTimeout    time.Duration
	rerankerURL             string
	watchDirs               map[string][]WatchRoot
	scanQueue               chan string
	scanJobs                map[string]*ScanJob
	scanJobsMu              sync.Mutex
	retrieveTraces          map[string]RetrieveTrace
	retrieveTraceMu         sync.Mutex
	resolvedModels          map[string]string
	resolvedModelsMu        sync.Mutex
	bm25                    *BM25Index
}

type WatchRoot struct {
	Path       string
	Scope      string
	Collection string
}

type epubContainer struct {
	Rootfiles []epubRootfile `xml:"rootfiles>rootfile"`
}

type epubRootfile struct {
	FullPath  string `xml:"full-path,attr"`
	MediaType string `xml:"media-type,attr"`
}

type epubPackage struct {
	Metadata epubMetadata  `xml:"metadata"`
	Manifest []epubItem    `xml:"manifest>item"`
	Spine    []epubItemRef `xml:"spine>itemref"`
}

type epubMetadata struct {
	Titles []string `xml:"title"`
}

type epubItem struct {
	ID        string `xml:"id,attr"`
	Href      string `xml:"href,attr"`
	MediaType string `xml:"media-type,attr"`
}

type epubItemRef struct {
	IDRef string `xml:"idref,attr"`
}

type MemoryItem struct {
	ID         string         `json:"id"`
	UserKey    string         `json:"user_key,omitempty"`
	Kind       string         `json:"kind"`
	SourceType string         `json:"source_type"`
	SourceURI  string         `json:"source_uri"`
	Title      string         `json:"title"`
	Body       string         `json:"body"`
	Tags       []string       `json:"tags"`
	Metadata   map[string]any `json:"metadata"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
}

type MemorySummary struct {
	ID         string         `json:"id"`
	Kind       string         `json:"kind"`
	SourceType string         `json:"source_type"`
	SourceURI  string         `json:"source_uri"`
	Title      string         `json:"title"`
	Preview    string         `json:"preview"`
	Tags       []string       `json:"tags"`
	Metadata   map[string]any `json:"metadata"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
}

type SearchResult struct {
	MemoryID      string         `json:"memory_id"`
	ChunkID       string         `json:"chunk_id"`
	Title         string         `json:"title"`
	Text          string         `json:"text"`
	ContextText   string         `json:"context_text,omitempty"`
	MetadataText  string         `json:"-"`
	Kind          string         `json:"kind"`
	SourceType    string         `json:"source_type"`
	SourceURI     string         `json:"source_uri"`
	Tags          []string       `json:"tags"`
	Metadata      map[string]any `json:"metadata"`
	ChunkMetadata map[string]any `json:"chunk_metadata"`
	ParentSummary string         `json:"-"`
	Score         float64        `json:"score"`
	ScoreParts    ScoreParts     `json:"score_parts"`
	Similarity    float64        `json:"similarity"`
	UpdatedAt     time.Time      `json:"updated_at"`
}

type textChunk struct {
	Text     string
	Metadata map[string]any
}

type ScoreParts struct {
	Mode             string  `json:"mode"`
	VectorSimilarity float64 `json:"vector_similarity"`
	LiteralScore     float64 `json:"literal_score"`
	TitleMatch       bool    `json:"title_match"`
	SourceMatch      bool    `json:"source_match"`
	BodyMatch        bool    `json:"body_match"`
}

type MemoryListResponse struct {
	Items []MemorySummary `json:"items"`
}

type UpsertMemoryRequest struct {
	Kind       string         `json:"kind"`
	SourceType string         `json:"source_type"`
	SourceURI  string         `json:"source_uri"`
	Title      string         `json:"title"`
	Body       string         `json:"body"`
	Tags       []string       `json:"tags"`
	Metadata   map[string]any `json:"metadata"`
}

type SearchRequest struct {
	RequestID           string    `json:"request_id,omitempty"`
	Query               string    `json:"query"`
	Kind                string    `json:"kind,omitempty"`
	SourceType          string    `json:"source_type,omitempty"`
	Scope               string    `json:"scope,omitempty"`
	Collection          string    `json:"collection,omitempty"`
	Limit               int       `json:"limit,omitempty"`
	Mode                string    `json:"mode,omitempty"`
	TimeFilter          string    `json:"time_filter,omitempty"`
	Tags                []string  `json:"tags,omitempty"`
	ChunkLimitPerSource int       `json:"chunk_limit_per_source,omitempty"`
	SourceTitles        []string  `json:"-"`
	Embedding           []float64 `json:"-"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
}

type RelatedMemoriesResponse struct {
	Items []MemorySummary `json:"items"`
}

type WatchRunResponse struct {
	Scanned    int `json:"scanned"`
	Upserts    int `json:"upserts"`
	Deleted    int `json:"deleted"`
	Skipped    int `json:"skipped"`
	Duplicates int `json:"duplicates"`
}

type ScanJob struct {
	ID        string            `json:"id"`
	UserKey   string            `json:"user_key,omitempty"`
	Status    string            `json:"status"`
	Result    *WatchRunResponse `json:"result,omitempty"`
	Error     string            `json:"error,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	StartedAt *time.Time        `json:"started_at,omitempty"`
	EndedAt   *time.Time        `json:"ended_at,omitempty"`
}

type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type EmbeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func main() {
	db, err := sql.Open("postgres", envOrDefault("MEMORY_PALACE_DATABASE_URL", defaultDatabaseURL))
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)

	if err := db.Ping(); err != nil {
		log.Fatalf("ping database: %v", err)
	}
	if err := migrate(db); err != nil {
		log.Fatalf("migrate database: %v", err)
	}

	server := &Server{
		db:                      db,
		httpClient:              &http.Client{Timeout: 45 * time.Second},
		embeddingURL:            envOrDefault("MEMORY_PALACE_EMBEDDING_URL", defaultEmbeddingURL),
		embeddingModel:          envOrDefault("MEMORY_PALACE_EMBEDDING_MODEL", defaultEmbeddingModel),
		embeddingKey:            strings.TrimSpace(os.Getenv("MEMORY_PALACE_EMBEDDING_API_KEY")),
		contextualizerURL:       envOrDefault("MEMORY_PALACE_CONTEXTUALIZER_URL", defaultContextualizerURL),
		contextualizerModel:     envOrDefault("MEMORY_PALACE_CONTEXTUALIZER_MODEL", defaultContextualizerModel),
		contextualizerBatchSize: parsePositiveInt(os.Getenv("MEMORY_PALACE_CONTEXTUALIZER_BATCH_SIZE"), 4),
		queryRewriterURL:        envOrDefault("MEMORY_PALACE_QUERY_REWRITER_URL", defaultQueryRewriterURL),
		queryRewriterModel:      envOrDefault("MEMORY_PALACE_QUERY_REWRITER_MODEL", defaultQueryRewriterModel),
		queryRewriterTimeout:    time.Duration(parsePositiveInt(os.Getenv("MEMORY_PALACE_QUERY_REWRITER_TIMEOUT_MS"), 4000)) * time.Millisecond,
		rerankerURL:             envOrDefault("MEMORY_PALACE_RERANKER_URL", defaultRerankerURL),
		watchDirs:               parseWatchDirs(os.Getenv("MEMORY_PALACE_WATCH_DIRS")),
		scanQueue:               make(chan string, 16),
		scanJobs:                map[string]*ScanJob{},
		retrieveTraces:          map[string]RetrieveTrace{},
		resolvedModels:          map[string]string{},
		bm25:                    NewBM25Index(),
	}
	if err := server.bm25.Load(context.Background(), db); err != nil {
		log.Fatalf("load BM25 index: %v", err)
	}
	if err := server.backfillParentEntityContext(context.Background()); err != nil {
		log.Fatalf("backfill parent entity context: %v", err)
	}
	go server.runScanWorker()
	go server.backfillRawEmbeddings()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/query", server.handleLegacyQuery)
	mux.HandleFunc("/v1/search", server.handleSearch)
	mux.HandleFunc("/v1/retrieve/traces/", server.handleRetrieveTrace)
	mux.HandleFunc("/v1/retrieve", server.handleRetrieve)
	mux.HandleFunc("/v1/memories", server.handleMemories)
	mux.HandleFunc("/v1/memories/", server.handleMemory)
	mux.HandleFunc("/v1/ingest/watch-dirs/run", server.handleWatchRun)
	mux.HandleFunc("/v1/ingest/watch-dirs/jobs", server.handleWatchJobs)
	mux.HandleFunc("/v1/ingest/watch-dirs/jobs/", server.handleWatchJob)

	addr := envOrDefault("MEMORY_PALACE_HTTP_ADDR", defaultHTTPAddr)
	log.Printf("memory_palace listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func migrate(db *sql.DB) error {
	statements := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE EXTENSION IF NOT EXISTS pg_trgm`,
		`CREATE TABLE IF NOT EXISTS memory_items (
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
		)`,
		`CREATE TABLE IF NOT EXISTS memory_chunks (
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
		)`,
		`CREATE TABLE IF NOT EXISTS memory_parent_nodes (
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
		)`,
		`CREATE TABLE IF NOT EXISTS memory_chunk_embeddings_3072 (
			chunk_id TEXT NOT NULL REFERENCES memory_chunks(id) ON DELETE CASCADE,
			user_key TEXT NOT NULL,
			model_key TEXT NOT NULL,
			model_name TEXT NOT NULL,
			embedding HALFVEC(3072) NOT NULL,
			metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY (chunk_id, model_key)
		)`,
		`ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS context_text TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS parent_id TEXT`,
		`ALTER TABLE memory_parent_nodes ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE memory_parent_nodes ADD COLUMN IF NOT EXISTS summary_embedding VECTOR(1024)`,
		`DO $$
		BEGIN
			IF EXISTS (
				SELECT 1 FROM information_schema.columns
				WHERE table_name = 'memory_chunks' AND column_name = 'retrieval_text'
			) AND NOT EXISTS (
				SELECT 1 FROM information_schema.columns
				WHERE table_name = 'memory_chunks' AND column_name = 'metadata_text'
			) THEN
				ALTER TABLE memory_chunks RENAME COLUMN retrieval_text TO metadata_text;
			END IF;
		END $$`,
		`ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS metadata_text TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS raw_embedding VECTOR(1024)`,
		`UPDATE memory_chunks SET metadata_text = text WHERE metadata_text = ''`,
		`DO $$
		BEGIN
			IF EXISTS (
				SELECT 1
				FROM pg_attribute a
				JOIN pg_class c ON c.oid = a.attrelid
				WHERE c.relname = 'memory_chunks'
					AND a.attname = 'embedding'
					AND a.atttypmod = -1
			) THEN
				ALTER TABLE memory_chunks
				ALTER COLUMN embedding TYPE VECTOR(1024)
				USING embedding::VECTOR(1024);
			END IF;
		END $$`,
		`CREATE TABLE IF NOT EXISTS memory_deletions (
			id TEXT PRIMARY KEY,
			user_key TEXT NOT NULL,
			memory_id TEXT NOT NULL,
			source_type TEXT NOT NULL DEFAULT '',
			source_uri TEXT NOT NULL DEFAULT '',
			title TEXT NOT NULL DEFAULT '',
			reason TEXT NOT NULL DEFAULT '',
			deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_user_updated ON memory_items(user_key, updated_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_source ON memory_items(user_key, source_type, source_uri)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_user ON memory_chunks(user_key)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_parent ON memory_chunks(user_key, parent_id, chunk_index)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_parent_nodes_user ON memory_parent_nodes(user_key, memory_id)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_parent_nodes_embedding_hnsw_cosine ON memory_parent_nodes USING hnsw (embedding vector_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_parent_nodes_summary_embedding_hnsw_cosine ON memory_parent_nodes USING hnsw (summary_embedding vector_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding_hnsw_cosine ON memory_chunks USING hnsw (embedding vector_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_raw_embedding_hnsw_cosine ON memory_chunks USING hnsw (raw_embedding vector_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunk_embeddings_3072_user_model ON memory_chunk_embeddings_3072(user_key, model_key)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunk_embeddings_3072_hnsw_cosine ON memory_chunk_embeddings_3072 USING hnsw (embedding halfvec_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_text_fts ON memory_chunks USING gin (to_tsvector('simple', text))`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_metadata_text_fts ON memory_chunks USING gin (to_tsvector('simple', metadata_text))`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_title_fts ON memory_items USING gin (to_tsvector('simple', title))`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_text_trgm ON memory_chunks USING gin (lower(text) gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_chunks_metadata_text_trgm ON memory_chunks USING gin (lower(metadata_text) gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_title_trgm ON memory_items USING gin (lower(title) gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_source_uri_trgm ON memory_items USING gin (lower(source_uri) gin_trgm_ops)`,
		`CREATE INDEX IF NOT EXISTS idx_memory_deletions_user_deleted ON memory_deletions(user_key, deleted_at DESC)`,
	}
	for _, statement := range statements {
		if _, err := db.Exec(statement); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":              true,
		"embedding_model": s.embeddingModel,
		"watch_users":     len(s.watchDirs),
	})
}

func (s *Server) handleLegacyQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	results, err := s.search(r.Context(), userKeyFromRequest(r), SearchRequest{Query: req.Query, Limit: req.Limit})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SearchResponse{Results: results})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req SearchRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	results, err := s.search(r.Context(), userKeyFromRequest(r), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SearchResponse{Results: results})
}

func (s *Server) handleMemories(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		userKey := userKeyFromRequest(r)
		query := strings.TrimSpace(r.URL.Query().Get("query"))
		if query != "" {
			limit := parseLimit(r.URL.Query().Get("limit"), 20)
			results, err := s.search(r.Context(), userKey, SearchRequest{
				Query:      query,
				Kind:       r.URL.Query().Get("kind"),
				SourceType: r.URL.Query().Get("source_type"),
				Scope:      r.URL.Query().Get("scope"),
				Collection: r.URL.Query().Get("collection"),
				Limit:      limit,
				Mode:       r.URL.Query().Get("mode"),
				TimeFilter: r.URL.Query().Get("time_filter"),
			})
			if err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			writeJSON(w, http.StatusOK, SearchResponse{Results: results})
			return
		}
		items, err := s.listMemories(r.Context(), userKey, r.URL.Query().Get("kind"), r.URL.Query().Get("source_type"), r.URL.Query().Get("scope"), r.URL.Query().Get("collection"), parseLimit(r.URL.Query().Get("limit"), 50))
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, MemoryListResponse{Items: items})
	case http.MethodPost:
		var req UpsertMemoryRequest
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		item, err := s.upsertMemory(r.Context(), userKeyFromRequest(r), req)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, item)
	default:
		writeError(w, http.StatusMethodNotAllowed, "unsupported method")
	}
}

func (s *Server) handleMemory(w http.ResponseWriter, r *http.Request) {
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/memories/"), "/")
	parts := strings.Split(path, "/")
	id := strings.TrimSpace(parts[0])
	if id == "" {
		writeError(w, http.StatusNotFound, "memory id required")
		return
	}
	userKey := userKeyFromRequest(r)
	if len(parts) > 1 && parts[1] == "related" {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "GET required")
			return
		}
		items, err := s.relatedMemories(r.Context(), userKey, id, parseLimit(r.URL.Query().Get("limit"), 6))
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, sql.ErrNoRows) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, RelatedMemoriesResponse{Items: items})
		return
	}
	switch r.Method {
	case http.MethodGet:
		item, err := s.getMemory(r.Context(), userKey, id)
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, sql.ErrNoRows) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, item)
	case http.MethodPatch:
		existing, err := s.getMemory(r.Context(), userKey, id)
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, sql.ErrNoRows) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		var req UpsertMemoryRequest
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		req.SourceType = firstNonEmpty(req.SourceType, existing.SourceType)
		req.SourceURI = firstNonEmpty(req.SourceURI, existing.SourceURI)
		req.Kind = firstNonEmpty(req.Kind, existing.Kind)
		req.Title = firstNonEmpty(req.Title, existing.Title)
		req.Body = firstNonEmpty(req.Body, existing.Body)
		if req.Tags == nil {
			req.Tags = existing.Tags
		}
		if req.Metadata == nil {
			req.Metadata = existing.Metadata
		}
		item, err := s.upsertMemoryWithID(r.Context(), userKey, id, req)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, item)
	case http.MethodDelete:
		if err := s.deleteMemory(r.Context(), userKey, id, "user_deleted"); err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, sql.ErrNoRows) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
	default:
		writeError(w, http.StatusMethodNotAllowed, "unsupported method")
	}
}

func (s *Server) handleWatchRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	userKey := userKeyFromRequest(r)
	response, err := s.runWatchDirs(r.Context(), userKey)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleWatchJobs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	userKey := userKeyFromRequest(r)
	if strings.TrimSpace(userKey) == "" {
		writeError(w, http.StatusBadRequest, "user key is required")
		return
	}
	if job := s.activeScanJobForUser(userKey); job != nil {
		writeJSON(w, http.StatusAccepted, job)
		return
	}
	job := &ScanJob{
		ID:        newID(),
		UserKey:   userKey,
		Status:    "queued",
		CreatedAt: time.Now().UTC(),
	}
	s.scanJobsMu.Lock()
	s.scanJobs[job.ID] = job
	s.scanJobsMu.Unlock()

	select {
	case s.scanQueue <- job.ID:
		writeJSON(w, http.StatusAccepted, cloneScanJob(job))
	default:
		s.updateScanJob(job.ID, func(job *ScanJob) {
			now := time.Now().UTC()
			job.Status = "failed"
			job.Error = "scan queue is full"
			job.EndedAt = &now
		})
		writeError(w, http.StatusTooManyRequests, "scan queue is full")
	}
}

func (s *Server) handleWatchJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	id := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/ingest/watch-dirs/jobs/"), "/")
	if id == "" {
		writeError(w, http.StatusNotFound, "scan job id required")
		return
	}
	job := s.getScanJob(id)
	if job == nil {
		writeError(w, http.StatusNotFound, "scan job not found")
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (s *Server) upsertMemory(ctx context.Context, userKey string, req UpsertMemoryRequest) (MemoryItem, error) {
	return s.upsertMemoryWithID(ctx, userKey, "", req)
}

func (s *Server) upsertMemoryWithID(ctx context.Context, userKey, id string, req UpsertMemoryRequest) (MemoryItem, error) {
	totalStart := time.Now()
	userKey = strings.TrimSpace(userKey)
	if userKey == "" {
		return MemoryItem{}, fmt.Errorf("user key is required")
	}
	req = normalizeUpsertRequest(req)
	if req.Body == "" {
		return MemoryItem{}, fmt.Errorf("body is required")
	}
	if id == "" {
		id = newID()
	}
	if req.SourceURI == "" {
		req.SourceURI = "memory://" + id
	}
	if req.Title == "" {
		req.Title = titleFromBody(req.Body)
	}
	contentHash := stringMapValue(req.Metadata, "content_hash")
	if contentHash == "" {
		contentHash = contentHashFor(req.Title, req.SourceURI, req.Body)
	}

	chunks := chunksForRequest(req)
	logIngestProgress := shouldLogIngestProgress(req)
	if logIngestProgress {
		log.Printf("memory.upsert status=start user=%q title=%q source_uri=%q path=%q chunks=%d body_chars=%d", userKey, req.Title, req.SourceURI, stringMapValue(req.Metadata, "path"), len(chunks), len([]rune(req.Body)))
	}
	contextStart := time.Now()
	contextMetadata := fallbackContextMetadata(req, chunks)
	contextMs := time.Since(contextStart).Milliseconds()
	embeddingInputs := make([]string, 0, len(chunks)*2)
	contextTexts := make([]string, len(chunks))
	metadataTexts := make([]string, len(chunks))
	parentNodes, chunkParentIDs := buildParentNodes(req, chunks)
	for i, chunk := range chunks {
		contextTexts[i] = chunkContextText(req, chunk, i)
		metadataTexts[i] = metadataTextForChunk(req, chunk, i, contextTexts[i], contextMetadata[i])
		embeddingInputs = append(embeddingInputs, embeddingTextForChunk(metadataTexts[i], chunk.Text))
	}
	for _, chunk := range chunks {
		embeddingInputs = append(embeddingInputs, rawEmbeddingInput(chunk.Text))
	}
	embedStart := time.Now()
	embeddings, err := s.embedBatched(ctx, embeddingInputs)
	embedMs := time.Since(embedStart).Milliseconds()
	if err != nil {
		return MemoryItem{}, err
	}
	if len(embeddings) != len(chunks)*2 {
		return MemoryItem{}, fmt.Errorf("embedding count mismatch: got %d want %d", len(embeddings), len(chunks)*2)
	}
	assignParentEmbeddings(parentNodes, embeddings[:len(chunks)])
	summaryStart := time.Now()
	parentSummaryCount, parentSummaryFallbacks := s.summarizeParentNodes(ctx, req, parentNodes)
	parentSummaryEmbeddings := s.embedParentSummaries(ctx, parentNodes)
	applyParentSummaryEmbeddings(parentNodes, parentSummaryEmbeddings)
	summaryMs := time.Since(summaryStart).Milliseconds()
	tagsJSON, err := json.Marshal(req.Tags)
	if err != nil {
		return MemoryItem{}, err
	}
	metadataJSON, err := json.Marshal(nonNilMap(req.Metadata))
	if err != nil {
		return MemoryItem{}, err
	}

	dbStart := time.Now()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return MemoryItem{}, err
	}
	defer tx.Rollback()

	var existingID string
	err = tx.QueryRowContext(ctx, `
		SELECT id FROM memory_items
		WHERE user_key = $1 AND source_type = $2 AND source_uri = $3
	`, userKey, req.SourceType, req.SourceURI).Scan(&existingID)
	if err == nil && id == "" {
		id = existingID
	} else if err == nil && existingID != id {
		id = existingID
	} else if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return MemoryItem{}, err
	}

	if _, err := tx.ExecContext(ctx, `
		INSERT INTO memory_items (id, user_key, kind, source_type, source_uri, title, body, tags, metadata, content_hash, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, NOW(), NOW())
		ON CONFLICT (user_key, source_type, source_uri) DO UPDATE SET
			kind = EXCLUDED.kind,
			title = EXCLUDED.title,
			body = EXCLUDED.body,
			tags = EXCLUDED.tags,
			metadata = EXCLUDED.metadata,
			content_hash = EXCLUDED.content_hash,
			updated_at = NOW()
	`, id, userKey, req.Kind, req.SourceType, req.SourceURI, req.Title, req.Body, string(tagsJSON), string(metadataJSON), contentHash); err != nil {
		return MemoryItem{}, err
	}
	if err := tx.QueryRowContext(ctx, `
		SELECT id FROM memory_items
		WHERE user_key = $1 AND source_type = $2 AND source_uri = $3
	`, userKey, req.SourceType, req.SourceURI).Scan(&id); err != nil {
		return MemoryItem{}, err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM memory_chunks WHERE memory_id = $1 AND user_key = $2`, id, userKey); err != nil {
		return MemoryItem{}, err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM memory_parent_nodes WHERE memory_id = $1 AND user_key = $2`, id, userKey); err != nil {
		return MemoryItem{}, err
	}
	for _, parent := range parentNodes {
		parentMetadataJSON, err := json.Marshal(parent.Metadata)
		if err != nil {
			return MemoryItem{}, err
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO memory_parent_nodes (id, memory_id, user_key, node_type, node_key, title, text, metadata_text, summary, embedding, summary_embedding, child_count, metadata, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11::vector, $12, $13::jsonb, NOW())
		`, parent.ID, id, userKey, parent.NodeType, parent.NodeKey, req.Title, parent.Text, parent.MetadataText, parent.Summary, vectorLiteral(parent.Embedding), nullableVectorLiteral(parent.SummaryEmbedding), len(parent.ChunkIndexes), string(parentMetadataJSON)); err != nil {
			return MemoryItem{}, err
		}
	}
	for i, chunk := range chunks {
		chunkMetadataJSON, err := json.Marshal(chunk.Metadata)
		if err != nil {
			return MemoryItem{}, err
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO memory_chunks (id, memory_id, user_key, chunk_index, text, context_text, metadata_text, parent_id, embedding, raw_embedding, metadata, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::vector, $11::jsonb, NOW())
		`, newID(), id, userKey, i, chunk.Text, contextTexts[i], metadataTexts[i], chunkParentIDs[i], vectorLiteral(embeddings[i]), vectorLiteral(embeddings[len(chunks)+i]), string(chunkMetadataJSON)); err != nil {
			return MemoryItem{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return MemoryItem{}, err
	}
	if s.bm25 != nil {
		if err := s.bm25.RefreshMemory(ctx, s.db, userKey, id); err != nil {
			log.Printf("memory.bm25 status=refresh_error user=%q memory_id=%q err=%q", userKey, id, err.Error())
		}
	}
	if logIngestProgress {
		log.Printf("memory.upsert status=ok user=%q title=%q source_uri=%q chunks=%d parent_summaries=%d parent_summary_fallbacks=%d contextualize_ms=%d embed_ms=%d summary_ms=%d db_ms=%d total_ms=%d", userKey, req.Title, req.SourceURI, len(chunks), parentSummaryCount, parentSummaryFallbacks, contextMs, embedMs, summaryMs, time.Since(dbStart).Milliseconds(), time.Since(totalStart).Milliseconds())
	}
	return s.getMemory(ctx, userKey, id)
}

func (s *Server) getMemory(ctx context.Context, userKey, id string) (MemoryItem, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, user_key, kind, source_type, source_uri, title, body, tags::text, metadata::text, created_at, updated_at
		FROM memory_items
		WHERE user_key = $1 AND id = $2
	`, userKey, id)
	return scanMemoryItem(row)
}

type rawEmbeddingChunk struct {
	ID   string
	Text string
}

func (s *Server) backfillRawEmbeddings() {
	start := time.Now()
	processed := 0
	log.Printf("memory.raw_embedding_backfill status=start batch=%d", rawEmbeddingBackfillBatch)
	for {
		rows, err := s.db.QueryContext(context.Background(), `
			SELECT id, text
			FROM memory_chunks
			WHERE raw_embedding IS NULL
			ORDER BY id
			LIMIT $1
		`, rawEmbeddingBackfillBatch)
		if err != nil {
			log.Printf("memory.raw_embedding_backfill status=error phase=load processed=%d err=%q", processed, err.Error())
			return
		}
		chunks := make([]rawEmbeddingChunk, 0, rawEmbeddingBackfillBatch)
		for rows.Next() {
			var chunk rawEmbeddingChunk
			if err := rows.Scan(&chunk.ID, &chunk.Text); err != nil {
				rows.Close()
				log.Printf("memory.raw_embedding_backfill status=error phase=scan processed=%d err=%q", processed, err.Error())
				return
			}
			chunks = append(chunks, chunk)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			log.Printf("memory.raw_embedding_backfill status=error phase=rows processed=%d err=%q", processed, err.Error())
			return
		}
		rows.Close()
		if len(chunks) == 0 {
			log.Printf("memory.raw_embedding_backfill status=done processed=%d total_ms=%d", processed, time.Since(start).Milliseconds())
			return
		}

		inputs := make([]string, len(chunks))
		for i, chunk := range chunks {
			inputs[i] = rawEmbeddingInput(chunk.Text)
		}
		embeddings, err := s.embedBatched(context.Background(), inputs)
		if err != nil {
			log.Printf("memory.raw_embedding_backfill status=error phase=embedding processed=%d batch=%d err=%q", processed, len(chunks), err.Error())
			return
		}
		if len(embeddings) != len(chunks) {
			log.Printf("memory.raw_embedding_backfill status=error phase=count processed=%d got=%d want=%d", processed, len(embeddings), len(chunks))
			return
		}
		for i, chunk := range chunks {
			if _, err := s.db.ExecContext(context.Background(), `
				UPDATE memory_chunks
				SET raw_embedding = $2::vector
				WHERE id = $1 AND raw_embedding IS NULL
			`, chunk.ID, vectorLiteral(embeddings[i])); err != nil {
				log.Printf("memory.raw_embedding_backfill status=error phase=update processed=%d chunk_id=%q err=%q", processed, chunk.ID, err.Error())
				return
			}
		}
		processed += len(chunks)
		if processed%512 == 0 {
			log.Printf("memory.raw_embedding_backfill status=progress processed=%d total_ms=%d", processed, time.Since(start).Milliseconds())
		}
	}
}

func (s *Server) listMemories(ctx context.Context, userKey, kind, sourceType, scope, collection string, limit int) ([]MemorySummary, error) {
	limit = clampLimit(limit)
	scope = normalizeMetadataFilter(scope)
	collection = normalizeMetadataFilter(collection)
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, kind, source_type, source_uri, title, body, tags::text, metadata::text, created_at, updated_at
		FROM memory_items
		WHERE user_key = $1
			AND ($2 = '' OR kind = $2)
			AND ($3 = '' OR source_type = $3)
			AND ($4 = '' OR metadata->>'scope' = $4)
			AND ($5 = '' OR metadata->>'collection' = $5)
			AND source_uri NOT LIKE '%/.trash/%'
		ORDER BY updated_at DESC
		LIMIT $6
	`, userKey, strings.TrimSpace(kind), strings.TrimSpace(sourceType), scope, collection, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []MemorySummary{}
	for rows.Next() {
		var item MemorySummary
		var body, tagsRaw, metadataRaw string
		if err := rows.Scan(&item.ID, &item.Kind, &item.SourceType, &item.SourceURI, &item.Title, &body, &tagsRaw, &metadataRaw, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		item.Preview = preview(body)
		item.Tags = parseStringSlice(tagsRaw)
		item.Metadata = parseMap(metadataRaw)
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Server) relatedMemories(ctx context.Context, userKey, id string, limit int) ([]MemorySummary, error) {
	item, err := s.getMemory(ctx, userKey, id)
	if err != nil {
		return nil, err
	}
	folderPrefix := sourceFolderPrefix(item.SourceURI)
	if folderPrefix == "" {
		return []MemorySummary{}, nil
	}
	scope := stringMapValue(item.Metadata, "scope")
	collection := stringMapValue(item.Metadata, "collection")
	limit = clampLimit(limit)
	sideLimit := limit / 2
	if sideLimit < 1 {
		sideLimit = 1
	}
	rows, err := s.db.QueryContext(ctx, `
		WITH related AS (
			(SELECT id, kind, source_type, source_uri, title, body, tags::text, metadata::text, created_at, updated_at, 1 AS side
			FROM memory_items
			WHERE user_key = $1 AND id <> $2 AND source_type = $3 AND source_uri LIKE $4 AND source_uri < $5
				AND source_uri NOT LIKE '%/.trash/%'
				AND ($8 = '' OR metadata->>'scope' = $8)
				AND ($9 = '' OR metadata->>'collection' = $9)
			ORDER BY source_uri DESC
			LIMIT $6)
			UNION ALL
			(SELECT id, kind, source_type, source_uri, title, body, tags::text, metadata::text, created_at, updated_at, 2 AS side
			FROM memory_items
			WHERE user_key = $1 AND id <> $2 AND source_type = $3 AND source_uri LIKE $4 AND source_uri > $5
				AND source_uri NOT LIKE '%/.trash/%'
				AND ($8 = '' OR metadata->>'scope' = $8)
				AND ($9 = '' OR metadata->>'collection' = $9)
			ORDER BY source_uri ASC
			LIMIT $6)
		)
		SELECT id, kind, source_type, source_uri, title, body, tags, metadata, created_at, updated_at
		FROM related
		ORDER BY side, source_uri
		LIMIT $7
	`, userKey, id, item.SourceType, folderPrefix+"/%", item.SourceURI, sideLimit, limit, scope, collection)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []MemorySummary{}
	for rows.Next() {
		var item MemorySummary
		var body, tagsRaw, metadataRaw string
		if err := rows.Scan(&item.ID, &item.Kind, &item.SourceType, &item.SourceURI, &item.Title, &body, &tagsRaw, &metadataRaw, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		item.Preview = preview(body)
		item.Tags = parseStringSlice(tagsRaw)
		item.Metadata = parseMap(metadataRaw)
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Server) search(ctx context.Context, userKey string, req SearchRequest) ([]SearchResult, error) {
	userKey = strings.TrimSpace(userKey)
	query := strings.TrimSpace(req.Query)
	if userKey == "" {
		return nil, fmt.Errorf("user key is required")
	}
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	limit := clampLimit(req.Limit)
	mode := normalizeSearchMode(req.Mode)
	if mode == searchModeLiteral {
		return s.searchLiteral(ctx, userKey, query, req, limit)
	}
	return s.searchVector(ctx, userKey, query, req, limit, mode)
}

func (s *Server) searchVector(ctx context.Context, userKey, query string, req SearchRequest, limit int, mode string) ([]SearchResult, error) {
	totalStart := time.Now()
	embedStart := time.Now()
	embedding := req.Embedding
	var err error
	if len(embedding) == 0 {
		embedding, err = s.embedOne(ctx, query, req.RequestID)
	}
	embedMs := time.Since(embedStart).Milliseconds()
	if err != nil {
		log.Printf("memory.search_vector request_id=%q status=error phase=embedding mode=%q query=%q query_chars=%d embed_ms=%d err=%q", req.RequestID, mode, query, len([]rune(query)), embedMs, err.Error())
		return nil, err
	}
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)
	resultRankLimit := searchResultRankLimit(req)
	vectorPoolLimit := vectorCandidatePoolLimit(limit)
	scoreExpression := "vector_similarity"
	if mode == searchModeHybrid {
		scoreExpression = "(vector_similarity * 0.70) + (literal_score * 0.30)"
	}
	dbStart := time.Now()
	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
		WITH nearest AS (
			SELECT c.memory_id, c.id AS chunk_id, i.title, c.text, c.context_text, c.metadata_text, i.kind, i.source_type, i.source_uri,
				i.tags::text AS tags, i.metadata::text AS metadata, c.metadata::text AS chunk_metadata,
				1 - (c.embedding <=> $2::vector) AS vector_similarity,
				i.updated_at
			FROM memory_chunks c
			JOIN memory_items i ON i.id = c.memory_id
			WHERE c.user_key = $1
				AND ($3 = '' OR i.kind = $3)
				AND ($4 = '' OR i.source_type = $4)
				AND ($6 = '' OR i.source_uri LIKE '%%' || $6 || '%%' OR i.title LIKE $6 || '%%' OR i.metadata->>'modified_at' LIKE $6 || '%%')
				AND ($7 = '' OR i.metadata->>'scope' = $7)
				AND ($8 = '' OR i.metadata->>'collection' = $8)
				AND (cardinality($9::text[]) = 0 OR i.title = ANY($9))
				AND i.source_uri NOT LIKE '%%/.trash/%%'
			ORDER BY c.embedding <=> $2::vector
			LIMIT $12
		),
		base AS (
			SELECT *,
				lower(title) LIKE '%%' || lower($5) || '%%' AS title_match,
				lower(source_uri) LIKE '%%' || lower($5) || '%%' AS source_match,
				lower(metadata_text) LIKE '%%' || lower($5) || '%%' AS body_match,
				ts_rank_cd(
					to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(metadata_text, '')),
					plainto_tsquery('simple', $5)
				) AS text_rank
			FROM nearest
		),
		scored AS (
			SELECT *,
				CASE
					WHEN lower(trim(metadata_text)) = lower(trim($5)) THEN 1.0
					WHEN lower(trim(title)) = lower(trim($5)) THEN 0.95
					WHEN body_match THEN 0.85
					WHEN title_match THEN 0.75
					WHEN text_rank > 0 THEN LEAST(0.70, text_rank * 8.0)
					WHEN source_match THEN 0.55
					ELSE 0.0
				END AS literal_score
			FROM base
		),
		ranked AS (
			SELECT *,
				%s AS score,
				ROW_NUMBER() OVER (
					PARTITION BY memory_id
					ORDER BY %s DESC, vector_similarity DESC, updated_at DESC
				) AS result_rank
			FROM scored
		)
		SELECT memory_id, chunk_id, title, text, context_text, metadata_text, kind, source_type, source_uri, tags, metadata, chunk_metadata,
			score, vector_similarity, literal_score, title_match, source_match, body_match, updated_at
		FROM ranked
				WHERE result_rank <= $11
			ORDER BY score DESC
			LIMIT $10
		`, scoreExpression, scoreExpression), userKey, vectorLiteral(embedding), strings.TrimSpace(req.Kind), strings.TrimSpace(req.SourceType), query, timeFilter, scope, collection, pq.Array(req.SourceTitles), limit, resultRankLimit, vectorPoolLimit)
	if err != nil {
		log.Printf("memory.search_vector request_id=%q status=error phase=db mode=%q query=%q query_chars=%d embed_ms=%d db_ms=%d err=%q", req.RequestID, mode, query, len([]rune(query)), embedMs, time.Since(dbStart).Milliseconds(), err.Error())
		return nil, err
	}
	results, err := scanSearchResults(rows, mode)
	dbMs := time.Since(dbStart).Milliseconds()
	if err != nil {
		log.Printf("memory.search_vector request_id=%q status=error phase=scan mode=%q query=%q query_chars=%d embed_ms=%d db_ms=%d total_ms=%d err=%q", req.RequestID, mode, query, len([]rune(query)), embedMs, dbMs, time.Since(totalStart).Milliseconds(), err.Error())
		return nil, err
	}
	log.Printf("memory.search_vector request_id=%q status=ok mode=%q query=%q query_chars=%d results=%d embed_ms=%d db_ms=%d total_ms=%d", req.RequestID, mode, query, len([]rune(query)), len(results), embedMs, dbMs, time.Since(totalStart).Milliseconds())
	return results, nil
}

func (s *Server) searchLiteral(ctx context.Context, userKey, query string, req SearchRequest, limit int) ([]SearchResult, error) {
	totalStart := time.Now()
	if s.bm25 != nil {
		req.Query = query
		results := s.bm25.Search(userKey, req, limit)
		log.Printf("memory.search_bm25 request_id=%q status=ok query=%q query_chars=%d results=%d total_ms=%d", req.RequestID, query, len([]rune(query)), len(results), time.Since(totalStart).Milliseconds())
		return results, nil
	}
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)
	resultRankLimit := searchResultRankLimit(req)
	recallTerms := literalRecallTerms(query)
	recallQuery := literalRecallTSQuery(recallTerms)
	recallTermString := strings.Join(recallTerms, " ")
	dbStart := time.Now()
	rows, err := s.db.QueryContext(ctx, `
		WITH query_terms AS (
			SELECT
				lower($2) AS lowered,
				plainto_tsquery('simple', $2) AS strict_tsq,
				CASE WHEN $10 = '' THEN NULL ELSE to_tsquery('simple', $10) END AS recall_tsq
		),
		candidate_chunks AS (
			SELECT c.id
			FROM memory_chunks c
			CROSS JOIN query_terms q
			WHERE c.user_key = $1
				AND (
					to_tsvector('simple', c.metadata_text) @@ q.strict_tsq
					OR (q.recall_tsq IS NOT NULL AND to_tsvector('simple', c.metadata_text) @@ q.recall_tsq)
					OR lower(c.metadata_text) LIKE '%' || q.lowered || '%'
				)
			UNION
			SELECT c.id
			FROM memory_items i
			JOIN memory_chunks c ON c.memory_id = i.id
			CROSS JOIN query_terms q
			WHERE i.user_key = $1
				AND i.source_uri NOT LIKE '%/.trash/%'
				AND c.chunk_index < $12
				AND (
					to_tsvector('simple', i.title) @@ q.strict_tsq
					OR (q.recall_tsq IS NOT NULL AND to_tsvector('simple', i.title) @@ q.recall_tsq)
					OR lower(i.title) LIKE '%' || q.lowered || '%'
					OR lower(i.source_uri) LIKE '%' || q.lowered || '%'
				)
		),
		base AS (
			SELECT c.memory_id, c.id AS chunk_id, i.title, c.text, c.context_text, c.metadata_text, i.kind, i.source_type, i.source_uri,
				i.tags::text AS tags, i.metadata::text AS metadata, c.metadata::text AS chunk_metadata,
				lower(i.title) LIKE '%' || q.lowered || '%' AS title_match,
				lower(i.source_uri) LIKE '%' || q.lowered || '%' AS source_match,
				lower(c.metadata_text) LIKE '%' || q.lowered || '%' AS body_match,
				ts_rank_cd(
					to_tsvector('simple', coalesce(i.title, '') || ' ' || coalesce(c.metadata_text, '')),
					q.strict_tsq
				) AS strict_rank,
				CASE
					WHEN q.recall_tsq IS NULL THEN 0.0
					ELSE ts_rank_cd(to_tsvector('simple', coalesce(i.title, '')), q.recall_tsq)
				END AS title_recall_rank,
				CASE
					WHEN q.recall_tsq IS NULL THEN 0.0
					ELSE ts_rank_cd(to_tsvector('simple', coalesce(i.source_uri, '')), q.recall_tsq)
				END AS source_recall_rank,
				CASE
					WHEN q.recall_tsq IS NULL THEN 0.0
					ELSE ts_rank_cd(
						to_tsvector('simple', coalesce(c.metadata_text, '')),
						q.recall_tsq
					)
				END AS recall_rank,
				(
					SELECT COUNT(*)
					FROM regexp_split_to_table($11, '\s+') AS term
					WHERE term <> ''
						AND to_tsvector('simple', coalesce(c.metadata_text, '')) @@ plainto_tsquery('simple', term)
				) AS body_term_matches,
				i.updated_at
			FROM candidate_chunks matched
			JOIN memory_chunks c ON c.id = matched.id
			JOIN memory_items i ON i.id = c.memory_id
			CROSS JOIN query_terms q
			WHERE c.user_key = $1
				AND ($3 = '' OR i.kind = $3)
				AND ($4 = '' OR i.source_type = $4)
				AND ($5 = '' OR i.source_uri LIKE '%' || $5 || '%' OR i.title LIKE $5 || '%' OR i.metadata->>'modified_at' LIKE $5 || '%')
				AND ($6 = '' OR i.metadata->>'scope' = $6)
				AND ($7 = '' OR i.metadata->>'collection' = $7)
				AND i.source_uri NOT LIKE '%/.trash/%'
		),
		scored AS (
			SELECT *,
				CASE
					WHEN lower(trim(metadata_text)) = lower(trim($2)) THEN 1.0
					WHEN lower(trim(title)) = lower(trim($2)) THEN 0.95
					WHEN body_match THEN 0.85
					WHEN title_match THEN 0.75
					WHEN strict_rank > 0 THEN LEAST(0.78, strict_rank * 8.0)
					WHEN title_recall_rank > 0 OR source_recall_rank > 0 OR recall_rank > 0 THEN LEAST(
						0.72,
						CASE WHEN title_recall_rank > 0 THEN 0.35 ELSE 0.0 END +
						CASE WHEN source_recall_rank > 0 THEN 0.05 ELSE 0.0 END +
						LEAST(0.10, title_recall_rank * 4.0) +
						LEAST(0.08, recall_rank * 1.0) +
						LEAST(0.22, body_term_matches * 0.11)
					)
					WHEN source_match THEN 0.55
					ELSE 0.0
				END AS literal_score
			FROM base
		),
		ranked AS (
			SELECT *,
				literal_score AS score,
				ROW_NUMBER() OVER (
					PARTITION BY memory_id
					ORDER BY literal_score DESC, updated_at DESC
				) AS result_rank
			FROM scored
		)
		SELECT memory_id, chunk_id, title, text, context_text, metadata_text, kind, source_type, source_uri, tags, metadata, chunk_metadata,
			score, 0.0 AS vector_similarity, literal_score, title_match, source_match, body_match, updated_at
		FROM ranked
		WHERE literal_score > 0 AND result_rank <= $9
		ORDER BY score DESC, updated_at DESC
		LIMIT $8
	`, userKey, query, strings.TrimSpace(req.Kind), strings.TrimSpace(req.SourceType), timeFilter, scope, collection, limit, resultRankLimit, recallQuery, recallTermString, titleSeedChunks)
	if err != nil {
		log.Printf("memory.search_literal request_id=%q status=error phase=db query=%q query_chars=%d db_ms=%d err=%q", req.RequestID, query, len([]rune(query)), time.Since(dbStart).Milliseconds(), err.Error())
		return nil, err
	}
	results, err := scanSearchResults(rows, searchModeLiteral)
	dbMs := time.Since(dbStart).Milliseconds()
	if err != nil {
		log.Printf("memory.search_literal request_id=%q status=error phase=scan query=%q query_chars=%d db_ms=%d total_ms=%d err=%q", req.RequestID, query, len([]rune(query)), dbMs, time.Since(totalStart).Milliseconds(), err.Error())
		return nil, err
	}
	log.Printf("memory.search_literal request_id=%q status=ok query=%q query_chars=%d results=%d db_ms=%d total_ms=%d", req.RequestID, query, len([]rune(query)), len(results), dbMs, time.Since(totalStart).Milliseconds())
	return results, nil
}

func scanSearchResults(rows *sql.Rows, mode string) ([]SearchResult, error) {
	defer rows.Close()
	results := []SearchResult{}
	for rows.Next() {
		var result SearchResult
		var tagsRaw, metadataRaw, chunkMetadataRaw string
		result.ScoreParts.Mode = mode
		if err := rows.Scan(
			&result.MemoryID,
			&result.ChunkID,
			&result.Title,
			&result.Text,
			&result.ContextText,
			&result.MetadataText,
			&result.Kind,
			&result.SourceType,
			&result.SourceURI,
			&tagsRaw,
			&metadataRaw,
			&chunkMetadataRaw,
			&result.Score,
			&result.ScoreParts.VectorSimilarity,
			&result.ScoreParts.LiteralScore,
			&result.ScoreParts.TitleMatch,
			&result.ScoreParts.SourceMatch,
			&result.ScoreParts.BodyMatch,
			&result.UpdatedAt,
		); err != nil {
			return nil, err
		}
		result.Similarity = result.ScoreParts.VectorSimilarity
		result.Tags = parseStringSlice(tagsRaw)
		result.Metadata = parseMap(metadataRaw)
		result.ChunkMetadata = parseMap(chunkMetadataRaw)
		results = append(results, result)
	}
	return results, rows.Err()
}

func (s *Server) runScanWorker() {
	for id := range s.scanQueue {
		job := s.getScanJob(id)
		if job == nil {
			continue
		}
		now := time.Now().UTC()
		s.updateScanJob(id, func(job *ScanJob) {
			job.Status = "running"
			job.StartedAt = &now
		})
		result, err := s.runWatchDirs(context.Background(), job.UserKey)
		endedAt := time.Now().UTC()
		s.updateScanJob(id, func(job *ScanJob) {
			job.EndedAt = &endedAt
			if err != nil {
				job.Status = "failed"
				job.Error = err.Error()
				return
			}
			job.Status = "succeeded"
			job.Result = &result
		})
	}
}

func (s *Server) activeScanJobForUser(userKey string) *ScanJob {
	userKey = strings.TrimSpace(userKey)
	s.scanJobsMu.Lock()
	defer s.scanJobsMu.Unlock()
	for _, job := range s.scanJobs {
		if job.UserKey == userKey && (job.Status == "queued" || job.Status == "running") {
			return cloneScanJob(job)
		}
	}
	return nil
}

func (s *Server) getScanJob(id string) *ScanJob {
	s.scanJobsMu.Lock()
	defer s.scanJobsMu.Unlock()
	return cloneScanJob(s.scanJobs[id])
}

func (s *Server) updateScanJob(id string, update func(*ScanJob)) {
	s.scanJobsMu.Lock()
	defer s.scanJobsMu.Unlock()
	if job := s.scanJobs[id]; job != nil {
		update(job)
	}
}

func cloneScanJob(job *ScanJob) *ScanJob {
	if job == nil {
		return nil
	}
	clone := *job
	if job.Result != nil {
		result := *job.Result
		clone.Result = &result
	}
	if job.StartedAt != nil {
		startedAt := *job.StartedAt
		clone.StartedAt = &startedAt
	}
	if job.EndedAt != nil {
		endedAt := *job.EndedAt
		clone.EndedAt = &endedAt
	}
	return &clone
}

func (s *Server) deleteMemory(ctx context.Context, userKey, id, reason string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var sourceType, sourceURI, title string
	if err := tx.QueryRowContext(ctx, `
		SELECT source_type, source_uri, title
		FROM memory_items
		WHERE user_key = $1 AND id = $2
	`, userKey, id).Scan(&sourceType, &sourceURI, &title); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `
		INSERT INTO memory_deletions (id, user_key, memory_id, source_type, source_uri, title, reason, deleted_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
	`, newID(), userKey, id, sourceType, sourceURI, title, reason); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM memory_items WHERE user_key = $1 AND id = $2`, userKey, id); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	if s.bm25 != nil {
		s.bm25.RemoveMemory(id)
	}
	return nil
}

func (s *Server) runWatchDirs(ctx context.Context, userKey string) (WatchRunResponse, error) {
	roots := s.watchDirs[userKey]
	if len(roots) == 0 {
		return WatchRunResponse{}, fmt.Errorf("no watched directories configured for %s", userKey)
	}
	scanStart := time.Now()
	log.Printf("memory.watch_scan status=start user=%q roots=%d", userKey, len(roots))
	seen := map[string]bool{}
	response := WatchRunResponse{}
	for _, root := range roots {
		root.Path = filepath.Clean(root.Path)
		rootStart := time.Now()
		scannedBefore := response.Scanned
		upsertsBefore := response.Upserts
		skippedBefore := response.Skipped
		duplicatesBefore := response.Duplicates
		deletedBefore := response.Deleted
		log.Printf("memory.watch_root status=start user=%q path=%q scope=%q collection=%q", userKey, root.Path, root.Scope, root.Collection)
		err := filepath.WalkDir(root.Path, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				response.Skipped++
				return nil
			}
			if entry.IsDir() {
				if ignoredWatchDirectory(entry.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if !supportedWatchFile(path) {
				response.Skipped++
				return nil
			}
			response.Scanned++
			abs, err := filepath.Abs(path)
			if err != nil {
				response.Skipped++
				return nil
			}
			sourceURI := "file://" + abs
			seen[sourceURI] = true
			info, err := entry.Info()
			if err != nil {
				response.Skipped++
				return nil
			}
			body, fileMetadata, err := readWatchFile(ctx, abs)
			if err != nil {
				response.Skipped++
				return nil
			}
			contentHash := contentHashFor(filepath.Base(abs), sourceURI, body, root.Scope, root.Collection)
			contentFingerprint := contentFingerprintFor(body)
			var existingID, existingHash string
			err = s.db.QueryRowContext(ctx, `
				SELECT id, content_hash FROM memory_items
				WHERE user_key = $1 AND source_type = 'watch_file' AND source_uri = $2
			`, userKey, sourceURI).Scan(&existingID, &existingHash)
			if err == nil && existingHash == contentHash {
				response.Skipped++
				return nil
			}
			if err != nil && !errors.Is(err, sql.ErrNoRows) {
				response.Skipped++
				return nil
			}
			duplicateID, err := s.duplicateContentMemoryID(ctx, userKey, sourceURI, contentFingerprint)
			if err != nil {
				response.Skipped++
				return nil
			}
			if duplicateID != "" {
				response.Duplicates++
				response.Skipped++
				if existingID != "" {
					if err := s.deleteMemory(ctx, userKey, existingID, "duplicate_content_hash"); err == nil {
						response.Deleted++
					}
				}
				return nil
			}
			metadata := map[string]any{
				"path":                abs,
				"size":                info.Size(),
				"modified_at":         info.ModTime().Format(time.RFC3339),
				"watched_root":        root.Path,
				"scope":               root.Scope,
				"collection":          root.Collection,
				"content_hash":        contentHash,
				"content_fingerprint": contentFingerprint,
				"file_ext":            strings.ToLower(filepath.Ext(abs)),
			}
			for key, value := range fileMetadata {
				metadata[key] = value
			}
			title := filepath.Base(abs)
			if bookTitle := stringMapValue(metadata, "book_title"); bookTitle != "" {
				title = bookTitle
			}
			_, err = s.upsertMemory(ctx, userKey, UpsertMemoryRequest{
				Kind:       "document",
				SourceType: "watch_file",
				SourceURI:  sourceURI,
				Title:      title,
				Body:       body,
				Tags:       []string{"watched"},
				Metadata:   metadata,
			})
			if err != nil {
				log.Printf("memory.watch_file status=error phase=upsert user=%q path=%q title=%q err=%q", userKey, abs, title, err.Error())
				response.Skipped++
				return nil
			}
			response.Upserts++
			return nil
		})
		if err != nil {
			return response, err
		}
		log.Printf("memory.watch_root status=done user=%q path=%q scope=%q collection=%q scanned=%d upserts=%d skipped=%d duplicates=%d deleted=%d ms=%d", userKey, root.Path, root.Scope, root.Collection, response.Scanned-scannedBefore, response.Upserts-upsertsBefore, response.Skipped-skippedBefore, response.Duplicates-duplicatesBefore, response.Deleted-deletedBefore, time.Since(rootStart).Milliseconds())
	}

	existing, err := s.listWatchedSourceURIs(ctx, userKey)
	if err != nil {
		return response, err
	}
	for sourceURI, id := range existing {
		if seen[sourceURI] {
			continue
		}
		if err := s.deleteMemory(ctx, userKey, id, "watch_file_removed"); err == nil {
			response.Deleted++
		}
	}
	log.Printf("memory.watch_scan status=done user=%q scanned=%d upserts=%d skipped=%d duplicates=%d deleted=%d ms=%d", userKey, response.Scanned, response.Upserts, response.Skipped, response.Duplicates, response.Deleted, time.Since(scanStart).Milliseconds())
	return response, nil
}

func (s *Server) listWatchedSourceURIs(ctx context.Context, userKey string) (map[string]string, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT source_uri, id
		FROM memory_items
		WHERE user_key = $1 AND source_type = 'watch_file'
	`, userKey)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := map[string]string{}
	for rows.Next() {
		var sourceURI, id string
		if err := rows.Scan(&sourceURI, &id); err != nil {
			return nil, err
		}
		items[sourceURI] = id
	}
	return items, rows.Err()
}

func (s *Server) duplicateContentMemoryID(ctx context.Context, userKey, sourceURI, contentFingerprint string) (string, error) {
	if strings.TrimSpace(contentFingerprint) == "" {
		return "", nil
	}
	var id string
	err := s.db.QueryRowContext(ctx, `
		SELECT id
		FROM memory_items
		WHERE user_key = $1
			AND source_type = 'watch_file'
			AND source_uri <> $2
			AND metadata->>'content_fingerprint' = $3
		ORDER BY updated_at ASC, source_uri ASC
		LIMIT 1
	`, userKey, sourceURI, contentFingerprint).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return id, nil
}

func readWatchFile(ctx context.Context, path string) (string, map[string]any, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".pdf":
		return extractPDFText(ctx, path)
	case ".epub":
		return extractEPUBText(path)
	default:
		bodyBytes, err := os.ReadFile(path)
		if err != nil {
			return "", nil, err
		}
		return strings.TrimSpace(string(bodyBytes)), map[string]any{}, nil
	}
}

func extractPDFText(ctx context.Context, path string) (string, map[string]any, error) {
	command := exec.CommandContext(ctx, "pdftotext", "-layout", "-enc", "UTF-8", path, "-")
	output, err := command.Output()
	if err != nil {
		return "", nil, err
	}
	body := strings.TrimSpace(string(output))
	if body == "" {
		return "", nil, fmt.Errorf("pdf text extraction returned no text")
	}
	return body, map[string]any{
		"extractor": "pdftotext",
		"pages":     countPDFPages(body),
	}, nil
}

func extractEPUBText(filePath string) (string, map[string]any, error) {
	reader, err := zip.OpenReader(filePath)
	if err != nil {
		return "", nil, err
	}
	defer reader.Close()

	containerBytes, err := readZipFile(&reader.Reader, "META-INF/container.xml")
	if err != nil {
		return "", nil, err
	}
	var container epubContainer
	if err := xml.Unmarshal(containerBytes, &container); err != nil {
		return "", nil, err
	}
	if len(container.Rootfiles) == 0 || strings.TrimSpace(container.Rootfiles[0].FullPath) == "" {
		return "", nil, fmt.Errorf("epub container has no rootfile")
	}
	opfPath := cleanZipPath(container.Rootfiles[0].FullPath)
	opfBytes, err := readZipFile(&reader.Reader, opfPath)
	if err != nil {
		return "", nil, err
	}
	var pkg epubPackage
	if err := xml.Unmarshal(opfBytes, &pkg); err != nil {
		return "", nil, err
	}
	manifest := map[string]epubItem{}
	for _, item := range pkg.Manifest {
		manifest[item.ID] = item
	}

	sections := []string{}
	for _, itemRef := range pkg.Spine {
		item, ok := manifest[itemRef.IDRef]
		if !ok || !isEPUBTextItem(item) {
			continue
		}
		itemPath := resolveZipPath(path.Dir(opfPath), item.Href)
		itemBytes, err := readZipFile(&reader.Reader, itemPath)
		if err != nil {
			continue
		}
		raw := string(itemBytes)
		cleaned := stripIgnoredHTMLBlocks(raw)
		text := stripMarkupText(cleaned)
		if shouldSkipEPUBSection(item.Href, text) {
			continue
		}
		if title := extractTitleTag(cleaned); title != "" && !looksLikeGeneratedEPUBMarker(title) {
			text = "# " + title + "\n\n" + text
		}
		sections = append(sections, text)
	}
	body := strings.TrimSpace(strings.Join(sections, "\f"))
	if body == "" {
		return "", nil, fmt.Errorf("epub text extraction returned no text")
	}
	metadata := map[string]any{
		"extractor": "native-epub",
		"chapters":  len(sections),
	}
	if title := firstNonEmpty(pkg.Metadata.Titles...); title != "" {
		metadata["book_title"] = title
	}
	return body, metadata, nil
}

func readZipFile(reader *zip.Reader, name string) ([]byte, error) {
	name = cleanZipPath(name)
	for _, file := range reader.File {
		if cleanZipPath(file.Name) != name {
			continue
		}
		handle, err := file.Open()
		if err != nil {
			return nil, err
		}
		defer handle.Close()
		return io.ReadAll(handle)
	}
	return nil, fmt.Errorf("zip entry not found: %s", name)
}

func resolveZipPath(baseDir, href string) string {
	if decoded, err := url.PathUnescape(href); err == nil {
		href = decoded
	}
	return cleanZipPath(path.Join(baseDir, href))
}

func cleanZipPath(value string) string {
	return strings.TrimPrefix(path.Clean(strings.ReplaceAll(value, "\\", "/")), "/")
}

func isEPUBTextItem(item epubItem) bool {
	mediaType := strings.ToLower(item.MediaType)
	extension := strings.ToLower(path.Ext(item.Href))
	return strings.Contains(mediaType, "html") || extension == ".html" || extension == ".xhtml" || extension == ".htm"
}

func extractTitleTag(raw string) string {
	lower := strings.ToLower(raw)
	start := strings.Index(lower, "<title")
	if start < 0 {
		return ""
	}
	openEnd := strings.Index(lower[start:], ">")
	if openEnd < 0 {
		return ""
	}
	contentStart := start + openEnd + 1
	closeStart := strings.Index(lower[contentStart:], "</title>")
	if closeStart < 0 {
		return ""
	}
	return strings.TrimSpace(html.UnescapeString(raw[contentStart : contentStart+closeStart]))
}

func stripIgnoredHTMLBlocks(raw string) string {
	raw = stripHTMLBlock(raw, "script")
	raw = stripHTMLBlock(raw, "style")
	return raw
}

func stripHTMLBlock(raw, tag string) string {
	lower := strings.ToLower(raw)
	openNeedle := "<" + tag
	closeNeedle := "</" + tag + ">"
	var builder strings.Builder
	for {
		start := strings.Index(lower, openNeedle)
		if start < 0 {
			builder.WriteString(raw)
			return builder.String()
		}
		builder.WriteString(raw[:start])
		closeStart := strings.Index(lower[start:], closeNeedle)
		if closeStart < 0 {
			return builder.String()
		}
		end := start + closeStart + len(closeNeedle)
		raw = raw[end:]
		lower = lower[end:]
	}
}

func shouldSkipEPUBSection(href, text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	lowerHref := strings.ToLower(href)
	lowerText := strings.ToLower(text)
	wordCount := len(strings.Fields(text))
	if strings.Contains(lowerHref, "cover") && wordCount < 80 {
		return true
	}
	if wordCount < 20 && strings.Contains(lowerText, "cover") {
		return true
	}
	if looksLikeGeneratedEPUBMarker(text) {
		return true
	}
	if looksLikeCSS(text) {
		return true
	}
	return false
}

func looksLikeGeneratedEPUBMarker(text string) bool {
	fields := strings.Fields(strings.ReplaceAll(text, "#", " "))
	if len(fields) == 0 || len(fields) > 3 {
		return false
	}
	for _, field := range fields {
		field = strings.Trim(field, ".:-_")
		if !isGeneratedEPUBMarkerToken(field) {
			return false
		}
	}
	return true
}

func isGeneratedEPUBMarkerToken(token string) bool {
	if len(token) < 2 || len(token) > 12 {
		return false
	}
	if token[0] != 'c' && token[0] != 'C' {
		return false
	}
	hasDigitOrUpper := false
	for index := 1; index < len(token); index++ {
		char := token[index]
		if (char >= '0' && char <= '9') || (char >= 'A' && char <= 'Z') {
			hasDigitOrUpper = true
		}
		if !((char >= '0' && char <= '9') ||
			(char >= 'a' && char <= 'z') ||
			(char >= 'A' && char <= 'Z')) {
			return false
		}
	}
	return hasDigitOrUpper
}

func looksLikeCSS(text string) bool {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return false
	}
	cssTokens := 0
	for _, field := range fields {
		field = strings.ToLower(field)
		if strings.Contains(field, "{") ||
			strings.Contains(field, "}") ||
			strings.Contains(field, ";") ||
			strings.Contains(field, "padding") ||
			strings.Contains(field, "margin") ||
			strings.Contains(field, "font-") ||
			strings.Contains(field, "text-align") {
			cssTokens++
		}
	}
	return len(fields) < 120 && cssTokens*3 >= len(fields)
}

func stripMarkupText(raw string) string {
	var builder strings.Builder
	for index := 0; index < len(raw); {
		if raw[index] != '<' {
			builder.WriteByte(raw[index])
			index++
			continue
		}
		end := strings.IndexByte(raw[index:], '>')
		if end < 0 {
			break
		}
		tag := strings.TrimSpace(raw[index+1 : index+end])
		name := strings.TrimPrefix(tag, "/")
		if space := strings.IndexAny(name, " \t\r\n/"); space >= 0 {
			name = name[:space]
		}
		name = strings.ToLower(name)
		switch name {
		case "p", "div", "li", "section", "article", "blockquote", "br", "hr",
			"h1", "h2", "h3", "h4", "h5", "h6", "tr":
			builder.WriteString("\n\n")
		default:
			builder.WriteByte(' ')
		}
		index += end + 1
	}
	lines := strings.Split(html.UnescapeString(builder.String()), "\n")
	for index, line := range lines {
		lines[index] = strings.Join(strings.Fields(line), " ")
	}
	result := strings.Join(lines, "\n")
	for strings.Contains(result, "\n\n\n") {
		result = strings.ReplaceAll(result, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(result)
}

func countPDFPages(text string) int {
	pages := 0
	for _, page := range strings.Split(text, "\f") {
		if strings.TrimSpace(page) != "" {
			pages++
		}
	}
	if pages == 0 && strings.TrimSpace(text) != "" {
		return 1
	}
	return pages
}

func (s *Server) embedOne(ctx context.Context, text string, requestID string) ([]float64, error) {
	embeddings, err := s.embed(ctx, []string{text}, requestID)
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("embedding service returned no vectors")
	}
	return embeddings[0], nil
}

func (s *Server) embedBatched(ctx context.Context, inputs []string) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no input to embed")
	}
	embeddings := make([][]float64, 0, len(inputs))
	for start := 0; start < len(inputs); {
		end := start
		estimatedTokens := 0
		for end < len(inputs) && end-start < embeddingBatchSize {
			nextTokens := estimateEmbeddingTokens(inputs[end])
			if end > start && estimatedTokens+nextTokens > embeddingBatchTokenBudget {
				break
			}
			estimatedTokens += nextTokens
			end++
		}
		if end == start {
			end++
		}
		batch, err := s.embed(ctx, inputs[start:end], "")
		if err != nil {
			return nil, err
		}
		embeddings = append(embeddings, batch...)
		start = end
	}
	return embeddings, nil
}

func estimateEmbeddingTokens(input string) int {
	asciiRunes := 0
	nonASCII := 0
	for _, r := range input {
		if r <= unicode.MaxASCII {
			if !unicode.IsSpace(r) {
				asciiRunes++
			}
			continue
		}
		nonASCII++
	}
	estimate := (asciiRunes+3)/4 + nonASCII + 8
	if estimate < 1 {
		return 1
	}
	return estimate
}

func (s *Server) embed(ctx context.Context, inputs []string, requestID string) ([][]float64, error) {
	start := time.Now()
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no input to embed")
	}
	inputChars := totalRuneCount(inputs)
	payload := EmbeddingRequest{Model: s.embeddingModel, Input: inputs}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.embeddingURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.embeddingKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.embeddingKey)
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d ms=%d err=%q", requestID, s.embeddingModel, len(inputs), inputChars, time.Since(start).Milliseconds(), err.Error())
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d ms=%d err=%q", requestID, s.embeddingModel, len(inputs), inputChars, time.Since(start).Milliseconds(), err.Error())
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d http_status=%d ms=%d body=%q", requestID, s.embeddingModel, len(inputs), inputChars, resp.StatusCode, time.Since(start).Milliseconds(), strings.TrimSpace(string(respBody)))
		return nil, fmt.Errorf("embedding status %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}
	var decoded EmbeddingResponse
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d phase=decode ms=%d err=%q", requestID, s.embeddingModel, len(inputs), inputChars, time.Since(start).Milliseconds(), err.Error())
		return nil, err
	}
	if len(decoded.Data) != len(inputs) {
		log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d phase=count got=%d ms=%d", requestID, s.embeddingModel, len(inputs), inputChars, len(decoded.Data), time.Since(start).Milliseconds())
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(decoded.Data), len(inputs))
	}
	embeddings := make([][]float64, len(decoded.Data))
	for i := range decoded.Data {
		if len(decoded.Data[i].Embedding) == 0 {
			log.Printf("memory.embedding request_id=%q status=error model=%q inputs=%d input_chars=%d phase=empty_vector index=%d ms=%d", requestID, s.embeddingModel, len(inputs), inputChars, i, time.Since(start).Milliseconds())
			return nil, fmt.Errorf("embedding %d is empty", i)
		}
		embeddings[i] = decoded.Data[i].Embedding
	}
	dimension := 0
	if len(embeddings) > 0 {
		dimension = len(embeddings[0])
	}
	elapsed := time.Since(start)
	if requestID != "" || elapsed >= 2*time.Second {
		log.Printf("memory.embedding request_id=%q status=ok model=%q inputs=%d input_chars=%d dimension=%d ms=%d", requestID, s.embeddingModel, len(inputs), inputChars, dimension, elapsed.Milliseconds())
	}
	return embeddings, nil
}

func scanMemoryItem(row *sql.Row) (MemoryItem, error) {
	var item MemoryItem
	var tagsRaw, metadataRaw string
	if err := row.Scan(&item.ID, &item.UserKey, &item.Kind, &item.SourceType, &item.SourceURI, &item.Title, &item.Body, &tagsRaw, &metadataRaw, &item.CreatedAt, &item.UpdatedAt); err != nil {
		return MemoryItem{}, err
	}
	item.Tags = parseStringSlice(tagsRaw)
	item.Metadata = parseMap(metadataRaw)
	return item, nil
}

func normalizeUpsertRequest(req UpsertMemoryRequest) UpsertMemoryRequest {
	req.Kind = firstNonEmpty(req.Kind, "note")
	req.SourceType = firstNonEmpty(req.SourceType, "manual")
	req.SourceURI = strings.TrimSpace(req.SourceURI)
	req.Title = strings.TrimSpace(req.Title)
	req.Body = strings.TrimSpace(req.Body)
	req.Tags = normalizeTags(req.Tags)
	req.Metadata = nonNilMap(req.Metadata)
	return req
}

func shouldLogIngestProgress(req UpsertMemoryRequest) bool {
	return req.SourceType == "watch_file"
}

func normalizeTags(tags []string) []string {
	seen := map[string]bool{}
	normalized := []string{}
	for _, tag := range tags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag == "" || seen[tag] {
			continue
		}
		seen[tag] = true
		normalized = append(normalized, tag)
	}
	sort.Strings(normalized)
	return normalized
}

func chunkText(text string) []string {
	runes := []rune(strings.TrimSpace(text))
	if len(runes) <= maxChunkRunes {
		return []string{string(runes)}
	}
	chunks := []string{}
	for start := 0; start < len(runes); {
		end := start + maxChunkRunes
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
		next := end - chunkOverlapRunes
		if next <= start {
			next = end
		}
		start = next
	}
	return chunks
}

func chunksForRequest(req UpsertMemoryRequest) []textChunk {
	if strings.EqualFold(stringMapValue(req.Metadata, "file_ext"), ".pdf") {
		return chunkSectionedTextWithMetadata(req.Body, "page")
	}
	if strings.EqualFold(stringMapValue(req.Metadata, "file_ext"), ".epub") {
		return chunkSectionedTextWithMetadata(req.Body, "chapter")
	}
	return chunkTextWithMetadata(req.Body)
}

func chunkSectionedTextWithMetadata(text string, sectionKey string) []textChunk {
	chunks := []textChunk{}
	for sectionIndex, section := range strings.Split(text, "\f") {
		sectionText := strings.TrimSpace(section)
		if sectionText == "" {
			continue
		}
		sectionChunks := chunkTextWithMetadata(sectionText)
		for _, chunk := range sectionChunks {
			chunk.Metadata[sectionKey] = sectionIndex + 1
			chunk.Metadata["chunk_index"] = len(chunks)
			chunks = append(chunks, chunk)
		}
	}
	if len(chunks) == 0 {
		return chunkTextWithMetadata(text)
	}
	return chunks
}

func chunkTextWithMetadata(text string) []textChunk {
	body := strings.TrimSpace(text)
	if body == "" {
		return []textChunk{}
	}

	chunks := []textChunk{}
	for _, paragraph := range paragraphRanges(body) {
		runes := []rune(paragraph.text)
		for start := 0; start < len(runes); {
			end := start + maxChunkRunes
			if end > len(runes) {
				end = len(runes)
			}
			byteStart := paragraph.start + len(string(runes[:start]))
			byteEnd := paragraph.start + len(string(runes[:end]))
			raw := string(runes[start:end])
			chunk := strings.TrimSpace(raw)
			if chunk != "" {
				metadata := map[string]any{
					"chunk_index": len(chunks),
					"line_start":  lineNumberAt(body, byteStart),
					"line_end":    lineNumberAt(body, byteEnd),
				}
				heading := nearestMarkdownHeading(body[:clampByteOffset(body, byteStart)])
				if heading == "" {
					heading = firstMarkdownHeading(raw)
				}
				if heading != "" && !looksLikeGeneratedEPUBMarker(heading) {
					metadata["heading"] = heading
				}
				chunks = append(chunks, textChunk{Text: chunk, Metadata: metadata})
			}
			if end == len(runes) {
				break
			}
			next := end - chunkOverlapRunes
			if next <= start {
				next = end
			}
			start = next
		}
	}
	return chunks
}

type paragraphRange struct {
	text       string
	start, end int
}

func paragraphRanges(body string) []paragraphRange {
	paragraphs := []paragraphRange{}
	paragraphStart := -1
	offset := 0
	for _, line := range strings.SplitAfter(body, "\n") {
		lineText := strings.TrimSpace(strings.TrimSuffix(line, "\n"))
		if lineText == "" {
			if paragraphStart >= 0 {
				paragraphs = append(paragraphs, paragraphRange{
					text:  strings.TrimSpace(body[paragraphStart:offset]),
					start: paragraphStart,
					end:   offset,
				})
				paragraphStart = -1
			}
			offset += len(line)
			continue
		}
		if paragraphStart < 0 {
			paragraphStart = offset
		}
		offset += len(line)
	}
	if paragraphStart >= 0 {
		paragraphs = append(paragraphs, paragraphRange{
			text:  strings.TrimSpace(body[paragraphStart:offset]),
			start: paragraphStart,
			end:   offset,
		})
	}
	return paragraphs
}

func clampByteOffset(text string, byteOffset int) int {
	if byteOffset < 0 {
		return 0
	}
	if byteOffset > len(text) {
		return len(text)
	}
	return byteOffset
}

func lineNumberAt(text string, byteOffset int) int {
	byteOffset = clampByteOffset(text, byteOffset)
	return strings.Count(text[:byteOffset], "\n") + 1
}

func nearestMarkdownHeading(prefix string) string {
	lines := strings.Split(prefix, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" || !strings.HasPrefix(line, "#") {
			continue
		}
		heading := strings.TrimSpace(strings.TrimLeft(line, "#"))
		if heading != "" && !looksLikeGeneratedEPUBMarker(heading) {
			return heading
		}
	}
	return ""
}

func firstMarkdownHeading(text string) string {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "#") {
			continue
		}
		heading := strings.TrimSpace(strings.TrimLeft(line, "#"))
		if heading != "" && !looksLikeGeneratedEPUBMarker(heading) {
			return heading
		}
	}
	return ""
}

func rawEmbeddingInput(chunk string) string {
	return strings.TrimSpace(chunk)
}

func metadataTextForChunk(req UpsertMemoryRequest, chunk textChunk, index int, contextText string, generated chunkContextMetadata) string {
	parts := []string{}
	if title := firstNonEmpty(stringMapValue(req.Metadata, "book_title"), req.Title); title != "" {
		parts = append(parts, "Title: "+title)
	}
	if source := strings.TrimSpace(req.SourceURI); source != "" {
		parts = append(parts, "Source: "+source)
	}
	if chapter := strings.TrimSpace(fmt.Sprint(chunk.Metadata["chapter"])); chapter != "" && chapter != "<nil>" {
		parts = append(parts, "Chapter: "+chapter)
	}
	if heading := stringMapValue(chunk.Metadata, "heading"); heading != "" {
		parts = append(parts, "Heading: "+heading)
	}
	entities := mentionedEntitiesForChunk(req, chunk)
	if len(entities) > 0 {
		parts = append(parts, "Entities: "+strings.Join(entities, "; "))
	}
	if pairs := entityCooccurrences(entities); len(pairs) > 0 {
		parts = append(parts, "Co-occurrences: "+strings.Join(pairs, "; "))
	}
	return strings.Join(parts, "\n\n")
}

func entityCooccurrences(entities []string) []string {
	entities = uniqueLimitedStrings(entities, 12)
	pairs := make([]string, 0, 24)
	for i := 0; i < len(entities); i++ {
		for j := i + 1; j < len(entities); j++ {
			left := strings.TrimSpace(entities[i])
			right := strings.TrimSpace(entities[j])
			if left == "" || right == "" || strings.EqualFold(left, right) {
				continue
			}
			pairs = append(pairs, left+" <-> "+right)
			if len(pairs) >= 24 {
				return pairs
			}
		}
	}
	return pairs
}

func embeddingTextForChunk(metadataText, chunkText string) string {
	parts := []string{}
	if strings.TrimSpace(metadataText) != "" {
		parts = append(parts, "Document metadata:\n"+limitRunes(strings.TrimSpace(metadataText), embeddingMetadataRunes))
	}
	if strings.TrimSpace(chunkText) != "" {
		parts = append(parts, "Passage:\n"+strings.TrimSpace(chunkText))
	}
	return strings.Join(parts, "\n\n")
}

func limitRunes(text string, limit int) string {
	runes := []rune(strings.TrimSpace(text))
	if limit <= 0 || len(runes) <= limit {
		return string(runes)
	}
	return string(runes[:limit]) + "..."
}

func contextualSummaryForChunk(req UpsertMemoryRequest, chunk textChunk, index int) string {
	sourceTitle := firstNonEmpty(stringMapValue(req.Metadata, "book_title"), req.Title)
	phrases := []string{}
	if sourceTitle != "" {
		phrases = append(phrases, "This chunk is from "+sourceTitle+".")
	}
	location := []string{}
	if chapter := fmt.Sprint(chunk.Metadata["chapter"]); strings.TrimSpace(chapter) != "" && chapter != "<nil>" {
		location = append(location, "chapter "+chapter)
	}
	if page := fmt.Sprint(chunk.Metadata["page"]); strings.TrimSpace(page) != "" && page != "<nil>" {
		location = append(location, "page "+page)
	}
	if heading := stringMapValue(chunk.Metadata, "heading"); heading != "" {
		location = append(location, "heading "+heading)
	}
	location = append(location, fmt.Sprintf("chunk %d", index))
	if len(location) > 0 {
		phrases = append(phrases, "Location: "+strings.Join(location, ", ")+".")
	}
	corpus := []string{}
	if collection := stringMapValue(req.Metadata, "collection"); collection != "" {
		corpus = append(corpus, "collection "+collection)
	}
	if scope := stringMapValue(req.Metadata, "scope"); scope != "" {
		corpus = append(corpus, "scope "+scope)
	}
	if len(corpus) > 0 {
		phrases = append(phrases, "Corpus: "+strings.Join(corpus, ", ")+".")
	}
	if source := readableSourcePath(req.SourceURI, req.Metadata); source != "" {
		phrases = append(phrases, "Source path: "+source+".")
	}
	return strings.Join(phrases, " ")
}

func mentionedEntitiesForChunk(req UpsertMemoryRequest, chunk textChunk) []string {
	candidates := []string{
		req.Title,
		stringMapValue(req.Metadata, "book_title"),
		stringMapValue(chunk.Metadata, "heading"),
	}
	candidates = append(candidates, extractMentionedEntities(chunk.Text)...)
	return uniqueLimitedStrings(candidates, 24)
}

func extractMentionedEntities(text string) []string {
	entities := []string{}
	current := []string{}
	flushEntity := func() {
		if len(current) == 0 {
			return
		}
		entity := strings.Join(current, " ")
		if !isEntityStopword(entity) {
			entities = append(entities, entity)
		}
		current = current[:0]
	}
	consumeToken := func(token string) {
		token = strings.Trim(token, "'-_")
		if token == "" {
			return
		}
		if isEntityStopword(token) {
			flushEntity()
			return
		}
		if isEntityToken(token) {
			current = append(current, token)
			return
		}
		flushEntity()
	}
	token := strings.Builder{}
	flushToken := func() {
		consumeToken(token.String())
		token.Reset()
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '-' || r == '_' {
			token.WriteRune(r)
			continue
		}
		flushToken()
		if !unicode.IsSpace(r) {
			flushEntity()
		}
	}
	flushToken()
	flushEntity()
	return entities
}

func isEntityToken(token string) bool {
	runes := []rune(token)
	if len(runes) == 0 {
		return false
	}
	return unicode.IsUpper(runes[0])
}

func isEntityStopword(entity string) bool {
	switch strings.ToLower(strings.TrimSpace(entity)) {
	case "a", "an", "and", "as", "but", "chapter", "he", "however", "i", "it", "later", "no", "of", "she", "that", "the", "then", "they", "this", "we", "yes", "you":
		return true
	default:
		return false
	}
}

func uniqueLimitedStrings(values []string, limit int) []string {
	seen := map[string]bool{}
	result := []string{}
	for _, value := range values {
		value = strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
		if value == "" {
			continue
		}
		key := strings.ToLower(value)
		if seen[key] {
			continue
		}
		seen[key] = true
		result = append(result, value)
		if limit > 0 && len(result) >= limit {
			break
		}
	}
	return result
}

func chunkContextText(req UpsertMemoryRequest, chunk textChunk, index int) string {
	parts := []string{}
	addContextPart(&parts, "Title", req.Title)
	if bookTitle := stringMapValue(req.Metadata, "book_title"); bookTitle != "" && !strings.EqualFold(bookTitle, req.Title) {
		addContextPart(&parts, "Book", bookTitle)
	}
	addContextPart(&parts, "Kind", req.Kind)
	addContextPart(&parts, "Source type", req.SourceType)
	addContextPart(&parts, "Collection", stringMapValue(req.Metadata, "collection"))
	addContextPart(&parts, "Scope", stringMapValue(req.Metadata, "scope"))
	addContextPart(&parts, "File type", stringMapValue(req.Metadata, "file_ext"))
	addContextPart(&parts, "Chapter", fmt.Sprint(chunk.Metadata["chapter"]))
	addContextPart(&parts, "Page", fmt.Sprint(chunk.Metadata["page"]))
	addContextPart(&parts, "Heading", stringMapValue(chunk.Metadata, "heading"))
	addContextPart(&parts, "Chunk", fmt.Sprintf("%d", index))
	addContextPart(&parts, "Modified", stringMapValue(req.Metadata, "modified_at"))
	addContextPart(&parts, "Source", readableSourcePath(req.SourceURI, req.Metadata))
	return strings.Join(parts, "\n")
}

func addContextPart(parts *[]string, label, value string) {
	value = strings.TrimSpace(value)
	if value == "" || value == "<nil>" {
		return
	}
	*parts = append(*parts, label+": "+value)
}

func readableSourcePath(sourceURI string, metadata map[string]any) string {
	source := strings.TrimPrefix(strings.TrimSpace(sourceURI), "file://")
	if pathValue := stringMapValue(metadata, "path"); pathValue != "" {
		source = pathValue
	}
	if root := stringMapValue(metadata, "watched_root"); root != "" {
		if rel, err := filepath.Rel(root, source); err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.ToSlash(source)
}

func vectorLiteral(values []float64) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprintf("%g", value)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func nullableVectorLiteral(values []float64) any {
	if len(values) == 0 {
		return nil
	}
	return vectorLiteral(values)
}

func parseWatchDirs(raw string) map[string][]WatchRoot {
	result := map[string][]WatchRoot{}
	for _, entry := range strings.Split(raw, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		user, dirsRaw, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		user = strings.TrimSpace(user)
		if user == "" {
			continue
		}
		for _, dir := range strings.Split(dirsRaw, ",") {
			root := parseWatchRoot(dir)
			if root.Path != "" {
				result[user] = append(result[user], root)
			}
		}
	}
	return result
}

func parseWatchRoot(raw string) WatchRoot {
	parts := strings.Split(raw, "|")
	root := WatchRoot{
		Path:       strings.TrimSpace(parts[0]),
		Scope:      defaultMemoryScope,
		Collection: "",
	}
	for _, part := range parts[1:] {
		key, value, ok := strings.Cut(strings.TrimSpace(part), "=")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		switch key {
		case "scope":
			root.Scope = normalizeMetadataValue(value, defaultMemoryScope)
		case "collection":
			root.Collection = normalizeMetadataValue(value, "")
		}
	}
	if root.Collection == "" {
		root.Collection = collectionFromPath(root.Path)
	}
	return root
}

func normalizeMetadataValue(value, fallback string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, " ", "-")
	if value == "" {
		return fallback
	}
	return value
}

func collectionFromPath(path string) string {
	base := strings.ToLower(filepath.Base(strings.TrimRight(strings.TrimSpace(path), "/")))
	switch base {
	case "", ".", "/":
		return ""
	case "obsidian":
		return "obsidian"
	default:
		return strings.ReplaceAll(base, " ", "-")
	}
}

func supportedWatchFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".txt", ".org", ".pdf", ".epub":
		return true
	default:
		return false
	}
}

func ignoredWatchDirectory(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case ".trash", ".caltrash", "manga", "review-to-delete":
		return true
	default:
		return false
	}
}

func userKeyFromRequest(r *http.Request) string {
	if value := strings.TrimSpace(r.Header.Get("X-Memory-User-Key")); value != "" {
		return value
	}
	return strings.TrimSpace(r.URL.Query().Get("user_key"))
}

func decodeJSON(r *http.Request, dst any) error {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write JSON: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func envOrDefault(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func nonNilMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

func parseStringSlice(raw string) []string {
	var values []string
	if err := json.Unmarshal([]byte(raw), &values); err != nil {
		return []string{}
	}
	return values
}

func parseMap(raw string) map[string]any {
	var value map[string]any
	if err := json.Unmarshal([]byte(raw), &value); err != nil || value == nil {
		return map[string]any{}
	}
	return value
}

func stringMapValue(values map[string]any, key string) string {
	value, ok := values[key]
	if !ok {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}

func parseLimit(raw string, fallback int) int {
	if strings.TrimSpace(raw) == "" {
		return fallback
	}
	var limit int
	if _, err := fmt.Sscanf(raw, "%d", &limit); err != nil {
		return fallback
	}
	return clampLimit(limit)
}

func totalRuneCount(values []string) int {
	total := 0
	for _, value := range values {
		total += len([]rune(value))
	}
	return total
}

func clampLimit(limit int) int {
	if limit <= 0 {
		return 20
	}
	if limit > 150 {
		return 150
	}
	return limit
}

func normalizeSearchMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case searchModeLiteral:
		return searchModeLiteral
	case searchModeHybrid:
		return searchModeHybrid
	default:
		return searchModeSemantic
	}
}

func literalRecallTSQuery(terms []string) string {
	if len(terms) == 0 {
		return ""
	}
	parts := make([]string, 0, len(terms))
	for _, term := range terms {
		parts = append(parts, "'"+strings.ReplaceAll(term, "'", "''")+"'")
	}
	return strings.Join(parts, " | ")
}

func literalRecallTerms(query string) []string {
	query = strings.NewReplacer("’", "'", "`", "'").Replace(strings.ToLower(strings.TrimSpace(query)))
	rawTerms := strings.FieldsFunc(query, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	seen := map[string]bool{}
	terms := []string{}
	for _, term := range rawTerms {
		term = strings.TrimSpace(strings.TrimSuffix(term, "'s"))
		if len([]rune(term)) < 2 || literalRecallStopword(term) || seen[term] {
			continue
		}
		seen[term] = true
		terms = append(terms, term)
	}
	if len(terms) > 16 {
		return terms[:16]
	}
	return terms
}

func literalRecallStopword(term string) bool {
	switch term {
	case "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does",
		"for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
		"it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "that",
		"the", "their", "them", "then", "there", "these", "they", "this", "to",
		"was", "we", "were", "what", "when", "where", "which", "who", "why", "with",
		"would", "you", "your":
		return true
	default:
		return false
	}
}

func searchResultRankLimit(req SearchRequest) int {
	if req.ChunkLimitPerSource > 0 {
		return clampLimit(req.ChunkLimitPerSource)
	}
	if strings.EqualFold(strings.TrimSpace(req.Scope), "library") {
		return libraryChunksPerDoc
	}
	return 1
}

func vectorCandidatePoolLimit(limit int) int {
	pool := limit * 10
	if pool < 300 {
		pool = 300
	}
	if pool > 1000 {
		pool = 1000
	}
	return pool
}

func normalizeTimeFilter(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || strings.EqualFold(value, "all") {
		return ""
	}
	for _, char := range value {
		if char < '0' || char > '9' {
			return ""
		}
	}
	return value
}

func normalizeMetadataFilter(value string) string {
	value = strings.TrimSpace(value)
	if strings.EqualFold(value, "all") {
		return ""
	}
	return value
}

func sourceFolderPrefix(sourceURI string) string {
	sourceURI = strings.TrimRight(strings.TrimSpace(sourceURI), "/")
	index := strings.LastIndex(sourceURI, "/")
	if index <= 0 {
		return ""
	}
	return sourceURI[:index]
}

func preview(body string) string {
	body = strings.Join(strings.Fields(body), " ")
	runes := []rune(body)
	if len(runes) <= 220 {
		return body
	}
	return string(runes[:220]) + "..."
}

func titleFromBody(body string) string {
	lines := strings.Split(body, "\n")
	for _, line := range lines {
		line = strings.Trim(strings.TrimSpace(line), "# ")
		if line != "" {
			runes := []rune(line)
			if len(runes) > 90 {
				return string(runes[:90])
			}
			return line
		}
	}
	return "Untitled memory"
}

func hashText(text string) string {
	// This is not a cryptographic identity boundary; it is a compact content-change marker.
	h := uint64(1469598103934665603)
	for _, b := range []byte(text) {
		h ^= uint64(b)
		h *= 1099511628211
	}
	return fmt.Sprintf("%016x", h)
}

func contentHashFor(title, sourceURI, body string, extraParts ...string) string {
	parts := []string{
		contentHashVersion,
		strings.TrimSpace(title),
		strings.TrimSpace(sourceURI),
		body,
	}
	for _, part := range extraParts {
		parts = append(parts, strings.TrimSpace(part))
	}
	return hashText(strings.Join(parts, "\n"))
}

func contentFingerprintFor(body string) string {
	return hashText(strings.TrimSpace(body))
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}
