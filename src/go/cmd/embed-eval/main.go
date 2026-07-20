package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

const (
	defaultDatabaseURL = "host=100.73.65.14 port=5438 dbname=pensieve user=postgres password=postgres sslmode=disable"
	defaultModelSpec   = "current|http://127.0.0.1:5021/v1/embeddings|Qwen3-Embedding-0.6B||"
)

type modelSpec struct {
	Name       string
	URL        string
	Model      string
	Dimensions int
	KeyEnv     string
}

type chunkRow struct {
	MemoryID     string
	ChunkID      string
	UserKey      string
	Title        string
	SourceURI    string
	Text         string
	MetadataText string
	ChunkIndex   int
}

type scoredChunk struct {
	Chunk      chunkRow
	Similarity float64
	Target     bool
}

type embeddingRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func main() {
	var (
		databaseURL   = flag.String("database-url", envOrDefault("MEMORY_PALACE_DATABASE_URL", defaultDatabaseURL), "Postgres connection string")
		userKey       = flag.String("user", "local", "Memory Palace user key")
		query         = flag.String("query", "", "Question/query to embed")
		target        = flag.String("target", "", "Substring that marks expected target chunks")
		expectedTitle = flag.String("expected-title", "", "Document title that marks expected target chunks")
		scope         = flag.String("scope", "library", "Optional memory scope filter")
		collection    = flag.String("collection", "naomi-novik", "Optional collection filter")
		limit         = flag.Int("limit", 0, "Maximum chunks to evaluate; 0 means all matching chunks")
		topK          = flag.Int("top", 10, "Number of top results to print per model")
		batchSize     = flag.Int("batch", 32, "Embedding request batch size")
		modelsRaw     = flag.String("models", envOrDefault("EMBED_EVAL_MODELS", defaultModelSpec), "Comma-separated model specs: name|url|model|dimensions|key_env")
		store         = flag.Bool("store", false, "Store chunk embeddings in the sidecar embedding table")
		storedSearch  = flag.Bool("stored-search", false, "Rank chunks through Postgres using stored sidecar embeddings")
		storeTable    = flag.String("store-table", "memory_chunk_embeddings_3072", "Sidecar embedding table for -store and -stored-search")
		storeType     = flag.String("store-vector-type", "halfvec", "Sidecar embedding type for casts: vector or halfvec")
		modelKey      = flag.String("model-key", "", "Stable sidecar model key; defaults to the model spec name")
		verbose       = flag.Bool("verbose", false, "Print embedding and storage timings to stderr")
		timeout       = flag.Duration("timeout", 10*time.Minute, "Per-model timeout")
	)
	flag.Parse()

	needsQuery := !*store || *storedSearch
	if needsQuery && strings.TrimSpace(*query) == "" {
		exitf("-query is required")
	}
	needsTarget := !*store || *storedSearch
	if needsTarget && strings.TrimSpace(*target) == "" && strings.TrimSpace(*expectedTitle) == "" {
		exitf("-target or -expected-title is required")
	}
	models, err := parseModelSpecs(*modelsRaw)
	if err != nil {
		exitf("parse models: %v", err)
	}

	db, err := sql.Open("postgres", *databaseURL)
	if err != nil {
		exitf("open database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	chunks, err := loadChunks(ctx, db, *userKey, *scope, *collection, *limit)
	if err != nil {
		exitf("load chunks: %v", err)
	}
	if len(chunks) == 0 {
		exitf("no chunks matched filters")
	}
	targetCount := countTargetChunks(chunks, *target, *expectedTitle)
	if needsTarget && targetCount == 0 {
		exitf("no chunks match target=%q expected_title=%q", *target, *expectedTitle)
	}
	if needsQuery {
		fmt.Printf("query: %s\n", *query)
	}
	fmt.Printf("filters: user=%s scope=%s collection=%s chunks=%d", *userKey, emptyAsAll(*scope), emptyAsAll(*collection), len(chunks))
	if needsTarget {
		fmt.Printf(" target_chunks=%d", targetCount)
		if strings.TrimSpace(*expectedTitle) != "" {
			fmt.Printf(" expected_title=%q", *expectedTitle)
		}
	}
	fmt.Println()
	fmt.Println()

	client := &http.Client{Timeout: *timeout}
	for _, model := range models {
		key := strings.TrimSpace(*modelKey)
		if key == "" {
			key = model.Name
		}
		if *store {
			if err := storeModelEmbeddings(ctx, db, client, model, *storeTable, *storeType, key, chunks, *batchSize, *verbose); err != nil {
				fmt.Printf("model=%s status=store_error err=%v\n\n", model.Name, err)
				continue
			}
			if !*storedSearch {
				continue
			}
		}
		var err error
		if *storedSearch {
			err = evalStoredModel(ctx, db, client, model, *storeTable, *storeType, key, chunks, *query, *target, *expectedTitle, *scope, *collection, *topK)
		} else {
			err = evalModel(ctx, client, model, chunks, *query, *target, *expectedTitle, *batchSize, *topK)
		}
		if err != nil {
			fmt.Printf("model=%s status=error err=%v\n\n", model.Name, err)
		}
	}
}

func evalModel(ctx context.Context, client *http.Client, model modelSpec, chunks []chunkRow, query, target, expectedTitle string, batchSize, topK int) error {
	start := time.Now()
	inputs := make([]string, 0, len(chunks)+1)
	inputs = append(inputs, query)
	for _, chunk := range chunks {
		inputs = append(inputs, evalEmbeddingInput(chunk.MetadataText, chunk.Text))
	}
	embeddings, err := embedBatched(ctx, client, model, inputs, batchSize)
	if err != nil {
		return err
	}
	if len(embeddings) != len(inputs) {
		return fmt.Errorf("embedding count mismatch: got %d want %d", len(embeddings), len(inputs))
	}
	queryEmbedding := embeddings[0]
	scores := make([]scoredChunk, 0, len(chunks))
	for i, chunk := range chunks {
		scores = append(scores, scoredChunk{
			Chunk:      chunk,
			Similarity: cosineSimilarity(queryEmbedding, embeddings[i+1]),
			Target:     targetMatches(chunk, target, expectedTitle),
		})
	}
	sort.SliceStable(scores, func(i, j int) bool {
		return scores[i].Similarity > scores[j].Similarity
	})
	bestRank := 0
	for i, score := range scores {
		if score.Target {
			bestRank = i + 1
			break
		}
	}
	dimension := 0
	if len(queryEmbedding) > 0 {
		dimension = len(queryEmbedding)
	}
	fmt.Printf("model=%s model_id=%s dimension=%d chunks=%d best_target_rank=%d elapsed_ms=%d\n", model.Name, model.Model, dimension, len(chunks), bestRank, time.Since(start).Milliseconds())
	printTop(scores, topK)
	printTargetWindow(scores, bestRank, 3)
	fmt.Println()
	return nil
}

func storeModelEmbeddings(ctx context.Context, db *sql.DB, client *http.Client, model modelSpec, tableName, vectorType, modelKey string, chunks []chunkRow, batchSize int, verbose bool) error {
	start := time.Now()
	table, err := safeIdent(tableName)
	if err != nil {
		return err
	}
	castType, err := safeVectorType(vectorType)
	if err != nil {
		return err
	}
	if batchSize <= 0 {
		batchSize = 32
	}
	existing, err := storedChunkIDs(ctx, db, table, firstChunkUserKey(chunks), modelKey)
	if err != nil {
		return err
	}
	pending := make([]chunkRow, 0, len(chunks))
	for _, chunk := range chunks {
		if !existing[chunk.ChunkID] {
			pending = append(pending, chunk)
		}
	}
	if len(pending) == 0 {
		fmt.Printf("model=%s model_key=%s stored_chunks=%d table=%s status=up_to_date elapsed_ms=%d\n", model.Name, modelKey, len(chunks), tableName, time.Since(start).Milliseconds())
		return nil
	}
	upsertSQL := fmt.Sprintf(`
		INSERT INTO %s (chunk_id, user_key, model_key, model_name, embedding, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5::%s, $6::jsonb, NOW(), NOW())
		ON CONFLICT (chunk_id, model_key) DO UPDATE
		SET user_key = EXCLUDED.user_key,
			model_name = EXCLUDED.model_name,
			embedding = EXCLUDED.embedding,
			metadata = EXCLUDED.metadata,
			updated_at = NOW()
	`, table, castType)
	for start := 0; start < len(pending); start += batchSize {
		end := start + batchSize
		if end > len(pending) {
			end = len(pending)
		}
		inputs := make([]string, 0, end-start)
		for _, chunk := range pending[start:end] {
			inputs = append(inputs, evalEmbeddingInput(chunk.MetadataText, chunk.Text))
		}
		embedStarted := time.Now()
		embeddings, err := embed(ctx, client, model, inputs)
		if err != nil {
			return fmt.Errorf("embed chunks %d-%d: %w", start+1, end, err)
		}
		if verbose {
			fmt.Fprintf(os.Stderr, "model=%s embedded=%d-%d elapsed_ms=%d\n", model.Name, start+1, end, time.Since(embedStarted).Milliseconds())
		}
		if len(embeddings) != len(inputs) {
			return fmt.Errorf("embedding count mismatch for chunks %d-%d: got %d want %d", start+1, end, len(embeddings), len(inputs))
		}
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return err
		}
		for i, chunk := range pending[start:end] {
			metadata, err := json.Marshal(map[string]any{
				"source_uri":  chunk.SourceURI,
				"title":       chunk.Title,
				"chunk_index": chunk.ChunkIndex,
			})
			if err != nil {
				tx.Rollback()
				return err
			}
			if _, err := tx.ExecContext(ctx, upsertSQL, chunk.ChunkID, chunk.UserKey, modelKey, model.Model, vectorLiteral(embeddings[i]), string(metadata)); err != nil {
				tx.Rollback()
				return fmt.Errorf("upsert chunk %s: %w", chunk.ChunkID, err)
			}
		}
		commitStarted := time.Now()
		if err := tx.Commit(); err != nil {
			return err
		}
		if verbose {
			fmt.Fprintf(os.Stderr, "model=%s committed=%d-%d elapsed_ms=%d\n", model.Name, start+1, end, time.Since(commitStarted).Milliseconds())
		}
		fmt.Printf("model=%s stored=%d/%d pending=%d/%d\n", model.Name, len(existing)+end, len(chunks), end, len(pending))
	}
	fmt.Printf("model=%s model_key=%s stored_chunks=%d table=%s elapsed_ms=%d\n", model.Name, modelKey, len(chunks), tableName, time.Since(start).Milliseconds())
	return nil
}

func storedChunkIDs(ctx context.Context, db *sql.DB, tableName, userKey, modelKey string) (map[string]bool, error) {
	rows, err := db.QueryContext(ctx, fmt.Sprintf(`
		SELECT chunk_id
		FROM %s
		WHERE user_key = $1 AND model_key = $2
	`, tableName), userKey, modelKey)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids[id] = true
	}
	return ids, rows.Err()
}

func evalStoredModel(ctx context.Context, db *sql.DB, client *http.Client, model modelSpec, tableName, vectorType, modelKey string, chunks []chunkRow, query, target, expectedTitle, scope, collection string, topK int) error {
	start := time.Now()
	table, err := safeIdent(tableName)
	if err != nil {
		return err
	}
	castType, err := safeVectorType(vectorType)
	if err != nil {
		return err
	}
	queryEmbedding, err := embedBatched(ctx, client, model, []string{query}, 1)
	if err != nil {
		return err
	}
	if len(queryEmbedding) != 1 {
		return fmt.Errorf("query embedding count mismatch: got %d want 1", len(queryEmbedding))
	}
	chunkByID := make(map[string]chunkRow, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ChunkID] = chunk
	}
	rows, err := db.QueryContext(ctx, fmt.Sprintf(`
		SELECT c.memory_id, c.id, c.user_key, i.title, i.source_uri, c.text, c.metadata_text, c.chunk_index,
			1 - (e.embedding <=> $3::%s) AS similarity
		FROM %s e
		JOIN memory_chunks c ON c.id = e.chunk_id
		JOIN memory_items i ON i.id = c.memory_id
		WHERE e.user_key = $1
			AND e.model_key = $2
			AND ($4 = '' OR i.metadata->>'scope' = $4)
			AND ($5 = '' OR i.metadata->>'collection' = $5)
			AND i.source_uri NOT LIKE '%%/.trash/%%'
		ORDER BY e.embedding <=> $3::%s
	`, castType, table, castType), firstChunkUserKey(chunks), modelKey, vectorLiteral(queryEmbedding[0]), strings.TrimSpace(scope), strings.TrimSpace(collection))
	if err != nil {
		return err
	}
	defer rows.Close()
	scores := []scoredChunk{}
	for rows.Next() {
		var score scoredChunk
		if err := rows.Scan(&score.Chunk.MemoryID, &score.Chunk.ChunkID, &score.Chunk.UserKey, &score.Chunk.Title, &score.Chunk.SourceURI, &score.Chunk.Text, &score.Chunk.MetadataText, &score.Chunk.ChunkIndex, &score.Similarity); err != nil {
			return err
		}
		if _, ok := chunkByID[score.Chunk.ChunkID]; !ok {
			continue
		}
		score.Target = targetMatches(score.Chunk, target, expectedTitle)
		scores = append(scores, score)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	bestRank := 0
	for i, score := range scores {
		if score.Target {
			bestRank = i + 1
			break
		}
	}
	dimension := 0
	if len(queryEmbedding[0]) > 0 {
		dimension = len(queryEmbedding[0])
	}
	fmt.Printf("model=%s model_id=%s model_key=%s table=%s dimension=%d chunks=%d best_target_rank=%d elapsed_ms=%d\n", model.Name, model.Model, modelKey, tableName, dimension, len(scores), bestRank, time.Since(start).Milliseconds())
	printTop(scores, topK)
	printTargetWindow(scores, bestRank, 3)
	fmt.Println()
	return nil
}

func printTop(scores []scoredChunk, topK int) {
	if topK > len(scores) {
		topK = len(scores)
	}
	fmt.Println("top_results:")
	for i := 0; i < topK; i++ {
		printScore(i+1, scores[i])
	}
}

func printTargetWindow(scores []scoredChunk, bestRank, radius int) {
	if bestRank <= 0 {
		fmt.Println("target_window: not found")
		return
	}
	start := bestRank - radius
	if start < 1 {
		start = 1
	}
	end := bestRank + radius
	if end > len(scores) {
		end = len(scores)
	}
	fmt.Println("target_window:")
	for rank := start; rank <= end; rank++ {
		printScore(rank, scores[rank-1])
	}
}

func printScore(rank int, score scoredChunk) {
	marker := " "
	if score.Target {
		marker = "*"
	}
	fmt.Printf("%s %4d %.6f %s chunk=%d %s\n", marker, rank, score.Similarity, score.Chunk.Title, score.Chunk.ChunkIndex, snippet(score.Chunk.Text, 140))
}

func loadChunks(ctx context.Context, db *sql.DB, userKey, scope, collection string, limit int) ([]chunkRow, error) {
	limitClause := ""
	args := []any{userKey, strings.TrimSpace(scope), strings.TrimSpace(collection)}
	if limit > 0 {
		limitClause = " LIMIT $4"
		args = append(args, limit)
	}
	rows, err := db.QueryContext(ctx, `
		SELECT c.memory_id, c.id, c.user_key, i.title, i.source_uri, c.text, c.metadata_text, c.chunk_index
		FROM memory_chunks c
		JOIN memory_items i ON i.id = c.memory_id
		WHERE c.user_key = $1
			AND i.source_uri NOT LIKE '%/.trash/%'
			AND ($2 = '' OR i.metadata->>'scope' = $2)
			AND ($3 = '' OR i.metadata->>'collection' = $3)
		ORDER BY i.title ASC, c.chunk_index ASC
	`+limitClause, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var chunks []chunkRow
	for rows.Next() {
		var chunk chunkRow
		if err := rows.Scan(&chunk.MemoryID, &chunk.ChunkID, &chunk.UserKey, &chunk.Title, &chunk.SourceURI, &chunk.Text, &chunk.MetadataText, &chunk.ChunkIndex); err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, rows.Err()
}

func embedBatched(ctx context.Context, client *http.Client, model modelSpec, inputs []string, batchSize int) ([][]float64, error) {
	if batchSize <= 0 {
		batchSize = 32
	}
	var embeddings [][]float64
	for start := 0; start < len(inputs); start += batchSize {
		end := start + batchSize
		if end > len(inputs) {
			end = len(inputs)
		}
		batch, err := embed(ctx, client, model, inputs[start:end])
		if err != nil {
			return nil, fmt.Errorf("batch %d-%d: %w", start, end, err)
		}
		embeddings = append(embeddings, batch...)
	}
	return embeddings, nil
}

func embed(ctx context.Context, client *http.Client, model modelSpec, inputs []string) ([][]float64, error) {
	payload := embeddingRequest{Model: model.Model, Input: inputs, Dimensions: model.Dimensions}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, model.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if model.KeyEnv != "" {
		key := strings.TrimSpace(os.Getenv(model.KeyEnv))
		if key == "" {
			return nil, fmt.Errorf("missing API key env %s", model.KeyEnv)
		}
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embedding status %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}
	var decoded embeddingResponse
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return nil, err
	}
	if len(decoded.Data) != len(inputs) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(decoded.Data), len(inputs))
	}
	embeddings := make([][]float64, len(decoded.Data))
	for i := range decoded.Data {
		if len(decoded.Data[i].Embedding) == 0 {
			return nil, fmt.Errorf("empty embedding at index %d", i)
		}
		embeddings[i] = decoded.Data[i].Embedding
	}
	return embeddings, nil
}

func parseModelSpecs(raw string) ([]modelSpec, error) {
	var specs []modelSpec
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		fields := strings.Split(part, "|")
		if len(fields) != 5 {
			return nil, fmt.Errorf("model spec %q must have 5 fields: name|url|model|dimensions|key_env", part)
		}
		dimensions := 0
		if strings.TrimSpace(fields[3]) != "" {
			parsed, err := strconv.Atoi(strings.TrimSpace(fields[3]))
			if err != nil {
				return nil, fmt.Errorf("invalid dimensions in %q: %w", part, err)
			}
			dimensions = parsed
		}
		specs = append(specs, modelSpec{
			Name:       strings.TrimSpace(fields[0]),
			URL:        strings.TrimSpace(fields[1]),
			Model:      strings.TrimSpace(fields[2]),
			Dimensions: dimensions,
			KeyEnv:     strings.TrimSpace(fields[4]),
		})
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("no model specs provided")
	}
	for _, spec := range specs {
		if spec.Name == "" || spec.URL == "" || spec.Model == "" {
			return nil, fmt.Errorf("model specs require name, url, and model")
		}
	}
	return specs, nil
}

func cosineSimilarity(left, right []float64) float64 {
	if len(left) == 0 || len(right) == 0 || len(left) != len(right) {
		return math.Inf(-1)
	}
	var dot, leftNorm, rightNorm float64
	for i := range left {
		dot += left[i] * right[i]
		leftNorm += left[i] * left[i]
		rightNorm += right[i] * right[i]
	}
	if leftNorm == 0 || rightNorm == 0 {
		return math.Inf(-1)
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
}

func evalEmbeddingInput(metadataText, chunk string) string {
	parts := []string{}
	if strings.TrimSpace(metadataText) != "" {
		parts = append(parts, "Document metadata:\n"+strings.TrimSpace(metadataText))
	}
	if strings.TrimSpace(chunk) != "" {
		parts = append(parts, "Passage:\n"+strings.TrimSpace(chunk))
	}
	return strings.Join(parts, "\n\n")
}

func snippet(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit]) + "..."
}

func countTargetChunks(chunks []chunkRow, target, expectedTitle string) int {
	count := 0
	for _, chunk := range chunks {
		if targetMatches(chunk, target, expectedTitle) {
			count++
		}
	}
	return count
}

func targetMatches(chunk chunkRow, target, expectedTitle string) bool {
	if expectedTitle = strings.TrimSpace(expectedTitle); expectedTitle != "" && strings.EqualFold(strings.TrimSpace(chunk.Title), expectedTitle) {
		return true
	}
	return strings.TrimSpace(target) != "" && containsFold(chunk.Text, target)
}

func containsFold(text, target string) bool {
	return strings.Contains(strings.ToLower(text), strings.ToLower(strings.TrimSpace(target)))
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstChunkUserKey(chunks []chunkRow) string {
	if len(chunks) == 0 {
		return ""
	}
	return chunks[0].UserKey
}

func safeIdent(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", fmt.Errorf("identifier is required")
	}
	for i, r := range value {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' {
			if i == 0 && r >= '0' && r <= '9' {
				return "", fmt.Errorf("identifier %q cannot start with a number", value)
			}
			continue
		}
		return "", fmt.Errorf("identifier %q must contain only lowercase letters, numbers, and underscores", value)
	}
	return value, nil
}

func safeVectorType(value string) (string, error) {
	value = strings.TrimSpace(strings.ToLower(value))
	switch value {
	case "vector", "halfvec":
		return value, nil
	default:
		return "", fmt.Errorf("unsupported vector type %q", value)
	}
}

func vectorLiteral(values []float64) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.FormatFloat(value, 'f', -1, 64)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func envOrDefault(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func emptyAsAll(value string) string {
	if strings.TrimSpace(value) == "" {
		return "(all)"
	}
	return strings.TrimSpace(value)
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
