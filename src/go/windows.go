package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
)

const (
	passageWindowTargetRunes    = 1600
	passageWindowStrideRunes    = 800
	passageWindowSearchLimit    = 32
	passageWindowLeavesPerMatch = 4
	passageWindowCandidateLimit = 64
)

type passageWindow struct {
	ID              string
	NodeID          string
	WindowIndex     int
	StartChunkIndex int
	EndChunkIndex   int
	Text            string
	Embedding       []float64
}

type passageWindowSearch struct {
	Results         []SearchResult
	MatchedWindows  int
	ExpandedLeaves  int
	CanonicalLeaves int
	ElapsedMs       int64
}

func buildPassageWindows(chunks []textChunk, nodeIDs []string) []passageWindow {
	if len(chunks) < 2 || len(chunks) != len(nodeIDs) {
		return nil
	}
	windows := []passageWindow{}
	for groupStart := 0; groupStart < len(chunks); {
		groupEnd := groupStart + 1
		for groupEnd < len(chunks) && nodeIDs[groupEnd] == nodeIDs[groupStart] {
			groupEnd++
		}
		windows = append(windows, buildPassageWindowsForGroup(chunks, nodeIDs[groupStart], groupStart, groupEnd)...)
		groupStart = groupEnd
	}
	for index := range windows {
		windows[index].WindowIndex = index
	}
	return windows
}

func buildPassageWindowsForGroup(chunks []textChunk, nodeID string, groupStart, groupEnd int) []passageWindow {
	if groupEnd-groupStart < 2 {
		return nil
	}
	result := []passageWindow{}
	for start := groupStart; start < groupEnd-1; {
		end := start
		totalRunes := 0
		parts := []string{}
		for end < groupEnd && (end == start || totalRunes < passageWindowTargetRunes) {
			text := strings.TrimSpace(chunks[end].Text)
			if text != "" {
				parts = append(parts, text)
				totalRunes += len([]rune(text))
				if len(parts) > 1 {
					totalRunes += 2
				}
			}
			end++
		}
		if end-start < 2 {
			break
		}
		result = append(result, passageWindow{
			ID:              newID(),
			NodeID:          nodeID,
			StartChunkIndex: start,
			EndChunkIndex:   end - 1,
			Text:            strings.Join(parts, "\n\n"),
		})
		if end == groupEnd {
			break
		}
		removedRunes := 0
		nextStart := start
		for nextStart < end-1 && removedRunes < passageWindowStrideRunes {
			removedRunes += len([]rune(strings.TrimSpace(chunks[nextStart].Text))) + 2
			nextStart++
		}
		if nextStart <= start {
			nextStart = start + 1
		}
		start = nextStart
	}
	return result
}

func passageWindowEmbeddingText(title, sourceURI, text string) string {
	var result strings.Builder
	if title = strings.TrimSpace(title); title != "" {
		result.WriteString("Title: ")
		result.WriteString(title)
		result.WriteString("\n")
	}
	if sourceURI = strings.TrimSpace(sourceURI); sourceURI != "" {
		result.WriteString("Source: ")
		result.WriteString(sourceURI)
		result.WriteString("\n")
	}
	result.WriteString("\nPassage window:\n")
	result.WriteString(strings.TrimSpace(text))
	return result.String()
}

func assignPassageWindowEmbeddings(windows []passageWindow, embeddings [][]float64) error {
	if len(windows) != len(embeddings) {
		return fmt.Errorf("window embedding count mismatch: got %d want %d", len(embeddings), len(windows))
	}
	for index := range windows {
		windows[index].Embedding = embeddings[index]
	}
	return nil
}

func (s *Server) searchPassageWindows(
	ctx context.Context,
	userKey string,
	embedding []float64,
	req RetrieveRequest,
) (passageWindowSearch, error) {
	started := time.Now()
	if len(embedding) == 0 {
		return passageWindowSearch{}, nil
	}
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)
	rows, err := s.db.QueryContext(ctx, `
		WITH nearest_base AS (
			SELECT w.id, w.memory_id, w.node_id, w.start_chunk_index, w.end_chunk_index,
				w.text AS window_text,
				1 - (w.embedding <=> $2::vector) AS window_similarity
			FROM memory_passage_windows w
			JOIN memory_items i ON i.id = w.memory_id
			WHERE w.user_key = $1
				AND ($3 = '' OR i.title LIKE $3 || '%' OR i.source_uri LIKE '%' || $3 || '%' OR i.metadata->>'modified_at' LIKE $3 || '%')
				AND ($4 = '' OR i.metadata->>'scope' = $4)
				AND ($5 = '' OR i.metadata->>'collection' = $5)
				AND i.source_uri NOT LIKE '%/.trash/%'
			ORDER BY w.embedding <=> $2::vector
			LIMIT $6
		),
		nearest AS (
			SELECT *, ROW_NUMBER() OVER (ORDER BY window_similarity DESC, id) AS window_rank
			FROM nearest_base
		),
		expanded AS (
			SELECT n.id AS window_id, n.window_text, n.window_rank, n.window_similarity,
				n.start_chunk_index, n.end_chunk_index,
				c.memory_id, c.id AS chunk_id, i.title, c.text, c.context_text, c.metadata_text,
				i.kind, i.source_type, i.source_uri, i.tags::text AS tags,
				i.metadata::text AS metadata, c.metadata AS chunk_metadata,
				1 - (c.embedding <=> $2::vector) AS leaf_similarity,
				i.updated_at,
				ROW_NUMBER() OVER (
					PARTITION BY n.id
					ORDER BY c.embedding <=> $2::vector, c.chunk_index
				) AS window_leaf_rank
			FROM nearest n
			JOIN memory_chunks c
				ON c.memory_id = n.memory_id
				AND c.node_id = n.node_id
				AND c.chunk_index BETWEEN n.start_chunk_index AND n.end_chunk_index
			JOIN memory_items i ON i.id = c.memory_id
		),
		scored AS (
			SELECT *,
				(window_similarity * 0.70) + (leaf_similarity * 0.30) AS score
			FROM expanded
			WHERE window_leaf_rank <= $7
		),
		deduped AS (
			SELECT *,
				ROW_NUMBER() OVER (
					PARTITION BY chunk_id
					ORDER BY score DESC, window_rank, window_id
				) AS canonical_rank,
				COUNT(*) OVER (PARTITION BY chunk_id) AS window_match_count
			FROM scored
		)
		SELECT memory_id, chunk_id, title, text, context_text, metadata_text, kind,
			source_type, source_uri, tags, metadata,
			(chunk_metadata || jsonb_build_object(
				'window_id', window_id,
				'window_rank', window_rank,
				'window_similarity', window_similarity,
				'window_leaf_rank', window_leaf_rank,
				'window_match_count', window_match_count,
				'window_start_chunk', start_chunk_index,
				'window_end_chunk', end_chunk_index,
				'window_text', window_text
			))::text,
			score, score AS vector_similarity, 0.0 AS literal_score,
			false AS title_match, false AS source_match, false AS body_match, updated_at
		FROM deduped
		WHERE canonical_rank = 1
		ORDER BY score DESC, window_rank
		LIMIT $8
	`, userKey, vectorLiteral(embedding), timeFilter, scope, collection, passageWindowSearchLimit, passageWindowLeavesPerMatch, passageWindowCandidateLimit)
	if err != nil {
		return passageWindowSearch{}, err
	}
	results, err := scanSearchResults(rows, searchModeSemantic)
	if err != nil {
		return passageWindowSearch{}, err
	}
	matchedWindows := map[string]bool{}
	expandedLeaves := 0
	for index := range results {
		windowID := stringMapValue(results[index].ChunkMetadata, "window_id")
		matchedWindows[windowID] = true
		expandedLeaves += maxInt(1, intMapValueOrZero(results[index].ChunkMetadata, "window_match_count"))
		windowText := stringMapValue(results[index].ChunkMetadata, "window_text")
		delete(results[index].ChunkMetadata, "window_text")
		results[index].RerankContext = centerTextAroundPassage(windowText, results[index].Text, rerankWindowChars)
	}
	return passageWindowSearch{
		Results:         results,
		MatchedWindows:  len(matchedWindows),
		ExpandedLeaves:  expandedLeaves,
		CanonicalLeaves: len(results),
		ElapsedMs:       time.Since(started).Milliseconds(),
	}, nil
}

func centerTextAroundPassage(windowText, passage string, limit int) string {
	windowText = strings.TrimSpace(windowText)
	if limit <= 0 || len([]rune(windowText)) <= limit {
		return windowText
	}
	windowRunes := []rune(windowText)
	passageRunes := []rune(strings.TrimSpace(passage))
	start := 0
	if len(passageRunes) > 0 {
		if index := strings.Index(windowText, string(passageRunes)); index >= 0 {
			start = len([]rune(windowText[:index])) + len(passageRunes)/2 - limit/2
		}
	}
	start = maxInt(0, minInt(start, len(windowRunes)-limit))
	return strings.TrimSpace(string(windowRunes[start : start+limit]))
}

func intMapValueOrZero(values map[string]any, key string) int {
	value, _ := intMapValue(values, key)
	return value
}

func (s *Server) backfillPassageWindows(ctx context.Context) error {
	started := time.Now()
	rows, err := s.db.QueryContext(ctx, `
		SELECT i.id, i.user_key, i.title, i.source_uri
		FROM memory_items i
		WHERE EXISTS (
			SELECT 1
			FROM memory_chunks c
			WHERE c.memory_id = i.id
			GROUP BY c.node_id
			HAVING COUNT(*) >= 2
		)
			AND NOT EXISTS (SELECT 1 FROM memory_passage_windows w WHERE w.memory_id = i.id)
		ORDER BY i.user_key, i.id
	`)
	if err != nil {
		return err
	}
	type pendingMemory struct {
		id, userKey, title, sourceURI string
	}
	pending := []pendingMemory{}
	for rows.Next() {
		var item pendingMemory
		if err := rows.Scan(&item.id, &item.userKey, &item.title, &item.sourceURI); err != nil {
			rows.Close()
			return err
		}
		pending = append(pending, item)
	}
	if err := rows.Close(); err != nil {
		return err
	}
	if len(pending) == 0 {
		log.Printf("memory.window_backfill status=complete memories=0 windows=0 total_ms=%d", time.Since(started).Milliseconds())
		return nil
	}
	log.Printf("memory.window_backfill status=start memories=%d", len(pending))
	totalWindows := 0
	for index, item := range pending {
		count, err := s.rebuildPassageWindowsForMemory(ctx, item.id, item.userKey, item.title, item.sourceURI)
		if err != nil {
			return fmt.Errorf("backfill windows for %s: %w", item.id, err)
		}
		totalWindows += count
		if (index+1)%10 == 0 || index+1 == len(pending) {
			log.Printf(
				"memory.window_backfill status=progress memories=%d/%d windows=%d total_ms=%d",
				index+1,
				len(pending),
				totalWindows,
				time.Since(started).Milliseconds(),
			)
		}
	}
	log.Printf("memory.window_backfill status=complete memories=%d windows=%d total_ms=%d", len(pending), totalWindows, time.Since(started).Milliseconds())
	return nil
}

func (s *Server) rebuildPassageWindowsForMemory(ctx context.Context, memoryID, userKey, title, sourceURI string) (int, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT chunk_index, text, node_id
		FROM memory_chunks
		WHERE memory_id = $1 AND user_key = $2
		ORDER BY chunk_index
	`, memoryID, userKey)
	if err != nil {
		return 0, err
	}
	chunks := []textChunk{}
	nodeIDs := []string{}
	for rows.Next() {
		var index int
		var text, nodeID string
		if err := rows.Scan(&index, &text, &nodeID); err != nil {
			rows.Close()
			return 0, err
		}
		chunks = append(chunks, textChunk{Text: text, Metadata: map[string]any{"chunk_index": index}})
		nodeIDs = append(nodeIDs, nodeID)
	}
	if err := rows.Close(); err != nil {
		return 0, err
	}
	windows := buildPassageWindows(chunks, nodeIDs)
	if len(windows) == 0 {
		return 0, nil
	}
	inputs := make([]string, len(windows))
	for index := range windows {
		inputs[index] = passageWindowEmbeddingText(title, sourceURI, windows[index].Text)
	}
	embeddings, err := s.embedBatched(ctx, inputs)
	if err != nil {
		return 0, err
	}
	if err := assignPassageWindowEmbeddings(windows, embeddings); err != nil {
		return 0, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `DELETE FROM memory_passage_windows WHERE memory_id = $1 AND user_key = $2`, memoryID, userKey); err != nil {
		return 0, err
	}
	if err := insertPassageWindows(ctx, tx, memoryID, userKey, windows); err != nil {
		return 0, err
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return len(windows), nil
}

func insertPassageWindows(ctx context.Context, tx *sql.Tx, memoryID, userKey string, windows []passageWindow) error {
	for _, window := range windows {
		metadata, err := json.Marshal(map[string]any{
			"window_version":    1,
			"target_runes":      passageWindowTargetRunes,
			"stride_runes":      passageWindowStrideRunes,
			"start_chunk_index": window.StartChunkIndex,
			"end_chunk_index":   window.EndChunkIndex,
		})
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO memory_passage_windows (
				id, memory_id, node_id, user_key, window_index,
				start_chunk_index, end_chunk_index, text, embedding, metadata, created_at
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::jsonb, NOW())
		`, window.ID, memoryID, window.NodeID, userKey, window.WindowIndex, window.StartChunkIndex, window.EndChunkIndex, window.Text, vectorLiteral(window.Embedding), string(metadata)); err != nil {
			return err
		}
	}
	return nil
}

func logPassageWindowBackfillError(err error) {
	if err != nil {
		log.Printf("memory.window_backfill status=error err=%q", err.Error())
	}
}
