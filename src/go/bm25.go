package main

import (
	"context"
	"database/sql"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

type bm25Document struct {
	userKey  string
	parentID string
	result   SearchResult
	terms    map[string]int
	length   int
}

type BM25Index struct {
	mu       sync.RWMutex
	docs     map[string]*bm25Document
	postings map[string]map[string]struct{}
}

func NewBM25Index() *BM25Index {
	return &BM25Index{
		docs:     map[string]*bm25Document{},
		postings: map[string]map[string]struct{}{},
	}
}

func (index *BM25Index) Load(ctx context.Context, db *sql.DB) error {
	rows, err := db.QueryContext(ctx, `
		SELECT c.user_key, c.parent_id, c.memory_id, c.id, i.title, c.text, c.context_text, c.metadata_text,
			i.kind, i.source_type, i.source_uri, i.tags::text, i.metadata::text,
			c.metadata::text, i.updated_at
		FROM memory_chunks c
		JOIN memory_items i ON i.id = c.memory_id
	`)
	if err != nil {
		return err
	}
	defer rows.Close()
	documents := make([]*bm25Document, 0)
	for rows.Next() {
		document, err := scanBM25Document(rows)
		if err != nil {
			return err
		}
		documents = append(documents, document)
	}
	if err := rows.Err(); err != nil {
		return err
	}

	index.mu.Lock()
	defer index.mu.Unlock()
	index.docs = map[string]*bm25Document{}
	index.postings = map[string]map[string]struct{}{}
	for _, document := range documents {
		index.addLocked(document)
	}
	return nil
}

func (index *BM25Index) RefreshMemory(ctx context.Context, db *sql.DB, userKey, memoryID string) error {
	rows, err := db.QueryContext(ctx, `
		SELECT c.user_key, c.parent_id, c.memory_id, c.id, i.title, c.text, c.context_text, c.metadata_text,
			i.kind, i.source_type, i.source_uri, i.tags::text, i.metadata::text,
			c.metadata::text, i.updated_at
		FROM memory_chunks c
		JOIN memory_items i ON i.id = c.memory_id
		WHERE c.user_key = $1 AND c.memory_id = $2
	`, userKey, memoryID)
	if err != nil {
		return err
	}
	defer rows.Close()
	documents := make([]*bm25Document, 0)
	for rows.Next() {
		document, err := scanBM25Document(rows)
		if err != nil {
			return err
		}
		documents = append(documents, document)
	}
	if err := rows.Err(); err != nil {
		return err
	}

	index.mu.Lock()
	defer index.mu.Unlock()
	index.removeMemoryLocked(memoryID)
	for _, document := range documents {
		index.addLocked(document)
	}
	return nil
}

func (index *BM25Index) RemoveMemory(memoryID string) {
	index.mu.Lock()
	defer index.mu.Unlock()
	index.removeMemoryLocked(memoryID)
}

func (index *BM25Index) Search(userKey string, req SearchRequest, limit int) []SearchResult {
	return index.search(userKey, req, limit, nil, 0)
}

func (index *BM25Index) SearchParents(userKey string, req SearchRequest, parentIDs []string, perParentLimit int) []SearchResult {
	if len(parentIDs) == 0 || perParentLimit <= 0 {
		return []SearchResult{}
	}
	allowedParents := make(map[string]bool, len(parentIDs))
	for _, parentID := range parentIDs {
		if parentID = strings.TrimSpace(parentID); parentID != "" {
			allowedParents[parentID] = true
		}
	}
	return index.search(userKey, req, len(allowedParents)*perParentLimit, allowedParents, perParentLimit)
}

func (index *BM25Index) search(userKey string, req SearchRequest, limit int, allowedParents map[string]bool, perParentLimit int) []SearchResult {
	terms := literalRecallTerms(req.Query)
	if len(terms) == 0 || limit <= 0 {
		return []SearchResult{}
	}
	searchMode := searchModeLiteral
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)

	index.mu.RLock()
	defer index.mu.RUnlock()
	eligible := make(map[string]*bm25Document)
	totalLength := 0
	for id, document := range index.docs {
		if document.userKey != userKey || !bm25DocumentMatches(document.result, req, timeFilter, scope, collection) {
			continue
		}
		if allowedParents != nil && !allowedParents[document.parentID] {
			continue
		}
		eligible[id] = document
		totalLength += document.length
	}
	if len(eligible) == 0 {
		return []SearchResult{}
	}

	queryTerms := make(map[string]bool, len(terms))
	for _, term := range terms {
		queryTerms[term] = true
	}
	documentFrequency := make(map[string]int, len(queryTerms))
	candidateIDs := map[string]bool{}
	for term := range queryTerms {
		for id := range index.postings[term] {
			if _, ok := eligible[id]; !ok {
				continue
			}
			documentFrequency[term]++
			candidateIDs[id] = true
		}
	}
	if len(candidateIDs) == 0 {
		return []SearchResult{}
	}
	averageLength := float64(totalLength) / float64(len(eligible))
	if averageLength <= 0 {
		averageLength = 1
	}

	type scored struct {
		document *bm25Document
		score    float64
	}
	scores := make([]scored, 0, len(candidateIDs))
	for id := range candidateIDs {
		document := eligible[id]
		score := 0.0
		for term := range queryTerms {
			tf := document.terms[term]
			if tf == 0 {
				continue
			}
			df := documentFrequency[term]
			idf := math.Log(1 + (float64(len(eligible)-df)+0.5)/(float64(df)+0.5))
			normalizedLength := 1 - bm25B + bm25B*(float64(document.length)/averageLength)
			score += idf * (float64(tf) * (bm25K1 + 1)) / (float64(tf) + bm25K1*normalizedLength)
		}
		scores = append(scores, scored{document: document, score: score})
	}
	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return scores[i].document.result.UpdatedAt.After(scores[j].document.result.UpdatedAt)
		}
		return scores[i].score > scores[j].score
	})

	groupLimit := searchResultRankLimit(req)
	if allowedParents != nil {
		groupLimit = perParentLimit
	}
	countByGroup := map[string]int{}
	results := make([]SearchResult, 0, limit)
	for _, item := range scores {
		groupID := item.document.result.MemoryID
		if allowedParents != nil {
			groupID = item.document.parentID
		}
		if countByGroup[groupID] >= groupLimit {
			continue
		}
		result := item.document.result
		if allowedParents != nil {
			result.ChunkMetadata = cloneAnyMap(result.ChunkMetadata)
			result.ChunkMetadata["parent_id"] = item.document.parentID
			result.ChunkMetadata["parent_bm25_rank"] = countByGroup[groupID] + 1
		}
		result.Score = item.score
		result.Similarity = 0
		result.ScoreParts = ScoreParts{
			Mode:         searchMode,
			LiteralScore: item.score,
			TitleMatch:   containsFold(result.Title, req.Query),
			SourceMatch:  containsFold(result.SourceURI, req.Query),
			BodyMatch:    containsFold(result.Text, req.Query),
		}
		results = append(results, result)
		countByGroup[groupID]++
		if len(results) >= limit {
			break
		}
	}
	return results
}

func (index *BM25Index) addLocked(document *bm25Document) {
	index.docs[document.result.ChunkID] = document
	for term := range document.terms {
		if index.postings[term] == nil {
			index.postings[term] = map[string]struct{}{}
		}
		index.postings[term][document.result.ChunkID] = struct{}{}
	}
}

func (index *BM25Index) removeMemoryLocked(memoryID string) {
	for id, document := range index.docs {
		if document.result.MemoryID != memoryID {
			continue
		}
		for term := range document.terms {
			delete(index.postings[term], id)
			if len(index.postings[term]) == 0 {
				delete(index.postings, term)
			}
		}
		delete(index.docs, id)
	}
}

func scanBM25Document(rows interface{ Scan(...any) error }) (*bm25Document, error) {
	var result SearchResult
	var userKey string
	var parentID sql.NullString
	var tagsRaw, metadataRaw, chunkMetadataRaw string
	if err := rows.Scan(
		&userKey,
		&parentID,
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
		&result.UpdatedAt,
	); err != nil {
		return nil, err
	}
	result.Tags = parseStringSlice(tagsRaw)
	result.Metadata = parseMap(metadataRaw)
	result.ChunkMetadata = parseMap(chunkMetadataRaw)
	terms := bm25TermFrequencies(bm25SearchText(result))
	return &bm25Document{userKey: userKey, parentID: parentID.String, result: result, terms: terms, length: termCount(terms)}, nil
}

func cloneAnyMap(source map[string]any) map[string]any {
	cloned := make(map[string]any, len(source)+2)
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}

func bm25SearchText(result SearchResult) string {
	return strings.Join([]string{result.MetadataText, result.Text}, " ")
}

func bm25TermFrequencies(text string) map[string]int {
	terms := map[string]int{}
	for _, token := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r))
	}) {
		if len([]rune(token)) < 2 || literalRecallStopword(token) {
			continue
		}
		terms[token]++
	}
	return terms
}

func termCount(terms map[string]int) int {
	total := 0
	for _, count := range terms {
		total += count
	}
	if total == 0 {
		return 1
	}
	return total
}

func bm25DocumentMatches(result SearchResult, req SearchRequest, timeFilter, scope, collection string) bool {
	if req.Kind != "" && result.Kind != strings.TrimSpace(req.Kind) {
		return false
	}
	if req.SourceType != "" && result.SourceType != strings.TrimSpace(req.SourceType) {
		return false
	}
	if timeFilter != "" && !strings.Contains(result.SourceURI, timeFilter) &&
		!strings.HasPrefix(strings.ToLower(result.Title), strings.ToLower(timeFilter)) &&
		!strings.HasPrefix(stringMapValue(result.Metadata, "modified_at"), timeFilter) {
		return false
	}
	if scope != "" && stringMapValue(result.Metadata, "scope") != scope {
		return false
	}
	if collection != "" && stringMapValue(result.Metadata, "collection") != collection {
		return false
	}
	if len(req.SourceTitles) > 0 {
		matched := false
		for _, title := range req.SourceTitles {
			if strings.EqualFold(strings.TrimSpace(result.Title), strings.TrimSpace(title)) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return !strings.Contains(result.SourceURI, "/.trash/")
}

func containsFold(value, query string) bool {
	return strings.Contains(strings.ToLower(value), strings.ToLower(strings.TrimSpace(query)))
}
