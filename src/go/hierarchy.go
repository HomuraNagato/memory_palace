package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/lib/pq"
)

const (
	hierarchyParentLimit       = 48
	hierarchyChildrenPerParent = 8
	hierarchyRoutedParents     = 12
	hierarchyRerankedChildren  = 20
	hierarchyVectorChildren    = 3
	hierarchyCandidateLimit    = 48
	hierarchyWindowRadius      = 3
	hierarchyWindowChunkChars  = 240
	hierarchySummaryShare      = 1
	hierarchyPooledShare       = 3
)

type retrievalParentNode struct {
	ID               string
	ParentID         string
	NodeType         string
	NodeKey          string
	Depth            int
	Ordinal          int
	Text             string
	MetadataText     string
	Summary          string
	Metadata         map[string]any
	ChunkIndexes     []int
	Embedding        []float64
	SummaryEmbedding []float64
}

type parentHit struct {
	ID           string
	MemoryID     string
	Summary      string
	MetadataText string
	Lane         string
	Rank         int
	Similarity   float64
	FusionScore  float64
}

func buildParentNodes(req UpsertMemoryRequest, chunks []textChunk) ([]retrievalParentNode, []string) {
	if len(chunks) == 0 {
		return nil, nil
	}
	parents := []retrievalParentNode{{
		ID:       newID(),
		NodeType: "document",
		NodeKey:  "document",
		Depth:    0,
		Ordinal:  0,
		Metadata: map[string]any{"node_type": "document", "node_key": "document", "label": req.Title},
	}}
	parentIDs := make([]string, len(chunks))
	byPath := map[string]int{"document": 0}
	byID := map[string]int{parents[0].ID: 0}
	for index, chunk := range chunks {
		path := "document"
		parentIndex := 0
		for _, part := range nodePathForChunk(chunk) {
			path += "/" + part.nodeType + ":" + part.nodeKey
			currentIndex, ok := byPath[path]
			if !ok {
				currentIndex = len(parents)
				byPath[path] = currentIndex
				metadata := cloneMap(part.metadata)
				metadata["node_type"] = part.nodeType
				metadata["node_key"] = path
				metadata["label"] = part.label
				parents = append(parents, retrievalParentNode{
					ID:       newID(),
					ParentID: parents[parentIndex].ID,
					NodeType: part.nodeType,
					NodeKey:  path,
					Depth:    parents[parentIndex].Depth + 1,
					Ordinal:  currentIndex,
					Metadata: metadata,
				})
				byID[parents[currentIndex].ID] = currentIndex
			}
			parentIndex = currentIndex
		}
		parentIDs[index] = parents[parentIndex].ID
		for nodeIndex := parentIndex; nodeIndex >= 0; {
			parent := &parents[nodeIndex]
			parent.ChunkIndexes = append(parent.ChunkIndexes, index)
			if parent.Text != "" {
				parent.Text += "\n\n"
			}
			parent.Text += strings.TrimSpace(chunk.Text)
			if parent.ParentID == "" {
				break
			}
			nodeIndex = byID[parent.ParentID]
		}
	}
	for index := range parents {
		parent := &parents[index]
		firstChunk := chunks[parent.ChunkIndexes[0]]
		parent.MetadataText = metadataTextForChunk(req, firstChunk, parent.ChunkIndexes[0], "", chunkContextMetadata{})
		if label := stringMapValue(parent.Metadata, "label"); label != "" {
			if parent.MetadataText != "" {
				parent.MetadataText += "\n\n"
			}
			parent.MetadataText += "Section: " + label
		}
		entities := rankedMentionedEntities(parent.Text, 64)
		parent.Metadata["entities"] = entities
		parent.Metadata["entity_context_version"] = 1
		parent.MetadataText = appendParentEntityContext(parent.MetadataText, entities)
	}
	return parents, parentIDs
}

type corpusNodePathPart struct {
	nodeType string
	nodeKey  string
	label    string
	metadata map[string]any
}

func nodePathForChunk(chunk textChunk) []corpusNodePathPart {
	parts := make([]corpusNodePathPart, 0, 3)
	if structured := structuredNodePath(chunk.Metadata["structure_path"]); len(structured) > 0 {
		return structured
	}
	if chapter := metadataValue(chunk.Metadata, "chapter"); chapter != "" {
		parts = append(parts, corpusNodePathPart{nodeType: "chapter", nodeKey: chapter, label: "Chapter " + chapter})
	} else if page := metadataValue(chunk.Metadata, "page"); page != "" {
		parts = append(parts, corpusNodePathPart{nodeType: "page", nodeKey: page, label: "Page " + page})
	}
	if heading := stringMapValue(chunk.Metadata, "heading"); heading != "" {
		parts = append(parts, corpusNodePathPart{nodeType: "section", nodeKey: heading, label: heading})
	}
	return parts
}

func structuredNodePath(value any) []corpusNodePathPart {
	entries, ok := value.([]map[string]any)
	if !ok {
		rawEntries, rawOK := value.([]any)
		if !rawOK {
			return nil
		}
		entries = make([]map[string]any, 0, len(rawEntries))
		for _, raw := range rawEntries {
			entry, entryOK := raw.(map[string]any)
			if entryOK {
				entries = append(entries, entry)
			}
		}
	}
	parts := make([]corpusNodePathPart, 0, len(entries))
	for _, entry := range entries {
		nodeType := stringMapValue(entry, "node_type")
		nodeKey := stringMapValue(entry, "node_key")
		label := stringMapValue(entry, "label")
		if nodeType == "" || nodeKey == "" || label == "" {
			continue
		}
		parts = append(parts, corpusNodePathPart{
			nodeType: nodeType,
			nodeKey:  nodeKey,
			label:    label,
			metadata: cloneMap(entry),
		})
	}
	return parts
}

func metadataValue(metadata map[string]any, key string) string {
	value := strings.TrimSpace(fmt.Sprint(metadata[key]))
	if value == "<nil>" {
		return ""
	}
	return value
}

func rankedMentionedEntities(text string, limit int) []string {
	type entityCount struct {
		value string
		count int
		first int
	}
	byKey := map[string]*entityCount{}
	for index, entity := range extractMentionedEntities(text) {
		entity = strings.Join(strings.Fields(entity), " ")
		if entity == "" {
			continue
		}
		key := strings.ToLower(entity)
		entry := byKey[key]
		if entry == nil {
			entry = &entityCount{value: entity, first: index}
			byKey[key] = entry
		}
		entry.count++
	}
	values := make([]entityCount, 0, len(byKey))
	for _, entry := range byKey {
		if entry.count < 2 && !strings.Contains(entry.value, " ") {
			continue
		}
		values = append(values, *entry)
	}
	sort.SliceStable(values, func(i, j int) bool {
		if values[i].count == values[j].count {
			return values[i].first < values[j].first
		}
		return values[i].count > values[j].count
	})
	if limit > 0 && len(values) > limit {
		values = values[:limit]
	}
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = value.value
	}
	return result
}

func appendParentEntityContext(metadataText string, entities []string) string {
	metadataText = strings.TrimSpace(metadataText)
	if len(entities) == 0 {
		return metadataText
	}
	if metadataText != "" {
		metadataText += "\n\n"
	}
	return metadataText + "Section entities: " + strings.Join(entities, ", ")
}

func (s *Server) backfillParentEntityContext(ctx context.Context) error {
	start := time.Now()
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, text, metadata_text
		FROM memory_nodes
		WHERE metadata->>'entity_context_version' IS DISTINCT FROM '1'
	`)
	if err != nil {
		return err
	}
	type parentContext struct {
		id, text, metadataText string
	}
	parents := []parentContext{}
	for rows.Next() {
		var parent parentContext
		if err := rows.Scan(&parent.id, &parent.text, &parent.metadataText); err != nil {
			rows.Close()
			return err
		}
		parents = append(parents, parent)
	}
	if err := rows.Close(); err != nil {
		return err
	}
	for _, parent := range parents {
		entities := rankedMentionedEntities(parent.text, 64)
		entitiesJSON, err := json.Marshal(entities)
		if err != nil {
			return err
		}
		if _, err := s.db.ExecContext(ctx, `
			UPDATE memory_nodes
			SET metadata_text = $2,
				metadata = jsonb_set(jsonb_set(metadata, '{entities}', $3::jsonb), '{entity_context_version}', '1'::jsonb)
			WHERE id = $1
		`, parent.id, appendParentEntityContext(parent.metadataText, entities), string(entitiesJSON)); err != nil {
			return err
		}
	}
	log.Printf("memory.parent_entity_backfill status=done parents=%d ms=%d", len(parents), time.Since(start).Milliseconds())
	return nil
}

func assignParentEmbeddings(parents []retrievalParentNode, childEmbeddings [][]float64) {
	for index := range parents {
		parent := &parents[index]
		if len(parent.ChunkIndexes) == 0 {
			continue
		}
		var sum []float64
		count := 0
		for _, chunkIndex := range parent.ChunkIndexes {
			if chunkIndex < 0 || chunkIndex >= len(childEmbeddings) || len(childEmbeddings[chunkIndex]) == 0 {
				continue
			}
			if sum == nil {
				sum = make([]float64, len(childEmbeddings[chunkIndex]))
			}
			if len(sum) != len(childEmbeddings[chunkIndex]) {
				continue
			}
			for dimension, value := range childEmbeddings[chunkIndex] {
				sum[dimension] += value
			}
			count++
		}
		if count == 0 {
			continue
		}
		for dimension := range sum {
			sum[dimension] /= float64(count)
		}
		parent.Embedding = normalizeVector(sum)
	}
}

func (s *Server) embedParentSummaries(ctx context.Context, parents []retrievalParentNode) map[string][]float64 {
	inputs := make([]string, 0, len(parents))
	ids := make([]string, 0, len(parents))
	for _, parent := range parents {
		if strings.TrimSpace(parent.Summary) == "" {
			continue
		}
		inputs = append(inputs, parentEmbeddingInput(parent))
		ids = append(ids, parent.ID)
	}
	if len(inputs) == 0 {
		return nil
	}
	embeddings, err := s.embedBatched(ctx, inputs)
	if err != nil || len(embeddings) != len(ids) {
		if err != nil {
			log.Printf("memory.parent_summary_embedding status=fallback summaries=%d err=%q", len(inputs), err.Error())
		} else {
			log.Printf("memory.parent_summary_embedding status=fallback summaries=%d got=%d want=%d", len(inputs), len(embeddings), len(ids))
		}
		return nil
	}
	result := make(map[string][]float64, len(ids))
	for index, id := range ids {
		result[id] = embeddings[index]
	}
	return result
}

func applyParentSummaryEmbeddings(parents []retrievalParentNode, summaryEmbeddings map[string][]float64) {
	for index := range parents {
		parents[index].SummaryEmbedding = summaryEmbeddings[parents[index].ID]
	}
}

func parentEmbeddingInput(parent retrievalParentNode) string {
	if strings.TrimSpace(parent.Summary) != "" {
		return "Document structure summary:\n" + strings.TrimSpace(parent.Summary)
	}
	return ""
}

func normalizeVector(values []float64) []float64 {
	result := append([]float64(nil), values...)
	var norm float64
	for _, value := range result {
		norm += value * value
	}
	if norm == 0 {
		return result
	}
	norm = math.Sqrt(norm)
	for index := range result {
		result[index] /= norm
	}
	return result
}

func (s *Server) searchParentChildren(ctx context.Context, userKey, query string, req SearchRequest, parentLimit, childLimit int, rerank bool) ([]SearchResult, error) {
	embedding := req.Embedding
	var err error
	if len(embedding) == 0 {
		embedding, err = s.embedOne(ctx, query, req.RequestID)
	}
	if err != nil {
		return nil, err
	}
	childParents, err := s.searchParentHits(ctx, userKey, embedding, req, "embedding", parentLimit)
	if err != nil {
		return nil, err
	}
	summaryParents := []parentHit{}
	if !s.summaryRoutingDisabled {
		summaryParents, err = s.searchParentHits(ctx, userKey, embedding, req, "summary_embedding", parentLimit)
		if err != nil {
			return nil, err
		}
	}
	parentByID := map[string]parentHit{}
	for _, parent := range append(childParents, summaryParents...) {
		existing, ok := parentByID[parent.ID]
		if !ok {
			existing = parent
		}
		if parent.Similarity > existing.Similarity {
			existing.Similarity = parent.Similarity
		}
		existing.FusionScore += reciprocalRankScore(parent.Rank)
		parentByID[parent.ID] = existing
	}
	parentLaneRanks := make(map[string]map[string]int, len(parentByID))
	for _, parent := range append(childParents, summaryParents...) {
		if parentLaneRanks[parent.ID] == nil {
			parentLaneRanks[parent.ID] = map[string]int{}
		}
		parentLaneRanks[parent.ID][parent.Lane] = parent.Rank
	}
	parents := make([]parentHit, 0, len(parentByID))
	for _, parent := range parentByID {
		parents = append(parents, parent)
	}
	sort.SliceStable(parents, func(i, j int) bool {
		if parents[i].FusionScore == parents[j].FusionScore {
			return parents[i].Similarity > parents[j].Similarity
		}
		return parents[i].FusionScore > parents[j].FusionScore
	})
	routedParentLimit := minInt(parentLimit, hierarchyRoutedParents)
	parents = selectRoutedParents(childParents, summaryParents, parents, routedParentLimit)
	parentIDs := make([]string, 0, len(parents))
	parentScores := make(map[string]float64, len(parents))
	parentFusionScores := make(map[string]float64, len(parents))
	parentRanks := make(map[string]int, len(parents))
	parentSummaries := make(map[string]string, len(parents))
	for index, parent := range parents {
		parentIDs = append(parentIDs, parent.ID)
		parentScores[parent.ID] = parent.Similarity
		parentFusionScores[parent.ID] = parent.FusionScore
		parentRanks[parent.ID] = index + 1
		parentSummaries[parent.ID] = parent.Summary
	}
	if len(parentIDs) == 0 {
		return nil, nil
	}

	childRows, err := s.db.QueryContext(ctx, `
		WITH RECURSIVE descendants(route_node_id, node_id) AS (
			SELECT id, id
			FROM memory_nodes
			WHERE user_key = $1 AND id = ANY($3)
			UNION ALL
			SELECT d.route_node_id, n.id
			FROM descendants d
			JOIN memory_nodes n ON n.parent_node_id = d.node_id
			WHERE n.user_key = $1
		),
		ranked AS (
			SELECT c.memory_id, c.id AS chunk_id, i.title, c.text, c.context_text, c.metadata_text,
				i.kind, i.source_type, i.source_uri, i.tags::text AS tags,
				i.metadata::text AS metadata, c.metadata::text AS chunk_metadata,
				1 - (c.embedding <=> $2::vector) AS child_similarity,
				c.node_id, d.route_node_id, i.updated_at,
				ROW_NUMBER() OVER (PARTITION BY d.route_node_id ORDER BY c.embedding <=> $2::vector, c.chunk_index ASC) AS child_rank
			FROM memory_chunks c
			JOIN descendants d ON d.node_id = c.node_id
			JOIN memory_items i ON i.id = c.memory_id
			WHERE c.user_key = $1
				AND i.source_uri NOT LIKE '%/.trash/%'
		)
		SELECT memory_id, chunk_id, title, text, context_text, metadata_text, kind, source_type, source_uri,
			tags, metadata, chunk_metadata, child_similarity, node_id, route_node_id, child_rank, updated_at
		FROM ranked
		WHERE child_rank <= $4
		ORDER BY route_node_id, child_rank
	`, userKey, vectorLiteral(embedding), pq.Array(parentIDs), childLimit)
	if err != nil {
		return nil, err
	}
	defer childRows.Close()
	results := make([]SearchResult, 0, len(parentIDs)*childLimit)
	for childRows.Next() {
		var result SearchResult
		var tagsRaw, metadataRaw, chunkMetadataRaw, nodeID, parentID string
		var childSimilarity float64
		var childRank int
		if err := childRows.Scan(
			&result.MemoryID, &result.ChunkID, &result.Title, &result.Text, &result.ContextText,
			&result.MetadataText, &result.Kind, &result.SourceType, &result.SourceURI,
			&tagsRaw, &metadataRaw, &chunkMetadataRaw, &childSimilarity, &nodeID, &parentID, &childRank, &result.UpdatedAt,
		); err != nil {
			return nil, err
		}
		result.Tags = parseStringSlice(tagsRaw)
		result.Metadata = parseMap(metadataRaw)
		result.ChunkMetadata = parseMap(chunkMetadataRaw)
		result.ChunkMetadata["node_id"] = nodeID
		result.ChunkMetadata["route_node_id"] = parentID
		result.ChunkMetadata["hierarchy_parent_hit"] = true
		result.ChunkMetadata["parent_similarity"] = parentScores[parentID]
		result.ChunkMetadata["parent_fusion_score"] = parentFusionScores[parentID]
		result.ChunkMetadata["parent_rank"] = parentRanks[parentID]
		result.ChunkMetadata["child_rank"] = childRank
		if rank := parentLaneRanks[parentID]["embedding"]; rank > 0 {
			result.ChunkMetadata["parent_pooled_rank"] = rank
		}
		if rank := parentLaneRanks[parentID]["summary_embedding"]; rank > 0 {
			result.ChunkMetadata["parent_summary_rank"] = rank
		}
		result.ParentSummary = parentSummaries[parentID]
		result.Score = childSimilarity
		result.Similarity = childSimilarity
		result.ScoreParts.Mode = searchModeSemantic
		result.ScoreParts.VectorSimilarity = childSimilarity
		results = append(results, result)
	}
	if err := childRows.Err(); err != nil {
		return nil, err
	}
	if s.bm25 != nil {
		nodeRoutes, err := s.descendantLeafNodeRoutes(ctx, userKey, parentIDs)
		if err != nil {
			return nil, err
		}
		nodeIDs := make([]string, 0, len(nodeRoutes))
		for nodeID := range nodeRoutes {
			nodeIDs = append(nodeIDs, nodeID)
		}
		lexicalChildren := s.bm25.SearchNodes(userKey, SearchRequest{
			Query:      query,
			TimeFilter: req.TimeFilter,
			Scope:      req.Scope,
			Collection: req.Collection,
		}, nodeIDs, childLimit)
		countByRoute := map[string]int{}
		filtered := lexicalChildren[:0]
		for index := range lexicalChildren {
			result := &lexicalChildren[index]
			nodeID := stringMapValue(result.ChunkMetadata, "node_id")
			parentID := nodeRoutes[nodeID]
			if parentID == "" || countByRoute[parentID] >= childLimit {
				continue
			}
			countByRoute[parentID]++
			result.ChunkMetadata["route_node_id"] = parentID
			result.ChunkMetadata["hierarchy_parent_hit"] = true
			result.ChunkMetadata["parent_similarity"] = parentScores[parentID]
			result.ChunkMetadata["parent_fusion_score"] = parentFusionScores[parentID]
			result.ChunkMetadata["parent_rank"] = parentRanks[parentID]
			result.ParentSummary = parentSummaries[parentID]
			filtered = append(filtered, *result)
		}
		results = fuseHierarchyChildren(results, filtered)
	}
	if len(results) > hierarchyCandidateLimit {
		results = results[:hierarchyCandidateLimit]
	}
	if !rerank || len(results) == 0 {
		return results, nil
	}
	selected, err := s.rerankHierarchyChildren(ctx, query, req.RequestID, results, hierarchyRerankedChildren)
	if err != nil {
		return nil, err
	}
	return s.expandHierarchyWindows(ctx, userKey, selected, hierarchyWindowRadius)
}

func selectRoutedParents(
	pooledParents,
	summaryParents,
	fusedParents []parentHit,
	limit int,
) []parentHit {
	if limit <= 0 {
		return nil
	}
	merged := make(map[string]parentHit, len(fusedParents))
	for _, parent := range fusedParents {
		merged[parent.ID] = parent
	}
	selected := make([]parentHit, 0, limit)
	seen := make(map[string]bool, limit)
	appendParents := func(parents []parentHit, count int) {
		for _, parent := range parents {
			if len(selected) >= limit || count <= 0 {
				return
			}
			if seen[parent.ID] {
				continue
			}
			candidate, ok := merged[parent.ID]
			if !ok {
				continue
			}
			seen[parent.ID] = true
			selected = append(selected, candidate)
			count--
		}
	}
	totalShare := hierarchyPooledShare + hierarchySummaryShare
	pooledQuota := (limit * hierarchyPooledShare) / totalShare
	appendParents(pooledParents, pooledQuota)
	appendParents(summaryParents, limit-pooledQuota)
	appendParents(fusedParents, limit-len(selected))
	sort.SliceStable(selected, func(i, j int) bool {
		if selected[i].FusionScore == selected[j].FusionScore {
			return selected[i].Similarity > selected[j].Similarity
		}
		return selected[i].FusionScore > selected[j].FusionScore
	})
	return selected
}

func (s *Server) expandHierarchyWindows(ctx context.Context, userKey string, results []SearchResult, radius int) ([]SearchResult, error) {
	if radius <= 0 {
		return results, nil
	}
	for index := range results {
		result := &results[index]
		chunkIndex, ok := intMapValue(result.ChunkMetadata, "chunk_index")
		if !ok {
			continue
		}
		nodeID := stringMapValue(result.ChunkMetadata, "node_id")
		rows, err := s.db.QueryContext(ctx, `
			SELECT id, text
			FROM memory_chunks
			WHERE user_key = $1
				AND memory_id = $2
				AND node_id = $3
				AND chunk_index BETWEEN $4 AND $5
			ORDER BY chunk_index ASC
		`, userKey, result.MemoryID, nodeID, chunkIndex-radius, chunkIndex+radius)
		if err != nil {
			return results, err
		}
		excerpts := []string{}
		chunkIDs := []string{}
		for rows.Next() {
			var chunkID, text string
			if err := rows.Scan(&chunkID, &text); err != nil {
				rows.Close()
				return results, err
			}
			excerpts = append(excerpts, truncateRunes(strings.TrimSpace(text), hierarchyWindowChunkChars))
			chunkIDs = append(chunkIDs, chunkID)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return results, err
		}
		rows.Close()
		if len(excerpts) <= 1 {
			continue
		}
		result.Text = strings.Join(excerpts, "\n\n---\n\n")
		result.ChunkMetadata["hierarchy_window_expanded"] = true
		result.ChunkMetadata["expanded_chunk_ids"] = chunkIDs
		result.ChunkMetadata["expanded_window"] = radius
	}
	return results, nil
}

func truncateRunes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return strings.TrimSpace(string(runes[:limit]))
}

func fuseHierarchyChildren(vectorChildren, lexicalChildren []SearchResult) []SearchResult {
	type fusedChild struct {
		result SearchResult
		score  float64
	}
	byID := make(map[string]*fusedChild, len(vectorChildren)+len(lexicalChildren))
	addLane := func(results []SearchResult) {
		for rank, result := range results {
			id := result.MemoryID + ":" + result.ChunkID
			entry := byID[id]
			if entry == nil {
				entry = &fusedChild{result: result}
				byID[id] = entry
			} else if result.ScoreParts.LiteralScore > 0 {
				entry.result.ScoreParts.LiteralScore = result.ScoreParts.LiteralScore
				entry.result.ChunkMetadata["node_bm25_rank"] = result.ChunkMetadata["node_bm25_rank"]
			}
			entry.score += reciprocalRankScore(rank + 1)
		}
	}
	addLane(vectorChildren)
	addLane(lexicalChildren)
	fused := make([]fusedChild, 0, len(byID))
	for _, entry := range byID {
		entry.result.Score = entry.score
		fused = append(fused, *entry)
	}
	sort.SliceStable(fused, func(i, j int) bool { return fused[i].score > fused[j].score })
	results := make([]SearchResult, len(fused))
	for index := range fused {
		results[index] = fused[index].result
	}
	return results
}

func (s *Server) rerankHierarchyChildren(ctx context.Context, query, requestID string, results []SearchResult, limit int) ([]SearchResult, error) {
	candidates := make([]rerankerCandidate, 0, len(results))
	byID := make(map[string]SearchResult, len(results))
	for _, result := range results {
		id := result.MemoryID + ":" + result.ChunkID
		candidates = append(candidates, rerankerCandidate{ID: id, Title: result.Title, Text: result.Text})
		byID[id] = result
	}
	if len(candidates) == 0 {
		return results, nil
	}
	start := time.Now()
	reranked, err := s.callReranker(ctx, query, requestID+":hierarchy", candidates)
	if err != nil {
		log.Printf("memory.retrieve.hierarchy_rerank request_id=%q status=fallback children=%d err=%q", requestID, len(candidates), err.Error())
		return results, nil
	}
	selected := make([]SearchResult, 0, minInt(limit, len(results)))
	for index, item := range reranked.Results {
		result, ok := byID[item.ID]
		if !ok {
			continue
		}
		result.ChunkMetadata["hierarchy_rerank_rank"] = index + 1
		if item.Score != nil {
			result.ChunkMetadata["hierarchy_rerank_score"] = *item.Score
		}
		selected = append(selected, result)
		if limit > 0 && len(selected) >= limit {
			break
		}
	}
	if len(selected) == 0 {
		log.Printf("memory.retrieve.hierarchy_rerank request_id=%q status=fallback children=%d err=%q", requestID, len(candidates), "reranker returned no known candidates")
		return results, nil
	}
	selected = retainHierarchyVectorChildren(selected, results, hierarchyVectorChildren)
	log.Printf("memory.retrieve.hierarchy_rerank request_id=%q status=ok children=%d selected=%d ms=%d model=%q", requestID, len(candidates), len(selected), time.Since(start).Milliseconds(), reranked.Model)
	return selected, nil
}

func retainHierarchyVectorChildren(selected, candidates []SearchResult, perParent int) []SearchResult {
	if perParent <= 0 {
		return selected
	}
	result := append([]SearchResult(nil), selected...)
	seen := make(map[string]bool, len(result))
	for _, candidate := range result {
		seen[candidate.MemoryID+":"+candidate.ChunkID] = true
	}
	for _, candidate := range candidates {
		childRank, ok := intMapValue(candidate.ChunkMetadata, "child_rank")
		if !ok || childRank <= 0 || childRank > perParent {
			continue
		}
		id := candidate.MemoryID + ":" + candidate.ChunkID
		if seen[id] {
			continue
		}
		candidate.ChunkMetadata["hierarchy_vector_rescue"] = true
		result = append(result, candidate)
		seen[id] = true
	}
	return result
}

func (s *Server) descendantLeafNodeRoutes(ctx context.Context, userKey string, routedNodeIDs []string) (map[string]string, error) {
	rows, err := s.db.QueryContext(ctx, `
		WITH RECURSIVE roots(route_node_id, route_rank) AS (
			SELECT n.id, requested.ordinality
			FROM unnest($2::text[]) WITH ORDINALITY AS requested(id, ordinality)
			JOIN memory_nodes n ON n.id = requested.id AND n.user_key = $1
		),
		descendants(route_node_id, route_rank, node_id) AS (
			SELECT route_node_id, route_rank, route_node_id
			FROM roots
			UNION ALL
			SELECT d.route_node_id, d.route_rank, n.id
			FROM descendants d
			JOIN memory_nodes n ON n.parent_node_id = d.node_id
			WHERE n.user_key = $1
		)
		SELECT DISTINCT ON (d.node_id) d.node_id, d.route_node_id
		FROM descendants d
		WHERE EXISTS (
			SELECT 1 FROM memory_chunks c
			WHERE c.user_key = $1 AND c.node_id = d.node_id
		)
		ORDER BY d.node_id, d.route_rank
	`, userKey, pq.Array(routedNodeIDs))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	routes := make(map[string]string)
	for rows.Next() {
		var nodeID, routeNodeID string
		if err := rows.Scan(&nodeID, &routeNodeID); err != nil {
			return nil, err
		}
		routes[nodeID] = routeNodeID
	}
	return routes, rows.Err()
}

func (s *Server) searchParentHits(ctx context.Context, userKey string, embedding []float64, req SearchRequest, column string, limit int) ([]parentHit, error) {
	if column != "embedding" && column != "summary_embedding" {
		return nil, fmt.Errorf("unsupported parent embedding column %q", column)
	}
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)
	nullFilter := ""
	if column == "summary_embedding" {
		nullFilter = "AND p.summary_embedding IS NOT NULL"
	}
	query := fmt.Sprintf(`
		SELECT p.id, p.memory_id, p.summary, p.metadata_text, 1 - (p.%s <=> $2::vector) AS similarity
		FROM memory_nodes p
		JOIN memory_items i ON i.id = p.memory_id
		WHERE p.user_key = $1
			%s
			AND ($3 = '' OR i.kind = $3)
			AND ($4 = '' OR i.source_type = $4)
			AND ($5 = '' OR i.source_uri LIKE '%%' || $5 || '%%' OR i.title LIKE $5 || '%%' OR i.metadata->>'modified_at' LIKE $5 || '%%')
			AND ($6 = '' OR i.metadata->>'scope' = $6)
			AND ($7 = '' OR i.metadata->>'collection' = $7)
				AND i.source_uri NOT LIKE '%%/.trash/%%'
			ORDER BY p.%s <=> $2::vector
			LIMIT $8
		`, column, nullFilter, column)
	rows, err := s.db.QueryContext(ctx, query, userKey, vectorLiteral(embedding), strings.TrimSpace(req.Kind), strings.TrimSpace(req.SourceType), timeFilter, scope, collection, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	parents := make([]parentHit, 0, limit)
	for rows.Next() {
		var parent parentHit
		if err := rows.Scan(&parent.ID, &parent.MemoryID, &parent.Summary, &parent.MetadataText, &parent.Similarity); err != nil {
			return nil, err
		}
		parent.Lane = column
		parent.Rank = len(parents) + 1
		parents = append(parents, parent)
	}
	return parents, rows.Err()
}
