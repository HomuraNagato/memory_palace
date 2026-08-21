package main

import (
	"context"
	"math"
)

const (
	documentRootCandidateLimit = 24
	documentRootReserve        = 2
	documentRootPassageDepth   = 6
)

type documentRootHit struct {
	MemoryID   string
	Rank       int
	Similarity float64
	Passages   int
}

func (s *Server) searchDocumentRoots(
	ctx context.Context,
	userKey string,
	embedding []float64,
	req RetrieveRequest,
	limit int,
) ([]documentRootHit, error) {
	if len(embedding) == 0 || limit <= 0 {
		return nil, nil
	}
	timeFilter := normalizeTimeFilter(req.TimeFilter)
	scope := normalizeMetadataFilter(req.Scope)
	collection := normalizeMetadataFilter(req.Collection)
	rows, err := s.db.QueryContext(ctx, `
		SELECT n.memory_id, 1 - (n.embedding <=> $2::vector) AS similarity, n.passage_count
		FROM memory_nodes n
		JOIN memory_items i ON i.id = n.memory_id
		WHERE n.user_key = $1
			AND n.node_type = 'document'
			AND ($3 = '' OR i.title LIKE $3 || '%' OR i.source_uri LIKE '%' || $3 || '%' OR i.metadata->>'modified_at' LIKE $3 || '%')
			AND ($4 = '' OR i.metadata->>'scope' = $4)
			AND ($5 = '' OR i.metadata->>'collection' = $5)
			AND i.source_uri NOT LIKE '%/.trash/%'
		ORDER BY n.embedding <=> $2::vector
		LIMIT $6
	`, userKey, vectorLiteral(embedding), timeFilter, scope, collection, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	hits := make([]documentRootHit, 0, limit)
	for rows.Next() {
		var hit documentRootHit
		if err := rows.Scan(&hit.MemoryID, &hit.Similarity, &hit.Passages); err != nil {
			return nil, err
		}
		hit.Rank = len(hits) + 1
		hits = append(hits, hit)
	}
	return hits, rows.Err()
}

func mergeDocumentRoutes(
	passageDocuments []firstPhaseDocument,
	rootHits []documentRootHit,
	limit int,
) []firstPhaseDocument {
	if limit <= 0 {
		return nil
	}
	passageByMemory := make(map[string]firstPhaseDocument, len(passageDocuments))
	for _, document := range passageDocuments {
		passageByMemory[document.memoryID] = document
	}
	for _, hit := range rootHits {
		document, ok := passageByMemory[hit.MemoryID]
		if !ok {
			continue
		}
		document.rootRank = hit.Rank
		document.rootSimilarity = hit.Similarity
		document.passageCount = hit.Passages
		passageByMemory[hit.MemoryID] = document
	}

	reserve := minInt(documentRootReserve, limit)
	passageLimit := minInt(len(passageDocuments), limit-reserve)
	selected := make([]firstPhaseDocument, 0, limit)
	seen := make(map[string]bool, limit)
	for _, original := range passageDocuments[:passageLimit] {
		document := passageByMemory[original.memoryID]
		selected = append(selected, document)
		seen[document.memoryID] = true
	}

	topPassageScore := 1.0
	if len(passageDocuments) > 0 && passageDocuments[0].score > 0 {
		topPassageScore = passageDocuments[0].score
	}
	topRootSimilarity := 0.0
	if len(rootHits) > 0 {
		topRootSimilarity = math.Max(0, rootHits[0].Similarity)
	}
	rootAdded := 0
	for _, hit := range rootHits {
		if rootAdded >= reserve || seen[hit.MemoryID] {
			continue
		}
		document, ok := passageByMemory[hit.MemoryID]
		if !ok {
			ratio := 1.0
			if topRootSimilarity > 0 {
				ratio = math.Max(0, hit.Similarity) / topRootSimilarity
			}
			document = firstPhaseDocument{
				memoryID: hit.MemoryID,
				score:    topPassageScore * 0.75 * ratio,
			}
		}
		document.rootRank = hit.Rank
		document.rootSimilarity = hit.Similarity
		document.passageCount = hit.Passages
		document.rootRouted = true
		selected = append(selected, document)
		seen[document.memoryID] = true
		rootAdded++
	}

	for _, original := range passageDocuments[passageLimit:] {
		if len(selected) >= limit {
			break
		}
		document := passageByMemory[original.memoryID]
		if seen[document.memoryID] {
			continue
		}
		selected = append(selected, document)
		seen[document.memoryID] = true
	}
	return selected
}

func rootRoutedMemoryIDs(documents []firstPhaseDocument) []string {
	result := make([]string, 0, documentRootReserve)
	for _, document := range documents {
		if document.rootRouted {
			result = append(result, document.memoryID)
		}
	}
	return result
}

func rootRoutedRepresentativeCandidates(
	documents []firstPhaseDocument,
	conditioned []CandidateSource,
) []CandidateSource {
	firstConditioned := make(map[string]CandidateSource)
	for _, candidate := range conditioned {
		if _, ok := firstConditioned[candidate.MemoryID]; !ok {
			firstConditioned[candidate.MemoryID] = candidate
		}
	}
	result := make([]CandidateSource, 0, documentRootReserve)
	for _, document := range documents {
		if !document.rootRouted {
			continue
		}
		var candidate CandidateSource
		if len(document.candidates) > 0 {
			candidate = document.candidates[0]
		} else {
			candidate = firstConditioned[document.memoryID]
		}
		if candidate.ID == "" {
			continue
		}
		candidate.ChunkMetadata = cloneMap(candidate.ChunkMetadata)
		candidate.ChunkMetadata["document_root_rank"] = document.rootRank
		candidate.ChunkMetadata["document_root_similarity"] = document.rootSimilarity
		result = append(result, candidate)
	}
	return result
}
