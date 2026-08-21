package main

import (
	"context"
	"log"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lib/pq"
)

const (
	conditionedSearchBudgetPerLane = 32
	conditionedDocumentWeightPower = 3
	conditionedPoolShare           = 1
	globalPoolShare                = 3
)

type conditionedRetrieval struct {
	Candidates []CandidateSource
	Budgets    []conditionedDocumentBudget
	ElapsedMs  int64
}

type conditionedDocumentBudget struct {
	MemoryID string
	Limit    int
}

func (s *Server) retrieveConditionedCandidates(
	ctx context.Context,
	userKey string,
	req RetrieveRequest,
	documents []firstPhaseDocument,
) conditionedRetrieval {
	started := time.Now()
	budgets := adaptiveConditionedBudgets(documents, conditionedSearchBudgetPerLane, firstPhaseDocumentLimit)
	if len(budgets) == 0 {
		return conditionedRetrieval{}
	}
	memoryIDs := conditionedBudgetMemoryIDs(budgets)
	maxDepth := maxConditionedBudget(budgets)

	type result struct {
		route   string
		spec    RetrievalSpec
		results []SearchResult
		err     error
	}
	out := make(chan result, 2)
	var wg sync.WaitGroup
	if normalizeSearchMode(req.Mode) != searchModeLiteral && len(req.Embedding) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			denseStarted := time.Now()
			results, err := s.searchConditionedDense(
				ctx,
				userKey,
				req.Embedding,
				memoryIDs,
				maxDepth,
			)
			results = filterConditionedResults(results, budgets)
			log.Printf(
				"memory.retrieve.conditioned_dense request_id=%q documents=%d depths=%v results=%d ms=%d err=%q",
				req.RequestID,
				len(memoryIDs),
				conditionedBudgetLimits(budgets),
				len(results),
				time.Since(denseStarted).Milliseconds(),
				errorString(err),
			)
			out <- result{
				route:   "conditioned_dense",
				spec:    RetrievalSpec{Query: strings.TrimSpace(req.Query), Mode: searchModeSemantic},
				results: results,
				err:     err,
			}
		}()
	}
	if s.bm25 != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			literalStarted := time.Now()
			results := s.bm25.SearchMemories(userKey, SearchRequest{
				Query:      strings.TrimSpace(req.Query),
				RequestID:  req.RequestID,
				TimeFilter: req.TimeFilter,
				Scope:      req.Scope,
				Collection: req.Collection,
			}, memoryIDs, maxDepth)
			results = filterConditionedResults(results, budgets)
			log.Printf(
				"memory.retrieve.conditioned_literal request_id=%q documents=%d depths=%v results=%d ms=%d",
				req.RequestID,
				len(memoryIDs),
				conditionedBudgetLimits(budgets),
				len(results),
				time.Since(literalStarted).Milliseconds(),
			)
			out <- result{
				route:   "conditioned_literal",
				spec:    RetrievalSpec{Query: strings.TrimSpace(req.Query), Mode: searchModeLiteral},
				results: results,
			}
		}()
	}
	wg.Wait()
	close(out)

	byID := make(map[string]*CandidateSource)
	response := conditionedRetrieval{}
	for current := range out {
		if current.err != nil {
			log.Printf(
				"memory.retrieve.conditioned request_id=%q route=%q status=fallback err=%q",
				req.RequestID,
				current.route,
				current.err.Error(),
			)
			continue
		}
		addRetrieveCandidates(byID, current.results, current.spec, current.route)
	}
	response.Candidates = candidateSources(byID)
	response.Budgets = budgets
	response.ElapsedMs = time.Since(started).Milliseconds()
	return response
}

func adaptiveConditionedBudgets(documents []firstPhaseDocument, total, documentLimit int) []conditionedDocumentBudget {
	if total <= 0 || len(documents) == 0 {
		return nil
	}
	documentLimit = minInt(documentLimit, minInt(total, len(documents)))
	documents = documents[:documentLimit]
	budgets := make([]conditionedDocumentBudget, documentLimit)
	weights := make([]float64, documentLimit)
	topScore := math.Max(0, documents[0].score)
	weightTotal := 0.0
	assignedBase := 0
	for index, document := range documents {
		base := 1
		if document.rootRouted {
			base = documentRootPassageDepth
			if document.passageCount > 0 {
				base = minInt(base, document.passageCount)
			}
		}
		base = minInt(base, total-assignedBase-(documentLimit-index-1))
		budgets[index] = conditionedDocumentBudget{MemoryID: document.memoryID, Limit: base}
		assignedBase += base
		ratio := 1.0
		if topScore > 0 {
			ratio = math.Max(0, document.score) / topScore
		}
		weights[index] = math.Pow(ratio, conditionedDocumentWeightPower)
		weightTotal += weights[index]
	}
	remaining := total - assignedBase
	if remaining <= 0 {
		return budgets
	}
	if weightTotal <= 0 {
		weightTotal = float64(documentLimit)
		for index := range weights {
			weights[index] = 1
		}
	}
	type remainder struct {
		index int
		value float64
	}
	remainders := make([]remainder, 0, documentLimit)
	assigned := 0
	for index, weight := range weights {
		exact := float64(remaining) * weight / weightTotal
		whole := int(math.Floor(exact))
		budgets[index].Limit += whole
		assigned += whole
		remainders = append(remainders, remainder{index: index, value: exact - float64(whole)})
	}
	sort.SliceStable(remainders, func(i, j int) bool {
		if remainders[i].value == remainders[j].value {
			return remainders[i].index < remainders[j].index
		}
		return remainders[i].value > remainders[j].value
	})
	for index := 0; index < remaining-assigned; index++ {
		budgets[remainders[index].index].Limit++
	}
	return budgets
}

func conditionedBudgetMemoryIDs(budgets []conditionedDocumentBudget) []string {
	result := make([]string, 0, len(budgets))
	for _, budget := range budgets {
		result = append(result, budget.MemoryID)
	}
	return result
}

func conditionedBudgetLimits(budgets []conditionedDocumentBudget) []int {
	result := make([]int, 0, len(budgets))
	for _, budget := range budgets {
		result = append(result, budget.Limit)
	}
	return result
}

func maxConditionedBudget(budgets []conditionedDocumentBudget) int {
	limit := 0
	for _, budget := range budgets {
		limit = maxInt(limit, budget.Limit)
	}
	return limit
}

func filterConditionedResults(results []SearchResult, budgets []conditionedDocumentBudget) []SearchResult {
	limits := make(map[string]int, len(budgets))
	for _, budget := range budgets {
		limits[budget.MemoryID] = budget.Limit
	}
	filtered := make([]SearchResult, 0, len(results))
	for _, result := range results {
		rank, ok := intMapValue(result.ChunkMetadata, "conditioned_rank")
		if !ok || rank <= limits[result.MemoryID] {
			filtered = append(filtered, result)
		}
	}
	return filtered
}

func (s *Server) searchConditionedDense(
	ctx context.Context,
	userKey string,
	embedding []float64,
	memoryIDs []string,
	perMemoryLimit int,
) ([]SearchResult, error) {
	if len(embedding) == 0 || len(memoryIDs) == 0 || perMemoryLimit <= 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `
		WITH ranked AS (
			SELECT c.memory_id, c.id AS chunk_id, i.title, c.text, c.context_text, c.metadata_text,
				i.kind, i.source_type, i.source_uri, i.tags::text AS tags,
				i.metadata::text AS metadata, c.metadata::text AS chunk_metadata,
				1 - (c.embedding <=> $2::vector) AS vector_similarity,
				i.updated_at,
				ROW_NUMBER() OVER (
					PARTITION BY c.memory_id
					ORDER BY c.embedding <=> $2::vector, c.chunk_index
				) AS conditioned_rank
			FROM memory_chunks c
			JOIN memory_items i ON i.id = c.memory_id
			WHERE c.user_key = $1
				AND c.memory_id = ANY($3::text[])
				AND i.source_uri NOT LIKE '%/.trash/%'
		)
		SELECT memory_id, chunk_id, title, text, context_text, metadata_text, kind,
			source_type, source_uri, tags, metadata,
			(chunk_metadata::jsonb || jsonb_build_object('conditioned_rank', conditioned_rank))::text,
			vector_similarity AS score, vector_similarity, 0.0 AS literal_score,
			false AS title_match, false AS source_match, false AS body_match, updated_at
		FROM ranked
		WHERE conditioned_rank <= $4
		ORDER BY vector_similarity DESC
	`, userKey, vectorLiteral(embedding), pq.Array(memoryIDs), perMemoryLimit)
	if err != nil {
		return nil, err
	}
	return scanSearchResults(rows, searchModeSemantic)
}

func documentRepresentativeCandidates(documents []firstPhaseDocument, limit int) []CandidateSource {
	if limit <= 0 || limit > len(documents) {
		limit = len(documents)
	}
	result := make([]CandidateSource, 0, limit)
	for index, document := range documents[:limit] {
		if len(document.candidates) == 0 {
			continue
		}
		candidate := document.candidates[0]
		candidate.ChunkMetadata = cloneMap(candidate.ChunkMetadata)
		candidate.ChunkMetadata["first_phase_document_rank"] = index + 1
		candidate.ChunkMetadata["first_phase_document_score"] = document.score
		result = append(result, candidate)
	}
	return result
}

func assembleRerankPool(
	globalCandidates []CandidateSource,
	conditionedCandidates []CandidateSource,
	documents []firstPhaseDocument,
	limit int,
) []CandidateSource {
	return assembleRerankPoolWithScoring(globalCandidates, conditionedCandidates, documents, limit, true)
}

func assembleRerankPoolWithScoring(
	globalCandidates []CandidateSource,
	conditionedCandidates []CandidateSource,
	documents []firstPhaseDocument,
	limit int,
	scoreFeatures bool,
) []CandidateSource {
	if limit <= 0 {
		return nil
	}
	merged := make(map[string]CandidateSource, len(globalCandidates)+len(conditionedCandidates))
	for _, candidate := range globalCandidates {
		merged[candidate.ID] = candidate
	}
	for _, candidate := range conditionedCandidates {
		if existing, ok := merged[candidate.ID]; ok {
			existing.RetrievalScore += candidate.RetrievalScore
			existing.Retrievals = append(existing.Retrievals, candidate.Retrievals...)
			if existing.ParentSummary == "" {
				existing.ParentSummary = candidate.ParentSummary
			}
			merged[candidate.ID] = existing
			continue
		}
		merged[candidate.ID] = candidate
	}

	documentRanks := make(map[string]int, len(documents))
	documentScores := make(map[string]float64, len(documents))
	for index, document := range documents {
		documentRanks[document.memoryID] = index + 1
		documentScores[document.memoryID] = document.score
	}
	withDocumentMetadata := func(candidate CandidateSource) CandidateSource {
		candidate.ChunkMetadata = cloneMap(candidate.ChunkMetadata)
		candidate.ChunkMetadata["first_phase_document_rank"] = documentRanks[candidate.MemoryID]
		candidate.ChunkMetadata["first_phase_document_score"] = documentScores[candidate.MemoryID]
		return candidate
	}
	allMerged := make([]CandidateSource, 0, len(merged))
	for _, candidate := range merged {
		allMerged = append(allMerged, withDocumentMetadata(candidate))
	}
	if scoreFeatures {
		allMerged = rankFirstPhaseCandidates(allMerged, documents)
	} else {
		sortRetrieveCandidates(allMerged)
	}
	for _, candidate := range allMerged {
		merged[candidate.ID] = candidate
	}

	conditionedByMemory := make(map[string][]CandidateSource)
	for _, candidate := range conditionedCandidates {
		candidate = merged[candidate.ID]
		conditionedByMemory[candidate.MemoryID] = append(conditionedByMemory[candidate.MemoryID], candidate)
	}
	for memoryID := range conditionedByMemory {
		sort.SliceStable(conditionedByMemory[memoryID], func(i, j int) bool {
			left := conditionedByMemory[memoryID][i].FirstPhaseScore
			right := conditionedByMemory[memoryID][j].FirstPhaseScore
			if left == right {
				return conditionedRetrievalScore(conditionedByMemory[memoryID][i]) > conditionedRetrievalScore(conditionedByMemory[memoryID][j])
			}
			return left > right
		})
	}

	result := make([]CandidateSource, 0, limit)
	seen := make(map[string]bool, limit)
	countByMemory := make(map[string]int)
	appendCandidate := func(candidate CandidateSource) bool {
		if len(result) >= limit || seen[candidate.ID] {
			return false
		}
		seen[candidate.ID] = true
		countByMemory[candidate.MemoryID]++
		result = append(result, candidate)
		return true
	}

	conditionedQuota := (limit * conditionedPoolShare) / (conditionedPoolShare + globalPoolShare)
	poolBudgets := adaptiveConditionedBudgets(documents, conditionedQuota, firstPhaseDocumentLimit)
	for _, budget := range poolBudgets {
		group := conditionedByMemory[budget.MemoryID]
		for _, candidate := range group {
			if countByMemory[budget.MemoryID] >= budget.Limit || len(result) >= conditionedQuota {
				break
			}
			appendCandidate(candidate)
		}
	}
	if len(result) < conditionedQuota {
		remaining := make([]CandidateSource, 0, len(conditionedCandidates))
		for _, candidate := range conditionedCandidates {
			remaining = append(remaining, merged[candidate.ID])
		}
		sortRetrieveCandidates(remaining)
		for _, candidate := range remaining {
			if len(result) >= conditionedQuota {
				break
			}
			appendCandidate(merged[candidate.ID])
		}
	}

	globalRanked := make([]CandidateSource, 0, len(globalCandidates))
	for _, candidate := range globalCandidates {
		globalRanked = append(globalRanked, merged[candidate.ID])
	}
	sortRetrieveCandidates(globalRanked)
	for _, candidate := range globalRanked {
		if len(result) >= limit {
			break
		}
		appendCandidate(merged[candidate.ID])
	}
	if len(result) < limit {
		for _, candidate := range allMerged {
			if len(result) >= limit {
				break
			}
			appendCandidate(candidate)
		}
	}
	return result
}

func conditionedRetrievalScore(candidate CandidateSource) float64 {
	score := 0.0
	for _, hit := range candidate.Retrievals {
		if strings.HasPrefix(hit.Route, "conditioned_") {
			score += hit.FusionScore
		}
	}
	return score
}

func candidateSources(byID map[string]*CandidateSource) []CandidateSource {
	result := make([]CandidateSource, 0, len(byID))
	for _, candidate := range byID {
		result = append(result, *candidate)
	}
	sortRetrieveCandidates(result)
	return result
}
