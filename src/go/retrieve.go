package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/lib/pq"
)

const (
	defaultRetrieveSources          = 12
	semanticRerankCandidates        = 40
	literalRerankCandidates         = 25
	defaultRerankCandidates         = 40
	defaultRetrieveSearchCandidates = 150
	retrieveChunkChars              = 1200
	retrieveEvidenceChars           = 1800
	rerankWindowChars               = 1400
	retrieveTraceTTL                = 20 * time.Minute
	retrieveTraceLimit              = 64
	nearbyChunkWindow               = 1
	maxSourcesPerMemory             = 3
	titleSeedChunks                 = 3
	literalRescueLimit              = 2
	literalRescueThreshold          = 0.65
	reciprocalRankConstant          = 60.0
	maxChunksPerChapter             = 12
	retrievalOrderWeight            = 0.25
)

type RetrieveRequest struct {
	RequestID           string    `json:"request_id,omitempty"`
	Query               string    `json:"query"`
	Mode                string    `json:"mode,omitempty"`
	TimeFilter          string    `json:"time_filter,omitempty"`
	Scope               string    `json:"scope,omitempty"`
	Collection          string    `json:"collection,omitempty"`
	Limit               int       `json:"limit,omitempty"`
	CandidateLimit      int       `json:"candidate_limit,omitempty"`
	ChunkLimitPerSource int       `json:"chunk_limit_per_source,omitempty"`
	CandidateOnly       bool      `json:"candidate_only,omitempty"`
	Embedding           []float64 `json:"-"`
}

type RetrieveResponse struct {
	Query       string              `json:"query"`
	Specs       []RetrievalSpec     `json:"specs"`
	Candidates  []CandidateSource   `json:"candidates,omitempty"`
	Sources     []CandidateSource   `json:"sources"`
	Diagnostics RetrieveDiagnostics `json:"diagnostics"`
}

type RetrieveTrace struct {
	RequestID string           `json:"request_id"`
	CreatedAt time.Time        `json:"created_at"`
	Response  RetrieveResponse `json:"response"`
}

type RetrievalSpec struct {
	Query string `json:"query"`
	Mode  string `json:"mode"`
}

type CandidateSource struct {
	ID             string         `json:"id"`
	MemoryID       string         `json:"memory_id"`
	ChunkID        string         `json:"chunk_id"`
	Title          string         `json:"title"`
	Text           string         `json:"text"`
	EvidenceText   string         `json:"evidence_text,omitempty"`
	ContextText    string         `json:"context_text,omitempty"`
	MetadataText   string         `json:"-"`
	ParentSummary  string         `json:"-"`
	RerankContext  string         `json:"-"`
	Kind           string         `json:"kind"`
	SourceType     string         `json:"source_type"`
	SourceURI      string         `json:"source_uri"`
	Tags           []string       `json:"tags"`
	Metadata       map[string]any `json:"metadata"`
	ChunkMetadata  map[string]any `json:"chunk_metadata"`
	UpdatedAt      string         `json:"updated_at"`
	MaxChars       int            `json:"max_chars"`
	Expanded       bool           `json:"expanded,omitempty"`
	RetrievalScore float64        `json:"retrieval_score"`
	RerankScore    *float64       `json:"rerank_score,omitempty"`
	Retrievals     []RetrievalHit `json:"retrievals"`
}

type RetrievalHit struct {
	Query            string  `json:"query"`
	Mode             string  `json:"mode"`
	Route            string  `json:"route"`
	Rank             int     `json:"rank"`
	Score            float64 `json:"score"`
	Similarity       float64 `json:"similarity"`
	LiteralScore     float64 `json:"literal_score"`
	VectorSimilarity float64 `json:"vector_similarity"`
	FusionScore      float64 `json:"fusion_score"`
	ParentRank       int     `json:"parent_rank,omitempty"`
	ChildRank        int     `json:"child_rank,omitempty"`
	ParentSimilarity float64 `json:"parent_similarity,omitempty"`
}

type retrieveNearbyChunk struct {
	id    string
	label string
	text  string
}

type RetrieveDiagnostics struct {
	Status                  string   `json:"status"`
	Reason                  string   `json:"reason,omitempty"`
	RetrievedCandidateCount int      `json:"retrieved_candidate_count"`
	CandidateCount          int      `json:"candidate_count"`
	RerankCandidateCount    int      `json:"rerank_candidate_count"`
	SelectedCount           int      `json:"selected_count"`
	QueryCount              int      `json:"query_count"`
	SearchMs                int64    `json:"search_ms,omitempty"`
	RerankMs                int64    `json:"rerank_ms,omitempty"`
	ExpandMs                int64    `json:"expand_ms,omitempty"`
	TotalMs                 int64    `json:"total_ms,omitempty"`
	ReturnedIDCount         int      `json:"returned_id_count,omitempty"`
	ChangedOrder            bool     `json:"changed_order,omitempty"`
	Reranker                string   `json:"reranker,omitempty"`
	RerankerModel           string   `json:"reranker_model,omitempty"`
	RewriteStatus           string   `json:"rewrite_status,omitempty"`
	RewriteQuery            string   `json:"rewrite_query,omitempty"`
	RewriteModel            string   `json:"rewrite_model,omitempty"`
	RewriteMs               int64    `json:"rewrite_ms,omitempty"`
	RewriteSearchMs         int64    `json:"rewrite_search_ms,omitempty"`
	Error                   string   `json:"error,omitempty"`
	UnknownIDs              []string `json:"unknown_ids,omitempty"`
}

type rerankerRequest struct {
	RequestID  string              `json:"request_id,omitempty"`
	Query      string              `json:"query"`
	TopK       int                 `json:"top_k"`
	Candidates []rerankerCandidate `json:"candidates"`
}

type rerankerCandidate struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
	Text  string `json:"text"`
}

type rerankerResponse struct {
	Model   string           `json:"model"`
	Detail  string           `json:"detail"`
	Results []rerankerResult `json:"results"`
}

type rerankerResult struct {
	ID    string   `json:"id"`
	Score *float64 `json:"score"`
}

func (s *Server) handleRetrieve(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req RetrieveRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	response, err := s.retrieve(r.Context(), userKeyFromRequest(r), req)
	if err != nil {
		log.Printf("memory.retrieve request_id=%q status=error user=%q query=%q query_chars=%d ms=%d err=%q", req.RequestID, userKeyFromRequest(r), req.Query, len([]rune(req.Query)), time.Since(start).Milliseconds(), err.Error())
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.storeRetrieveTrace(req.RequestID, response)
	log.Printf(
		"memory.retrieve request_id=%q status=ok user=%q query=%q query_chars=%d request_limit=%d request_candidate_limit=%d specs=%d retrieved_candidates=%d candidates=%d selected=%d search_ms=%d rewrite_status=%q rewrite_ms=%d rewrite_search_ms=%d rerank_ms=%d total_ms=%d reranker=%q model=%q",
		req.RequestID,
		userKeyFromRequest(r),
		req.Query,
		len([]rune(req.Query)),
		req.Limit,
		req.CandidateLimit,
		len(response.Specs),
		response.Diagnostics.RetrievedCandidateCount,
		response.Diagnostics.CandidateCount,
		response.Diagnostics.SelectedCount,
		response.Diagnostics.SearchMs,
		response.Diagnostics.RewriteStatus,
		response.Diagnostics.RewriteMs,
		response.Diagnostics.RewriteSearchMs,
		response.Diagnostics.RerankMs,
		response.Diagnostics.TotalMs,
		response.Diagnostics.Reranker,
		response.Diagnostics.RerankerModel,
	)
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleRetrieveTrace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	requestID := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/retrieve/traces/"), "/")
	if requestID == "" {
		writeError(w, http.StatusNotFound, "request id required")
		return
	}
	trace, ok := s.retrieveTrace(requestID)
	if !ok {
		writeError(w, http.StatusNotFound, "retrieve trace not found")
		return
	}
	writeJSON(w, http.StatusOK, trace)
}

func (s *Server) retrieve(ctx context.Context, userKey string, req RetrieveRequest) (RetrieveResponse, error) {
	totalStart := time.Now()
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return RetrieveResponse{}, fmt.Errorf("query is required")
	}
	sourceLimit := retrieveLimit(req.Limit, defaultRetrieveSources)
	rerankCandidateLimit := retrieveLimit(req.CandidateLimit, defaultRerankCandidates)
	if req.CandidateOnly {
		rerankCandidateLimit = candidateEvaluationLimit(req.CandidateLimit)
	}
	if rerankCandidateLimit < sourceLimit {
		rerankCandidateLimit = sourceLimit
	}
	searchCandidateLimit := maxInt(defaultRetrieveSearchCandidates, semanticRerankCandidates)

	preferredMode := normalizeSearchMode(req.Mode)
	specs := retrievalSpecsFor(query, preferredMode)
	if preferredMode != searchModeLiteral {
		embedding, embeddingErr := s.embedOne(ctx, query, req.RequestID)
		if embeddingErr != nil {
			return RetrieveResponse{}, embeddingErr
		}
		req.Embedding = embedding
	}
	searchStart := time.Now()
	titles, titleErr := s.queryRewriteTitles(ctx, userKey, query, req.Scope)
	if titleErr != nil {
		log.Printf("memory.retrieve.source_titles request_id=%q status=error query=%q err=%q", req.RequestID, query, titleErr.Error())
		titles = nil
	}
	rewriteResults := make(chan queryRewriteResult, 1)
	rewriteCtx, cancelRewrite := context.WithCancel(ctx)
	defer cancelRewrite()
	go func() {
		rewriteResults <- s.rewriteRetrievalQueryForTitles(rewriteCtx, req.RequestID, query, titles)
	}()
	type retrievalLaneResult struct {
		candidates []CandidateSource
		err        error
	}
	generalLane := make(chan retrievalLaneResult, 1)
	sourceLane := make(chan retrievalLaneResult, 1)
	go func() {
		candidates, laneErr := s.retrieveCandidates(ctx, userKey, req, specs, searchCandidateLimit, literalRerankCandidates)
		generalLane <- retrievalLaneResult{candidates: candidates, err: laneErr}
	}()
	go func() {
		if len(titles) == 0 {
			sourceLane <- retrievalLaneResult{}
			return
		}
		candidates, laneErr := s.retrieveSourceCandidates(ctx, userKey, req, RetrievalSpec{Query: query, Mode: preferredMode}, titles)
		sourceLane <- retrievalLaneResult{candidates: candidates, err: laneErr}
	}()
	general := <-generalLane
	source := <-sourceLane
	if general.err != nil {
		cancelRewrite()
		return RetrieveResponse{}, general.err
	}
	if source.err != nil {
		log.Printf("memory.retrieve.source_search request_id=%q status=error query=%q titles=%q err=%q", req.RequestID, query, titles, source.err.Error())
	}
	retrievedCandidates := mergeRetrieveCandidates(general.candidates, source.candidates)
	rewrite := <-rewriteResults
	rewriteSearchMs := int64(0)
	searchMs := time.Since(searchStart).Milliseconds()
	candidates := selectRerankCandidates(retrievedCandidates, rerankCandidateLimit)
	var sources []CandidateSource
	var diagnostics RetrieveDiagnostics
	if req.CandidateOnly {
		sources = append([]CandidateSource(nil), candidates[:minInt(sourceLimit, len(candidates))]...)
		diagnostics = RetrieveDiagnostics{
			Status:               "candidate_only",
			Reason:               "reranker_skipped",
			CandidateCount:       len(candidates),
			RerankCandidateCount: len(candidates),
			SelectedCount:        len(sources),
		}
	} else {
		windowStart := time.Now()
		windowedCandidates, windowErr := s.addRerankWindows(ctx, userKey, candidates, nearbyChunkWindow)
		if windowErr != nil {
			log.Printf("memory.retrieve.rerank_windows request_id=%q status=fallback candidates=%d err=%q", req.RequestID, len(candidates), windowErr.Error())
		} else {
			candidates = windowedCandidates
			log.Printf("memory.retrieve.rerank_windows request_id=%q status=ok candidates=%d ms=%d", req.RequestID, len(candidates), time.Since(windowStart).Milliseconds())
		}
		rerankQuery := query
		if rewrite.Query != "" {
			rerankQuery = query + "\nRetrieval interpretation: " + rewrite.Query
		}
		if relationshipQuery := relationshipVocabularyExpansion(query); relationshipQuery != "" {
			rerankQuery += "\nRelationship vocabulary: " + relationshipQuery
		}
		sources, diagnostics = s.selectRetrieveSources(ctx, query, rerankQuery, candidates, sourceLimit, rerankCandidateLimit, req.RequestID)
	}
	diagnostics.RetrievedCandidateCount = len(retrievedCandidates)
	expandStart := time.Now()
	expandedSources, expandErr := s.expandRetrieveSources(ctx, userKey, sources, nearbyChunkWindow)
	if expandErr != nil {
		log.Printf("memory.retrieve.expand request_id=%q status=error query=%q selected=%d err=%q", req.RequestID, query, len(sources), expandErr.Error())
	} else {
		sources = expandedSources
		diagnostics.ExpandMs = time.Since(expandStart).Milliseconds()
	}
	diagnostics.QueryCount = len(specs)
	diagnostics.SearchMs = searchMs
	diagnostics.RewriteStatus = rewrite.Status
	diagnostics.RewriteQuery = rewrite.Query
	diagnostics.RewriteModel = rewrite.Model
	diagnostics.RewriteMs = rewrite.Ms
	diagnostics.RewriteSearchMs = rewriteSearchMs
	diagnostics.TotalMs = time.Since(totalStart).Milliseconds()
	return RetrieveResponse{
		Query:       query,
		Specs:       specs,
		Candidates:  candidates,
		Sources:     sources,
		Diagnostics: diagnostics,
	}, nil
}

func (s *Server) addRerankWindows(ctx context.Context, userKey string, candidates []CandidateSource, radius int) ([]CandidateSource, error) {
	if radius <= 0 || len(candidates) == 0 {
		return candidates, nil
	}
	targetIDs := make([]string, 0, len(candidates))
	memoryIDs := make([]string, 0, len(candidates))
	chunkIndexes := make([]int64, 0, len(candidates))
	for _, candidate := range candidates {
		chunkIndex, ok := intMapValue(candidate.ChunkMetadata, "chunk_index")
		if !ok {
			continue
		}
		targetIDs = append(targetIDs, candidate.ID)
		memoryIDs = append(memoryIDs, candidate.MemoryID)
		chunkIndexes = append(chunkIndexes, int64(chunkIndex))
	}
	if len(targetIDs) == 0 {
		return candidates, nil
	}
	rows, err := s.db.QueryContext(ctx, `
		WITH targets(target_id, memory_id, chunk_index) AS (
			SELECT * FROM unnest($2::text[], $3::text[], $4::bigint[])
		)
		SELECT t.target_id, c.id, c.chunk_index, c.text
		FROM targets t
		JOIN memory_chunks c
			ON c.user_key = $1
			AND c.memory_id = t.memory_id
			AND c.chunk_index BETWEEN t.chunk_index - $5 AND t.chunk_index + $5
		ORDER BY t.target_id, c.chunk_index
	`, userKey, pq.Array(targetIDs), pq.Array(memoryIDs), pq.Array(chunkIndexes), radius)
	if err != nil {
		return candidates, err
	}
	defer rows.Close()
	windows := make(map[string][]retrieveNearbyChunk, len(targetIDs))
	selected := make(map[string]int, len(targetIDs))
	for rows.Next() {
		var targetID, chunkID, text string
		var chunkIndex int
		if err := rows.Scan(&targetID, &chunkID, &chunkIndex, &text); err != nil {
			return candidates, err
		}
		windows[targetID] = append(windows[targetID], retrieveNearbyChunk{
			id:    chunkID,
			label: fmt.Sprintf("Chunk %d", chunkIndex),
			text:  strings.TrimSpace(text),
		})
	}
	if err := rows.Err(); err != nil {
		return candidates, err
	}
	result := append([]CandidateSource(nil), candidates...)
	for index := range result {
		chunks := windows[result[index].ID]
		for chunkIndex := range chunks {
			if chunks[chunkIndex].id == result[index].ChunkID {
				selected[result[index].ID] = chunkIndex
				break
			}
		}
		if len(chunks) > 0 {
			result[index].RerankContext = centeredEvidenceText(chunks, selected[result[index].ID], rerankWindowChars)
		}
	}
	return result, nil
}

func (s *Server) retrieveSourceCandidates(ctx context.Context, userKey string, req RetrieveRequest, spec RetrievalSpec, titles []string) ([]CandidateSource, error) {
	start := time.Now()
	sourceQuery := strings.TrimSpace(spec.Query + " " + strings.Join(titles, " "))
	sourceEmbedding, err := s.embedOne(ctx, sourceQuery, req.RequestID)
	if err != nil {
		return nil, err
	}
	type sourceLaneResult struct {
		results []SearchResult
		err     error
	}
	directLane := make(chan sourceLaneResult, 1)
	parentLane := make(chan sourceLaneResult, 1)
	literalLane := make(chan sourceLaneResult, 1)
	go func() {
		results, searchErr := s.search(ctx, userKey, SearchRequest{
			Query:               sourceQuery,
			RequestID:           req.RequestID,
			Mode:                searchModeSemantic,
			Scope:               req.Scope,
			Collection:          req.Collection,
			Limit:               defaultRetrieveSearchCandidates,
			ChunkLimitPerSource: defaultRetrieveSearchCandidates,
			SourceTitles:        titles,
			Embedding:           sourceEmbedding,
		})
		directLane <- sourceLaneResult{results: results, err: searchErr}
	}()
	go func() {
		results, searchErr := s.searchParentChildren(ctx, userKey, spec.Query, SearchRequest{
			Query:        spec.Query,
			RequestID:    req.RequestID,
			Mode:         searchModeSemantic,
			TimeFilter:   req.TimeFilter,
			Scope:        req.Scope,
			Collection:   req.Collection,
			SourceTitles: titles,
			Embedding:    req.Embedding,
		}, sourceHierarchyParents, sourceHierarchyChildren, false)
		parentLane <- sourceLaneResult{results: results, err: searchErr}
	}()
	go func() {
		if s.bm25 == nil {
			literalLane <- sourceLaneResult{}
			return
		}
		literalQuery := spec.Query
		if relationshipQuery := relationshipVocabularyExpansion(spec.Query); relationshipQuery != "" {
			literalQuery += " " + relationshipQuery
		}
		results := s.bm25.Search(userKey, SearchRequest{
			Query:               literalQuery,
			TimeFilter:          req.TimeFilter,
			Scope:               req.Scope,
			Collection:          req.Collection,
			SourceTitles:        titles,
			ChunkLimitPerSource: defaultRerankCandidates,
		}, defaultRetrieveSearchCandidates)
		literalLane <- sourceLaneResult{results: results}
	}()
	direct := <-directLane
	parent := <-parentLane
	literal := <-literalLane
	if direct.err != nil {
		return nil, direct.err
	}
	if parent.err != nil {
		return nil, parent.err
	}
	byID := map[string]*CandidateSource{}
	addRetrieveCandidates(byID, direct.results, spec, "source_direct")
	addRetrieveCandidates(byID, parent.results, spec, "source_parent")
	addRetrieveCandidates(byID, literal.results, RetrievalSpec{Query: spec.Query, Mode: searchModeLiteral}, "source_literal")
	candidates := make([]CandidateSource, 0, len(byID))
	for _, candidate := range byID {
		candidates = append(candidates, *candidate)
	}
	sortRetrieveCandidates(candidates)
	log.Printf("memory.retrieve.source_search request_id=%q status=ok query=%q titles=%q results=%d ms=%d", req.RequestID, spec.Query, titles, len(candidates), time.Since(start).Milliseconds())
	return candidates, nil
}

func (s *Server) retrieveCandidates(ctx context.Context, userKey string, req RetrieveRequest, specs []RetrievalSpec, limit, literalLimit int) ([]CandidateSource, error) {
	type specResult struct {
		spec    RetrievalSpec
		results []SearchResult
		err     error
	}
	chunkLimitPerSource := req.ChunkLimitPerSource
	if chunkLimitPerSource <= 0 {
		chunkLimitPerSource = limit
	}
	out := make(chan specResult, len(specs))
	parentOut := make(chan specResult, 1)
	var wg sync.WaitGroup
	go func() {
		preferredMode := normalizeSearchMode(req.Mode)
		if preferredMode == searchModeLiteral || len(req.Embedding) == 0 {
			parentOut <- specResult{}
			return
		}
		start := time.Now()
		results, parentErr := s.searchParentChildren(ctx, userKey, strings.TrimSpace(req.Query), SearchRequest{
			Query:      strings.TrimSpace(req.Query),
			RequestID:  req.RequestID,
			Mode:       searchModeSemantic,
			TimeFilter: req.TimeFilter,
			Scope:      req.Scope,
			Collection: req.Collection,
			Embedding:  req.Embedding,
		}, hierarchyParentLimit, hierarchyChildrenPerParent, false)
		log.Printf("memory.retrieve.parent_search request_id=%q query=%q parents=%d children=%d ms=%d", req.RequestID, strings.TrimSpace(req.Query), hierarchyParentLimit, len(results), time.Since(start).Milliseconds())
		parentOut <- specResult{spec: RetrievalSpec{Query: strings.TrimSpace(req.Query), Mode: preferredMode}, results: results, err: parentErr}
	}()
	for _, spec := range specs {
		spec := spec
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			specLimit := limit
			if spec.Mode == searchModeLiteral && literalLimit > 0 {
				specLimit = literalLimit
			}
			results, err := s.search(ctx, userKey, SearchRequest{
				Query:               spec.Query,
				RequestID:           req.RequestID,
				Mode:                spec.Mode,
				TimeFilter:          req.TimeFilter,
				Scope:               req.Scope,
				Collection:          req.Collection,
				Limit:               specLimit,
				ChunkLimitPerSource: chunkLimitPerSource,
			})
			log.Printf(
				"memory.retrieve.search_spec request_id=%q mode=%q query=%q query_chars=%d results=%d ms=%d err=%q",
				req.RequestID,
				spec.Mode,
				spec.Query,
				len([]rune(spec.Query)),
				len(results),
				time.Since(start).Milliseconds(),
				errorString(err),
			)
			out <- specResult{spec: spec, results: results, err: err}
		}()
	}
	wg.Wait()
	close(out)

	byID := map[string]*CandidateSource{}
	for result := range out {
		if result.err != nil {
			return nil, result.err
		}
		addRetrieveCandidates(byID, result.results, result.spec, "direct")
	}
	parent := <-parentOut
	if parent.err != nil {
		return nil, parent.err
	}
	addRetrieveCandidates(byID, parent.results, parent.spec, "parent")
	candidates := make([]CandidateSource, 0, len(byID))
	for _, candidate := range byID {
		candidates = append(candidates, *candidate)
	}
	sortRetrieveCandidates(candidates)
	return candidates, nil
}

func (s *Server) selectRetrieveSources(ctx context.Context, query, rerankQuery string, candidates []CandidateSource, sourceLimit, candidateLimit int, requestID string) ([]CandidateSource, RetrieveDiagnostics) {
	topCandidates := append([]CandidateSource(nil), candidates[:minInt(candidateLimit, len(candidates))]...)
	base := RetrieveDiagnostics{
		Status:               "fallback",
		CandidateCount:       len(candidates),
		RerankCandidateCount: len(topCandidates),
		SelectedCount:        minInt(sourceLimit, len(topCandidates)),
	}
	if len(topCandidates) <= sourceLimit {
		base.Status = "skipped"
		base.Reason = "not_enough_candidates"
		return topCandidates, base
	}
	rerankStart := time.Now()
	reranked, result, err := s.rerankRetrieveSources(ctx, rerankQuery, topCandidates, requestID)
	base.RerankMs = time.Since(rerankStart).Milliseconds()
	if err != nil {
		base.Reranker = "retrieval_score"
		base.Error = err.Error()
		return topCandidates[:sourceLimit], base
	}
	base.Status = "reranked"
	reranked = fuseRetrieveAndRerankOrders(topCandidates, reranked)
	reranked = filterQueryEchoSources(query, reranked)
	reranked = diversifyRetrieveSources(reranked, sourceLimit, maxSourcesPerMemory)
	reranked = rescueLiteralRetrieveSources(query, reranked, topCandidates, sourceLimit, literalRescueLimit, literalRescueThreshold)
	base.SelectedCount = len(reranked)
	base.ReturnedIDCount = len(result.Results)
	base.ChangedOrder = retrieveOrderChanged(topCandidates, reranked)
	base.Reranker = "cross_encoder"
	base.RerankerModel = result.Model
	return reranked, base
}

func fuseRetrieveAndRerankOrders(retrieved, reranked []CandidateSource) []CandidateSource {
	type fusedCandidate struct {
		source        CandidateSource
		score         float64
		retrievalRank int
		rerankRank    int
	}
	byID := make(map[string]*fusedCandidate, len(retrieved))
	for rank, source := range retrieved {
		byID[source.ID] = &fusedCandidate{source: source, score: retrievalOrderWeight * reciprocalRankScore(rank+1), retrievalRank: rank + 1}
	}
	for rank, source := range reranked {
		entry := byID[source.ID]
		if entry == nil {
			entry = &fusedCandidate{source: source}
			byID[source.ID] = entry
		} else {
			entry.source = source
		}
		entry.rerankRank = rank + 1
		entry.score += reciprocalRankScore(rank + 1)
	}
	fused := make([]fusedCandidate, 0, len(byID))
	for _, entry := range byID {
		entry.source.ChunkMetadata = cloneAnyMap(entry.source.ChunkMetadata)
		entry.source.ChunkMetadata["retrieval_rank"] = entry.retrievalRank
		entry.source.ChunkMetadata["final_rerank_rank"] = entry.rerankRank
		entry.source.ChunkMetadata["final_fusion_score"] = entry.score
		fused = append(fused, *entry)
	}
	sort.SliceStable(fused, func(i, j int) bool {
		if fused[i].score == fused[j].score {
			return fused[i].source.RetrievalScore > fused[j].source.RetrievalScore
		}
		return fused[i].score > fused[j].score
	})
	result := make([]CandidateSource, len(fused))
	for index := range fused {
		result[index] = fused[index].source
	}
	return result
}

func filterQueryEchoSources(query string, sources []CandidateSource) []CandidateSource {
	normalizedQuery := normalizeQueryEchoText(query)
	if normalizedQuery == "" {
		return sources
	}
	filtered := make([]CandidateSource, 0, len(sources))
	for _, source := range sources {
		if normalizeQueryEchoText(source.Text) == normalizedQuery {
			continue
		}
		filtered = append(filtered, source)
	}
	return filtered
}

func normalizeQueryEchoText(value string) string {
	terms := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	return strings.Join(terms, " ")
}

func (s *Server) rerankRetrieveSources(ctx context.Context, query string, candidates []CandidateSource, requestID string) ([]CandidateSource, rerankerResponse, error) {
	rerankerCandidates := make([]rerankerCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		rerankerCandidates = append(rerankerCandidates, rerankerCandidate{
			ID:    candidate.ID,
			Title: candidate.Title,
			Text:  candidate.rerankText(),
		})
	}
	result, err := s.callReranker(ctx, query, requestID, rerankerCandidates)
	if err != nil {
		return nil, result, err
	}

	byID := map[string]CandidateSource{}
	for _, candidate := range candidates {
		byID[candidate.ID] = candidate
	}
	selected := make([]CandidateSource, 0, len(candidates))
	for rank, item := range result.Results {
		id := strings.TrimSpace(item.ID)
		candidate, ok := byID[id]
		if !ok || containsCandidate(selected, id) {
			continue
		}
		candidate.RerankScore = item.Score
		candidate.ChunkMetadata = cloneAnyMap(candidate.ChunkMetadata)
		candidate.ChunkMetadata["final_rerank_rank"] = rank + 1
		selected = append(selected, candidate)
	}
	for _, candidate := range candidates {
		if containsCandidate(selected, candidate.ID) {
			continue
		}
		selected = append(selected, candidate)
	}
	return selected, result, nil
}

func (s *Server) callReranker(ctx context.Context, query, requestID string, candidates []rerankerCandidate) (rerankerResponse, error) {
	baseURL := strings.TrimRight(strings.TrimSpace(s.rerankerURL), "/")
	if baseURL == "" {
		return rerankerResponse{}, fmt.Errorf("reranker URL is not configured")
	}
	payload := rerankerRequest{
		RequestID:  requestID,
		Query:      query,
		TopK:       len(candidates),
		Candidates: candidates,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return rerankerResponse{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return rerankerResponse{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	if requestID != "" {
		request.Header.Set("X-Request-ID", requestID)
	}
	response, err := s.httpClient.Do(request)
	if err != nil {
		return rerankerResponse{}, err
	}
	defer response.Body.Close()
	var result rerankerResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return rerankerResponse{}, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		if strings.TrimSpace(result.Detail) != "" {
			return result, fmt.Errorf(result.Detail)
		}
		return result, fmt.Errorf("reranker status %s", response.Status)
	}
	return result, nil
}

func (s *Server) expandRetrieveSources(ctx context.Context, userKey string, sources []CandidateSource, window int) ([]CandidateSource, error) {
	if window <= 0 || len(sources) == 0 {
		for index := range sources {
			sources[index].EvidenceText = truncateRunes(strings.TrimSpace(sources[index].Text), retrieveEvidenceChars)
		}
		return sources, nil
	}
	expanded := make([]CandidateSource, 0, len(sources))
	for _, source := range sources {
		if source.ChunkMetadata["hierarchy_window_expanded"] == true {
			source.Expanded = true
			source.EvidenceText = truncateRunes(strings.TrimSpace(source.Text), retrieveEvidenceChars)
			expanded = append(expanded, source)
			continue
		}
		chunkIndex, ok := intMapValue(source.ChunkMetadata, "chunk_index")
		if !ok {
			source.EvidenceText = truncateRunes(strings.TrimSpace(source.Text), retrieveEvidenceChars)
			expanded = append(expanded, source)
			continue
		}
		rows, err := s.db.QueryContext(ctx, `
			SELECT id, text, metadata::text
			FROM memory_chunks
			WHERE user_key = $1
				AND memory_id = $2
				AND chunk_index BETWEEN $3 AND $4
			ORDER BY chunk_index ASC
		`, userKey, source.MemoryID, chunkIndex-window, chunkIndex+window)
		if err != nil {
			return sources, err
		}
		chunks := []retrieveNearbyChunk{}
		chunkIDs := []string{}
		selectedIndex := -1
		for rows.Next() {
			var id, text, metadataRaw string
			if err := rows.Scan(&id, &text, &metadataRaw); err != nil {
				rows.Close()
				return sources, err
			}
			label := "Nearby chunk"
			metadata := parseMap(metadataRaw)
			if index, ok := intMapValue(metadata, "chunk_index"); ok {
				label = fmt.Sprintf("Chunk %d", index)
			}
			chunks = append(chunks, retrieveNearbyChunk{id: id, label: label, text: strings.TrimSpace(text)})
			chunkIDs = append(chunkIDs, id)
			if id == source.ChunkID {
				selectedIndex = len(chunks) - 1
			}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return sources, err
		}
		rows.Close()
		if len(chunks) <= 1 {
			source.EvidenceText = strings.TrimSpace(source.Text)
			expanded = append(expanded, source)
			continue
		}
		formatted := make([]string, 0, len(chunks))
		for _, chunk := range chunks {
			formatted = append(formatted, fmt.Sprintf("%s\n%s", chunk.label, chunk.text))
		}
		source.Text = strings.Join(formatted, "\n\n---\n\n")
		source.EvidenceText = centeredEvidenceText(chunks, selectedIndex, retrieveEvidenceChars)
		source.Expanded = true
		if source.ChunkMetadata == nil {
			source.ChunkMetadata = map[string]any{}
		}
		source.ChunkMetadata["expanded_chunk_ids"] = chunkIDs
		source.ChunkMetadata["expanded_window"] = window
		expanded = append(expanded, source)
	}
	return expanded, nil
}

func centeredEvidenceText(chunks []retrieveNearbyChunk, selectedIndex, limit int) string {
	if len(chunks) == 0 || limit <= 0 {
		return ""
	}
	if selectedIndex < 0 || selectedIndex >= len(chunks) {
		selectedIndex = len(chunks) / 2
	}
	selected := chunks[selectedIndex]
	selectedText := strings.TrimSpace(selected.label + "\n" + selected.text)
	selectedText = truncateRunes(selectedText, limit)
	remaining := limit - len([]rune(selectedText))
	if remaining <= 0 {
		return selectedText
	}

	before := ""
	after := ""
	if selectedIndex > 0 {
		chunk := chunks[selectedIndex-1]
		before = strings.TrimSpace(chunk.label + "\n" + chunk.text)
	}
	if selectedIndex+1 < len(chunks) {
		chunk := chunks[selectedIndex+1]
		after = strings.TrimSpace(chunk.label + "\n" + chunk.text)
	}
	separatorRunes := 0
	if before != "" {
		separatorRunes += len([]rune("\n\n"))
	}
	if after != "" {
		separatorRunes += len([]rune("\n\n"))
	}
	remaining = maxInt(0, remaining-separatorRunes)
	beforeBudget := remaining / 2
	afterBudget := remaining - beforeBudget
	if after == "" {
		beforeBudget = remaining
	}
	if before == "" {
		afterBudget = remaining
	}
	if len([]rune(before)) > beforeBudget {
		runes := []rune(before)
		before = "..." + strings.TrimSpace(string(runes[len(runes)-beforeBudget:]))
	}
	if len([]rune(after)) > afterBudget {
		after = strings.TrimSpace(truncateRunes(after, afterBudget)) + "..."
	}
	parts := []string{}
	if before != "" {
		parts = append(parts, before)
	}
	parts = append(parts, selectedText)
	if after != "" {
		parts = append(parts, after)
	}
	return strings.Join(parts, "\n\n")
}

func diversifyRetrieveSources(sources []CandidateSource, limit, perMemoryLimit int) []CandidateSource {
	if len(sources) <= limit || perMemoryLimit <= 0 {
		if len(sources) > limit {
			return sources[:limit]
		}
		return sources
	}
	selected := make([]CandidateSource, 0, limit)
	deferred := make([]CandidateSource, 0, len(sources))
	countByMemory := map[string]int{}
	for _, source := range sources {
		if countByMemory[source.MemoryID] >= perMemoryLimit {
			deferred = append(deferred, source)
			continue
		}
		selected = append(selected, source)
		countByMemory[source.MemoryID]++
		if len(selected) >= limit {
			return selected
		}
	}
	for _, source := range deferred {
		selected = append(selected, source)
		if len(selected) >= limit {
			break
		}
	}
	return selected
}

func rescueLiteralRetrieveSources(query string, selected, candidates []CandidateSource, limit, rescueLimit int, threshold float64) []CandidateSource {
	if limit <= 0 || rescueLimit <= 0 || len(candidates) == 0 {
		return selected
	}
	rescueQuery := query
	if relationshipQuery := relationshipVocabularyExpansion(query); relationshipQuery != "" {
		rescueQuery += " " + relationshipQuery
	}
	terms := literalRecallTerms(rescueQuery)
	type rescueCandidate struct {
		source CandidateSource
		score  float64
	}
	rescueCandidates := make([]rescueCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if containsCandidate(selected, candidate.ID) || maxLiteralScore(candidate) < threshold {
			continue
		}
		score := literalBodyEvidenceScore(query, terms, candidate)
		if score <= 0 {
			continue
		}
		rescueCandidates = append(rescueCandidates, rescueCandidate{source: candidate, score: score})
	}
	sort.SliceStable(rescueCandidates, func(i, j int) bool {
		if rescueCandidates[i].score == rescueCandidates[j].score {
			return maxLiteralScore(rescueCandidates[i].source) > maxLiteralScore(rescueCandidates[j].source)
		}
		return rescueCandidates[i].score > rescueCandidates[j].score
	})
	if len(rescueCandidates) > rescueLimit {
		rescueCandidates = rescueCandidates[:rescueLimit]
	}
	if len(rescueCandidates) == 0 {
		return selected
	}
	result := append([]CandidateSource(nil), selected...)
	replaceIndex := len(result) - 1
	for _, rescue := range rescueCandidates {
		source := rescue.source
		if containsCandidate(result, source.ID) {
			continue
		}
		if len(result) < limit {
			result = append(result, source)
			continue
		}
		if replaceIndex < 0 {
			break
		}
		result[replaceIndex] = source
		replaceIndex--
	}
	return result
}

func literalBodyEvidenceScore(query string, terms []string, source CandidateSource) float64 {
	bodyTerms := tokenizeTermSet(source.Text)
	score := 0.0
	for _, term := range terms {
		if bodyTerms[term] {
			score += 1
		}
	}
	return score
}

func tokenizeTermSet(text string) map[string]bool {
	terms := map[string]bool{}
	for _, term := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') &&
			!(r >= 'A' && r <= 'Z') &&
			!(r >= '0' && r <= '9') &&
			!(r >= 0x80)
	}) {
		term = strings.TrimSpace(strings.TrimSuffix(term, "'s"))
		if term != "" {
			terms[term] = true
		}
	}
	return terms
}

func maxLiteralScore(source CandidateSource) float64 {
	best := 0.0
	for _, hit := range source.Retrievals {
		if hit.LiteralScore > best {
			best = hit.LiteralScore
		}
	}
	return best
}

func (s *Server) storeRetrieveTrace(requestID string, response RetrieveResponse) {
	requestID = strings.TrimSpace(requestID)
	if requestID == "" {
		return
	}
	s.retrieveTraceMu.Lock()
	defer s.retrieveTraceMu.Unlock()
	now := time.Now().UTC()
	s.retrieveTraces[requestID] = RetrieveTrace{
		RequestID: requestID,
		CreatedAt: now,
		Response:  response,
	}
	for id, trace := range s.retrieveTraces {
		if now.Sub(trace.CreatedAt) > retrieveTraceTTL {
			delete(s.retrieveTraces, id)
		}
	}
	if len(s.retrieveTraces) <= retrieveTraceLimit {
		return
	}
	type traceAge struct {
		id        string
		createdAt time.Time
	}
	ages := make([]traceAge, 0, len(s.retrieveTraces))
	for id, trace := range s.retrieveTraces {
		ages = append(ages, traceAge{id: id, createdAt: trace.CreatedAt})
	}
	sort.Slice(ages, func(i, j int) bool {
		return ages[i].createdAt.Before(ages[j].createdAt)
	})
	for len(s.retrieveTraces) > retrieveTraceLimit && len(ages) > 0 {
		delete(s.retrieveTraces, ages[0].id)
		ages = ages[1:]
	}
}

func (s *Server) retrieveTrace(requestID string) (RetrieveTrace, bool) {
	s.retrieveTraceMu.Lock()
	defer s.retrieveTraceMu.Unlock()
	trace, ok := s.retrieveTraces[strings.TrimSpace(requestID)]
	if !ok || time.Since(trace.CreatedAt) > retrieveTraceTTL {
		if ok {
			delete(s.retrieveTraces, requestID)
		}
		return RetrieveTrace{}, false
	}
	return trace, true
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func intMapValue(values map[string]any, key string) (int, bool) {
	value, ok := values[key]
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		return int(typed), true
	case json.Number:
		parsed, err := typed.Int64()
		return int(parsed), err == nil
	default:
		var parsed int
		if _, err := fmt.Sscanf(fmt.Sprint(typed), "%d", &parsed); err == nil {
			return parsed, true
		}
		return 0, false
	}
}

func retrievalSpecsFor(query, preferred string) []RetrievalSpec {
	specs := []RetrievalSpec{}
	query = strings.TrimSpace(query)
	normalized := strings.TrimSpace(normalizeRetrievalQuery(query))
	preferredQuery := query
	if preferred == searchModeLiteral && normalized != "" {
		preferredQuery = normalized
	}
	if preferredQuery != "" {
		specs = append(specs, RetrievalSpec{Query: preferredQuery, Mode: preferred})
	}
	if preferred != searchModeLiteral {
		for _, expanded := range deterministicQueryExpansions(query) {
			specs = append(specs, RetrievalSpec{Query: expanded, Mode: preferred})
		}
	}
	if preferred != searchModeLiteral {
		literalQuery := normalized
		if literalQuery == "" {
			literalQuery = query
		}
		if literalQuery != "" {
			specs = append(specs, RetrievalSpec{Query: literalQuery, Mode: searchModeLiteral})
		}
		if relationshipQuery := relationshipVocabularyExpansion(query); relationshipQuery != "" {
			specs = append(specs, RetrievalSpec{Query: relationshipQuery, Mode: searchModeLiteral})
		}
	}
	return dedupeRetrievalSpecs(specs)
}

func deterministicQueryExpansions(query string) []string {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}
	variants := []string{}
	compact := strings.TrimSpace(normalizeRetrievalQuery(query))
	terms := retrievalQueryTerms(compact)
	if len(terms) >= 2 {
		variants = append(variants, strings.Join(terms, " "))
	}
	for index, term := range terms {
		if !relationshipTerm(term) || index == 0 {
			continue
		}
		person := terms[index-1]
		if retrievalQueryStopword(person) {
			continue
		}
		variants = append(variants, term+" of "+person+" name")
		if synonyms := relationshipSynonyms(term); len(synonyms) > 0 {
			variants = append(variants, strings.Join(synonyms, " ")+" of "+person+" name")
		}
		break
	}
	return dedupeStrings(variants, 3)
}

func relationshipVocabularyExpansion(query string) string {
	terms := retrievalQueryTerms(normalizeRetrievalQuery(query))
	for index, term := range terms {
		if index == 0 || !relationshipTerm(term) {
			continue
		}
		person := terms[index-1]
		synonyms := relationshipSynonyms(term)
		if retrievalQueryStopword(person) || len(synonyms) == 0 {
			continue
		}
		return strings.Join(synonyms, " ") + " " + person + " name"
	}
	return ""
}

func relationshipSynonyms(term string) []string {
	switch strings.ToLower(strings.TrimSpace(term)) {
	case "mother", "mum", "mom":
		return []string{"mother", "mum", "mom"}
	case "father", "dad":
		return []string{"father", "dad"}
	case "parent":
		return []string{"parent", "mother", "father", "mum", "mom", "dad"}
	case "daughter", "son", "child":
		return []string{"daughter", "son", "child"}
	case "wife", "husband", "partner", "spouse":
		return []string{"wife", "husband", "partner", "spouse"}
	default:
		return nil
	}
}

func retrievalQueryTerms(query string) []string {
	terms := []string{}
	for _, term := range strings.Fields(strings.ToLower(query)) {
		term = strings.TrimSpace(term)
		if term == "" || retrievalQueryStopword(term) {
			continue
		}
		terms = append(terms, term)
	}
	return terms
}

func retrievalQueryStopword(term string) bool {
	switch strings.ToLower(strings.TrimSpace(term)) {
	case "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "with", "would", "you", "your", "name":
		return true
	default:
		return false
	}
}

func relationshipTerm(term string) bool {
	switch strings.ToLower(strings.TrimSpace(term)) {
	case "mother", "mum", "mom", "father", "dad", "parent", "daughter", "son", "sister", "brother", "wife", "husband", "partner", "spouse", "child", "friend":
		return true
	default:
		return false
	}
}

func dedupeStrings(values []string, limit int) []string {
	seen := map[string]bool{}
	result := []string{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		key := strings.ToLower(value)
		if value == "" || seen[key] {
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

func addRetrieveCandidates(candidates map[string]*CandidateSource, results []SearchResult, spec RetrievalSpec, route string) {
	rankedResults := append([]SearchResult(nil), results...)
	sort.SliceStable(rankedResults, func(i, j int) bool {
		if route == "parent" || route == "source_parent" {
			return hierarchicalFusionScore(rankedResults[i]) > hierarchicalFusionScore(rankedResults[j])
		}
		left := maxFloat(rankedResults[i].Score, rankedResults[i].Similarity)
		right := maxFloat(rankedResults[j].Score, rankedResults[j].Similarity)
		return left > right
	})
	for index, result := range rankedResults {
		memoryID := strings.TrimSpace(result.MemoryID)
		chunkID := strings.TrimSpace(result.ChunkID)
		text := strings.TrimSpace(result.Text)
		if memoryID == "" || chunkID == "" || text == "" {
			continue
		}
		id := memoryID + ":" + chunkID
		fusionScore := reciprocalRankScore(index + 1)
		parentRank, _ := intMapValue(result.ChunkMetadata, "parent_rank")
		childRank, _ := intMapValue(result.ChunkMetadata, "child_rank")
		if route == "parent" || route == "source_parent" {
			if hierarchicalScore := hierarchicalFusionScore(result); hierarchicalScore > 0 {
				fusionScore = hierarchicalScore
			}
		}
		hit := RetrievalHit{
			Query:            spec.Query,
			Mode:             spec.Mode,
			Route:            route,
			Rank:             index + 1,
			Score:            maxFloat(result.Score, result.Similarity),
			Similarity:       result.Similarity,
			LiteralScore:     result.ScoreParts.LiteralScore,
			VectorSimilarity: result.ScoreParts.VectorSimilarity,
			FusionScore:      fusionScore,
			ParentRank:       parentRank,
			ChildRank:        childRank,
			ParentSimilarity: floatMapValue(result.ChunkMetadata, "parent_similarity"),
		}
		if existing := candidates[id]; existing != nil {
			existing.RetrievalScore += hit.FusionScore
			if existing.ParentSummary == "" {
				existing.ParentSummary = result.ParentSummary
			}
			existing.Retrievals = append(existing.Retrievals, hit)
			continue
		}
		title := strings.TrimSpace(result.Title)
		if title == "" {
			title = "Untitled memory"
		}
		candidates[id] = &CandidateSource{
			ID:             id,
			MemoryID:       memoryID,
			ChunkID:        chunkID,
			Title:          title,
			Text:           text,
			ContextText:    result.ContextText,
			MetadataText:   result.MetadataText,
			ParentSummary:  result.ParentSummary,
			Kind:           result.Kind,
			SourceType:     result.SourceType,
			SourceURI:      result.SourceURI,
			Tags:           result.Tags,
			Metadata:       result.Metadata,
			ChunkMetadata:  result.ChunkMetadata,
			UpdatedAt:      result.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
			MaxChars:       retrieveChunkChars,
			RetrievalScore: hit.FusionScore,
			Retrievals:     []RetrievalHit{hit},
		}
	}
}

func hierarchicalFusionScore(result SearchResult) float64 {
	if rerankRank, ok := intMapValue(result.ChunkMetadata, "hierarchy_rerank_rank"); ok && rerankRank > 0 {
		return reciprocalRankScore(rerankRank)
	}
	parentScore := floatMapValue(result.ChunkMetadata, "parent_fusion_score")
	childRank := bestHierarchyChildRank(result.ChunkMetadata)
	if parentScore <= 0 || childRank <= 0 {
		return 0
	}
	return 0.5*parentScore + 0.5*reciprocalRankScore(childRank)
}

func bestHierarchyChildRank(metadata map[string]any) int {
	vectorRank, _ := intMapValue(metadata, "child_rank")
	literalRank, _ := intMapValue(metadata, "parent_bm25_rank")
	if vectorRank <= 0 {
		return literalRank
	}
	if literalRank <= 0 || vectorRank < literalRank {
		return vectorRank
	}
	return literalRank
}

func floatMapValue(values map[string]any, key string) float64 {
	value, ok := values[key]
	if !ok {
		return 0
	}
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case json.Number:
		parsed, _ := typed.Float64()
		return parsed
	default:
		var parsed float64
		fmt.Sscanf(fmt.Sprint(typed), "%f", &parsed)
		return parsed
	}
}

func reciprocalRankScore(rank int) float64 {
	if rank <= 0 {
		return 0
	}
	return 1 / (reciprocalRankConstant + float64(rank))
}

func sortRetrieveCandidates(candidates []CandidateSource) {
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].RetrievalScore == candidates[j].RetrievalScore {
			return candidates[i].UpdatedAt > candidates[j].UpdatedAt
		}
		return candidates[i].RetrievalScore > candidates[j].RetrievalScore
	})
}

func selectRerankCandidates(candidates []CandidateSource, limit int) []CandidateSource {
	ranked := append([]CandidateSource(nil), candidates...)
	sortRetrieveCandidates(ranked)
	if limit <= 0 || len(ranked) <= limit {
		return ranked
	}
	hierarchyCandidates := make([]CandidateSource, 0, len(ranked))
	literalCandidates := make([]CandidateSource, 0, len(ranked))
	sourceLiteralCandidates := make([]CandidateSource, 0, len(ranked))
	for _, candidate := range ranked {
		if candidate.ChunkMetadata["hierarchy_parent_hit"] == true {
			hierarchyCandidates = append(hierarchyCandidates, candidate)
		}
		if bestLiteralScore(candidate) > 0 {
			literalCandidates = append(literalCandidates, candidate)
		}
		if candidateHasRoute(candidate, "source_literal") {
			sourceLiteralCandidates = append(sourceLiteralCandidates, candidate)
		}
	}
	hierarchyCandidates = roundRobinHierarchyCandidates(hierarchyCandidates)
	sortRetrieveCandidates(literalCandidates)
	sourceLiteralCandidates = roundRobinLiteralCandidatesBySource(sourceLiteralCandidates)

	result := make([]CandidateSource, 0, limit)
	seen := make(map[string]bool, limit)
	chapterCounts := map[string]int{}
	appendLimited := func(values []CandidateSource, quota int, enforceDiversity bool) {
		for _, candidate := range values {
			if quota <= 0 || len(result) >= limit {
				return
			}
			if seen[candidate.ID] {
				continue
			}
			group := candidateChapterKey(candidate)
			if enforceDiversity && group != "" && chapterCounts[group] >= maxChunksPerChapter {
				continue
			}
			seen[candidate.ID] = true
			chapterCounts[group]++
			result = append(result, candidate)
			quota--
		}
	}
	if len(sourceLiteralCandidates) > 0 {
		sourceLiteralQuota := minInt(30, maxInt(1, (limit*3)/4))
		appendLimited(sourceLiteralCandidates, sourceLiteralQuota, false)
		appendLimited(hierarchyCandidates, minInt(8, limit-len(result)), false)
		appendLimited(ranked, limit-len(result), true)
		if len(result) < limit {
			appendLimited(ranked, limit-len(result), false)
		}
		sortRetrieveCandidates(result)
		return result
	}
	hierarchyQuota := minInt(40, maxInt(1, (limit*3)/4))
	literalQuota := minInt(25, maxInt(1, limit/4))
	appendLimited(hierarchyCandidates, hierarchyQuota, false)
	appendLimited(literalCandidates, literalQuota, true)
	appendLimited(ranked, limit-len(result), true)
	if len(result) < limit {
		appendLimited(ranked, limit-len(result), false)
	}
	sortRetrieveCandidates(result)
	return result
}

func candidateHasRoute(candidate CandidateSource, route string) bool {
	for _, hit := range candidate.Retrievals {
		if hit.Route == route {
			return true
		}
	}
	return false
}

func roundRobinLiteralCandidatesBySource(candidates []CandidateSource) []CandidateSource {
	groups := map[string][]CandidateSource{}
	order := []string{}
	for _, candidate := range candidates {
		key := candidate.MemoryID
		if _, ok := groups[key]; !ok {
			order = append(order, key)
		}
		groups[key] = append(groups[key], candidate)
	}
	for key := range groups {
		sort.SliceStable(groups[key], func(i, j int) bool {
			left := bestLiteralScore(groups[key][i])
			right := bestLiteralScore(groups[key][j])
			if left == right {
				return groups[key][i].RetrievalScore > groups[key][j].RetrievalScore
			}
			return left > right
		})
	}
	sort.SliceStable(order, func(i, j int) bool {
		return bestLiteralScore(groups[order[i]][0]) > bestLiteralScore(groups[order[j]][0])
	})
	result := make([]CandidateSource, 0, len(candidates))
	for offset := 0; len(result) < len(candidates); offset++ {
		added := false
		for _, key := range order {
			if offset >= len(groups[key]) {
				continue
			}
			result = append(result, groups[key][offset])
			added = true
		}
		if !added {
			break
		}
	}
	return result
}

func candidateChapterKey(candidate CandidateSource) string {
	if parentID := stringMapValue(candidate.ChunkMetadata, "parent_id"); parentID != "" {
		return parentID
	}
	chapter := stringMapValue(candidate.ChunkMetadata, "chapter")
	if chapter == "" {
		return ""
	}
	return candidate.MemoryID + ":chapter:" + chapter
}

func roundRobinHierarchyCandidates(candidates []CandidateSource) []CandidateSource {
	groups := map[string][]CandidateSource{}
	groupOrder := []string{}
	for _, candidate := range candidates {
		parentID := stringMapValue(candidate.ChunkMetadata, "parent_id")
		if parentID == "" {
			parentID = candidate.MemoryID
		}
		if _, ok := groups[parentID]; !ok {
			groupOrder = append(groupOrder, parentID)
		}
		groups[parentID] = append(groups[parentID], candidate)
	}
	for _, group := range groups {
		sort.SliceStable(group, func(i, j int) bool {
			left := group[i].RetrievalScore
			right := group[j].RetrievalScore
			if left == right {
				return group[i].UpdatedAt > group[j].UpdatedAt
			}
			return left > right
		})
	}
	sort.SliceStable(groupOrder, func(i, j int) bool {
		left := groups[groupOrder[i]][0].RetrievalScore
		right := groups[groupOrder[j]][0].RetrievalScore
		if left == right {
			return groupOrder[i] < groupOrder[j]
		}
		return left > right
	})
	result := make([]CandidateSource, 0, len(candidates))
	for offset := 0; ; offset++ {
		added := false
		for _, parentID := range groupOrder {
			group := groups[parentID]
			if offset >= len(group) {
				continue
			}
			result = append(result, group[offset])
			added = true
		}
		if !added {
			return result
		}
	}
}

func bestVectorSimilarity(source CandidateSource) float64 {
	best := 0.0
	for _, hit := range source.Retrievals {
		if hit.VectorSimilarity > best {
			best = hit.VectorSimilarity
		}
	}
	return best
}

func bestLiteralScore(source CandidateSource) float64 {
	best := 0.0
	for _, hit := range source.Retrievals {
		if hit.Mode != searchModeLiteral {
			continue
		}
		if hit.LiteralScore > best {
			best = hit.LiteralScore
		}
	}
	return best
}

func dedupeRetrievalSpecs(specs []RetrievalSpec) []RetrievalSpec {
	seen := map[string]bool{}
	result := []RetrievalSpec{}
	for _, spec := range specs {
		key := strings.ToLower(strings.TrimSpace(spec.Mode) + ":" + strings.TrimSpace(spec.Query))
		if seen[key] {
			continue
		}
		seen[key] = true
		result = append(result, spec)
	}
	return result
}

func normalizeRetrievalQuery(query string) string {
	replacer := strings.NewReplacer("’", "'", "`", "'")
	query = replacer.Replace(query)
	query = strings.ReplaceAll(query, "'s", "")
	fields := strings.FieldsFunc(query, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') &&
			!(r >= 'A' && r <= 'Z') &&
			!(r >= '0' && r <= '9') &&
			!(r >= 0x80)
	})
	return strings.Join(fields, " ")
}

func retrieveLimit(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	if value > 150 {
		return 150
	}
	return value
}

func candidateEvaluationLimit(value int) int {
	if value <= 0 {
		return defaultRerankCandidates
	}
	if value > 1000 {
		return 1000
	}
	return value
}

func retrieveOrderChanged(original, reranked []CandidateSource) bool {
	for index, source := range reranked {
		if index >= len(original) || original[index].ID != source.ID {
			return true
		}
	}
	return false
}

func containsCandidate(candidates []CandidateSource, id string) bool {
	for _, candidate := range candidates {
		if candidate.ID == id {
			return true
		}
	}
	return false
}

func maxFloat(left, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func (source CandidateSource) rerankText() string {
	parts := []string{}
	text := strings.TrimSpace(source.RerankContext)
	if text == "" {
		text = strings.TrimSpace(source.Text)
	}
	if text != "" {
		parts = append(parts, "Original chunk with local context:\n"+text)
	}
	if strings.TrimSpace(source.ContextText) != "" {
		parts = append(parts, strings.TrimSpace(source.ContextText))
	}
	if strings.TrimSpace(source.ParentSummary) != "" {
		parts = append(parts, "Parent chapter summary:\n"+strings.TrimSpace(source.ParentSummary))
	}
	if strings.TrimSpace(source.MetadataText) != "" {
		parts = append(parts, strings.TrimSpace(source.MetadataText))
	}
	return strings.Join(parts, "\n\n")
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}
