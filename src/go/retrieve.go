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
	minSourcesPerMemory             = 3
	lookupSourceShareNumerator      = 3
	lookupSourceShareDenominator    = 4
	firstPhaseRRFWeight             = 0.30
	firstPhaseDenseWeight           = 0.20
	firstPhaseLiteralWeight         = 0.20
	firstPhaseHierarchyWeight       = 0.08
	firstPhaseConditionedWeight     = 0.08
	firstPhaseDocumentWeight        = 0.07
	firstPhaseSupportWeight         = 0.04
	firstPhaseFieldMatchWeight      = 0.03
	reciprocalRankConstant          = 60.0
	firstPhaseDocumentLimit         = 8
	firstPhaseDocumentShare         = 3
	firstPhaseGlobalShare           = 1
	retrievalPlanLookup             = "lookup"
	retrievalPlanOverview           = "overview"
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
	TraceStages         bool      `json:"trace_stages,omitempty"`
	Plan                string    `json:"plan,omitempty"`
	Embedding           []float64 `json:"-"`
}

type RetrieveResponse struct {
	Query       string              `json:"query"`
	Specs       []RetrievalSpec     `json:"specs"`
	Candidates  []CandidateSource   `json:"candidates,omitempty"`
	Sources     []CandidateSource   `json:"sources"`
	Diagnostics RetrieveDiagnostics `json:"diagnostics"`
	StageTrace  *RetrieveStageTrace `json:"stage_trace,omitempty"`
}

type RetrieveStageTrace struct {
	Candidates []RetrieveTraceCandidate `json:"candidates"`
	Stages     []RetrieveTraceStage     `json:"stages"`
}

type RetrieveTraceCandidate struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Text      string `json:"text"`
	SourceURI string `json:"source_uri"`
}

type RetrieveTraceStage struct {
	Name         string   `json:"name"`
	CandidateIDs []string `json:"candidate_ids"`
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
	ID              string              `json:"id"`
	MemoryID        string              `json:"memory_id"`
	ChunkID         string              `json:"chunk_id"`
	Title           string              `json:"title"`
	Text            string              `json:"text"`
	EvidenceText    string              `json:"evidence_text,omitempty"`
	ContextText     string              `json:"context_text,omitempty"`
	MetadataText    string              `json:"-"`
	ParentSummary   string              `json:"-"`
	RerankContext   string              `json:"-"`
	Kind            string              `json:"kind"`
	SourceType      string              `json:"source_type"`
	SourceURI       string              `json:"source_uri"`
	Tags            []string            `json:"tags"`
	Metadata        map[string]any      `json:"metadata"`
	ChunkMetadata   map[string]any      `json:"chunk_metadata"`
	UpdatedAt       string              `json:"updated_at"`
	MaxChars        int                 `json:"max_chars"`
	Expanded        bool                `json:"expanded,omitempty"`
	RetrievalScore  float64             `json:"retrieval_score"`
	FirstPhaseScore float64             `json:"first_phase_score,omitempty"`
	FirstPhase      *FirstPhaseFeatures `json:"first_phase,omitempty"`
	RerankScore     *float64            `json:"rerank_score,omitempty"`
	Retrievals      []RetrievalHit      `json:"retrievals"`
}

type FirstPhaseFeatures struct {
	ReciprocalRankFusion float64 `json:"reciprocal_rank_fusion"`
	DenseSimilarity      float64 `json:"dense_similarity"`
	LiteralScore         float64 `json:"literal_score"`
	HierarchyScore       float64 `json:"hierarchy_score"`
	ConditionedScore     float64 `json:"conditioned_score"`
	DocumentScore        float64 `json:"document_score"`
	RouteSupport         int     `json:"route_support"`
	TitleMatch           bool    `json:"title_match,omitempty"`
	SourceMatch          bool    `json:"source_match,omitempty"`
}

type RetrievalHit struct {
	Query             string  `json:"query"`
	Mode              string  `json:"mode"`
	Route             string  `json:"route"`
	Rank              int     `json:"rank"`
	Score             float64 `json:"score"`
	Similarity        float64 `json:"similarity"`
	LiteralScore      float64 `json:"literal_score"`
	VectorSimilarity  float64 `json:"vector_similarity"`
	FusionScore       float64 `json:"fusion_score"`
	ParentRank        int     `json:"parent_rank,omitempty"`
	ChildRank         int     `json:"child_rank,omitempty"`
	ParentPooledRank  int     `json:"parent_pooled_rank,omitempty"`
	ParentSummaryRank int     `json:"parent_summary_rank,omitempty"`
	ParentSimilarity  float64 `json:"parent_similarity,omitempty"`
	TitleMatch        bool    `json:"title_match,omitempty"`
	SourceMatch       bool    `json:"source_match,omitempty"`
	BodyMatch         bool    `json:"body_match,omitempty"`
}

type retrieveNearbyChunk struct {
	id    string
	label string
	text  string
}

type RetrieveDiagnostics struct {
	Status                     string   `json:"status"`
	Reason                     string   `json:"reason,omitempty"`
	RetrievedCandidateCount    int      `json:"retrieved_candidate_count"`
	GlobalCandidateCount       int      `json:"global_candidate_count"`
	DocumentRootCandidateCount int      `json:"document_root_candidate_count"`
	DocumentRootSelectedCount  int      `json:"document_root_selected_count"`
	DocumentRootMemoryIDs      []string `json:"document_root_memory_ids,omitempty"`
	WindowMatchedCount         int      `json:"window_matched_count"`
	WindowExpandedLeafCount    int      `json:"window_expanded_leaf_count"`
	WindowCandidateCount       int      `json:"window_candidate_count"`
	ConditionedCandidateCount  int      `json:"conditioned_candidate_count"`
	ConditionedDepths          []int    `json:"conditioned_depths,omitempty"`
	StructuralTargetCount      int      `json:"structural_target_count,omitempty"`
	StructuredCandidateCount   int      `json:"structured_candidate_count,omitempty"`
	StructuredDocumentCount    int      `json:"structured_document_count,omitempty"`
	StructuredDocumentScope    string   `json:"structured_document_scope,omitempty"`
	StructuredRoleConstrained  bool     `json:"structured_role_constrained,omitempty"`
	StructuralFallback         string   `json:"structural_fallback,omitempty"`
	CandidateCount             int      `json:"candidate_count"`
	RerankCandidateCount       int      `json:"rerank_candidate_count"`
	SelectedCount              int      `json:"selected_count"`
	SelectedDocumentCount      int      `json:"selected_document_count"`
	QueryCount                 int      `json:"query_count"`
	GlobalSearchMs             int64    `json:"global_search_ms,omitempty"`
	WindowSearchMs             int64    `json:"window_search_ms,omitempty"`
	DocumentRootSearchMs       int64    `json:"document_root_search_ms,omitempty"`
	ConditionedSearchMs        int64    `json:"conditioned_search_ms,omitempty"`
	StructuredSearchMs         int64    `json:"structured_search_ms,omitempty"`
	SearchMs                   int64    `json:"search_ms,omitempty"`
	RerankMs                   int64    `json:"rerank_ms,omitempty"`
	ExpandMs                   int64    `json:"expand_ms,omitempty"`
	TotalMs                    int64    `json:"total_ms,omitempty"`
	ReturnedIDCount            int      `json:"returned_id_count,omitempty"`
	ChangedOrder               bool     `json:"changed_order,omitempty"`
	Reranker                   string   `json:"reranker,omitempty"`
	RerankerModel              string   `json:"reranker_model,omitempty"`
	Error                      string   `json:"error,omitempty"`
	UnknownIDs                 []string `json:"unknown_ids,omitempty"`
	Plan                       string   `json:"plan,omitempty"`
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

type retrieveSelection struct {
	Sources     []CandidateSource
	Reranked    []CandidateSource
	Diagnostics RetrieveDiagnostics
}

type retrieveCandidateSearch struct {
	Candidates []CandidateSource
	Windows    passageWindowSearch
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
	userKey := s.userKeyFromRequest(r)
	response, err := s.retrieve(r.Context(), userKey, req)
	if err != nil {
		log.Printf("memory.retrieve request_id=%q status=error user=%q query=%q query_chars=%d ms=%d err=%q", req.RequestID, userKey, req.Query, len([]rune(req.Query)), time.Since(start).Milliseconds(), err.Error())
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.storeRetrieveTrace(req.RequestID, response)
	if !req.TraceStages {
		response.StageTrace = nil
	}
	log.Printf(
		"memory.retrieve request_id=%q status=ok user=%q query=%q query_chars=%d request_limit=%d request_candidate_limit=%d specs=%d global_candidates=%d window_matches=%d window_expanded_leaves=%d window_candidates=%d root_candidates=%d root_selected=%d documents=%d conditioned_candidates=%d structural_targets=%d structured_candidates=%d structural_fallback=%q candidates=%d selected=%d global_search_ms=%d window_search_ms=%d root_search_ms=%d conditioned_search_ms=%d structured_search_ms=%d search_ms=%d rerank_ms=%d total_ms=%d reranker=%q model=%q",
		req.RequestID,
		userKey,
		req.Query,
		len([]rune(req.Query)),
		req.Limit,
		req.CandidateLimit,
		len(response.Specs),
		response.Diagnostics.GlobalCandidateCount,
		response.Diagnostics.WindowMatchedCount,
		response.Diagnostics.WindowExpandedLeafCount,
		response.Diagnostics.WindowCandidateCount,
		response.Diagnostics.DocumentRootCandidateCount,
		response.Diagnostics.DocumentRootSelectedCount,
		response.Diagnostics.SelectedDocumentCount,
		response.Diagnostics.ConditionedCandidateCount,
		response.Diagnostics.StructuralTargetCount,
		response.Diagnostics.StructuredCandidateCount,
		response.Diagnostics.StructuralFallback,
		response.Diagnostics.CandidateCount,
		response.Diagnostics.SelectedCount,
		response.Diagnostics.GlobalSearchMs,
		response.Diagnostics.WindowSearchMs,
		response.Diagnostics.DocumentRootSearchMs,
		response.Diagnostics.ConditionedSearchMs,
		response.Diagnostics.StructuredSearchMs,
		response.Diagnostics.SearchMs,
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
	plan, err := normalizeRetrievalPlan(req.Plan, query)
	if err != nil {
		return RetrieveResponse{}, err
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
	type documentRootResult struct {
		hits      []documentRootHit
		elapsedMs int64
		err       error
	}
	rootResultChannel := make(chan documentRootResult, 1)
	if len(req.Embedding) > 0 {
		go func() {
			started := time.Now()
			hits, rootErr := s.searchDocumentRoots(ctx, userKey, req.Embedding, req, documentRootCandidateLimit)
			rootResultChannel <- documentRootResult{
				hits:      hits,
				elapsedMs: time.Since(started).Milliseconds(),
				err:       rootErr,
			}
		}()
	} else {
		rootResultChannel <- documentRootResult{}
	}
	retrieved, err := s.retrieveCandidates(ctx, userKey, req, specs, searchCandidateLimit, literalRerankCandidates)
	if err != nil {
		return RetrieveResponse{}, err
	}
	retrievedCandidates := retrieved.Candidates
	if !s.firstPhaseDisabled {
		retrievedCandidates = rankFirstPhaseCandidates(retrievedCandidates, nil)
	}
	globalSearchMs := time.Since(searchStart).Milliseconds()
	rootResult := <-rootResultChannel
	if rootResult.err != nil {
		log.Printf("memory.retrieve.document_roots request_id=%q status=fallback err=%q", req.RequestID, rootResult.err.Error())
		rootResult.hits = nil
	} else {
		log.Printf("memory.retrieve.document_roots request_id=%q status=ok candidates=%d ms=%d", req.RequestID, len(rootResult.hits), rootResult.elapsedMs)
	}
	passageDocuments := rankFirstPhaseDocuments(retrievedCandidates, plan)
	documents := mergeDocumentRoutes(passageDocuments, rootResult.hits, firstPhaseDocumentLimit)
	conditionedChannel := make(chan conditionedRetrieval, 1)
	structuredChannel := make(chan structuredRetrieval, 1)
	go func() { conditionedChannel <- s.retrieveConditionedCandidates(ctx, userKey, req, documents) }()
	go func() { structuredChannel <- s.retrieveStructuredCandidates(ctx, userKey, req, documents) }()
	conditioned := <-conditionedChannel
	structured := <-structuredChannel
	candidates := assembleGuardedRerankPool(
		retrievedCandidates,
		conditioned.Candidates,
		structured.Candidates,
		documents,
		rerankCandidateLimit,
		!s.firstPhaseDisabled,
	)
	if len(conditioned.Candidates) == 0 && len(structured.Candidates) == 0 {
		candidates = selectFirstPhaseCandidates(retrievedCandidates, rerankCandidateLimit, plan)
	}
	logRetrieveDocumentFlow(req.RequestID, documents, retrievedCandidates, conditioned.Candidates, candidates)
	searchMs := time.Since(searchStart).Milliseconds()
	var selection retrieveSelection
	if req.CandidateOnly {
		selection.Sources = append([]CandidateSource(nil), candidates[:minInt(sourceLimit, len(candidates))]...)
		selection.Diagnostics = RetrieveDiagnostics{
			Status:               "candidate_only",
			Reason:               "reranker_skipped",
			CandidateCount:       len(candidates),
			RerankCandidateCount: len(candidates),
			SelectedCount:        len(selection.Sources),
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
		logRetrieveCandidates("rerank_input", req.RequestID, candidates, false)
		selection = s.selectRetrieveSources(ctx, query, candidates, sourceLimit, rerankCandidateLimit, plan, req.RequestID)
	}
	sources := selection.Sources
	diagnostics := selection.Diagnostics
	diagnostics.RetrievedCandidateCount = len(retrievedCandidates)
	diagnostics.GlobalCandidateCount = len(retrievedCandidates)
	diagnostics.DocumentRootCandidateCount = len(rootResult.hits)
	diagnostics.DocumentRootMemoryIDs = rootRoutedMemoryIDs(documents)
	diagnostics.DocumentRootSelectedCount = len(diagnostics.DocumentRootMemoryIDs)
	diagnostics.WindowMatchedCount = retrieved.Windows.MatchedWindows
	diagnostics.WindowExpandedLeafCount = retrieved.Windows.ExpandedLeaves
	diagnostics.WindowCandidateCount = retrieved.Windows.CanonicalLeaves
	diagnostics.WindowSearchMs = retrieved.Windows.ElapsedMs
	diagnostics.ConditionedCandidateCount = len(conditioned.Candidates)
	diagnostics.ConditionedDepths = conditionedBudgetLimits(conditioned.Budgets)
	diagnostics.StructuralTargetCount = len(structured.Targets)
	diagnostics.StructuredCandidateCount = len(structured.Candidates)
	diagnostics.StructuredDocumentCount = structured.DocumentCount
	diagnostics.StructuredDocumentScope = structured.DocumentScope
	diagnostics.StructuredRoleConstrained = structured.RoleConstrained
	diagnostics.StructuralFallback = structured.Fallback
	diagnostics.SelectedDocumentCount = minInt(firstPhaseDocumentLimit, len(documents))
	diagnostics.GlobalSearchMs = globalSearchMs
	diagnostics.DocumentRootSearchMs = rootResult.elapsedMs
	diagnostics.ConditionedSearchMs = conditioned.ElapsedMs
	diagnostics.StructuredSearchMs = structured.ElapsedMs
	expandStart := time.Now()
	expandedSources, expandErr := s.expandRetrieveSources(ctx, userKey, sources, nearbyChunkWindow)
	if expandErr != nil {
		log.Printf("memory.retrieve.expand request_id=%q status=error query=%q selected=%d err=%q", req.RequestID, query, len(sources), expandErr.Error())
	} else {
		sources = expandedSources
		diagnostics.ExpandMs = time.Since(expandStart).Milliseconds()
	}
	logRetrieveCandidates("selected_sources", req.RequestID, sources, true)
	diagnostics.QueryCount = len(specs)
	diagnostics.Plan = plan
	diagnostics.SearchMs = searchMs
	diagnostics.TotalMs = time.Since(totalStart).Milliseconds()
	traceFinal := sources
	if req.CandidateOnly {
		traceFinal = nil
	}
	stageTrace := buildRetrieveStageTrace(
		retrievedCandidates,
		documentRepresentativeCandidates(documents, firstPhaseDocumentLimit),
		rootRoutedRepresentativeCandidates(documents, conditioned.Candidates),
		conditioned.Candidates,
		structured.Candidates,
		candidates,
		selection.Reranked,
		traceFinal,
	)
	return RetrieveResponse{
		Query:       query,
		Specs:       specs,
		Candidates:  candidates,
		Sources:     sources,
		Diagnostics: diagnostics,
		StageTrace:  stageTrace,
	}, nil
}

type retrieveDocumentLog struct {
	Rank                  int     `json:"rank"`
	Title                 string  `json:"title"`
	MemoryID              string  `json:"memory_id"`
	Score                 float64 `json:"score"`
	RootRank              int     `json:"root_rank,omitempty"`
	RootSimilarity        float64 `json:"root_similarity,omitempty"`
	GlobalCandidates      int     `json:"global_candidates"`
	ConditionedCandidates int     `json:"conditioned_candidates"`
	RerankCandidates      int     `json:"rerank_candidates"`
}

type retrieveCandidateLog struct {
	Rank             int      `json:"rank"`
	Title            string   `json:"title"`
	Page             int      `json:"page,omitempty"`
	Chapter          int      `json:"chapter,omitempty"`
	Chunk            int      `json:"chunk,omitempty"`
	StructuralTarget string   `json:"structural_target,omitempty"`
	StructuralRole   string   `json:"structural_role,omitempty"`
	TargetConfidence float64  `json:"target_confidence,omitempty"`
	FirstPhaseScore  float64  `json:"first_phase_score,omitempty"`
	RerankScore      *float64 `json:"rerank_score,omitempty"`
	Preview          string   `json:"preview,omitempty"`
}

func logRetrieveDocumentFlow(requestID string, documents []firstPhaseDocument, global, conditioned, rerank []CandidateSource) {
	counts := func(candidates []CandidateSource) map[string]int {
		result := make(map[string]int)
		for _, candidate := range candidates {
			result[candidate.MemoryID]++
		}
		return result
	}
	globalCounts := counts(global)
	conditionedCounts := counts(conditioned)
	rerankCounts := counts(rerank)
	titles := make(map[string]string)
	for _, candidates := range [][]CandidateSource{global, conditioned, rerank} {
		for _, candidate := range candidates {
			if titles[candidate.MemoryID] == "" {
				titles[candidate.MemoryID] = candidate.Title
			}
		}
	}
	entries := make([]retrieveDocumentLog, 0, len(documents))
	for index, document := range documents {
		title := titles[document.memoryID]
		if len(document.candidates) > 0 {
			title = document.candidates[0].Title
		}
		entries = append(entries, retrieveDocumentLog{
			Rank: index + 1, Title: title, MemoryID: document.memoryID, Score: document.score,
			RootRank: document.rootRank, RootSimilarity: document.rootSimilarity,
			GlobalCandidates: globalCounts[document.memoryID], ConditionedCandidates: conditionedCounts[document.memoryID],
			RerankCandidates: rerankCounts[document.memoryID],
		})
	}
	encoded, err := json.Marshal(entries)
	if err != nil {
		log.Printf("memory.retrieve.document_flow request_id=%q status=error err=%q", requestID, err.Error())
		return
	}
	log.Printf("memory.retrieve.document_flow request_id=%q documents=%s", requestID, encoded)
}

func logRetrieveCandidates(stage, requestID string, candidates []CandidateSource, includePreview bool) {
	entries := make([]retrieveCandidateLog, 0, len(candidates))
	for index, candidate := range candidates {
		entry := retrieveCandidateLog{
			Rank: index + 1, Title: candidate.Title, FirstPhaseScore: candidate.FirstPhaseScore,
			RerankScore: candidate.RerankScore,
		}
		entry.Page, _ = intMapValue(candidate.ChunkMetadata, "page")
		entry.Chapter, _ = intMapValue(candidate.ChunkMetadata, "chapter")
		entry.Chunk, _ = intMapValue(candidate.ChunkMetadata, "chunk_index")
		entry.StructuralTarget = stringMapValue(candidate.ChunkMetadata, "structural_target_label")
		entry.StructuralRole = stringMapValue(candidate.ChunkMetadata, "semantic_role")
		entry.TargetConfidence = floatMapValue(candidate.ChunkMetadata, "structural_target_confidence")
		if includePreview {
			preview := candidate.EvidenceText
			if strings.TrimSpace(preview) == "" {
				preview = candidate.Text
			}
			entry.Preview = truncateRunes(strings.Join(strings.Fields(preview), " "), 180)
		}
		entries = append(entries, entry)
	}
	encoded, err := json.Marshal(entries)
	if err != nil {
		log.Printf("memory.retrieve.%s request_id=%q status=error err=%q", stage, requestID, err.Error())
		return
	}
	log.Printf("memory.retrieve.%s request_id=%q candidates=%s", stage, requestID, encoded)
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
			if strings.TrimSpace(result[index].RerankContext) == "" {
				result[index].RerankContext = centeredEvidenceText(chunks, selected[result[index].ID], rerankWindowChars)
			}
		}
	}
	return result, nil
}

func (s *Server) retrieveCandidates(ctx context.Context, userKey string, req RetrieveRequest, specs []RetrievalSpec, limit, literalLimit int) (retrieveCandidateSearch, error) {
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
	windowOut := make(chan passageWindowSearch, 1)
	var wg sync.WaitGroup
	go func() {
		if s.passageWindowsDisabled || len(req.Embedding) == 0 || normalizeSearchMode(req.Mode) == searchModeLiteral {
			windowOut <- passageWindowSearch{}
			return
		}
		result, windowErr := s.searchPassageWindows(ctx, userKey, req.Embedding, req)
		if windowErr != nil {
			log.Printf("memory.retrieve.window_search request_id=%q status=fallback err=%q", req.RequestID, windowErr.Error())
			windowOut <- passageWindowSearch{}
			return
		}
		log.Printf(
			"memory.retrieve.window_search request_id=%q status=ok windows=%d expanded_leaves=%d canonical_leaves=%d ms=%d",
			req.RequestID,
			result.MatchedWindows,
			result.ExpandedLeaves,
			result.CanonicalLeaves,
			result.ElapsedMs,
		)
		windowOut <- result
	}()
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
				Embedding:           embeddingForSpec(req.Embedding, spec.Mode),
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
			return retrieveCandidateSearch{}, result.err
		}
		addRetrieveCandidates(byID, result.results, result.spec, "direct")
	}
	parent := <-parentOut
	if parent.err != nil {
		return retrieveCandidateSearch{}, parent.err
	}
	addRetrieveCandidates(byID, parent.results, parent.spec, "parent")
	windowResult := <-windowOut
	addRetrieveCandidates(byID, windowResult.Results, RetrievalSpec{Query: strings.TrimSpace(req.Query), Mode: searchModeSemantic}, "window")
	candidates := make([]CandidateSource, 0, len(byID))
	for _, candidate := range byID {
		candidates = append(candidates, *candidate)
	}
	sortRetrieveCandidates(candidates)
	return retrieveCandidateSearch{Candidates: candidates, Windows: windowResult}, nil
}

func (s *Server) selectRetrieveSources(ctx context.Context, query string, candidates []CandidateSource, sourceLimit, candidateLimit int, plan, requestID string) retrieveSelection {
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
		return retrieveSelection{Sources: topCandidates, Diagnostics: base}
	}
	rerankStart := time.Now()
	reranked, result, err := s.rerankRetrieveSources(ctx, query, topCandidates, requestID)
	base.RerankMs = time.Since(rerankStart).Milliseconds()
	if err != nil {
		base.Reranker = "retrieval_score"
		base.Error = err.Error()
		return retrieveSelection{Sources: topCandidates[:sourceLimit], Diagnostics: base}
	}
	base.Status = "reranked"
	rawReranked := append([]CandidateSource(nil), reranked...)
	reranked = filterQueryEchoSources(query, reranked)
	perMemoryLimit := lookupPerMemoryLimit(sourceLimit)
	if plan == retrievalPlanOverview {
		perMemoryLimit = sourceLimit
	}
	sources := diversifyRetrieveSources(reranked, sourceLimit, perMemoryLimit)
	base.SelectedCount = len(sources)
	base.ReturnedIDCount = len(result.Results)
	base.ChangedOrder = retrieveOrderChanged(topCandidates, sources)
	base.Reranker = "cross_encoder"
	base.RerankerModel = result.Model
	return retrieveSelection{Sources: sources, Reranked: rawReranked, Diagnostics: base}
}

func buildRetrieveStageTrace(
	retrieved,
	documentSelection,
	documentRootSelection,
	conditioned,
	structured,
	rerankPool,
	reranked,
	final []CandidateSource,
) *RetrieveStageTrace {
	type lane struct {
		name       string
		candidates []CandidateSource
		match      func(RetrievalHit) bool
	}
	lanes := []lane{
		{name: "direct_hybrid", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "direct" && hit.Mode == searchModeHybrid }},
		{name: "direct_semantic", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "direct" && hit.Mode == searchModeSemantic }},
		{name: "direct_literal", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "direct" && hit.Mode == searchModeLiteral }},
		{name: "passage_windows", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "window" }},
		{name: "hierarchy", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "parent" }},
		{name: "hierarchy_pooled", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "parent" && hit.ParentPooledRank > 0 }},
		{name: "hierarchy_summary", candidates: retrieved, match: func(hit RetrievalHit) bool { return hit.Route == "parent" && hit.ParentSummaryRank > 0 }},
		{name: "conditioned_dense", candidates: conditioned, match: func(hit RetrievalHit) bool { return hit.Route == "conditioned_dense" }},
		{name: "conditioned_literal", candidates: conditioned, match: func(hit RetrievalHit) bool { return hit.Route == "conditioned_literal" }},
		{name: "structured_dense", candidates: structured, match: func(hit RetrievalHit) bool { return hit.Route == "structured_dense" }},
		{name: "structured_literal", candidates: structured, match: func(hit RetrievalHit) bool { return hit.Route == "structured_literal" }},
	}
	orderedStages := make([]struct {
		name       string
		candidates []CandidateSource
	}, 0, len(lanes)+5)
	for _, current := range lanes {
		candidates := rankRetrieveLaneCandidates(current.candidates, current.match)
		if len(candidates) > 0 {
			orderedStages = append(orderedStages, struct {
				name       string
				candidates []CandidateSource
			}{name: current.name, candidates: candidates})
		}
	}
	for _, stage := range []struct {
		name       string
		candidates []CandidateSource
	}{
		{name: "global_matched", candidates: retrieved},
		{name: "document_selection", candidates: documentSelection},
		{name: "document_root_selection", candidates: documentRootSelection},
		{name: "rerank_pool", candidates: rerankPool},
		{name: "cross_encoder", candidates: reranked},
		{name: "final", candidates: final},
	} {
		if len(stage.candidates) > 0 {
			orderedStages = append(orderedStages, stage)
		}
	}

	trace := &RetrieveStageTrace{}
	seenCandidates := make(map[string]bool)
	for _, stage := range orderedStages {
		ids := make([]string, 0, len(stage.candidates))
		seenIDs := make(map[string]bool, len(stage.candidates))
		for _, candidate := range stage.candidates {
			if candidate.ID == "" || seenIDs[candidate.ID] {
				continue
			}
			seenIDs[candidate.ID] = true
			ids = append(ids, candidate.ID)
			if !seenCandidates[candidate.ID] {
				seenCandidates[candidate.ID] = true
				trace.Candidates = append(trace.Candidates, RetrieveTraceCandidate{
					ID:        candidate.ID,
					Title:     candidate.Title,
					Text:      candidate.Text,
					SourceURI: candidate.SourceURI,
				})
			}
		}
		trace.Stages = append(trace.Stages, RetrieveTraceStage{Name: stage.name, CandidateIDs: ids})
	}
	return trace
}

func rankRetrieveLaneCandidates(candidates []CandidateSource, matches func(RetrievalHit) bool) []CandidateSource {
	type laneCandidate struct {
		candidate CandidateSource
		score     float64
		bestRank  int
	}
	ranked := make([]laneCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		entry := laneCandidate{candidate: candidate}
		for _, hit := range candidate.Retrievals {
			if !matches(hit) {
				continue
			}
			entry.score += hit.FusionScore
			if entry.bestRank == 0 || hit.Rank < entry.bestRank {
				entry.bestRank = hit.Rank
			}
		}
		if entry.bestRank > 0 {
			ranked = append(ranked, entry)
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].score == ranked[j].score {
			if ranked[i].bestRank == ranked[j].bestRank {
				return ranked[i].candidate.ID < ranked[j].candidate.ID
			}
			return ranked[i].bestRank < ranked[j].bestRank
		}
		return ranked[i].score > ranked[j].score
	})
	result := make([]CandidateSource, len(ranked))
	for index := range ranked {
		result[index] = ranked[index].candidate
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

func lookupPerMemoryLimit(sourceLimit int) int {
	shareLimit := (sourceLimit*lookupSourceShareNumerator + lookupSourceShareDenominator - 1) / lookupSourceShareDenominator
	return maxInt(minSourcesPerMemory, minInt(sourceLimit, shareLimit))
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

func selectEvidenceSet(sources []CandidateSource, limit, perMemoryLimit int) []CandidateSource {
	if limit <= 0 || len(sources) == 0 {
		return nil
	}
	representative := make([]CandidateSource, 0, len(sources))
	repeated := make([]CandidateSource, 0, len(sources))
	seenGroups := map[string]bool{}
	for _, source := range sources {
		group := candidateChapterKey(source)
		if group == "" {
			group = source.MemoryID + ":" + source.ChunkID
		}
		if seenGroups[group] {
			repeated = append(repeated, source)
			continue
		}
		seenGroups[group] = true
		representative = append(representative, source)
	}
	ordered := append(representative, repeated...)
	return diversifyRetrieveSources(ordered, limit, perMemoryLimit)
}

func candidateChapterKey(candidate CandidateSource) string {
	if parentID := stringMapValue(candidate.ChunkMetadata, "route_node_id"); parentID != "" {
		return parentID
	}
	chapter := stringMapValue(candidate.ChunkMetadata, "chapter")
	if chapter == "" {
		return ""
	}
	return candidate.MemoryID + ":chapter:" + chapter
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
	query = strings.TrimSpace(query)
	normalized := strings.TrimSpace(normalizeRetrievalQuery(query))
	preferredQuery := query
	if preferred == searchModeLiteral && normalized != "" {
		preferredQuery = normalized
	}
	specs := []RetrievalSpec{}
	if preferredQuery != "" {
		specs = append(specs, RetrievalSpec{Query: preferredQuery, Mode: preferred})
	}
	if preferred != searchModeLiteral && normalized != "" {
		specs = append(specs, RetrievalSpec{Query: normalized, Mode: searchModeLiteral})
	}
	return dedupeRetrievalSpecs(specs)
}

func embeddingForSpec(embedding []float64, mode string) []float64 {
	if mode == searchModeLiteral {
		return nil
	}
	return embedding
}

func addRetrieveCandidates(candidates map[string]*CandidateSource, results []SearchResult, spec RetrievalSpec, route string) {
	rankedResults := append([]SearchResult(nil), results...)
	sort.SliceStable(rankedResults, func(i, j int) bool {
		if route == "parent" {
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
		rank := index + 1
		if strings.HasPrefix(route, "conditioned_") {
			if conditionedRank, ok := intMapValue(result.ChunkMetadata, "conditioned_rank"); ok && conditionedRank > 0 {
				rank = conditionedRank
			}
		}
		fusionScore := reciprocalRankScore(rank)
		parentRank, _ := intMapValue(result.ChunkMetadata, "parent_rank")
		childRank, _ := intMapValue(result.ChunkMetadata, "child_rank")
		parentPooledRank, _ := intMapValue(result.ChunkMetadata, "parent_pooled_rank")
		parentSummaryRank, _ := intMapValue(result.ChunkMetadata, "parent_summary_rank")
		if route == "parent" {
			if hierarchicalScore := hierarchicalFusionScore(result); hierarchicalScore > 0 {
				fusionScore = hierarchicalScore
			}
		}
		hit := RetrievalHit{
			Query:             spec.Query,
			Mode:              spec.Mode,
			Route:             route,
			Rank:              rank,
			Score:             maxFloat(result.Score, result.Similarity),
			Similarity:        result.Similarity,
			LiteralScore:      result.ScoreParts.LiteralScore,
			VectorSimilarity:  result.ScoreParts.VectorSimilarity,
			FusionScore:       fusionScore,
			ParentRank:        parentRank,
			ChildRank:         childRank,
			ParentPooledRank:  parentPooledRank,
			ParentSummaryRank: parentSummaryRank,
			ParentSimilarity:  floatMapValue(result.ChunkMetadata, "parent_similarity"),
			TitleMatch:        result.ScoreParts.TitleMatch,
			SourceMatch:       result.ScoreParts.SourceMatch,
			BodyMatch:         result.ScoreParts.BodyMatch,
		}
		if existing := candidates[id]; existing != nil {
			existing.RetrievalScore += hit.FusionScore
			if existing.ParentSummary == "" {
				existing.ParentSummary = result.ParentSummary
			}
			if existing.RerankContext == "" && result.RerankContext != "" {
				existing.RerankContext = result.RerankContext
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
			RerankContext:  result.RerankContext,
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
	literalRank, _ := intMapValue(metadata, "node_bm25_rank")
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
		left := candidateFirstPhaseOrderScore(candidates[i])
		right := candidateFirstPhaseOrderScore(candidates[j])
		if left == right {
			if candidates[i].UpdatedAt == candidates[j].UpdatedAt {
				return candidates[i].ID < candidates[j].ID
			}
			return candidates[i].UpdatedAt > candidates[j].UpdatedAt
		}
		return left > right
	})
}

func candidateFirstPhaseOrderScore(candidate CandidateSource) float64 {
	if candidate.FirstPhase != nil {
		return candidate.FirstPhaseScore
	}
	return candidate.RetrievalScore
}

func rankFirstPhaseCandidates(candidates []CandidateSource, documents []firstPhaseDocument) []CandidateSource {
	if len(candidates) == 0 {
		return nil
	}
	documentScores := make(map[string]float64, len(documents))
	for _, document := range documents {
		documentScores[document.memoryID] = document.score
	}
	result := append([]CandidateSource(nil), candidates...)
	maxima := FirstPhaseFeatures{}
	for index := range result {
		features := extractFirstPhaseFeatures(result[index])
		features.DocumentScore = documentScores[result[index].MemoryID]
		result[index].FirstPhase = &features
		maxima.ReciprocalRankFusion = maxFloat(maxima.ReciprocalRankFusion, features.ReciprocalRankFusion)
		maxima.DenseSimilarity = maxFloat(maxima.DenseSimilarity, features.DenseSimilarity)
		maxima.LiteralScore = maxFloat(maxima.LiteralScore, features.LiteralScore)
		maxima.HierarchyScore = maxFloat(maxima.HierarchyScore, features.HierarchyScore)
		maxima.ConditionedScore = maxFloat(maxima.ConditionedScore, features.ConditionedScore)
		maxima.DocumentScore = maxFloat(maxima.DocumentScore, features.DocumentScore)
		maxima.RouteSupport = maxInt(maxima.RouteSupport, features.RouteSupport)
	}
	for index := range result {
		features := result[index].FirstPhase
		fieldMatch := 0.0
		if features.TitleMatch || features.SourceMatch {
			fieldMatch = 1
		}
		result[index].FirstPhaseScore =
			firstPhaseRRFWeight*normalizePositiveFeature(features.ReciprocalRankFusion, maxima.ReciprocalRankFusion) +
				firstPhaseDenseWeight*normalizePositiveFeature(features.DenseSimilarity, maxima.DenseSimilarity) +
				firstPhaseLiteralWeight*normalizePositiveFeature(features.LiteralScore, maxima.LiteralScore) +
				firstPhaseHierarchyWeight*normalizePositiveFeature(features.HierarchyScore, maxima.HierarchyScore) +
				firstPhaseConditionedWeight*normalizePositiveFeature(features.ConditionedScore, maxima.ConditionedScore) +
				firstPhaseDocumentWeight*normalizePositiveFeature(features.DocumentScore, maxima.DocumentScore) +
				firstPhaseSupportWeight*normalizePositiveFeature(float64(features.RouteSupport), float64(maxima.RouteSupport)) +
				firstPhaseFieldMatchWeight*fieldMatch
	}
	sortRetrieveCandidates(result)
	return result
}

func extractFirstPhaseFeatures(candidate CandidateSource) FirstPhaseFeatures {
	features := FirstPhaseFeatures{ReciprocalRankFusion: candidate.RetrievalScore}
	routes := map[string]bool{}
	for _, hit := range candidate.Retrievals {
		features.DenseSimilarity = maxFloat(features.DenseSimilarity, maxFloat(hit.VectorSimilarity, hit.Similarity))
		features.LiteralScore = maxFloat(features.LiteralScore, hit.LiteralScore)
		features.TitleMatch = features.TitleMatch || hit.TitleMatch
		features.SourceMatch = features.SourceMatch || hit.SourceMatch
		routes[hit.Route+":"+hit.Mode] = true
		switch {
		case hit.Route == "parent":
			features.HierarchyScore = maxFloat(features.HierarchyScore, hit.FusionScore)
		case strings.HasPrefix(hit.Route, "conditioned_"):
			features.ConditionedScore += hit.FusionScore
		}
	}
	features.RouteSupport = len(routes)
	return features
}

func normalizePositiveFeature(value, maximum float64) float64 {
	if value <= 0 || maximum <= 0 {
		return 0
	}
	return minFloat(1, value/maximum)
}

func normalizeRetrievalPlan(value, query string) (string, error) {
	switch plan := strings.ToLower(strings.TrimSpace(value)); plan {
	case "":
		return retrievalPlanForQuery(query), nil
	case retrievalPlanLookup, retrievalPlanOverview:
		return plan, nil
	default:
		return "", fmt.Errorf("plan must be lookup or overview")
	}
}

func retrievalPlanForQuery(query string) string {
	query = strings.ToLower(strings.TrimSpace(query))
	for _, cue := range []string{
		"overview", "summarize", "summary", "main ideas", "major ideas",
		"what is in chapter", "what happens in chapter", "what is this chapter about",
		"what is the book about", "describe the chapter", "describe the book",
	} {
		if strings.Contains(query, cue) {
			return retrievalPlanOverview
		}
	}
	return retrievalPlanLookup
}

type firstPhaseDocument struct {
	memoryID       string
	score          float64
	candidates     []CandidateSource
	rootRank       int
	rootSimilarity float64
	passageCount   int
	rootRouted     bool
}

func selectFirstPhaseCandidates(candidates []CandidateSource, limit int, plan string) []CandidateSource {
	ranked := append([]CandidateSource(nil), candidates...)
	sortRetrieveCandidates(ranked)
	if limit <= 0 || len(ranked) <= limit {
		return ranked
	}
	documents := rankFirstPhaseDocuments(ranked, plan)
	documentRanks := make(map[string]int, len(documents))
	documentScores := make(map[string]float64, len(documents))
	for index, document := range documents {
		documentRanks[document.memoryID] = index + 1
		documentScores[document.memoryID] = document.score
	}
	reservedDocuments := documents
	if len(reservedDocuments) > firstPhaseDocumentLimit {
		reservedDocuments = reservedDocuments[:firstPhaseDocumentLimit]
	}
	result := make([]CandidateSource, 0, limit)
	seen := make(map[string]bool, limit)
	appendCandidate := func(candidate CandidateSource) bool {
		if len(result) >= limit || seen[candidate.ID] {
			return false
		}
		candidate.ChunkMetadata = cloneMap(candidate.ChunkMetadata)
		candidate.ChunkMetadata["first_phase_document_rank"] = documentRanks[candidate.MemoryID]
		candidate.ChunkMetadata["first_phase_document_score"] = documentScores[candidate.MemoryID]
		seen[candidate.ID] = true
		result = append(result, candidate)
		return true
	}

	documentQuota := (limit * firstPhaseDocumentShare) / (firstPhaseDocumentShare + firstPhaseGlobalShare)
	for offset := 0; len(result) < documentQuota; offset++ {
		added := false
		for _, document := range reservedDocuments {
			if offset >= len(document.candidates) {
				continue
			}
			if appendCandidate(document.candidates[offset]) {
				added = true
			}
			if len(result) >= documentQuota {
				break
			}
		}
		if !added {
			break
		}
	}
	for _, candidate := range ranked {
		if len(result) >= limit {
			break
		}
		appendCandidate(candidate)
	}
	return result
}

func rankFirstPhaseDocuments(candidates []CandidateSource, plan string) []firstPhaseDocument {
	ranked := append([]CandidateSource(nil), candidates...)
	sortRetrieveCandidates(ranked)
	byMemory := make(map[string][]CandidateSource)
	memoryOrder := make([]string, 0)
	globalRank := make(map[string]int, len(ranked))
	for index, candidate := range ranked {
		globalRank[candidate.ID] = index + 1
		if _, ok := byMemory[candidate.MemoryID]; !ok {
			memoryOrder = append(memoryOrder, candidate.MemoryID)
		}
		byMemory[candidate.MemoryID] = append(byMemory[candidate.MemoryID], candidate)
	}
	documents := make([]firstPhaseDocument, 0, len(memoryOrder))
	for _, memoryID := range memoryOrder {
		documentCandidates := byMemory[memoryID]
		sortRetrieveCandidates(documentCandidates)
		topCount := minInt(3, len(documentCandidates))
		sum := 0.0
		for index := 0; index < topCount; index++ {
			sum += candidateFirstPhaseOrderScore(documentCandidates[index])
		}
		score := candidateFirstPhaseOrderScore(documentCandidates[0]) + 0.5*(sum/float64(topCount))
		score += 0.25 * reciprocalRankScore(globalRank[documentCandidates[0].ID])
		if plan == retrievalPlanOverview {
			sort.SliceStable(documentCandidates, func(i, j int) bool {
				leftHierarchy := documentCandidates[i].ChunkMetadata["hierarchy_parent_hit"] == true
				rightHierarchy := documentCandidates[j].ChunkMetadata["hierarchy_parent_hit"] == true
				if leftHierarchy != rightHierarchy {
					return leftHierarchy
				}
				return candidateFirstPhaseOrderScore(documentCandidates[i]) > candidateFirstPhaseOrderScore(documentCandidates[j])
			})
		}
		documents = append(documents, firstPhaseDocument{
			memoryID:   memoryID,
			score:      score,
			candidates: documentCandidates,
		})
	}
	sort.SliceStable(documents, func(i, j int) bool {
		if documents[i].score == documents[j].score {
			return documents[i].memoryID < documents[j].memoryID
		}
		return documents[i].score > documents[j].score
	})
	return documents
}

func cloneMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source)+2)
	for key, value := range source {
		result[key] = value
	}
	return result
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

func minFloat(left, right float64) float64 {
	if left < right {
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
