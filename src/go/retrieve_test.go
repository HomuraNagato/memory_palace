package main

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestBuildRetrieveStageTraceKeepsActiveLaneOrders(t *testing.T) {
	answer := CandidateSource{
		ID:        "answer",
		Title:     "Book",
		Text:      "answer evidence",
		SourceURI: "file:///book.epub",
		Retrievals: []RetrievalHit{
			{Route: "direct", Mode: searchModeHybrid, Rank: 5},
			{Route: "direct", Mode: searchModeLiteral, Rank: 1},
			{Route: "window", Mode: searchModeSemantic, Rank: 2},
		},
	}
	denseFirst := CandidateSource{
		ID:         "dense-first",
		Retrievals: []RetrievalHit{{Route: "direct", Mode: searchModeHybrid, Rank: 1}},
	}
	hierarchy := CandidateSource{
		ID:         "hierarchy",
		Retrievals: []RetrievalHit{{Route: "parent", Mode: searchModeSemantic, Rank: 1, ParentPooledRank: 2, ParentSummaryRank: 1}},
	}
	conditionedAnswer := answer
	conditionedAnswer.Retrievals = append(conditionedAnswer.Retrievals,
		RetrievalHit{Route: "conditioned_literal", Mode: searchModeLiteral, Rank: 1},
	)
	structuredAnswer := answer
	structuredAnswer.Retrievals = append(structuredAnswer.Retrievals,
		RetrievalHit{Route: "structured_dense", Mode: searchModeSemantic, Rank: 1},
	)

	trace := buildRetrieveStageTrace(
		[]CandidateSource{answer, denseFirst, hierarchy},
		[]CandidateSource{hierarchy},
		nil,
		[]CandidateSource{conditionedAnswer},
		[]CandidateSource{structuredAnswer},
		[]CandidateSource{hierarchy, conditionedAnswer},
		[]CandidateSource{answer, hierarchy},
		[]CandidateSource{answer},
	)

	if len(trace.Candidates) != 3 {
		t.Fatalf("expected deduplicated candidate catalog, got %#v", trace.Candidates)
	}
	assertTraceStage(t, trace, "direct_hybrid", []string{"dense-first", "answer"})
	assertTraceStage(t, trace, "direct_literal", []string{"answer"})
	assertTraceStage(t, trace, "passage_windows", []string{"answer"})
	assertTraceStage(t, trace, "hierarchy", []string{"hierarchy"})
	assertTraceStage(t, trace, "hierarchy_pooled", []string{"hierarchy"})
	assertTraceStage(t, trace, "hierarchy_summary", []string{"hierarchy"})
	assertTraceStage(t, trace, "conditioned_literal", []string{"answer"})
	assertTraceStage(t, trace, "structured_dense", []string{"answer"})
	assertTraceStage(t, trace, "global_matched", []string{"answer", "dense-first", "hierarchy"})
	assertTraceStage(t, trace, "document_selection", []string{"hierarchy"})
	assertTraceStage(t, trace, "rerank_pool", []string{"hierarchy", "answer"})
	assertTraceStage(t, trace, "cross_encoder", []string{"answer", "hierarchy"})
	assertTraceStage(t, trace, "final", []string{"answer"})
}

func TestRetrieveResponseOmitsDisabledStageTrace(t *testing.T) {
	encoded, err := json.Marshal(RetrieveResponse{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "stage_trace") {
		t.Fatalf("disabled trace leaked into response: %s", encoded)
	}
}

func TestRetrievalSpecsUseGenericNormalizedAndLiteralQueries(t *testing.T) {
	specs := retrievalSpecsFor("Niko's mother's name?", searchModeHybrid)
	want := []RetrievalSpec{
		{Query: "Niko's mother's name?", Mode: searchModeHybrid},
		{Query: "Niko mother name", Mode: searchModeLiteral},
	}
	if len(specs) != len(want) {
		t.Fatalf("expected %d specs, got %#v", len(want), specs)
	}
	for index := range want {
		if specs[index] != want[index] {
			t.Fatalf("spec %d = %#v, want %#v", index, specs[index], want[index])
		}
	}
}

func TestAssembleRerankPoolReservesConditionedAndGlobalCandidates(t *testing.T) {
	global := []CandidateSource{
		{ID: "a-global", MemoryID: "a", RetrievalScore: 1, ChunkMetadata: map[string]any{}},
		{ID: "outside", MemoryID: "outside", RetrievalScore: 0.9, ChunkMetadata: map[string]any{}},
	}
	conditioned := []CandidateSource{
		{
			ID:             "a-answer",
			MemoryID:       "a",
			RetrievalScore: reciprocalRankScore(1),
			Retrievals:     []RetrievalHit{{Route: "conditioned_dense", Rank: 1, FusionScore: reciprocalRankScore(1)}},
			ChunkMetadata:  map[string]any{},
		},
		{
			ID:             "b-answer",
			MemoryID:       "b",
			RetrievalScore: reciprocalRankScore(1),
			Retrievals:     []RetrievalHit{{Route: "conditioned_literal", Rank: 1, FusionScore: reciprocalRankScore(1)}},
			ChunkMetadata:  map[string]any{},
		},
	}
	documents := []firstPhaseDocument{
		{memoryID: "a", score: 1},
		{memoryID: "b", score: 0.8},
	}

	selected := assembleRerankPool(global, conditioned, documents, 4)
	for _, id := range []string{"a-answer", "b-answer", "outside"} {
		if !containsCandidate(selected, id) {
			t.Fatalf("expected %q in bounded pool, got %#v", id, selected)
		}
	}
	if len(selected) != 4 {
		t.Fatalf("expected full pool of 4, got %d", len(selected))
	}
}

func TestAssembleRerankPoolAddsQuarterConditionedCandidates(t *testing.T) {
	documents := []firstPhaseDocument{{memoryID: "a", score: 1}, {memoryID: "b", score: 0.9}}
	global := make([]CandidateSource, 0, 8)
	conditioned := make([]CandidateSource, 0, 8)
	for index := 0; index < 8; index++ {
		global = append(global, CandidateSource{
			ID:             "g-" + string(rune('a'+index)),
			MemoryID:       "global",
			RetrievalScore: 1 - float64(index)/100,
			ChunkMetadata:  map[string]any{},
		})
		memoryID := "a"
		if index%2 == 1 {
			memoryID = "b"
		}
		conditioned = append(conditioned, CandidateSource{
			ID:             "c-" + string(rune('a'+index)),
			MemoryID:       memoryID,
			RetrievalScore: 1 - float64(index)/100,
			Retrievals:     []RetrievalHit{{Route: "conditioned_dense", Rank: index/2 + 1, FusionScore: reciprocalRankScore(index/2 + 1)}},
			ChunkMetadata:  map[string]any{},
		})
	}

	selected := assembleRerankPool(global, conditioned, documents, 8)
	conditionedCount := 0
	globalCount := 0
	for _, candidate := range selected {
		if strings.HasPrefix(candidate.ID, "g") {
			globalCount++
		} else {
			conditionedCount++
		}
	}
	if conditionedCount != 2 || globalCount != 6 {
		t.Fatalf("expected 2 conditioned and 6 global candidates, got %d and %d", conditionedCount, globalCount)
	}
}

func TestAdaptiveConditionedBudgetsFollowDocumentConfidence(t *testing.T) {
	documents := []firstPhaseDocument{
		{memoryID: "a", score: 1},
		{memoryID: "b", score: 0.5},
		{memoryID: "c", score: 0.5},
		{memoryID: "d", score: 0.25},
	}
	budgets := adaptiveConditionedBudgets(documents, 12, 8)
	want := []conditionedDocumentBudget{
		{MemoryID: "a", Limit: 7},
		{MemoryID: "b", Limit: 2},
		{MemoryID: "c", Limit: 2},
		{MemoryID: "d", Limit: 1},
	}
	if !reflect.DeepEqual(budgets, want) {
		t.Fatalf("budgets = %#v, want %#v", budgets, want)
	}
}

func TestAdaptiveConditionedBudgetsRemainBroadWhenScoresAreEven(t *testing.T) {
	documents := []firstPhaseDocument{
		{memoryID: "a", score: 1},
		{memoryID: "b", score: 1},
		{memoryID: "c", score: 1},
		{memoryID: "d", score: 1},
	}
	budgets := adaptiveConditionedBudgets(documents, 12, 8)
	for _, budget := range budgets {
		if budget.Limit != 3 {
			t.Fatalf("expected even depth allocation, got %#v", budgets)
		}
	}
}

func TestMergeDocumentRoutesReservesTwoUniqueRootDocuments(t *testing.T) {
	passages := make([]firstPhaseDocument, 0, 9)
	for index := 0; index < 9; index++ {
		passages = append(passages, firstPhaseDocument{
			memoryID:   "passage-" + string(rune('a'+index)),
			score:      1 - float64(index)/10,
			candidates: []CandidateSource{{ID: "candidate-" + string(rune('a'+index))}},
		})
	}
	roots := []documentRootHit{
		{MemoryID: "passage-a", Rank: 1, Similarity: 0.9, Passages: 12},
		{MemoryID: "root-only", Rank: 2, Similarity: 0.8, Passages: 3},
		{MemoryID: "passage-h", Rank: 3, Similarity: 0.7, Passages: 8},
	}

	selected := mergeDocumentRoutes(passages, roots, 8)
	if len(selected) != 8 {
		t.Fatalf("expected 8 selected documents, got %#v", selected)
	}
	for index, memoryID := range []string{"passage-a", "passage-b", "passage-c", "passage-d", "passage-e", "passage-f", "root-only", "passage-h"} {
		if selected[index].memoryID != memoryID {
			t.Fatalf("selected[%d] = %q, want %q", index, selected[index].memoryID, memoryID)
		}
	}
	if selected[0].rootRouted {
		t.Fatal("a duplicate root hit already selected by passages must not consume root reserve")
	}
	if !selected[6].rootRouted || !selected[7].rootRouted {
		t.Fatalf("expected reserved roots to be marked, got %#v", selected)
	}
}

func TestAdaptiveConditionedBudgetsGiveRootRoutesBoundedInitialDepth(t *testing.T) {
	documents := []firstPhaseDocument{
		{memoryID: "passage", score: 1},
		{memoryID: "short-root", score: 0.8, rootRouted: true, passageCount: 3},
		{memoryID: "long-root", score: 0.7, rootRouted: true, passageCount: 20},
	}
	budgets := adaptiveConditionedBudgets(documents, 12, 8)
	if budgets[1].Limit < 3 || budgets[2].Limit < 6 {
		t.Fatalf("expected root-routed initial depths of 3 and 6, got %#v", budgets)
	}
	total := 0
	for _, budget := range budgets {
		total += budget.Limit
	}
	if total != 12 {
		t.Fatalf("expected exact total budget 12, got %d from %#v", total, budgets)
	}
}

func TestFilterConditionedResultsUsesPerDocumentDepth(t *testing.T) {
	results := []SearchResult{
		{MemoryID: "a", ChunkID: "a1", ChunkMetadata: map[string]any{"conditioned_rank": 1}},
		{MemoryID: "a", ChunkID: "a2", ChunkMetadata: map[string]any{"conditioned_rank": 2}},
		{MemoryID: "b", ChunkID: "b1", ChunkMetadata: map[string]any{"conditioned_rank": 1}},
		{MemoryID: "b", ChunkID: "b2", ChunkMetadata: map[string]any{"conditioned_rank": 2}},
	}
	filtered := filterConditionedResults(results, []conditionedDocumentBudget{
		{MemoryID: "a", Limit: 2},
		{MemoryID: "b", Limit: 1},
	})
	if len(filtered) != 3 || filtered[2].ChunkID != "b1" {
		t.Fatalf("unexpected filtered results: %#v", filtered)
	}
}

func TestRetrievalSpecsForLiteralUsesSingleNormalizedQuery(t *testing.T) {
	specs := retrievalSpecsFor("Niko's mother's name?", searchModeLiteral)
	if len(specs) != 1 || specs[0] != (RetrievalSpec{Query: "Niko mother name", Mode: searchModeLiteral}) {
		t.Fatalf("unexpected literal specs: %#v", specs)
	}
}

func TestAddRetrieveCandidatesAccumulatesReciprocalRanks(t *testing.T) {
	candidates := map[string]*CandidateSource{}
	result := SearchResult{MemoryID: "memory", ChunkID: "chunk", Title: "Title", Text: "Evidence"}
	spec := RetrievalSpec{Query: "question", Mode: searchModeSemantic}
	addRetrieveCandidates(candidates, []SearchResult{result}, spec, "direct")
	addRetrieveCandidates(candidates, []SearchResult{result}, spec, "parent")

	got := candidates["memory:chunk"]
	want := 2 * reciprocalRankScore(1)
	if got == nil || math.Abs(got.RetrievalScore-want) > 0.0000001 {
		t.Fatalf("expected fused score %f, got %#v", want, got)
	}
	if len(got.Retrievals) != 2 || got.Retrievals[1].Route != "parent" {
		t.Fatalf("expected both retrieval routes, got %#v", got.Retrievals)
	}
}

func TestAddRetrieveCandidatesCollapsesWindowAndLeafHits(t *testing.T) {
	candidates := map[string]*CandidateSource{}
	spec := RetrievalSpec{Query: "question", Mode: searchModeSemantic}
	direct := SearchResult{MemoryID: "memory", ChunkID: "chunk", Title: "Title", Text: "Evidence"}
	window := direct
	window.RerankContext = "Surrounding context with Evidence in the middle."

	addRetrieveCandidates(candidates, []SearchResult{direct}, spec, "direct")
	addRetrieveCandidates(candidates, []SearchResult{window}, spec, "window")

	if len(candidates) != 1 {
		t.Fatalf("expected one canonical leaf, got %#v", candidates)
	}
	got := candidates["memory:chunk"]
	if got == nil || len(got.Retrievals) != 2 {
		t.Fatalf("expected merged retrieval signals, got %#v", got)
	}
	if got.RerankContext != window.RerankContext {
		t.Fatalf("expected window context to survive deduplication, got %q", got.RerankContext)
	}
}

func TestRankFirstPhaseCandidatesPreservesComplementaryFeatures(t *testing.T) {
	candidates := []CandidateSource{
		{
			ID:             "dense-only",
			MemoryID:       "a",
			RetrievalScore: 0.02,
			Retrievals: []RetrievalHit{{
				Route: "direct", Mode: searchModeSemantic, VectorSimilarity: 0.9,
			}},
		},
		{
			ID:             "supported",
			MemoryID:       "b",
			RetrievalScore: 0.03,
			Retrievals: []RetrievalHit{
				{Route: "direct", Mode: searchModeSemantic, VectorSimilarity: 0.75},
				{Route: "direct", Mode: searchModeLiteral, LiteralScore: 4},
				{Route: "parent", Mode: searchModeSemantic, FusionScore: 0.02},
			},
		},
	}
	ranked := rankFirstPhaseCandidates(candidates, nil)
	if ranked[0].ID != "supported" {
		t.Fatalf("expected complementary evidence first, got %#v", ranked)
	}
	features := ranked[0].FirstPhase
	if features == nil || features.LiteralScore != 4 || features.HierarchyScore != 0.02 || features.RouteSupport != 3 {
		t.Fatalf("first-phase features were not preserved: %#v", features)
	}
}

func TestRankFirstPhaseCandidatesUsesDocumentEvidence(t *testing.T) {
	candidates := []CandidateSource{
		{ID: "a", MemoryID: "a", RetrievalScore: 1, Retrievals: []RetrievalHit{{Route: "direct", Mode: searchModeSemantic, VectorSimilarity: 0.7}}},
		{ID: "b", MemoryID: "b", RetrievalScore: 1, Retrievals: []RetrievalHit{{Route: "direct", Mode: searchModeSemantic, VectorSimilarity: 0.7}}},
	}
	documents := []firstPhaseDocument{{memoryID: "b", score: 1}, {memoryID: "a", score: 0.5}}
	ranked := rankFirstPhaseCandidates(candidates, documents)
	if ranked[0].ID != "b" || ranked[0].FirstPhase.DocumentScore != 1 {
		t.Fatalf("expected stronger document evidence first, got %#v", ranked)
	}
}

func TestSelectFirstPhaseCandidatesBalancesDocumentsAndGlobalRank(t *testing.T) {
	candidates := make([]CandidateSource, 0, 24)
	for document := 0; document < 3; document++ {
		for chunk := 0; chunk < 8; chunk++ {
			candidates = append(candidates, CandidateSource{
				ID:             string(rune('a'+document)) + string(rune('0'+chunk)),
				MemoryID:       string(rune('a' + document)),
				RetrievalScore: 1 - float64(document*8+chunk)/100,
				ChunkMetadata:  map[string]any{},
			})
		}
	}

	selected := selectFirstPhaseCandidates(candidates, 12, retrievalPlanLookup)
	if len(selected) != 12 {
		t.Fatalf("expected 12 candidates, got %d", len(selected))
	}
	counts := map[string]int{}
	for _, candidate := range selected {
		counts[candidate.MemoryID]++
		if _, ok := candidate.ChunkMetadata["first_phase_document_rank"]; !ok {
			t.Fatalf("missing document rank on %#v", candidate)
		}
	}
	for _, memoryID := range []string{"a", "b", "c"} {
		if counts[memoryID] == 0 {
			t.Fatalf("expected document %q to be represented, got %#v", memoryID, counts)
		}
	}
}

func TestCrossEncoderOrderRemainsAuthoritativeThroughDiversity(t *testing.T) {
	reranked := []CandidateSource{
		{ID: "a1", MemoryID: "a"},
		{ID: "a2", MemoryID: "a"},
		{ID: "b1", MemoryID: "b"},
		{ID: "a3", MemoryID: "a"},
	}
	selected := diversifyRetrieveSources(reranked, 3, 2)
	want := []string{"a1", "a2", "b1"}
	for index, id := range want {
		if selected[index].ID != id {
			t.Fatalf("selected order = %#v, want %#v", selected, want)
		}
	}
}

func TestLookupPerMemoryLimitPreservesStrongSingleSourceEvidence(t *testing.T) {
	tests := []struct {
		sourceLimit int
		want        int
	}{
		{sourceLimit: 3, want: 3},
		{sourceLimit: 8, want: 6},
		{sourceLimit: 12, want: 9},
	}
	for _, test := range tests {
		if got := lookupPerMemoryLimit(test.sourceLimit); got != test.want {
			t.Fatalf("lookupPerMemoryLimit(%d) = %d, want %d", test.sourceLimit, got, test.want)
		}
	}
}

func TestRetrievalPlanRecognizesOverviewQuestions(t *testing.T) {
	if got := retrievalPlanForQuery("What is in chapter 1 of the probability textbook?"); got != retrievalPlanOverview {
		t.Fatalf("expected overview plan, got %q", got)
	}
	if got := retrievalPlanForQuery("What is Niko's mother's name?"); got != retrievalPlanLookup {
		t.Fatalf("expected lookup plan, got %q", got)
	}
}

func TestSelectEvidenceSetPrefersDistinctStructuralNodes(t *testing.T) {
	sources := []CandidateSource{
		{ID: "a1", MemoryID: "book", ChunkID: "1", ChunkMetadata: map[string]any{"route_node_id": "chapter-a"}},
		{ID: "a2", MemoryID: "book", ChunkID: "2", ChunkMetadata: map[string]any{"route_node_id": "chapter-a"}},
		{ID: "b1", MemoryID: "book", ChunkID: "3", ChunkMetadata: map[string]any{"route_node_id": "chapter-b"}},
	}
	selected := selectEvidenceSet(sources, 2, 2)
	if len(selected) != 2 || selected[0].ID != "a1" || selected[1].ID != "b1" {
		t.Fatalf("expected one representative per structural node, got %#v", selected)
	}
}

func TestCenteredEvidenceTextPreservesSelectedChunk(t *testing.T) {
	chunks := []retrieveNearbyChunk{
		{id: "before", label: "Chunk 1", text: strings.Repeat("before ", 200)},
		{id: "selected", label: "Chunk 2", text: "Talia Voss is Niko's mother."},
		{id: "after", label: "Chunk 3", text: strings.Repeat("after ", 200)},
	}
	evidence := centeredEvidenceText(chunks, 1, 300)
	if !strings.Contains(evidence, "Talia Voss is Niko's mother.") {
		t.Fatalf("selected evidence was lost: %q", evidence)
	}
	if len([]rune(evidence)) > 306 {
		t.Fatalf("evidence exceeded bounded context: %d", len([]rune(evidence)))
	}
}

func assertTraceStage(t *testing.T, trace *RetrieveStageTrace, name string, want []string) {
	t.Helper()
	for _, stage := range trace.Stages {
		if stage.Name != name {
			continue
		}
		if len(stage.CandidateIDs) != len(want) {
			t.Fatalf("%s = %#v, want %#v", name, stage.CandidateIDs, want)
		}
		for index := range want {
			if stage.CandidateIDs[index] != want[index] {
				t.Fatalf("%s = %#v, want %#v", name, stage.CandidateIDs, want)
			}
		}
		return
	}
	t.Fatalf("missing trace stage %q", name)
}
