package main

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestRetrievalSpecsForHybridIncludesExpandedSemanticQueries(t *testing.T) {
	specs := retrievalSpecsFor("Niko's mother's name?", searchModeHybrid)

	if len(specs) != 6 {
		t.Fatalf("expected 6 specs, got %#v", specs)
	}
	if specs[0] != (RetrievalSpec{Query: "Niko's mother's name?", Mode: searchModeHybrid}) {
		t.Fatalf("unexpected preferred spec: %#v", specs[0])
	}
	if specs[1] != (RetrievalSpec{Query: "niko mother", Mode: searchModeHybrid}) {
		t.Fatalf("unexpected compact expansion: %#v", specs[1])
	}
	if specs[2] != (RetrievalSpec{Query: "mother of niko name", Mode: searchModeHybrid}) {
		t.Fatalf("unexpected relationship expansion: %#v", specs[2])
	}
	if specs[3] != (RetrievalSpec{Query: "mother mum mom of niko name", Mode: searchModeHybrid}) {
		t.Fatalf("unexpected relationship vocabulary expansion: %#v", specs[3])
	}
	if specs[4] != (RetrievalSpec{Query: "Niko mother name", Mode: searchModeLiteral}) {
		t.Fatalf("unexpected literal spec: %#v", specs[4])
	}
	if specs[5] != (RetrievalSpec{Query: "mother mum mom niko name", Mode: searchModeLiteral}) {
		t.Fatalf("unexpected literal relationship vocabulary spec: %#v", specs[5])
	}
}

func TestRetrievalSpecsForLiteralUsesSingleNormalizedQuery(t *testing.T) {
	specs := retrievalSpecsFor("Niko's mother's name?", searchModeLiteral)

	if len(specs) != 1 {
		t.Fatalf("expected 1 spec, got %#v", specs)
	}
	if specs[0] != (RetrievalSpec{Query: "Niko mother name", Mode: searchModeLiteral}) {
		t.Fatalf("unexpected literal spec: %#v", specs[0])
	}
}

func TestRescueLiteralRetrieveSourcesKeepsStrongLiteralHit(t *testing.T) {
	selected := []CandidateSource{
		{ID: "semantic-1"},
		{ID: "semantic-2"},
		{ID: "semantic-3"},
	}
	candidates := []CandidateSource{
		{
			ID:   "literal-answer",
			Text: "This chunk contains the answer name directly.",
			Retrievals: []RetrievalHit{
				{LiteralScore: 0.72},
			},
		},
	}

	result := rescueLiteralRetrieveSources("answer name", selected, candidates, 3, 1, 0.65)

	if len(result) != 3 {
		t.Fatalf("expected result to stay at limit, got %d", len(result))
	}
	if result[2].ID != "literal-answer" {
		t.Fatalf("expected weak tail source to be replaced by literal hit, got %#v", result)
	}
}

func TestRescueLiteralRetrieveSourcesIgnoresWeakLiteralHit(t *testing.T) {
	selected := []CandidateSource{
		{ID: "semantic-1"},
		{ID: "semantic-2"},
	}
	candidates := []CandidateSource{
		{
			ID: "weak-literal",
			Retrievals: []RetrievalHit{
				{LiteralScore: 0.40},
			},
		},
	}

	result := rescueLiteralRetrieveSources("answer name", selected, candidates, 2, 1, 0.65)

	if result[1].ID != "semantic-2" {
		t.Fatalf("expected weak literal hit to be ignored, got %#v", result)
	}
}

func TestRescueLiteralRetrieveSourcesKeepsMultipleHits(t *testing.T) {
	selected := []CandidateSource{{ID: "semantic-1"}, {ID: "semantic-2"}, {ID: "semantic-3"}, {ID: "semantic-4"}}
	candidates := []CandidateSource{
		{ID: "name", Text: "Mum's name is recorded here.", Retrievals: []RetrievalHit{{LiteralScore: 2}}},
		{ID: "relation", Text: "Her mother appears here.", Retrievals: []RetrievalHit{{LiteralScore: 1}}},
	}

	result := rescueLiteralRetrieveSources("What is her mother's name?", selected, candidates, 4, 2, 0.65)
	if !containsCandidate(result, "name") || !containsCandidate(result, "relation") {
		t.Fatalf("expected both rescue passages to survive, got %#v", result)
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

func TestAddRetrieveCandidatesUsesParentAndChildRanks(t *testing.T) {
	candidates := map[string]*CandidateSource{}
	result := SearchResult{
		MemoryID: "memory",
		ChunkID:  "chunk",
		Title:    "Title",
		Text:     "Evidence",
		ChunkMetadata: map[string]any{
			"parent_rank":         13,
			"child_rank":          5,
			"parent_similarity":   0.5,
			"parent_fusion_score": reciprocalRankScore(20) + reciprocalRankScore(21),
		},
	}
	spec := RetrievalSpec{Query: "question", Mode: searchModeSemantic}
	addRetrieveCandidates(candidates, []SearchResult{result}, spec, "parent")

	got := candidates["memory:chunk"]
	want := 0.5*(reciprocalRankScore(20)+reciprocalRankScore(21)) + 0.5*reciprocalRankScore(5)
	if got == nil || math.Abs(got.RetrievalScore-want) > 0.0000001 {
		t.Fatalf("expected hierarchical score %f, got %#v", want, got)
	}
	hit := got.Retrievals[0]
	if hit.ParentRank != 13 || hit.ChildRank != 5 || hit.ParentSimilarity != 0.5 {
		t.Fatalf("expected hierarchy diagnostics, got %#v", hit)
	}
}

func TestHierarchyRerankOverridesParentChildFusion(t *testing.T) {
	result := SearchResult{ChunkMetadata: map[string]any{
		"hierarchy_rerank_rank": 2,
		"parent_fusion_score":   reciprocalRankScore(1),
		"child_rank":            8,
	}}

	if got, want := hierarchicalFusionScore(result), reciprocalRankScore(2); math.Abs(got-want) > 0.0000001 {
		t.Fatalf("expected hierarchy rerank score %f, got %f", want, got)
	}
}

func TestFilterQueryEchoSourcesRemovesQuestionOnlyDocument(t *testing.T) {
	sources := []CandidateSource{
		{ID: "echo", Text: "Who designed the lunar observatory?"},
		{ID: "answer", Text: "Mira Patel designed the lunar observatory."},
	}

	filtered := filterQueryEchoSources("who designed the lunar observatory", sources)
	if len(filtered) != 1 || filtered[0].ID != "answer" {
		t.Fatalf("expected only answer evidence, got %#v", filtered)
	}
}

func TestRerankTextPlacesOriginalEvidenceFirst(t *testing.T) {
	source := CandidateSource{
		Text:          "answer evidence",
		ContextText:   "context",
		ParentSummary: "summary",
		MetadataText:  "metadata",
	}

	text := source.rerankText()
	if !strings.HasPrefix(text, "Original chunk with local context:\nanswer evidence") {
		t.Fatalf("expected original evidence first, got %q", text)
	}
}

func TestRerankTextPrefersLocalWindow(t *testing.T) {
	source := CandidateSource{Text: "isolated sentence", RerankContext: "neighbor context\nanswer sentence"}
	text := source.rerankText()
	if !strings.Contains(text, "neighbor context\nanswer sentence") || strings.Contains(text, "isolated sentence") {
		t.Fatalf("expected reranker-local context, got %q", text)
	}
}

func TestFuseRetrieveAndRerankOrdersRewardsAgreement(t *testing.T) {
	retrieved := []CandidateSource{{ID: "evidence"}, {ID: "retrieval-only"}, {ID: "shared"}}
	reranked := []CandidateSource{{ID: "shared"}, {ID: "rerank-only"}, {ID: "evidence"}}

	result := fuseRetrieveAndRerankOrders(retrieved, reranked)
	if len(result) != 4 || !((result[0].ID == "evidence" && result[1].ID == "shared") || (result[0].ID == "shared" && result[1].ID == "evidence")) {
		t.Fatalf("expected candidates supported by both orders first, got %#v", result)
	}
}

func TestFuseRetrieveAndRerankOrdersLetsRerankerCorrectRetrieval(t *testing.T) {
	retrieved := make([]CandidateSource, 50)
	reranked := make([]CandidateSource, 50)
	for index := range retrieved {
		retrieved[index] = CandidateSource{ID: fmt.Sprintf("candidate-%d", index)}
		reranked[len(reranked)-1-index] = retrieved[index]
	}

	result := fuseRetrieveAndRerankOrders(retrieved, reranked)
	if result[0].ID != "candidate-49" {
		t.Fatalf("expected reranker to correct retrieval order, got %q", result[0].ID)
	}
}

func TestSelectRerankCandidatesCapsAndDiversifiesChapters(t *testing.T) {
	candidates := make([]CandidateSource, 0, 12)
	for index := 0; index < 8; index++ {
		candidates = append(candidates, CandidateSource{
			ID:             fmt.Sprintf("same-%d", index),
			MemoryID:       "book",
			RetrievalScore: 1 - float64(index)/100,
			ChunkMetadata:  map[string]any{"chapter": 1},
		})
	}
	for index := 0; index < 4; index++ {
		candidates = append(candidates, CandidateSource{
			ID:             fmt.Sprintf("other-%d", index),
			MemoryID:       "other-book",
			RetrievalScore: 0.5 - float64(index)/100,
			ChunkMetadata:  map[string]any{"chapter": index + 1},
		})
	}

	selected := selectRerankCandidates(candidates, 8)
	if len(selected) != 8 {
		t.Fatalf("expected 8 candidates, got %d", len(selected))
	}
	sameChapter := 0
	for _, candidate := range selected {
		if candidate.MemoryID == "book" {
			sameChapter++
		}
	}
	if sameChapter > maxChunksPerChapter {
		t.Fatalf("expected at most %d same-chapter candidates, got %d", maxChunksPerChapter, sameChapter)
	}
}

func TestDeterministicQueryExpansionsIncludesRelationshipVocabulary(t *testing.T) {
	expansions := deterministicQueryExpansions("What is El's mother's name?")
	joined := strings.Join(expansions, "\n")
	if !strings.Contains(joined, "mother mum mom of el name") {
		t.Fatalf("expected general mother vocabulary expansion, got %q", expansions)
	}
}

func TestHierarchicalFusionScoreUsesLexicalChildRank(t *testing.T) {
	result := SearchResult{ChunkMetadata: map[string]any{
		"parent_fusion_score": 0.02,
		"parent_bm25_rank":    2,
	}}
	if score := hierarchicalFusionScore(result); score <= 0.01 {
		t.Fatalf("expected lexical child rank to contribute to hierarchy score, got %f", score)
	}
}

func TestSelectRerankCandidatesBalancesSourceLiteralLane(t *testing.T) {
	candidates := make([]CandidateSource, 0, 60)
	for source := 0; source < 3; source++ {
		for rank := 0; rank < 20; rank++ {
			candidates = append(candidates, CandidateSource{
				ID:             fmt.Sprintf("source-%d-%d", source, rank),
				MemoryID:       fmt.Sprintf("source-%d", source),
				RetrievalScore: 1 - float64(source*20+rank)/1000,
				Retrievals: []RetrievalHit{{
					Route:        "source_literal",
					LiteralScore: float64(100 - rank),
				}},
			})
		}
	}

	selected := selectRerankCandidates(candidates, 40)
	counts := map[string]int{}
	for _, candidate := range selected {
		counts[candidate.MemoryID]++
	}
	for source := 0; source < 3; source++ {
		if count := counts[fmt.Sprintf("source-%d", source)]; count < 9 {
			t.Fatalf("expected deep evidence from every identified source, got counts %#v", counts)
		}
	}
}

func TestCenteredEvidenceTextPreservesSelectedChunk(t *testing.T) {
	chunks := []retrieveNearbyChunk{
		{id: "before", label: "Chunk 1", text: strings.Repeat("before ", 200)},
		{id: "selected", label: "Chunk 2", text: "Gwen Higgins is El's mother."},
		{id: "after", label: "Chunk 3", text: strings.Repeat("after ", 200)},
	}

	evidence := centeredEvidenceText(chunks, 1, 300)
	if !strings.Contains(evidence, "Gwen Higgins is El's mother.") {
		t.Fatalf("selected evidence was lost: %q", evidence)
	}
	if len([]rune(evidence)) > 306 {
		t.Fatalf("evidence exceeded bounded context: %d", len([]rune(evidence)))
	}
}
