package main

import (
	"fmt"
	"testing"
)

func TestParseStructuralQueryIsSourceAgnostic(t *testing.T) {
	tests := []struct {
		query, nodeType, ordinal, role string
	}{
		{"What is a problem in ch 1 of a probability text?", "chapter", "1", "problem"},
		{"What does the first chapter of the probability text cover?", "chapter", "1", ""},
		{"What does chapter first of the probability text cover?", "chapter", "1", ""},
		{"Summarize section 2.3 of the operations manual", "section", "2.3", ""},
		{"Give an example from Appendix A", "appendix", "a", "example"},
		{"Which chapter of the manual covers setup?", "", "", ""},
	}
	for _, test := range tests {
		got := parseStructuralQuery(test.query)
		if got.NodeType != test.nodeType || got.Ordinal != test.ordinal || got.Role != test.role {
			t.Fatalf("parse %q = %#v", test.query, got)
		}
	}
}

func TestNormalizeStructureOrdinalMatchesWordsAndDigits(t *testing.T) {
	if normalizeStructureOrdinal("One") != "1" || normalizeStructureOrdinal("first") != "1" || normalizeStructureOrdinal("1") != "1" {
		t.Fatal("expected cardinal, ordinal, and digit to normalize equally")
	}
}

func TestAssembleGuardedRerankPoolReservesFallbackLanes(t *testing.T) {
	makeCandidates := func(prefix string, count int, route string) []CandidateSource {
		result := make([]CandidateSource, 0, count)
		for index := 0; index < count; index++ {
			id := fmt.Sprintf("%s-%02d", prefix, index)
			result = append(result, CandidateSource{
				ID: id, MemoryID: prefix, ChunkID: id, RetrievalScore: float64(count - index),
				Retrievals: []RetrievalHit{{Route: route, Mode: searchModeSemantic, Rank: index + 1}},
			})
		}
		return result
	}
	structured := makeCandidates("structured", 25, "structured_dense")
	conditioned := makeCandidates("conditioned", 15, "conditioned_dense")
	global := makeCandidates("global", 15, "direct")
	pool := assembleGuardedRerankPool(global, conditioned, structured, nil, 40, true)
	if len(pool) != 40 {
		t.Fatalf("expected bounded 40-candidate pool, got %d", len(pool))
	}
	counts := map[string]int{}
	for _, candidate := range pool {
		counts[candidate.MemoryID]++
	}
	if counts["structured"] != 20 || counts["conditioned"] != 10 || counts["global"] != 10 {
		t.Fatalf("expected 20/10/10 reservations, got %#v", counts)
	}
}

func TestSemanticRoleMatchesProblemAndExercise(t *testing.T) {
	if !semanticRoleMatches("exercise", "problem") || !semanticRoleMatches("problems", "exercise") {
		t.Fatal("expected problem and exercise roles to be compatible")
	}
	if semanticRoleMatches("theorem", "problem") {
		t.Fatal("unrelated structural roles must not match")
	}
}

func TestSemanticRoleAlternativesComposeProblemsAndExercises(t *testing.T) {
	got := semanticRoleAlternatives("problems")
	if len(got) != 2 || got[0] != "problem" || got[1] != "exercise" {
		t.Fatalf("unexpected compatible roles: %#v", got)
	}
}

func TestSelectStructuredDocumentsUsesGenericQueryIdentity(t *testing.T) {
	document := func(id, title, source string) firstPhaseDocument {
		return firstPhaseDocument{memoryID: id, candidates: []CandidateSource{{MemoryID: id, Title: title, SourceURI: source}}}
	}
	documents := []firstPhaseDocument{
		document("upfal", "Probability and Computing", "books/upfal/probability-and-computing.pdf"),
		document("billingsley", "Probability and Measure", "books/billingsley/probability-and-measure.pdf"),
		document("other", "Introduction to Algorithms", "books/cormen/introduction-to-algorithms.pdf"),
	}
	selected, scope := selectStructuredDocuments("What is a problem in ch 1 of Upfal's probability textbook?", documents)
	if scope != "query_identity" || len(selected) != 1 || selected[0].memoryID != "upfal" {
		t.Fatalf("unexpected document selection scope=%q documents=%#v", scope, selected)
	}
	selected, scope = selectStructuredDocuments("What is the first problem in ch 2?", documents)
	if scope != "ranked" || len(selected) != len(documents) {
		t.Fatalf("query without source identity should preserve ranked fallback, scope=%q count=%d", scope, len(selected))
	}
}

func TestSelectStructuredDocumentsCanRetainSeriesDocuments(t *testing.T) {
	documents := []firstPhaseDocument{
		{memoryID: "one", candidates: []CandidateSource{{Title: "A Deadly Education (The Scholomance)"}}},
		{memoryID: "two", candidates: []CandidateSource{{Title: "The Last Graduate (The Scholomance)"}}},
		{memoryID: "three", candidates: []CandidateSource{{Title: "The Golden Enclaves (The Scholomance)"}}},
	}
	selected, scope := selectStructuredDocuments("In the Scholomance trilogy, what happens next?", documents)
	if scope != "query_identity" || len(selected) != 3 {
		t.Fatalf("expected all explicitly named series documents, scope=%q count=%d", scope, len(selected))
	}
}
