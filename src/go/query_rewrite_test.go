package main

import "testing"

func TestCleanQueryRewriteRemovesReasoningAndLabel(t *testing.T) {
	got := cleanQueryRewrite("<think>internal reasoning</think>\nQuery: contents of chapter one of Probability and Measure")
	want := "contents of chapter one of Probability and Measure"
	if got != want {
		t.Fatalf("cleanQueryRewrite() = %q, want %q", got, want)
	}
}

func TestValidQueryRewriteRequiresGrounding(t *testing.T) {
	titles := []string{"Billingsley - Probability and Measure.pdf"}
	if !validQueryRewrite(
		"what is in ch 1 of Billingsley probability textbook?",
		"contents of chapter one of Billingsley Probability and Measure",
		titles,
	) {
		t.Fatal("expected grounded rewrite to be accepted")
	}
	if validQueryRewrite("Billingsley chapter", "a completely unrelated answer", titles) {
		t.Fatal("expected unrelated rewrite to be rejected")
	}
}

func TestValidQueryRewriteRejectsLostIntentAndTitleLists(t *testing.T) {
	titles := []string{
		"Northwind Rising.pdf",
		"Northwind Falling.pdf",
		"A Different Series.pdf",
	}
	if validQueryRewrite(
		"What is Mira's mentor's name in the Northwind trilogy?",
		"Northwind Rising Northwind Falling A Different Series",
		titles,
	) {
		t.Fatal("expected a title list that loses the requested relationship to be rejected")
	}
	if !validQueryRewrite(
		"What is Mira's mentor's name in the Northwind trilogy?",
		"Mira mentor name Northwind trilogy Northwind Rising",
		titles,
	) {
		t.Fatal("expected an intent-preserving rewrite to be accepted")
	}
	if validQueryRewrite(
		"What is Mira's mentor's name in the Northwind trilogy?",
		"Northwind trilogy Mira mentor name",
		titles,
	) {
		t.Fatal("expected a rewrite that only removes stopwords to be rejected")
	}
}

func TestQueryRewriteTermsDropsGenericDocumentTerms(t *testing.T) {
	got := queryRewriteTerms("What is in chapter 1 of the Billingsley probability textbook?")
	want := []string{"billingsley", "probability"}
	if len(got) != len(want) {
		t.Fatalf("queryRewriteTerms() = %#v, want %#v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("queryRewriteTerms() = %#v, want %#v", got, want)
		}
	}
}

func TestMergeRetrieveCandidatesCombinesRetrievalEvidence(t *testing.T) {
	first := CandidateSource{ID: "same", RetrievalScore: 1, Retrievals: []RetrievalHit{{Query: "original"}}}
	second := CandidateSource{ID: "same", RetrievalScore: 2, Retrievals: []RetrievalHit{{Query: "rewrite"}}}
	merged := mergeRetrieveCandidates([]CandidateSource{first}, []CandidateSource{second})
	if len(merged) != 1 || merged[0].RetrievalScore != 3 || len(merged[0].Retrievals) != 2 {
		t.Fatalf("unexpected merged candidates: %#v", merged)
	}
}

func TestRetainHierarchyVectorChildren(t *testing.T) {
	selected := []SearchResult{{MemoryID: "memory", ChunkID: "reranked", ChunkMetadata: map[string]any{}}}
	candidates := []SearchResult{
		{MemoryID: "memory", ChunkID: "first", ChunkMetadata: map[string]any{"child_rank": 1}},
		{MemoryID: "memory", ChunkID: "third", ChunkMetadata: map[string]any{"child_rank": 3}},
		{MemoryID: "memory", ChunkID: "fourth", ChunkMetadata: map[string]any{"child_rank": 4}},
	}
	got := retainHierarchyVectorChildren(selected, candidates, 3)
	if len(got) != 3 || got[1].ChunkID != "first" || got[2].ChunkID != "third" {
		t.Fatalf("unexpected retained hierarchy children: %#v", got)
	}
}
