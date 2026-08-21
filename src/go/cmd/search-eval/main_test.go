package main

import (
	"testing"
	"time"
)

func TestMatchesSource(t *testing.T) {
	result := searchResult{
		Title:     "2024.10.03 - Thursday.md",
		SourceURI: "file:///home/src/data/obsidian/diary/2024/10%20October/2024.10.03%20-%20Thursday.md",
	}
	expected := expectedSource{
		Title:        "2024.10.03 - Thursday.md",
		RelativePath: "diary/2024/10 October/2024.10.03 - Thursday.md",
	}
	if !matchesSource(result, expected) {
		t.Fatal("expected source to match")
	}
	expected.RelativePath = "diary/2024/10 October/other.md"
	if matchesSource(result, expected) {
		t.Fatal("unexpected source match")
	}
}

func TestValidateSuiteRejectsMissingEvidence(t *testing.T) {
	loaded := validSuite(evalCase{
		ID:         "case",
		Difficulty: "hard",
		Category:   "relationship",
		Query:      "question",
	})
	if err := validateSuite(loaded); err == nil {
		t.Fatal("expected missing evidence validation error")
	}
}

func TestValidateSuiteAcceptsUnanswerableCase(t *testing.T) {
	loaded := validSuite(evalCase{
		ID:         "case",
		Difficulty: "hard",
		Category:   "unanswerable",
		Query:      "question",
		Expected: expectedCase{
			Unanswerable: true,
			Answer:       expectedAnswer{AnyOf: []string{"not found"}},
		},
	})
	if err := validateSuite(loaded); err != nil {
		t.Fatal(err)
	}
}

func TestLoadEvaluationSuite(t *testing.T) {
	loaded, err := loadSuite("../../../../eval/search_eval.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Queries) != 31 {
		t.Fatalf("expected 31 cases, got %d", len(loaded.Queries))
	}
}

func TestEvaluateEvidenceAcrossChunks(t *testing.T) {
	source := expectedSource{RelativePath: "books/book.epub", Title: "Book"}
	item := evalCase{Expected: expectedCase{Evidence: []expectedEvidence{{
		Source: source,
		AllOf:  []string{"first fact", "second fact"},
	}}}}
	candidates := []searchResult{
		{Title: "Other", SourceURI: "file:///library/books/other.epub", Text: "first fact"},
		{Title: "Book", SourceURI: "file:///library/books/book.epub", Text: "first fact"},
		{Title: "Book", SourceURI: "file:///library/books/book.epub", Text: "second fact"},
	}
	result := evaluateResults(item, 50, candidates, time.Second)
	if result.DocumentRank != 2 || result.EvidenceRank != 3 {
		t.Fatalf("unexpected ranks: document=%d evidence=%d", result.DocumentRank, result.EvidenceRank)
	}
}

func TestEvaluateRequiresEveryEvidenceGroup(t *testing.T) {
	item := evalCase{Expected: expectedCase{Evidence: []expectedEvidence{
		{Source: expectedSource{RelativePath: "one.md", Title: "One"}, AnyOf: []string{"alpha"}},
		{Source: expectedSource{RelativePath: "two.md", Title: "Two"}, AnyOf: []string{"beta"}},
	}}}
	candidates := []searchResult{{Title: "One", SourceURI: "file:///notes/one.md", Text: "alpha"}}
	result := evaluateResults(item, 50, candidates, time.Second)
	if result.DocumentRank != 0 || result.EvidenceRank != 0 {
		t.Fatalf("partial evidence must not count as a hit: %#v", result)
	}
	if result.MatchedEvidenceGroups != 1 {
		t.Fatalf("expected one diagnostic group hit, got %d", result.MatchedEvidenceGroups)
	}
}

func TestRequestFiltersDoNotUseExpectedSource(t *testing.T) {
	item := evalCase{Expected: expectedCase{Evidence: []expectedEvidence{{
		Source: expectedSource{RelativePath: "book.epub", Title: "Book"},
		AnyOf:  []string{"fact"},
	}}}}
	defaults := suiteDefaults{Scope: "", Collection: "", Mode: "hybrid"}
	if requestScope(defaults, item) != "" || requestCollection(defaults, item) != "" {
		t.Fatal("expected source leaked into request filters")
	}
}

func TestMatchesExpectedAnswer(t *testing.T) {
	expected := expectedAnswer{
		AnyOf: []string{"swamp adder", "snake"},
		AllOf: []string{"milk", "whipcord"},
	}
	if !matchesExpectedAnswer("Milk and whipcord revealed a swamp adder.", expected) {
		t.Fatal("expected answer to match")
	}
	if matchesExpectedAnswer("A swamp adder was responsible.", expected) {
		t.Fatal("answer missing required details matched")
	}
}

func TestContainsFoldNormalizesExtractedWhitespace(t *testing.T) {
	value := "a training ground for future generals\nand\u00a0statesmen"
	if !containsFold(value, "a training ground for future generals and statesmen") {
		t.Fatal("expected wrapped extracted text to match normalized evidence")
	}
}

func TestSetStageRanksEvaluatesEveryOrderedStage(t *testing.T) {
	item := evalCase{
		ID: "case",
		Expected: expectedCase{Evidence: []expectedEvidence{{
			Source: expectedSource{RelativePath: "book.epub", Title: "Book"},
			AnyOf:  []string{"answer evidence"},
		}}},
	}
	response := retrieveResponse{StageTrace: &retrieveStageTrace{
		Candidates: []retrieveTraceCandidate{
			{ID: "wrong", Title: "Book", Text: "related passage", SourceURI: "file:///books/book.epub"},
			{ID: "answer", Title: "Book", Text: "answer evidence", SourceURI: "file:///books/book.epub"},
		},
		Stages: []retrieveTraceStage{
			{Name: "direct_hybrid", CandidateIDs: []string{"wrong", "answer"}},
			{Name: "final", CandidateIDs: []string{"answer"}},
		},
	}}
	result := evaluationResult{Case: item}
	result.setStageRanks(response)

	if len(result.StageRanks) != 2 {
		t.Fatalf("expected two stage ranks, got %#v", result.StageRanks)
	}
	if result.StageRanks[0].DocumentRank != 1 || result.StageRanks[0].EvidenceRank != 2 {
		t.Fatalf("unexpected direct ranks: %#v", result.StageRanks[0])
	}
	if result.StageRanks[1].EvidenceRank != 1 {
		t.Fatalf("unexpected final rank: %#v", result.StageRanks[1])
	}
}

func TestParseCandidateCaps(t *testing.T) {
	caps, err := parseCandidateCaps("50,75,100,150,75")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{50, 75, 100, 150}
	if len(caps) != len(want) {
		t.Fatalf("unexpected caps: %#v", caps)
	}
	for index := range want {
		if caps[index] != want[index] {
			t.Fatalf("unexpected caps: %#v", caps)
		}
	}
}

func TestParseFinalLimits(t *testing.T) {
	limits, err := parseFinalLimits("8,12,16,12", 8)
	if err != nil {
		t.Fatal(err)
	}
	expected := []int{8, 12, 16}
	if len(limits) != len(expected) {
		t.Fatalf("unexpected final limits: %#v", limits)
	}
	for index := range expected {
		if limits[index] != expected[index] {
			t.Fatalf("unexpected final limits: %#v", limits)
		}
	}
	if _, err := parseFinalLimits("51", 8); err == nil {
		t.Fatal("expected oversized final limit to fail")
	}
}

func TestLatencyAggregates(t *testing.T) {
	values := []int64{40, 10, 30, 20}
	if average := averageInt64(values); average != 25 {
		t.Fatalf("unexpected average: %d", average)
	}
	if median := percentileInt64(values, 0.5); median != 20 {
		t.Fatalf("unexpected median: %d", median)
	}
	if p95 := percentileInt64(values, 0.95); p95 != 40 {
		t.Fatalf("unexpected p95: %d", p95)
	}
}

func TestOrderOracleCandidatesUsesCompleteRerankerOrder(t *testing.T) {
	candidates := []searchResult{{ChunkID: "a"}, {ChunkID: "b"}, {ChunkID: "c"}}
	response := oracleRerankResponse{Results: []oracleRerankResult{{ID: "c"}, {ID: "a"}, {ID: "b"}}}
	ordered, err := orderOracleCandidates(candidates, response)
	if err != nil {
		t.Fatal(err)
	}
	if ordered[0].ChunkID != "c" || ordered[1].ChunkID != "a" || ordered[2].ChunkID != "b" {
		t.Fatalf("unexpected oracle order: %#v", ordered)
	}
	response.Results = response.Results[:2]
	if _, err := orderOracleCandidates(candidates, response); err == nil {
		t.Fatal("expected incomplete reranker response to fail")
	}
}

func TestSanitizeStageName(t *testing.T) {
	if got := sanitizeStageName("BAAI/bge-reranker-v2-m3"); got != "baai_bge_reranker_v2_m3" {
		t.Fatalf("stage name = %q", got)
	}
}

func TestAuditStatus(t *testing.T) {
	if auditStatus(nil) != "valid" {
		t.Fatal("nil audit error must be valid")
	}
	if auditStatus(assertionError{}) != "invalid" {
		t.Fatal("audit error must be invalid")
	}
}

type assertionError struct{}

func (assertionError) Error() string { return "invalid" }

func validSuite(item evalCase) suite {
	return suite{
		Version:  3,
		Defaults: suiteDefaults{UserKey: "local", ExpectedTopK: 8},
		Queries:  []evalCase{item},
	}
}
