package main

import "testing"

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
	loaded := suite{
		Version:  2,
		Defaults: suiteDefaults{UserKey: "local", ExpectedTopK: 5},
		Queries: []evalCase{{
			ID:    "case",
			Query: "question",
			Expected: expectedCase{Source: expectedSource{
				RelativePath: "note.md",
				Title:        "note.md",
			}},
		}},
	}
	if err := validateSuite(loaded); err == nil {
		t.Fatal("expected missing evidence validation error")
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
