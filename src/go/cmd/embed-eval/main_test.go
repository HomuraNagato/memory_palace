package main

import (
	"math"
	"testing"
)

func TestParseModelSpecs(t *testing.T) {
	specs, err := parseModelSpecs("current|http://127.0.0.1:5021/v1/embeddings|Qwen3-Embedding-0.6B||,openai|https://api.openai.com/v1/embeddings|text-embedding-3-large|1024|OPENAI_API_KEY")
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 2 {
		t.Fatalf("expected 2 specs, got %d", len(specs))
	}
	if specs[0].Name != "current" || specs[0].Dimensions != 0 || specs[0].KeyEnv != "" {
		t.Fatalf("unexpected first spec: %#v", specs[0])
	}
	if specs[1].Name != "openai" || specs[1].Dimensions != 1024 || specs[1].KeyEnv != "OPENAI_API_KEY" {
		t.Fatalf("unexpected second spec: %#v", specs[1])
	}
}

func TestCosineSimilarity(t *testing.T) {
	got := cosineSimilarity([]float64{1, 0}, []float64{0, 1})
	if math.Abs(got) > 1e-12 {
		t.Fatalf("expected orthogonal vectors to be 0, got %f", got)
	}
	got = cosineSimilarity([]float64{1, 1}, []float64{1, 1})
	if math.Abs(got-1) > 1e-12 {
		t.Fatalf("expected same vectors to be 1, got %f", got)
	}
}

func TestSafeIdent(t *testing.T) {
	got, err := safeIdent("memory_chunk_embeddings_3072")
	if err != nil {
		t.Fatal(err)
	}
	if got != "memory_chunk_embeddings_3072" {
		t.Fatalf("unexpected identifier %q", got)
	}
	if _, err := safeIdent("memory-chunk-embeddings"); err == nil {
		t.Fatal("expected invalid identifier to fail")
	}
}

func TestSafeVectorType(t *testing.T) {
	got, err := safeVectorType("HALFVEC")
	if err != nil {
		t.Fatal(err)
	}
	if got != "halfvec" {
		t.Fatalf("unexpected vector type %q", got)
	}
	if _, err := safeVectorType("text"); err == nil {
		t.Fatal("expected invalid vector type to fail")
	}
}

func TestVectorLiteral(t *testing.T) {
	got := vectorLiteral([]float64{1, -0.5, 0.25})
	if got != "[1,-0.5,0.25]" {
		t.Fatalf("unexpected vector literal %q", got)
	}
}

func TestTargetMatches(t *testing.T) {
	chunk := chunkRow{Title: "starbucks.md", Text: "tall pistachio creme frappuccino with soy milk"}
	if !targetMatches(chunk, "", "STARBUCKS.MD") {
		t.Fatal("expected title match")
	}
	if !targetMatches(chunk, "pistachio", "") {
		t.Fatal("expected text match")
	}
	if targetMatches(chunk, "", "korean greetings.md") {
		t.Fatal("unexpected title match")
	}
}
