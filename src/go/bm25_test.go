package main

import (
	"testing"
	"time"
)

func TestBM25SearchRanksDistinctiveTerms(t *testing.T) {
	index := NewBM25Index()
	index.mu.Lock()
	index.addLocked(newBM25TestDocument("memory-1", "chunk-1", "Mira Voss is Niko's mother."))
	index.addLocked(newBM25TestDocument("memory-2", "chunk-2", "Niko went to school with Daren."))
	index.addLocked(newBM25TestDocument("memory-3", "chunk-3", "The mother spoke to the students."))
	index.mu.Unlock()

	results := index.Search("local", SearchRequest{Query: "Mira mother"}, 3)
	if len(results) == 0 {
		t.Fatal("expected BM25 results")
	}
	if results[0].ChunkID != "chunk-1" {
		t.Fatalf("expected distinctive relationship chunk first, got %#v", results)
	}
}

func TestBM25SearchNodesFiltersAndLimitsEachNode(t *testing.T) {
	index := NewBM25Index()
	index.mu.Lock()
	first := newBM25TestDocument("book", "chapter-1-a", "A rare spellbook was received at school.")
	first.nodeID = "chapter-1"
	index.addLocked(first)
	second := newBM25TestDocument("book", "chapter-1-b", "Another school spellbook appears here.")
	second.nodeID = "chapter-1"
	index.addLocked(second)
	other := newBM25TestDocument("book", "chapter-2-a", "This spellbook belongs to another chapter.")
	other.nodeID = "chapter-2"
	index.addLocked(other)
	index.mu.Unlock()

	results := index.SearchNodes("local", SearchRequest{Query: "received spellbook school"}, []string{"chapter-1"}, 1)
	if len(results) != 1 || stringMapValue(results[0].ChunkMetadata, "node_id") != "chapter-1" {
		t.Fatalf("expected one result from the selected node, got %#v", results)
	}
}

func TestBM25SearchMemoriesFiltersAndLimitsEachDocument(t *testing.T) {
	index := NewBM25Index()
	index.mu.Lock()
	for _, document := range []*bm25Document{
		newBM25TestDocument("book-a", "a-answer", "Talia Voss is Niko's mother."),
		newBM25TestDocument("book-a", "a-related", "Niko remembers his mother."),
		newBM25TestDocument("book-b", "b-answer", "Mira Voss is Niko's mother."),
		newBM25TestDocument("book-c", "excluded", "Another mother named Aria appears."),
	} {
		index.addLocked(document)
	}
	index.mu.Unlock()

	results := index.SearchMemories(
		"local",
		SearchRequest{Query: "mother name"},
		[]string{"book-a", "book-b"},
		1,
	)
	if len(results) != 2 {
		t.Fatalf("expected one result from each selected document, got %#v", results)
	}
	seen := map[string]bool{}
	for _, result := range results {
		seen[result.MemoryID] = true
		if result.ChunkID == "excluded" {
			t.Fatal("result from an unselected document was returned")
		}
		if rank, ok := intMapValue(result.ChunkMetadata, "conditioned_rank"); !ok || rank != 1 {
			t.Fatalf("missing conditioned rank: %#v", result.ChunkMetadata)
		}
	}
	if !seen["book-a"] || !seen["book-b"] {
		t.Fatalf("selected documents were not both represented: %#v", seen)
	}

	index.RemoveMemory("book-a")
	results = index.SearchMemories("local", SearchRequest{Query: "mother name"}, []string{"book-a"}, 2)
	if len(results) != 0 {
		t.Fatalf("removed document remained in conditioned index: %#v", results)
	}
}

func newBM25TestDocument(memoryID, chunkID, text string) *bm25Document {
	result := SearchResult{
		MemoryID:  memoryID,
		ChunkID:   chunkID,
		Text:      text,
		UpdatedAt: time.Unix(0, 0),
	}
	terms := bm25TermFrequencies(bm25SearchText(result))
	return &bm25Document{userKey: "local", result: result, terms: terms, length: termCount(terms)}
}
