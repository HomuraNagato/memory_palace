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

func TestBM25SearchParentsFiltersAndLimitsEachParent(t *testing.T) {
	index := NewBM25Index()
	index.mu.Lock()
	first := newBM25TestDocument("book", "chapter-1-a", "A rare spellbook was received at school.")
	first.parentID = "chapter-1"
	index.addLocked(first)
	second := newBM25TestDocument("book", "chapter-1-b", "Another school spellbook appears here.")
	second.parentID = "chapter-1"
	index.addLocked(second)
	other := newBM25TestDocument("book", "chapter-2-a", "This spellbook belongs to another chapter.")
	other.parentID = "chapter-2"
	index.addLocked(other)
	index.mu.Unlock()

	results := index.SearchParents("local", SearchRequest{Query: "received spellbook school"}, []string{"chapter-1"}, 1)
	if len(results) != 1 || stringMapValue(results[0].ChunkMetadata, "parent_id") != "chapter-1" {
		t.Fatalf("expected one result from the selected parent, got %#v", results)
	}
}

func TestBM25SearchRestrictsIdentifiedSourceTitles(t *testing.T) {
	index := NewBM25Index()
	index.mu.Lock()
	target := newBM25TestDocument("book-1", "target", "Her mum is Gwen Higgins.")
	target.result.Title = "The Scholomance"
	index.addLocked(target)
	other := newBM25TestDocument("book-2", "other", "Another mum appears here.")
	other.result.Title = "Another Book"
	index.addLocked(other)
	index.mu.Unlock()

	results := index.Search("local", SearchRequest{Query: "mum", SourceTitles: []string{"The Scholomance"}}, 10)
	if len(results) != 1 || results[0].ChunkID != "target" {
		t.Fatalf("expected only the identified source result, got %#v", results)
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
