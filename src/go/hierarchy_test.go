package main

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func TestBuildParentNodesGroupsLogicalSections(t *testing.T) {
	req := UpsertMemoryRequest{Title: "Example", SourceURI: "file:///example.epub"}
	chunks := []textChunk{
		{Text: "First", Metadata: map[string]any{"chapter": 1, "chunk_index": 0}},
		{Text: "Second", Metadata: map[string]any{"chapter": 1, "chunk_index": 1}},
		{Text: "Third", Metadata: map[string]any{"chapter": 2, "chunk_index": 2}},
	}
	parents, childParentIDs := buildParentNodes(req, chunks)
	if len(parents) != 3 {
		t.Fatalf("expected document root and two chapter nodes, got %d", len(parents))
	}
	if childParentIDs[0] != childParentIDs[1] || childParentIDs[1] == childParentIDs[2] {
		t.Fatalf("unexpected child parent IDs: %#v", childParentIDs)
	}
	if parents[0].NodeType != "document" || parents[0].Text != "First\n\nSecond\n\nThird" {
		t.Fatalf("unexpected document root: %#v", parents[0])
	}
	if parents[1].Text != "First\n\nSecond" || parents[1].ParentID != parents[0].ID {
		t.Fatalf("unexpected first chapter: %#v", parents[1])
	}
	if parents[1].Metadata["label"] != "Chapter 1" {
		t.Fatalf("unexpected first chapter metadata: %#v", parents[1].Metadata)
	}
}

func TestBuildParentNodesNestsSectionsInsideChapters(t *testing.T) {
	req := UpsertMemoryRequest{Title: "Example", SourceURI: "file:///example.epub"}
	chunks := []textChunk{
		{Text: "First", Metadata: map[string]any{"chapter": 1, "heading": "Opening"}},
		{Text: "Second", Metadata: map[string]any{"chapter": 1, "heading": "Opening"}},
	}
	nodes, chunkNodeIDs := buildParentNodes(req, chunks)
	if len(nodes) != 3 {
		t.Fatalf("expected document, chapter, and section nodes, got %#v", nodes)
	}
	if nodes[2].NodeType != "section" || nodes[2].ParentID != nodes[1].ID || nodes[2].Depth != 2 {
		t.Fatalf("unexpected section hierarchy: %#v", nodes[2])
	}
	if chunkNodeIDs[0] != nodes[2].ID || chunkNodeIDs[1] != nodes[2].ID {
		t.Fatalf("expected passages attached to the section node, got %#v", chunkNodeIDs)
	}
}

func TestRankedMentionedEntitiesKeepsNamesAndFiltersIncidentalWords(t *testing.T) {
	entities := rankedMentionedEntities("Yes, Mira Voss arrived. Daren waited. Mira Voss answered. Later Daren left.", 8)
	if !slices.Contains(entities, "Mira Voss") || !slices.Contains(entities, "Daren") {
		t.Fatalf("expected repeated and multi-word names, got %#v", entities)
	}
	if slices.Contains(entities, "Yes") || slices.Contains(entities, "Later") {
		t.Fatalf("expected incidental sentence starts to be filtered, got %#v", entities)
	}
}

func TestAssignParentEmbeddingsNormalizesMean(t *testing.T) {
	parents := []retrievalParentNode{{ChunkIndexes: []int{0, 1}}}
	assignParentEmbeddings(parents, [][]float64{{1, 0}, {0, 1}})
	if len(parents[0].Embedding) != 2 {
		t.Fatalf("unexpected parent embedding: %#v", parents[0].Embedding)
	}
	if math.Abs(parents[0].Embedding[0]-parents[0].Embedding[1]) > 0.0001 {
		t.Fatalf("expected equal pooled dimensions: %#v", parents[0].Embedding)
	}
	if math.Abs(parents[0].Embedding[0]*parents[0].Embedding[0]+parents[0].Embedding[1]*parents[0].Embedding[1]-1) > 0.0001 {
		t.Fatalf("expected normalized parent embedding: %#v", parents[0].Embedding)
	}
}

func TestFuseHierarchyChildrenCombinesVectorAndLexicalLanes(t *testing.T) {
	sharedVector := SearchResult{MemoryID: "book", ChunkID: "shared", ScoreParts: ScoreParts{VectorSimilarity: 0.8}, ChunkMetadata: map[string]any{}}
	sharedLexical := SearchResult{MemoryID: "book", ChunkID: "shared", ScoreParts: ScoreParts{LiteralScore: 3.2}, ChunkMetadata: map[string]any{"node_bm25_rank": 1}}
	vectorOnly := SearchResult{MemoryID: "book", ChunkID: "vector", ChunkMetadata: map[string]any{}}

	results := fuseHierarchyChildren([]SearchResult{sharedVector, vectorOnly}, []SearchResult{sharedLexical})
	if len(results) != 2 || results[0].ChunkID != "shared" {
		t.Fatalf("expected the result present in both lanes first, got %#v", results)
	}
	if results[0].ScoreParts.LiteralScore != 3.2 {
		t.Fatalf("expected lexical diagnostics to survive fusion, got %#v", results[0])
	}
}

func TestSelectRoutedParentsReservesSummaryWithoutDisplacingPooledLane(t *testing.T) {
	pooled := []parentHit{
		{ID: "p1", FusionScore: 1},
		{ID: "p2", FusionScore: 0.9},
		{ID: "p3", FusionScore: 0.8},
		{ID: "p4", FusionScore: 0.7},
	}
	summaries := []parentHit{
		{ID: "p1", FusionScore: 1},
		{ID: "s1", FusionScore: 0.6},
		{ID: "s2", FusionScore: 0.5},
	}
	fused := append(append([]parentHit{}, pooled...), summaries[1:]...)

	selected := selectRoutedParents(pooled, summaries, fused, 4)
	ids := make(map[string]bool, len(selected))
	for _, parent := range selected {
		ids[parent.ID] = true
	}
	for _, id := range []string{"p1", "p2", "p3", "s1"} {
		if !ids[id] {
			t.Fatalf("expected %q in routed parents, got %#v", id, selected)
		}
	}
}

func TestTruncateRunesPreservesUnicodeBoundaries(t *testing.T) {
	got := truncateRunes("alpha 世界 omega", 8)
	if got != "alpha 世界" || strings.ContainsRune(got, '\ufffd') {
		t.Fatalf("unexpected rune-safe truncation %q", got)
	}
}
