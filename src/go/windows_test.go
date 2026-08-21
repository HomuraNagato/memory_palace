package main

import (
	"strings"
	"testing"
)

func TestBuildPassageWindowsOverlapsWithinNodeBoundaries(t *testing.T) {
	chunks := []textChunk{
		{Text: strings.Repeat("a", 600)},
		{Text: strings.Repeat("b", 600)},
		{Text: strings.Repeat("c", 600)},
		{Text: strings.Repeat("d", 600)},
		{Text: strings.Repeat("e", 600)},
		{Text: strings.Repeat("f", 600)},
	}
	windows := buildPassageWindows(chunks, []string{"chapter-1", "chapter-1", "chapter-1", "chapter-1", "chapter-2", "chapter-2"})
	if len(windows) != 3 {
		t.Fatalf("expected three windows, got %#v", windows)
	}
	if windows[0].StartChunkIndex != 0 || windows[0].EndChunkIndex != 2 {
		t.Fatalf("unexpected first window: %#v", windows[0])
	}
	if windows[1].StartChunkIndex != 2 || windows[1].EndChunkIndex != 3 {
		t.Fatalf("expected an overlapping second window, got %#v", windows[1])
	}
	if windows[2].StartChunkIndex != 4 || windows[2].EndChunkIndex != 5 {
		t.Fatalf("window crossed a node boundary: %#v", windows[2])
	}
}

func TestBuildPassageWindowsSkipsSingleLeafNodes(t *testing.T) {
	chunks := []textChunk{{Text: "first"}, {Text: "second"}}
	if windows := buildPassageWindows(chunks, []string{"one", "two"}); len(windows) != 0 {
		t.Fatalf("expected no single-leaf windows, got %#v", windows)
	}
}

func TestCenterTextAroundPassageKeepsCanonicalLeaf(t *testing.T) {
	passage := "Gwen Higgins is your mother"
	window := strings.Repeat("before ", 100) + passage + strings.Repeat(" after", 100)
	centered := centerTextAroundPassage(window, passage, 240)
	if len([]rune(centered)) > 240 {
		t.Fatalf("centered window exceeded limit: %d", len([]rune(centered)))
	}
	if !strings.Contains(centered, passage) {
		t.Fatalf("centered window lost the canonical passage: %q", centered)
	}
}
