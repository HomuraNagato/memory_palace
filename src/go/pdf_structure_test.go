package main

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestExtractPDFDocumentIntegration(t *testing.T) {
	path := os.Getenv("MEMORY_PALACE_TEST_PDF")
	if path == "" {
		t.Skip("MEMORY_PALACE_TEST_PDF is not set")
	}
	body, metadata, structure, err := extractPDFDocument(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if structure == nil || metadata["structure_extractor"] != "pdftohtml-xml" {
		t.Fatalf("expected structured extraction, got metadata=%#v structure=%#v", metadata, structure)
	}
	chunks := chunksForRequest(UpsertMemoryRequest{Body: body, Metadata: map[string]any{"file_ext": ".pdf"}, Structure: structure})
	t.Logf("structured chunks: %d", len(chunks))
	foundChapter := false
	foundExercise := false
	var chapterMetadata map[string]any
	var exerciseMetadata map[string]any
	for _, chunk := range chunks {
		normalizedText := strings.Join(strings.Fields(chunk.Text), " ")
		if strings.Contains(normalizedText, "This chapter introduces the notion of randomized algorithms") {
			chapterMetadata = chunk.Metadata
			path := structuredNodePath(chunk.Metadata["structure_path"])
			if len(path) > 0 && path[0].metadata["ordinal"] == "1" {
				foundChapter = true
			}
		}
		if strings.Contains(normalizedText, "We flip a fair coin ten times") {
			exerciseMetadata = chunk.Metadata
			if semanticRoleMatches(stringMapValue(chunk.Metadata, "semantic_role"), "problem") {
				foundExercise = true
			}
		}
	}
	if !foundChapter {
		t.Fatalf("expected known passage to be assigned to chapter 1; metadata=%#v", chapterMetadata)
	}
	if !foundExercise {
		t.Fatalf("expected a known exercise to receive a generic exercise/problem role; metadata=%#v", exerciseMetadata)
	}
}

func TestPDFStructuredChunkCountIntegration(t *testing.T) {
	path := os.Getenv("MEMORY_PALACE_TEST_PDF_COUNT")
	if path == "" {
		t.Skip("MEMORY_PALACE_TEST_PDF_COUNT is not set")
	}
	body, _, structure, err := extractPDFDocument(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	structured := chunksForRequest(UpsertMemoryRequest{Body: body, Metadata: map[string]any{"file_ext": ".pdf"}, Structure: structure})
	fallback := chunkSectionedTextWithMetadata(body, "page")
	chapters := map[string]bool{}
	for _, chunk := range structured {
		for _, node := range structuredNodePath(chunk.Metadata["structure_path"]) {
			if node.nodeType == "chapter" || node.nodeType == "appendix" {
				chapters[node.nodeKey] = true
			}
		}
	}
	t.Logf("structured chunks=%d fallback chunks=%d chapter nodes=%d", len(structured), len(fallback), len(chapters))
	if len(structured) > maxInt(100, len(fallback)*3) {
		t.Fatalf("pathological structured expansion: structured=%d fallback=%d", len(structured), len(fallback))
	}
}

func TestChunksForRequestUsesCanonicalTextWithFragmentedStructure(t *testing.T) {
	blocks := make([]documentBlock, 0, 101)
	for index := 0; index < 101; index++ {
		blocks = append(blocks, documentBlock{Top: float64(index * 20), FontSize: 10, Text: "A separate fragment."})
	}
	req := UpsertMemoryRequest{
		Title: "Fragmented.pdf", Body: "Canonical fallback text.",
		Metadata:  map[string]any{"file_ext": ".pdf"},
		Structure: &documentStructure{Extractor: "fixture", Pages: []documentPage{{Number: 1, Blocks: blocks}}},
	}
	chunks := chunksForRequest(req)
	if len(chunks) != 1 || chunks[0].Text != "Canonical fallback text." {
		t.Fatalf("expected canonical page fallback, got %d chunks: %#v", len(chunks), chunks)
	}
}

func TestParsePDFStructureBuildsGenericHierarchy(t *testing.T) {
	raw := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<pdf2xml>
  <page number="1" width="600" height="800">
    <fontspec id="0" size="10" family="Body"/>
    <fontspec id="1" size="22" family="Heading"/>
    <fontspec id="2" size="15" family="Heading"/>
    <text top="20" left="60" width="200" height="12" font="0">Repeated Manual Header</text>
    <text top="100" left="60" width="300" height="26" font="1"><b>Chapter 1 Foundations</b></text>
    <text top="150" left="60" width="250" height="18" font="2"><b>1.1 Sample Spaces</b></text>
    <text top="190" left="60" width="450" height="12" font="0">A sample space contains every possible outcome.</text>
    <text top="730" left="60" width="200" height="12" font="0">Repeated Manual Footer</text>
  </page>
  <page number="2" width="600" height="800">
    <text top="20" left="60" width="200" height="12" font="0">Repeated Manual Header</text>
    <text top="100" left="60" width="250" height="18" font="2"><b>Exercises</b></text>
    <text top="140" left="60" width="450" height="12" font="0">We flip a fair coin ten times. Find the probability.</text>
    <text top="730" left="60" width="200" height="12" font="0">Repeated Manual Footer</text>
  </page>
  <page number="3" width="600" height="800">
    <text top="20" left="60" width="200" height="12" font="0">Repeated Manual Header</text>
    <text top="100" left="60" width="300" height="26" font="1"><b>Chapter 2 Variables</b></text>
    <text top="150" left="60" width="450" height="12" font="0">A random variable maps outcomes to values.</text>
    <text top="730" left="60" width="200" height="12" font="0">Repeated Manual Footer</text>
  </page>
</pdf2xml>`)

	structure, err := parsePDFStructureXML(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, page := range structure.Pages {
		for _, block := range page.Blocks {
			if strings.Contains(block.Text, "Repeated Manual") {
				t.Fatalf("expected repeated margin text removed, got %#v", block)
			}
		}
	}

	chunks := chunksForDocumentStructure(structure)
	if len(chunks) != 3 {
		t.Fatalf("expected three body chunks, got %d: %#v", len(chunks), chunks)
	}
	path := structuredNodePath(chunks[0].Metadata["structure_path"])
	if len(path) != 2 || path[0].nodeType != "chapter" || path[1].nodeType != "section" {
		t.Fatalf("unexpected hierarchy path: %#v", path)
	}
	if chunks[1].Metadata["semantic_role"] != "exercise" {
		t.Fatalf("expected generic exercise role, got %#v", chunks[1].Metadata)
	}
	if chunks[2].Metadata["heading"] != "Chapter 2 Variables" {
		t.Fatalf("expected second chapter boundary, got %#v", chunks[2].Metadata)
	}
}

func TestDocumentStructureFallsBackToPagesWithoutHeadings(t *testing.T) {
	structure := documentStructure{
		Extractor: "fixture",
		Pages:     []documentPage{{Number: 4, Height: 800, Blocks: []documentBlock{{Top: 100, FontSize: 10, Text: "Ordinary prose without a heading."}}}},
	}
	chunks := chunksForDocumentStructure(structure)
	if len(chunks) != 1 || chunks[0].Metadata["page"] != 4 {
		t.Fatalf("expected page metadata fallback, got %#v", chunks)
	}
	if _, ok := chunks[0].Metadata["structure_path"]; ok {
		t.Fatalf("did not expect invented structure: %#v", chunks[0].Metadata)
	}
	path := nodePathForChunk(chunks[0])
	if len(path) != 1 || path[0].nodeType != "page" || path[0].nodeKey != "4" {
		t.Fatalf("expected page hierarchy fallback, got %#v", path)
	}
}

func TestTOCLeaderIsNotClassifiedAsHeading(t *testing.T) {
	block := documentBlock{Text: "1.2 Conditional Probability ........ 37", FontSize: 16, Bold: true}
	if _, ok := classifyPDFHeading(block, 10, 3); ok {
		t.Fatal("table-of-contents leader must not create a structural boundary")
	}
}

func TestParsePDFStructureRecoversMalformedInlineMarkup(t *testing.T) {
	raw := []byte(`<pdf2xml><page number="1" width="600" height="800"><fontspec id="0" size="12" family="Body"/><text top="100" left="50" width="400" height="14" font="0"><i>Probability text</b></text></page></pdf2xml>`)
	structure, err := parsePDFStructureXML(raw)
	if err != nil {
		t.Fatal(err)
	}
	if structure.Extractor != "pdftohtml-xml-sanitized" || len(structure.Pages) != 1 || structure.Pages[0].Blocks[0].Text != "Probability text" {
		t.Fatalf("unexpected sanitized structure: %#v", structure)
	}
}

func TestChapterLabelAndDisplayTitleFormOneNode(t *testing.T) {
	structure := documentStructure{Extractor: "fixture", Pages: []documentPage{{
		Number: 23, Height: 1080, Blocks: []documentBlock{
			{Top: 86, FontSize: 25, Bold: true, Text: "chapter one"},
			{Top: 126, FontSize: 33, Bold: true, Text: "Events and Probability"},
			{Top: 384, FontSize: 16, Text: "This chapter introduces randomized algorithms and probability theory."},
		},
	}}}
	chunks := chunksForDocumentStructure(structure)
	if len(chunks) != 1 {
		t.Fatalf("expected one passage, got %#v", chunks)
	}
	path := structuredNodePath(chunks[0].Metadata["structure_path"])
	if len(path) != 1 || path[0].metadata["ordinal"] != "1" || path[0].label != "chapter one: Events and Probability" {
		t.Fatalf("expected one chapter node with label and title, got %#v", path)
	}
}

func TestStructureAnnotationCanAdvancePastUnmatchedHeading(t *testing.T) {
	structure := documentStructure{Extractor: "fixture", Pages: []documentPage{{
		Number: 1, Height: 800, Blocks: []documentBlock{
			{Top: 80, FontSize: 16, Bold: true, Text: "1.1 Introduction"},
			{Top: 500, FontSize: 16, Bold: true, Text: "Exercises"},
		},
	}}}
	chunks := []textChunk{{Text: "Exercises\n\nSolve the following problem.", Metadata: map[string]any{"page": 1}}}
	annotated := applyDocumentStructureToChunks(chunks, structure)
	if annotated[0].Metadata["semantic_role"] != "exercise" {
		t.Fatalf("expected later matched role to advance prior structural events, got %#v", annotated[0].Metadata)
	}
}
