package main

import (
	"archive/zip"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCoreContextRequiresTenant(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/v1/core-context", nil)
	recorder := httptest.NewRecorder()
	(&Server{}).handleCoreContext(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func TestFixedInstanceUserKeyOverridesRequestIdentity(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://memory.test/v1/memories?user_key=query-user", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Memory-User-Key", "header-user")

	if got := (&Server{instanceUserKey: "kimichuwu"}).userKeyFromRequest(request); got != "kimichuwu" {
		t.Fatalf("fixed instance user = %q, want kimichuwu", got)
	}
	if got := (&Server{}).userKeyFromRequest(request); got != "header-user" {
		t.Fatalf("request user = %q, want header-user", got)
	}
}

func TestWatchDirsForInstanceKeepsOnlyFixedTenant(t *testing.T) {
	watchDirs := map[string][]WatchRoot{
		"garnetmoon": {{Path: "/data/garnetmoon"}},
		"kimichuwu":  {{Path: "/data/kimichuwu"}},
	}
	filtered, err := watchDirsForInstance(watchDirs, "kimichuwu")
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 1 || len(filtered["kimichuwu"]) != 1 || filtered["kimichuwu"][0].Path != "/data/kimichuwu" {
		t.Fatalf("unexpected fixed watch roots: %#v", filtered)
	}
	if _, err := watchDirsForInstance(watchDirs, "unknown"); err == nil {
		t.Fatal("expected mismatched fixed tenant to fail")
	}
}

func TestExtractEPUBText(t *testing.T) {
	path := filepath.Join(t.TempDir(), "book.epub")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	entries := map[string]string{
		"META-INF/container.xml": `<?xml version="1.0"?>
			<container>
				<rootfiles>
					<rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
				</rootfiles>
			</container>`,
		"OPS/package.opf": `<?xml version="1.0"?>
			<package xmlns:dc="http://purl.org/dc/elements/1.1/">
				<metadata><dc:title>Example Book</dc:title></metadata>
				<manifest>
					<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
					<item id="marker" href="c9.xhtml" media-type="application/xhtml+xml"/>
					<item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
					<item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
				</manifest>
				<spine>
					<itemref idref="cover"/>
					<itemref idref="marker"/>
					<itemref idref="chapter1"/>
					<itemref idref="chapter2"/>
				</spine>
			</package>`,
		"OPS/cover.xhtml":    `<html><head><style>body { text-align: center; padding: 0pt; margin: 0pt; }</style><title>Cover</title></head><body>Cover</body></html>`,
		"OPS/c9.xhtml":       `<html><head><title>c9</title></head><body><h1>c9</h1></body></html>`,
		"OPS/chapter1.xhtml": `<html><head><title>First Chapter</title></head><body><h1>First</h1><p>Hello &amp; welcome.</p></body></html>`,
		"OPS/chapter2.xhtml": `<html><body><p>Second chapter text.</p></body></html>`,
	}
	for name, body := range entries {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	body, metadata, err := extractEPUBText(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, "Hello & welcome") {
		t.Fatalf("expected unescaped chapter text, got %q", body)
	}
	if strings.Contains(body, "padding") || strings.Contains(body, "text-align") {
		t.Fatalf("expected style text to be stripped, got %q", body)
	}
	if strings.Contains(body, "# c9") {
		t.Fatalf("expected generated epub marker to be skipped, got %q", body)
	}
	if !strings.Contains(body, "\f") {
		t.Fatalf("expected chapter separator, got %q", body)
	}
	if metadata["extractor"] != "native-epub" {
		t.Fatalf("unexpected extractor metadata: %#v", metadata)
	}
	if metadata["chapters"] != 2 {
		t.Fatalf("unexpected chapter count: %#v", metadata["chapters"])
	}
	if metadata["book_title"] != "Example Book" {
		t.Fatalf("unexpected book title: %#v", metadata["book_title"])
	}
}

func TestMetadataTextForChunkUsesStructureAndCooccurrences(t *testing.T) {
	req := UpsertMemoryRequest{
		Kind:       "document",
		SourceType: "watch_file",
		SourceURI:  "file:///library/The Glass Archive.epub",
		Title:      "The Glass Archive",
		Metadata: map[string]any{
			"book_title": "The Glass Archive",
			"collection": "fixture-books",
			"scope":      "library",
			"path":       "/library/The Glass Archive.epub",
		},
	}
	chunk := textChunk{
		Text: "He paused. \"Mira Voss is your mother?\" Niko ignored Daren and Lysa.",
		Metadata: map[string]any{
			"chapter":     7,
			"chunk_index": 214,
		},
	}
	context := chunkContextText(req, chunk, 214)
	got := metadataTextForChunk(req, chunk, 214, context, chunkContextMetadata{})

	for _, want := range []string{
		"Title: The Glass Archive",
		"Source: file:///library/The Glass Archive.epub",
		"Chapter: 7",
		"Entities:",
		"Co-occurrences:",
		"Mira Voss",
		"Niko",
		"Daren",
		"Mira Voss <-> Niko",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected retrieval text to contain %q, got:\n%s", want, got)
		}
	}
}

func TestEmbeddingTextForChunkIncludesRawChunk(t *testing.T) {
	got := embeddingTextForChunk("Contextual summary:\nA helper.", "Raw passage text.")
	if !strings.Contains(got, "Document metadata:\nContextual summary:") || !strings.Contains(got, "Passage:\nRaw passage text.") {
		t.Fatalf("expected embedding text to include helper and raw text, got %q", got)
	}
}

func TestChunkTextWithMetadataUsesParagraphsBeforeSplitting(t *testing.T) {
	chunks := chunkTextWithMetadata("First paragraph.\n\nSecond paragraph.")
	if len(chunks) != 2 {
		t.Fatalf("expected two paragraph chunks, got %d: %#v", len(chunks), chunks)
	}
	if chunks[0].Text != "First paragraph." || chunks[1].Text != "Second paragraph." {
		t.Fatalf("unexpected paragraph chunks: %#v", chunks)
	}
}

func TestChunkTextWithMetadataSplitsOversizedParagraph(t *testing.T) {
	text := strings.Repeat("a", maxChunkRunes) + strings.Repeat("b", maxChunkRunes)
	chunks := chunkTextWithMetadata(text)
	if len(chunks) < 2 {
		t.Fatalf("expected oversized paragraph to split, got %d chunks", len(chunks))
	}
	if chunks[0].Text == chunks[1].Text {
		t.Fatalf("expected split chunks to differ")
	}
}

func TestStripMarkupTextPreservesParagraphBoundaries(t *testing.T) {
	got := stripMarkupText("<p>First paragraph.</p><p>Second paragraph.</p><h2>Heading</h2><div>Body.</div>")
	if !strings.Contains(got, "First paragraph.\n\nSecond paragraph.") {
		t.Fatalf("expected paragraph boundary, got %q", got)
	}
	if !strings.Contains(got, "Heading\n\nBody.") {
		t.Fatalf("expected heading boundary, got %q", got)
	}
}

func TestParseContextualizerResults(t *testing.T) {
	got, err := parseContextualizerResults("```text\nBEGIN_CHUNK 0\nSUMMARY: A family discussion.\nENTITIES: Mira Voss | Niko\nEND_CHUNK\n```", 1)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Summary != "A family discussion." {
		t.Fatalf("unexpected summary: %#v", got[0])
	}
	if !strings.EqualFold(got[0].Entities[0], "Mira Voss") {
		t.Fatalf("unexpected entities: %#v", got[0].Entities)
	}
}

func TestParseContextualizerResultsAcceptsMissingFinalEndMarker(t *testing.T) {
	got, err := parseContextualizerResults("BEGIN_CHUNK 0\nSUMMARY: A family discussion.\nENTITIES: Mira Voss | Niko", 1)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Summary != "A family discussion." {
		t.Fatalf("unexpected summary: %#v", got[0])
	}
}

func TestIgnoredWatchDirectory(t *testing.T) {
	for _, name := range []string{".trash", ".caltrash", "manga", "review-to-delete", "Review-To-Delete"} {
		if !ignoredWatchDirectory(name) {
			t.Errorf("expected %q to be ignored", name)
		}
	}
	if ignoredWatchDirectory("textbooks") {
		t.Fatal("textbooks should remain watchable")
	}
}

func TestWatchedSourceExistsDetectsMovedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.md")
	if err := os.WriteFile(path, []byte("memory"), 0o600); err != nil {
		t.Fatal(err)
	}
	sourceURI := "file://" + path
	if !watchedSourceExists(sourceURI) {
		t.Fatal("expected existing watched file")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if watchedSourceExists(sourceURI) {
		t.Fatal("expected removed watched file to be treated as a move candidate")
	}
}

func TestContentHashVersionInvalidatesOnlyPDFs(t *testing.T) {
	common := []string{"Title", "body", "library", "collection"}
	note := contentHashFor(common[0], "file:///library/note.md", common[1], common[2:]...)
	book := contentHashFor(common[0], "file:///library/book.pdf", common[1], common[2:]...)
	if note == book {
		t.Fatal("expected PDF and non-PDF ingestion versions to differ")
	}
	wantNote := hashText(strings.Join([]string{contentHashVersion, "Title", "file:///library/note.md", "body", "library", "collection"}, "\n"))
	if note != wantNote {
		t.Fatal("non-PDF content hash must retain the previous ingestion version")
	}
}
