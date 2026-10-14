package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveMemorySourcePathAllowsSupportedWatchedFile(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "book.pdf")
	if err := os.WriteFile(source, []byte("pdf"), 0o600); err != nil {
		t.Fatal(err)
	}
	item := sourceDocument{
		SourceType: "watch_file",
		SourceURI:  "file://" + source,
		SourcePath: source,
	}

	got, err := resolveMemorySourcePath(item, []WatchRoot{{Path: root}})
	if err != nil {
		t.Fatal(err)
	}
	if got != source {
		t.Fatalf("resolved path = %q, want %q", got, source)
	}
}

func TestResolveMemorySourcePathRejectsOutsideRoot(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "book.epub")
	if err := os.WriteFile(outside, []byte("epub"), 0o600); err != nil {
		t.Fatal(err)
	}
	item := sourceDocument{SourceType: "watch_file", SourcePath: outside}

	_, err := resolveMemorySourcePath(item, []WatchRoot{{Path: root}})
	if !errors.Is(err, errSourceForbidden) {
		t.Fatalf("error = %v, want forbidden", err)
	}
}

func TestResolveMemorySourcePathRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "book.md")
	if err := os.WriteFile(outside, []byte("# Book"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "book.md")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}
	item := sourceDocument{SourceType: "watch_file", SourcePath: link}

	_, err := resolveMemorySourcePath(item, []WatchRoot{{Path: root}})
	if !errors.Is(err, errSourceForbidden) {
		t.Fatalf("error = %v, want forbidden", err)
	}
}

func TestResolveMemorySourcePathRejectsUnsupportedFormat(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "notes.txt")
	if err := os.WriteFile(source, []byte("notes"), 0o600); err != nil {
		t.Fatal(err)
	}
	item := sourceDocument{SourceType: "watch_file", SourcePath: source}

	_, err := resolveMemorySourcePath(item, []WatchRoot{{Path: root}})
	if !errors.Is(err, errSourceUnsupported) {
		t.Fatalf("error = %v, want unsupported", err)
	}
}

func TestReaderContentType(t *testing.T) {
	tests := map[string]string{
		"book.pdf":       "application/pdf",
		"book.epub":      "application/epub+zip",
		"notes.md":       "text/markdown; charset=utf-8",
		"notes.markdown": "text/markdown; charset=utf-8",
	}
	for source, want := range tests {
		if got := readerContentType(source); got != want {
			t.Errorf("readerContentType(%q) = %q, want %q", source, got, want)
		}
	}
}

func TestReaderFormat(t *testing.T) {
	tests := []struct {
		name      string
		format    string
		sourceURI string
		want      string
	}{
		{name: "metadata", format: " .EPUB ", sourceURI: "file:///book.pdf", want: ".epub"},
		{name: "URI", sourceURI: "file:///notes.MD?download=1#intro", want: ".md"},
		{name: "missing", sourceURI: "memory://note", want: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := readerFormat(test.format, test.sourceURI); got != test.want {
				t.Fatalf("readerFormat(%q, %q) = %q, want %q", test.format, test.sourceURI, got, test.want)
			}
		})
	}
}

func TestServeMemorySourceFileSupportsRangesAndHead(t *testing.T) {
	source := filepath.Join(t.TempDir(), "book.pdf")
	if err := os.WriteFile(source, []byte("0123456789"), 0o600); err != nil {
		t.Fatal(err)
	}

	rangeRequest := httptest.NewRequest(http.MethodGet, "/source", nil)
	rangeRequest.Header.Set("Range", "bytes=2-5")
	rangeResponse := httptest.NewRecorder()
	if err := serveMemorySourceFile(rangeResponse, rangeRequest, source); err != nil {
		t.Fatal(err)
	}
	if rangeResponse.Code != http.StatusPartialContent || rangeResponse.Body.String() != "2345" {
		t.Fatalf("range response = %d %q", rangeResponse.Code, rangeResponse.Body.String())
	}
	if rangeResponse.Header().Get("Content-Range") != "bytes 2-5/10" {
		t.Fatalf("content range = %q", rangeResponse.Header().Get("Content-Range"))
	}

	headRequest := httptest.NewRequest(http.MethodHead, "/source", nil)
	headResponse := httptest.NewRecorder()
	if err := serveMemorySourceFile(headResponse, headRequest, source); err != nil {
		t.Fatal(err)
	}
	if headResponse.Code != http.StatusOK || strings.TrimSpace(headResponse.Body.String()) != "" {
		t.Fatalf("HEAD response = %d %q", headResponse.Code, headResponse.Body.String())
	}
	if headResponse.Header().Get("Content-Length") != "10" || headResponse.Header().Get("Accept-Ranges") != "bytes" {
		t.Fatalf("HEAD headers = %#v", headResponse.Header())
	}
}
