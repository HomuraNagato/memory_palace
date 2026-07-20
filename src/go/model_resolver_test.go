package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestResolveCompletionModelDiscoversAndCachesServedModel(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"served-model"}]}`))
	}))
	defer server.Close()

	memory := &Server{httpClient: server.Client(), resolvedModels: map[string]string{}}
	for attempt := 0; attempt < 2; attempt++ {
		model, err := memory.resolveCompletionModel(context.Background(), server.URL+"/v1/chat/completions", "auto")
		if err != nil {
			t.Fatal(err)
		}
		if model != "served-model" {
			t.Fatalf("unexpected model %q", model)
		}
	}
	if requests != 1 {
		t.Fatalf("expected one discovery request, got %d", requests)
	}
}

func TestResolveCompletionModelHonorsExplicitOverride(t *testing.T) {
	memory := &Server{}
	model, err := memory.resolveCompletionModel(context.Background(), "not-a-url", "configured-model")
	if err != nil {
		t.Fatal(err)
	}
	if model != "configured-model" {
		t.Fatalf("unexpected model %q", model)
	}
}
