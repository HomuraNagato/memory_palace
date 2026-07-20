package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strings"
)

type openAIModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

func (s *Server) resolveCompletionModel(ctx context.Context, completionURL, configured string) (string, error) {
	configured = strings.TrimSpace(configured)
	if configured != "" && !strings.EqualFold(configured, "auto") {
		return configured, nil
	}

	modelsURL, err := openAIModelsURL(completionURL)
	if err != nil {
		return "", err
	}
	s.resolvedModelsMu.Lock()
	model := s.resolvedModels[modelsURL]
	s.resolvedModelsMu.Unlock()
	if model != "" {
		return model, nil
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return "", err
	}
	response, err := s.httpClient.Do(request)
	if err != nil {
		return "", fmt.Errorf("discover completion model: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("discover completion model: status %s", response.Status)
	}
	var decoded openAIModelsResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return "", fmt.Errorf("discover completion model: %w", err)
	}
	for _, candidate := range decoded.Data {
		if model = strings.TrimSpace(candidate.ID); model != "" {
			break
		}
	}
	if model == "" {
		return "", fmt.Errorf("discover completion model: server returned no models")
	}

	s.resolvedModelsMu.Lock()
	if existing := s.resolvedModels[modelsURL]; existing != "" {
		model = existing
	} else {
		s.resolvedModels[modelsURL] = model
		log.Printf("memory.completion_model status=discovered url=%q model=%q", modelsURL, model)
	}
	s.resolvedModelsMu.Unlock()
	return model, nil
}

func openAIModelsURL(completionURL string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(completionURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("invalid completion URL %q", completionURL)
	}
	parsed.Path = strings.TrimSuffix(parsed.Path, "/v1/chat/completions")
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/v1/models"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}
