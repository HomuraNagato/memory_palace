package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	chapterSummaryMaxRunes  = 18000
	chapterSummaryMaxOutput = 1200
	chapterSummaryWorkers   = 2
)

type chunkContextMetadata struct {
	Summary  string
	Entities []string
}

type contextualizerMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type contextualizerRequest struct {
	Model       string                  `json:"model,omitempty"`
	Messages    []contextualizerMessage `json:"messages"`
	Temperature float64                 `json:"temperature"`
	MaxTokens   int                     `json:"max_tokens"`
}

type contextualizerResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (s *Server) summarizeParentNodes(ctx context.Context, req UpsertMemoryRequest, parents []retrievalParentNode) (int, int) {
	if len(parents) == 0 {
		return 0, 0
	}
	targets := make([]int, 0, len(parents))
	for index, parent := range parents {
		if parent.NodeType == "chapter" {
			targets = append(targets, index)
		}
	}
	if len(targets) == 0 {
		return 0, 0
	}
	start := time.Now()
	log.Printf("memory.parent_summary status=start title=%q source_uri=%q chapters=%d workers=%d", req.Title, req.SourceURI, len(targets), chapterSummaryWorkers)
	type summaryResult struct {
		index   int
		summary string
		err     error
	}
	jobs := make(chan int)
	results := make(chan summaryResult, len(targets))
	var workers sync.WaitGroup
	workerCount := minInt(chapterSummaryWorkers, len(targets))
	for worker := 0; worker < workerCount; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for index := range jobs {
				summary, err := s.summarizeParent(ctx, req, parents[index])
				results <- summaryResult{index: index, summary: summary, err: err}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for _, index := range targets {
			jobs <- index
		}
	}()
	workers.Wait()
	close(results)

	generated := 0
	fallbacks := 0
	var firstErr error
	for result := range results {
		parent := &parents[result.index]
		if result.err != nil || strings.TrimSpace(result.summary) == "" {
			fallbacks++
			if firstErr == nil {
				firstErr = result.err
			}
			parent.Summary = deterministicParentSummary(*parent)
			parent.Metadata["summary_source"] = "deterministic_fallback"
			continue
		}
		generated++
		parent.Summary = result.summary
		parent.Metadata["summary_source"] = "reasoning_model"
	}
	if firstErr != nil {
		log.Printf("memory.parent_summary status=fallback title=%q source_uri=%q generated=%d fallback=%d ms=%d err=%q", req.Title, req.SourceURI, generated, fallbacks, time.Since(start).Milliseconds(), firstErr.Error())
	} else {
		log.Printf("memory.parent_summary status=ok title=%q source_uri=%q generated=%d fallback=%d ms=%d", req.Title, req.SourceURI, generated, fallbacks, time.Since(start).Milliseconds())
	}
	return generated, fallbacks
}

func (s *Server) summarizeParent(ctx context.Context, req UpsertMemoryRequest, parent retrievalParentNode) (string, error) {
	if strings.TrimSpace(s.contextualizerURL) == "" {
		return "", fmt.Errorf("contextualizer URL is not configured")
	}
	model, err := s.resolveCompletionModel(ctx, s.contextualizerURL, s.contextualizerModel)
	if err != nil {
		return "", err
	}
	prompt := strings.Builder{}
	prompt.WriteString("Write one concise, grounded summary of this chapter for future question answering.\n")
	prompt.WriteString("Preserve important names, relationships, definitions, events, and answer-bearing facts.\n")
	prompt.WriteString("Use only information explicitly present in the supplied text. Do not invent facts.\n")
	prompt.WriteString("Return plain text only: one paragraph, no JSON, markdown, labels, or commentary.\n\n")
	prompt.WriteString("Document title: ")
	prompt.WriteString(req.Title)
	prompt.WriteString("\nSection: ")
	prompt.WriteString(stringMapValue(parent.Metadata, "label"))
	prompt.WriteString("\n\nChapter text:\n")
	prompt.WriteString(limitParentRunes(parent.Text, chapterSummaryMaxRunes))

	body, err := json.Marshal(contextualizerRequest{
		Model: model,
		Messages: []contextualizerMessage{
			{Role: "system", Content: "You produce concise, factual summaries for retrieval. Return only the requested plain-text summary."},
			{Role: "user", Content: prompt.String()},
		},
		Temperature: 0,
		MaxTokens:   256,
	})
	if err != nil {
		return "", err
	}
	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, strings.TrimRight(s.contextualizerURL, "/"), bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := s.httpClient.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return "", err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("contextualizer status %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	var decoded contextualizerResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return "", err
	}
	if len(decoded.Choices) == 0 {
		return "", fmt.Errorf("contextualizer returned no choices")
	}
	summary := cleanParentSummary(decoded.Choices[0].Message.Content)
	if summary == "" {
		return "", fmt.Errorf("contextualizer returned an empty summary")
	}
	return summary, nil
}

func deterministicParentSummary(parent retrievalParentNode) string {
	parts := make([]string, 0, 3)
	for _, paragraph := range strings.Split(strings.TrimSpace(parent.Text), "\n\n") {
		paragraph = strings.Join(strings.Fields(paragraph), " ")
		if paragraph == "" {
			continue
		}
		parts = append(parts, paragraph)
		if len(parts) == 2 {
			break
		}
	}
	paragraphs := strings.Split(strings.TrimSpace(parent.Text), "\n\n")
	if len(paragraphs) > 2 {
		last := strings.Join(strings.Fields(paragraphs[len(paragraphs)-1]), " ")
		if last != "" {
			parts = append(parts, last)
		}
	}
	return limitParentRunes(strings.Join(parts, " "), chapterSummaryMaxOutput)
}

func cleanParentSummary(raw string) string {
	raw = strings.TrimSpace(raw)
	if start := strings.Index(raw, "<think>"); start >= 0 {
		if end := strings.Index(raw[start+len("<think>"):], "</think>"); end >= 0 {
			raw = raw[:start] + raw[start+len("<think>")+end+len("</think>"):]
		}
	}
	raw = strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(raw, "```"), "```text"))
	raw = strings.TrimSpace(strings.TrimPrefix(raw, "SUMMARY:"))
	raw = strings.TrimSpace(strings.TrimLeft(raw, "-• "))
	return limitParentRunes(strings.Join(strings.Fields(raw), " "), chapterSummaryMaxOutput)
}

func limitParentRunes(value string, max int) string {
	runes := []rune(strings.TrimSpace(value))
	if max <= 0 || len(runes) <= max {
		return string(runes)
	}
	left := max / 2
	right := max - left
	return string(runes[:left]) + " ... [middle omitted] ... " + string(runes[len(runes)-right:])
}

func (s *Server) contextualizeChunks(ctx context.Context, req UpsertMemoryRequest, chunks []textChunk) ([]chunkContextMetadata, error) {
	result := fallbackContextMetadata(req, chunks)
	if len(chunks) == 0 || strings.TrimSpace(s.contextualizerURL) == "" {
		return result, nil
	}
	batchSize := s.contextualizerBatchSize
	if batchSize <= 0 {
		batchSize = 8
	}
	logProgress := shouldLogIngestProgress(req) && len(chunks) >= 20
	contextStart := time.Now()
	if logProgress {
		log.Printf("memory.contextualize status=start title=%q source_uri=%q chunks=%d batch_size=%d", req.Title, req.SourceURI, len(chunks), batchSize)
	}
	var firstErr error
	for start := 0; start < len(chunks); start += batchSize {
		end := start + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch, err := s.contextualizeBatchResilient(ctx, req, chunks[start:end], start)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
		}
		copy(result[start:end], batch)
	}
	if logProgress {
		status := "ok"
		if firstErr != nil {
			status = "fallback"
		}
		log.Printf("memory.contextualize status=%s title=%q source_uri=%q chunks=%d ms=%d", status, req.Title, req.SourceURI, len(chunks), time.Since(contextStart).Milliseconds())
	}
	return result, firstErr
}

func (s *Server) contextualizeBatchResilient(ctx context.Context, req UpsertMemoryRequest, chunks []textChunk, offset int) ([]chunkContextMetadata, error) {
	result, err := s.contextualizeBatch(ctx, req, chunks, offset)
	if err == nil {
		return result, nil
	}
	if len(chunks) <= 1 {
		return fallbackContextMetadata(req, chunks), err
	}
	middle := len(chunks) / 2
	left, leftErr := s.contextualizeBatchResilient(ctx, req, chunks[:middle], offset)
	right, rightErr := s.contextualizeBatchResilient(ctx, req, chunks[middle:], offset+middle)
	combined := append(left, right...)
	switch {
	case leftErr != nil && rightErr != nil:
		return combined, fmt.Errorf("contextualizer split failures: %v; %v", leftErr, rightErr)
	case leftErr != nil:
		return combined, leftErr
	default:
		return combined, rightErr
	}
}

func (s *Server) contextualizeBatch(ctx context.Context, req UpsertMemoryRequest, chunks []textChunk, offset int) ([]chunkContextMetadata, error) {
	start := time.Now()
	model, err := s.resolveCompletionModel(ctx, s.contextualizerURL, s.contextualizerModel)
	if err != nil {
		return nil, err
	}
	var prompt strings.Builder
	prompt.WriteString("Generate retrieval metadata for the supplied text chunks.\n")
	prompt.WriteString("Return one plain-text block per chunk, in the same order, using exactly these markers:\n")
	prompt.WriteString("BEGIN_CHUNK 0\nSUMMARY: one concise sentence\nENTITIES: Name One | Name Two\nEND_CHUNK\n")
	prompt.WriteString("Use the actual chunk number in each BEGIN_CHUNK marker. Do not use JSON, markdown, bullets, or additional commentary.\n")
	prompt.WriteString("Use only information explicitly present in the supplied text and metadata.\n")
	prompt.WriteString("Write one concise contextual summary for retrieval. Preserve important names and topics.\n")
	prompt.WriteString("List only relevant named entities explicitly mentioned in the text. Use [] when there are none.\n")
	prompt.WriteString("Do not invent facts. Do not add a separate relationship list, answer, or question.\n\n")
	prompt.WriteString("Document title: ")
	prompt.WriteString(req.Title)
	prompt.WriteString("\n\nChunks:\n")
	for index, chunk := range chunks {
		fmt.Fprintf(&prompt, "--- id=%d ---\n", offset+index)
		if chapter := fmt.Sprint(chunk.Metadata["chapter"]); chapter != "<nil>" && chapter != "" {
			fmt.Fprintf(&prompt, "chapter: %s\n", chapter)
		}
		if page := fmt.Sprint(chunk.Metadata["page"]); page != "<nil>" && page != "" {
			fmt.Fprintf(&prompt, "page: %s\n", page)
		}
		prompt.WriteString("text:\n")
		prompt.WriteString(strings.TrimSpace(chunk.Text))
		prompt.WriteString("\n")
	}

	body, err := json.Marshal(contextualizerRequest{
		Model: model,
		Messages: []contextualizerMessage{
			{Role: "system", Content: "You produce compact, grounded retrieval metadata. Follow the requested plain-text block format exactly."},
			{Role: "user", Content: prompt.String()},
		},
		Temperature: 0,
		MaxTokens:   maxInt(512, len(chunks)*160),
	})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(s.contextualizerURL, "/"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := s.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("contextualizer status %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	var decoded contextualizerResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return nil, err
	}
	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("contextualizer returned no choices")
	}
	results, err := parseContextualizerResults(decoded.Choices[0].Message.Content, len(chunks))
	if err != nil {
		return nil, err
	}
	elapsed := time.Since(start)
	if shouldLogIngestProgress(req) && elapsed >= 5*time.Second {
		log.Printf("memory.contextualize_batch status=slow title=%q source_uri=%q offset=%d chunks=%d prompt_chars=%d response_chars=%d ms=%d", req.Title, req.SourceURI, offset, len(chunks), len([]rune(prompt.String())), len([]rune(decoded.Choices[0].Message.Content)), elapsed.Milliseconds())
	}
	return results, nil
}

func parseContextualizerResults(raw string, expected int) ([]chunkContextMetadata, error) {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```text")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)
	result := make([]chunkContextMetadata, 0, expected)
	position := 0
	for len(result) < expected {
		begin := strings.Index(raw[position:], "BEGIN_CHUNK")
		if begin < 0 {
			return nil, fmt.Errorf("contextualizer returned %d blocks, expected %d", len(result), expected)
		}
		begin += position
		endMarker := strings.Index(raw[begin:], "END_CHUNK")
		if endMarker < 0 {
			if len(result) != expected-1 {
				return nil, fmt.Errorf("contextualizer block %d was incomplete", len(result))
			}
			endMarker = len(raw) - begin
		}
		endMarker += begin
		block := raw[begin:endMarker]
		summary := ""
		entities := []string{}
		for _, line := range strings.Split(block, "\n") {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "SUMMARY:"):
				summary = strings.TrimSpace(strings.TrimPrefix(line, "SUMMARY:"))
			case strings.HasPrefix(line, "ENTITIES:"):
				rawEntities := strings.TrimSpace(strings.TrimPrefix(line, "ENTITIES:"))
				if rawEntities != "" && rawEntities != "none" && rawEntities != "[]" {
					for _, entity := range strings.Split(rawEntities, "|") {
						entities = append(entities, strings.TrimSpace(entity))
					}
				}
			}
		}
		if summary == "" {
			return nil, fmt.Errorf("contextualizer block %d had no summary", len(result))
		}
		result = append(result, chunkContextMetadata{Summary: summary, Entities: uniqueLimitedStrings(entities, 24)})
		position = endMarker + len("END_CHUNK")
	}
	return result, nil
}

func fallbackContextMetadata(req UpsertMemoryRequest, chunks []textChunk) []chunkContextMetadata {
	result := make([]chunkContextMetadata, len(chunks))
	for index, chunk := range chunks {
		result[index] = chunkContextMetadata{
			Summary:  contextualSummaryForChunk(req, chunk, index),
			Entities: mentionedEntitiesForChunk(req, chunk),
		}
	}
	return result
}

func parsePositiveInt(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}
