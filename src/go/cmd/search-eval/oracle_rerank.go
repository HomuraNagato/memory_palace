package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

const (
	defaultOracleDatabaseURL = "host=100.73.65.14 port=5438 dbname=pensieve user=postgres password=postgres sslmode=disable"
	defaultOracleRerankerURL = "http://127.0.0.1:5022"
)

type oracleRerankCandidate struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
	Text  string `json:"text"`
}

type oracleRerankRequest struct {
	RequestID  string                  `json:"request_id,omitempty"`
	Query      string                  `json:"query"`
	Candidates []oracleRerankCandidate `json:"candidates"`
	TopK       int                     `json:"top_k"`
}

type oracleRerankResponse struct {
	Model   string               `json:"model"`
	Results []oracleRerankResult `json:"results"`
}

type oracleRerankResult struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
	Index int     `json:"index"`
}

func runSuiteAudit(ctx context.Context, databaseURL string, defaults suiteDefaults, cases []evalCase, summaryOnly bool) []evaluationResult {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return []evaluationResult{{Err: fmt.Errorf("open audit database: %w", err)}}
	}
	defer db.Close()
	if err := db.PingContext(ctx); err != nil {
		return []evaluationResult{{Err: fmt.Errorf("ping audit database: %w", err)}}
	}

	results := make([]evaluationResult, 0, len(cases))
	for _, item := range cases {
		result := evaluationResult{Case: item, Strategy: "suite_audit"}
		if item.Expected.Unanswerable {
			result.Skipped = true
			results = append(results, result)
			continue
		}
		candidates, err := loadOracleSourceChunks(ctx, db, defaults.UserKey, item.Expected.Evidence)
		if err != nil {
			result.Err = err
		} else {
			evaluated := evaluateResults(item, len(candidates), candidates, 0)
			result.DocumentRank = evaluated.DocumentRank
			result.EvidenceRank = evaluated.EvidenceRank
			result.MatchedDocumentGroups = evaluated.MatchedDocumentGroups
			result.MatchedEvidenceGroups = evaluated.MatchedEvidenceGroups
			result.RetrievedCandidates = len(candidates)
			if result.EvidenceRank == 0 {
				result.Err = fmt.Errorf("indexed source is missing one or more evidence anchors")
			}
		}
		results = append(results, result)
		if !summaryOnly {
			fmt.Printf(
				"audit id=%s source_chunks=%d document_groups=%d/%d evidence_groups=%d/%d status=%s\n",
				item.ID,
				result.RetrievedCandidates,
				result.MatchedDocumentGroups,
				len(item.Expected.Evidence),
				result.MatchedEvidenceGroups,
				len(item.Expected.Evidence),
				auditStatus(result.Err),
			)
		}
	}

	var checked, valid, oversized int
	for _, result := range results {
		if result.Skipped {
			continue
		}
		checked++
		if result.Err == nil {
			valid++
		}
		if result.RetrievedCandidates > 150 {
			oversized++
		}
	}
	fmt.Printf("audit_summary checked=%d valid=%d invalid=%d sources_over_150_chunks=%d\n", checked, valid, checked-valid, oversized)
	return results
}

func auditStatus(err error) string {
	if err != nil {
		return "invalid"
	}
	return "valid"
}

func runOracleSourceEvaluation(ctx context.Context, client *http.Client, databaseURL, rerankerURL string, defaults suiteDefaults, cases []evalCase, topK int, summaryOnly bool) []evaluationResult {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return []evaluationResult{{Err: fmt.Errorf("open oracle database: %w", err)}}
	}
	defer db.Close()
	if err := db.PingContext(ctx); err != nil {
		return []evaluationResult{{Err: fmt.Errorf("ping oracle database: %w", err)}}
	}
	results := make([]evaluationResult, 0, len(cases))
	for _, item := range cases {
		result := runOracleSourceCase(ctx, client, db, strings.TrimRight(rerankerURL, "/"), defaults, item, topK)
		results = append(results, result)
		if !summaryOnly {
			printCaseResult(result)
		}
	}
	printSummary(results, topK, "oracle_source_cross_encoder")
	return results
}

func runOracleSourceCase(ctx context.Context, client *http.Client, db *sql.DB, rerankerURL string, defaults suiteDefaults, item evalCase, topK int) evaluationResult {
	started := time.Now()
	base := evaluationResult{Case: item, Strategy: "oracle_source_cross_encoder", FinalLimit: topK}
	if item.Expected.Unanswerable {
		base.Skipped = true
		return base
	}
	candidates, err := loadOracleSourceChunks(ctx, db, defaults.UserKey, item.Expected.Evidence)
	if err != nil {
		base.Err = err
		base.Elapsed = time.Since(started)
		return base
	}
	base.RetrievedCandidates = len(candidates)
	base.SelectedCandidates = len(candidates)
	rerankStarted := time.Now()
	ranked, model, err := rerankOracleSource(ctx, client, rerankerURL, item.ID, item.Query, candidates)
	base.RerankMs = time.Since(rerankStarted).Milliseconds()
	base.PipelineMs = time.Since(started).Milliseconds()
	base.Elapsed = time.Since(started)
	if err != nil {
		base.Err = err
		return base
	}
	full := evaluateResults(item, len(ranked), ranked, base.Elapsed)
	limited := ranked[:minInt(topK, len(ranked))]
	selected := evaluateResults(item, topK, limited, base.Elapsed)
	base.CandidateDocumentRank = full.DocumentRank
	base.CandidateEvidenceRank = full.EvidenceRank
	base.DocumentRank = selected.DocumentRank
	base.EvidenceRank = selected.EvidenceRank
	base.MatchedDocumentGroups = selected.MatchedDocumentGroups
	base.MatchedEvidenceGroups = selected.MatchedEvidenceGroups
	base.StageRanks = []evaluationStageRank{{Name: "oracle_source_cross_encoder_" + sanitizeStageName(model), DocumentRank: selected.DocumentRank, EvidenceRank: selected.EvidenceRank}}
	return base
}

func runCandidateRerankerCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase, cap, topK int, rerankerURLs []string) []evaluationResult {
	if item.Expected.Unanswerable {
		return nil
	}
	started := time.Now()
	request := retrieveRequest{
		Query:          item.Query,
		Scope:          requestScope(defaults, item),
		Collection:     requestCollection(defaults, item),
		Mode:           requestMode(defaults, item),
		CandidateLimit: cap,
		CandidateOnly:  true,
	}
	var decoded retrieveResponse
	if err := postJSON(client, baseURL+"/v1/retrieve", defaults.UserKey, request, &decoded); err != nil {
		return []evaluationResult{{Case: item, Cap: cap, Elapsed: time.Since(started), Err: err, Strategy: "candidate_cross_encoder_retrieval"}}
	}
	candidateResult := evaluateResults(item, cap, decoded.Candidates, time.Since(started))
	results := make([]evaluationResult, 0, len(rerankerURLs))
	for _, rerankerURL := range rerankerURLs {
		rerankStarted := time.Now()
		ranked, model, err := rerankOracleSource(context.Background(), client, strings.TrimRight(rerankerURL, "/"), item.ID, item.Query, decoded.Candidates)
		rerankMs := time.Since(rerankStarted).Milliseconds()
		modelStrategy := "candidate_cross_encoder_" + sanitizeStageName(model)
		if model == "" {
			modelStrategy = "candidate_cross_encoder_" + sanitizeStageName(rerankerURL)
		}
		result := evaluationResult{
			Case:                  item,
			Cap:                   cap,
			CandidateDocumentRank: candidateResult.DocumentRank,
			CandidateEvidenceRank: candidateResult.EvidenceRank,
			RetrievedCandidates:   len(decoded.Candidates),
			WindowMatches:         decoded.Diagnostics.WindowMatchedCount,
			WindowExpandedLeaves:  decoded.Diagnostics.WindowExpandedLeafCount,
			WindowCandidates:      decoded.Diagnostics.WindowCandidateCount,
			SelectedCandidates:    len(decoded.Candidates),
			SearchMs:              decoded.Diagnostics.SearchMs,
			WindowSearchMs:        decoded.Diagnostics.WindowSearchMs,
			RerankMs:              rerankMs,
			PipelineMs:            decoded.Diagnostics.TotalMs + rerankMs,
			Elapsed:               time.Since(started),
			Err:                   err,
			Strategy:              modelStrategy,
			FinalLimit:            topK,
		}
		if err == nil {
			limited := ranked[:minInt(minPositive(topK, cap), len(ranked))]
			evaluated := evaluateResults(item, topK, limited, result.Elapsed)
			result.DocumentRank = evaluated.DocumentRank
			result.EvidenceRank = evaluated.EvidenceRank
			result.MatchedDocumentGroups = evaluated.MatchedDocumentGroups
			result.MatchedEvidenceGroups = evaluated.MatchedEvidenceGroups
			result.StageRanks = []evaluationStageRank{{Name: modelStrategy, DocumentRank: result.DocumentRank, EvidenceRank: result.EvidenceRank}}
		}
		results = append(results, result)
	}
	return results
}

func loadOracleSourceChunks(ctx context.Context, db *sql.DB, userKey string, evidence []expectedEvidence) ([]searchResult, error) {
	byID := map[string]searchResult{}
	for _, expected := range evidence {
		rows, err := db.QueryContext(ctx, `
			SELECT c.id, i.title, c.text, i.source_uri
			FROM memory_items i
			JOIN memory_chunks c ON c.memory_id = i.id
			WHERE i.user_key = $1
				AND lower(i.title) = lower($2)
				AND i.source_uri NOT LIKE '%/.trash/%'
			ORDER BY c.chunk_index
		`, userKey, expected.Source.Title)
		if err != nil {
			return nil, err
		}
		matchedSource := 0
		for rows.Next() {
			var candidate searchResult
			if err := rows.Scan(&candidate.ChunkID, &candidate.Title, &candidate.Text, &candidate.SourceURI); err != nil {
				rows.Close()
				return nil, err
			}
			if !matchesSource(candidate, expected.Source) {
				continue
			}
			byID[candidate.ChunkID] = candidate
			matchedSource++
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
		if matchedSource == 0 {
			return nil, fmt.Errorf("oracle source not found: %s (%s)", expected.Source.Title, expected.Source.RelativePath)
		}
	}
	candidates := make([]searchResult, 0, len(byID))
	for _, candidate := range byID {
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ChunkID < candidates[j].ChunkID })
	return candidates, nil
}

func rerankOracleSource(ctx context.Context, client *http.Client, rerankerURL, caseID, query string, candidates []searchResult) ([]searchResult, string, error) {
	payload := oracleRerankRequest{
		RequestID:  "oracle-source-" + caseID,
		Query:      query,
		TopK:       len(candidates),
		Candidates: make([]oracleRerankCandidate, len(candidates)),
	}
	for index, candidate := range candidates {
		payload.Candidates[index] = oracleRerankCandidate{ID: candidate.ChunkID, Title: candidate.Title, Text: candidate.Text}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, rerankerURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, "", err
	}
	defer response.Body.Close()
	var decoded oracleRerankResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return nil, "", err
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, decoded.Model, fmt.Errorf("oracle reranker status %s", response.Status)
	}
	ranked, err := orderOracleCandidates(candidates, decoded)
	return ranked, decoded.Model, err
}

func orderOracleCandidates(candidates []searchResult, response oracleRerankResponse) ([]searchResult, error) {
	byID := make(map[string]searchResult, len(candidates))
	for _, candidate := range candidates {
		byID[candidate.ChunkID] = candidate
	}
	ordered := make([]searchResult, 0, len(candidates))
	seen := make(map[string]bool, len(candidates))
	for _, result := range response.Results {
		candidate, ok := byID[result.ID]
		if !ok || seen[result.ID] {
			continue
		}
		ordered = append(ordered, candidate)
		seen[result.ID] = true
	}
	if len(ordered) != len(candidates) {
		return nil, fmt.Errorf("oracle reranker returned %d of %d candidates", len(ordered), len(candidates))
	}
	return ordered, nil
}

func sanitizeStageName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.Map(func(char rune) rune {
		if char >= 'a' && char <= 'z' || char >= '0' && char <= '9' {
			return char
		}
		return '_'
	}, value)
	return strings.Trim(value, "_")
}

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

func maxDuration(left, right time.Duration) time.Duration {
	if left > right {
		return left
	}
	return right
}
