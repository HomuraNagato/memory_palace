package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	defaultBaseURL   = "http://127.0.0.1:8013"
	defaultSuitePath = "../../eval/search_eval.yaml"
)

type suite struct {
	Version     int           `yaml:"version"`
	Name        string        `yaml:"name"`
	Description string        `yaml:"description"`
	Defaults    suiteDefaults `yaml:"defaults"`
	Queries     []evalCase    `yaml:"queries"`
}

type suiteDefaults struct {
	UserKey      string `yaml:"user_key"`
	Scope        string `yaml:"scope"`
	Collection   string `yaml:"collection"`
	Mode         string `yaml:"mode"`
	ExpectedTopK int    `yaml:"expected_top_k"`
}

type evalCase struct {
	ID         string       `yaml:"id"`
	Difficulty string       `yaml:"difficulty"`
	Category   string       `yaml:"category"`
	Query      string       `yaml:"query"`
	Request    evalRequest  `yaml:"request,omitempty"`
	Expected   expectedCase `yaml:"expected"`
}

type evalRequest struct {
	Scope      string `yaml:"scope,omitempty"`
	Collection string `yaml:"collection,omitempty"`
	Mode       string `yaml:"mode,omitempty"`
}

type expectedCase struct {
	Evidence     []expectedEvidence `yaml:"evidence,omitempty"`
	Answer       expectedAnswer     `yaml:"answer,omitempty"`
	Unanswerable bool               `yaml:"unanswerable,omitempty"`
}

type expectedEvidence struct {
	Source expectedSource `yaml:"source"`
	AnyOf  []string       `yaml:"any_of,omitempty"`
	AllOf  []string       `yaml:"all_of,omitempty"`
}

type expectedAnswer struct {
	AnyOf []string `yaml:"any_of,omitempty"`
	AllOf []string `yaml:"all_of,omitempty"`
}

type expectedSource struct {
	RelativePath string `yaml:"relative_path"`
	Title        string `yaml:"title"`
}

type searchRequest struct {
	Query      string `json:"query"`
	Scope      string `json:"scope,omitempty"`
	Collection string `json:"collection,omitempty"`
	Mode       string `json:"mode,omitempty"`
	Limit      int    `json:"limit,omitempty"`
}

type searchResponse struct {
	Results []searchResult `json:"results"`
}

type retrieveRequest struct {
	Query          string `json:"query"`
	Scope          string `json:"scope,omitempty"`
	Collection     string `json:"collection,omitempty"`
	Mode           string `json:"mode,omitempty"`
	Limit          int    `json:"limit,omitempty"`
	CandidateLimit int    `json:"candidate_limit"`
	CandidateOnly  bool   `json:"candidate_only"`
	TraceStages    bool   `json:"trace_stages"`
}

type retrieveResponse struct {
	Candidates  []searchResult `json:"candidates"`
	Sources     []searchResult `json:"sources"`
	Diagnostics struct {
		RetrievedCandidateCount   int   `json:"retrieved_candidate_count"`
		GlobalCandidateCount      int   `json:"global_candidate_count"`
		WindowMatchedCount        int   `json:"window_matched_count"`
		WindowExpandedLeafCount   int   `json:"window_expanded_leaf_count"`
		WindowCandidateCount      int   `json:"window_candidate_count"`
		ConditionedCandidateCount int   `json:"conditioned_candidate_count"`
		CandidateCount            int   `json:"candidate_count"`
		RerankCandidateCount      int   `json:"rerank_candidate_count"`
		SelectedCount             int   `json:"selected_count"`
		SelectedDocumentCount     int   `json:"selected_document_count"`
		GlobalSearchMs            int64 `json:"global_search_ms"`
		WindowSearchMs            int64 `json:"window_search_ms"`
		ConditionedSearchMs       int64 `json:"conditioned_search_ms"`
		SearchMs                  int64 `json:"search_ms"`
		RerankMs                  int64 `json:"rerank_ms"`
		TotalMs                   int64 `json:"total_ms"`
	} `json:"diagnostics"`
	StageTrace *retrieveStageTrace `json:"stage_trace"`
}

type retrieveStageTrace struct {
	Candidates []retrieveTraceCandidate `json:"candidates"`
	Stages     []retrieveTraceStage     `json:"stages"`
}

type retrieveTraceCandidate struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Text      string `json:"text"`
	SourceURI string `json:"source_uri"`
}

type retrieveTraceStage struct {
	Name         string   `json:"name"`
	CandidateIDs []string `json:"candidate_ids"`
}

type searchResult struct {
	ChunkID   string `json:"chunk_id"`
	Title     string `json:"title"`
	Text      string `json:"text"`
	SourceURI string `json:"source_uri"`
}

type evaluationResult struct {
	Case                  evalCase
	Cap                   int
	DocumentRank          int
	EvidenceRank          int
	MatchedDocumentGroups int
	MatchedEvidenceGroups int
	CandidateDocumentRank int
	CandidateEvidenceRank int
	RetrievedCandidates   int
	GlobalCandidates      int
	WindowMatches         int
	WindowExpandedLeaves  int
	WindowCandidates      int
	ConditionedCandidates int
	SelectedCandidates    int
	SelectedDocuments     int
	GlobalSearchMs        int64
	WindowSearchMs        int64
	ConditionedSearchMs   int64
	SearchMs              int64
	RerankMs              int64
	PipelineMs            int64
	AnswerHit             bool
	AnswerTested          bool
	Skipped               bool
	Elapsed               time.Duration
	Err                   error
	StageRanks            []evaluationStageRank
	Strategy              string
	FinalLimit            int
}

type evaluationStageRank struct {
	Name         string
	DocumentRank int
	EvidenceRank int
}

func main() {
	var (
		suitePath     = flag.String("suite", defaultSuitePath, "Path to the YAML evaluation suite")
		baseURL       = flag.String("base-url", defaultBaseURL, "Memory Palace HTTP base URL")
		caseIDs       = flag.String("case", "", "Comma-separated case IDs to run; empty runs every case")
		candidateCaps = flag.String("candidate-caps", "", "Comma-separated candidate caps; uses candidate-only retrieval when set")
		candidateTopK = flag.Int("candidate-top-k", 0, "Evaluate only the first K candidates")
		finalLimits   = flag.String("final-limits", "", "Comma-separated final passage limits; defaults to expected_top_k")
		rerankerURLs  = flag.String("candidate-reranker-urls", "", "Comma-separated cross-encoder URLs applied to the same candidate pool")
		beemoURL      = flag.String("beemo-url", "", "Optional Beemo UI URL for cases with expected.answer")
		auditSuite    = flag.Bool("audit-suite", false, "Verify expected sources and evidence anchors against the indexed corpus")
		oracleSource  = flag.Bool("oracle-source-rerank", false, "Rerank every chunk from each case's expected source")
		oracleDBURL   = flag.String("database-url", envOrDefault("MEMORY_PALACE_DATABASE_URL", defaultOracleDatabaseURL), "Postgres connection string for oracle diagnostics")
		oracleRerank  = flag.String("reranker-url", defaultOracleRerankerURL, "Cross-encoder URL for oracle diagnostics")
		oracleTopK    = flag.Int("oracle-top-k", 0, "Oracle reranker cutoff; defaults to expected_top_k")
		summaryOnly   = flag.Bool("summary-only", false, "Suppress per-case output and print aggregate retrieval summaries")
		timeout       = flag.Duration("timeout", 45*time.Second, "Per-search request timeout")
	)
	flag.Parse()

	suite, err := loadSuite(*suitePath)
	if err != nil {
		exitf("load suite: %v", err)
	}
	selected, err := selectCases(suite.Queries, *caseIDs)
	if err != nil {
		exitf("select cases: %v", err)
	}
	client := &http.Client{Timeout: *timeout}
	if *auditSuite {
		results := runSuiteAudit(context.Background(), *oracleDBURL, suite.Defaults, selected, *summaryOnly)
		if hasFailures(results) {
			os.Exit(1)
		}
		return
	}
	if *oracleSource {
		topK := *oracleTopK
		if topK <= 0 {
			topK = suite.Defaults.ExpectedTopK
		}
		oracleClient := &http.Client{Timeout: maxDuration(*timeout, 10*time.Minute)}
		results := runOracleSourceEvaluation(context.Background(), oracleClient, *oracleDBURL, *oracleRerank, suite.Defaults, selected, topK, *summaryOnly)
		if hasFailures(results) {
			os.Exit(1)
		}
		return
	}
	if strings.TrimSpace(*beemoURL) != "" {
		results := make([]evaluationResult, 0, len(selected))
		for _, item := range selected {
			if !hasExpectedAnswer(item.Expected.Answer) {
				continue
			}
			result := runAnswerCase(client, strings.TrimRight(*beemoURL, "/"), suite.Defaults, item)
			results = append(results, result)
			if !*summaryOnly {
				printCaseResult(result)
			}
		}
		if len(results) == 0 {
			exitf("no selected cases define expected.answer")
		}
		if hasFailures(results) {
			os.Exit(1)
		}
		return
	}
	caps, err := parseCandidateCaps(*candidateCaps)
	if err != nil {
		exitf("parse candidate caps: %v", err)
	}
	if len(caps) > 0 {
		candidateRerankerURLs := nonBlankStrings(strings.Split(*rerankerURLs, ","))
		if *candidateTopK < 0 || *candidateTopK > 150 {
			exitf("-candidate-top-k must be between 0 and 150")
		}
		var allResults []evaluationResult
		for _, cap := range caps {
			if len(candidateRerankerURLs) > 0 {
				byModel := map[string][]evaluationResult{}
				modelOrder := []string{}
				for _, item := range selected {
					compared := runCandidateRerankerCase(client, strings.TrimRight(*baseURL, "/"), suite.Defaults, item, cap, *candidateTopK, candidateRerankerURLs)
					for _, result := range compared {
						if _, exists := byModel[result.Strategy]; !exists {
							modelOrder = append(modelOrder, result.Strategy)
						}
						byModel[result.Strategy] = append(byModel[result.Strategy], result)
						allResults = append(allResults, result)
						if !*summaryOnly {
							printCaseResult(result)
						}
					}
				}
				summaryK := minPositive(*candidateTopK, cap)
				for _, model := range modelOrder {
					printSummary(byModel[model], summaryK, model)
				}
				continue
			}
			results := make([]evaluationResult, 0, len(selected))
			for _, item := range selected {
				result := runCandidateCase(client, strings.TrimRight(*baseURL, "/"), suite.Defaults, item, cap, *candidateTopK)
				results = append(results, result)
				if !*summaryOnly {
					printCaseResult(result)
				}
			}
			summaryK := minPositive(*candidateTopK, cap)
			printSummary(results, summaryK, "first_phase")
			allResults = append(allResults, results...)
		}
		if hasFailures(allResults) {
			os.Exit(1)
		}
		return
	}
	if strings.TrimSpace(*rerankerURLs) != "" {
		exitf("-candidate-reranker-urls requires -candidate-caps")
	}
	limits, err := parseFinalLimits(*finalLimits, suite.Defaults.ExpectedTopK)
	if err != nil {
		exitf("parse final limits: %v", err)
	}
	var allResults []evaluationResult
	for _, finalLimit := range limits {
		results := make([]evaluationResult, 0, len(selected))
		for _, item := range selected {
			result := runCase(client, strings.TrimRight(*baseURL, "/"), suite.Defaults, item, finalLimit)
			results = append(results, result)
			if !*summaryOnly {
				printCaseResult(result)
			}
		}
		printSummary(results, finalLimit, "pipeline")
		allResults = append(allResults, results...)
	}
	if hasFailures(allResults) {
		os.Exit(1)
	}
}

type answerRequest struct {
	SessionID string            `json:"session_id"`
	Messages  []answerMessage   `json:"messages"`
	Options   map[string]string `json:"options"`
}

type answerMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type answerResponse struct {
	Text string `json:"text"`
}

func runAnswerCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase) evaluationResult {
	started := time.Now()
	request := answerRequest{
		SessionID: "memory-eval-" + item.ID,
		Messages:  []answerMessage{{Role: "user", Content: item.Query}},
		Options: map[string]string{
			"request_id":         "memory-eval-" + item.ID,
			"memory_ask":         "true",
			"memory_user_key":    defaults.UserKey,
			"memory_search_mode": requestMode(defaults, item),
			"memory_scope":       requestScope(defaults, item),
			"memory_collection":  requestCollection(defaults, item),
		},
	}
	var decoded answerResponse
	if err := postJSON(client, baseURL+"/api/chat", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, AnswerTested: true, Elapsed: time.Since(started), Err: err}
	}
	return evaluationResult{
		Case:         item,
		AnswerTested: true,
		AnswerHit:    matchesExpectedAnswer(decoded.Text, item.Expected.Answer),
		Elapsed:      time.Since(started),
	}
}

func loadSuite(filePath string) (suite, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return suite{}, err
	}
	defer file.Close()
	decoder := yaml.NewDecoder(file)
	decoder.KnownFields(true)
	var loaded suite
	if err := decoder.Decode(&loaded); err != nil {
		return suite{}, err
	}
	if err := validateSuite(loaded); err != nil {
		return suite{}, err
	}
	return loaded, nil
}

func validateSuite(loaded suite) error {
	if loaded.Version != 3 {
		return fmt.Errorf("unsupported suite version %d", loaded.Version)
	}
	if strings.TrimSpace(loaded.Defaults.UserKey) == "" {
		return errors.New("defaults.user_key is required")
	}
	if loaded.Defaults.ExpectedTopK <= 0 {
		return errors.New("defaults.expected_top_k must be positive")
	}
	if len(loaded.Queries) == 0 {
		return errors.New("queries cannot be empty")
	}
	seen := map[string]bool{}
	for _, item := range loaded.Queries {
		if strings.TrimSpace(item.ID) == "" || strings.TrimSpace(item.Query) == "" {
			return errors.New("every query requires id and query")
		}
		if seen[item.ID] {
			return fmt.Errorf("duplicate query id %q", item.ID)
		}
		seen[item.ID] = true
		if !containsString([]string{"easy", "medium", "hard"}, item.Difficulty) {
			return fmt.Errorf("query %q has invalid difficulty %q", item.ID, item.Difficulty)
		}
		if strings.TrimSpace(item.Category) == "" {
			return fmt.Errorf("query %q requires category", item.ID)
		}
		if item.Expected.Unanswerable {
			if len(item.Expected.Evidence) > 0 {
				return fmt.Errorf("unanswerable query %q cannot define evidence", item.ID)
			}
			if !hasExpectedAnswer(item.Expected.Answer) {
				return fmt.Errorf("unanswerable query %q requires expected.answer", item.ID)
			}
			continue
		}
		if len(item.Expected.Evidence) == 0 {
			return fmt.Errorf("query %q requires expected.evidence", item.ID)
		}
		for evidenceIndex, evidence := range item.Expected.Evidence {
			if strings.TrimSpace(evidence.Source.RelativePath) == "" || strings.TrimSpace(evidence.Source.Title) == "" {
				return fmt.Errorf("query %q evidence %d requires source.relative_path and source.title", item.ID, evidenceIndex+1)
			}
			if len(nonBlankStrings(evidence.AnyOf))+len(nonBlankStrings(evidence.AllOf)) == 0 {
				return fmt.Errorf("query %q evidence %d requires any_of or all_of", item.ID, evidenceIndex+1)
			}
		}
	}
	return nil
}

func selectCases(cases []evalCase, rawIDs string) ([]evalCase, error) {
	rawIDs = strings.TrimSpace(rawIDs)
	if rawIDs == "" {
		return cases, nil
	}
	wanted := map[string]bool{}
	for _, id := range strings.Split(rawIDs, ",") {
		id = strings.TrimSpace(id)
		if id != "" {
			wanted[id] = true
		}
	}
	selected := make([]evalCase, 0, len(wanted))
	for _, item := range cases {
		if wanted[item.ID] {
			selected = append(selected, item)
			delete(wanted, item.ID)
		}
	}
	if len(wanted) > 0 {
		unknown := make([]string, 0, len(wanted))
		for id := range wanted {
			unknown = append(unknown, id)
		}
		return nil, fmt.Errorf("unknown case IDs: %s", strings.Join(unknown, ", "))
	}
	return selected, nil
}

func parseCandidateCaps(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	seen := map[int]bool{}
	caps := []int{}
	for _, part := range strings.Split(raw, ",") {
		capValue, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || capValue <= 0 || capValue > 150 {
			return nil, fmt.Errorf("candidate cap %q must be between 1 and 150", strings.TrimSpace(part))
		}
		if !seen[capValue] {
			seen[capValue] = true
			caps = append(caps, capValue)
		}
	}
	return caps, nil
}

func parseFinalLimits(raw string, fallback int) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return []int{fallback}, nil
	}
	limits, err := parseCandidateCaps(raw)
	if err != nil {
		return nil, err
	}
	for _, limit := range limits {
		if limit > 50 {
			return nil, fmt.Errorf("final passage limit %d must not exceed 50", limit)
		}
	}
	return limits, nil
}

func runCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase, finalLimit int) evaluationResult {
	started := time.Now()
	if item.Expected.Unanswerable {
		return evaluationResult{Case: item, Skipped: true, Strategy: "pipeline", FinalLimit: finalLimit}
	}
	request := retrieveRequest{
		Query:          item.Query,
		Scope:          requestScope(defaults, item),
		Collection:     requestCollection(defaults, item),
		Mode:           requestMode(defaults, item),
		Limit:          finalLimit,
		CandidateLimit: 50,
		TraceStages:    true,
	}
	var decoded retrieveResponse
	if err := postJSON(client, baseURL+"/v1/retrieve", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, Elapsed: time.Since(started), Err: err, Strategy: "pipeline", FinalLimit: finalLimit}
	}
	result := evaluateResults(item, 0, decoded.Sources, time.Since(started))
	result.Strategy = "pipeline"
	result.FinalLimit = finalLimit
	candidateResult := evaluateResults(item, 50, decoded.Candidates, result.Elapsed)
	result.CandidateDocumentRank = candidateResult.DocumentRank
	result.CandidateEvidenceRank = candidateResult.EvidenceRank
	result.setDiagnostics(decoded)
	result.setStageRanks(decoded)
	return result
}

func runCandidateCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase, cap, topK int) evaluationResult {
	started := time.Now()
	if item.Expected.Unanswerable {
		return evaluationResult{Case: item, Cap: cap, Skipped: true, Strategy: "first_phase"}
	}
	request := retrieveRequest{
		Query:          item.Query,
		Scope:          requestScope(defaults, item),
		Collection:     requestCollection(defaults, item),
		Mode:           requestMode(defaults, item),
		CandidateLimit: cap,
		CandidateOnly:  true,
		TraceStages:    true,
	}
	var decoded retrieveResponse
	if err := postJSON(client, baseURL+"/v1/retrieve", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, Cap: cap, Elapsed: time.Since(started), Err: err, Strategy: "first_phase"}
	}
	evaluationCandidates := decoded.Candidates
	if topK > 0 && topK < len(evaluationCandidates) {
		evaluationCandidates = evaluationCandidates[:topK]
	}
	result := evaluateResults(item, minPositive(topK, cap), evaluationCandidates, time.Since(started))
	result.Strategy = "first_phase"
	result.CandidateDocumentRank = result.DocumentRank
	result.CandidateEvidenceRank = result.EvidenceRank
	result.setDiagnostics(decoded)
	result.setStageRanks(decoded)
	return result
}

func (result *evaluationResult) setDiagnostics(response retrieveResponse) {
	result.RetrievedCandidates = response.Diagnostics.RetrievedCandidateCount
	result.GlobalCandidates = response.Diagnostics.GlobalCandidateCount
	result.WindowMatches = response.Diagnostics.WindowMatchedCount
	result.WindowExpandedLeaves = response.Diagnostics.WindowExpandedLeafCount
	result.WindowCandidates = response.Diagnostics.WindowCandidateCount
	result.ConditionedCandidates = response.Diagnostics.ConditionedCandidateCount
	result.SelectedCandidates = response.Diagnostics.CandidateCount
	result.SelectedDocuments = response.Diagnostics.SelectedDocumentCount
	result.GlobalSearchMs = response.Diagnostics.GlobalSearchMs
	result.WindowSearchMs = response.Diagnostics.WindowSearchMs
	result.ConditionedSearchMs = response.Diagnostics.ConditionedSearchMs
	result.SearchMs = response.Diagnostics.SearchMs
	result.RerankMs = response.Diagnostics.RerankMs
	result.PipelineMs = response.Diagnostics.TotalMs
}

func (result *evaluationResult) setStageRanks(response retrieveResponse) {
	if response.StageTrace == nil {
		return
	}
	byID := make(map[string]searchResult, len(response.StageTrace.Candidates))
	for _, candidate := range response.StageTrace.Candidates {
		byID[candidate.ID] = searchResult{
			Title:     candidate.Title,
			Text:      candidate.Text,
			SourceURI: candidate.SourceURI,
		}
	}
	for _, stage := range response.StageTrace.Stages {
		candidates := make([]searchResult, 0, len(stage.CandidateIDs))
		for _, id := range stage.CandidateIDs {
			if candidate, ok := byID[id]; ok {
				candidates = append(candidates, candidate)
			}
		}
		evaluated := evaluateResults(result.Case, 0, candidates, 0)
		result.StageRanks = append(result.StageRanks, evaluationStageRank{
			Name:         stage.Name,
			DocumentRank: evaluated.DocumentRank,
			EvidenceRank: evaluated.EvidenceRank,
		})
	}
}

func evaluateResults(item evalCase, cap int, candidates []searchResult, elapsed time.Duration) evaluationResult {
	result := evaluationResult{Case: item, Cap: cap, Elapsed: elapsed}
	for _, evidence := range item.Expected.Evidence {
		documentRank, evidenceRank := evaluateEvidenceGroup(candidates, evidence)
		if documentRank > 0 {
			result.MatchedDocumentGroups++
			result.DocumentRank = maxInt(result.DocumentRank, documentRank)
		}
		if evidenceRank > 0 {
			result.MatchedEvidenceGroups++
			result.EvidenceRank = maxInt(result.EvidenceRank, evidenceRank)
		}
	}
	if result.MatchedDocumentGroups != len(item.Expected.Evidence) {
		result.DocumentRank = 0
	}
	if result.MatchedEvidenceGroups != len(item.Expected.Evidence) {
		result.EvidenceRank = 0
	}
	return result
}

func evaluateEvidenceGroup(candidates []searchResult, expected expectedEvidence) (int, int) {
	documentRank := 0
	anyRank := 0
	allRank := 0
	remaining := nonBlankStrings(expected.AllOf)
	for index, candidate := range candidates {
		if !matchesSource(candidate, expected.Source) {
			continue
		}
		rank := index + 1
		if documentRank == 0 {
			documentRank = rank
		}
		if anyRank == 0 && containsAnyFold(candidate.Text, expected.AnyOf) {
			anyRank = rank
		}
		for fragmentIndex := len(remaining) - 1; fragmentIndex >= 0; fragmentIndex-- {
			if containsFold(candidate.Text, remaining[fragmentIndex]) {
				remaining = append(remaining[:fragmentIndex], remaining[fragmentIndex+1:]...)
				allRank = maxInt(allRank, rank)
			}
		}
	}
	if len(expected.AnyOf) > 0 && anyRank == 0 || len(remaining) > 0 {
		return documentRank, 0
	}
	return documentRank, maxInt(anyRank, allRank)
}

func postJSON(client *http.Client, endpoint, userKey string, payload, decoded any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Memory-User-Key", userKey)
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("request status %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	return json.Unmarshal(responseBody, decoded)
}

func matchesSource(result searchResult, expected expectedSource) bool {
	if !strings.EqualFold(strings.TrimSpace(result.Title), strings.TrimSpace(expected.Title)) {
		return false
	}
	relativePath := strings.Trim(strings.TrimSpace(expected.RelativePath), "/")
	if relativePath == "" {
		return true
	}
	parsed, err := url.Parse(result.SourceURI)
	if err != nil {
		return false
	}
	actualPath, err := url.PathUnescape(parsed.Path)
	if err != nil {
		return false
	}
	actualPath = path.Clean(actualPath)
	return strings.HasSuffix(actualPath, "/"+relativePath)
}

func containsFold(value, fragment string) bool {
	return strings.Contains(normalizeComparableText(value), normalizeComparableText(fragment))
}

func normalizeComparableText(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}

func containsAnyFold(value string, fragments []string) bool {
	for _, fragment := range nonBlankStrings(fragments) {
		if containsFold(value, fragment) {
			return true
		}
	}
	return false
}

func containsAllFold(value string, fragments []string) bool {
	for _, fragment := range nonBlankStrings(fragments) {
		if !containsFold(value, fragment) {
			return false
		}
	}
	return true
}

func hasExpectedAnswer(expected expectedAnswer) bool {
	return len(nonBlankStrings(expected.AnyOf))+len(nonBlankStrings(expected.AllOf)) > 0
}

func matchesExpectedAnswer(value string, expected expectedAnswer) bool {
	if len(nonBlankStrings(expected.AnyOf)) > 0 && !containsAnyFold(value, expected.AnyOf) {
		return false
	}
	return containsAllFold(value, expected.AllOf)
}

func nonBlankStrings(values []string) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func requestScope(defaults suiteDefaults, item evalCase) string {
	return firstNonBlank(item.Request.Scope, defaults.Scope)
}

func requestCollection(defaults suiteDefaults, item evalCase) string {
	return firstNonBlank(item.Request.Collection, defaults.Collection)
}

func requestMode(defaults suiteDefaults, item evalCase) string {
	return firstNonBlank(item.Request.Mode, defaults.Mode)
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func minPositive(value, fallback int) int {
	if value > 0 && value < fallback {
		return value
	}
	return fallback
}

func printCaseResult(result evaluationResult) {
	prefix := ""
	if result.Strategy != "" {
		prefix = fmt.Sprintf("strategy=%s ", result.Strategy)
	}
	if result.FinalLimit > 0 {
		prefix += fmt.Sprintf("final_limit=%d ", result.FinalLimit)
	}
	if result.Cap > 0 {
		prefix += fmt.Sprintf("cap=%d ", result.Cap)
	}
	if result.Err != nil {
		fmt.Printf("FAIL %sid=%s difficulty=%s error=%q elapsed_ms=%d\n", prefix, result.Case.ID, result.Case.Difficulty, result.Err.Error(), result.Elapsed.Milliseconds())
		return
	}
	if result.Skipped {
		fmt.Printf("SKIP %sid=%s difficulty=%s reason=answer_only\n", prefix, result.Case.ID, result.Case.Difficulty)
		return
	}
	if result.AnswerTested {
		fmt.Printf("%sid=%s difficulty=%s answer_hit=%t elapsed_ms=%d\n", prefix, result.Case.ID, result.Case.Difficulty, result.AnswerHit, result.Elapsed.Milliseconds())
		return
	}
	fmt.Printf(
		"%sid=%s difficulty=%s candidate_document_rank=%d candidate_evidence_rank=%d document_rank=%d evidence_rank=%d document_groups=%d/%d evidence_groups=%d/%d global_candidates=%d window_matches=%d window_expanded_leaves=%d window_candidates=%d documents=%d conditioned_candidates=%d selected=%d global_search_ms=%d window_search_ms=%d conditioned_search_ms=%d search_ms=%d rerank_ms=%d pipeline_ms=%d elapsed_ms=%d\n",
		prefix,
		result.Case.ID,
		result.Case.Difficulty,
		result.CandidateDocumentRank,
		result.CandidateEvidenceRank,
		result.DocumentRank,
		result.EvidenceRank,
		result.MatchedDocumentGroups,
		len(result.Case.Expected.Evidence),
		result.MatchedEvidenceGroups,
		len(result.Case.Expected.Evidence),
		result.GlobalCandidates,
		result.WindowMatches,
		result.WindowExpandedLeaves,
		result.WindowCandidates,
		result.SelectedDocuments,
		result.ConditionedCandidates,
		result.SelectedCandidates,
		result.GlobalSearchMs,
		result.WindowSearchMs,
		result.ConditionedSearchMs,
		result.SearchMs,
		result.RerankMs,
		result.PipelineMs,
		result.Elapsed.Milliseconds(),
	)
	if len(result.StageRanks) > 0 {
		fmt.Printf("stages strategy=%s id=%s", result.Strategy, result.Case.ID)
		for _, stage := range result.StageRanks {
			fmt.Printf(" %s=%d/%d", stage.Name, stage.DocumentRank, stage.EvidenceRank)
		}
		fmt.Println()
	}
}

func printSummary(results []evaluationResult, cap int, strategy string) {
	var eligible, completed, documentHits, evidenceHits int
	globalSearchTimes := []int64{}
	windowSearchTimes := []int64{}
	conditionedSearchTimes := []int64{}
	searchTimes := []int64{}
	rerankTimes := []int64{}
	pipelineTimes := []int64{}
	byDifficulty := map[string][]evaluationResult{}
	for _, result := range results {
		if result.Skipped {
			continue
		}
		eligible++
		byDifficulty[result.Case.Difficulty] = append(byDifficulty[result.Case.Difficulty], result)
		if result.Err != nil {
			continue
		}
		completed++
		globalSearchTimes = append(globalSearchTimes, result.GlobalSearchMs)
		windowSearchTimes = append(windowSearchTimes, result.WindowSearchMs)
		conditionedSearchTimes = append(conditionedSearchTimes, result.ConditionedSearchMs)
		searchTimes = append(searchTimes, result.SearchMs)
		rerankTimes = append(rerankTimes, result.RerankMs)
		pipelineTimes = append(pipelineTimes, result.PipelineMs)
		if result.DocumentRank > 0 {
			documentHits++
		}
		if result.EvidenceRank > 0 {
			evidenceHits++
		}
	}
	fmt.Printf("summary strategy=%s k=%d cases=%d eligible=%d completed=%d document_recall_at_k=%d/%d evidence_recall_at_k=%d/%d\n", strategy, cap, len(results), eligible, completed, documentHits, eligible, evidenceHits, eligible)
	for _, difficulty := range []string{"hard", "medium", "easy"} {
		group := byDifficulty[difficulty]
		if len(group) == 0 {
			continue
		}
		var documents, evidence int
		for _, result := range group {
			if result.DocumentRank > 0 {
				documents++
			}
			if result.EvidenceRank > 0 {
				evidence++
			}
		}
		fmt.Printf("summary strategy=%s k=%d difficulty=%s cases=%d document_recall_at_k=%d/%d evidence_recall_at_k=%d/%d\n", strategy, cap, difficulty, len(group), documents, len(group), evidence, len(group))
	}
	if completed > 0 {
		fmt.Printf(
			"latency_summary strategy=%s k=%d global_search_avg_ms=%d window_search_avg_ms=%d conditioned_search_avg_ms=%d search_avg_ms=%d rerank_avg_ms=%d pipeline_avg_ms=%d pipeline_p50_ms=%d pipeline_p95_ms=%d\n",
			strategy,
			cap,
			averageInt64(globalSearchTimes),
			averageInt64(windowSearchTimes),
			averageInt64(conditionedSearchTimes),
			averageInt64(searchTimes),
			averageInt64(rerankTimes),
			averageInt64(pipelineTimes),
			percentileInt64(pipelineTimes, 0.50),
			percentileInt64(pipelineTimes, 0.95),
		)
	}
	printStageSummaries(results, strategy)
}

func averageInt64(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return total / int64(len(values))
}

func percentileInt64(values []int64, percentile float64) int64 {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	index := int(math.Ceil(percentile*float64(len(ordered)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(ordered) {
		index = len(ordered) - 1
	}
	return ordered[index]
}

func printStageSummaries(results []evaluationResult, strategy string) {
	type stageSummary struct {
		cases     int
		documents int
		evidence  int
	}
	order := []string{}
	byStage := map[string]*stageSummary{}
	for _, result := range results {
		if result.Err != nil || result.Skipped {
			continue
		}
		for _, rank := range result.StageRanks {
			summary := byStage[rank.Name]
			if summary == nil {
				summary = &stageSummary{}
				byStage[rank.Name] = summary
				order = append(order, rank.Name)
			}
			summary.cases++
			if rank.DocumentRank > 0 {
				summary.documents++
			}
			if rank.EvidenceRank > 0 {
				summary.evidence++
			}
		}
	}
	for _, name := range order {
		summary := byStage[name]
		fmt.Printf(
			"stage_summary strategy=%s stage=%s cases=%d document_recall=%d/%d evidence_recall=%d/%d\n",
			strategy,
			name,
			summary.cases,
			summary.documents,
			summary.cases,
			summary.evidence,
			summary.cases,
		)
	}
}

func hasFailures(results []evaluationResult) bool {
	for _, result := range results {
		if result.Skipped {
			continue
		}
		if result.Err != nil || result.AnswerTested && !result.AnswerHit || !result.AnswerTested && (result.DocumentRank == 0 || result.EvidenceRank == 0) {
			return true
		}
	}
	return false
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
