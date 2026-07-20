package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
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
	Expected   expectedCase `yaml:"expected"`
}

type expectedCase struct {
	Source           expectedSource `yaml:"source"`
	EvidenceContains string         `yaml:"evidence_contains"`
	AnswerContains   string         `yaml:"answer_contains,omitempty"`
}

type expectedSource struct {
	Scope        string `yaml:"scope"`
	Collection   string `yaml:"collection"`
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
}

type retrieveResponse struct {
	Candidates []searchResult `json:"candidates"`
	Sources    []searchResult `json:"sources"`
}

type searchResult struct {
	ChunkID   string `json:"chunk_id"`
	Title     string `json:"title"`
	Text      string `json:"text"`
	SourceURI string `json:"source_uri"`
}

type evaluationResult struct {
	Case         evalCase
	Cap          int
	DocumentRank int
	EvidenceRank int
	AnswerHit    bool
	AnswerTested bool
	Elapsed      time.Duration
	Err          error
}

func main() {
	var (
		suitePath     = flag.String("suite", defaultSuitePath, "Path to the YAML evaluation suite")
		baseURL       = flag.String("base-url", defaultBaseURL, "Memory Palace HTTP base URL")
		caseIDs       = flag.String("case", "", "Comma-separated case IDs to run; empty runs every case")
		candidateCaps = flag.String("candidate-caps", "", "Comma-separated candidate caps; uses candidate-only retrieval when set")
		beemoURL      = flag.String("beemo-url", "", "Optional Beemo UI URL for cases with expected.answer_contains")
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
	if strings.TrimSpace(*beemoURL) != "" {
		results := make([]evaluationResult, 0, len(selected))
		for _, item := range selected {
			if strings.TrimSpace(item.Expected.AnswerContains) == "" {
				continue
			}
			result := runAnswerCase(client, strings.TrimRight(*beemoURL, "/"), suite.Defaults, item)
			results = append(results, result)
			printCaseResult(result)
		}
		if len(results) == 0 {
			exitf("no selected cases define expected.answer_contains")
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
		var allResults []evaluationResult
		for _, cap := range caps {
			results := make([]evaluationResult, 0, len(selected))
			for _, item := range selected {
				result := runCandidateCase(client, strings.TrimRight(*baseURL, "/"), suite.Defaults, item, cap)
				results = append(results, result)
				printCaseResult(result)
			}
			printSummary(results, cap)
			allResults = append(allResults, results...)
		}
		if hasFailures(allResults) {
			os.Exit(1)
		}
		return
	}
	results := make([]evaluationResult, 0, len(selected))
	for _, item := range selected {
		result := runCase(client, strings.TrimRight(*baseURL, "/"), suite.Defaults, item)
		results = append(results, result)
		printCaseResult(result)
	}
	printSummary(results, suite.Defaults.ExpectedTopK)
	if hasFailures(results) {
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
			"request_id":        "memory-eval-" + item.ID,
			"memory_ask":        "true",
			"memory_user_key":   defaults.UserKey,
			"memory_search_mode": defaults.Mode,
			"memory_scope":      firstNonBlank(item.Expected.Source.Scope, defaults.Scope),
			"memory_collection": firstNonBlank(item.Expected.Source.Collection, defaults.Collection),
		},
	}
	var decoded answerResponse
	if err := postJSON(client, baseURL+"/api/chat", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, AnswerTested: true, Elapsed: time.Since(started), Err: err}
	}
	return evaluationResult{
		Case:         item,
		AnswerTested: true,
		AnswerHit:    containsFold(decoded.Text, item.Expected.AnswerContains),
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
	if loaded.Version != 2 {
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
		if strings.TrimSpace(item.Expected.Source.RelativePath) == "" || strings.TrimSpace(item.Expected.Source.Title) == "" {
			return fmt.Errorf("query %q requires expected.source.relative_path and expected.source.title", item.ID)
		}
		if strings.TrimSpace(item.Expected.EvidenceContains) == "" {
			return fmt.Errorf("query %q requires expected.evidence_contains", item.ID)
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

func runCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase) evaluationResult {
	started := time.Now()
	scope := firstNonBlank(item.Expected.Source.Scope, defaults.Scope)
	collection := firstNonBlank(item.Expected.Source.Collection, defaults.Collection)
	request := retrieveRequest{
		Query:          item.Query,
		Scope:          scope,
		Collection:     collection,
		Mode:           defaults.Mode,
		Limit:          defaults.ExpectedTopK,
		CandidateLimit: 50,
	}
	var decoded retrieveResponse
	if err := postJSON(client, baseURL+"/v1/retrieve", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, Elapsed: time.Since(started), Err: err}
	}
	return evaluateResults(item, 0, decoded.Sources, time.Since(started))
}

func runCandidateCase(client *http.Client, baseURL string, defaults suiteDefaults, item evalCase, cap int) evaluationResult {
	started := time.Now()
	request := retrieveRequest{
		Query:          item.Query,
		Scope:          firstNonBlank(item.Expected.Source.Scope, defaults.Scope),
		Collection:     firstNonBlank(item.Expected.Source.Collection, defaults.Collection),
		Mode:           defaults.Mode,
		CandidateLimit: cap,
		CandidateOnly:  true,
	}
	var decoded retrieveResponse
	if err := postJSON(client, baseURL+"/v1/retrieve", defaults.UserKey, request, &decoded); err != nil {
		return evaluationResult{Case: item, Cap: cap, Elapsed: time.Since(started), Err: err}
	}
	return evaluateResults(item, cap, decoded.Candidates, time.Since(started))
}

func evaluateResults(item evalCase, cap int, candidates []searchResult, elapsed time.Duration) evaluationResult {
	result := evaluationResult{Case: item, Cap: cap, Elapsed: elapsed}
	for index, candidate := range candidates {
		if !matchesSource(candidate, item.Expected.Source) {
			continue
		}
		if result.DocumentRank == 0 {
			result.DocumentRank = index + 1
		}
		if result.EvidenceRank == 0 && containsFold(candidate.Text, item.Expected.EvidenceContains) {
			result.EvidenceRank = index + 1
		}
	}
	return result
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
	return strings.Contains(strings.ToLower(value), strings.ToLower(strings.TrimSpace(fragment)))
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func printCaseResult(result evaluationResult) {
	prefix := ""
	if result.Cap > 0 {
		prefix = fmt.Sprintf("cap=%d ", result.Cap)
	}
	if result.Err != nil {
		fmt.Printf("FAIL %sid=%s difficulty=%s error=%q elapsed_ms=%d\n", prefix, result.Case.ID, result.Case.Difficulty, result.Err.Error(), result.Elapsed.Milliseconds())
		return
	}
	if result.AnswerTested {
		fmt.Printf("%sid=%s difficulty=%s answer_hit=%t elapsed_ms=%d\n", prefix, result.Case.ID, result.Case.Difficulty, result.AnswerHit, result.Elapsed.Milliseconds())
		return
	}
	fmt.Printf("%sid=%s difficulty=%s document_rank=%d evidence_rank=%d elapsed_ms=%d\n", prefix, result.Case.ID, result.Case.Difficulty, result.DocumentRank, result.EvidenceRank, result.Elapsed.Milliseconds())
}

func printSummary(results []evaluationResult, cap int) {
	var completed, documentHits, evidenceHits int
	byDifficulty := map[string][]evaluationResult{}
	for _, result := range results {
		byDifficulty[result.Case.Difficulty] = append(byDifficulty[result.Case.Difficulty], result)
		if result.Err != nil {
			continue
		}
		completed++
		if result.DocumentRank > 0 {
			documentHits++
		}
		if result.EvidenceRank > 0 {
			evidenceHits++
		}
	}
	fmt.Printf("summary k=%d cases=%d completed=%d document_recall_at_k=%d/%d evidence_recall_at_k=%d/%d\n", cap, len(results), completed, documentHits, len(results), evidenceHits, len(results))
	for _, difficulty := range []string{"hard", "easy"} {
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
		fmt.Printf("summary k=%d difficulty=%s cases=%d document_recall_at_k=%d/%d evidence_recall_at_k=%d/%d\n", cap, difficulty, len(group), documents, len(group), evidence, len(group))
	}
}

func hasFailures(results []evaluationResult) bool {
	for _, result := range results {
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
