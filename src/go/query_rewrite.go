package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/lib/pq"
)

const (
	queryRewriteCatalogLimit = 8
	queryRewriteMaxRunes     = 320
)

type queryRewriteResult struct {
	Query        string
	Status       string
	Model        string
	Ms           int64
	SourceTitles []string
}

type queryRewriteTitle struct {
	Title string
	Score int
}

func (s *Server) rewriteRetrievalQuery(ctx context.Context, userKey, requestID, query, scope string) queryRewriteResult {
	start := time.Now()
	titles, err := s.queryRewriteTitles(ctx, userKey, query, scope)
	if err != nil {
		result := queryRewriteResult{Status: "catalog_error", Model: s.queryRewriterModel}
		result.Status = "catalog_error"
		result.Ms = time.Since(start).Milliseconds()
		log.Printf("memory.query_rewrite request_id=%q status=error phase=catalog query=%q ms=%d err=%q", requestID, query, result.Ms, err.Error())
		return result
	}
	return s.rewriteRetrievalQueryForTitles(ctx, requestID, query, titles)
}

func (s *Server) rewriteRetrievalQueryForTitles(ctx context.Context, requestID, query string, titles []string) queryRewriteResult {
	start := time.Now()
	result := queryRewriteResult{Status: "skipped", Model: s.queryRewriterModel, SourceTitles: titles}
	if strings.TrimSpace(s.queryRewriterURL) == "" {
		result.Status = "disabled"
		return result
	}
	if len(titles) == 0 {
		result.Status = "no_catalog_match"
		result.Ms = time.Since(start).Milliseconds()
		return result
	}
	model, err := s.resolveCompletionModel(ctx, s.queryRewriterURL, s.queryRewriterModel)
	if err != nil {
		result.Status = "model_error"
		result.Ms = time.Since(start).Milliseconds()
		log.Printf("memory.query_rewrite request_id=%q status=error phase=model_discovery query=%q ms=%d err=%q", requestID, query, result.Ms, err.Error())
		return result
	}
	result.Model = model
	prompt := strings.Builder{}
	prompt.WriteString("Rewrite the request as one precise query for searching stored documents and memories.\n")
	prompt.WriteString("Expand all abbreviations into full words; do not leave abbreviations in the output. Preserve every concept, relationship, name, and constraint in the request, and clarify references only when supported by the candidate titles.\n")
	prompt.WriteString("Candidate titles are hints, not text to repeat. Never list or combine titles; when more than one title may apply, preserve the user's broader reference.\n")
	prompt.WriteString("Use an exact candidate title only when the request clearly refers to it. Do not answer the request or introduce facts.\n")
	prompt.WriteString("Return exactly one plain-text search query with no label, explanation, JSON, markdown, or quotation marks.\n\n")
	prompt.WriteString("Request: ")
	prompt.WriteString(strings.TrimSpace(query))
	prompt.WriteString("\n\nCandidate titles:\n")
	for _, title := range titles {
		prompt.WriteString("- ")
		prompt.WriteString(title)
		prompt.WriteByte('\n')
	}

	body, err := json.Marshal(contextualizerRequest{
		Model: model,
		Messages: []contextualizerMessage{
			{Role: "system", Content: "You rewrite retrieval queries. Return one plain-text search query only."},
			{Role: "user", Content: prompt.String()},
		},
		Temperature: 0,
		MaxTokens:   96,
	})
	if err != nil {
		result.Status = "request_error"
		result.Ms = time.Since(start).Milliseconds()
		return result
	}
	requestCtx, cancel := context.WithTimeout(ctx, s.queryRewriterTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, strings.TrimRight(s.queryRewriterURL, "/"), bytes.NewReader(body))
	if err != nil {
		result.Status = "request_error"
		result.Ms = time.Since(start).Milliseconds()
		return result
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := s.httpClient.Do(request)
	if err != nil {
		result.Status = "model_error"
		result.Ms = time.Since(start).Milliseconds()
		log.Printf("memory.query_rewrite request_id=%q status=error phase=model query=%q candidates=%q ms=%d err=%q", requestID, query, titles, result.Ms, err.Error())
		return result
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil || response.StatusCode < 200 || response.StatusCode >= 300 {
		result.Status = "model_error"
		result.Ms = time.Since(start).Milliseconds()
		log.Printf("memory.query_rewrite request_id=%q status=error phase=response query=%q candidates=%q http_status=%d ms=%d", requestID, query, titles, response.StatusCode, result.Ms)
		return result
	}
	var decoded contextualizerResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil || len(decoded.Choices) == 0 {
		result.Status = "parse_error"
		result.Ms = time.Since(start).Milliseconds()
		return result
	}

	rewrite := cleanQueryRewrite(decoded.Choices[0].Message.Content)
	if !validQueryRewrite(query, rewrite, titles) {
		result.Status = "rejected"
		result.Ms = time.Since(start).Milliseconds()
		log.Printf("memory.query_rewrite request_id=%q status=rejected query=%q rewrite=%q candidates=%q model=%q ms=%d", requestID, query, rewrite, titles, result.Model, result.Ms)
		return result
	}
	result.Query = rewrite
	result.Status = "accepted"
	result.Ms = time.Since(start).Milliseconds()
	log.Printf("memory.query_rewrite request_id=%q status=accepted query=%q rewrite=%q candidates=%q model=%q ms=%d", requestID, query, rewrite, titles, result.Model, result.Ms)
	return result
}

func (s *Server) queryRewriteTitles(ctx context.Context, userKey, query, scope string) ([]string, error) {
	terms := queryRewriteTerms(query)
	if len(terms) == 0 {
		return nil, nil
	}
	patterns := make([]string, 0, len(terms))
	for _, term := range terms {
		patterns = append(patterns, "%"+term+"%")
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT DISTINCT title
		FROM memory_items
		WHERE user_key = $1
		  AND source_type = 'watch_file'
		  AND ($2 = '' OR metadata->>'scope' = $2)
		  AND (LOWER(title) LIKE ANY($3::text[]) OR LOWER(source_uri) LIKE ANY($3::text[]))
		LIMIT 64
	`, userKey, normalizeMetadataFilter(scope), pq.Array(patterns))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	candidates := make([]queryRewriteTitle, 0)
	for rows.Next() {
		var title string
		if err := rows.Scan(&title); err != nil {
			return nil, err
		}
		titleTerms := tokenizeTermSet(title)
		score := 0
		for _, term := range terms {
			if titleTerms[term] || strings.Contains(strings.ToLower(title), term) {
				score++
			}
		}
		if score > 0 {
			candidates = append(candidates, queryRewriteTitle{Title: title, Score: score})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			return candidates[i].Title < candidates[j].Title
		}
		return candidates[i].Score > candidates[j].Score
	})
	titles := make([]string, 0, minInt(queryRewriteCatalogLimit, len(candidates)))
	for _, candidate := range candidates {
		titles = append(titles, candidate.Title)
		if len(titles) >= queryRewriteCatalogLimit {
			break
		}
	}
	return titles, nil
}

func queryRewriteTerms(query string) []string {
	terms := make([]string, 0)
	for _, term := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}) {
		if len([]rune(term)) < 3 || retrievalQueryStopword(term) || genericQueryRewriteTerm(term) {
			continue
		}
		terms = append(terms, term)
	}
	return dedupeStrings(terms, 8)
}

func genericQueryRewriteTerm(term string) bool {
	switch term {
	case "book", "books", "chapter", "chapters", "section", "sections", "series", "textbook", "textbooks", "trilogy", "trilogies", "contents":
		return true
	default:
		return false
	}
}

func cleanQueryRewrite(raw string) string {
	cleaned := cleanParentSummary(raw)
	for _, prefix := range []string{"search query:", "query:", "rewritten query:"} {
		if strings.HasPrefix(strings.ToLower(cleaned), prefix) {
			cleaned = strings.TrimSpace(cleaned[len(prefix):])
		}
	}
	cleaned = strings.Trim(cleaned, "\"'` ")
	return strings.Join(strings.Fields(cleaned), " ")
}

func validQueryRewrite(original, rewrite string, titles []string) bool {
	if rewrite == "" || normalizeQueryEchoText(rewrite) == normalizeQueryEchoText(original) {
		return false
	}
	if len([]rune(rewrite)) > queryRewriteMaxRunes || strings.ContainsAny(rewrite, "{}[]") {
		return false
	}
	if !preservesQueryIntent(original, rewrite) {
		return false
	}
	if !addsRetrievalMeaning(original, rewrite) {
		return false
	}
	return matchedCandidateTitleCount(rewrite, titles) <= 1
}

func addsRetrievalMeaning(original, rewrite string) bool {
	originalTerms := tokenizeTermSet(original)
	for term := range tokenizeTermSet(rewrite) {
		if !originalTerms[term] && !retrievalQueryStopword(term) {
			return true
		}
	}
	return false
}

func preservesQueryIntent(original, rewrite string) bool {
	originalTerms := queryRewriteTerms(original)
	if len(originalTerms) == 0 {
		return false
	}
	rewriteTerms := tokenizeTermSet(rewrite)
	matched := 0
	for _, term := range originalTerms {
		if rewriteTerms[term] {
			matched++
		}
	}
	return matched*4 >= len(originalTerms)*3
}

func matchedCandidateTitleCount(rewrite string, titles []string) int {
	normalizedRewrite := normalizeQueryEchoText(rewrite)
	matches := 0
	for _, title := range titles {
		normalizedTitle := normalizeQueryEchoText(strings.TrimSuffix(title, ".pdf"))
		if normalizedTitle != "" && strings.Contains(normalizedRewrite, normalizedTitle) {
			matches++
		}
	}
	return matches
}

func containsRetrievalSpec(specs []RetrievalSpec, target RetrievalSpec) bool {
	key := strings.ToLower(strings.TrimSpace(target.Mode) + ":" + strings.TrimSpace(target.Query))
	for _, spec := range specs {
		if strings.ToLower(strings.TrimSpace(spec.Mode)+":"+strings.TrimSpace(spec.Query)) == key {
			return true
		}
	}
	return false
}

func mergeRetrieveCandidates(groups ...[]CandidateSource) []CandidateSource {
	byID := make(map[string]*CandidateSource)
	for _, group := range groups {
		for _, candidate := range group {
			if existing := byID[candidate.ID]; existing != nil {
				existing.RetrievalScore += candidate.RetrievalScore
				existing.Retrievals = append(existing.Retrievals, candidate.Retrievals...)
				continue
			}
			copy := candidate
			byID[candidate.ID] = &copy
		}
	}
	merged := make([]CandidateSource, 0, len(byID))
	for _, candidate := range byID {
		merged = append(merged, *candidate)
	}
	sortRetrieveCandidates(merged)
	return merged
}
