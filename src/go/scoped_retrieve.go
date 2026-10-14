package main

import (
	"context"
	"encoding/json"
	"log"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/lib/pq"
)

const (
	structuralDocumentLimit      = 3
	structuralTargetsPerDocument = 2
	structuredRerankShare        = 20
	documentRerankShare          = 10
	globalRerankShare            = 10
)

type SearchConstraints struct {
	MemoryIDs       []string
	AncestorNodeIDs []string
}

type RetrievalTarget struct {
	MemoryID   string  `json:"memory_id"`
	NodeID     string  `json:"node_id"`
	NodeType   string  `json:"node_type"`
	Label      string  `json:"label"`
	Ordinal    string  `json:"ordinal,omitempty"`
	Role       string  `json:"role,omitempty"`
	Confidence float64 `json:"confidence"`
	Evidence   string  `json:"evidence"`
}

type structuralQuery struct {
	NodeType string
	Ordinal  string
	Role     string
}

type structuredRetrieval struct {
	Candidates      []CandidateSource
	Targets         []RetrievalTarget
	DocumentCount   int
	DocumentScope   string
	RoleConstrained bool
	ElapsedMs       int64
	Fallback        string
}

var (
	structureLocatorPattern      = regexp.MustCompile(`(?i)\b(chapter|ch|section|sec|appendix|part|page)\.?\s+([[:alnum:]]+(?:\.[0-9]+)*)\b`)
	leadingOrdinalLocatorPattern = regexp.MustCompile(`(?i)\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s+(chapter|section|appendix|part|page)\b`)
	structureOrdinalPattern      = regexp.MustCompile(`^(?:[0-9]+(?:\.[0-9]+)*|[a-z]|[ivxlcdm]+)$`)
)

func parseStructuralQuery(query string) structuralQuery {
	parsed := structuralQuery{}
	if match := leadingOrdinalLocatorPattern.FindStringSubmatch(strings.TrimSpace(query)); len(match) > 0 {
		parsed.NodeType = normalizeLocatorType(match[2])
		parsed.Ordinal = normalizeStructureOrdinal(match[1])
	} else if match := structureLocatorPattern.FindStringSubmatch(strings.TrimSpace(query)); len(match) > 0 {
		ordinal := normalizeStructureOrdinal(match[2])
		if isStructureOrdinal(ordinal) {
			parsed.NodeType = normalizeLocatorType(match[1])
			parsed.Ordinal = ordinal
		}
	}
	lower := strings.ToLower(query)
	for _, role := range []string{"exercise", "problem", "question", "example", "theorem", "definition", "proof"} {
		if regexp.MustCompile(`\b` + role + `s?\b`).MatchString(lower) {
			parsed.Role = role
			break
		}
	}
	return parsed
}

func isStructureOrdinal(value string) bool {
	return structureOrdinalPattern.MatchString(value)
}

func normalizeLocatorType(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "ch":
		return "chapter"
	case "sec":
		return "section"
	case "part":
		return "division"
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func (s *Server) retrieveStructuredCandidates(
	ctx context.Context,
	userKey string,
	req RetrieveRequest,
	documents []firstPhaseDocument,
) structuredRetrieval {
	started := time.Now()
	query := parseStructuralQuery(req.Query)
	if query.NodeType == "" && query.Role == "" {
		return structuredRetrieval{}
	}
	documents, documentScope := selectStructuredDocuments(req.Query, documents)
	if len(documents) > structuralDocumentLimit {
		documents = documents[:structuralDocumentLimit]
	}
	targets, err := s.resolveRetrievalTargets(ctx, userKey, req, query, documents)
	if err != nil {
		log.Printf("memory.retrieve.structure request_id=%q status=fallback phase=resolve err=%q", req.RequestID, err.Error())
		return structuredRetrieval{DocumentCount: len(documents), DocumentScope: documentScope, ElapsedMs: time.Since(started).Milliseconds(), Fallback: "target_resolution_error"}
	}
	if len(targets) == 0 {
		return structuredRetrieval{DocumentCount: len(documents), DocumentScope: documentScope, ElapsedMs: time.Since(started).Milliseconds(), Fallback: "no_structural_target"}
	}
	routes, err := s.descendantLeafNodeRoutes(ctx, userKey, targetNodeIDs(targets))
	if err != nil || len(routes) == 0 {
		log.Printf("memory.retrieve.structure request_id=%q status=fallback phase=descendants err=%q", req.RequestID, errorString(err))
		return structuredRetrieval{Targets: targets, DocumentCount: len(documents), DocumentScope: documentScope, ElapsedMs: time.Since(started).Milliseconds(), Fallback: "no_structural_descendants"}
	}
	roleConstrained := false
	fallback := ""
	if query.Ordinal != "" && query.Role != "" {
		roleRoutes, roleErr := s.filterLeafNodeRoutesByRole(ctx, userKey, routes, query.Role)
		switch {
		case roleErr != nil:
			fallback = "role_filter_error"
			log.Printf("memory.retrieve.structure request_id=%q status=fallback phase=role_filter role=%q err=%q", req.RequestID, query.Role, roleErr.Error())
		case len(roleRoutes) == 0:
			fallback = "no_role_descendants"
		case len(roleRoutes) > 0:
			routes = roleRoutes
			roleConstrained = true
		}
	}

	type result struct {
		route string
		hits  []SearchResult
		err   error
	}
	out := make(chan result, 2)
	leafIDs := mapKeys(routes)
	if len(req.Embedding) > 0 && normalizeSearchMode(req.Mode) != searchModeLiteral {
		go func() {
			hits, denseErr := s.searchStructuredDense(ctx, userKey, req.Embedding, leafIDs, structuredRerankShare*2)
			out <- result{route: "structured_dense", hits: hits, err: denseErr}
		}()
	} else {
		out <- result{route: "structured_dense"}
	}
	if s.bm25 != nil {
		go func() {
			hits := s.bm25.SearchNodes(userKey, SearchRequest{
				Query: req.Query, RequestID: req.RequestID, TimeFilter: req.TimeFilter,
				Scope: req.Scope, Collection: req.Collection,
			}, leafIDs, 2)
			if len(hits) > structuredRerankShare*2 {
				hits = hits[:structuredRerankShare*2]
			}
			out <- result{route: "structured_literal", hits: hits}
		}()
	} else {
		out <- result{route: "structured_literal"}
	}

	targetByNode := make(map[string]RetrievalTarget, len(targets))
	for _, target := range targets {
		targetByNode[target.NodeID] = target
	}
	byID := make(map[string]*CandidateSource)
	for index := 0; index < 2; index++ {
		current := <-out
		if current.err != nil {
			log.Printf("memory.retrieve.structure request_id=%q status=fallback route=%q err=%q", req.RequestID, current.route, current.err.Error())
			continue
		}
		annotateStructuredResults(current.hits, routes, targetByNode, query.Role)
		addRetrieveCandidates(byID, current.hits, RetrievalSpec{Query: req.Query, Mode: routeSearchMode(current.route)}, current.route)
	}
	candidates := candidateSources(byID)
	logRetrievalTargets(req.RequestID, targets, len(documents), documentScope, roleConstrained, len(leafIDs), len(candidates), fallback, time.Since(started).Milliseconds())
	return structuredRetrieval{
		Candidates: candidates, Targets: targets, DocumentCount: len(documents), DocumentScope: documentScope,
		RoleConstrained: roleConstrained, ElapsedMs: time.Since(started).Milliseconds(), Fallback: fallback,
	}
}

func (s *Server) resolveRetrievalTargets(
	ctx context.Context,
	userKey string,
	req RetrieveRequest,
	query structuralQuery,
	documents []firstPhaseDocument,
) ([]RetrievalTarget, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	memoryIDs := make([]string, 0, len(documents))
	for _, document := range documents {
		memoryIDs = append(memoryIDs, document.memoryID)
	}
	roles := semanticRoleAlternatives(query.Role)
	rows, err := s.db.QueryContext(ctx, `
		WITH ranked AS (
			SELECT n.id, n.memory_id, n.node_type, n.metadata->>'label' AS label,
				n.metadata->>'ordinal' AS ordinal, n.metadata->>'semantic_role' AS role,
				COALESCE((n.metadata->>'confidence')::double precision, 0.5) AS confidence,
				CASE
					WHEN $3 <> '' AND n.node_type = $3 AND lower(COALESCE(n.metadata->>'ordinal', '')) = $4 THEN 4
					WHEN $4 <> '' AND lower(COALESCE(n.metadata->>'ordinal', '')) = $4 THEN 3
					WHEN $4 = '' AND lower(COALESCE(n.metadata->>'semantic_role', '')) = ANY($5::text[]) THEN 2
					ELSE 1
				END AS exactness,
				0.0 AS similarity,
				ROW_NUMBER() OVER (
					PARTITION BY n.memory_id
					ORDER BY
						CASE
							WHEN $3 <> '' AND n.node_type = $3 AND lower(COALESCE(n.metadata->>'ordinal', '')) = $4 THEN 4
							WHEN $4 <> '' AND lower(COALESCE(n.metadata->>'ordinal', '')) = $4 THEN 3
							WHEN $4 = '' AND lower(COALESCE(n.metadata->>'semantic_role', '')) = ANY($5::text[]) THEN 2
							ELSE 1
						END DESC,
						n.ordinal
				) AS document_rank
			FROM memory_nodes n
			WHERE n.user_key = $1
				AND n.memory_id = ANY($2::text[])
				AND n.node_type NOT IN ('document', 'page')
				AND (
					($4 <> '' AND lower(COALESCE(n.metadata->>'ordinal', '')) = $4)
					OR ($4 = '' AND lower(COALESCE(n.metadata->>'semantic_role', '')) = ANY($5::text[]))
				)
		)
		SELECT id, memory_id, node_type, label, ordinal, role, confidence, exactness, similarity
		FROM ranked
		WHERE document_rank <= $6
		ORDER BY array_position($2::text[], memory_id), exactness DESC, similarity DESC
	`, userKey, pq.Array(memoryIDs), query.NodeType, query.Ordinal, pq.Array(roles), structuralTargetsPerDocument)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	targets := []RetrievalTarget{}
	for rows.Next() {
		var target RetrievalTarget
		var exactness int
		var similarity float64
		if err := rows.Scan(&target.NodeID, &target.MemoryID, &target.NodeType, &target.Label, &target.Ordinal, &target.Role, &target.Confidence, &exactness, &similarity); err != nil {
			return nil, err
		}
		target.Confidence = minFloat(1, 0.55*target.Confidence+0.30*(float64(exactness)/4)+0.15*maxFloat(0, similarity))
		target.Evidence = "structure_metadata"
		if exactness >= 4 {
			target.Evidence = "exact_type_and_ordinal"
		} else if exactness >= 3 {
			target.Evidence = "exact_ordinal"
		} else if exactness == 2 {
			target.Evidence = "semantic_role"
		}
		targets = append(targets, target)
	}
	return targets, rows.Err()
}

func (s *Server) filterLeafNodeRoutesByRole(ctx context.Context, userKey string, routes map[string]string, role string) (map[string]string, error) {
	if len(routes) == 0 || strings.TrimSpace(role) == "" {
		return routes, nil
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id
		FROM memory_nodes
		WHERE user_key = $1
			AND id = ANY($2::text[])
			AND lower(COALESCE(metadata->>'semantic_role', '')) = ANY($3::text[])
	`, userKey, pq.Array(mapKeys(routes)), pq.Array(semanticRoleAlternatives(role)))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	filtered := make(map[string]string)
	for rows.Next() {
		var nodeID string
		if err := rows.Scan(&nodeID); err != nil {
			return nil, err
		}
		filtered[nodeID] = routes[nodeID]
	}
	return filtered, rows.Err()
}

func (s *Server) searchStructuredDense(ctx context.Context, userKey string, embedding []float64, nodeIDs []string, limit int) ([]SearchResult, error) {
	if len(embedding) == 0 || len(nodeIDs) == 0 || limit <= 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT c.memory_id, c.id, i.title, c.text, c.context_text, c.metadata_text,
			i.kind, i.source_type, i.source_uri, i.tags::text, i.metadata::text,
			(c.metadata || jsonb_build_object('node_id', c.node_id))::text,
			1 - (c.embedding <=> $2::vector) AS score,
			1 - (c.embedding <=> $2::vector) AS vector_similarity,
			0.0 AS literal_score, false AS title_match, false AS source_match,
			false AS body_match, i.updated_at
		FROM memory_chunks c
		JOIN memory_items i ON i.id = c.memory_id
		WHERE c.user_key = $1 AND c.node_id = ANY($3::text[])
		ORDER BY c.embedding <=> $2::vector
		LIMIT $4
	`, userKey, vectorLiteral(embedding), pq.Array(nodeIDs), limit)
	if err != nil {
		return nil, err
	}
	return scanSearchResults(rows, searchModeSemantic)
}

func annotateStructuredResults(results []SearchResult, routes map[string]string, targets map[string]RetrievalTarget, requestedRole string) {
	for index := range results {
		result := &results[index]
		leafID := stringMapValue(result.ChunkMetadata, "node_id")
		targetID := routes[leafID]
		target := targets[targetID]
		result.ChunkMetadata = cloneMap(result.ChunkMetadata)
		result.ChunkMetadata["structural_target_node_id"] = targetID
		result.ChunkMetadata["structural_target_label"] = target.Label
		result.ChunkMetadata["structural_target_confidence"] = target.Confidence
		if requestedRole != "" && semanticRoleMatches(stringMapValue(result.ChunkMetadata, "semantic_role"), requestedRole) {
			result.Score += 0.05
			result.ChunkMetadata["structural_role_match"] = true
		}
	}
}

func semanticRoleMatches(actual, requested string) bool {
	actual, requested = normalizeSemanticRole(actual), normalizeSemanticRole(requested)
	if actual == requested {
		return true
	}
	return (actual == "exercise" && requested == "problem") || (actual == "problem" && requested == "exercise")
}

func semanticRoleAlternatives(role string) []string {
	role = normalizeSemanticRole(role)
	if role == "problem" || role == "exercise" {
		return []string{"problem", "exercise"}
	}
	if role == "" {
		return []string{}
	}
	return []string{role}
}

func selectStructuredDocuments(query string, documents []firstPhaseDocument) ([]firstPhaseDocument, string) {
	if len(documents) <= 1 {
		return documents, "ranked"
	}
	queryTokens := documentIdentityQueryTokens(query)
	if len(queryTokens) == 0 {
		return documents, "ranked"
	}
	type scoredDocument struct {
		document firstPhaseDocument
		score    float64
	}
	documentTokens := make([]map[string]bool, len(documents))
	documentFrequency := make(map[string]int)
	for index, document := range documents {
		documentTokens[index] = firstPhaseDocumentIdentityTokens(document)
		for token := range queryTokens {
			if identityTokenMatches(token, documentTokens[index]) {
				documentFrequency[token]++
			}
		}
	}
	scored := make([]scoredDocument, 0, len(documents))
	best := 0.0
	for index, document := range documents {
		score := 0.0
		for token := range queryTokens {
			if identityTokenMatches(token, documentTokens[index]) {
				score += 1 / float64(documentFrequency[token])
			}
		}
		best = maxFloat(best, score)
		scored = append(scored, scoredDocument{document: document, score: score})
	}
	if best == 0 {
		return documents, "ranked"
	}
	selected := make([]firstPhaseDocument, 0, len(documents))
	for _, current := range scored {
		if current.score == best {
			selected = append(selected, current.document)
		}
	}
	return selected, "query_identity"
}

func documentIdentityQueryTokens(query string) map[string]bool {
	stop := map[string]bool{
		"a": true, "about": true, "an": true, "and": true, "are": true, "at": true, "before": true,
		"book": true, "chapter": true, "ch": true, "describe": true, "does": true, "example": true,
		"exercise": true, "first": true, "for": true, "from": true, "give": true, "in": true, "is": true,
		"major": true, "of": true, "part": true, "problem": true, "question": true, "section": true,
		"summarize": true, "text": true, "textbook": true, "the": true, "theorem": true, "to": true,
		"what": true, "which": true, "with": true,
	}
	result := map[string]bool{}
	for _, token := range strings.Fields(strings.ToLower(normalizeRetrievalQuery(query))) {
		if !stop[token] && !isStructureOrdinal(normalizeStructureOrdinal(token)) {
			result[token] = true
		}
	}
	return result
}

func firstPhaseDocumentIdentityTokens(document firstPhaseDocument) map[string]bool {
	result := map[string]bool{}
	for _, candidate := range document.candidates {
		for _, token := range strings.Fields(strings.ToLower(normalizeRetrievalQuery(candidate.Title + " " + candidate.SourceURI))) {
			result[token] = true
		}
	}
	return result
}

func identityTokenMatches(token string, documentTokens map[string]bool) bool {
	if documentTokens[token] {
		return true
	}
	return len(token) > 4 && strings.HasSuffix(token, "s") && documentTokens[strings.TrimSuffix(token, "s")]
}

func assembleGuardedRerankPool(global, conditioned, structured []CandidateSource, documents []firstPhaseDocument, limit int, scoreFeatures bool) []CandidateSource {
	if len(structured) == 0 {
		return assembleRerankPoolWithScoring(global, conditioned, documents, limit, scoreFeatures)
	}
	allConditioned := append(append([]CandidateSource(nil), conditioned...), structured...)
	uniqueCount := uniqueCandidateCount(global, allConditioned)
	ranked := assembleRerankPoolWithScoring(global, allConditioned, documents, uniqueCount, scoreFeatures)
	structuredIDs := candidateIDSet(structured)
	conditionedIDs := candidateIDSet(conditioned)
	globalIDs := candidateIDSet(global)
	result := make([]CandidateSource, 0, limit)
	seen := make(map[string]bool, limit)
	appendLane := func(ids map[string]bool, target int) {
		for _, candidate := range ranked {
			if len(result) >= target || seen[candidate.ID] || !ids[candidate.ID] {
				continue
			}
			seen[candidate.ID] = true
			result = append(result, candidate)
		}
	}
	structuredLimit := minInt(limit, structuredRerankShare)
	documentLimit := minInt(limit, structuredLimit+documentRerankShare)
	globalLimit := minInt(limit, documentLimit+globalRerankShare)
	appendLane(structuredIDs, structuredLimit)
	appendLane(conditionedIDs, documentLimit)
	appendLane(globalIDs, globalLimit)
	for _, candidate := range ranked {
		if len(result) >= limit {
			break
		}
		if !seen[candidate.ID] {
			seen[candidate.ID] = true
			result = append(result, candidate)
		}
	}
	return result
}

func targetNodeIDs(targets []RetrievalTarget) []string {
	ids := make([]string, 0, len(targets))
	for _, target := range targets {
		ids = append(ids, target.NodeID)
	}
	return ids
}

func mapKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func routeSearchMode(route string) string {
	if strings.HasSuffix(route, "literal") {
		return searchModeLiteral
	}
	return searchModeSemantic
}

func candidateIDSet(candidates []CandidateSource) map[string]bool {
	result := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		result[candidate.ID] = true
	}
	return result
}

func uniqueCandidateCount(groups ...[]CandidateSource) int {
	ids := map[string]bool{}
	for _, candidates := range groups {
		for _, candidate := range candidates {
			ids[candidate.ID] = true
		}
	}
	return len(ids)
}

func logRetrievalTargets(requestID string, targets []RetrievalTarget, documents int, documentScope string, roleConstrained bool, descendants, candidates int, fallback string, elapsedMs int64) {
	encoded, err := json.Marshal(targets)
	if err != nil {
		log.Printf("memory.retrieve.structure request_id=%q status=error phase=log err=%q", requestID, err.Error())
		return
	}
	log.Printf("memory.retrieve.structure request_id=%q status=ok targets=%s documents=%d document_scope=%q role_constrained=%t descendants=%d candidates=%d fallback=%q ms=%d", requestID, encoded, documents, documentScope, roleConstrained, descendants, candidates, fallback, elapsedMs)
}
