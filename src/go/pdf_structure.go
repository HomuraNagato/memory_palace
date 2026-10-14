package main

import (
	"context"
	"encoding/xml"
	"fmt"
	"log"
	"math"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

const (
	pdfHeaderFooterBand  = 0.09
	pdfRepeatedPageShare = 0.40
	pdfMaxHeadingRunes   = 180
)

type documentStructure struct {
	Extractor       string
	Profile         string
	Pages           []documentPage
	RepeatedMargins map[string]bool
}

type documentPage struct {
	Number    int
	Width     float64
	Height    float64
	PlainText string
	Blocks    []documentBlock
}

type documentBlock struct {
	Top      float64
	Left     float64
	Width    float64
	Height   float64
	FontSize float64
	Bold     bool
	Text     string
}

type structurePathEntry struct {
	NodeType   string
	NodeKey    string
	Label      string
	Level      int
	Ordinal    string
	Confidence float64
	Method     string
	PageStart  int
	Role       string
}

type pdfXMLDocument struct {
	Fonts []pdfXMLFont `xml:"fontspec"`
	Pages []pdfXMLPage `xml:"page"`
}

type pdfXMLFont struct {
	ID     int     `xml:"id,attr"`
	Size   float64 `xml:"size,attr"`
	Family string  `xml:"family,attr"`
}

type pdfXMLPage struct {
	Number int          `xml:"number,attr"`
	Width  float64      `xml:"width,attr"`
	Height float64      `xml:"height,attr"`
	Fonts  []pdfXMLFont `xml:"fontspec"`
	Texts  []pdfXMLText `xml:"text"`
}

type pdfXMLText struct {
	Top      float64 `xml:"top,attr"`
	Left     float64 `xml:"left,attr"`
	Width    float64 `xml:"width,attr"`
	Height   float64 `xml:"height,attr"`
	Font     int     `xml:"font,attr"`
	InnerXML string  `xml:",innerxml"`
}

var (
	pdfDivisionHeading = regexp.MustCompile(`(?i)^\s*(part|book)\s+([[:alnum:]IVXLC]+)\b(?:\s*[:.\-–—]\s*|\s+)?(.*)$`)
	pdfChapterHeading  = regexp.MustCompile(`(?i)^\s*(chapter|appendix)\s+([[:alnum:]IVXLC]+)\b(?:\s*[:.\-–—]\s*|\s+)?(.*)$`)
	pdfSectionHeading  = regexp.MustCompile(`^\s*(\d+(?:\.\d+){1,3})\s*[.:\-–—]?\s+(.+)$`)
	pdfTopHeading      = regexp.MustCompile(`^\s*(\d+)\s*[.:\-–—]?\s+([[:upper:]][^.!?]{1,140})$`)
	pdfRoleHeading     = regexp.MustCompile(`(?i)^\s*(exercises?|problems?|questions?|examples?|theorems?|definitions?|proofs?)\b(?:\s+([[:alnum:].-]+))?\s*[:.]?\s*$`)
	pdfTOCLine         = regexp.MustCompile(`\.{3,}\s*[ivxlcdm\d]+\s*$`)
	pdfInlineMarkup    = regexp.MustCompile(`(?i)</?(?:a|b|i|u|em|strong|sub|sup)(?:\s+[^>]*)?>`)
)

func extractPDFDocument(ctx context.Context, path string) (string, map[string]any, *documentStructure, error) {
	body, metadata, err := extractPDFText(ctx, path)
	if err != nil {
		return "", nil, nil, err
	}
	command := exec.CommandContext(ctx, "pdftohtml", "-xml", "-hidden", "-nodrm", "-i", "-stdout", path)
	output, structuredErr := command.Output()
	if structuredErr != nil {
		log.Printf("memory.pdf_structure status=fallback path=%q phase=extract err=%q", path, structuredErr.Error())
		metadata["structure_extractor"] = "page-fallback"
		return body, metadata, nil, nil
	}
	structure, structuredErr := parsePDFStructureXML(output)
	if structuredErr != nil || len(structure.Pages) == 0 {
		log.Printf("memory.pdf_structure status=fallback path=%q phase=parse err=%q", path, errorString(structuredErr))
		metadata["structure_extractor"] = "page-fallback"
		return body, metadata, nil, nil
	}
	assignPDFPageText(&structure, body)
	metadata["structure_extractor"] = structure.Extractor
	metadata["document_profile"] = structure.Profile
	metadata["structured_pages"] = len(structure.Pages)
	return body, metadata, &structure, nil
}

func parsePDFStructureXML(raw []byte) (documentStructure, error) {
	var parsed pdfXMLDocument
	extractor := "pdftohtml-xml"
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		sanitized := pdfInlineMarkup.ReplaceAll(raw, nil)
		if sanitizedErr := xml.Unmarshal(sanitized, &parsed); sanitizedErr != nil {
			return documentStructure{}, err
		}
		extractor = "pdftohtml-xml-sanitized"
	}
	fonts := make(map[int]pdfXMLFont, len(parsed.Fonts))
	for _, font := range parsed.Fonts {
		fonts[font.ID] = font
	}
	structure := documentStructure{Extractor: extractor, Profile: "structured-prose", RepeatedMargins: map[string]bool{}}
	for _, page := range parsed.Pages {
		for _, font := range page.Fonts {
			fonts[font.ID] = font
		}
		converted := documentPage{Number: page.Number, Width: page.Width, Height: page.Height}
		for _, item := range page.Texts {
			text := strings.Join(strings.Fields(stripMarkupText(item.InnerXML)), " ")
			if text == "" {
				continue
			}
			font := fonts[item.Font]
			converted.Blocks = append(converted.Blocks, documentBlock{
				Top: item.Top, Left: item.Left, Width: item.Width, Height: item.Height,
				FontSize: font.Size, Bold: strings.Contains(strings.ToLower(item.InnerXML), "<b>"), Text: text,
			})
		}
		sort.SliceStable(converted.Blocks, func(i, j int) bool {
			if math.Abs(converted.Blocks[i].Top-converted.Blocks[j].Top) < 2 {
				return converted.Blocks[i].Left < converted.Blocks[j].Left
			}
			return converted.Blocks[i].Top < converted.Blocks[j].Top
		})
		structure.Pages = append(structure.Pages, converted)
	}
	removeRepeatedPDFMargins(&structure)
	return structure, nil
}

func removeRepeatedPDFMargins(structure *documentStructure) {
	if structure == nil || len(structure.Pages) < 3 {
		return
	}
	pageOccurrences := make(map[string]map[int]bool)
	for _, page := range structure.Pages {
		for _, block := range page.Blocks {
			if !isPDFMarginBlock(page, block) {
				continue
			}
			key := normalizeRepeatedPDFText(block.Text)
			if key == "" || isPageNumberText(key) {
				continue
			}
			if pageOccurrences[key] == nil {
				pageOccurrences[key] = make(map[int]bool)
			}
			pageOccurrences[key][page.Number] = true
		}
	}
	minimum := maxInt(3, int(math.Ceil(float64(len(structure.Pages))*pdfRepeatedPageShare)))
	for pageIndex := range structure.Pages {
		page := &structure.Pages[pageIndex]
		kept := page.Blocks[:0]
		for _, block := range page.Blocks {
			key := normalizeRepeatedPDFText(block.Text)
			if isPDFMarginBlock(*page, block) && len(pageOccurrences[key]) >= minimum {
				structure.RepeatedMargins[key] = true
				continue
			}
			kept = append(kept, block)
		}
		page.Blocks = kept
	}
}

func assignPDFPageText(structure *documentStructure, body string) {
	if structure == nil {
		return
	}
	pages := strings.Split(body, "\f")
	for index := range structure.Pages {
		if index >= len(pages) {
			break
		}
		structure.Pages[index].PlainText = strings.TrimSpace(pages[index])
	}
}

func isPDFMarginBlock(page documentPage, block documentBlock) bool {
	if page.Height <= 0 {
		return false
	}
	return block.Top <= page.Height*pdfHeaderFooterBand || block.Top+block.Height >= page.Height*(1-pdfHeaderFooterBand)
}

func normalizeRepeatedPDFText(text string) string {
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(text)))
	for index, field := range fields {
		if _, err := strconv.Atoi(strings.Trim(field, "-–—()[]")); err == nil {
			fields[index] = "#"
		}
	}
	return strings.Join(fields, " ")
}

func isPageNumberText(text string) bool {
	text = strings.Trim(text, " -–—()[]")
	_, err := strconv.Atoi(text)
	return err == nil
}

func chunksForDocumentStructure(structure documentStructure) []textChunk {
	bodyFont := medianPDFBodyFont(structure)
	path := []structurePathEntry{}
	currentRole := ""
	chunks := []textChunk{}
	for _, page := range structure.Pages {
		events := pdfPageHeadingEvents(page, bodyFont)
		plainText := page.PlainText
		if strings.TrimSpace(plainText) == "" {
			plainText = pdfBlocksAsText(page.Blocks)
		}
		lines := strings.Split(plainText, "\n")
		buffer := []string{}
		eventIndex := 0
		matchedHeading := false
		flush := func() {
			text := strings.TrimSpace(strings.Join(buffer, "\n"))
			buffer = buffer[:0]
			appendPDFStructuredChunks(&chunks, text, page.Number, structure.Extractor, path, currentRole)
		}
		for _, line := range lines {
			lineText := strings.TrimSpace(line)
			if lineText == "" {
				buffer = append(buffer, "")
				continue
			}
			if structure.RepeatedMargins[normalizeRepeatedPDFText(lineText)] {
				continue
			}
			lineMatched := false
			remainingLine := lineText
			for eventIndex < len(events) && pdfHeadingLineMatches(lineText, events[eventIndex].matchText) {
				flush()
				path, currentRole = applyPDFHeadingEvent(path, currentRole, events[eventIndex])
				remainingLine = removePDFHeadingText(remainingLine, events[eventIndex].matchText)
				eventIndex++
				matchedHeading = true
				lineMatched = true
			}
			if lineMatched && remainingLine != "" {
				buffer = append(buffer, remainingLine)
			} else if !lineMatched {
				buffer = append(buffer, line)
			}
		}
		if !matchedHeading && len(events) > 0 {
			for eventIndex < len(events) && events[eventIndex].entry.Level == 1 {
				path, currentRole = applyPDFHeadingEvent(path, currentRole, events[eventIndex])
				eventIndex++
			}
		}
		flush()
	}
	return chunks
}

func applyDocumentStructureToChunks(chunks []textChunk, structure documentStructure) []textChunk {
	if len(chunks) == 0 || len(structure.Pages) == 0 {
		return chunks
	}
	bodyFont := medianPDFBodyFont(structure)
	pageByNumber := make(map[int]documentPage, len(structure.Pages))
	for _, page := range structure.Pages {
		pageByNumber[page.Number] = page
	}
	result := append([]textChunk(nil), chunks...)
	path := []structurePathEntry{}
	currentRole := ""
	currentPage := 0
	events := []pdfHeadingEvent{}
	eventIndex := 0
	for index := range result {
		chunk := &result[index]
		pageNumber, ok := intMapValue(chunk.Metadata, "page")
		if !ok {
			continue
		}
		if pageNumber != currentPage {
			currentPage = pageNumber
			events = pdfPageHeadingEvents(pageByNumber[pageNumber], bodyFont)
			eventIndex = 0
		}
		matched := false
		for searchIndex := eventIndex; searchIndex < len(events); searchIndex++ {
			if !pdfHeadingLineMatches(chunk.Text, events[searchIndex].matchText) {
				continue
			}
			for eventIndex <= searchIndex {
				path, currentRole = applyPDFHeadingEvent(path, currentRole, events[eventIndex])
				eventIndex++
			}
			matched = true
			break
		}
		if !matched && eventIndex == 0 {
			for eventIndex < len(events) && events[eventIndex].entry.Level == 1 {
				path, currentRole = applyPDFHeadingEvent(path, currentRole, events[eventIndex])
				eventIndex++
			}
		}
		annotatePDFChunkStructure(chunk, structure.Extractor, path, currentRole)
	}
	return result
}

func annotatePDFChunkStructure(chunk *textChunk, extractor string, path []structurePathEntry, role string) {
	chunk.Metadata["structure_extractor"] = extractor
	if len(path) > 0 {
		chunk.Metadata["structure_path"] = structurePathMetadata(path)
		chunk.Metadata["heading"] = path[len(path)-1].Label
		chunk.Metadata["structure_confidence"] = path[len(path)-1].Confidence
	}
	if role != "" {
		chunk.Metadata["semantic_role"] = role
	}
}

type pdfHeadingEvent struct {
	entry        structurePathEntry
	matchText    string
	continuation bool
}

func pdfPageHeadingEvents(page documentPage, bodyFont float64) []pdfHeadingEvent {
	events := []pdfHeadingEvent{}
	for _, block := range page.Blocks {
		heading, ok := classifyPDFHeading(block, bodyFont, page.Number)
		if !ok {
			continue
		}
		event := pdfHeadingEvent{entry: heading, matchText: heading.Label}
		if len(events) > 0 && isPDFHeadingTitleContinuation(events[len(events)-1].entry, heading) {
			event.continuation = true
		}
		events = append(events, event)
	}
	return events
}

func applyPDFHeadingEvent(path []structurePathEntry, currentRole string, event pdfHeadingEvent) ([]structurePathEntry, string) {
	if event.continuation && len(path) > 0 {
		path[len(path)-1].Label += ": " + event.entry.Label
		return path, currentRole
	}
	path = updateStructurePath(path, event.entry)
	return path, event.entry.Role
}

func appendPDFStructuredChunks(chunks *[]textChunk, text string, page int, extractor string, path []structurePathEntry, role string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	for _, chunk := range chunkTextWithMetadata(text) {
		chunk.Metadata["page"] = page
		chunk.Metadata["page_start"] = page
		chunk.Metadata["page_end"] = page
		chunk.Metadata["structure_extractor"] = extractor
		if len(path) > 0 {
			chunk.Metadata["structure_path"] = structurePathMetadata(path)
			chunk.Metadata["heading"] = path[len(path)-1].Label
			chunk.Metadata["structure_confidence"] = path[len(path)-1].Confidence
		}
		if role != "" {
			chunk.Metadata["semantic_role"] = role
		}
		chunk.Metadata["chunk_index"] = len(*chunks)
		*chunks = append(*chunks, chunk)
	}
}

func pdfBlocksAsText(blocks []documentBlock) string {
	lines := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if text := strings.TrimSpace(block.Text); text != "" {
			lines = append(lines, text)
		}
	}
	return strings.Join(lines, "\n\n")
}

func pdfHeadingLineMatches(line, heading string) bool {
	line = normalizePDFHeadingText(line)
	heading = normalizePDFHeadingText(heading)
	if line == "" || heading == "" {
		return false
	}
	return line == heading || (len([]rune(heading)) >= 8 && strings.Contains(line, heading))
}

func normalizePDFHeadingText(value string) string {
	value = strings.ToLower(strings.Join(strings.Fields(value), " "))
	return strings.Trim(value, " .:;-–—")
}

func removePDFHeadingText(line, heading string) string {
	lowerLine := strings.ToLower(line)
	lowerHeading := strings.ToLower(strings.TrimSpace(heading))
	index := strings.Index(lowerLine, lowerHeading)
	if index < 0 {
		return ""
	}
	return strings.TrimSpace(strings.TrimLeft(line[index+len(heading):], " .:;-–—"))
}

func isPDFHeadingTitleContinuation(previous, current structurePathEntry) bool {
	return previous.Level == 1 && previous.PageStart == current.PageStart &&
		previous.Method == "explicit-label" && current.Method == "layout" && current.Ordinal == ""
}

func medianPDFBodyFont(structure documentStructure) float64 {
	values := []float64{}
	for _, page := range structure.Pages {
		for _, block := range page.Blocks {
			if block.FontSize <= 0 || len([]rune(block.Text)) < 20 {
				continue
			}
			values = append(values, block.FontSize)
		}
	}
	if len(values) == 0 {
		return 12
	}
	sort.Float64s(values)
	return values[(len(values)-1)/2]
}

func mergePDFBlocks(blocks []documentBlock) []documentBlock {
	result := []documentBlock{}
	for _, block := range blocks {
		if len(result) == 0 {
			result = append(result, block)
			continue
		}
		previous := &result[len(result)-1]
		gap := block.Top - (previous.Top + previous.Height)
		similarFont := math.Abs(block.FontSize-previous.FontSize) <= 1
		similarIndent := math.Abs(block.Left-previous.Left) <= maxFloat(18, previous.FontSize*1.5)
		join := gap <= maxFloat(8, maxFloat(block.Height, previous.Height)*0.9) && similarFont && similarIndent
		if !join || likelyStandalonePDFHeading(*previous) || likelyStandalonePDFHeading(block) {
			result = append(result, block)
			continue
		}
		separator := " "
		if strings.HasSuffix(previous.Text, "-") && len(block.Text) > 0 && unicode.IsLower([]rune(block.Text)[0]) {
			previous.Text = strings.TrimSuffix(previous.Text, "-")
			separator = ""
		}
		previous.Text += separator + block.Text
		previous.Width = maxFloat(previous.Width, block.Left+block.Width-previous.Left)
		previous.Height = block.Top + block.Height - previous.Top
		previous.Bold = previous.Bold && block.Bold
	}
	return result
}

func likelyStandalonePDFHeading(block documentBlock) bool {
	text := strings.TrimSpace(block.Text)
	return pdfDivisionHeading.MatchString(text) || pdfChapterHeading.MatchString(text) || pdfSectionHeading.MatchString(text) || pdfRoleHeading.MatchString(text)
}

func classifyPDFHeading(block documentBlock, bodyFont float64, page int) (structurePathEntry, bool) {
	text := strings.Join(strings.Fields(block.Text), " ")
	if text == "" || len([]rune(text)) > pdfMaxHeadingRunes || pdfTOCLine.MatchString(text) {
		return structurePathEntry{}, false
	}
	entry := structurePathEntry{Label: text, PageStart: page, Method: "layout", Confidence: 0.62}
	headingStyle := block.Bold || (bodyFont > 0 && block.FontSize >= bodyFont*1.12)
	if match := pdfDivisionHeading.FindStringSubmatch(text); len(match) > 0 && headingStyle {
		entry.NodeType, entry.Level, entry.Ordinal, entry.Method, entry.Confidence = "division", 1, normalizeStructureOrdinal(match[2]), "explicit-label", 0.98
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	if match := pdfChapterHeading.FindStringSubmatch(text); len(match) > 0 && headingStyle {
		entry.NodeType, entry.Level, entry.Ordinal, entry.Method, entry.Confidence = "chapter", 1, normalizeStructureOrdinal(match[2]), "explicit-label", 0.98
		if strings.EqualFold(match[1], "appendix") {
			entry.NodeType = "appendix"
		}
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	if match := pdfRoleHeading.FindStringSubmatch(text); len(match) > 0 {
		entry.NodeType, entry.Level, entry.Role, entry.Method, entry.Confidence = "section", 2, normalizeSemanticRole(match[1]), "explicit-role", 0.94
		entry.Ordinal = match[2]
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	if match := pdfSectionHeading.FindStringSubmatch(text); len(match) > 0 && headingStyle {
		entry.Ordinal = match[1]
		entry.Level = minInt(4, strings.Count(match[1], ".")+1)
		entry.NodeType = structureNodeTypeForLevel(entry.Level)
		entry.Method, entry.Confidence = "numbering", 0.88
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	large := bodyFont > 0 && block.FontSize >= bodyFont*1.18
	if match := pdfTopHeading.FindStringSubmatch(text); len(match) > 0 && (large || block.Bold) {
		entry.NodeType, entry.Level, entry.Ordinal, entry.Method, entry.Confidence = "chapter", 1, match[1], "numbering-layout", 0.80
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	if large && len([]rune(text)) <= 100 && !strings.HasSuffix(text, ".") {
		entry.NodeType, entry.Level = "section", 2
		entry.NodeKey = structureNodeKey(entry)
		return entry, true
	}
	return structurePathEntry{}, false
}

func updateStructurePath(path []structurePathEntry, entry structurePathEntry) []structurePathEntry {
	for len(path) > 0 && path[len(path)-1].Level >= entry.Level {
		path = path[:len(path)-1]
	}
	return append(path, entry)
}

func structureNodeTypeForLevel(level int) string {
	switch level {
	case 1:
		return "chapter"
	case 2:
		return "section"
	default:
		return "subsection"
	}
}

func structureNodeKey(entry structurePathEntry) string {
	key := entry.Ordinal
	if key == "" {
		key = strings.ToLower(strings.Map(func(r rune) rune {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				return r
			}
			return '-'
		}, entry.Label))
		key = strings.Trim(strings.Join(strings.FieldsFunc(key, func(r rune) bool { return r == '-' }), "-"), "-")
	}
	return fmt.Sprintf("%s:%s:p%d", entry.NodeType, key, entry.PageStart)
}

func structurePathMetadata(path []structurePathEntry) []map[string]any {
	result := make([]map[string]any, 0, len(path))
	for _, entry := range path {
		result = append(result, map[string]any{
			"node_type": entry.NodeType, "node_key": entry.NodeKey, "label": entry.Label,
			"level": entry.Level, "ordinal": entry.Ordinal, "confidence": entry.Confidence,
			"method": entry.Method, "page_start": entry.PageStart, "semantic_role": entry.Role,
		})
	}
	return result
}

func normalizeSemanticRole(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.TrimSuffix(value, "s")
	if value == "exercis" {
		return "exercise"
	}
	return value
}

func normalizeStructureOrdinal(value string) string {
	value = strings.ToLower(strings.Trim(strings.TrimSpace(value), ".:;,-–—()[]"))
	words := map[string]string{
		"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
		"six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
		"eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
		"fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
		"nineteen": "19", "twenty": "20",
		"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
		"sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
		"eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
		"fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
		"nineteenth": "19", "twentieth": "20",
	}
	if normalized := words[value]; normalized != "" {
		return normalized
	}
	return value
}
