# Search North Star

## Purpose

Memory Palace should retrieve the evidence needed to answer questions about a
private corpus of memories, books, and textbooks quickly, accurately, and with
traceable sources.

Google Search is a useful north star because its public design principles show
how strong retrieval systems combine indexing, several retrieval methods,
staged ranking, passage selection, and continuous evaluation. It is not a
blueprint: Google's production implementation is proprietary, its corpus and
feedback are radically different from ours, and returning a useful web page is
usually easier than answering a relationship or synthesis question from a long
book.

This document records durable principles and experiment directions. Concrete
thresholds and model choices belong in the evaluation documentation because
they will change.

## What Google Teaches Us

### Perform expensive work before the query

Search begins with offline work: discovering content, parsing it, identifying
duplicates, understanding document structure, and building indexes. The online
request should use those prepared representations instead of rereading the
corpus.

For Memory Palace, ingestion should produce reusable document, chapter,
section, paragraph, lexical, and vector representations. Query-time generation
should be reserved for work that depends on the user's question.

### Use several retrieval signals

Google publicly describes multiple ranking systems rather than one universal
score. These include lexical and conceptual matching, passage ranking, link
analysis, freshness, deduplication, and specialized systems for particular
information needs. See Google's
[guide to Search ranking systems](https://developers.google.com/search/docs/appearance/ranking-systems-guide).

Memory Palace has no web-scale link graph or click history, but the general
lesson holds: exact terms, semantic similarity, title and structure matches,
parent context, and cross-encoder relevance capture different kinds of signal.
No single embedding distance should decide the result.

### Retrieve broadly, rank progressively

A practical search pipeline uses inexpensive methods to find a broad candidate
set, then applies more precise and expensive ranking to a bounded subset. Fast
approximate vector indexes are one component of this process, not the complete
ranking system. Google's published ScaNN work illustrates the speed and recall
tradeoffs involved in large vector indexes:
[SOAR and ScaNN](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/).

Our pipeline should therefore preserve a measurable distinction between:

1. Corpus indexing.
2. Candidate generation.
3. Candidate fusion and selection.
4. Cross-encoder reranking.
5. Evidence assembly.
6. Answer generation.

### Rank passages, not only documents

Google describes passage ranking as understanding whether an individual
section of a page is relevant even when the complete page covers a broader
topic. That distinction is essential for books: finding the correct novel or
textbook is not sufficient if the answer-bearing paragraph is absent.

Memory Palace must measure both document recall and evidence recall. Document
recall is useful diagnostic information, but evidence recall is the retrieval
success criterion.

### Deduplicate without losing canonical identity

Duplicate content consumes ranking capacity and can make results appear more
confident than they are. A canonical source identity should survive file moves
and renames, while exact duplicate content should be indexed once. Canonical
identity must not leak into a query as knowledge of the expected answer.

### Evaluate continuously

Google's public documentation emphasizes ongoing evaluation of ranking
changes. Our equivalent is a versioned suite of realistic questions with
source and evidence ground truth. Improvements must be measured across the
whole suite and private holdouts, not demonstrated on one favored question.

## Why Books And Textbooks Differ From Web Search

Web search commonly returns a page that discusses the query. Memory Palace is
expected to return evidence that can answer it.

Novels create retrieval problems involving:

- names and aliases introduced far from the relevant event;
- pronouns and relationships resolved across passages;
- facts implied by dialogue or narrative rather than stated as definitions;
- developments spread across chapters or an entire series;
- many passages containing the same characters but different events.

Textbooks create different problems:

- chapter and section references;
- definitions separated from consequences or examples;
- equations, tables, captions, and notation;
- questions requiring several propositions or steps;
- several books covering similar concepts with different terminology.

Personal memories add short, irregular documents, dates, private names, and
facts that may be mentioned only once. The same retrieval pipeline must handle
all three without assuming that every source resembles a web page.

## Memory Palace Pipeline

The intended architecture is:

```text
files
  -> parse and normalize
  -> canonicalize and deduplicate
  -> preserve document/chapter/section/paragraph hierarchy
  -> build lexical, vector, and parent representations

question
  -> interpret without adding unsupported facts
  -> run lexical, dense, and hierarchy lanes concurrently
  -> score documents from the broad matches
  -> reserve a small independent document-root routing lane
  -> retrieve dense and lexical passages inside the top documents
  -> combine conditioned candidates with a global escape path
  -> attach local evidence windows
  -> cross-encode and rerank
  -> diversify and select evidence
  -> ask Beemo to answer from cited evidence or abstain
```

Current implementations already cover much of this shape:

- PostgreSQL full-text and trigram indexes provide lexical retrieval.
- Qwen embeddings and pgvector provide dense retrieval.
- Parent nodes preserve document hierarchy and chapter-level routing.
- Reciprocal-rank-style fusion combines retrieval lanes.
- Document-conditioned dense and BM25 lanes adapt a fixed search budget across
  the top eight documents from their measured first-phase confidence.
- A concurrent pooled document-root lane may contribute at most two of those
  eight documents; it routes passage retrieval but never supplies evidence.
- A transparent first-phase ranker preserves normalized RRF, dense, lexical,
  hierarchy, conditioned, document, route-support, and field-match features.
- A fixed reranker pool reserves 25% for conditioned passages and 75% for
  globally ranked candidates.
- GTE ModernBERT reranks a bounded set on GPU.
- Beemo receives selected evidence and source metadata.
- Retrieval traces and evaluation output expose ranks and stage timings.

## Current Measured Gap

The version 4 evaluation baseline is documented in
[`eval/search_eval.md`](../eval/search_eval.md). As of 2026-07-27:

```text
matched document recall:     24/27
matched evidence recall:     17/27
conditioned dense evidence:  14/27
conditioned lexical evidence: 12/27
reranker-pool evidence:      18/27
cross-encoder evidence:      18/27
final document recall@8:     24/27
final evidence recall@8:     18/27
average pipeline latency:    824 ms
pipeline p95 latency:       1097 ms
```

Per-stage tracing shows that broad matching finds 17/27 answer-bearing
passages. Adaptive document retrieval finds 14/27 through dense matching and
12/27 through lexical matching. The independent document-root route recovers
one otherwise absent diary source; the bounded reranker pool and cross-encoder
therefore reach 18/27, and final evidence expansion returns 18/27. The
remaining problems are primarily passages absent from all candidate lanes or
poorly ranked by the current passage representation and cross-encoder.

## Experiment Priorities

### 1. Explain candidate losses

For every evaluation miss, record the best rank in each production phase:

- direct hybrid, semantic, and literal matching;
- hierarchy routing;
- merged global matching;
- document selection;
- document-root selection;
- conditioned dense and lexical passage retrieval;
- bounded reranker-pool assembly;
- cross-encoder order;
- final evidence selection.

An experiment is actionable only when it identifies the stage where expected
evidence disappeared.

The trace reflects the same contextual chunk embeddings and hierarchy used by
production retrieval. Retired comparison lanes do not remain in the request
path; model and selector experiments should consume a fixed traced candidate
pool instead.

### 2. Improve multi-resolution retrieval

Index and retrieve at several granularities:

- document and book;
- chapter or major section;
- subsection;
- paragraph-sized child passage;
- bounded neighboring paragraphs.

Parent matches should route into children, but parent summaries must not crowd
out raw answer-bearing text. Child passages remain the evidence supplied to
the reranker and answer model.

### 3. Strengthen complementary retrieval lanes

Candidate generation should preserve genuinely different signals rather than
several variations of the same cosine search. Useful experiments include:

- field-aware BM25 for title, heading, metadata, and body;
- phrase and proximity scoring;
- dense retrieval over raw passages and structural context;
- learned sparse retrieval;
- late-interaction retrieval such as ColBERT-style token matching;
- query decomposition for questions requiring multiple facts.

Each experiment must report incremental evidence recall, latency, and unique
candidates contributed after fusion.

The next model-level experiments should preserve BM25 as the exact lexical
baseline while comparing:

- BGE-M3 learned sparse output against BM25;
- BGE-M3 dense output against Qwen dense embeddings;
- BGE-M3 multi-vector or ColBERT/PLAID against single-vector dense retrieval;
- late-interaction reranking over the existing fused pool before building a
  corpus-wide token-vector index.

### 4. Improve bounded candidate selection

Increasing the pool indefinitely moves cost downstream and hides weak
retrieval. Prefer a stable upper bound, currently around 50 candidates for the
cross-encoder, with explicit quotas only when they improve suite-wide recall.

Selection should prevent one book, chapter, or retrieval lane from consuming
the entire pool while still allowing several passages from one source when a
question requires synthesis.

### 5. Calibrate final reranking

Once evidence consistently enters the pool:

- compare pure cross-encoder order with fused retrieval and reranker order;
- measure where known evidence is demoted;
- test final evidence counts such as 8, 10, and 12;
- preserve multiple required passages rather than selecting near-duplicates;
- benchmark quality and latency separately on GarnetMoon and Legion Go.

### 6. Validate answer grounding

Beemo should receive only source-attributed evidence and should:

- answer every requested part supported by the evidence;
- distinguish statements from inference;
- cite the source passages used;
- abstain when required evidence is missing;
- avoid filling gaps from model memory when the task is corpus QA.

Retrieval and answer evaluation remain separate so a fluent answer cannot hide
a retrieval failure.

### 7. Optimize latency after quality is understood

Parallel lanes, bounded work, batching, dynamic padding, efficient vector
indexes, and smaller intermediate representations should be optimized before
adding result caches. Caching is valuable near the end, but repeated-query
speed does not improve first-query retrieval quality.

## Guardrails

Do not improve a score by:

- adding aliases or rules for a particular evaluation question;
- putting expected source titles into requests unless the user supplied them;
- embedding expected answers into summaries or query rewrites;
- increasing candidate limits without measuring downstream cost;
- replacing raw evidence with generated summaries;
- tuning only against private holdouts;
- reporting document recall as if it were answer evidence recall.

A valid change should be general, explainable, and beneficial across memories,
novels, and textbooks.

## Decision Rubric

Before adopting a search change, answer:

1. Which pipeline stage does it change?
2. Which previously missed evidence does it recover?
3. Does it preserve or improve the rest of the suite?
4. What does it add to median and tail latency?
5. How many unique candidates does it contribute?
6. Does it work without knowing the expected source or answer?
7. Can it run under both GarnetMoon and Legion Go profiles?
8. Is its behavior visible in diagnostics and logs?

The north-star outcome is not a particular model or algorithm. It is a system
that reliably moves answer-bearing evidence through every bounded stage, makes
failures observable, and answers only from evidence it can show.
