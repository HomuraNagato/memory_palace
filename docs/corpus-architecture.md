# Corpus And Retrieval Architecture

Memory Palace stores source files as a hierarchy while retaining passages as
the only answer evidence:

```text
memory_items
└── memory_nodes: document
    ├── memory_nodes: chapter or page
    │   └── memory_nodes: section
    │       └── memory_chunks: passage
    └── memory_chunks: passage
```

Structural nodes have a parent, depth, stable path key, source order, pooled
passage embedding, optional summary embedding, and descendant passage count.
EPUB spine entries become chapters, PDF form-feed sections become pages, and
recognized headings become sections. Unstructured notes still receive a
document root and may receive heading sections. Raw passages remain the source
of truth for answers and citations.

Chapter and structured-document summaries are generated during ingestion.
They route overview and conceptual questions but never replace supporting
passages. Summary-derived parents are limited to 25% of the routed parent set;
pooled passage embeddings retain the other 75%. This keeps useful summary
recall without allowing generated interpretations to dominate routing. Plain
notes do not incur document-summary model calls.

## Retrieval Plans

`lookup` is the default. It runs global hybrid, literal, and hierarchical lanes
concurrently, ranks documents from their strongest passages, and independently
searches pooled document-root embeddings. Up to two root-routed documents may
replace the lowest passage-routed documents in the fixed set of eight.
Conditioned search then finds additional dense and lexical passages inside
those documents. It has a fixed per-lane budget; every selected document
receives one recall slot, root-routed replacements receive up to six initial
slots, and the remainder is allocated from normalized document scores. This
searches deeply when one source clearly leads and remains broad when document
scores are close. Document roots are routing signals only and are never
returned as answer evidence.

Before cross-encoding, candidates retain raw RRF, dense, lexical, hierarchy,
conditioned, document, route-support, and field-match features. An explicit
normalized first-phase score orders the fixed pool. The cross-encoder receives
at most 50 passages, with a 25% document-conditioned share and a 75% global
escape path. Final lookup results may use up to 75% of their passage budget
from one document. This preserves alternatives without replacing strong
cross-encoder results with much lower-ranked passages solely for diversity.

`overview` is selected for chapter, book-summary, and main-idea questions. It
prioritizes hierarchy-routed passages while scoring documents and permits the
final evidence set to contain more passages from one document.

Clients may set `plan` explicitly to `lookup` or `overview`; otherwise Memory
Palace chooses deterministically from the query. The request embedding is
computed once and reused by global and document-conditioned dense retrieval.
Document selection uses only retrieved evidence; expected titles or answers
never become request filters.

## Answer Loop

Beemo normally retrieves once and answers. If the first reasoning pass finds
the supplied evidence insufficient, it may return one plain-text
`RETRIEVE: <query>` request. Beemo performs exactly one additional global
retrieval, merges and deduplicates both evidence sets, and then produces the
answer. There is no unbounded agent loop.

Memory Palace owns parsing, indexes, retrieval plans, reranking orchestration,
and evidence packaging. Beemo owns the bounded tool loop and final response.
Embedding, reranking, and reasoning remain independent model services.
