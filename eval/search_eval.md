# Retrieval Evaluation

`search_eval.yaml` is source-and-evidence ground truth for Memory Palace
retrieval. It avoids generated chunk IDs because rechunking changes those IDs.

Each case specifies:

- `expected.source.relative_path`: stable path beneath a watched root.
- `expected.source.title`: a second source-identity check.
- `expected.evidence_contains`: the relevant passage that must occur in a
  returned chunk.

The runner reports two complementary measures at the suite's `expected_top_k`:

- Document recall: any result from the expected source was retrieved.
- Evidence recall: a result from that source also contains the expected anchor.

Run the complete suite from the Go directory:

```sh
cd memory_palace/src/go
go run ./cmd/search-eval
```

Run selected cases with stable IDs:

```sh
go run ./cmd/search-eval \
  -case relationship_mark_interview_support,book_scholomance_orion_rescues
```

Use `-base-url` when Memory Palace is not available at `http://127.0.0.1:8013`.

Compare the bounded candidate list without invoking the cross-encoder:

```sh
go run ./cmd/search-eval -candidate-caps 50,75,100,150
```

## Baseline: 2026-07-11

With the current Qwen3-Embedding-0.6B semantic/hybrid search path:

```text
document_recall@5: 19/20
evidence_recall@5: 14/20
hard evidence_recall@5: 7/13
easy evidence_recall@5: 7/7
```

The difference between document and evidence recall is intentional signal: it
identifies queries where the correct note or book was found, but not the
passage needed to answer the question.

## Candidate Fusion Baseline: 2026-07-13

Reciprocal rank fusion combines direct semantic, expanded semantic, BM25, and
parent-routed child results. The candidate-only evaluation produced:

| Candidate cap | Document recall | Evidence recall | Hard evidence | Easy evidence |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 20/20 | 18/20 | 11/13 | 7/7 |
| 75 | 20/20 | 18/20 | 11/13 | 7/7 |
| 100 | 20/20 | 18/20 | 11/13 | 7/7 |
| 150 | 20/20 | 18/20 | 11/13 | 7/7 |

The two misses are `book_scholomance_orion_rescues` and
`book_scholomance_liu_friendship` at every cap. Increasing the cap does not fix
them, so they remain retrieval-ranking issues rather than truncation losses.

## Hierarchical Cascade: 2026-07-13

The bounded top-50 path routes the original query through a hierarchical,
hybrid cascade before merging it with direct semantic and BM25 candidates:

1. Retrieve 48 pooled-text and 48 summary-embedding parent candidates.
2. Fuse those parent ranks and retain 24 sections without using a lossy parent
   cross-encoder gate.
3. Retrieve up to eight vector and eight BM25 children per retained section.
4. Fuse and cross-encode those children, retaining 20 hierarchy anchors.
5. Attach bounded three-paragraph context windows to the selected anchors.
6. Fuse hierarchy results with direct retrieval and send at most 50 candidates
   to the final cross-encoder.
7. Fuse final retrieval and cross-encoder ranks at a 2:1 weight before source
   diversification.

The 21-case top-50 candidate evaluation is now:

```text
document_recall: 21/21
evidence_recall: 19/21
hard evidence_recall: 12/14
easy evidence_recall: 7/7
```

For the retired relationship diagnostic, the answer-bearing passage improved
from raw fused rank 499 to a returned source at rank 8. Measured local
latency was 5.36 seconds for retrieval and hierarchy reranking, 1.57 seconds
for the final 50-chunk rerank, and 6.93 seconds total before reranker runtime
optimization. Dynamic padding and length-bucketed ONNX batches reduced the
same pipeline to 1.22 seconds for retrieval, 0.98 seconds for the final rerank,
and 2.20 seconds total without changing the answer rank or evaluation recall.
