# Retrieval Evaluation

`search_eval.yaml` is the version 3 ground-truth suite for mixed-corpus Memory
Palace retrieval and Beemo answers. Private holdout questions are supplied at
runtime and must not be added to the repository.

## Case Contract

Every answerable case defines one or more evidence groups:

```yaml
expected:
  evidence:
    - source:
        relative_path: "books/author/book/book.epub"
        title: "Book"
      any_of: ["one acceptable passage anchor"]
  answer:
    all_of: ["required answer detail"]
```

- Every evidence group must be satisfied.
- `any_of` requires at least one listed fragment.
- `all_of` requires every listed fragment, which may occur in separate chunks
  from the expected source.
- Source paths remain stable across rechunking; generated chunk IDs are not
  ground truth.
- Unanswerable controls contain no evidence and are evaluated only through
  Beemo.

## Running

Run the full retrieval pipeline from `memory_palace/src/go`:

```sh
go run ./cmd/search-eval
```

Useful bounded diagnostics:

```sh
go run ./cmd/search-eval -summary-only
go run ./cmd/search-eval -audit-suite
go run ./cmd/search-eval -candidate-caps 25,50,75,150
go run ./cmd/search-eval -final-limits 8,10,12,16 -summary-only
```

Compare cross-encoders against the same first-phase candidates:

```sh
go run ./cmd/search-eval \
  -candidate-caps 50 \
  -candidate-reranker-urls http://127.0.0.1:5022,http://127.0.0.1:5023
```

Measure cross-encoder ordering independently by reranking every chunk from each
expected source:

```sh
go run ./cmd/search-eval \
  -oracle-source-rerank \
  -oracle-top-k 8 \
  -timeout 10m
```

This is an exhaustive diagnostic, not a production retrieval configuration.

Run answer cases through Beemo:

```sh
go run ./cmd/search-eval -beemo-url http://127.0.0.1:5017
```

## Pipeline Contract

Stage tracing reports the current production phases:

1. `direct_hybrid`, `direct_semantic`, and `direct_literal` broad matching.
2. `hierarchy` parent-to-child matching.
3. `global_matched` merged broad retrieval candidates.
4. `document_selection` the highest-scoring documents from the broad pass.
5. `document_root_selection` up to two additional documents routed by their
   pooled document-root embeddings.
6. `conditioned_dense` and `conditioned_literal` passage retrieval restricted
   to those documents.
7. `rerank_pool` a bounded pool containing 25% document-conditioned candidates
   and a 75% global path.
8. `cross_encoder` authoritative reranker order.
9. `final` diversified evidence returned to Beemo.

The evaluator reports document and evidence recall at every available stage,
plus global search, conditioned search, combined search, rerank, and total
pipeline latency. Expected source metadata never changes request filters;
cases exercise the same broad retrieval path as user queries.

## Suite Composition

The 31 cases comprise:

- 28 answer-bearing questions spanning personal memories, fiction, and
  textbooks;
- 3 unanswerable controls.

Record new baselines only after a complete corpus rebuild and full suite run.
Do not retain retired selector or model experiments in this document.

## Overlapping Passage Windows: 2026-07-30

Memory Palace indexes one deterministic retrieval window of approximately
1,600 characters every 800 characters without crossing chapter, page, or
section nodes. Window hits expand to at most four leaf candidates. Overlapping
hits collapse by canonical leaf ID before the bounded reranker pool, and only
raw leaf passages remain answer evidence.

The initial backfill created 48,407 windows across 194 documents in 678
seconds. It used the embedding service only; no summaries or contextualizer
calls were generated. Four multi-chunk notes produced no windows because every
node contained exactly one isolated leaf.

```text
window-lane evidence recall:          12/28
global matched evidence recall:       20/28  (was 18/28)
reranker-pool evidence recall:        18/28  (was 17/28)
controlled GTE evidence recall@8:     18/28  (was 17/28)
full pipeline evidence recall@8:      18/28  (unchanged)
average window search:                20 ms
average pipeline:                   1009 ms  (was 1002 ms)
pipeline p95:                       1304 ms  (was 1274 ms)
```

The Scholomance mother-name diagnostic still misses: its answer-bearing window
ranks 441 globally and 73 within the correct book under the embedding model,
although the canonical answer leaf ranks first inside that window. This
isolates the remaining failure to query-to-window ranking rather than window
construction, overlap collapse, or reranking.

Set `MEMORY_PALACE_DISABLE_PASSAGE_WINDOWS=true` for controlled ablations.

## Baseline: 2026-07-27

The document-conditioned pipeline allocates a fixed 32-passage budget per lane
across up to eight documents. Every selected document receives one slot; the
remaining depth follows normalized document confidence. A transparent
first-phase score combines preserved retrieval features before assembling a
25% conditioned and 75% global reranker pool. Hierarchy routing reserves 75%
of parents for pooled passage embeddings and up to 25% for unique
generated-summary hits. Final lookup selection permits up to 75% of returned
passages from one document. A concurrent document-root lane may replace the
lowest two passage-selected documents and gives each replacement at most six
conditioned passages. Document roots route retrieval only; answer evidence
still consists exclusively of raw passages.

```text
global matched evidence recall: 17/27
conditioned dense evidence:      14/27
conditioned literal evidence:    12/27
reranker-pool evidence recall:   18/27
cross-encoder evidence recall:   18/27
final evidence recall@8:         18/27
hard evidence recall@8:           5/11
medium evidence recall@8:         7/10
easy evidence recall@8:           6/6
average global search:          305 ms
average conditioned search:     118 ms
average rerank:                 389 ms
average pipeline:               824 ms
pipeline p95:                  1097 ms
```

A controlled summary-routing ablation produced equal aggregate recall with
summary routing enabled and disabled, but each configuration passed a
different case. Reserving three of twelve routed parents for unique summary
hits preserves their complementary value. Replacing the fixed
three-passages-per-document final cap with the bounded 75% share improved
suite-wide evidence recall and kept both diagnostic cases that previously
traded places.

Generated summaries remain routing signals only. An audit found useful names
and events, but also truncation, unsupported inferences, and boilerplate in a
material minority of summaries. Raw passages remain the only evidence used for
answers and citations.

The first-phase scorer was also evaluated with an environment-controlled
ablation. Enabled and disabled runs returned the same evidence cases, while
the feature score improved several candidate ranks, including the hard
`Uprooted` case from 47 to 25 and the Strang textbook case from 46 to 35.
Repeated enabled runs produced identical stage recall. The switch is retained
as `MEMORY_PALACE_DISABLE_FIRST_PHASE_SCORING` for future controlled
experiments, not as a user-facing retrieval mode.

The document-root lane recovered `hard_nightmare_reason_to_live`, whose source
was absent from every broad passage lane, without losing an existing passing
case or increasing measured average latency. It did not recover two mixed
diary entries whose relevant details were diluted in pooled document
embeddings. This lane is therefore deliberately limited to two documents and
runs concurrently with broad retrieval; it is not a substitute for better
passage candidate generation.

## End-to-End Beemo Baseline: 2026-07-27

With Qwen3 4B Instruct on GPU 0, Qwen3 Embedding and GTE ModernBERT reranking
on GPU 1, and Beemo using the same 50-candidate and 8-source limits as the
retrieval evaluation:

```text
answer-bearing questions: 21/27
easy answers:              6/6
medium answers:            8/10
hard answers:              7/11
unanswerable controls:     3/3
average end-to-end:       3338 ms
end-to-end p50:           3203 ms
end-to-end p95:           6319 ms
```

The six remaining answer failures are
`medium_social_bubble_observer`, `medium_ender_battleroom_orientation`,
`hard_game_montage_nostalgia`, `hard_scholomance_enclave_text`,
`synthesis_sherlock_clues_and_cause`, and `synthesis_hyperion_cruciform`.
These correspond to absent or poorly ranked evidence rather than a general
failure to phrase retrieved facts.

Beemo validates citation-only model output and performs one bounded answer
retry. First-person questions explicitly prefer direct personal-memory
evidence over analogous fictional passages. Answer evaluation accepts
semantically equivalent abstentions and mathematical explanations rather than
requiring one exact surface form.

## Oracle-Source Diagnosis: 2026-07-27

The oracle diagnostic reranked every indexed passage from each expected source
without using expected titles or answers as production query filters:

```text
oracle document recall@8: 27/27
oracle evidence recall@8: 22/27
hard evidence recall@8:     7/11
medium evidence recall@8:   9/10
easy evidence recall@8:     6/6
average oracle rerank:    4936 ms
oracle rerank p95:       13702 ms
```

The 9 production misses divide by their first unresolved stage:

| Failure class | Cases | Interpretation |
| --- | --- | --- |
| Candidate generation | `medium_social_bubble_observer`, `hard_game_montage_nostalgia`, `synthesis_sherlock_clues_and_cause` | The existing reranker places the evidence in oracle top 8, but production does not supply all required passages. |
| Reranker or passage representation | `medium_ender_battleroom_orientation`, `hard_legendborn_maternal_magic`, `hard_scholomance_enclave_text`, `synthesis_hyperion_cruciform`, `synthesis_uprooted_selection_and_resentment` | Evidence exists in the indexed source but ranks 14, 31, 14, 99, and 22 respectively under exhaustive reranking. |
| Final selection | `medium_islr_model_selection` | Evidence enters the production pool and reaches cross-encoder rank 7, then falls outside the final source-diversity allocation. |

Candidate-generation changes have an observed oracle ceiling of `22/27` with
the current reranker. Reranker changes must be evaluated against the fixed
oracle-source set so passage-ranking quality is not conflated with retrieval
recall.

The initial oracle report undercounted both `easy_sinegard_purpose` and
`textbook_billingsley_chapter_one_scope` because extracted text wrapped
evidence phrases across line breaks. Evidence and answer matching now collapse
Unicode whitespace before case-insensitive comparison. Both sources pass
production and exhaustive oracle retrieval at final rank 1 without reindexing.
