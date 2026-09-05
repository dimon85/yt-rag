# yt-rag

Retrieval evaluation over YouTube AI-tutorial transcripts. Measures which
chunking + embedding + reranking configuration actually retrieves the right
transcript spans, on a frozen question set.

The system returns **spans with timestamps and source links**. It does not
generate answers. Full spec: `docs/spec.md`.

## Hard invariants — never violate these

1. **Every row in `chunks` has a `chunk_set_id`.** Chunks from different
   strategies and embedding models share the table. Without this column the
   ablation is unrecoverable.
2. **One tokenizer for all configurations.** Chunk sizes are counted with
   `gpt-tokenizer`, never with `text.length / 4` or a per-provider tokenizer.
   Inconsistent counting silently invalidates every comparison.
3. **The golden set is frozen.** `golden/questions.yaml` is not edited to
   improve results. If it changes, that is a new version: bump it, record the
   new `golden_sha`, and do not compare across versions. Growing the set counts
   as changing it — a table mixing two sizes has two denominators.
4. **Metric code is tested before any results are trusted.** `packages/evals`
   has unit tests with hand-computed expected values for `recall@k`, MRR,
   contradiction coverage. Prior experience on this codebase: grader bugs
   outnumbered real differences between configurations.
5. **Transcripts are fetched once.** Raw transcripts are cached in `cache/` as
   JSON. Never re-fetch during a run — it hits rate limits and wastes a day.
6. **Only compare runs with identical `golden_sha` and `git_sha`.** Mixed
   comparisons go in the bin, not in the report.
7. **No answer generation.** If a task seems to need an LLM to phrase an
   answer, stop and ask. Retrieval quality is the object of measurement.
8. **Public output is short spans only.** Transcripts are third-party content:
   return brief excerpts with timestamp and source URL, never paragraphs.
9. **`chunks.embedding` is `vector(1536)` — a storage width, not a model
   dimension.** The model's real width lives in `chunk_sets.dim` and must be
   ≤ 1536. Shorter vectors are zero-padded up to 1536 by `toStorage()` — both
   chunks and query vectors, through that one helper and nowhere else. Padding
   preserves cosine distance exactly; comparing across models never happens,
   because every query filters on `chunk_set_id`.
10. **The set is sized by a power calculation, not by feel.** 90 questions, of
    which 76 carry `recall@k` (negatives have no gold spans). That detects a
    10.5 pp difference at power 0.8. The number is printed in the run header
    and sits above the results table, never only in a limitations section.
11. **The baseline must land at 60–85% on `recall@5`.** Against the ceiling no
    configuration difference is measurable at any sample size. If the set comes
    out too easy, it gets harder questions — never fewer of the ones the
    baseline answered correctly. Selecting on the retriever's own output tunes
    the set to the retriever.
12. **Per-`kind` results are descriptive, not comparative.** 18 contradiction
    and 14 negative questions do not support "A beats B on contradictions".
    Report a proportion with a Clopper-Pearson interval instead.

## Conventions

- `packages/chunking` is pure functions: segments in, chunks out. No database
  access, no network. This is where bugs hide, so it stays trivially testable.
- Every provider call records its cost. Embedding and rerank spend land in
  `chunk_sets.index_cost_usd` and `run_results.cost_usd`.
- One JSONL file per run, appended as results arrive. `packages/report`
  aggregates; it never re-runs anything.
- Each configuration runs three times. Retrieval should be deterministic —
  verify that, don't assume it.
- `manual` and `generated` transcripts stay distinguishable
  (`videos.transcript_kind`). Their quality differs and it's a useful cut.

## Commands

```
pnpm ingest            # fetch transcripts → cache → db
pnpm chunk --config N  # build a chunk_set from configs.yaml
pnpm embed --set N     # embed a chunk_set
pnpm eval --run-all    # all configurations against the golden set
pnpm report            # aggregate JSONL → results table
pnpm test              # unit tests, including metric code
```

## Ask before doing

- editing `golden/questions.yaml`
- changing the `chunks` or `runs` schema, including the `vector(1536)` storage
  width — that one is expensive to undo
- adding an embedding model whose `dim > 1536`. It needs either the API's
  `dimensions` parameter or a separate unindexed table, and that is a decision,
  not an implementation detail. A model with `dim ≤ 1536` is an ordinary change
  and needs no permission.
- anything that would make an existing run's results non-comparable

## Not goals

Chat interface, answer generation, RAG-as-a-product, supporting arbitrary
corpora. This is a measurement tool that happens to also be useful for
research.