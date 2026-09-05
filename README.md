# yt-rag

Retrieval evaluation over YouTube AI-tutorial transcripts. It measures which
combination of chunking strategy, embedding model and reranker actually
retrieves the right transcript spans, on a frozen set of questions.

The system returns spans with timestamps and source links. It does not generate
answers — retrieval quality is the object of measurement, and an LLM writing
prose on top would hide exactly the thing being measured.

## Results

Not yet. The ablation has not run.

What can be said before it does — because it is a property of the question set,
not of the results — is how large a difference this set is able to see:

```
90 questions, 76 carrying recall@k (negatives have no gold spans).
Detectable difference: 10.5 pp (paired, power 0.8, α 0.05).
```

That line will sit directly above the results table rather than in a footnote,
because a table read without it says something different from what it means.

| configuration | recall@5 | MRR | contradiction coverage | $/1000 queries |
|---|---|---|---|---|
| _pending_ | | | | |

## What makes this different from a RAG tutorial

The corpus **contains contradictions on purpose**. Channels were selected across
a hype / practical / skeptical axis, and videos span 12–18 months, so different
authors say opposite things about the same tool — sometimes because they
disagree, sometimes because the tool changed and one of them is out of date.

That makes two metrics possible that standard RAG benchmarks leave out:

- **contradiction coverage** — when both sides of a disagreement are annotated
  (`side: pro` / `side: contra`), did retrieval surface both, or only one? A
  system that confidently returns one side of a live disagreement is worse than
  one that returns neither, and ordinary recall scores them the same.
- **false-positive rate on negative questions** — four tools are kept out of the
  corpus deliberately. Questions about them have an empty correct answer. Most
  RAG systems fail here and few benchmarks measure it.

## Design decisions worth knowing

**The question set is sized by a power calculation, not by feel.** 90 questions,
not the 40 originally planned. Differences between chunking strategies live in
the 5–15 pp range; a 40-question set only detects 22.5 pp, so its most likely
result is "every configuration looks the same" — a statement about the question
set that is easy to mistake for a statement about chunking.

**The set is built so the baseline lands at 60–85% on recall@5.** Against a 93%
baseline no configuration difference is visible at any sample size, because the
metric is against its ceiling. If the set comes out too easy it gets harder
questions — never fewer of the ones the baseline answered correctly, which would
tune the set to the retriever using the retriever's own output.

**One chunks table at `vector(1536)`, treated as a storage width rather than a
model dimension.** Shorter vectors are zero-padded up to it, which preserves
cosine distance exactly. The binding constraint is the index, not the
arithmetic: HNSW and IVFFlat cap at 2000 dimensions, so a 3072-wide vector
stores fine and cannot be indexed at all.

**Every configuration runs three times.** Retrieval ought to be deterministic.
That is checked rather than assumed — and with a hosted reranker in the loop it
also checks whether the vendor's model drifts between runs.

## Reproducing

```
pnpm ingest            # fetch transcripts → cache → db
pnpm chunk --config N  # build a chunk_set from configs.yaml
pnpm embed --set N     # embed a chunk_set
pnpm eval --run-all    # all configurations against the golden set
pnpm report            # aggregate JSONL → results table
pnpm test              # unit tests, including the metric code
```

Transcripts are fetched once and cached on disk; re-runs never hit YouTube.
Runs are only comparable when `golden_sha` and `git_sha` match — the report
refuses to mix them.

## Limitations

- **Corpus size.** Tens of videos, not thousands. Confidence intervals are wide
  and the results describe this corpus, not YouTube.
- **Sample selection.** Twelve channels, chosen by hand for a spread of opinion
  about a handful of AI coding tools. The stance split is roughly 33% hype, 44%
  practical, 23% skeptical, and that mix shapes the contradiction questions.
- **Transcript quality varies.** Auto-generated transcripts arrive without
  punctuation and with recognition errors. `manual` and `generated` stay
  distinguishable so this can be reported as a separate cut rather than
  averaged away.
- **One reranker.** No local cross-encoder exists for TypeScript, so reranking
  is measured against a single hosted API. The on/off comparison is valid; it is
  not a comparison between rerankers.
- **Per-kind results are descriptive.** 18 contradiction and 14 negative
  questions cannot support "configuration A beats B on contradictions". Those
  metrics are reported as proportions with confidence intervals, not as
  comparisons.
- **Third-party content.** Transcripts belong to their authors. Any public
  endpoint returns short excerpts with a timestamp and a link to the source,
  never full paragraphs.

## Status

Pre-implementation. Specification, conventions and corpus definition are in
place; ingest has not run.

- [docs/spec.md](docs/spec.md) — schema, ablation configurations, metrics
- [CONVENTIONS.md](CONVENTIONS.md) — invariants the implementation has to respect
- [corpus.yaml](corpus.yaml) — tools, channels, topics, selection rules
