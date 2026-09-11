# yt-rag

Retrieval evaluation over YouTube AI-tutorial transcripts. It measures which
combination of chunking strategy, embedding model and reranker actually
retrieves the right transcript spans, on a frozen set of questions.

The system returns spans with timestamps and source links. It does not generate
answers — retrieval quality is the object of measurement, and an LLM writing
prose on top would hide exactly the thing being measured.

## Results

Read this line before the table, because a table read without it says
something different from what it means:

```
104 questions, 72 carrying recall@k (negatives have no gold spans).
Detectable difference: 11.0 pp (paired, power 0.8, α 0.05).
```

Two configurations differing by less than 11 pp are indistinguishable on this
set, however far apart their percentages look. The design called for a pool of
76, which would have given 10.5 pp; the set came in at 72.

Best configurations out of the 105 cells of one run directory: the 63-cell
matrix under the local model, and its 42 dense cells again under `baai/bge-m3`
through OpenRouter. Every figure below comes from that one run.

| configuration | recall@5 | MRR | contradiction coverage | FP@median | p50 | $/1000 queries |
|---|---|---|---|---|---|---|
| `fixed-512-ov128__hybrid__cohere-rerank` /local | **63.2%** | **0.613** | **2/11** | **0.0%** | 717 ms | $0 * |
| `fixed-512-ov128__hybrid__none` /local | 58.4% | 0.527 | 0/11 | 9.4% | 4.3 ms | $0 |
| `fixed-512-ov128__hybrid__none` /bge-m3 | 57.2% | 0.494 | 1/11 | 15.6% | 3.9 ms | $0.0002 |
| `window-90-ov30__hybrid__none` /bge-m3 | 55.1% | **0.577** | 0/11 | 12.5% | 6.6 ms | $0.0002 |
| `window-60__vector__none` /bge-m3 | 51.4% | 0.421 | 0/11 | **0.0%** | 5.2 ms | $0.0002 |
| `fixed-1024-ov256__bm25__none` | 50.6% | 0.447 | 1/11 | 34.4% | 0.3 ms | $0 |

`golden_sha 76841db0c595d0d6`, `git_sha d08c97f71b14`. Runs are comparable only
when both match, and the report refuses to mix them.

**The headline number is 58.4%, and the gate holds.** The ceiling rule below
requires the baseline to stay under 85% for differences to be visible at all;
at 58.4% there is room to measure in.

**Chunk size moves the two retrievers in opposite directions.** BM25 rises with
chunk length and never reverses — 37.1%, 44.0%, 44.0%, 47.3%, 50.6% from 128 to
1024 tokens — because a longer document holds more query terms. The local
embedding model peaks in the middle and falls off at both ends, because mean
pooling washes a short answer out of a long chunk. So "which chunking is best"
has no answer without naming the retriever, and the report never averages that
axis away.

**Hybrid beats both halves, and the paired test can see it.** Against
`fixed-512-ov128__vector__none`: 39 questions both found, 17 neither, 2 only
vector, 14 only hybrid — p = 0.0042. The counts matter more than the p-value:
"2 against 14" is the shape of the result, and 56 of the 72 questions carried no
information about the difference at all. The largest gap in the matrix is
against a vector configuration MMR had already damaged — 0 against 34 — which
is a statement about MMR rather than about fusion.

**The recall/abstention trade-off is real and large.** The leading
configuration answers 9.4% of the questions that have no answer.
`window-60__vector__none` answers **0%** of them, at 35.4% recall. Chosen on
recall alone, a system would confidently answer questions with no answer in the
corpus — which is why the false-positive column sits beside recall rather than
in an appendix.

**MMR is a measured negative result.** At λ 0.7 it lost recall@5 in 18 of 21
configurations, gained in 2 and tied in 1. It was added to move contradiction
coverage, and moved it up in 2 configurations while moving it down in 4. It
diversifies by surface dissimilarity, and two passages arguing opposite sides
of one question share their topic vocabulary — so it reads them as redundant
and pulls against the metric it was added to help. It stays in the matrix as a
row worth having.

**The reranker improves every number, and the set cannot confirm any of
them.** Cohere Rerank over the best configuration moves recall@1 from 28.1% to
41.3%, recall@5 to 63.2%, MRR from 0.527 to 0.613, false positives from 9.4% to
**0.0%**, and contradiction coverage from 0/11 to 2/11 — the only thing in the
project that has moved that metric off zero on this configuration. It helps most
where help is needed: +6.7 pp on the questions term matching cannot answer,
against +1.8 pp on the ones it can.

And not one paired comparison reaches significance. At recall@5, 4 questions
went to `none` and 5 to the reranker (p = 1.0); at recall@1, 8 against 16
(p = 0.15); at recall@10, 1 against 4 (p = 0.38). So a 13-point gain at rank 1
is what this set calls indistinguishable from chance, which is exactly what the
detectable-difference line above the table exists to say. The honest summary is
that every point estimate favours reranking and 72 questions cannot establish
it.

It is also the one configuration where latency matters: **717 ms at p50 and
1,665 ms at p95**, against 4.3 ms without it. The reranking round trip is
167 times the retrieval it corrects.

\* It cost no money and is not free: the axis ran on a Cohere trial key, which
allows 1,000 calls a month at 10 a minute. One cell — 104 questions × 3 repeats
— is 312 of them.

**The hosted embedding model wins, and the set can see it.** `baai/bge-m3`
through OpenRouter beats the local 384-dimension model in 26 of the 28 dense
configurations. Of the 28 paired comparisons, the set resolves 11 at p <= 0.05
— **all 11 favouring bge-m3, none favouring local**. The gains concentrate
where the dense retriever works alone over short chunks:
`fixed-128__vector__none` goes 29.0% to 48.5%, `window-60__vector__none` 35.4%
to 51.4%. Hybrid configurations over long chunks are indistinguishable, which
is what a lexical half doing most of the work looks like.

That closes the embedding axis of the design, and it cost **$0.0355** for the
whole matrix — 3.2M tokens at $0.01 per million, measured from the vendor's own
`usage.cost` rather than inferred from a price table. Gemini's free tier could
not run it at any schedule: the allowance is spent per text against a daily cap
of roughly a thousand, and this corpus is 3,238 chunks.

Two things the local model had hidden:

- **"Long chunks wash out a dense retriever" was a property of that model.**
  The local one peaks in the middle of the size range and collapses at 128
  tokens; bge-m3 is at its best there. A conclusion about chunk size drawn from
  a 384-dimension MiniLM would have been a conclusion about MiniLM.
- **bge-m3 abstains perfectly on this corpus.** Every `vector` configuration
  under it scores 0.0% false positives at its own median threshold — 32
  unanswerable questions, none answered. Recall rose and the abstention did not
  pay for it, which is not the usual shape of that trade-off.

**Latency does not discriminate at this scale, and that is the finding.**
Everything is single-digit milliseconds: the best configuration costs 3.8 ms,
the cheapest 0.3 ms at 50.6% recall. So no configuration is disqualified on
speed and the choice rests entirely on recall and abstention. The shape is still
informative — dense latency is linear in chunk count (1.5 ms over 534 chunks,
9.5 ms over 3,238) because the vector search is a full scan with no index, and
MMR adds a flat ~5.5 ms regardless of corpus size. At a hundred times this
corpus, that linearity is the first thing that would break.

The embedder is the exception, and it moves latency by an order of magnitude:
one query embedding through the hosted model is a **33.8 ms** round trip,
against 0.3–10.2 ms for the retrieval it feeds. So the statement holds among
retrieval configurations and not between embedders — the local model's recall
costs less in milliseconds than it looks.

**Retrieval is deterministic.** All three repeats of all 42 cells returned
identical rankings — 0 disagreements over 3,024 question-cells. That is what
the repeats are for, and it is what licenses treating noise as zero in the power
calculation instead of assuming it.

### The metric the corpus was built for

Contradiction coverage asks whether retrieval surfaced **both** sides of a
disagreement, not just one. The contrast is the most interesting number in the
project:

| | best configuration |
|---|---|
| found at least one side (recall@5) | 9/11 — 82% [48–98%] |
| found both sides (coverage@5) | **0/11 — 0%** |

A retriever that reliably finds one side of a live disagreement and never finds
both is precisely the failure ordinary recall cannot see, and here it is, on a
metric built to see it. Best coverage anywhere in the matrix was 2 of 11.

With 11 measurable questions against a design of 18, this is descriptive: a
difference of one question moves it by 9 pp. It is reported as a proportion
with an interval, never as a comparison between configurations.

### Where retrieval actually struggles

recall@5 for the best configuration, split by whether term matching alone
answers the question at rank 1:

| term-matchable | needs more than term matching |
|---|---|
| 76.7% (n=27) | 47.4% (n=45) |

The split reads down a column, not across chunking rows: the label comes from a
BM25 pass over *that configuration's own* chunks, so the two halves are a
different partition of the same 72 questions in every row.

By question kind, best configuration, recall@5 with exact intervals:

| kind | | |
|---|---|---|
| contradiction | 9/11 | 82% [48–98%] |
| factual | 28/36 | 78% [61–90%] |
| comparative | 16/25 | 64% [43–82%] |

## What did not run

Of 63 cells — 7 chunking × 3 retrieval × 3 reranking — **42 ran and 21 did
not**, and the gaps are worth stating plainly rather than leaving as holes in a
table.

| blocked | cells | why |
|---|---|---|
| `cohere-rerank` | 21 | `COHERE_API_KEY` was never set. The client is written and tested; with a key each cell needs up to 312 billed calls |
| Gemini embeddings | 28 under `--embedder gemini` | the free-tier allowance is spent per text against a daily cap of roughly a thousand, and this corpus is 3,238 chunks at 128 tokens. Run through OpenRouter instead, billed per token — see the embedding comparison above |

Both blocked steps have since run. The embedding-model comparison went
through OpenRouter for $0.0355; the reranker-on/off comparison ran on a Cohere
trial key, on the winning configuration, for no money and 312 of a monthly
allowance of 1,000 calls. The 21 `cohere-rerank` cells of the full matrix remain
unrun — at 312 calls each, the allowance covers three a month.

A skipped cell is recorded in its file as "not run" with the reason, never as a
zero and never as a blank, because a missing cell that looks like a bad result
is worse than an empty one.

## What makes this different from a RAG tutorial

The corpus **contains contradictions on purpose**. Channels were selected across
a hype / practical / skeptical axis, and videos span 12–18 months, so different
authors say opposite things about the same tool — sometimes because they
disagree, sometimes because the tool changed and one of them is out of date.

That makes two metrics possible that standard RAG benchmarks leave out:

- **contradiction coverage** — when both sides of a disagreement are annotated
  (`side: pro` / `side: contra`), did retrieval surface both, or only one? A
  system that confidently returns one side of a live disagreement is worse than
  one that returns neither, and ordinary recall scores them the same. Measured
  above: 82% one side, 0% both.
- **false-positive rate on negative questions** — four tools are kept out of the
  corpus deliberately. Questions about them have an empty correct answer. Most
  RAG systems fail here and few benchmarks measure it. Measured above: from 0%
  to 34.4% depending on configuration, at each configuration's own median score.

## Design decisions worth knowing

**The question set is sized by a power calculation, not by feel.** 72 questions
carry `recall@k`, not the 40 originally planned. Differences between chunking
strategies live in the 5–15 pp range; a 40-question set only detects 22.5 pp, so
its most likely result is "every configuration looks the same" — a statement
about the question set that is easy to mistake for a statement about chunking.

Negatives are counted separately, at 32 rather than a proportional 14. They have
no gold spans, so they do not move that number at all; their count is set by what
false-positive rate needs. They are also the cheapest question to write, and an
extra hour there halves the confidence interval on a headline metric.

**The baseline has to stay below 85% on recall@5.** Against a 93% baseline no
configuration difference is visible at any sample size, because the metric is
against its ceiling. There is no lower bound — the best configuration at 58.4%
is further from the ceiling and so more sensitive, not less. If the set comes out
too easy it gets harder questions, never fewer of the ones the baseline answered
correctly, which would tune the set to the retriever using the retriever's own
output.

**What counts as a hit is one rule, written down and identical everywhere.** A
gold annotation is an interval of a video and a retrieved chunk is another, so
`overlap / min(gold_length, chunk_length) >= 0.5`. Dividing by the shorter
interval is deliberate: under *any overlap*, a 1024-token chunk crosses more
gold spans than a 512-token one regardless of relevance and wins the chunk-size
comparison before a number is interpreted; under *IoU >= 0.5* a 90-second chunk
containing a 10-second answer scores 0.11 and misses, punishing a correct chunk
for being long. The threshold travels in every run's header, so a file written
under a different rule is visible rather than silently averaged in.

**Vectors are padded to a storage width of 1536, which preserves cosine
distance exactly.** The binding constraint was never the arithmetic but the
index: HNSW and IVFFlat cap at 2000 dimensions, so a 3072-wide vector stores
fine and cannot be indexed at all. The padding is implemented; the database is
not — see *No database* below.

**Every configuration runs three times, and the repeats are not averaged.**
They exist to check that retrieval is deterministic. Averaging them would be
either a no-op or a way of hiding a finding, so a disagreement is reported as a
finding and the table shows the first repeat. Timings are the exception: the
rankings being identical makes the three passes three independent measurements
of the same work, which is what a percentile wants more of.

**Cost is recorded in units and priced at read time.** A run writes what the
configuration consumes — texts, tokens, rerank searches — and `prices.yaml`
supplies the tariff when the table is printed, so a corrected price is a re-read
rather than a re-run. A price there is either a number with a source and a date,
or null with a reason; there is no third form. Both hosted vendors are null,
because this project never paid either of them, and the report prints the
reasons where the dollars would go.

**No database.** The specification describes a `chunks` table with a mandatory
`chunk_set_id`; there is no database in the repository. Everything is derived
from `cache/transcripts/*.json` on each invocation, and a JSONL file is one cell
of the matrix by construction — its header names the chunking, retrieval and
reranking configuration, so the `cell` field is the chunk-set identity for these
runs. The vector search is consequently a full scan, which the latency figures
show directly.

**The report cannot retrieve anything.** `pnpm eval` retrieves and writes;
`packages/report` reads and computes, and imports no retriever, no embedder and
no chunker. A test walks its import graph to keep it that way — not for tidiness,
but so that no figure in the table can have come from retrieving again instead of
from reading what a run recorded.

## Reproducing

```
pnpm test               # unit tests, including the tests for the metric code
pnpm golden             # validate the question set, print the recall pool
pnpm ingest             # fetch transcripts → cache/ (once; never re-fetches)
pnpm eval --dry-run --run-all   # print the matrix and what it would cost
pnpm eval --run-all     # 63 cells × 3 repeats → runs/<stamp>/*.jsonl
pnpm report             # aggregate the newest run directory into the tables above
```

The full matrix takes about 90 seconds against a warm embedding cache and
computes no embeddings. `pnpm eval --chunking fixed-512 --retrieval bm25,hybrid`
narrows it to named cells; an id that matches nothing is an error rather than a
quietly empty run.

## Limitations

- **Corpus size.** 78 videos across 12 channels, not thousands. Confidence
  intervals are wide and the results describe this corpus, not YouTube.
- **Sample selection.** Twelve channels, chosen by hand for a spread of opinion
  about a handful of AI coding tools — 4 hype, 5 practical, 3 skeptical. That mix
  shapes the contradiction questions.
- **Two embedding models, and the hosted one is a single vendor.** The
  comparison is local MiniLM against `baai/bge-m3`; Gemini, which the design
  named, never ran, because its free tier is capped per day rather than priced
  per token. "A hosted model beats a small local one here" is what the data
  supports, not a ranking of hosted models.
- **The hosted embedder is not reproducible to the bit.** The same text
  embedded twice returns vectors that differ at 1e-4. Rankings are stable
  because the disk cache is, so the three repeats verify the cache rather than
  the vendor.
- **The reranker was measured on one configuration, not on the matrix.** A
  trial key allows 1,000 calls a month and one cell costs 312, so the on/off
  comparison ran on the winner alone. It is also a comparison against **one**
  reranker: no local cross-encoder exists for TypeScript, so there is nothing
  to compare Cohere against but MMR, which is a different kind of thing.
- **Contradiction coverage rests on 11 questions**, against a design of 18. The
  corpus turned out to hold less disagreement than the design assumed. One
  question moves that metric by 9 pp, so it is descriptive only.
- **Transcript quality varies.** 72 of the 78 videos have auto-generated
  captions, which arrive without punctuation and with recognition errors. The
  manual/auto flag is in `cache/meta`, but no report cut uses it yet, so this is
  a caveat rather than a measured comparison.
- **Sentence chunking is absent.** Configuration #6 in the specification needs
  punctuation restoration, which does not exist in TypeScript here, so the
  chunking axis is fixed-size and time-window only.
- **Per-kind results are descriptive.** 11 contradiction and 32 negative
  questions cannot support "configuration A beats B on contradictions". Those
  metrics are reported as proportions with exact intervals, not as comparisons.
- **Third-party content.** Transcripts belong to their authors. Any public
  endpoint returns short excerpts with a timestamp and a link to the source,
  never full paragraphs.

## Status

The ablation has run, under two embedding models: 105 cells in one run
directory, results above. What remains is the reranker axis, which needs a paid
key, the seven contradiction questions the design asked for, and deployment.

- [docs/spec.md](docs/spec.md) — schema, ablation configurations, metrics, and
  what the runner actually measured
- [CONVENTIONS.md](CONVENTIONS.md) — invariants the implementation has to respect
- [configs.yaml](configs.yaml) — the ablation matrix, read by the runner
- [prices.yaml](prices.yaml) — tariffs, with a source and a date or a reason
- [corpus.yaml](corpus.yaml) — tools, channels, topics, selection rules
