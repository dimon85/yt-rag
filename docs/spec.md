# yt-rag — retrieval over AI-tutorial transcripts, with an eval set

Technical specification. The goal is not to "build a RAG" but to **measure
which retrieval strategies actually work on transcripts**, and to end up with a
working tool.

---

## The whole thing in one paragraph

The system pulls transcripts from YouTube videos about AI tooling, indexes them
several different ways, and measures — on a fixed question set — which way
retrieves the right spans best. It does not generate answers: it returns spans
with timestamps and a link to the source.

The key difference from a typical RAG project: the corpus **contains
contradictions**. Different authors say opposite things about the same tool. So
the metrics cover not only "did we find something relevant" but also "did we
find both sides".

---

## Stack

| Layer | Choice | Note |
|---|---|---|
| Language | TypeScript, Node 22 | Primary language of the project |
| Monorepo | pnpm | Standard for a Node monorepo |
| Transcripts | `youtube-transcript-plus` | Maintained fork, no API keys |
| Tokenization | `gpt-tokenizer` or `js-tiktoken` | Honest token counts per chunk |
| DB + vectors | Supabase Postgres + `pgvector` | Managed Postgres, vectors without a separate store |
| Embeddings | 2 models over HTTP APIs | Language-agnostic |
| Reranker | Cohere Rerank API | A compromise: no local cross-encoders in JS |
| Runner and report | **existing harness** | JSONL runner + `report.mjs` — reuse, don't rewrite |
| UI (optional) | Next.js | Week 3, not required for the metrics |

**Main decision:** the eval infrastructure is not written from scratch. A JSONL
runner, per-run isolation via git worktree, ablation configs and a `report.mjs`
aggregator already exist. Only the object of measurement differs here — a
retriever instead of an LLM. That saves a week.

---

## Database schema

The main architectural decision: **chunks from every configuration live in one
table, but with a mandatory `chunk_set_id`**. Without it, a week later there is
no way to tell which embeddings came from which strategy.

```sql
create extension if not exists vector;

-- 1. Raw data. Immutable. Fetched once.
create table videos (
  id            bigserial primary key,
  youtube_id    text unique not null,
  channel       text,
  title         text,
  published_at  timestamptz,
  duration_s    int,
  view_count    bigint,
  transcript_kind text,          -- 'manual' | 'generated'
  fetched_at    timestamptz default now()
);

create table transcript_segments (
  id         bigserial primary key,
  video_id   bigint references videos(id) on delete cascade,
  start_s    numeric(10,2) not null,
  end_s      numeric(10,2) not null,
  text       text not null
);
create index on transcript_segments (video_id, start_s);

-- 2. Indexing configurations
create table chunk_sets (
  id          bigserial primary key,
  name        text unique not null,   -- 'fixed512-ov0-oai-small'
  strategy    text not null,          -- fixed | time_window | sentence
  params      jsonb not null,         -- {"tokens":512,"overlap":0}
  embed_model text not null,
  dim         int not null check (dim between 1 and 1536),  -- real model width
  index_cost_usd numeric(10,4),       -- what it cost to index
  created_at  timestamptz default now()
);

create table chunks (
  id            bigserial primary key,
  chunk_set_id  bigint references chunk_sets(id) on delete cascade,
  video_id      bigint references videos(id) on delete cascade,
  text          text not null,
  start_s       numeric(10,2) not null,
  end_s         numeric(10,2) not null,
  token_count   int,
  embedding     vector(1536) not null -- STORAGE width, not the model dimension
);
create index on chunks (chunk_set_id, video_id);

-- 3. Golden set. Frozen, under git.
create table questions (
  id    bigserial primary key,
  slug  text unique not null,   -- stable identifier matching the YAML
  text  text not null,
  kind  text not null           -- factual | comparative | contradiction | negative
);

create table question_gold (
  question_id bigint references questions(id) on delete cascade,
  video_id    bigint references videos(id),
  start_s     numeric(10,2),
  end_s       numeric(10,2),
  side        text,      -- for contradiction: 'pro' | 'contra'
  note        text
);

-- 4. Runs
create table runs (
  id           bigserial primary key,
  chunk_set_id bigint references chunk_sets(id),
  retriever    text not null,   -- vector | hybrid
  reranker     text,            -- null | 'cohere-rerank-v3'
  top_k        int not null,
  golden_sha   text not null,   -- hash of the golden-set YAML
  git_sha      text not null,
  created_at   timestamptz default now()
);

create table run_results (
  run_id      bigint references runs(id) on delete cascade,
  question_id bigint references questions(id),
  rank        int not null,
  chunk_id    bigint references chunks(id),
  score       numeric,
  is_hit      boolean,
  latency_ms  int,
  cost_usd    numeric(10,6)
);
```

**On dimensions.** Models produce different widths — 1536 (`text-embedding-3-small`),
1024 (Cohere `embed-v3`, `voyage-3`, BGE-m3), 3072 (`text-embedding-3-large`).
The decision here: **one table, `vector(1536)`, shorter vectors zero-padded up,
3072 not allowed.**

Zero-padding is safe. `cos(a,b) = a·b / (|a|·|b|)`; appending zeros to the tail
of *both* vectors leaves the dot product and both norms untouched, so the
distance is bit-for-bit identical. Padding would only be a problem when
comparing vectors from *different* models — which never happens here, because
every query filters on `chunk_set_id`, and the query vector is padded by the
same code as the chunks. Inside one chunk_set every vector comes from one model
at one real width.

The real constraint is the index, not the arithmetic. `vector` stores up to
16000 dimensions, but HNSW and IVFFlat index at most **2000** (`halfvec`: 4000).
So `vector(3072)` stores fine and cannot be indexed at all. There is a cast
workaround — `create index ... using hnsw ((embedding::halfvec(3072))
halfvec_cosine_ops)` — but that is half precision, which adds a variable to the
experiment that nobody asked for.

3072 is not needed anyway: `text-embedding-3-large` takes a `dimensions`
parameter and returns 1536 natively — it is Matryoshka-trained, not blindly
truncated. That has a side benefit worth stating plainly: **with both models at
1536, width is held constant as a variable.** The comparison then measures
model quality rather than "bigger vector vs. smaller one", which is the
cleaner experiment.

`chunk_sets.dim` records the model's real width and is constrained to ≤ 1536.
Adding a model at or below that width needs no migration and no new table.

**On the HNSW index.** Do not create one at the start. 100 videos across a
dozen configurations is tens of thousands of rows, and every query runs under
`where chunk_set_id = N` — one to three thousand vectors. A sequential scan
there costs single-digit milliseconds and gives *exact* recall rather than
approximate. Add a partial HNSW index per `chunk_set_id` later, and measure
what it costs in recall: that is one more number for the README.

Both chunk and query vectors go through one helper, and padding happens nowhere
else:

```ts
export const STORAGE_DIM = 1536;

/**
 * Pads a model vector up to the storage width. Chunks and query vectors must
 * both go through this — a query padded differently silently ruins recall.
 */
export function toStorage(vec: number[], expectedDim: number): number[] {
  if (vec.length !== expectedDim) {
    throw new Error(`model returned ${vec.length}, chunk_set says ${expectedDim}`);
  }
  if (expectedDim > STORAGE_DIM) {
    throw new Error(`dim ${expectedDim} exceeds storage width ${STORAGE_DIM}`);
  }
  return [...vec, ...new Array(STORAGE_DIM - expectedDim).fill(0)];
}
```

Throwing rather than `assert`: Node strips assertions under
`--no-force-node-api-uncaught-exceptions-policy`-style flags and
`NODE_OPTIONS`, and this check has to survive into production ingest.

The check on the real width matters more than it looks: if a provider quietly
changes its default, ingest fails immediately instead of producing a corrupted
metrics table discovered a week later. Same class of bug as the metric-code
bugs — the ones that outnumber the differences between configurations.

### The pgvector-from-TypeScript trap

`pg` does not know the `vector` type and will hand it over as a string in both
directions. Writing needs `'[0.1,0.2,...]'`; reading needs parsing back. This is
the single most common place where projects like this break silently: pass a JS
array straight in, the driver serializes it as `{0.1,0.2}` — Postgres *array*
syntax — Postgres casts it without complaint, and the vectors come out garbage.

```ts
export const toPgVector = (v: number[]) => `[${v.join(",")}]`;
export const fromPgVector = (s: string): number[] => JSON.parse(s);
```

The invariant the whole decision rests on gets its own test:

```ts
test("padding preserves cosine", () => {
  const a = Array.from({ length: 1024 }, () => Math.random() - 0.5);
  const b = Array.from({ length: 1024 }, () => Math.random() - 0.5);
  expect(cosine(toStorage(a, 1024), toStorage(b, 1024))).toBeCloseTo(cosine(a, b), 12);
});

test("vector survives the pg string round-trip", () => {
  const v = toStorage(Array.from({ length: 1024 }, () => Math.random() - 0.5), 1024);
  expect(fromPgVector(toPgVector(v))).toEqual(v);
});
```

The round-trip that actually matters is through a live database — insert, select
back, compare — but that belongs in an integration test, not next to the pure
functions. The string codec above is what unit tests can pin down.

---

## Repository layout

```
yt-rag/
├── packages/
│   ├── ingest/           # transcripts → DB, with an on-disk cache
│   ├── chunking/         # strategies, pure functions — easy to test
│   ├── embed/            # embedding-provider clients + cost accounting
│   ├── retrieve/         # vector and hybrid search, reranking
│   ├── evals/            # metrics + TESTS FOR THE METRICS
│   └── report/           # JSONL aggregation → table
├── golden/
│   └── questions.yaml    # frozen set, under git
├── cache/                # raw transcripts, .gitignore
└── configs.yaml          # ablation configurations
```

`chunking` as pure functions is deliberate: chunking strategies break most
easily and test most easily. Input is an array of segments, output is an array
of chunks. No database inside.

---

## Config validation

`corpus.yaml` carries a planned balance across channel stances. A plan that
drifts from its own targets is worth catching before ingest spends a day
fetching, not after. Schema and balance check live in one Zod object, so there
is no way to load the config while skipping validation:

```ts
import { z } from "zod";

const Stance = z.enum(["hype", "practical", "skeptical"]);

export const CorpusConfig = z.object({
  targets: z.object({
    stance_share: z.record(Stance, z.number()),
    tolerance: z.number().default(0.03),
    min_bucket_videos: z.number().int().default(20),
  }),
  channels: z.array(z.object({
    name: z.string(),
    handle: z.string(),
    stance: Stance,
    target_videos: z.number().int().positive(),
  })),
}).superRefine((cfg, ctx) => {
  const planned = new Map<string, number>();
  for (const ch of cfg.channels) {
    planned.set(ch.stance, (planned.get(ch.stance) ?? 0) + ch.target_videos);
  }
  const total = [...planned.values()].reduce((s, n) => s + n, 0);

  for (const [stance, want] of Object.entries(cfg.targets.stance_share)) {
    const count = planned.get(stance) ?? 0;
    const got = count / total;
    if (Math.abs(got - want) > cfg.targets.tolerance) {
      ctx.addIssue({
        code: "custom",
        path: ["targets", "stance_share", stance],
        message: `plan gives ${got.toFixed(3)}, target ${want.toFixed(3)}`,
      });
    }
    if (count < cfg.targets.min_bucket_videos) {
      ctx.addIssue({
        code: "custom",
        path: ["channels"],
        message: `stance ${stance}: only ${count} videos planned`,
      });
    }
  }
});

export type CorpusConfig = z.infer<typeof CorpusConfig>;
```

Then `CorpusConfig.parse(YAML.parse(raw))`, and an invalid plan physically
cannot reach ingest.

---

## Ablation configurations

**`configs.yaml` is the source of truth, not this table.** It is loaded and
validated by `packages/evals/src/configs.ts`, and the table below is a
description of it that has already drifted once — the sizes here were the ones
the project started with, and the pilot moved them. Where the two disagree, the
file is right.

What the file declares now, and why it differs:

| id | Strategy | Parameters | What it tests |
|---|---|---|---|
| `fixed-128` | fixed | 128 tokens, overlap 0 | The short end, where embeddings recover their signal |
| `fixed-256` | fixed | 256, overlap 0 | Midpoint of the range where the two retrievers cross over |
| `fixed-512` | fixed | 512, overlap 0 | The original baseline, best for BM25 in the pilot |
| `fixed-512-ov128` | fixed | 512, overlap 25% | Whether overlap rescues answers split across a boundary |
| `fixed-1024-ov256` | fixed | 1024, overlap 25% | Longer context vs. precision; control for the size trend |
| `window-60` | time_window | 60 s windows | Natural transcript boundaries instead of token counts |
| `window-90-ov30` | time_window | 90 s, overlap 30 s | Same, with overlap |
| — | sentence | sentence boundaries, up to 512 | **Not implemented.** See "Configuration #6 without Python" |

Seven rows, not six. 128 and 256 were added and `sentence` was dropped, both
for reasons recorded in the comment at the top of `configs.yaml`: a pilot on 18
questions showed chunk size acting on the two retrievers in **opposite
directions**, with the interesting range below 512 rather than above it —

| retriever | 128 tokens | 512 tokens | |
|---|---|---|---|
| bm25 | 43.8% | 50.0% | longer documents hold more query terms |
| local | 37.5% | 18.8% | mean pooling washes the answer out |

— so "which chunking strategy is best" has no answer without naming the
retriever, and 1024 is unlikely to be where anything happens. It is kept as the
control that shows the trend continuing.

`window-90-ov30` and `window-60` produce the **same number of chunks**, 1,756
each: a 90-second window with 30 seconds of overlap advances 60 seconds a
window, which is exactly what a 60-second window with no overlap does. The pair
differs only in chunk length, 227 against 336 mean tokens. They are not
replicates of each other.

The full matrix is chunking × retrieval × reranking — 7 × 3 × 3 = **63 cells**,
each repeated 3 times. Not the "roughly 12 runs" this document originally
planned: that number assumed a hand-picked best-2 rather than the whole grid,
and the grid turned out cheap enough to run whole once chunking and embedding
were cached per configuration rather than per cell.

**Three times is not a formality.** Vector search ought to be deterministic,
but HNSW is approximate, and with concurrent inserts the ordering can differ.
That is a result worth recording rather than assuming.

### Reranking: Cohere only, and what that costs

A local cross-encoder is off the table in TypeScript. It means
`sentence-transformers`, so either a Python service alongside or an ONNX
runtime with a hand-rolled tokenizer — both cost more than this step is worth
in a three-week project. So Cohere Rerank is not one option among two; it is
the only one. Three consequences, stated up front rather than discovered later:

- Reranking becomes paid and networked. The cost-per-1000-queries metric grows
  a second line, and p95 latency gains an API round-trip.
- The reranker-on/off comparison stays valid, but it is a comparison against
  **one** reranker. That goes in the README as a limitation, not left unsaid.
- Determinism stops being a formality. Three runs per configuration were
  already planned; for a hosted reranker they now check whether the vendor's
  model drifts between runs. If it does, that shows up in our own data — and
  it is a genuinely interesting line in the README.

### Configuration #6 without Python

Auto-generated transcripts arrive without punctuation, so sentence chunking
makes no sense without preparation. There are no local punctuation-restoration
models in TS, which leaves two options:

- **Cheap:** treat segment boundaries as pseudo-sentences, merged up to the
  target length. Technically a variant of #4, but with uneven windows
- **More interesting:** a batched LLM call that punctuates the transcript. Done
  once, the result cached in `transcript_segments`. And then **the cost of that
  step becomes a metric too** — "is punctuation worth the money" is a
  meaningful question nobody has answered

The second is the one to build. It costs a few dollars more and yields a
separate finding for the README.

---

## Metrics

**What counts as a hit.** A gold annotation is an interval of a video and a
retrieved chunk is another interval, so "found it" needs a rule. `is_hit` is
true when `overlap / min(gold_length, chunk_length) >= 0.5`.

The alternatives are not neutral. Under *any non-zero overlap*, a 1024-token
chunk crosses more gold spans than a 512-token one regardless of relevance, and
configuration #3 wins the chunk-size comparison before a single number is
interpreted. Under *IoU >= 0.5*, a 90-second chunk that fully contains a
10-second answer scores 0.11 and misses — the rule punishes a correct chunk for
being long. Dividing by the shorter interval keeps both good cases: a chunk
containing the whole answer, and a chunk sitting inside a three-minute one.

It is not obviously the best rule. It is written down, tested, and identical
across every configuration, which matters more than which rule it is.

**Core:**
- `recall@k` for k = 1, 3, 5, 10 — the headline number
- `MRR` — how high the first correct span lands

Both are computed over the questions that have gold spans. Negatives have none:
MRR on a question with no correct answer is undefined rather than zero, and
scoring it 0 would drag the headline number down by an amount that depends only
on how many negatives the set contains. They are measured by false-positive
rate, which is what they are for.

**What tutorials leave out:**
- **contradiction coverage** — for `contradiction` questions: did top-k include
  sources from both sides (`side = pro` and `side = contra`)
- **false positive rate on negative questions** — does the system confidently
  return garbage when the answer is not in the corpus at all. Most RAG systems
  fail here and nobody measures it

  It needs a threshold below which the system declines to answer, and the
  threshold is the whole metric: a plain top-k retriever always returns k
  results, so without an abstention rule the rate is 1 by construction. It also
  needs enough negatives to say anything. On the first 32, a threshold keeping
  90% of answerable questions admitted 94% of negatives under BM25 — the score
  distributions overlap almost entirely. Under local embeddings the same
  threshold admitted 72%, and at the median answerable score 6% against BM25's
  22%.

  That is the first real trade-off the project has found: the retriever that is
  worse at recall (37.5% against 50.0%) is markedly better at knowing when to
  stay quiet. Chosen on recall alone, the system would confidently answer three
  out of four questions that have no answer.

**Cost and speed:**
- cost per 1000 queries (query embedding + rerank)
- one-off indexing cost per configuration (stored in `chunk_sets`)
- p50 / p95 latency

All three are in `pnpm report`, and three decisions about them are worth
stating because each one is a place the obvious version misleads.

**Units are recorded; dollars are computed at read time.** A run writes what
the configuration consumes — chunk texts and tokens, query texts and tokens,
rerank searches and documents — and `prices.yaml` supplies the tariff. A price
is a claim about a vendor on a date, so freezing one into the JSONL would make
a corrected price a reason to re-run the matrix. This way it is a re-read.

**The units are what the configuration REQUIRES, not what the run paid.**
Every embedding here is cached, so a second run of a cell spends nothing;
recording that would make the cost column a measurement of cache warmth, and
the same cell would cost $0 today and $4 on the machine that first ran it.

**An unpriced input makes the figure null, not a partial sum.** This project
never paid for an embedding — the Gemini work was free-tier allowance counted
in texts against a daily cap — and never paid for a rerank, since no key was
ever set. So `prices.yaml` carries nulls with reasons rather than list prices
for tiers the project did not use, and the report prints the reasons in place
of the dollars. "$0.02, reranker not counted" reads as complete and is not.

Latency is timed per question and per repeat, so p95 is a query that actually
took that long. It covers search, fusion and reranking — a cross-encoder's
round trip included — and NOT embedding the query, which happens once per run
before the loop. That half is measured separately and reported as its own
column, because a dense retriever in production pays it on every query and a
p95 describing only the second half of the path would be the flattering
number rather than the true one.

---

## Golden set: 106 questions

The most important and most tedious part. Without it the project is a demo.

- **36 factual** — the answer sits in one specific place in one video
- **22 comparative** — requires ≥2 different videos
- **18 contradiction** — different authors say the opposite, both sides labelled
- **30 negative** — the answer is not in the corpus, the correct result is empty

The first three carry `recall@k`; 36 + 22 + 18 = 76 is the number the power
calculation below is about. Negatives contribute nothing to it, so their count
is set by what false-positive rate needs rather than by proportion.

**The split between the first three is a budget, not a requirement.** Only the
total of 76 governs what the set can detect, and 18 contradictions turned out
to be the one line item the corpus may not be able to fill. Searching every
tool over the whole corpus for passages that cannot both be true returns the
same four disputes, each with several independent sources — see `pnpm pairs --relation clash`.
Four is a lower bound from one method with known blind spots: it only reads
windows that name a tool, and it rejects two claims from one author separated
in time, which `selection_rules` names as the second source of contradictions.
But it is not close to eighteen.

Two further searches confirmed the shape rather than the number. Mining the
second source — one author contradicting his own earlier video, which
`selection_rules` names and nothing was using — added five, and searching by
subject across every tool instead of by tool added one, the strongest in the
set: one claim that a million-token window removes the need to compress, denied
by three authors for three different reasons. Beyond those, every search now
returns disputes it has already returned. Eleven is what the corpus gave up
under three relations and four search axes.

The shortfall belongs in comparative questions, and the arithmetic is in
`pnpm power --contradictions 5`: at 36 + 22 + 5 the pool is 63 and the
detectable difference widens from 10.5 to 12.5 pp, while 36 + 35 + 5 is 76
again and detects the same 10.5 pp as the original design. Comparative
questions need two videos on one subject, which this corpus has in quantity;
a flat disagreement between two authors it evidently does not. Moving the
budget costs nothing measurable. Lowering the total costs 2 pp of resolution
in the range where chunking differences actually live.

**Negatives carry two topics, not one.** `absent` says the answer is not in the
corpus; the second says what the question is *about*. Of the 32 written, 19 have
a real counterpart in the corpus — an MCP question about a tool that is absent,
asked of a corpus holding 21 passages on MCP setup — and 13 do not. Those are
different questions in everything but their label, and one false-positive rate
averages them into a number that hides which half a system fails on.

The split is assigned by reading the question, never from a retriever's score.
Labelling by score would tune the set to the retriever it exists to test, which
is the mistake the ceiling gate forbids in the other direction. The hardness is
real: top-1 BM25 scores across the 32 range from 9.4 to 19.3.

A proportional split would give 14, scaled from an earlier 40-question design —
inherited arithmetic rather than a decision. The four kinds differ in both cost
and value. Negatives are the cheapest question in the set: no transcript to
read, no span to pin down, `gold: []`, and the four absent tools were verified
at zero mentions across all 78 transcripts. They are also the only input to
false-positive rate, one of the two metrics that distinguish this project.

At 14 questions a perfect result — no false positives at all — reads as
"somewhere between 0% and 23%" with a Clopper-Pearson interval. At 30 it reads
as "0% to 12%". Roughly an extra hour of writing halves the interval on a
headline number, which is the best trade available anywhere in the set.

### Why 90 and not 40

The size is a power calculation, not a round number. Negative questions have no
gold spans, so they contribute nothing to `recall@k`: the pool that carries the
configuration comparison is 36 + 22 + 18 = **76**.

| total | recall pool | detectable difference | annotation |
|---|---|---|---|
| 40 | 34 | 22.5 pp | 4 h |
| 60 | 51 | 15.5 pp | 6 h |
| **90** | **76** | **10.5 pp** | **9 h** |
| 120 | 102 | 8.0 pp | 12 h |

Recomputable with `pnpm power`, which is how the four figures above are now
checked — a test asserts them, so a change to the set cannot quietly leave the
table stale. They were written down once before anything could re-derive them.

Paired design, power 0.8, α 0.05, deterministic retrieval. Differences between
chunking strategies realistically live in the 5–15 pp range. At 40 questions the
most likely outcome is "every configuration looks the same" with no way to tell
whether that is true or simply unmeasured — a statement about the question set,
not about chunking. 120 buys 2.5 pp more for three extra hours, the worst trade
in the table.

### The ceiling gate

Size is necessary but not sufficient. If the baseline configuration scores 93%
on `recall@5`, no difference is visible at any n, because the metric is against
its ceiling. **The set is built so the baseline stays below 85%**, and that is
verified before the full ablation runs, not after.

Only the upper bound matters. An early pilot on 18 questions put the best
configuration at 50% — further from the ceiling, and so more sensitive to a
difference, not less. Below roughly 30% the thing to suspect is the pipeline
rather than the questions.

A set that comes out too easy gets harder questions, not a different metric.
Difficulty is set by choosing the lever — multi-video answers, time-separated
claims, near-miss phrasing — never by dropping the questions the baseline got
right. Dropping those tunes the set to the retriever using the retriever's own
output.

### How the questions get written

Not one at a time. One question at a time means inventing a question and then
hunting the corpus for a place that answers it — the most expensive possible
order, and backwards relative to rule 4 below.

Instead, per lever: take one `tool × topic` cell, grep the transcripts for the
tool's `aliases` (already listed in `corpus.yaml` for exactly this), read twenty
hits in a row, and write eight to ten questions from what is actually said. The
spans are on screen while the questions are being written, so the expensive half
of annotation disappears. Six tools × seven topics is more cells than 90
questions need.

Levers also give control pairs for free — the same question shape with and
without a superseding marker in the transcript ("it used to be 5 hours, now
it's 50"). That is a finding for the README, not just a bigger set.

### Generating questions, and what it costs

Writing 106 questions by hand is the largest single cost in the project, so
generation was measured rather than argued about. Five passages, one question
each, two methods, overlap measured against the original passage:

| method | mean overlap | range |
|---|---|---|
| naive — read the passage, write a question | 64% | 45-100% |
| two-stage bottleneck | 25% | 0-43% |
| written by hand, for comparison (48 questions) | ~33% | 0-70% |

The naive method inherits the passage's vocabulary, which is the failure that
matters: a question found by term matching alone scores well for every
configuration and drops out of the comparison. One of the five came out at 100%.

The bottleneck is two calls with an information gap between them. The first
states the claim in its own words; the second sees *only* that sentence, never
the passage, and writes the question. Its distribution lands on top of the
hand-written one.

It is not sufficient on its own. Overlap says nothing about whether the passage
actually answers the generated question, and a plausible-looking question about
something the passage does not address is a worse defect than high overlap,
because nothing automated catches it. So generation is followed by two checks:
the overlap thresholds, and a person reading the span to confirm the answer is
in it. The second is quick — the span is already on screen — but it does not
automate away.

**Wording is checked from both ends.** `pnpm golden` warns above 75% content-word
overlap with the span a question points at, and below 15%. Neither is a verdict:
a question about the context window has to say "context window", and one phrased
in entirely different words can still be findable by meaning.

**The floor flags two different things and cannot tell them apart.** One is a
question the corpus does not answer — a wrong annotation. The other is a good
question phrased in entirely different words from the passage that answers it.

The second is not a defect. It is the most valuable question type in the set,
because it is the only kind that separates lexical retrieval from semantic
retrieval, and separating those is what the ablation is for. One question sits
at 10% overlap, is never found by BM25 at any chunk size, and *is* found by
local embeddings at 128 tokens. Raising its overlap would delete exactly the
signal it carries.

That happened once. Both questions tripping the floor were rephrased on the
grounds that no retriever found them, and for one of them that was simply
untrue: only the 512-token runs had been checked. It has been reverted, and the
note on it now says why it stays at 10%. The other, at 0%, was found by nothing
at any size and stays rephrased.

So the floor is a prompt to check that the answer really is in the span, not a
prompt to rewrite.

The measure also does not stem, which showed up in the same place: that question
says "skills" and "replaced" where the passage says "skill" and "replacement",
so part of its 10% is an artefact rather than a real gap. It reads lower than a
person would judge — the safe direction for a warning, but worth knowing when
reading one.

### What 90 does not buy

Per-`kind` comparisons. 18 contradiction and 30 negative questions cannot
support "configuration A beats B on contradictions" — at that size the
detectable difference is far larger than any real effect. Contradiction coverage
and false-positive rate stay the metrics that distinguish this project, but they
are reported as a proportion with a Clopper-Pearson interval, not as a
comparison between configurations.

The numbers, since the claim is checkable (`pnpm power`): at 18 paired
questions the detectable difference is 40.5 pp, and a 20 pp difference — far
larger than anything chunking produces — is found 13% of the time. At the 5
written so far, no difference in the whole range from 0 to 100 pp reaches 80%
power, which is why `mdePaired` returns nothing rather than a large number.
This is not a shortfall against the design; the design never claimed
otherwise. It is worth stating in figures because a printed 2/5 against 0/5
invites exactly the comparison this section forbids, and `pnpm ceiling` used
to drop its warning about that at n=5.

### Rules

1. Questions are written **before** looking at what the system retrieves
2. Each one gets a concrete `video_id + start_s..end_s`, not "somewhere in the
   video"
3. The YAML is committed to git and `runs` records `golden_sha`. Changing the
   set makes a new version; old results are not comparable — including growing
   it. A table that mixes 76 items with 102 is two different denominators
4. Go from the corpus to the questions, not the other way round
5. The detectable difference is printed in the run header and stands above the
   results table, not in a limitations section at the bottom. The table should
   not be readable without it

---

## The runner

Two commands, and the split between them is load-bearing.

```
pnpm eval --run-all                 # every cell of configs.yaml → runs/<stamp>/*.jsonl
pnpm eval --chunking fixed-128 --retrieval bm25,vector --reranking none
pnpm eval --dry-run --run-all       # print the matrix and the cost, run nothing
pnpm report                         # aggregate the newest run directory
```

`pnpm eval` retrieves and writes; `packages/report` reads and computes. The
report package imports no retriever, no embedder and no chunker, and a test
walks its import graph to keep it that way. That is not tidiness: it is what
makes every number in the table traceable to a line in a file whose header
records the question set and the code that produced it.

**No database.** Invariants 1 and 9 describe a `chunks` table with a mandatory
`chunk_set_id`, and there isn't one — everything is derived from
`cache/transcripts/*.json` on each invocation. A JSONL file is one cell of the
matrix by construction, and its `cell` field is the chunk-set identity for
these runs. If a table ever lands, the mapping is
`chunk_set_id ↔ (chunking id, embedder)`.

**One file per cell, appended as results arrive**, so a run that dies keeps
what it did. The cost of appending is that a second run writing the same
filename extends the file rather than replacing it, giving it two headers and a
doubled question set. `pnpm eval` refuses an `--out` directory that already
holds the cells it is about to write (`--force` replaces them), and
`readCell` refuses a two-header file outright. The default timestamped
directory never collides, which is exactly why the check has to exist: the
failure is invisible with the default and reachable with `--out`.

**A skipped cell gets a file too**, with its header and a `skipped` line
carrying the reason. Never a zero and never a blank — a missing cell that looks
like a bad result is worse than an empty one.

### What actually ran, and what did not

Of 63 cells, **42 ran** and 21 did not:

| blocked | cells | reason |
|---|---|---|
| `cohere-rerank` | 21 | `COHERE_API_KEY` is not set. The client exists (`packages/retrieve/src/cohere.ts`); with a key each cell needs up to 312 billed calls and `--spend-rerank` |
| Gemini embeddings | 28 under `--embedder gemini` | The spend cap is exhausted |

So the "on the winner, reranker on/off" step of the plan is **not closed**: the
code is there and tested, the calls are not paid for. And the embedding-model
axis is not a comparison either — the whole matrix ran on the local model.

The Gemini gap is worth stating precisely, because the obvious version of it is
wrong. The disk cache covers `fixed-128` and `fixed-512` completely, 3,238 and
825 chunks. It covers **45 of the 104 question texts**. A query needs a vector
too, so a `vector` cell on `fixed-128` needs 59 new calls despite every chunk
being on disk — the pre-flight checks queries and chunks separately for that
reason. Quota is spent per text, so 59 units would unblock 8 cells; the other
20 still need their chunks embedded.

### Reranking is the one axis where the repeats really cost

Chunking, embedding and rerank calls are all cached on disk by a hash of their
input, which is what makes three repeats of 63 cells affordable — the second
full run computed 0 embeddings and read 11,505 from cache. Reranking is the
exception, and deliberately so: for a hosted model the repeats check whether
the **vendor** drifts between calls, and a cached repeat would answer that with
the first repeat's answer. So `rerank` reads the cache and `rerankFresh` does
not, and repeats past the first always spend.

### The embedding axis, closed through a different gateway

The design asks for the best configurations against two embedding models. That
step was blocked, and never on code: Gemini's free tier spends its allowance per
TEXT against a daily cap of roughly a thousand, and the corpus is 3,238 chunks
at 128 tokens.

OpenRouter bills the same work per token. The whole matrix — 3.2M tokens across
seven chunking configurations — was embedded through `baai/bge-m3` for
**$0.0355**, measured from the vendor's own `usage.cost` rather than inferred.

`baai/bge-m3` wins. Against the local model on the same 28 dense
configurations it is ahead in 26, and of the 28 paired comparisons the set can
resolve 11 at p <= 0.05 — **all 11 favouring bge-m3, none favouring local**.
The gains concentrate where the dense retriever works alone and the chunks are
short: `fixed-128__vector__none` moves 29.0% to 48.5%, `window-60__vector__none`
35.4% to 51.4%. Hybrid configurations over long chunks are indistinguishable,
which is what a fused lexical half doing most of the work looks like.

Two things follow that the local model had hidden.

**"Long chunks wash out a dense retriever" was a property of that model, not of
dense retrieval.** The local model peaked in the middle of the size range and
collapsed at 128 tokens; bge-m3 is at its best there. Any conclusion about
chunk size and embeddings drawn from a 384-dimension MiniLM would have been a
conclusion about MiniLM.

**bge-m3 abstains perfectly on this corpus.** Every `vector` configuration
under it scores a false-positive rate of 0.0% at its own median threshold — 32
unanswerable questions, none answered. The local model scored 0–3.1%. Recall
went up and the abstention did not pay for it, which is not the usual shape of
that trade-off.

Three practical findings came out of running it:

- **The token estimate is 10% low.** The dry run priced 3,216k tokens from
  gpt-tokenizer; the vendor billed 3,552k. Both counts are honest and they
  count different things — the caveat in `costUnits` now has a number on it.
- **A hosted embedder dominates query latency.** One query embedding is a
  33.8 ms round trip, against 0.3–10.2 ms for the retrieval it feeds. The
  conclusion that latency does not discriminate holds *among retrieval
  configurations*; the embedder choice moves it by an order of magnitude.
- **The vendor is not deterministic.** The same text embedded twice in one
  request differs — cosine 0.999998877, largest element difference 1.6e-4, no
  two of 1,024 values equal. Too small to move a ranking, but it means the
  three repeats verify that the disk cache is stable rather than that the
  vendor is.

One structural fix made the comparison possible at all. Cell files were named
`chunking__retrieval__reranking`, with no embedder, so a second run under a
different model overwrote the first and the two could never appear in one
table. Dense cells now carry the embedder in the filename; lexical cells, which
embed nothing, keep the bare name and are shared.

### Determinism

All three repeats of all 42 cells returned identical rankings. That is the
result the repeats exist to produce, and it is what licenses `noise = 0` in
`mdePaired` — measured rather than assumed. Every ranking sorts by score and
then by index; without the tiebreak, equal BM25 scores would make two repeats
differ and it would be indistinguishable from a real effect.

### Results

Recall pool 72 of 104 questions, so the detectable difference is **11.0 pp**,
not the 10.5 pp the table in "Why 90 and not 40" gives for 76. `pnpm report`
prints the live figure and the spec's side by side rather than reprinting a
stale number.

Best cell per retriever, at recall@5 (`/local` embeddings throughout):

| configuration | r@1 | r@5 | r@10 | MRR | FP@median |
|---|---|---|---|---|---|
| `fixed-1024-ov256__bm25__none` | 27.8% | 50.6% | 57.1% | 0.447 | 34.4% |
| `fixed-512-ov128__vector__none` | 15.3% | 45.9% | 55.2% | 0.350 | 3.1% |
| `fixed-512-ov128__hybrid__none` | 28.1% | 58.4% | 63.5% | 0.527 | 9.4% |

Four things the run established:

1. **The ceiling gate holds.** The best configuration reaches 58.4% on
   recall@5, well below 85%, so differences are measurable in principle.
2. **Chunk size still moves the retrievers in opposite directions**, which is
   why the table never collapses that axis. BM25 climbs monotonically with
   chunk size, 37.1% → 50.6%; the local model peaks in the middle.
3. **MMR remains a measured negative result.** At λ 0.7 it lost recall@5 in 18
   of 21 configurations and improved contradiction coverage in 2. It is in the
   matrix as a row worth having, not as a fix.
4. **The recall/abstention trade-off is real and large.**
   `fixed-512-ov128__hybrid__none` leads recall at 58.4% and answers 9.4% of
   the unanswerable questions; `window-60__vector__none` answers **0%** of them
   at 35.4% recall. Chosen on recall alone, the system would confidently answer
   questions that have no answer.

Contradiction coverage is 0–2 of **11** measurable questions throughout. The
design calls for 18; at 11 a difference is one or two questions, so the column
is descriptive only (invariant 14).

### One caveat on the term-matchable split

`pnpm report` prints recall@5 split by whether BM25 alone answers a question at
rank 1. The denominators differ by chunking row — 21, 23, 18, 27 — because the
label is computed from a BM25 pass over *that configuration's own chunks*, and
a question BM25 answers over 512-token chunks is not always one it answers over
128-token chunks.

So the column reads **down**, comparing retrievers within one chunking row, and
not **across** chunking rows: 79.5% (n=21) against 83.3% (n=24) compares two
different subsets of the same 72 questions. The alternative — one partition
from a single reference chunking — would make the rows comparable and would be
wrong in a worse way, labelling a question by how a configuration nobody is
measuring behaves. The report says this above the table.

---

## Three-week roadmap

**Week 1 — ingest and baseline**
- 50–100 videos: fetch transcripts, cache them on disk as JSON
- Schema in Supabase
- Configuration #1 works end-to-end: chunking → embeddings → retrieval
- Manual check on 5 questions

**Week 2 — metrics and ablation**
- Golden set of 106 questions, 76 of them carrying recall (see above).
  **Delivered as 104, with a recall pool of 72** — see "The runner" for what
  that does to the detectable difference
- `pnpm eval` and `packages/report` — written here rather than reused. The
  "existing harness" this document counted on was a JSONL runner and a
  `report.mjs` from a previous project; only `power.ts` was actually portable
- Metric code + **tests for the metric code**
- 63 configurations × 3 repeats
- Report

**Week 3 — presentation**
- README with the results table
- Deploy
- A limitations section, written honestly

---

## Traps that will eat time

**Transcripts:**
- Fetch **once and cache on disk**. On repeat runs never pull from YouTube —
  that means rate limits and a lost day
- Segments arrive in 2–8 second pieces. Merge them before chunking, otherwise a
  "chunk" is half a sentence
- Keep `manual` and `generated` transcripts apart in `videos.transcript_kind`.
  Their quality differs, and it is a useful separate cut of the results
- The same video gets reuploaded on several channels. Deduplicate on normalized
  text, not on `youtube_id`
- Some videos simply have no transcript. Handle that as a normal case, not an
  exception

**Metrics:**
- The conclusion from the previous harness repeats here: **bugs in metric code
  outnumber the differences between configurations**. Unit tests for `recall@k`
  with known-good answers come before the first table, not after
- Only compare runs with identical `golden_sha` and `git_sha`

**Embeddings:**
- `text-embedding-3-large` needs `dimensions: 1536` passed explicitly. There is
  no SDK default that suits us: without the field it returns 3072 and dies in
  `toStorage`. Dying is correct, but it should die when the config is read, not
  on the hundredth chunk — validate `dim` against the model at config load

**Tokens:**
- "512 tokens" must mean the same thing for every configuration. Use one
  tokenizer for splitting, not `text.length / 4`. Otherwise the comparison is
  invalid

**Legal:**
- Transcripts are third-party content. For a private tool and metrics that is
  fine, but a public endpoint returns **short excerpts with a timestamp and a
  link to the source**, not paragraphs. State this in the README from day one

---

## README: what goes in it, and in what order

The main artifact of the project. People read it; they do not read the code.

1. **One paragraph**: what this is and what question it answers
2. **The results table immediately**, before any install instructions:
   `configuration | recall@5 | MRR | contradiction coverage | $/1000 queries`,
   with the detectable-difference line directly above it
3. **Three findings in words** — what turned out to be unexpected
4. How to reproduce
5. **Limitations, honestly**: corpus size, bias in the video sample, what the
   system cannot do, and that reranking was measured against a single hosted
   reranker

Points 2 and 3 are what separate a portfolio project from a tutorial.
