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

| # | Strategy | Parameters | What it tests |
|---|---|---|---|
| 1 | fixed | 512 tokens, overlap 0 | Baseline |
| 2 | fixed | 512, overlap 25% | Whether overlap helps on conversational text |
| 3 | fixed | 1024, overlap 25% | Longer context vs. precision |
| 4 | time_window | 60 s windows | Natural transcript boundaries |
| 5 | time_window | 90 s, overlap 30 s | Same, with overlap |
| 6 | sentence | sentence boundaries, up to 512 | See below — needs punctuation |

Then: **the best 2 configurations × 2 embedding models**, and on the winner,
reranker on/off. Roughly 12 runs in total, each repeated 3 times.

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

## Three-week roadmap

**Week 1 — ingest and baseline**
- 50–100 videos: fetch transcripts, cache them on disk as JSON
- Schema in Supabase
- Configuration #1 works end-to-end: chunking → embeddings → retrieval
- Manual check on 5 questions

**Week 2 — metrics and ablation**
- Golden set of 106 questions, 76 of them carrying recall (see above)
- Wire up the existing JSONL runner and `report.mjs`
- Metric code + **tests for the metric code**
- 12 configurations × 3 runs
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
