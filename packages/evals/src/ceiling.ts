// The ceiling check.
//
//   pnpm ceiling                    # both retrievers
//   pnpm ceiling --retriever bm25   # lexical only, no model download
//
// Answers one question before the remaining questions get written: can this
// corpus separate one retrieval configuration from another at all?
//
// A baseline at 90%+ means no difference is visible at any sample size,
// because the metric is against its ceiling — and the remaining questions
// would need to be harder, so writing them first would waste the work. Only
// the upper bound matters: a baseline at 50% sits further from the ceiling and
// is more sensitive to a difference, not less. Below roughly 20-30% the thing
// to suspect is the pipeline rather than the questions.
//
// Nothing here writes to a database. The full ablation needs one; a gate does
// not, and building it first would have delayed the answer by a session.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { fixedChunks } from "../../chunking/src/fixed.ts";
import { embedLocal, LOCAL_DIM } from "../../embed/src/local.ts";
import { GeminiEmbedder, GEMINI_DIM } from "../../embed/src/gemini.ts";
import { cosine, toStorage } from "../../embed/src/storage.ts";
import { CACHE_DIR, loadCorpus } from "../../ingest/src/corpus.ts";
import type { Segment } from "../../ingest/src/text.ts";
import { buildIndex, search, tokenize } from "../../retrieve/src/bm25.ts";
import { rrf } from "../../retrieve/src/fuse.ts";
import { jaccard, mmrRerank } from "../../retrieve/src/mmr.ts";
import { goldenSha, loadGolden, type Question } from "./golden.ts";
import {
  contradictionCoverage, falsePositiveRate, lexicallyTrivial, mrr, recallAtK,
  type Retrieved,
} from "./metrics.ts";

const HELP = `
usage: pnpm ceiling [--retriever bm25|local|gemini|hybrid|both|all] [--tokens N] [--overlap N]
                    [--top-k N] [--mmr LAMBDA] [--gemini-batch N]

  --mmr LAMBDA   rerank a pool of top-k*5 for diversity. 1 is plain relevance,
                 0 ignores the query. Contradiction coverage cannot move
                 without it: plain top-k returns one side of a disagreement.

  --retriever hybrid
                 reciprocal rank fusion of bm25 and gemini. Worth running
                 because the two fail on different questions: at 128 tokens
                 gemini reached two the others placed nowhere in fifty, and
                 bm25 reached one gemini never finds at any chunk size.

  --gemini-batch N
                 texts per Gemini request, at most 100. A request spends one
                 unit of the daily quota per text, so a smaller batch is what
                 gets anything at all once the day's budget is nearly gone.
`.trimStart();

const argv = process.argv.slice(2)
  .filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));
const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        retriever: { type: "string", default: "both" },
        tokens: { type: "string", default: "512" },
        overlap: { type: "string", default: "0" },
        "top-k": { type: "string", default: "10" },
        mmr: { type: "string" },
        "gemini-batch": { type: "string" },
      },
      allowPositionals: false,
    });
  } catch (e) {
    console.error(`${(e as Error).message}\n\n${HELP}`);
    process.exit(2);
  }
})();

const which = values.retriever!;
if (!["bm25", "local", "gemini", "hybrid", "both", "all"].includes(which)) {
  console.error(`--retriever must be bm25, local, gemini, hybrid, both or all\n\n${HELP}`);
  process.exit(2);
}
/** Reported in the results. */
const wants = (name: string) =>
  which === name || which === "all" ||
  (which === "both" && name !== "gemini" && name !== "hybrid");
/**
 * Built at all. Fusion needs both of its inputs computed even when only the
 * fused ranking is asked for, so this is deliberately wider than `wants`.
 */
const needs = (name: string) =>
  wants(name) || (wants("hybrid") && (name === "bm25" || name === "gemini"));
const targetTokens = Number(values.tokens);
const overlapTokens = Number(values.overlap);
const topK = Number(values["top-k"]);
const lambda = values.mmr === undefined ? null : Number(values.mmr);
if (lambda !== null && !(lambda >= 0 && lambda <= 1)) {
  console.error(`--mmr must be a number in [0,1]\n\n${HELP}`);
  process.exit(2);
}
// MMR can only promote something it was given, so it reranks a wider pool than
// it returns. Without this it would reorder the same top-k and never reach the
// other side of a disagreement, which is the entire reason it is here.
const poolK = lambda === null ? topK : topK * 5;

const cfg = loadCorpus();
const golden = loadGolden();
if (golden.questions.length === 0) {
  console.error("golden/questions.yaml is empty — nothing to measure.");
  process.exit(2);
}

// ─── chunk ───────────────────────────────────────────────────────────────────

type IndexedChunk = Retrieved & { text: string };
const chunks: IndexedChunk[] = [];

for (const v of cfg.videos) {
  const segments: Segment[] = JSON.parse(
    readFileSync(join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`), "utf8"),
  ).segments;
  for (const c of fixedChunks(segments, targetTokens, overlapTokens)) {
    chunks.push({
      video: v.youtube_id,
      start_s: c.start_s,
      end_s: c.end_s,
      text: c.text,
      score: 0,
    });
  }
}

const tokenTotal = chunks.length * targetTokens;
console.log(
  `corpus: ${cfg.videos.length} videos → ${chunks.length} chunks ` +
  `(${targetTokens} tokens, overlap ${overlapTokens})`,
);
console.log(`golden: ${golden.questions.length} questions, sha ${goldenSha(golden)}\n`);

// ─── retrievers ──────────────────────────────────────────────────────────────

type Retriever = { name: string; run: (query: string) => Promise<Retrieved[]> };
const retrievers: Retriever[] = [];

// Ranked chunk indices from each side of the fusion, kept so `hybrid` can
// consult them without re-running anything.
let lexPool: ((query: string) => number[]) | null = null;
let vecPool: ((query: string) => Promise<number[]>) | null = null;

if (needs("bm25")) {
  const index = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
  const tokenSets = chunks.map((c) => new Set(tokenize(c.text)));
  lexPool = (query) => search(index, query, poolK).map(({ id }) => id);
  if (wants("bm25")) retrievers.push({
    name: lambda === null ? "bm25" : `bm25+mmr${lambda}`,
    run: async (query) => {
      const pool = search(index, query, poolK)
        .map(({ id, score }) => ({ id, chunk: { ...chunks[id]!, score } }));
      if (lambda === null) return pool.map((p) => p.chunk);
      // BM25 scores are unbounded while jaccard lives in [0,1], so lambda would
      // not mean what it looks like without rescaling the relevance side.
      const top = pool[0]?.chunk.score ?? 1;
      return mmrRerank(
        pool.map((p) => ({ item: p, relevance: top === 0 ? 0 : p.chunk.score / top })),
        (a, b) => jaccard(tokenSets[a.id]!, tokenSets[b.id]!),
        { lambda, k: topK },
      ).map((r) => r.item.chunk);
    },
  });
}

if (needs("local")) {
  process.stdout.write("embedding chunks locally");
  const vectors = (await embedLocal(chunks.map((c) => c.text), {
    onProgress: (done, total) => {
      if (done % 320 === 0 || done === total) process.stdout.write(`\rembedding chunks locally  ${done}/${total}`);
    },
  })).map((v) => toStorage(v, LOCAL_DIM));
  process.stdout.write("\n\n");

  if (wants("local")) retrievers.push({
    name: lambda === null ? "local" : `local+mmr${lambda}`,
    run: async (query) => {
      const [q] = await embedLocal([query]);
      const padded = toStorage(q!, LOCAL_DIM);
      const pool = chunks
        .map((c, i) => ({ id: i, chunk: { ...c, score: cosine(padded, vectors[i]!) } }))
        // id as tiebreak: three runs per configuration are planned, and an
        // unstable order would be indistinguishable from a real effect.
        .sort((a, b) => b.chunk.score - a.chunk.score || a.id - b.id)
        .slice(0, poolK);
      if (lambda === null) return pool.slice(0, topK).map((p) => p.chunk);
      // Both sides are cosine here, so lambda means what it says without rescaling.
      return mmrRerank(
        pool.map((p) => ({ item: p, relevance: p.chunk.score })),
        (a, b) => cosine(vectors[a.id]!, vectors[b.id]!),
        { lambda, k: topK },
      ).map((r) => r.item.chunk);
    },
  });
}

if (needs("gemini")) {
  // 1536 dimensions by request, which is the storage width exactly — the one
  // configuration where toStorage has nothing to do.
  const embedder = new GeminiEmbedder(join(CACHE_DIR, "embeddings"));
  process.stdout.write("embedding chunks with gemini");
  const vectors = await embedder.embedAll(chunks.map((c) => c.text), {
    batchSize: values["gemini-batch"] ? Number(values["gemini-batch"]) : undefined,
    onProgress: (done, total) => process.stdout.write(`\rembedding chunks with gemini  ${done}/${total}`),
  });
  process.stdout.write(
    `\n  ${embedder.usage.calls} requests, ${embedder.usage.texts} embedded, ` +
    `${embedder.usage.cached} from cache` +
    (embedder.usage.waitedMs ? `, ${Math.round(embedder.usage.waitedMs / 1000)}s waiting on quota` : "") +
    `\n\n`,
  );

  vecPool = async (query) => {
    const [q] = await embedder.batch([query]);
    const padded = toStorage(q!, GEMINI_DIM);
    return chunks
      .map((c, i) => ({ id: i, score: cosine(padded, vectors[i]!) }))
      .sort((a, b) => b.score - a.score || a.id - b.id)
      .slice(0, poolK)
      .map(({ id }) => id);
  };

  if (wants("gemini")) retrievers.push({
    name: lambda === null ? "gemini" : `gemini+mmr${lambda}`,
    run: async (query) => {
      const [q] = await embedder.batch([query]);
      const padded = toStorage(q!, GEMINI_DIM);
      const pool = chunks
        .map((c, i) => ({ id: i, chunk: { ...c, score: cosine(padded, vectors[i]!) } }))
        .sort((a, b) => b.chunk.score - a.chunk.score || a.id - b.id)
        .slice(0, poolK);
      if (lambda === null) return pool.slice(0, topK).map((p) => p.chunk);
      return mmrRerank(
        pool.map((p) => ({ item: p, relevance: p.chunk.score })),
        (a, b) => cosine(vectors[a.id]!, vectors[b.id]!),
        { lambda, k: topK },
      ).map((r) => r.item.chunk);
    },
  });
}

if (wants("hybrid")) {
  if (!lexPool || !vecPool) throw new Error("hybrid needs both bm25 and gemini built");
  const lex = lexPool, vec = vecPool;
  retrievers.push({
    name: "hybrid",
    run: async (query) => {
      // Ids as strings only because fusion is generic over what it ranks; the
      // fused score replaces both input scores, which are on scales that
      // cannot be compared and so are deliberately not carried through.
      const fused = rrf([lex(query).map(String), (await vec(query)).map(String)]);
      return fused.slice(0, topK).map(({ id, score }) => ({ ...chunks[Number(id)]!, score }));
    },
  });
}

// ─── measure ─────────────────────────────────────────────────────────────────

// Which questions term matching alone already answers. Computed once, from a
// plain BM25 run over the same chunks, and used only to split the report — a
// question every configuration gets right cannot separate them, and averaging
// it in hides how much the harder half differs.
const lexIndex = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
const trivial = new Map<string, boolean>();
for (const q of golden.questions) {
  if (q.gold.length === 0) continue;
  const ranked = search(lexIndex, q.text, 1).map(({ id, score }) => ({ ...chunks[id]!, score }));
  trivial.set(q.slug, lexicallyTrivial(q.gold, ranked) ?? false);
}
const nTrivial = [...trivial.values()].filter(Boolean).length;
console.log(
  `lexically trivial: ${nTrivial} of ${trivial.size} questions are answered by ` +
  `term matching at rank 1\n`,
);

const KS = [1, 3, 5, 10];
const withGold = golden.questions.filter((q) => q.gold.length > 0);
const negatives = golden.questions.filter((q) => q.gold.length === 0);

type PerQuestion = { q: Question; ranked: Retrieved[] };

for (const r of retrievers) {
  const results: PerQuestion[] = [];
  for (const q of golden.questions) results.push({ q, ranked: await r.run(q.text) });

  console.log(`── ${r.name} ${"─".repeat(56 - r.name.length)}`);

  const line = KS.map((k) => {
    const scores = results
      .filter(({ q }) => q.gold.length > 0)
      .map(({ q, ranked }) => recallAtK(q.gold, ranked, k)!)
      ;
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    return `recall@${k} ${(mean * 100).toFixed(1)}%`;
  }).join("   ");
  console.log(`  ${line}   (n=${withGold.length})`);

  const mrrs = results
    .filter(({ q }) => q.gold.length > 0)
    .map(({ q, ranked }) => mrr(q.gold, ranked)!);
  console.log(`  MRR ${(mrrs.reduce((a, b) => a + b, 0) / mrrs.length).toFixed(3)}`);

  // The split that matters. Every configuration gets the trivial questions, so
  // an average over both halves understates the difference between them.
  for (const [label, want] of [["term-matchable", true], ["needs more", false]] as const) {
    const subset = results.filter(({ q }) => q.gold.length > 0 && trivial.get(q.slug) === want);
    if (subset.length === 0) continue;
    const r5 = subset.map(({ q, ranked }) => recallAtK(q.gold, ranked, 5)!);
    console.log(
      `  recall@5 on ${label.padEnd(14)} ${((r5.reduce((a, b) => a + b, 0) / r5.length) * 100).toFixed(1)}%` +
      `   (n=${subset.length})`,
    );
  }

  // Reported at both cut-offs. A single number at k=5 shows zero where the
  // truth is "the second side was two ranks lower", and those are different
  // problems: one is ranking, the other is retrieval.
  const cov = (k: number) => {
    const c = results
      .map(({ q, ranked }) => contradictionCoverage(q.gold, ranked, k))
      .filter((x): x is boolean => x !== null);
    return `${c.filter(Boolean).length}/${c.length}`;
  };
  const measurable = results
    .map(({ q, ranked }) => contradictionCoverage(q.gold, ranked, 5))
    .filter((x) => x !== null).length;
  // The caveat is tied to the target from the power calculation, not to a
  // round number. It used to lift at 5, which is where the figure stops being
  // absurd and starts being merely unusable: 2 of 5 against 0 of 5 is one
  // question either way. Silence from the tool at that point reads as
  // permission to compare, so it stays until the set can actually support one.
  const CONTRADICTION_TARGET = 18;
  console.log(
    `  contradiction coverage — @5 ${cov(5)}, @10 ${cov(10)}` +
    (measurable < CONTRADICTION_TARGET
      ? `  (${measurable} of ${CONTRADICTION_TARGET} written — descriptive only, ` +
        `a difference here is one or two questions)`
      : ""),
  );

  // Thresholds come from this retriever's own score distribution, not from a
  // fixed 0-1 grid. Cosine lives in [-1,1] and BM25 does not live anywhere in
  // particular — the first version printed "100% at every threshold" for BM25
  // because its top-1 scores were 10.15 and 13.54, off the end of the grid.
  const negTop = results
    .filter(({ q }) => q.gold.length === 0)
    .map(({ ranked }) => ranked[0]?.score ?? 0);
  const answered = results
    .filter(({ q }) => q.gold.length > 0)
    .map(({ ranked }) => ranked[0]?.score ?? 0)
    .sort((a, b) => a - b);
  const quantile = (p: number) =>
    answered.length === 0 ? 0 : answered[Math.min(answered.length - 1, Math.floor(p * answered.length))]!;

  console.log(
    `  false positives on ${negatives.length} negatives, at thresholds taken from\n` +
    `  the score distribution of the answerable questions:`,
  );
  for (const p of [0.1, 0.25, 0.5]) {
    const t = quantile(p);
    const fp = (falsePositiveRate(negTop, t) ?? 0) * 100;
    console.log(`    threshold ${t.toFixed(3)} (p${(p * 100).toFixed(0)} of answerable) → ${fp.toFixed(0)}%`);
  }
  console.log(`  top-1 on negatives: ${negTop.map((s) => s.toFixed(3)).join(", ")}`);
  console.log(
    `  top-1 on answerable: min ${quantile(0).toFixed(3)}, median ${quantile(0.5).toFixed(3)}`,
  );

  // Per-question, so a metric that says "bad" can be checked against the eye.
  console.log("\n  per question (recall@5):");
  for (const { q, ranked } of results) {
    const rec = recallAtK(q.gold, ranked, 5);
    const mark = rec === null ? "  n/a" : rec === 1 ? "  ✓  " : rec === 0 ? "  ✗  " : ` ${rec.toFixed(2)}`;
    console.log(`   ${mark} ${q.kind.padEnd(14)} ${q.slug}`);
  }
  console.log();
}

// ─── verdict ─────────────────────────────────────────────────────────────────

console.log("─".repeat(64));
console.log(
  "Gate: baseline recall@5 below 85%. Above it, no configuration difference is\n" +
  "measurable at any sample size, whatever n. There is no lower bound — further\n" +
  "from the ceiling is more sensitive, not less. Below roughly 30%, suspect the\n" +
  "pipeline before the questions.",
);
