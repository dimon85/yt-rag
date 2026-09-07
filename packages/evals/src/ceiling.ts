// The ceiling check.
//
//   pnpm ceiling                    # both retrievers
//   pnpm ceiling --retriever bm25   # lexical only, no model download
//
// Answers one question before the remaining questions get written: can this
// corpus separate one retrieval configuration from another at all?
//
// A baseline at 90%+ means no difference is visible at any sample size,
// because the metric is against its ceiling — and the remaining 72 questions
// would need to be harder, so writing them first would waste the work. A
// baseline near 20% means a broken pipeline rather than hard questions. The
// target band is 60-85%.
//
// Nothing here writes to a database. The full ablation needs one; a gate does
// not, and building it first would have delayed the answer by a session.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { fixedChunks } from "../../chunking/src/fixed.ts";
import { embedLocal, LOCAL_DIM } from "../../embed/src/local.ts";
import { cosine, toStorage } from "../../embed/src/storage.ts";
import { CACHE_DIR, loadCorpus } from "../../ingest/src/corpus.ts";
import type { Segment } from "../../ingest/src/text.ts";
import { buildIndex, search } from "../../retrieve/src/bm25.ts";
import { goldenSha, loadGolden, type Question } from "./golden.ts";
import {
  contradictionCoverage, falsePositiveRate, mrr, recallAtK, type Retrieved,
} from "./metrics.ts";

const HELP = `
usage: pnpm ceiling [--retriever bm25|local|both] [--tokens N] [--overlap N] [--top-k N]
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
      },
      allowPositionals: false,
    });
  } catch (e) {
    console.error(`${(e as Error).message}\n\n${HELP}`);
    process.exit(2);
  }
})();

const which = values.retriever!;
if (!["bm25", "local", "both"].includes(which)) {
  console.error(`--retriever must be bm25, local or both\n\n${HELP}`);
  process.exit(2);
}
const targetTokens = Number(values.tokens);
const overlapTokens = Number(values.overlap);
const topK = Number(values["top-k"]);

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

if (which === "bm25" || which === "both") {
  const index = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
  retrievers.push({
    name: "bm25",
    run: async (query) =>
      search(index, query, topK).map(({ id, score }) => ({ ...chunks[id]!, score })),
  });
}

if (which === "local" || which === "both") {
  process.stdout.write("embedding chunks locally");
  const vectors = (await embedLocal(chunks.map((c) => c.text), {
    onProgress: (done, total) => {
      if (done % 320 === 0 || done === total) process.stdout.write(`\rembedding chunks locally  ${done}/${total}`);
    },
  })).map((v) => toStorage(v, LOCAL_DIM));
  process.stdout.write("\n\n");

  retrievers.push({
    name: "local",
    run: async (query) => {
      const [q] = await embedLocal([query]);
      const padded = toStorage(q!, LOCAL_DIM);
      return chunks
        .map((c, i) => ({ ...c, score: cosine(padded, vectors[i]!) }))
        // id as tiebreak: three runs per configuration are planned, and an
        // unstable order would be indistinguishable from a real effect.
        .sort((a, b) => b.score - a.score || chunks.indexOf(a) - chunks.indexOf(b))
        .slice(0, topK);
    },
  });
}

// ─── measure ─────────────────────────────────────────────────────────────────

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

  const coverage = results
    .map(({ q, ranked }) => contradictionCoverage(q.gold, ranked, 5))
    .filter((c): c is boolean => c !== null);
  console.log(
    `  contradiction coverage@5 ${coverage.filter(Boolean).length}/${coverage.length}` +
    (coverage.length < 5 ? "  (too few to compare configurations — descriptive only)" : ""),
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
  "Gate: baseline recall@5 in 60-85%. Above it, no configuration difference\n" +
  "is measurable at any sample size. Below ~30%, suspect the pipeline before\n" +
  "the questions.",
);
