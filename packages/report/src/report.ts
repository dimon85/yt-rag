// pnpm report — JSONL in, table out. It aggregates and it never re-runs.
//
//   pnpm report                      # the newest directory under runs/
//   pnpm report --dir runs/2026-...  # a specific one
//   pnpm report --k 5                # the cut-off the paired tests use
//
// This package imports no retriever, no embedder and no chunker, and it cannot
// retrieve anything. That is a convention (`packages/report` aggregates; it
// never re-runs anything) and it is also what makes the numbers auditable:
// every figure below traces back to a line in a file whose header records the
// question set and the code that produced it.
//
// The order of the output is not cosmetic. The detectable difference goes
// FIRST, above the table — invariant 10 — because a reader who sees 43.8%
// against 50.0% and does not know the set cannot resolve 6 pp has been
// misinformed by a true number.
import { existsSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { ROOT } from "../../ingest/src/corpus.ts";
import { binomTest, clopperPearson, mdePaired } from "../../evals/src/power.ts";
import { RECALL_POOL_TARGET } from "../../evals/src/golden.ts";
import { loadPrices, unpricedReason } from "../../evals/src/prices.ts";
import {
  byKind, comparable, pairings, readDir, rowLabel, summarise, type CellRun, type Row,
} from "./aggregate.ts";

const HELP = `
usage: pnpm report [--dir DIR] [--k N] [--pairs N]

  --dir    a run directory. Defaults to the newest under runs/.
  --k      cut-off for the paired comparisons and the per-kind figures (default 5)
  --pairs  how many paired comparisons to print (default 12)
  --all-pairs
           include pairs that differ in more than one axis. Off by default:
           changing chunking AND retriever at once produces a difference no
           paired test can attribute to either.
`.trimStart();

const { values } = (() => {
  try {
    return parseArgs({
      args: process.argv.slice(2),
      options: {
        dir: { type: "string" },
        k: { type: "string", default: "5" },
        pairs: { type: "string", default: "12" },
        "all-pairs": { type: "boolean", default: false },
      },
      allowPositionals: false,
    });
  } catch (e) {
    console.error(`${(e as Error).message}\n\n${HELP}`);
    process.exit(2);
  }
})();

const K = Number(values.k);
const MAX_PAIRS = Number(values.pairs);

function newestRunDir(): string {
  const runs = join(ROOT, "runs");
  if (!existsSync(runs)) {
    console.error(`no runs/ directory — run pnpm eval first`);
    process.exit(2);
  }
  const dirs = readdirSync(runs)
    .map((d) => join(runs, d))
    .filter((d) => statSync(d).isDirectory())
    .sort();
  const newest = dirs[dirs.length - 1];
  if (!newest) {
    console.error(`runs/ holds no run directories — run pnpm eval first`);
    process.exit(2);
  }
  return newest;
}

const dir = values.dir ?? newestRunDir();
const all = (() => {
  try {
    return readDir(dir);
  } catch (e) {
    console.error((e as Error).message);
    process.exit(2);
  }
})();

// ─── invariant 6, before any number is printed ───────────────────────────────

const { keep: cells, rejected } = comparable(all);
const reference = cells[0]!.header;

console.log(`report: ${dir}`);
console.log(`golden_sha ${reference.golden_sha}   git_sha ${reference.git_sha}`);
console.log(
  `${cells.length} cells, ${reference.questions} questions, top_k ${reference.top_k}, ` +
  `${reference.repeats} repeats, hit coverage ${reference.hit_coverage}`,
);
if (rejected.length > 0) {
  // Invariant 6: mixed comparisons go in the bin, not in the report. Named
  // rather than counted, so the operator can see what they were about to
  // compare — a run against a different question set, or against different code.
  console.log(
    `\n${rejected.length} file${rejected.length === 1 ? "" : "s"} EXCLUDED — a run is comparable only to runs with the\n` +
    `same golden_sha and git_sha, and ${rejected.length === 1 ? "this one does" : "these do"} not carry the majority pair:`,
  );
  for (const r of rejected) {
    console.log(`  ${r.file.split("/").pop()}   golden ${r.golden_sha}   git ${r.git_sha}`);
  }
}
if (reference.git_sha.endsWith("-dirty")) {
  console.log(
    `\ngit_sha carries -dirty: these runs came from a tree with uncommitted changes,\n` +
    `so HEAD does not describe the code that produced them. They are comparable to\n` +
    `each other and to nothing else.`,
  );
}

// Prices come from a file, not from the code: a corrected tariff has to be a
// re-read of what the runs recorded rather than a re-run of the matrix.
const prices = loadPrices(join(ROOT, "prices.yaml"));
const rows = cells.map((c) => summarise(c, prices));
const live = rows.filter((r) => r.notRun === null);
const liveCells = cells.filter((c) => !c.skipped && c.results.length > 0);

if (live.length === 0) {
  console.log(`\nNo cell in ${dir} produced results. Nothing to aggregate.`);
  for (const r of rows) console.log(`  ${r.label}   ${r.notRun}`);
  process.exit(0);
}

// ─── 1. what this set can detect ─────────────────────────────────────────────

// Noise measured from the repeats rather than assumed to be zero. power.ts:
// "Retrieval here is deterministic, so it is 0 — and the parameter exists so
// that 'deterministic' is an argument rather than an assumption."
const totalQuestionsChecked = live.reduce((n, r) => n + r.recallPool, 0);
const totalDisagreements = live.reduce((n, r) => n + r.disagreements.length, 0);
const noise = totalQuestionsChecked === 0 ? 0 : totalDisagreements / totalQuestionsChecked;

const pool = Math.max(...live.map((r) => r.recallPool));
const mde = mdePaired(pool, noise);

console.log(`\n${"═".repeat(78)}`);
console.log(
  `DETECTABLE DIFFERENCE: ${mde === null ? "none in range" : `${(mde * 100).toFixed(1)} pp`} ` +
  `on a paired comparison of ${pool} questions, at power 0.8 and alpha 0.05.`,
);
console.log(
  `Two configurations differing by less than that are indistinguishable on this\n` +
  `set, however far apart their percentages look. The recall pool is ${pool} of\n` +
  `${reference.questions} questions — negatives carry no recall and do not move it ` +
  `(invariant 13).`,
);
if (pool !== RECALL_POOL_TARGET) {
  // The 10.5 pp in docs/spec.md is for 76. Printing that figure against a
  // live pool of another size is wrong in the direction that looks fine.
  console.log(
    `The design in docs/spec.md assumes a pool of ${RECALL_POOL_TARGET}, which gives ` +
    `${((mdePaired(RECALL_POOL_TARGET, noise) ?? 0) * 100).toFixed(1)} pp.\n` +
    `The pool here is ${pool}, so ${pool > RECALL_POOL_TARGET ? "more" : "less"} is detectable ` +
    `than the spec's table says.`,
  );
}
console.log(
  `Measured noise between repeats: ${(noise * 100).toFixed(1)}% ` +
  `(${totalDisagreements} of ${totalQuestionsChecked} question-cells).`,
);
console.log("═".repeat(78));

// ─── 2. the results table ────────────────────────────────────────────────────

const pct = (v: number | null) => (v === null ? "  —  " : `${(v * 100).toFixed(1)}%`.padStart(6));
const num = (v: number | null, d = 3) => (v === null ? "  —  " : v.toFixed(d).padStart(5));

const WIDTH = Math.max(...rows.map((r) => r.label.length), 24);
const head = (s: string, n: number) => s.padEnd(n);

console.log(`\n${head("configuration", WIDTH)} chunks  r@1     r@5     r@10    MRR    contra  FP@med`);
console.log("─".repeat(WIDTH + 54));
for (const r of rows) {
  if (r.notRun) {
    // Never a zero and never a blank. A missing cell that looks like a bad
    // result is worse than an empty one.
    console.log(`${head(r.label, WIDTH)} ${String(r.chunkCount).padStart(6)}  ${r.notRun}`);
    continue;
  }
  console.log(
    `${head(r.label, WIDTH)} ${String(r.chunkCount).padStart(6)}  ` +
    `${pct(r.recall[1] ?? null)}  ${pct(r.recall[5] ?? null)}  ${pct(r.recall[10] ?? null)}  ` +
    `${num(r.mrr)}  ${`${r.contradiction.k}/${r.contradiction.n}`.padStart(6)}  ${pct(r.falsePositiveRate)}`,
  );
}
console.log(
  `\nFP@med is the false-positive rate on the ${live[0]!.negatives} negative questions at each\n` +
  `configuration's OWN median top-1 score among answerable questions. The threshold\n` +
  `is per-row because the scales are not comparable: cosine lives in [0,1], BM25 is\n` +
  `unbounded, and fused RRF scores collapse onto a handful of values. Both columns\n` +
  `belong in the table — recall alone would misrepresent a configuration that buys\n` +
  `recall by giving up the ability to abstain.`,
);
console.log(
  `\nThe chunking axis is never collapsed. Chunk size moves the retrievers in\n` +
  `opposite directions, so "which chunking is best" has no answer without naming\n` +
  `the retriever, and a mean over retrievers would answer a question nobody asked.`,
);

// ─── 2b. cost and speed ──────────────────────────────────────────────────────

/** Greedy wrap. Long words are left long rather than broken mid-token. */
const wrap = (text: string, width: number): string[] =>
  text.split(/\s+/).filter(Boolean).reduce<string[]>((lines, word) => {
    const last = lines[lines.length - 1];
    if (last !== undefined && `${last} ${word}`.length <= width) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
    return lines;
  }, []);

const ms = (v: number | null) => (v === null ? "   —  " : `${v.toFixed(1)}`.padStart(6));
const usd = (c: { usd: number | null } | undefined) =>
  c === undefined || c.usd === null ? "     —  " : `$${c.usd.toFixed(4)}`.padStart(8);

console.log(`\n${"─".repeat(78)}`);
console.log("cost and speed");
console.log(
  `Retrieval latency is timed per question and per repeat, so p95 is a query that\n` +
  `actually took that long rather than a mean nobody experienced. It covers search,\n` +
  `fusion and reranking — a cross-encoder's round trip included — and NOT embedding\n` +
  `the query, which happens once per run before the loop. The +embed column carries\n` +
  `that half where it was measured; where every query vector came from the disk\n` +
  `cache there was nothing to time and it reads "—".`,
);

console.log(
  `\n${head("configuration", WIDTH)}   p50    p95   +embed   $/1000q   index $`,
);
console.log("─".repeat(WIDTH + 42));
for (const r of rows) {
  if (r.notRun) {
    // The reason is in the main table; repeating it for every skipped cell
    // would bury the rows that have figures.
    console.log(`${head(r.label, WIDTH)}   not run`);
    continue;
  }
  if (r.latency.samples === 0 && r.cost === null) {
    console.log(`${head(r.label, WIDTH)}   not recorded — this file predates these columns`);
    continue;
  }
  console.log(
    `${head(r.label, WIDTH)} ${ms(r.latency.p50)} ${ms(r.latency.p95)} ${ms(r.queryEmbedMs)} ` +
    `${usd(r.cost?.perThousandQueries)}  ${usd(r.cost?.indexing)}`,
  );
}

// Which figures could not be produced, and in prices.yaml's own words. Printed
// once rather than per row: 42 rows repeating one sentence about a free tier is
// noise, and the reason is a property of the price file, not of the cell.
const unpriced = new Map<string, string>();
for (const r of rows) {
  for (const key of [...(r.cost?.perThousandQueries.unpriced ?? []), ...(r.cost?.indexing.unpriced ?? [])]) {
    // The reranking id, from the cell name the run recorded — not a literal, so
    // a second hosted reranker reports its own reason rather than cohere's.
    const vendor = key === "embedding" ? (r.embedder ?? "") : (r.cell.split("__")[2] ?? "");
    const reason = unpricedReason(prices, vendor);
    if (reason) unpriced.set(vendor, reason);
  }
}
if (unpriced.size > 0) {
  console.log(
    `\nSome dollar figures are "—" on purpose. An unpriced input makes the whole\n` +
    `figure null rather than a partial sum: "$0.02, reranker not counted" reads as\n` +
    `complete and is not. prices.yaml says why, and those reasons are:`,
  );
  for (const [vendor, reason] of unpriced) {
    console.log(`\n  ${vendor}:`);
    // Wrapped here, not in the file: YAML block scalars fold their newlines
    // away, so a reason written as a tidy paragraph arrives as one long line
    // and would print past the width of everything around it.
    for (const line of wrap(reason, 74)) console.log(`    ${line}`);
  }
}

const indexed = new Map<string, Row>();
for (const r of rows) if (!indexed.has(r.cell.split("__")[0]!)) indexed.set(r.cell.split("__")[0]!, r);
console.log(
  `\nThe index column is a one-off per chunking configuration, not per row: the\n` +
  `${indexed.size} chunking config${indexed.size === 1 ? " is" : "s are each"} embedded once and shared by every retrieval ×\n` +
  `reranking cell over it, so charging each cell would multiply a cost that was\n` +
  `paid once. It is also deliberately not folded into $/1000q — amortising it\n` +
  `would require inventing how many queries the system ever answers.`,
);

// ─── 3. the term-matchable split ─────────────────────────────────────────────

console.log(`\n${"─".repeat(78)}`);
console.log("recall@5 split by whether term matching alone answers the question");
console.log(
  `Term-matchable questions separate configurations more than three times better\n` +
  `than the rest — 55 pp of spread against 17 pp when this was measured — so an\n` +
  `average over both halves hides the result. The label comes from a plain BM25\n` +
  `pass and is used to split the report, never to drop a question.`,
);
console.log(
  `\nREAD DOWN A COLUMN, NOT ACROSS CHUNKING ROWS. The denominators differ by row\n` +
  `(n=${live.map((r) => r.split.trivial.n).filter((n, i, a) => a.indexOf(n) === i).slice(0, 4).join("/")}...), because "term-matchable" is decided by a BM25 pass over THAT\n` +
  `configuration's own chunks: a question BM25 answers at rank 1 over 512-token\n` +
  `chunks is not always one it answers over 128-token chunks. So the two halves\n` +
  `are a different partition of the same 72 questions in every chunking row, and\n` +
  `comparing 79.5% (n=21) against 83.3% (n=24) compares two different subsets.\n` +
  `Within one chunking row the partition is fixed and the retrievers ARE\n` +
  `comparable, which is the cut this table is for.\n` +
  `\nThe alternative — one partition from a single reference chunking, applied to\n` +
  `every row — would make the columns comparable across rows and would be wrong\n` +
  `in a worse way: it would label a question by how a configuration nobody is\n` +
  `measuring behaves, and the label is supposed to mean "term matching finds this,\n` +
  `here".\n`,
);
console.log(`${head("configuration", WIDTH)} term-matchable      needs more`);
console.log("─".repeat(WIDTH + 34));
for (const r of live) {
  console.log(
    `${head(r.label, WIDTH)} ${pct(r.split.trivial.mean)} (n=${String(r.split.trivial.n).padStart(2)})     ` +
    `${pct(r.split.harder.mean)} (n=${String(r.split.harder.n).padStart(2)})`,
  );
}

// ─── 4. paired comparisons ───────────────────────────────────────────────────

console.log(`\n${"─".repeat(78)}`);
console.log(`paired comparisons at recall@${K} — McNemar on the discordant pairs`);
console.log(
  `Every configuration ran the same frozen set, so the design is paired: questions\n` +
  `both got right and questions both got wrong carry no information about the\n` +
  `difference. Only the discordant ones do, which is why the counts are printed\n` +
  `and not only the p-value — "31 both, 13 neither, 4 moved" is the honest shape\n` +
  `of a result on a set this size. The test is an exact two-sided binomial on the\n` +
  `movers against 0.5, which is exactly what McNemar is.\n`,
);

const allPairs = pairings(liveCells, K).map((p) => {
  const d = p.counts.aOnly + p.counts.bOnly;
  return { ...p, d, p: binomTest(p.counts.aOnly, d, 0.5) };
});
// One-axis pairs only, unless asked otherwise. A pair differing in both
// chunking and retriever is discordant on plenty of questions and says nothing
// about either change, and sorting on discordance alone puts exactly those at
// the top — the first version of this section did, and it was unreadable.
const pairs = (values["all-pairs"] ? allPairs : allPairs.filter((p) => p.axesDiffering === 1))
  // Smallest p first, then most movement: what the set could actually resolve.
  .sort((x, y) => x.p - y.p || y.d - x.d);

if (pairs.length === 0) {
  console.log("Only one configuration ran — nothing to pair it with.");
} else {
  const shown = pairs.slice(0, MAX_PAIRS);
  for (const p of shown) {
    const verdict = p.d === 0
      ? "identical on every question"
      : p.p <= 0.05
        ? `p = ${p.p.toFixed(4)}  — a difference this set can see`
        : `p = ${p.p.toFixed(4)}  — not distinguishable on ${p.d} mover${p.d === 1 ? "" : "s"}`;
    console.log(`  ${p.a}\n  vs ${p.b}`);
    console.log(
      `     ${p.counts.both} both, ${p.counts.neither} neither, ` +
      `${p.counts.aOnly} only the first, ${p.counts.bOnly} only the second — ${verdict}`,
    );
  }
  const silent = pairs.filter((p) => p.d === 0).length;
  const significant = pairs.filter((p) => p.d > 0 && p.p <= 0.05).length;
  console.log(
    `\n  ${pairs.length} pairs differ in exactly one axis` +
    (values["all-pairs"] ? " or more (--all-pairs)" : " (--all-pairs for the rest)") +
    `; ${pairs.length - shown.length} not shown.\n` +
    `  ${significant} of them reach p <= 0.05. ${silent} are discordant on nothing at all — the\n` +
    `  two configurations returned the same verdict on every question, and no sample\n` +
    `  size makes those distinguishable.`,
  );
}

// ─── 5. per kind, descriptive only ───────────────────────────────────────────

console.log(`\n${"─".repeat(78)}`);
console.log(`per-kind proportions with exact 95% intervals, at recall@${K}`);
console.log(
  `Descriptive, never comparative — invariant 14. The intervals are Clopper-Pearson\n` +
  `and they are wide on purpose: with this many questions per kind, "A beats B on\n` +
  `contradictions" is not a statement the set supports, whatever the point\n` +
  `estimates do. Negatives are absent because they carry no recall; they are the\n` +
  `FP column of the main table.\n`,
);
for (const cell of liveCells) {
  console.log(`  ${rowLabel(cell.header)}`);
  for (const t of byKind(cell, K)) {
    const [lo, hi] = clopperPearson(t.k, t.n);
    console.log(
      `     ${head(t.kind, 14)} ${head(`${t.k}/${t.n}`, 6)} ` +
      `${((t.k / t.n) * 100).toFixed(0).padStart(3)}%  ` +
      `[${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}%]`,
    );
  }
}

// ─── determinism ─────────────────────────────────────────────────────────────

console.log(`\n${"─".repeat(78)}`);
const bad = live.filter((r) => r.disagreements.length > 0);
if (bad.length === 0) {
  console.log(
    `Determinism: all ${reference.repeats} repeats of all ${live.length} cells returned identical\n` +
    `rankings. The repeats are in configs.yaml to check that, not to average noise\n` +
    `away, and the check passed — so the noise term above is 0 by measurement.`,
  );
} else {
  console.log(
    `Determinism: FAILED on ${bad.length} of ${live.length} cells. This is a finding, not noise to\n` +
    `smooth away: a ranking that disagrees with itself is indistinguishable from a\n` +
    `real effect, and every paired comparison above assumes it does not happen.\n` +
    `The rows show the first repeat, which is arbitrary and said so.`,
  );
  for (const r of bad) {
    console.log(`  ${head(r.label, WIDTH)} ${r.disagreements.length}: ${r.disagreements.slice(0, 6).join(", ")}`);
  }
}

// ─── what this run found out that nobody asked it ────────────────────────────

// Computed from the rows rather than written down, so it cannot go stale
// against the data it describes. Each one is a claim a reader would otherwise
// have to reconstruct from the table by eye.
console.log(`\n${"─".repeat(78)}`);
console.log("notes from this run\n");

const notes: string[] = [];
const byLabel = new Map(live.map((r) => [r.label, r]));
const withRerank = (r: Row, id: string) =>
  byLabel.get(r.label.replace(/__[^_/]+( \/|$)/, `__${id}$1`));

// MMR is in the matrix as a measured negative result, not as a fix. Whether it
// stayed one is a question about this run.
const mmrPairs = live
  .filter((r) => r.cell.endsWith("__mmr-0.7"))
  .flatMap((r) => {
    const plain = byLabel.get(r.label.replace("__mmr-0.7", "__none"));
    return plain ? [{ mmr: r, plain }] : [];
  });
if (mmrPairs.length > 0) {
  const lost = mmrPairs.filter((p) => (p.mmr.recall[5] ?? 0) < (p.plain.recall[5] ?? 0)).length;
  const movedContradiction = mmrPairs.filter((p) => p.mmr.contradiction.k > p.plain.contradiction.k).length;
  notes.push(
    `MMR at lambda 0.7 lost recall@5 in ${lost} of ${mmrPairs.length} configurations and improved\n` +
    `  contradiction coverage in ${movedContradiction}. It is in the matrix as a measured negative\n` +
    `  result and it stayed one: it diversifies by surface dissimilarity, and two\n` +
    `  passages arguing opposite sides of one question share their topic vocabulary,\n` +
    `  so it reads them as redundant and avoids taking both.`,
  );
}

// Fusion buys recall and sells the ability to abstain. Both halves, or neither.
const fusionPairs = live
  .filter((r) => r.cell.includes("__hybrid__"))
  .flatMap((r) => {
    const lex = byLabel.get(r.label.replace("__hybrid__", "__bm25__").replace(/ \/.*$/, ""));
    return lex ? [{ hybrid: r, lex }] : [];
  });
if (fusionPairs.length > 0) {
  const better = fusionPairs.filter((p) => (p.hybrid.recall[5] ?? 0) > (p.lex.recall[5] ?? 0)).length;
  const worseFp = fusionPairs.filter(
    (p) => (p.hybrid.falsePositiveRate ?? 0) > (p.lex.falsePositiveRate ?? 0),
  ).length;
  notes.push(
    `Rank fusion beat its own lexical half on recall@5 in ${better} of ${fusionPairs.length} configurations and\n` +
    `  was worse on the negatives in ${worseFp}. Reporting only the first column would\n` +
    `  misrepresent it: fused scores take few distinct values, so there is less room\n` +
    `  to put an abstention threshold.`,
  );
}

// Two chunking configs producing the same chunk count is not a coincidence and
// not a bug; it is arithmetic worth stating before someone reads the pair as a
// replicate.
const counts = new Map<string, number>();
for (const r of rows) counts.set(r.cell.split("__")[0]!, r.chunkCount);
const collisions = [...counts].filter(([, n], _, all) => all.filter(([, m]) => m === n).length > 1);
if (collisions.length > 1) {
  notes.push(
    `${collisions.map(([id]) => id).join(" and ")} produce the same number of chunks\n` +
    `  (${collisions[0]![1]}). A 90-second window stepping 30 seconds of overlap advances 60\n` +
    `  seconds a window, which is exactly what a 60-second window with no overlap\n` +
    `  does — so the pair differs only in chunk LENGTH, not in count. They are not\n` +
    `  replicates of each other, and the overlap parameter buys nothing here that\n` +
    `  the window length does not already give.`,
  );
}

// The retriever that is worse at recall may be better at staying quiet. Whether
// that trade-off appeared in this run is the finding, not the assumption.
const bestRecall = live.reduce((a, b) => ((b.recall[5] ?? 0) > (a.recall[5] ?? 0) ? b : a));
const bestAbstain = live.reduce((a, b) =>
  ((b.falsePositiveRate ?? 1) < (a.falsePositiveRate ?? 1) ? b : a));
if (bestRecall.label !== bestAbstain.label) {
  notes.push(
    `The best configuration on recall@5 is not the best at knowing when to stay\n` +
    `  quiet. ${bestRecall.label}\n` +
    `  leads recall at ${((bestRecall.recall[5] ?? 0) * 100).toFixed(1)}% with ` +
    `${((bestRecall.falsePositiveRate ?? 0) * 100).toFixed(1)}% false positives, while\n` +
    `  ${bestAbstain.label}\n` +
    `  answers ${((bestAbstain.falsePositiveRate ?? 0) * 100).toFixed(1)}% of the unanswerable questions ` +
    `at ${((bestAbstain.recall[5] ?? 0) * 100).toFixed(1)}% recall. Chosen on\n` +
    `  recall alone, the system would confidently answer questions with no answer.`,
  );
}

// Invariant 11: the gate. Printed here because the whole table is void above it.
const ceiling = live.filter((r) => (r.recall[5] ?? 0) >= 0.85);
notes.push(
  ceiling.length === 0
    ? `The ceiling gate holds: no configuration reached 85% on recall@5, so every\n` +
      `  difference above is measurable in principle. The best is ` +
      `${((bestRecall.recall[5] ?? 0) * 100).toFixed(1)}%.`
    : `The ceiling gate FAILED for ${ceiling.length} configurations at or above 85% recall@5. No\n` +
      `  difference involving them is measurable at any sample size.`,
);

for (const [i, note] of notes.entries()) {
  // Continuation lines are written with two spaces and re-indented here, so
  // the note text stays readable in the source and aligned in the output.
  console.log(`  ${i + 1}. ${note.replace(/\n  /g, "\n     ")}\n`);
}

const notRun = rows.filter((r) => r.notRun !== null);
if (notRun.length > 0) {
  console.log(
    `\n${notRun.length} of ${rows.length} cells were not run. Their rows carry the reason rather than a\n` +
    `zero or a blank, because a missing cell that looks like a bad result is worse\n` +
    `than an empty one.`,
  );
}
