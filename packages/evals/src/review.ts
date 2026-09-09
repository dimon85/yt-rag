// Which questions are worth a second look, and why.
//
//   pnpm review              # ranked, worst first
//   pnpm review --all        # every question carrying gold spans
//
// Reviewing 78 questions in order is a waste of the one thing that cannot be
// automated. This ranks them by how little they contribute to the measurement,
// so the reading starts where it pays.
//
// Nothing here edits or rejects anything. A question found by term matching is
// a perfectly ordinary question — it just cannot separate one configuration
// from another, so a set made mostly of those measures less than it appears to.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { fixedChunks } from "../../chunking/src/fixed.ts";
import { CACHE_DIR, loadCorpus } from "../../ingest/src/corpus.ts";
import { buildIndex, search } from "../../retrieve/src/bm25.ts";
import { lexicalOverlap, loadGolden, OVERLAP_FLOOR } from "./golden.ts";
import { lexicallyTrivial, recallAtK, type Retrieved } from "./metrics.ts";

const { values } = parseArgs({
  args: process.argv.slice(2).filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--"))),
  options: { all: { type: "boolean", default: false } },
  allowPositionals: false,
});

const cfg = loadCorpus();
const chunks: (Retrieved & { text: string })[] = [];
for (const v of cfg.videos) {
  const segs = JSON.parse(readFileSync(join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`), "utf8")).segments;
  for (const c of fixedChunks(segs, 512, 0)) {
    chunks.push({ video: v.youtube_id, start_s: c.start_s, end_s: c.end_s, text: c.text, score: 0 });
  }
}
const index = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
const run = (q: string, k: number) => search(index, q, k).map(({ id, score }) => ({ ...chunks[id]!, score }));

const spanOf = (g: { video: string; start_s: number; end_s: number }) =>
  (JSON.parse(readFileSync(join(CACHE_DIR, "transcripts", `${g.video}.json`), "utf8")).segments as any[])
    .filter((s) => s.end_s > g.start_s && s.start_s < g.end_s).map((s) => s.text).join(" ");

const set = loadGolden();
const rows = set.questions.filter((q) => q.gold.length > 0).map((q) => {
  const overlap = Math.max(...q.gold.map((g) => lexicalOverlap(q.text, spanOf(g))));
  const trivial = lexicallyTrivial(q.gold, run(q.text, 1)) ?? false;
  const words = q.text.split(/\s+/).filter(Boolean).length;

  // Reasons, in the order they matter. A trivial question contributes nothing
  // to a comparison; a very short one is probably fine and just scores oddly.
  const flags: string[] = [];
  if (trivial) flags.push("found by term matching at rank 1");
  if (overlap > 0.7) flags.push(`${Math.round(overlap * 100)}% of its words are in the answer`);
  if (words <= 8) flags.push(`only ${words} words`);
  if (overlap < OVERLAP_FLOOR) flags.push(`shares ${Math.round(overlap * 100)}% with its answer — check it is answerable`);
  if (q.source === "generated") flags.push("generated");
  if (!/\?$/.test(q.text.trim())) flags.push("not phrased as a question");

  // Weighted so the top of the list is where rewriting changes the measurement.
  const priority =
    (trivial ? 4 : 0) + (overlap > 0.7 ? 2 : 0) + (q.source === "generated" ? 1 : 0) +
    (!/\?$/.test(q.text.trim()) ? 2 : 0) + (overlap < OVERLAP_FLOOR ? 1 : 0);

  return { q, overlap, trivial, words, flags, priority, recall: recallAtK(q.gold, run(q.text, 5), 5)! };
});

const shown = values.all ? rows : rows.filter((r) => r.priority > 0);
shown.sort((a, b) => b.priority - a.priority || b.overlap - a.overlap);

console.log(
  `${rows.length} questions carry gold spans. ${rows.filter((r) => r.trivial).length} are ` +
  `answered by term matching alone.\n`,
);

for (const r of shown) {
  console.log(`${r.q.slug}  [${r.q.kind}, ${r.q.source}]`);
  console.log(`  ${r.q.text}`);
  console.log(`  ${r.flags.join(" · ")}`);
  console.log(`  span: youtu.be/${r.q.gold[0]!.video}?t=${Math.floor(r.q.gold[0]!.start_s)}\n`);
}

const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
console.log("─".repeat(64));
for (const s of ["hand", "generated"] as const) {
  const g = rows.filter((r) => r.q.source === s);
  if (g.length === 0) continue;
  console.log(
    `${s.padEnd(10)} n=${String(g.length).padStart(2)}  recall@5 ${(mean(g.map((r) => r.recall)) * 100).toFixed(1)}%  ` +
    `term-matchable ${g.filter((r) => r.trivial).length}/${g.length}  ` +
    `overlap ${(mean(g.map((r) => r.overlap)) * 100).toFixed(0)}%  words ${mean(g.map((r) => r.words)).toFixed(1)}`,
  );
}
console.log(
  "\nRewriting a term-matchable question so it is not one raises what the set\n" +
  "can measure. Rewriting the others mostly does not.",
);
