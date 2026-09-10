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

  // Where a lexical retriever puts the gold span, within a deep-ish window.
  // Rank 1 means the question is handed to every configuration for free; not
  // found at all means it may be handed to none of them, which is the same
  // amount of information. What a comparison needs is the middle.
  const deep = run(q.text, 20);
  const rank = 1 + deep.findIndex((r) =>
    q.gold.some((g) => g.video === r.video && r.end_s > g.start_s && r.start_s < g.end_s));
  const unreachable = rank === 0;

  // Reasons, in the order they matter. A trivial question contributes nothing
  // to a comparison; a very short one is probably fine and just scores oddly.
  const flags: string[] = [];
  if (trivial) flags.push("term matching puts it first — which does not mean every retriever finds it");
  if (unreachable) flags.push("not in the top 20 for term matching — check any retriever can reach it");
  if (!trivial && !unreachable) flags.push(`term matching puts it at rank ${rank}`);
  if (overlap > 0.7) flags.push(`${Math.round(overlap * 100)}% of its words are in the answer`);
  if (words <= 8) flags.push(`only ${words} words`);
  if (overlap < OVERLAP_FLOOR) flags.push(`shares ${Math.round(overlap * 100)}% with its answer`);
  if (q.source === "generated") flags.push("generated");
  if (!/\?$/.test(q.text.trim())) flags.push("not phrased as a question");

  // Weighted so the top of the list is where rewriting changes the measurement.
  //
  // Term-matchability scores NOTHING, which is a reversal of what this used to
  // do. It carried the largest weight on the reasoning that every
  // configuration answers such a question, so it takes no part in the
  // comparison. Measured, that is backwards: on 15 term-matchable questions
  // recall@5 was 91.3% for BM25, 74.7% for hosted embeddings and 36.0% for the
  // local model, against a 17 pp spread on the other 34. They are the most
  // discriminating group in the set, and this list was sending them to the top
  // of the pile to be rewritten. Two separate measurements now say to leave
  // them alone: this one, and the eight rewrites of which seven became
  // unreachable.
  //
  // `source` scores nothing: it correlates strongly with the problem — 16 of 30
  // generated questions are term-matchable against 6 of 16 hand-written — but
  // being generated is not itself a defect, and scoring it counts the same
  // evidence twice.
  //
  // High overlap scores nothing either, for a reason that took a measurement to
  // see. "Why is the default plan so slow?" shares 100% of its content words
  // with its answer and still comes back at rank 2, because those words appear
  // in other passages too. Overlap says how much wording a question shares;
  // triviality says whether that wording is enough to identify one passage.
  // They are different questions, and only the second one matters here.
  // Unreachable weighs as much as trivial. Measured the hard way: rewriting
  // eight term-matchable questions to remove the shared wording moved seven of
  // them out of reach of BM25 *and* of local embeddings entirely. A question no
  // retriever finds tells a comparison exactly as little as one they all find.
  const priority =
    (unreachable ? 4 : 0) + (!/\?$/.test(q.text.trim()) ? 2 : 0);

  return { q, overlap, trivial, unreachable, words, flags, priority, recall: recallAtK(q.gold, run(q.text, 5), 5)! };
});

const shown = values.all ? rows : rows.filter((r) => r.priority > 0);
shown.sort((a, b) => b.priority - a.priority || b.overlap - a.overlap);

console.log(
  `${rows.length} questions carry gold spans. ${rows.filter((r) => r.trivial).length} are put ` +
  `first by term matching, which is not the\ndefect it looks like — those separate ` +
  `configurations best of any group in the set.\n${rows.filter((r) => r.unreachable).length} are ` +
  `out of reach of term matching entirely, and those are worth a look.\n`,
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
  "\nRemoving the wording a question shares with its answer was measured twice and\n" +
  "it overshoots both ways. Seven of eight questions rewritten that way became\n" +
  "unreachable for BM25 and local embeddings alike. And the questions term\n" +
  "matching finds first are the ones that separate configurations best — 55 pp\n" +
  "of spread against 17 pp for the rest. Rewrite for phrasing, not for overlap.",
);
