// Validates golden/questions.yaml against the corpus and reports where it
// stands against the design.
//
//   pnpm golden
//
// Exits 1 on any error, so it can gate a run rather than merely inform one.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CACHE_DIR, loadCorpus } from "../../ingest/src/corpus.ts";
import type { Segment } from "../../ingest/src/text.ts";
import {
  countByKind, goldenSha, lexicalOverlap, loadGolden, OVERLAP_FLOOR, OVERLAP_WARN,
  RECALL_POOL_TARGET, TARGET, validateStructure, type Issue, type Kind,
} from "./golden.ts";

const corpus = loadCorpus();
const durations = new Map(corpus.videos.map((v) => [v.youtube_id, v.duration_s]));

let set;
try {
  set = loadGolden();
} catch (e) {
  console.error(`golden/questions.yaml did not parse:\n${(e as Error).message}`);
  process.exit(1);
}

const issues: Issue[] = validateStructure(set, { durations });

// Overlap needs the transcript, so it lives here rather than in the pure part.
const spanText = (video: string, start_s: number, end_s: number): string => {
  const path = join(CACHE_DIR, "transcripts", `${video}.json`);
  const cached = JSON.parse(readFileSync(path, "utf8")) as { kind: string; segments?: Segment[] };
  if (cached.kind !== "ok" || !cached.segments) return "";
  return cached.segments
    .filter((s) => s.end_s > start_s && s.start_s < end_s)
    .map((s) => s.text)
    .join(" ");
};

for (const q of set.questions) {
  // The floor applies to the question's best span: a comparative question is
  // fine if it connects to one of its sources, and demanding overlap with
  // every one of them would push it back towards restating the passage.
  const overlaps = q.gold
    .filter((g) => durations.has(g.video))
    .map((g) => lexicalOverlap(q.text, spanText(g.video, g.start_s, g.end_s)));
  if (overlaps.length > 0 && Math.max(...overlaps) < OVERLAP_FLOOR) {
    issues.push({
      level: "warning",
      slug: q.slug,
      message:
        `only ${Math.round(Math.max(...overlaps) * 100)}% of the question's words appear in ` +
        `any span it points at. Check the answer really is in the span. If it is, ` +
        `keep the question — a pure paraphrase is the only kind that separates lexical ` +
        `from semantic retrieval, and raising its overlap deletes that signal`,
    });
  }

  for (const g of q.gold) {
    if (!durations.has(g.video)) continue;
    const overlap = lexicalOverlap(q.text, spanText(g.video, g.start_s, g.end_s));
    if (overlap > OVERLAP_WARN) {
      issues.push({
        level: "warning",
        slug: q.slug,
        message:
          `${Math.round(overlap * 100)}% of the question's words appear in the span it points at ` +
          `— term matching alone will find it, so it cannot separate one retriever from another`,
      });
    }
  }
}

// ─── report ──────────────────────────────────────────────────────────────────

const errors = issues.filter((i) => i.level === "error");
const warnings = issues.filter((i) => i.level === "warning");
const counts = countByKind(set);
const pool = counts.factual + counts.comparative + counts.contradiction;

console.log(`golden/questions.yaml — ${set.questions.length} questions, sha ${goldenSha(set)}\n`);
console.log("kind            have  target");
for (const k of Object.keys(TARGET) as Kind[]) {
  const short = TARGET[k] - counts[k];
  console.log(
    `${k.padEnd(14)} ${String(counts[k]).padStart(5)} ${String(TARGET[k]).padStart(7)}` +
    (short > 0 ? `   ${short} to go` : short < 0 ? `   ${-short} over` : "   done"),
  );
}
console.log(
  `\nrecall pool: ${pool} of ${RECALL_POOL_TARGET} — negatives carry no gold spans, so they do not ` +
  `contribute to recall@k or to what the set can detect`,
);

for (const i of errors) console.error(`ERROR   ${i.slug}: ${i.message}`);
for (const i of warnings) console.warn(`WARN    ${i.slug}: ${i.message}`);

if (errors.length > 0) {
  console.error(`\n${errors.length} error${errors.length === 1 ? "" : "s"}. Not usable for a run.`);
  process.exit(1);
}
console.log(`\nno errors${warnings.length > 0 ? `, ${warnings.length} warning(s) worth a reread` : ""}.`);
