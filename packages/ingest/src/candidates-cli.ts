// Ranked passages for annotation.
//
//   pnpm candidates --tool claude-code            # best passages for a tool
//   pnpm candidates --tool cursor --topic cost    # one cell of the matrix
//   pnpm candidates --pairs --tool cursor         # pro/contra candidates side by side
//   pnpm candidates --cells                       # how much material each cell has
//
// Ranking decides reading order and nothing else. It does not write questions:
// a question phrased from the passage is found by term matching alone and stops
// separating one retriever from another.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { dedupeOverlapping, score, windows, type Scored } from "./candidates.ts";
import { CACHE_DIR, loadCorpus, type Video } from "./corpus.ts";
import type { Segment } from "./text.ts";

const HELP = `
usage: pnpm candidates [--tool ID] [--topic ID] [--stance S] [--limit N] [--width S]
       pnpm candidates --pairs --tool ID
       pnpm candidates --cells

  --tool ID     restrict to passages mentioning this tool
  --topic ID    restrict to passages matching the topic's spoken keywords
  --stance S    hype | practical | skeptical
  --pairs       group skeptical passages with hype ones on the same tool
  --cells       material per tool x topic, before reading anything
  --limit N     passages to print (default 15)
  --width S     window width in seconds (default 30)
`.trimStart();

function usage(message: string): never {
  console.error(`${message}\n\n${HELP}`);
  process.exit(2);
}

const argv = process.argv.slice(2).filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));
const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        tool: { type: "string" },
        topic: { type: "string" },
        stance: { type: "string" },
        pairs: { type: "boolean", default: false },
        cells: { type: "boolean", default: false },
        limit: { type: "string", default: "15" },
        width: { type: "string", default: "30" },
      },
      allowPositionals: false,
    });
  } catch (e) {
    usage(String((e as Error).message));
  }
})();

const cfg = loadCorpus();
const stanceOf = new Map(cfg.channels.map((c) => [c.name, c.stance]));
const topics = cfg.topics.map((t) => ({ id: t.id, keywords: t.keywords }));
const limit = Number(values.limit);
const width = Number(values.width);
if (!Number.isInteger(limit) || limit < 1) usage("--limit must be a positive integer");
if (!Number.isFinite(width) || width < 5) usage("--width must be at least 5");

if (values.tool && !cfg.tools.some((t) => t.id === values.tool)) {
  usage(`unknown tool ${values.tool} — have: ${cfg.tools.map((t) => t.id).join(", ")}`);
}
if (values.topic && !topics.some((t) => t.id === values.topic)) {
  usage(`unknown topic ${values.topic} — have: ${topics.map((t) => t.id).join(", ")}`);
}

const segmentsOf = (v: Video): Segment[] =>
  JSON.parse(readFileSync(join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`), "utf8")).segments;

type Row = Scored & { v: Video };

const all: Row[] = [];
for (const v of cfg.videos) {
  const stance = stanceOf.get(v.channel);
  if (values.stance && stance !== values.stance) continue;
  for (const w of windows(segmentsOf(v), width)) {
    const s = score(w, cfg.tools, topics, { scarceStance: stance === "skeptical" });
    if (s.total === 0) continue;
    if (values.tool && !s.tools.includes(values.tool)) continue;
    if (values.topic && !s.topics.includes(values.topic)) continue;
    all.push({ ...{ window: w, score: s }, v });
  }
}

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function print(rows: Row[], n: number) {
  rows.slice(0, n).forEach((r, i) => {
    const stance = stanceOf.get(r.v.channel) ?? "?";
    console.log(`\n[${i + 1}] score ${r.score.total}  ${r.score.reasons.join(", ")}`);
    console.log(`    ${r.v.channel} · ${stance} · ${r.v.published_at}`);
    console.log(
      `    https://youtu.be/${r.v.youtube_id}?t=${Math.floor(r.window.start_s)}` +
      `   ${mmss(r.window.start_s)}–${mmss(r.window.end_s)}`,
    );
    console.log(`    ${r.window.text.replace(/\s+/g, " ").slice(0, 400)}`);
    console.log(
      `    gold: { video: ${r.v.youtube_id}, start_s: ${r.window.start_s}, end_s: ${r.window.end_s} }`,
    );
  });
}

// ─── --cells ─────────────────────────────────────────────────────────────────

if (values.cells) {
  console.log("Scoring passages per cell. Skeptical count is the ceiling on");
  console.log("contradiction questions: no contra side, no contradiction.\n");
  console.log("tool           topic            passages  skeptical");
  for (const t of cfg.tools) {
    for (const tp of topics) {
      const rows = all.filter((r) => r.score.tools.includes(t.id) && r.score.topics.includes(tp.id));
      if (rows.length === 0) continue;
      const sk = rows.filter((r) => stanceOf.get(r.v.channel) === "skeptical").length;
      console.log(
        `${t.id.padEnd(14)} ${tp.id.padEnd(16)} ${String(rows.length).padStart(8)} ${String(sk).padStart(10)}`,
      );
    }
  }
  process.exit(0);
}

// ─── --pairs ─────────────────────────────────────────────────────────────────

if (values.pairs) {
  if (!values.tool) usage("--pairs needs --tool");
  const sk = dedupeOverlapping(all.filter((r) => stanceOf.get(r.v.channel) === "skeptical")) as Row[];
  const rest = dedupeOverlapping(all.filter((r) => stanceOf.get(r.v.channel) !== "skeptical")) as Row[];

  console.log(`Contra candidates for ${values.tool} — the scarce side, ${sk.length} found.`);
  print(sk, Math.min(limit, sk.length));
  console.log(`\n${"─".repeat(64)}`);
  console.log(`Pro candidates, ${rest.length} found. Pair one against a contra above.`);
  print(rest, Math.min(limit, rest.length));
  console.log(
    `\nA pair is only a contradiction if both talk about the same claim. ` +
    `Same tool is not enough — read both before writing the question.`,
  );
  process.exit(0);
}

// ─── ranked list ─────────────────────────────────────────────────────────────

const ranked = dedupeOverlapping(all) as Row[];
print(ranked, limit);
const byStance = ["hype", "practical", "skeptical"]
  .map((s) => `${s} ${ranked.filter((r) => stanceOf.get(r.v.channel) === s).length}`)
  .join(", ");
console.log(`\n${ranked.length} passages after merging overlaps, showing ${Math.min(limit, ranked.length)}`);
console.log(`by stance: ${byStance}`);
