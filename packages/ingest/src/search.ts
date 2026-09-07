// Search the cached transcripts. This is the tool the golden set is written
// with: pick a tool × topic cell, read what was actually said about it, and
// write questions from that — the corpus-first order the spec asks for.
//
//   pnpm grep -- "context window"
//   pnpm grep -- "rate limit" --stance skeptical --limit 30
//   pnpm grep -- claude --tool claude-code --context 40
//   pnpm grep -- --cells                 # what material each tool has
//
// Prints a timestamped link per hit, so the span for a gold annotation is a
// click away rather than a hunt through the video.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { CACHE_DIR, loadCorpus, type Video } from "./corpus.ts";
import { normalizeTranscript, toolMentions, type Segment } from "./text.ts";

const HELP = `
usage: pnpm grep <query> [--stance S] [--channel NAME] [--context CHARS] [--limit N]
       pnpm grep --cells

  <query>            phrase to find, case-insensitive, matched across segment
                     boundaries so a phrase split over two captions still hits
  --stance S         hype | practical | skeptical
  --channel NAME     substring of the channel name
  --context CHARS    characters of surrounding text per hit (default 220)
  --limit N          hits to print (default 20)
  --cells            what material each tool has, by stance, before reading any

Each hit prints a timestamped link and a ready-to-paste gold span.
`.trimStart();

/** Bad input: say what was wrong, then how to call it. */
function usage(message: string): never {
  console.error(`${message}\n\n${HELP}`);
  process.exit(2);
}

const argv = process.argv.slice(2).filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));
const parsed = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        tool: { type: "string" },
        stance: { type: "string" },
        channel: { type: "string" },
        context: { type: "string", default: "220" },
        limit: { type: "string", default: "20" },
        cells: { type: "boolean", default: false },
      },
      allowPositionals: true,
    });
  } catch (e) {
    usage(String((e as Error).message));
  }
})();
const { values } = parsed;
const query = parsed.positionals.join(" ").trim();

const cfg = loadCorpus();
const stanceOf = new Map(cfg.channels.map((c) => [c.name, c.stance]));

type Loaded = { v: Video; segments: Segment[] };
const load = (v: Video): Loaded => ({
  v,
  segments: JSON.parse(readFileSync(join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`), "utf8")).segments,
});

let pool = cfg.videos;
if (values.stance) pool = pool.filter((v) => stanceOf.get(v.channel) === values.stance);
if (values.channel) pool = pool.filter((v) => v.channel.toLowerCase().includes(values.channel!.toLowerCase()));

// ─── --cells: what material exists, before spending time reading ─────────────

if (values.cells) {
  const loaded = pool.map(load);
  // Mention counts come from toolMentions, not a plain substring test. A bare
  // includes() matches the alias "cc" inside "soccer" and "accurate", which
  // reported claude-code in 69 of 78 videos.
  const counted = loaded.map((l) => ({
    ...l,
    mentions: toolMentions(normalizeTranscript(l.segments), cfg.tools),
  }));

  console.log("tool           videos>=1  >=3   hype  practical  skeptical  channels>=3");
  for (const t of cfg.tools) {
    const at = (k: number) => counted.filter((c) => (c.mentions.get(t.id) ?? 0) >= k);
    const solid = at(3);
    const by = (st: string) => solid.filter((h) => stanceOf.get(h.v.channel) === st).length;
    const chans = new Set(solid.map((h) => h.v.channel)).size;
    console.log(
      `${t.id.padEnd(14)} ${String(at(1).length).padStart(9)} ${String(solid.length).padStart(4)} ` +
      `${String(by("hype")).padStart(6)} ${String(by("practical")).padStart(10)} ` +
      `${String(by("skeptical")).padStart(10)} ${String(chans).padStart(12)}`,
    );
  }
  console.log("\nStance columns count videos at 3+ mentions — sustained discussion, not a passing name-drop.");
  console.log("A cell with no skeptical videos cannot carry a contradiction question.");
  process.exit(0);
}

// A bare `pnpm grep` is a request for help, not a mistake: print it and exit
// cleanly, so the shell does not decorate it with a failed-command trace.
if (!query) {
  console.log(HELP);
  process.exit(0);
}

// ─── search ──────────────────────────────────────────────────────────────────

const contextChars = Number(values.context);
const limit = Number(values.limit);
if (!Number.isInteger(contextChars) || contextChars < 20) usage("--context must be at least 20");
if (!Number.isInteger(limit) || limit < 1) usage("--limit must be a positive integer");

const needle = query.toLowerCase();

/**
 * Segments arrive in 2-8 second pieces, so a phrase routinely straddles two of
 * them and a per-segment search would miss it. The transcript is joined into
 * one string with an index back to the segment each character came from.
 */
function joined(segments: Segment[]) {
  let text = "";
  const owner: number[] = [];
  for (let i = 0; i < segments.length; i++) {
    const piece = (i > 0 ? " " : "") + segments[i]!.text;
    for (let c = 0; c < piece.length; c++) owner.push(i);
    text += piece;
  }
  return { text, lower: text.toLowerCase(), owner };
}

type Hit = { v: Video; at: number; excerpt: string; start_s: number; end_s: number };
const hits: Hit[] = [];

for (const v of pool) {
  if (values.tool && !(v.tools.includes(values.tool) || cfg.tools.some((t) => t.id === values.tool))) continue;
  const { segments } = load(v);
  const { text, lower, owner } = joined(segments);

  let from = 0;
  for (;;) {
    const at = lower.indexOf(needle, from);
    if (at === -1) break;
    const a = Math.max(0, at - Math.floor(contextChars / 2));
    const b = Math.min(text.length, at + needle.length + Math.floor(contextChars / 2));
    hits.push({
      v,
      at,
      excerpt: (a > 0 ? "…" : "") + text.slice(a, b).replace(/\s+/g, " ") + (b < text.length ? "…" : ""),
      start_s: segments[owner[a] ?? 0]!.start_s,
      end_s: segments[owner[Math.min(b, owner.length - 1)] ?? segments.length - 1]!.end_s,
    });
    from = at + needle.length;
  }
}

const shown = hits.slice(0, limit);
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

shown.forEach((h, i) => {
  const stance = stanceOf.get(h.v.channel) ?? "?";
  console.log(
    `\n[${i + 1}] ${h.v.channel} · ${stance} · ${h.v.published_at} · ${h.v.transcript_kind}`,
  );
  console.log(`    ${h.v.title.slice(0, 70)}`);
  console.log(`    https://youtu.be/${h.v.youtube_id}?t=${Math.floor(h.start_s)}   ${mmss(h.start_s)}–${mmss(h.end_s)}`);
  console.log(`    ${h.excerpt}`);
  console.log(`    gold: { video: ${h.v.youtube_id}, start_s: ${h.start_s}, end_s: ${h.end_s} }`);
});

console.log(
  `\n${hits.length} hits in ${new Set(hits.map((h) => h.v.youtube_id)).size} videos` +
  (hits.length > shown.length ? `, showing ${shown.length} — raise --limit for more` : ""),
);
const byStance = ["hype", "practical", "skeptical"]
  .map((s) => `${s} ${hits.filter((h) => stanceOf.get(h.v.channel) === s).length}`)
  .join(", ");
console.log(`by stance: ${byStance}`);
