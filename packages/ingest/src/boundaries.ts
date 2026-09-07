// Where fixed-token chunk boundaries fall in a transcript.
//
//   pnpm boundaries Q7n0PGbMW_U
//   pnpm boundaries Q7n0PGbMW_U --tokens 1024
//
// Exists so that a gold span annotated as "straddles a chunk boundary" is a
// checkable claim rather than a note someone wrote once. Boundaries are counted
// the way fixed-512-overlap-0 will: merge the segments in order, count tokens
// with gpt-tokenizer, split every N.
//
// One tokenizer for every configuration is invariant 2 — a boundary computed
// from text.length / 4 would land somewhere else entirely.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { encode } from "gpt-tokenizer";
import { CACHE_DIR, loadCorpus } from "./corpus.ts";
import type { Segment } from "./text.ts";

const HELP = `
usage: pnpm boundaries <youtube_id> [--tokens N]

  --tokens N   chunk size in tokens (default 512)

Prints the timestamp of every chunk boundary and what is being said across it.
A gold span containing one of these times is split by chunking.
`.trimStart();

const argv = process.argv.slice(2)
  .filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));

const parsed = (() => {
  try {
    return parseArgs({
      args: argv,
      options: { tokens: { type: "string", default: "512" } },
      allowPositionals: true,
    });
  } catch (e) {
    console.error(`${(e as Error).message}\n\n${HELP}`);
    process.exit(2);
  }
})();

const id = parsed.positionals[0];
if (!id) {
  console.log(HELP);
  process.exit(0);
}

const size = Number(parsed.values.tokens);
if (!Number.isInteger(size) || size < 1) {
  console.error(`--tokens must be a positive integer\n\n${HELP}`);
  process.exit(2);
}

const cfg = loadCorpus();
const video = cfg.videos.find((v) => v.youtube_id === id);
if (!video) {
  console.error(`${id} is not in corpus.yaml`);
  process.exit(2);
}

const segments: Segment[] = JSON.parse(
  readFileSync(join(CACHE_DIR, "transcripts", `${id}.json`), "utf8"),
).segments;

let cumulative = 0;
const crossings: { token: number; at: number; index: number }[] = [];
for (const [i, s] of segments.entries()) {
  const before = cumulative;
  // The leading space matters: a tokenizer treats " word" and "word" as
  // different tokens, and the merged transcript joins segments with a space.
  cumulative += encode(`${i > 0 ? " " : ""}${s.text}`).length;
  for (let b = Math.floor(before / size) + 1; b <= Math.floor(cumulative / size); b++) {
    crossings.push({ token: b * size, at: s.start_s, index: i });
  }
}

const mmss = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

console.log(
  `${id} · ${video.channel} · ${cumulative} tokens · ` +
  `${Math.ceil(cumulative / size)} chunks of ${size}\n`,
);
for (const c of crossings) {
  const around = segments.slice(Math.max(0, c.index - 2), c.index + 3).map((s) => s.text).join(" ");
  console.log(`token ${String(c.token).padStart(6)}  at ${mmss(c.at).padStart(6)}  (${c.at.toFixed(2)}s)`);
  console.log(`   …${around.replace(/\s+/g, " ").slice(0, 120)}…`);
}
