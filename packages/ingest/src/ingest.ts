// Fetches a transcript for every video in corpus.yaml and caches it on disk.
//
//   pnpm ingest                 # everything still missing from the cache
//   pnpm ingest -- --sleep 800  # ms between requests that miss the cache
//   pnpm ingest -- --limit 10   # stop after N fetches, for a first look
//
// This is the last step that touches the network. Everything downstream —
// chunking, embedding, retrieval, metrics — reads cache/transcripts and never
// makes a request, which is what keeps two runs comparable.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { CACHE_DIR, loadCorpus, type Video } from "./corpus.ts";
import { coverageSeconds, fingerprint, normalizeTranscript, toolsInText, type Segment } from "./text.ts";
import { fetchTranscript } from "./transcripts.ts";

const DIR = join(CACHE_DIR, "transcripts");

function usage(message: string): never {
  console.error(`${message}\n\nusage: pnpm ingest [--sleep MS] [--limit N]`);
  process.exit(2);
}

const argv = process.argv.slice(2).filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));
const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: { sleep: { type: "string", default: "800" }, limit: { type: "string" } },
      allowPositionals: false,
    });
  } catch (e) {
    usage(String((e as Error).message));
  }
})();

const sleepMs = Number(values.sleep);
if (!Number.isInteger(sleepMs) || sleepMs < 0) usage(`--sleep must be a non-negative integer`);
const limit = values.limit === undefined ? Infinity : Number(values.limit);
if (!(limit > 0)) usage(`--limit must be a positive integer`);

const cfg = loadCorpus();
if (cfg.videos.length === 0) {
  console.error("corpus.yaml has no videos yet — run `pnpm discover` first.");
  process.exit(2);
}

mkdirSync(DIR, { recursive: true });
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Cached = { kind: "ok"; segments: Segment[] } | { kind: "gone"; reason: string };

/** Fetched once, cached forever. Throttling is never written to the cache. */
async function transcript(v: Video): Promise<Cached | null> {
  const path = join(DIR, `${v.youtube_id}.json`);
  if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8")) as Cached;

  for (let attempt = 0; attempt < 5; attempt++) {
    const r = await fetchTranscript(v.youtube_id);
    if (r.kind === "throttled") {
      const backoff = Math.min(60_000, 2_000 * 2 ** attempt);
      process.stdout.write(`    throttled, waiting ${backoff / 1000}s — ${r.reason}\n`);
      await wait(backoff);
      continue;
    }
    const settled: Cached = r.kind === "ok"
      ? { kind: "ok", segments: r.segments }
      : { kind: "gone", reason: r.reason };
    writeFileSync(path, JSON.stringify(settled));
    if (sleepMs > 0) await wait(sleepMs);
    return settled;
  }
  return null;
}

const seen = new Map<string, string>();          // fingerprint → youtube_id
const gone: { id: string; reason: string }[] = [];
const dupes: { id: string; of: string }[] = [];
const ok: { v: Video; segments: Segment[]; tools: string[] }[] = [];
let fetched = 0;

for (const v of cfg.videos) {
  if (fetched >= limit) break;
  const before = existsSync(join(DIR, `${v.youtube_id}.json`));
  const c = await transcript(v);
  if (!before) fetched++;

  if (c === null) {
    console.error(
      `\nstill throttled after 5 attempts on ${v.youtube_id}.\n` +
      `Cached transcripts are kept, so a later run resumes from here.`,
    );
    process.exit(3);
  }
  if (c.kind === "gone") {
    gone.push({ id: v.youtube_id, reason: c.reason });
    continue;
  }

  // The authoritative duplicate check. select.ts compares titles, which a
  // reupload under a new name slips past; this compares what was actually said.
  const normalized = normalizeTranscript(c.segments);
  const fp = fingerprint(normalized);
  const prior = seen.get(fp);
  if (prior) {
    dupes.push({ id: v.youtube_id, of: prior });
    continue;
  }
  seen.set(fp, v.youtube_id);

  ok.push({ v, segments: c.segments, tools: toolsInText(normalized, cfg.tools) });
}

// ─── report ───────────────────────────────────────────────────────────────────

console.log("\n" + "─".repeat(64));
console.log(`videos in corpus:  ${cfg.videos.length}`);
console.log(`transcripts ok:    ${ok.length}   (${fetched} fetched this run, rest from cache)`);
console.log(`no transcript:     ${gone.length}`);
console.log(`duplicate text:    ${dupes.length}`);

for (const d of dupes) console.log(`  ${d.id} duplicates ${d.of}`);

// A transcript covering a fraction of the video usually means captions stop
// partway, and a chunk set built on it would be quietly missing the ending.
const thin = ok.filter(({ v, segments }) => coverageSeconds(segments) < v.duration_s * 0.6);
console.log(`thin coverage:     ${thin.length}  (transcript spans under 60% of the video)`);
for (const t of thin.slice(0, 5)) {
  const pct = Math.round((coverageSeconds(t.segments) / t.v.duration_s) * 100);
  console.log(`  ${t.v.youtube_id}  ${pct}%  ${t.v.title.slice(0, 48)}`);
}

const byKind = ok.reduce<Record<string, number>>(
  (a, { v }) => ((a[v.transcript_kind] = (a[v.transcript_kind] ?? 0) + 1), a), {});
console.log(`transcript kind:   ${JSON.stringify(byKind)}`);

// The per-tool rule can finally be checked: aliases now run against what was
// said, not against titles, where four of six tools scored zero channels.
const stanceOf = new Map(cfg.channels.map((c) => [c.name, c.stance]));
console.log("\nchannels per tool (from transcript text):");
for (const t of cfg.tools) {
  const chans = new Set(ok.filter((o) => o.tools.includes(t.id)).map((o) => o.v.channel));
  const nonHype = [...chans].filter((c) => stanceOf.get(c) !== "hype").length;
  const unmet = chans.size < 3 || nonHype === 0 ? "  RULE UNMET" : "";
  console.log(`  ${t.id.padEnd(14)} ${String(chans.size).padStart(2)} channels, ${nonHype} non-hype${unmet}`);
}

const untagged = ok.filter((o) => o.tools.length === 0).length;
console.log(`\nno tool mentioned: ${untagged} of ${ok.length}`);
console.log(`\ntranscripts cached in ${DIR}`);
