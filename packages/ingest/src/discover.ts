// Fills `videos:` in corpus.yaml.
//
//   pnpm discover                  # every channel
//   pnpm discover -- --channel @t3dotgg
//   pnpm discover -- --scan 60     # how deep into each channel to look
//   pnpm discover -- --sleep 400   # ms between requests that miss the cache
//   pnpm discover -- --cookies chrome   # authenticate when YouTube starts
//                                       # asking us to confirm we're not a bot
//   pnpm discover -- --dry-run     # report only, corpus.yaml untouched
//
// Metadata is cached per video under cache/meta/. Re-runs read the cache and
// never re-request: transcripts and metadata are fetched once (invariant 5).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { CACHE_DIR, loadCorpus, writeVideos, type Video } from "./corpus.ts";
import { select, spreadMonths, spreadPick, toolsInTitle, type Candidate } from "./select.ts";
import { fetchMeta, listChannel, type VideoMeta } from "./ytdlp.ts";

const META_DIR = join(CACHE_DIR, "meta");

function usage(message: string): never {
  console.error(`${message}\n\nusage: pnpm discover [--channel @handle] [--scan N] [--sleep MS] [--cookies BROWSER] [--dry-run]`);
  process.exit(2);
}

// `pnpm discover -- --channel X` hands the script a literal "--" as argv[2],
// which parseArgs reads as the end-of-options marker and turns everything after
// it into positionals. Dropping leading separators makes both spellings work.
const argv = process.argv.slice(2).filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));

const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        channel: { type: "string" },
        scan: { type: "string", default: "60" },
        sleep: { type: "string", default: "400" },
        cookies: { type: "string" },
        "dry-run": { type: "boolean", default: false },
      },
      allowPositionals: false,
    });
  } catch (e) {
    usage(String((e as Error).message));
  }
})();

const scan = Number(values.scan);
if (!Number.isInteger(scan) || scan < 1) usage(`--scan must be a positive integer, got ${values.scan}`);

const sleepMs = Number(values.sleep);
if (!Number.isInteger(sleepMs) || sleepMs < 0) usage(`--sleep must be a non-negative integer, got ${values.sleep}`);

const cfg = loadCorpus();
const channels = values.channel
  ? cfg.channels.filter((c) => c.handle === values.channel)
  : cfg.channels;
if (channels.length === 0) usage(`no channel with handle ${values.channel}`);

const [minDurationS, maxDurationS] = cfg.targets.duration_s;
const window = { minDurationS, maxDurationS, months: cfg.targets.window_months, now: new Date() };

mkdirSync(META_DIR, { recursive: true });

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Cached = { kind: "ok"; meta: VideoMeta } | { kind: "gone"; reason: string };

/**
 * One request per video, then never again.
 *
 * Only settled outcomes are cached. A throttled request is not an answer about
 * the video, so writing it to the cache would turn a temporary block into a
 * permanent verdict — which is exactly what happened on the first full run.
 *
 * The pause applies only to requests that actually go out: a cache hit costs
 * nothing, or a re-run over 600 cached videos would idle for minutes doing no
 * work.
 */
async function meta(id: string): Promise<Cached | null> {
  const path = join(META_DIR, `${id}.json`);
  if (existsSync(path)) {
    const raw = JSON.parse(readFileSync(path, "utf8"));
    if (raw && typeof raw === "object" && "kind" in raw) return raw as Cached;
    // Pre-fix cache entry. `null` there meant "unavailable", which was often a
    // throttle. Ignore it and re-fetch rather than trusting it.
    if (raw !== null) return { kind: "ok", meta: raw as VideoMeta };
  }

  for (let attempt = 0; attempt < 5; attempt++) {
    const r = await fetchMeta(id, values.cookies);
    if (r.kind === "throttled") {
      const backoff = Math.min(60_000, 2_000 * 2 ** attempt);
      process.stdout.write(`    throttled, waiting ${backoff / 1000}s — ${r.reason.slice(0, 70)}\n`);
      await wait(backoff);
      continue;
    }
    const settled: Cached = r.kind === "ok"
      ? { kind: "ok", meta: r.meta }
      : { kind: "gone", reason: r.reason };
    writeFileSync(path, JSON.stringify(settled, null, 2));
    if (sleepMs > 0) await wait(sleepMs);
    return settled;
  }
  return null;   // still throttled after five attempts
}

const kept: Video[] = [];
const skipped: Record<string, number> = {};
const perChannel: { name: string; scanned: number; kept: number }[] = [];

for (const ch of channels) {
  process.stdout.write(`\n${ch.name} (${ch.handle}) — listing ${scan}\n`);

  const flat = await listChannel(ch.channel_id, scan);
  // Cheap filter first: duration is already known, so videos outside the length
  // bounds never cost a per-video request.
  const worthFetching = flat.filter(
    (e) => e.duration_s != null && e.duration_s >= minDurationS && e.duration_s <= maxDurationS,
  );
  process.stdout.write(`  ${flat.length} listed, ${worthFetching.length} within length bounds\n`);

  const candidates: Candidate[] = [];
  for (const e of worthFetching) {
    const c = await meta(e.youtube_id);
    if (c === null) {
      // Five backoffs and still blocked. Stopping beats finishing with a
      // corpus that is quietly missing whole channels.
      console.error(
        `\nstill throttled after 5 attempts on ${e.youtube_id}.\n` +
        `Cached results are kept, so a later run resumes where this one stopped.\n` +
        `Try a longer --sleep, or --cookies chrome to authenticate.`,
      );
      process.exit(3);
    }
    if (c.kind === "gone") {
      skipped["gone"] = (skipped["gone"] ?? 0) + 1;
      continue;
    }
    const m = c.meta;
    candidates.push({
      youtube_id: m.youtube_id,
      channel: ch.name,
      title: m.title || e.title,
      duration_s: m.duration_s ?? e.duration_s,
      published_at: m.published_at,
      view_count: m.view_count ?? e.view_count,
      has_manual_en: m.has_manual_en,
      has_auto_en: m.has_auto_en,
    });
  }

  const report = select(candidates, window);
  for (const r of report.rejected) skipped[r.reason] = (skipped[r.reason] ?? 0) + 1;

  // Not the newest N: yt-dlp returns newest-first, and slicing the top produced
  // a set spanning 0.3 months on the first real run. Spread across the window.
  const take = spreadPick(report.selected, ch.target_videos, (s) => s.candidate.published_at!);
  for (const s of take) {
    kept.push({
      youtube_id: s.candidate.youtube_id,
      channel: ch.name,
      title: s.candidate.title,
      published_at: s.candidate.published_at!,
      duration_s: s.candidate.duration_s!,
      view_count: s.candidate.view_count ?? undefined,
      transcript_kind: s.transcript_kind,
      tools: toolsInTitle(s.candidate.title, cfg.tools),
      topics: [],
    });
  }
  perChannel.push({ name: ch.name, scanned: flat.length, kept: take.length });
  process.stdout.write(`  kept ${take.length} of ${ch.target_videos} target\n`);
}

// ─── report ───────────────────────────────────────────────────────────────────

console.log("\n" + "─".repeat(64));
for (const c of perChannel) {
  const target = cfg.channels.find((x) => x.name === c.name)!.target_videos;
  const short = c.kept < target ? `  SHORT by ${target - c.kept}` : "";
  console.log(`${c.name.padEnd(22)} scanned ${String(c.scanned).padStart(3)}  kept ${String(c.kept).padStart(3)}${short}`);
}

console.log("\nskipped:");
for (const [reason, n] of Object.entries(skipped).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${reason.padEnd(18)} ${n}`);
}

const spread = spreadMonths(kept.map((v) => v.published_at));
console.log(`\ntotal kept:      ${kept.length}  (target ${cfg.targets.total_videos}, floor ${cfg.targets.min_videos})`);
console.log(`date spread:     ${spread.toFixed(1)} months` + (spread < 12 ? "  BELOW the 12-month rule" : ""));

const byKind = kept.reduce<Record<string, number>>((a, v) => ((a[v.transcript_kind] = (a[v.transcript_kind] ?? 0) + 1), a), {});
console.log(`transcript kind: ${JSON.stringify(byKind)}`);

const untagged = kept.filter((v) => v.tools.length === 0).length;
console.log(`no tool in title: ${untagged}  (tools are matched on title only — annotate the rest by hand)`);

// Rule: at least 3 channels per tool, one of them not hype.
const stanceOf = new Map(cfg.channels.map((c) => [c.name, c.stance]));
console.log("\nchannels per tool:");
for (const t of cfg.tools) {
  const chans = new Set(kept.filter((v) => v.tools.includes(t.id)).map((v) => v.channel));
  const nonHype = [...chans].filter((c) => stanceOf.get(c) !== "hype").length;
  const bad = chans.size < 3 || nonHype === 0 ? "  RULE UNMET" : "";
  console.log(`  ${t.id.padEnd(14)} ${String(chans.size).padStart(2)} channels, ${nonHype} non-hype${bad}`);
}

if (values["dry-run"]) {
  console.log("\n--dry-run: corpus.yaml not written");
} else {
  writeVideos(kept);
  console.log(`\nwrote ${kept.length} videos to corpus.yaml`);
}
