// Selection rules from corpus.yaml as pure functions: candidates in, decisions out.
// No network, no yt-dlp, no filesystem — so every rule is testable without a
// single request, and a rejected video always carries the reason it was rejected.

export type Candidate = {
  youtube_id: string;
  channel: string;
  title: string;
  duration_s: number | null;
  published_at: string | null;   // ISO date
  view_count?: number | null;
  has_manual_en?: boolean;
  has_auto_en?: boolean;
};

export type Rejection =
  | "no-duration" | "too-short" | "too-long"
  | "no-date" | "outside-window"
  | "no-transcript" | "duplicate-title";

export type Decision =
  | { keep: true; transcript_kind: "manual" | "generated" }
  | { keep: false; reason: Rejection };

export type Window = { minDurationS: number; maxDurationS: number; months: number; now: Date };

/**
 * Title normalization for deduplication. The same video gets reuploaded across
 * channels, so identity cannot be youtube_id.
 *
 * This is the cheap first pass, on titles. The authoritative check is on
 * normalized transcript text and happens after fetching — a reupload often
 * carries a different title, which this pass cannot see.
 */
export function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/\p{Extended_Pictographic}/gu, " ")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

export function decide(c: Candidate, w: Window): Decision {
  if (c.duration_s == null) return { keep: false, reason: "no-duration" };
  if (c.duration_s < w.minDurationS) return { keep: false, reason: "too-short" };
  if (c.duration_s > w.maxDurationS) return { keep: false, reason: "too-long" };

  if (!c.published_at) return { keep: false, reason: "no-date" };
  const published = new Date(c.published_at + "T00:00:00Z");
  if (Number.isNaN(published.getTime())) return { keep: false, reason: "no-date" };
  const cutoff = new Date(w.now);
  cutoff.setUTCMonth(cutoff.getUTCMonth() - w.months);
  if (published < cutoff) return { keep: false, reason: "outside-window" };

  // A video without a transcript is a normal case, not an exception: it is
  // logged with its reason and dropped, never retried as an error.
  if (c.has_manual_en) return { keep: true, transcript_kind: "manual" };
  if (c.has_auto_en) return { keep: true, transcript_kind: "generated" };
  return { keep: false, reason: "no-transcript" };
}

/** Which tools a video mentions, matched on the aliases from corpus.yaml. */
export function toolsInTitle(
  title: string,
  tools: { id: string; aliases: string[] }[],
): string[] {
  const hay = ` ${normalizeTitle(title)} `;
  return tools
    .filter((t) => t.aliases.some((a) => hay.includes(` ${normalizeTitle(a)} `)))
    .map((t) => t.id);
}

export type Selected = { candidate: Candidate; transcript_kind: "manual" | "generated" };

export type SelectionReport = {
  selected: Selected[];
  rejected: { youtube_id: string; title: string; reason: Rejection }[];
};

/** Applies `decide` across candidates and drops title-level duplicates. */
export function select(candidates: Candidate[], w: Window): SelectionReport {
  const selected: Selected[] = [];
  const rejected: SelectionReport["rejected"] = [];
  const seen = new Set<string>();

  for (const c of candidates) {
    const d = decide(c, w);
    if (!d.keep) {
      rejected.push({ youtube_id: c.youtube_id, title: c.title, reason: d.reason });
      continue;
    }
    const key = normalizeTitle(c.title);
    if (seen.has(key)) {
      rejected.push({ youtube_id: c.youtube_id, title: c.title, reason: "duplicate-title" });
      continue;
    }
    seen.add(key);
    selected.push({ candidate: c, transcript_kind: d.transcript_kind });
  }
  return { selected, rejected };
}

/**
 * How many months the selection actually spans.
 *
 * The rule asks for a 12-18 month range and says why: outdated claims are the
 * second source of contradictions, alongside differences of opinion. A set that
 * is entirely from the last two months satisfies every per-video filter and
 * still fails the rule, so the spread is reported rather than assumed.
 */
export function spreadMonths(dates: string[]): number {
  if (dates.length < 2) return 0;
  const ts = dates.map((d) => new Date(d + "T00:00:00Z").getTime()).sort((a, b) => a - b);
  const days = (ts[ts.length - 1]! - ts[0]!) / 86_400_000;
  return days / 30.44;
}

/**
 * Picks `n` items spread across the time window instead of taking the newest n.
 *
 * yt-dlp returns a channel newest-first, so slicing the top gives a set from the
 * last few weeks: on a prolific channel, 8 videos spanned 0.3 months. That
 * satisfies every per-video filter and still breaks the rule the corpus is built
 * on — outdated claims are the second source of contradictions, and there are
 * none if everything was published at once.
 *
 * Candidates are bucketed by month and taken round-robin, oldest bucket first,
 * so the months present in the window are represented before any month is
 * doubled up.
 */
export function spreadPick<T>(items: T[], n: number, dateOf: (t: T) => string): T[] {
  if (n <= 0) return [];
  if (items.length <= n) return [...items];

  const buckets = new Map<string, T[]>();
  for (const it of items) {
    const key = dateOf(it).slice(0, 7);          // YYYY-MM
    const bucket = buckets.get(key);
    if (bucket) bucket.push(it);
    else buckets.set(key, [it]);
  }
  const keys = [...buckets.keys()].sort();

  // More months than picks: take months at evenly spaced positions, ends
  // included. Walking the sorted keys in order would take the n OLDEST months
  // and collapse the spread — which is the bug this function exists to fix.
  if (keys.length >= n) {
    const chosen = n === 1
      ? [keys[Math.floor((keys.length - 1) / 2)]!]
      : Array.from({ length: n }, (_, i) => keys[Math.round((i * (keys.length - 1)) / (n - 1))]!);
    return chosen.map((k) => buckets.get(k)![0]!);
  }

  // Fewer months than picks: every month is represented, then round-robin for
  // the remainder so no single month absorbs the surplus.
  const out: T[] = [];
  for (let round = 0; out.length < n; round++) {
    let progressed = false;
    for (const k of keys) {
      const bucket = buckets.get(k)!;
      if (round < bucket.length) {
        out.push(bucket[round]!);
        progressed = true;
        if (out.length === n) break;
      }
    }
    if (!progressed) break;
  }
  return out;
}
