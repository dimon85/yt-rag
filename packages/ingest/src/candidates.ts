// Ranking passages for annotation. Pure functions: segments in, scored windows
// out, no filesystem and no network.
//
// The problem this solves is volume. The corpus holds 3,883 thirty-second
// windows; 358 mention a tool at all. Reading all of them to write 90 questions
// is the bulk of the work, and most of what gets read is unusable — a passing
// mention with no claim attached cannot become a question with a gold span.
//
// What it deliberately does NOT do is write the question. Scoring finds where
// claims are; the phrasing has to come from somewhere other than the passage,
// or the question is found by term matching alone and stops separating one
// retriever from another.
import { normalizeTranscript, toolMentions, type Segment } from "./text.ts";

export type Window = {
  segments: Segment[];
  start_s: number;
  end_s: number;
  text: string;
};

/**
 * Slices a transcript into overlapping windows.
 *
 * Overlapping because a claim that straddles a boundary would otherwise be
 * split across two low-scoring windows and rank below a single window holding
 * something weaker. The step is half the width, so every claim appears whole in
 * at least one window.
 */
export function windows(segments: Segment[], widthS = 30): Window[] {
  if (segments.length === 0) return [];
  const out: Window[] = [];
  const stepS = widthS / 2;
  const last = segments[segments.length - 1]!.end_s;

  for (let from = segments[0]!.start_s; from < last; from += stepS) {
    const to = from + widthS;
    const inside = segments.filter((s) => s.end_s > from && s.start_s < to);
    if (inside.length === 0) continue;
    out.push({
      segments: inside,
      start_s: inside[0]!.start_s,
      end_s: inside[inside.length - 1]!.end_s,
      text: inside.map((s) => s.text).join(" "),
    });
  }
  return out;
}

/** A number attached to a unit — the anchor of a checkable factual claim. */
const NUMBER = /\b\d+([.,]\d+)?\s*(percent|%|dollars?|bucks?|hours?|minutes?|seconds?|tokens?|times|x|k|m)\b/gi;

/** Phrases that mark an assertion rather than narration. */
const CLAIM = /\b(better than|worse than|instead of|switched from|used to|the problem is|doesn'?t work|does not work|i recommend|you should|turned out|in my experience|the trick is|broke|failed|never use|stopped using)\b/gi;

/** Phrases that mark a stance someone else could contradict. */
const OPINION = /\b(honestly|hot take|i disagree|overhyped|underrated|everyone says|people think|in my opinion|the truth is)\b/gi;

export type Score = {
  total: number;
  reasons: string[];
  tools: string[];
  topics: string[];
};

const count = (text: string, re: RegExp) => (text.match(re) ?? []).length;

/**
 * Whole-word phrase match against normalized text.
 *
 * A plain includes() found the keyword "ci" inside "decision", "specific" and
 * "efficiency", which reported 97 testing passages for one tool. Padding both
 * the haystack and the needle with spaces is enough here, because normalization
 * has already reduced every separator to a single space.
 */
export function hasPhrase(normalized: string, phrase: string): boolean {
  const needle = ` ${phrase.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim()} `;
  return needle.trim() !== "" && ` ${normalized} `.includes(needle);
}

/**
 * How likely a window is to yield an annotatable question.
 *
 * The weights are a judgement, not a measurement, and they only decide reading
 * order — a low score hides nothing, it just sinks. What matters is that the
 * reasons are printed alongside, so a bad ranking is visible rather than
 * mysterious.
 */
export function score(
  w: Window,
  tools: { id: string; aliases: string[] }[],
  topics: { id: string; keywords: string[] }[],
  opts: { scarceStance?: boolean } = {},
): Score {
  const normalized = normalizeTranscript(w.segments);
  const reasons: string[] = [];
  let total = 0;

  const mentioned = toolMentions(normalized, tools);
  const toolIds = [...mentioned.keys()];
  if (toolIds.length > 0) {
    // Capped: ten mentions of one tool is not ten times more useful than two.
    const points = Math.min(3, [...mentioned.values()].reduce((a, b) => a + b, 0));
    total += points;
    reasons.push(`tool:${toolIds.join("+")}`);
  }

  const topicIds = topics
    .filter((t) => t.keywords.some((k) => hasPhrase(normalized, k)))
    .map((t) => t.id);
  if (topicIds.length > 0) {
    total += 2;
    reasons.push(`topic:${topicIds.join("+")}`);
  }

  const numbers = count(w.text, NUMBER);
  if (numbers > 0) {
    total += Math.min(4, numbers * 2);
    reasons.push(`${numbers} number${numbers === 1 ? "" : "s"}`);
  }

  const claims = count(w.text, CLAIM);
  if (claims > 0) {
    total += Math.min(4, claims * 2);
    reasons.push(`${claims} claim marker${claims === 1 ? "" : "s"}`);
  }

  const opinions = count(w.text, OPINION);
  if (opinions > 0) {
    total += 1;
    reasons.push(`${opinions} opinion marker${opinions === 1 ? "" : "s"}`);
  }

  // Skeptical channels hold the whole contra side of the corpus — 18 videos of
  // 78 — so a passage from one is worth surfacing earlier than its raw score.
  if (opts.scarceStance && total > 0) {
    total += 2;
    reasons.push("scarce stance");
  }

  return { total, reasons, tools: toolIds, topics: topicIds };
}

export type Scored = { window: Window; score: Score };

/**
 * Drops windows that overlap a higher-scoring one.
 *
 * Windows step by half their width, so a strong claim produces two or three
 * near-identical neighbours. Without this the top of the list is the same
 * passage three times.
 */
export function dedupeOverlapping(scored: Scored[]): Scored[] {
  const kept: Scored[] = [];
  for (const s of [...scored].sort((a, b) => b.score.total - a.score.total)) {
    const clash = kept.some(
      (k) => k.window.end_s > s.window.start_s && k.window.start_s < s.window.end_s,
    );
    if (!clash) kept.push(s);
  }
  return kept;
}

/**
 * Partitions a transcript into non-overlapping windows.
 *
 * Distinct from `windows`, which steps by half its width so a claim on a
 * boundary survives whole somewhere. That overlap is right for a reading list
 * and wrong for sourcing gold spans, and the difference is not small: greedy
 * non-overlap selection over half-stepped windows collapses 770 candidates to
 * 52, drawn from 21 of 78 videos, because one strong window kills its
 * neighbours. Tiling the same corpus at the same score floor gives 139 spans
 * across 39 videos.
 *
 * A golden set sourced from the first number would sit almost entirely in a
 * quarter of the corpus, and every gold span would be the densest passage in
 * its neighbourhood — which is to say the easiest to retrieve. Recall would
 * come out high for reasons that have nothing to do with the retriever.
 */
export function tile(segments: Segment[], widthS = 30): Window[] {
  if (segments.length === 0) return [];
  const out: Window[] = [];
  const last = segments[segments.length - 1]!.end_s;

  for (let from = segments[0]!.start_s; from < last; from += widthS) {
    const to = from + widthS;
    const inside = segments.filter((s) => s.end_s > from && s.start_s < to);
    if (inside.length === 0) continue;
    out.push({
      segments: inside,
      start_s: inside[0]!.start_s,
      end_s: Math.max(...inside.map((s) => s.end_s)),
      text: inside.map((s) => s.text).join(" "),
    });
  }
  return out;
}

/**
 * Samples evenly across score bands rather than taking the top n.
 *
 * Taking the top would fill the golden set with the corpus's densest passages,
 * which are its most findable ones. Stratifying keeps harder spans in, so
 * recall measures the retriever instead of the selection.
 */
export function stratify<T extends { score: Score }>(rows: T[], n: number, bands = 4): T[] {
  if (rows.length <= n) return [...rows];
  const sorted = [...rows].sort((a, b) => b.score.total - a.score.total);
  const perBand = Math.ceil(n / bands);
  const size = Math.ceil(sorted.length / bands);
  const out: T[] = [];

  for (let b = 0; b < bands && out.length < n; b++) {
    const slice = sorted.slice(b * size, (b + 1) * size);
    // Even spacing inside the band, so a band is not represented by its top
    // few either.
    const step = Math.max(1, Math.floor(slice.length / perBand));
    for (let i = 0; i < slice.length && out.length < n; i += step) out.push(slice[i]!);
  }
  return out;
}

/**
 * Sponsor reads and calls to action.
 *
 * They score well — a sponsor segment is dense with product names and numbers —
 * and they are useless as gold spans, because the claim is about a product
 * outside the corpus axis. One reached a generated batch: "get started with
 * SERP API using 250 free credits", perfectly answerable and about none of the
 * four tools being measured.
 *
 * The word "sponsor" alone does not catch them. That one said "clicking the
 * link in the description" and "scan the QR code".
 */
/**
 * Read-outs that are advertising rather than a claim about a tool.
 *
 * Extended after one slipped into a candidate pair: "I highly recommend you go
 * and check out firecro" matched none of the original forms, and the passage
 * became half a gold span before it was read. The blunter cause was that this
 * pattern was exported, tested, and applied nowhere — an ad filter that
 * filtered nothing. `pnpm pairs` now marks passages it matches instead of
 * dropping them, because a 90-second window can hold an ad read and a real
 * claim, and the person reading the candidate is the one who can tell.
 */
export const AD_PATTERN =
  /sponsor|word from|link in the description|link below|scan the qr|use my code|discount code|free credits|sign up (?:for|using)|recommend you go|go and check out|\d+% off/i;

/**
 * Whether a window overlaps anything already chosen.
 *
 * `tile` partitions by time, but a window's reported span comes from the
 * segments it contains, and captions in this corpus overlap in time — 13,232
 * of the segment pairs do. Two tiles can therefore report spans that touch.
 * Two adjacent picks produced near-duplicate questions in the first batch,
 * eight seconds apart and both about the same claim.
 */
export function clashes(
  candidate: { start_s: number; end_s: number },
  taken: { start_s: number; end_s: number }[],
): boolean {
  return taken.some((t) => t.end_s > candidate.start_s && t.start_s < candidate.end_s);
}
