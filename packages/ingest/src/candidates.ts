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
