// What counts as finding the answer.
//
// A gold annotation is an interval of a video; a retrieved chunk is another
// interval. Whether they "match" needs a rule, and the schema's `is_hit
// boolean` does not contain one. Left unstated it gets decided by accident,
// and the accident has a direction:
//
//   any non-zero overlap    — longer chunks win automatically, because a
//                             1024-token chunk crosses more gold intervals
//                             than a 512-token one no matter how relevant it
//                             is. Configuration #3 would take the ablation
//                             before a single number was interpreted.
//   gold centre inside      — cheap, but a chunk that clips one second either
//                             side of the centre counts the same as one that
//                             contains the whole answer.
//   IoU >= t                — symmetric, and punishes a correct chunk for
//                             being long even when it fully contains the
//                             answer. Under IoU 0.5 a 90-second chunk that
//                             contains a 10-second gold span scores 0.11 and
//                             misses.
//
// The rule used here: the shorter interval must be at least half covered by
// the other. A chunk containing the whole answer is a hit regardless of chunk
// length; a chunk that clips the edge of a long answer is not.
//
//   overlap / min(len_gold, len_chunk) >= 0.5
//
// It is not obviously the best rule. It is written down, tested, and identical
// across every configuration — which matters more than which rule it is.

export type Interval = { start_s: number; end_s: number };

export const HIT_COVERAGE = 0.5;

export function overlapSeconds(a: Interval, b: Interval): number {
  return Math.max(0, Math.min(a.end_s, b.end_s) - Math.max(a.start_s, b.start_s));
}

/**
 * Coverage of the shorter interval by the other, in [0, 1].
 *
 * Using min() rather than the gold length is what makes the rule symmetric in
 * the case that matters: a chunk sitting entirely inside a three-minute gold
 * span is fully covered and counts, and so does a chunk that fully contains a
 * five-second one. Dividing by the gold length alone would fail the first;
 * dividing by the union would fail both.
 */
export function coverage(gold: Interval, chunk: Interval): number {
  const lenGold = gold.end_s - gold.start_s;
  const lenChunk = chunk.end_s - chunk.start_s;
  if (lenGold <= 0 || lenChunk <= 0) return 0;
  return overlapSeconds(gold, chunk) / Math.min(lenGold, lenChunk);
}

export function isHit(gold: Interval, chunk: Interval, threshold = HIT_COVERAGE): boolean {
  return coverage(gold, chunk) >= threshold;
}

/**
 * Which gold spans a chunk covers.
 *
 * Returned as indices rather than a count: a question with several gold spans
 * must not be credited twice for one chunk that happens to cross two of them,
 * and the caller needs to know *which* were found to union across ranks.
 */
export function hitsFor(
  golds: Interval[],
  chunk: Interval,
  threshold = HIT_COVERAGE,
): number[] {
  return golds.flatMap((g, i) => (isHit(g, chunk, threshold) ? [i] : []));
}
