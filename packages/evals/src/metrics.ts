// The metrics. Written against tests that already existed, with the expected
// values worked out by hand — invariant 4, because on the previous harness
// grader bugs outnumbered the real differences between configurations, and a
// wrong metric still produces a plausible table.
//
// Every number here says what it does NOT distinguish, because that is the
// thing a metric name hides.
import type { Gold } from "./golden.ts";
import { hitsFor, HIT_COVERAGE, type Interval } from "./hit.ts";

export type Retrieved = Interval & { video: string; score: number };

const covers = (g: Gold, r: Retrieved, threshold: number) =>
  g.video === r.video && hitsFor([g], r, threshold).length > 0;

/**
 * The first rank at which each gold span is covered, or null if never.
 *
 * Ranks rather than a count, so a single chunk crossing two gold spans credits
 * both once rather than counting twice, and so the caller can union finds
 * across ranks instead of re-scanning.
 */
export function foundByRank(
  golds: Gold[],
  ranked: Retrieved[],
  threshold = HIT_COVERAGE,
): (number | null)[] {
  return golds.map((g) => {
    const at = ranked.findIndex((r) => covers(g, r, threshold));
    return at === -1 ? null : at + 1;
  });
}

/**
 * Fraction of a question's gold spans found within the top k.
 *
 * Blind to: which sources were found. Two spans with one found scores 0.5
 * whichever half it was, and for a comparative question the two halves are not
 * interchangeable.
 *
 * The alternative — "any hit scores 1" — was rejected because it makes a
 * comparative question indistinguishable from a factual one, and the whole
 * point of a comparative question is that one source is not enough.
 *
 * Returns null for questions with no gold spans. Negatives have no correct
 * answer; they are measured by falsePositiveRate.
 */
export function recallAtK(
  golds: Gold[],
  ranked: Retrieved[],
  k: number,
  threshold = HIT_COVERAGE,
): number | null {
  if (golds.length === 0) return null;
  const found = foundByRank(golds, ranked.slice(0, k), threshold);
  return found.filter((r) => r !== null).length / golds.length;
}

/**
 * Reciprocal rank of the first chunk that finds any gold span.
 *
 * Blind to: everything after the first hit. A run that finds one source at
 * rank 1 and never finds the second scores the same as one that finds both.
 * That is why it is reported next to recall rather than instead of it.
 *
 * Null, not zero, for questions with no gold spans: MRR over a question with
 * no correct answer is undefined, and scoring it 0 would move the headline
 * number by an amount that depends only on how many negatives the set holds.
 */
export function mrr(golds: Gold[], ranked: Retrieved[], threshold = HIT_COVERAGE): number | null {
  if (golds.length === 0) return null;
  const ranks = foundByRank(golds, ranked, threshold).filter((r): r is number => r !== null);
  return ranks.length === 0 ? 0 : 1 / Math.min(...ranks);
}

/**
 * Whether top-k contains both sides of a contradiction.
 *
 * The failure this exists for: a system that confidently returns one side of a
 * live disagreement. Plain recall scores that as a success, and it is worse
 * than returning nothing.
 *
 * Blind to: how many chunks support each side, and whether the sides are
 * balanced. One chunk per side is full coverage.
 *
 * Null when the question does not carry both labels — there is nothing to
 * measure, and reporting false would understate coverage across the set.
 */
export function contradictionCoverage(
  golds: Gold[],
  ranked: Retrieved[],
  k: number,
  threshold = HIT_COVERAGE,
): boolean | null {
  const sides = new Set(golds.map((g) => g.side).filter(Boolean));
  if (!(sides.has("pro") && sides.has("contra"))) return null;

  const found = foundByRank(golds, ranked.slice(0, k), threshold);
  const seen = new Set(golds.filter((_, i) => found[i] !== null).map((g) => g.side));
  return seen.has("pro") && seen.has("contra");
}

/**
 * Share of negative questions the system answered anyway.
 *
 * Takes the top-1 score for each negative question and a threshold below which
 * the system is treated as declining to answer.
 *
 * The threshold is the whole metric. A plain top-k retriever always returns k
 * results, so with no abstention rule this is 1 by construction — which is the
 * finding, not a bug. Picking the threshold on this same question set would be
 * tuning on the test set, so it belongs in the report as a curve over
 * thresholds rather than a single number chosen after the fact.
 */
export function falsePositiveRate(topScores: number[], threshold: number): number | null {
  if (topScores.length === 0) return null;
  return topScores.filter((s) => s >= threshold).length / topScores.length;
}

/**
 * Whether term matching alone finds this question's answer.
 *
 * True when a lexical retriever puts a gold span first. Such a question is not
 * bad — it is a perfectly ordinary thing to ask.
 *
 * It is also NOT true that every configuration gets it, which is what this
 * function was originally documented and weighted on. Measured on 15 such
 * questions at 128-token chunks, recall@5 was 91.3% for BM25, 74.7% for
 * hosted embeddings and 36.0% for the local model — a spread of 55 pp. On the
 * other 34 questions the spread was 17 pp. So these questions separate
 * configurations more than three times better than the rest of the set, and
 * treating them as dead weight had it backwards.
 *
 * What the flag is good for is the split in the report: "how much better is
 * retrieval than grep" is only a question about the half where grep works,
 * and averaging the two halves hides both answers.
 *
 * This replaced a threshold on question/passage word overlap, which was a proxy
 * and a poor one: overlap is a fraction of the *question's* words, so a
 * four-word question with all four in the passage scores 100% for being short.
 *
 * Used to split the report, never to select questions. Selecting on a
 * retriever's output tunes the set to that retriever; labelling for the report
 * discards nothing and answers the question worth asking — how much better is
 * retrieval than grep, on the questions where grep does not work.
 */
export function lexicallyTrivial(
  golds: Gold[],
  lexicalRanked: Retrieved[],
  threshold = HIT_COVERAGE,
): boolean | null {
  if (golds.length === 0) return null;
  const first = lexicalRanked[0];
  if (!first) return false;
  return golds.some((g) => covers(g, first, threshold));
}
