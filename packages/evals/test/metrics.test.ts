// Tests before implementation, and every expected value worked out by hand.
// Invariant 4 exists because on the previous harness grader bugs outnumbered
// the real differences between configurations — a metric that is wrong still
// produces a plausible table.
import { describe, expect, test } from "vitest";
import {
  contradictionCoverage, falsePositiveRate, foundByRank, mrr, recallAtK,
  type Retrieved,
} from "../src/metrics.ts";
import type { Gold } from "../src/golden.ts";

const gold = (video: string, start_s: number, end_s: number, side?: "pro" | "contra"): Gold =>
  ({ video, start_s, end_s, ...(side ? { side } : {}) });
const hit = (video: string, start_s: number, end_s: number, score = 1): Retrieved =>
  ({ video, start_s, end_s, score });

describe("foundByRank", () => {
  test("reports the first rank at which each gold span is covered", () => {
    const golds = [gold("v1", 0, 20), gold("v2", 0, 20)];
    // rank 1 covers gold 0; rank 3 covers gold 1; rank 2 covers nothing.
    const ranked = [hit("v1", 0, 20), hit("v9", 0, 20), hit("v2", 0, 20)];
    expect(foundByRank(golds, ranked)).toEqual([1, 3]);
  });

  test("a gold span never covered has no rank", () => {
    expect(foundByRank([gold("v1", 0, 20)], [hit("v2", 0, 20)])).toEqual([null]);
  });

  test("one chunk covering two gold spans credits both at the same rank", () => {
    const golds = [gold("v1", 0, 20), gold("v1", 20, 40)];
    expect(foundByRank(golds, [hit("v1", 0, 40)])).toEqual([1, 1]);
  });

  test("a later chunk does not overwrite an earlier find", () => {
    const golds = [gold("v1", 0, 20)];
    expect(foundByRank(golds, [hit("v1", 0, 20), hit("v1", 0, 20)])).toEqual([1]);
  });

  test("the same video is not enough — the interval has to overlap", () => {
    expect(foundByRank([gold("v1", 0, 20)], [hit("v1", 500, 520)])).toEqual([null]);
  });
});

describe("recallAtK", () => {
  test("one gold span found is 1, missed is 0", () => {
    const g = [gold("v1", 0, 20)];
    expect(recallAtK(g, [hit("v1", 0, 20)], 5)).toBe(1);
    expect(recallAtK(g, [hit("v2", 0, 20)], 5)).toBe(0);
  });

  test("two gold spans, one found, is 0.5 — not 1", () => {
    // The alternative definition, "any hit counts", would score this 1 and make
    // a comparative question indistinguishable from a factual one: finding one
    // of two sources is genuinely half the answer.
    const golds = [gold("v1", 0, 20), gold("v2", 0, 20)];
    expect(recallAtK(golds, [hit("v1", 0, 20)], 5)).toBe(0.5);
  });

  test("k truncates the list", () => {
    const golds = [gold("v2", 0, 20)];
    const ranked = [hit("v9", 0, 20), hit("v8", 0, 20), hit("v2", 0, 20)];
    expect(recallAtK(golds, ranked, 2)).toBe(0);
    expect(recallAtK(golds, ranked, 3)).toBe(1);
  });

  test("a question with no gold spans is not scored", () => {
    // Negatives have no correct answer, so they belong to false-positive rate.
    expect(recallAtK([], [hit("v1", 0, 20)], 5)).toBeNull();
  });
});

describe("mrr", () => {
  test("reciprocal of the first rank that finds anything", () => {
    const golds = [gold("v1", 0, 20)];
    expect(mrr(golds, [hit("v1", 0, 20)])).toBe(1);
    expect(mrr(golds, [hit("v9", 0, 20), hit("v1", 0, 20)])).toBe(0.5);
    expect(mrr(golds, [hit("v9", 0, 20), hit("v8", 0, 20), hit("v1", 0, 20)]))
      .toBeCloseTo(1 / 3, 10);
  });

  test("nothing found is 0", () => {
    expect(mrr([gold("v1", 0, 20)], [hit("v2", 0, 20)])).toBe(0);
  });

  test("the earliest find wins, even when a later chunk covers more", () => {
    const golds = [gold("v1", 0, 20), gold("v2", 0, 20)];
    expect(mrr(golds, [hit("v1", 0, 20), hit("v2", 0, 20)])).toBe(1);
  });

  test("a question with no gold spans is undefined, not zero", () => {
    // Scoring it 0 would drag the headline down by an amount that depends only
    // on how many negatives the set happens to contain.
    expect(mrr([], [hit("v1", 0, 20)])).toBeNull();
  });
});

describe("contradictionCoverage", () => {
  const pro = gold("v1", 0, 20, "pro");
  const contra = gold("v2", 0, 20, "contra");

  test("both sides in top-k is covered", () => {
    expect(contradictionCoverage([pro, contra], [hit("v1", 0, 20), hit("v2", 0, 20)], 5)).toBe(true);
  });

  test("one side only is not covered, however many chunks agree with it", () => {
    // The failure the metric exists for: confidently returning one side of a
    // live disagreement, which plain recall scores as a success.
    expect(contradictionCoverage([pro, contra], [hit("v1", 0, 20), hit("v1", 0, 20)], 5)).toBe(false);
  });

  test("k cuts off the second side", () => {
    const ranked = [hit("v1", 0, 20), hit("v9", 0, 20), hit("v2", 0, 20)];
    expect(contradictionCoverage([pro, contra], ranked, 2)).toBe(false);
    expect(contradictionCoverage([pro, contra], ranked, 3)).toBe(true);
  });

  test("both sides in one video still counts when different chunks carry them", () => {
    const same = [gold("v1", 0, 20, "pro"), gold("v1", 600, 620, "contra")];
    expect(contradictionCoverage(same, [hit("v1", 0, 20), hit("v1", 600, 620)], 5)).toBe(true);
  });

  test("a question without both sides labelled is not measurable", () => {
    expect(contradictionCoverage([pro], [hit("v1", 0, 20)], 5)).toBeNull();
  });
});

describe("falsePositiveRate", () => {
  test("counts negatives whose top result clears the threshold", () => {
    // Two of four negatives return something the system is confident about.
    const scores = [0.9, 0.8, 0.4, 0.3];
    expect(falsePositiveRate(scores, 0.75)).toBe(0.5);
  });

  test("the threshold is inclusive", () => {
    expect(falsePositiveRate([0.5], 0.5)).toBe(1);
  });

  test("with no threshold every top-k answer is a false positive", () => {
    // A plain top-k retriever always returns k results, so without an
    // abstention rule this metric is 1 by construction. That is the point:
    // the number is meaningless until the system can decline to answer.
    expect(falsePositiveRate([0.9, 0.1, 0.01], 0)).toBe(1);
  });

  test("no negatives means nothing to report", () => {
    expect(falsePositiveRate([], 0.5)).toBeNull();
  });
});
