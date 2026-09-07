import { describe, expect, test } from "vitest";
import { jaccard, mmrRerank, type Candidate } from "../src/mmr.ts";

/** Documents are points on a line; similarity falls off with distance. */
const near = (a: number, b: number) => Math.max(0, 1 - Math.abs(a - b) / 10);
const c = (item: number, relevance: number): Candidate<number> => ({ item, relevance });

describe("mmrRerank", () => {
  test("lambda 1 is plain relevance ranking", () => {
    const out = mmrRerank([c(0, 0.5), c(1, 0.9), c(2, 0.7)], near, { lambda: 1 });
    expect(out.map((x) => x.item)).toEqual([1, 2, 0]);
  });

  test("the first pick is the most relevant whenever lambda is above 0", () => {
    for (const lambda of [0.1, 0.3, 0.5, 1]) {
      const out = mmrRerank([c(0, 0.4), c(1, 0.95), c(2, 0.6)], near, { lambda });
      expect(out[0]!.item).toBe(1);
    }
  });

  test("lambda 0 ignores the query entirely, including for the first pick", () => {
    // Nothing has been picked yet, so redundancy is 0 for every candidate and
    // the score is 0 * relevance = 0 across the board. The formula says
    // relevance does not count at lambda 0, and that has to include the first
    // selection — special-casing it would make the documented formula a
    // half-truth in the one place it is easiest to misread.
    const out = mmrRerank([c(0, 0.4), c(1, 0.95), c(2, 0.6)], near, { lambda: 0 });
    expect(out[0]!.item).toBe(0);
  });

  test("the second pick avoids a near-duplicate of the first", () => {
    // 1 and 2 sit on top of each other; 30 is far away and slightly less
    // relevant. Plain ranking takes 2 second; MMR takes 30.
    const cands = [c(1, 0.9), c(2, 0.88), c(30, 0.8)];
    expect(mmrRerank(cands, near, { lambda: 1 }).map((x) => x.item)).toEqual([1, 2, 30]);
    expect(mmrRerank(cands, near, { lambda: 0.5 }).map((x) => x.item)).toEqual([1, 30, 2]);
  });

  test("this is the contradiction case in miniature", () => {
    // Four chunks arguing one side, one arguing the other and slightly less
    // relevant. Top-3 by relevance never reaches the other side; MMR does.
    const oneSide = [c(0, 0.9), c(1, 0.89), c(2, 0.88), c(3, 0.87)];
    const otherSide = c(50, 0.7);
    const cands = [...oneSide, otherSide];

    expect(mmrRerank(cands, near, { lambda: 1, k: 3 }).map((x) => x.item)).not.toContain(50);
    expect(mmrRerank(cands, near, { lambda: 0.5, k: 3 }).map((x) => x.item)).toContain(50);
  });

  test("k truncates", () => {
    expect(mmrRerank([c(0, 0.5), c(1, 0.9), c(2, 0.7)], near, { k: 2 })).toHaveLength(2);
  });

  test("every candidate appears exactly once", () => {
    const out = mmrRerank([c(0, 0.5), c(1, 0.9), c(2, 0.7), c(3, 0.6)], near, { lambda: 0.5 });
    expect(new Set(out.map((x) => x.item)).size).toBe(4);
  });

  test("identical input gives identical output", () => {
    const cands = [c(0, 0.5), c(1, 0.5), c(2, 0.5)];
    expect(mmrRerank(cands, near, { lambda: 0.5 })).toEqual(mmrRerank(cands, near, { lambda: 0.5 }));
  });

  test("an empty list is not a crash", () => {
    expect(mmrRerank([], near)).toEqual([]);
  });

  test("lambda outside [0,1] is rejected", () => {
    expect(() => mmrRerank([c(0, 1)], near, { lambda: 1.5 })).toThrow(/lambda must be/);
    expect(() => mmrRerank([c(0, 1)], near, { lambda: -0.1 })).toThrow(/lambda must be/);
  });
});

describe("jaccard", () => {
  test("identical sets are 1, disjoint are 0", () => {
    expect(jaccard(new Set(["a", "b"]), new Set(["a", "b"]))).toBe(1);
    expect(jaccard(new Set(["a"]), new Set(["b"]))).toBe(0);
  });

  test("half-shared is one third, not one half", () => {
    // |A ∩ B| = 1, |A ∪ B| = 3.
    expect(jaccard(new Set(["a", "b"]), new Set(["b", "c"]))).toBeCloseTo(1 / 3, 10);
  });

  test("two empty sets are 0 rather than NaN", () => {
    expect(jaccard(new Set(), new Set())).toBe(0);
  });
});
