import { describe, expect, test } from "vitest";
import { binomPmf, binomTest, clopperPearson, mdePaired, powerPaired } from "../src/power.ts";

describe("the spec's own table, reproduced", () => {
  // docs/spec.md, "Why 90 and not 40", claims these four numbers. They were
  // computed once and written down, which means nothing in the repo could
  // check them. These are the check.
  test.each([
    [34, 0.225],
    [51, 0.155],
    [76, 0.105],
    [102, 0.08],
  ])("a recall pool of %i detects %f", (n, expected) => {
    expect(mdePaired(n, 0)).toBeCloseTo(expected, 3);
  });
});

describe("small n", () => {
  test("five items detect nothing at all, and say so rather than returning a number", () => {
    // A large number here would read as "we can measure something, just
    // barely". null is the honest answer: no effect in [0,1] reaches 80%
    // power, including an effect of 100 percentage points.
    expect(mdePaired(5, 0)).toBeNull();
    expect(powerPaired(5, 1.0, 0)).toBeLessThan(0.8);
  });

  test("power rises with n at a fixed effect", () => {
    const at = (n: number) => powerPaired(n, 0.2, 0);
    expect(at(10)).toBeLessThan(at(30));
    expect(at(30)).toBeLessThan(at(80));
  });

  test("the target of 18 contradictions could not compare configurations either", () => {
    // The spec says so in words ("What 90 does not buy"); this is the number.
    // A 20pp difference on 18 paired items is found 13% of the time.
    expect(powerPaired(18, 0.2, 0)).toBeLessThan(0.2);
    expect(mdePaired(18, 0)).toBeGreaterThan(0.4);
  });
});

describe("noise", () => {
  test("items that flip on their own cost power", () => {
    // Retrieval here is deterministic, so noise is 0 — but the parameter has
    // to exist for the claim "deterministic" to be doing any work.
    expect(powerPaired(76, 0.1, 0)).toBeGreaterThan(powerPaired(76, 0.1, 0.2));
  });

  test("a perfect effect against total noise does not produce NaN", () => {
    // pb + pc must not exceed 1: they are exclusive states of one item.
    const p = powerPaired(20, 1.0, 1.0);
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  test("noise caps how large an effect can even be posited", () => {
    expect(mdePaired(76, 1.0)).toBeNull();
  });
});

describe("binomial pieces", () => {
  test("the density sums to one", () => {
    const total = Array.from({ length: 21 }, (_, k) => binomPmf(20, k, 0.3))
      .reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 10);
  });

  test("degenerate p is exact rather than a limit", () => {
    expect(binomPmf(5, 0, 0)).toBe(1);
    expect(binomPmf(5, 3, 0)).toBe(0);
    expect(binomPmf(5, 5, 1)).toBe(1);
  });

  test("out-of-range k has no density", () => {
    expect(binomPmf(5, -1, 0.5)).toBe(0);
    expect(binomPmf(5, 6, 0.5)).toBe(0);
  });

  test("the two-sided test is symmetric and never exceeds one", () => {
    expect(binomTest(3, 10, 0.5)).toBeCloseTo(binomTest(7, 10, 0.5), 12);
    expect(binomTest(5, 10, 0.5)).toBeLessThanOrEqual(1);
    expect(binomTest(0, 0, 0.5)).toBe(1);
  });

  test("an extreme split is significant, an even one is not", () => {
    expect(binomTest(10, 10, 0.5)).toBeLessThan(0.05);
    expect(binomTest(5, 10, 0.5)).toBeGreaterThan(0.05);
  });
});

describe("clopper-pearson", () => {
  test("a clean sweep is not reported as certainty", () => {
    // The reason the interval is exact rather than normal: on 0 of 23 the
    // normal approximation gives [0, 0] — "cannot happen" — which the data
    // does not say.
    const [lo, hi] = clopperPearson(0, 23);
    expect(lo).toBe(0);
    expect(hi).toBeGreaterThan(0.1);
    expect(hi).toBeLessThan(0.15);
  });

  test("the negatives argument in the spec holds up", () => {
    // "At 14 questions a perfect result reads as 0% to 23%. At 30, 0% to 12%."
    expect(clopperPearson(0, 14)[1]).toBeCloseTo(0.23, 2);
    expect(clopperPearson(0, 30)[1]).toBeCloseTo(0.12, 2);
  });

  test("five items say almost nothing about a proportion", () => {
    const [lo, hi] = clopperPearson(2, 5);
    expect(lo).toBeLessThan(0.1);
    expect(hi).toBeGreaterThan(0.8);
  });

  test("an empty sample is the whole range, not a division by zero", () => {
    expect(clopperPearson(0, 0)).toEqual([0, 1]);
  });

  test("the interval contains the estimate and narrows with n", () => {
    for (const n of [10, 50, 200]) {
      const [lo, hi] = clopperPearson(Math.round(n / 2), n);
      expect(lo).toBeLessThan(0.5);
      expect(hi).toBeGreaterThan(0.5);
    }
    const width = (n: number) => {
      const [lo, hi] = clopperPearson(Math.round(n / 2), n);
      return hi - lo;
    };
    expect(width(200)).toBeLessThan(width(50));
    expect(width(50)).toBeLessThan(width(10));
  });
});
