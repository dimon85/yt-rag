import { describe, expect, test } from "vitest";
import { coverage, hitsFor, isHit, overlapSeconds } from "../src/hit.ts";

const iv = (start_s: number, end_s: number) => ({ start_s, end_s });

describe("overlapSeconds", () => {
  test("disjoint intervals overlap by nothing", () => {
    expect(overlapSeconds(iv(0, 10), iv(20, 30))).toBe(0);
  });

  test("touching at a point is not overlap", () => {
    expect(overlapSeconds(iv(0, 10), iv(10, 20))).toBe(0);
  });

  test("partial overlap is the shared seconds", () => {
    expect(overlapSeconds(iv(0, 10), iv(6, 20))).toBe(4);
  });

  test("is symmetric", () => {
    expect(overlapSeconds(iv(0, 10), iv(6, 20))).toBe(overlapSeconds(iv(6, 20), iv(0, 10)));
  });
});

describe("coverage", () => {
  test("a chunk containing the whole answer covers it fully", () => {
    // The case IoU gets wrong: a 90s chunk holding a 10s answer scores 0.11
    // under IoU and misses, though it contains the answer completely.
    expect(coverage(iv(100, 110), iv(60, 150))).toBe(1);
  });

  test("a chunk inside a long answer is fully covered too", () => {
    // The reverse case: a 3-minute gold span, a 30s chunk inside it. The chunk
    // is entirely about the answer, so it counts.
    expect(coverage(iv(0, 180), iv(60, 90))).toBe(1);
  });

  test("half the shorter interval is exactly the threshold", () => {
    expect(coverage(iv(0, 20), iv(10, 40))).toBe(0.5);
  });

  test("zero-length intervals cover nothing rather than dividing by zero", () => {
    expect(coverage(iv(10, 10), iv(0, 20))).toBe(0);
    expect(coverage(iv(0, 20), iv(10, 10))).toBe(0);
  });
});

describe("isHit", () => {
  test("clipping the edge of a long answer is not a hit", () => {
    // 5 seconds of a 100-second gold span, in a 60-second chunk.
    expect(isHit(iv(0, 100), iv(95, 155))).toBe(false);
  });

  test("the threshold is inclusive", () => {
    expect(isHit(iv(0, 20), iv(10, 40))).toBe(true);
  });

  test("a longer chunk does not win by being longer", () => {
    // The failure this rule exists to prevent: under "any overlap" this is a
    // hit, and the 1024-token configuration takes the ablation for reasons
    // that have nothing to do with retrieval quality.
    expect(isHit(iv(0, 60), iv(59, 600))).toBe(false);
  });

  test("a very short chunk clipping a long answer IS a hit — known property", () => {
    // min() protects the case that matters — a chunk sitting inside a
    // three-minute answer — but the same protection credits a 1.5-second chunk
    // that catches one second of a one-minute answer, because the chunk is
    // mostly inside the span.
    //
    // Harmless at the chunk sizes this project uses: 512 tokens is roughly
    // 60-90 seconds of speech, so intervals this short do not occur. Recorded
    // rather than patched, because a future configuration producing very short
    // chunks would need this revisited, and a silent property is the kind that
    // decides an ablation quietly.
    expect(isHit(iv(0, 60), iv(59, 60.5))).toBe(true);
  });

  test("an exact match is a hit", () => {
    expect(isHit(iv(10, 40), iv(10, 40))).toBe(true);
  });
});

describe("hitsFor", () => {
  test("reports which gold spans were covered, not how many times", () => {
    // One chunk crossing two gold spans must not credit the question twice;
    // the caller unions indices across ranks instead.
    const golds = [iv(0, 20), iv(15, 35), iv(200, 220)];
    expect(hitsFor(golds, iv(0, 40))).toEqual([0, 1]);
  });

  test("no coverage returns nothing", () => {
    expect(hitsFor([iv(0, 20)], iv(500, 600))).toEqual([]);
  });

  test("an empty gold list can never be hit — the negative case", () => {
    expect(hitsFor([], iv(0, 100))).toEqual([]);
  });
});
