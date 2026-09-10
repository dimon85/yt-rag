import { describe, expect, test } from "vitest";
import type { Question } from "../src/golden.ts";
import type { Retrieved } from "../src/metrics.ts";
import {
  chunkSha, discordance, falsePositives, foundWithin, Header, Line, meanOf,
  medianThreshold, repeatDisagreements, Result, scoreQuestion, tally,
} from "../src/run.ts";

const span = (start_s: number, end_s: number, score = 1, video = "aaaaaaaaaaa"): Retrieved =>
  ({ video, start_s, end_s, score });

const question = (over: Partial<Question> = {}): Question => ({
  slug: "q",
  text: "a question long enough to pass validation",
  kind: "factual",
  topics: [],
  tools: [],
  source: "hand",
  gold: [{ video: "aaaaaaaaaaa", start_s: 100, end_s: 120 }],
  ...over,
});

describe("scoreQuestion", () => {
  test("records recall at every cut-off and the spans they came from", () => {
    // Gold is 100-120, 20 s long. Rank 3 is the only chunk that covers it:
    // 105-125 overlaps 15 s of a 20 s span, coverage 0.75, a hit. Rank 1 and 2
    // touch nothing. So recall@1 = 0, recall@3 = 1, MRR = 1/3.
    const ranked = [span(0, 20, 9), span(40, 60, 8), span(105, 125, 7)];
    const r = scoreQuestion(question(), ranked, 1, false);

    expect(r.recall["1"]).toBe(0);
    expect(r.recall["3"]).toBe(1);
    expect(r.recall["5"]).toBe(1);
    expect(r.mrr).toBeCloseTo(1 / 3, 10);
    expect(r.spans).toHaveLength(3);
    expect(r.top1).toBe(9);
  });

  test("a negative question records nulls, not zeros", () => {
    // Invariant 13. A zero here would drag the headline number by an amount
    // that depends only on how many negatives the set holds.
    const r = scoreQuestion(question({ kind: "negative", gold: [] }), [span(0, 20, 4)], 1, null);
    expect(r.recall["1"]).toBeNull();
    expect(r.recall["10"]).toBeNull();
    expect(r.mrr).toBeNull();
    // The top-1 score IS recorded: it is the only input false-positive rate
    // needs, and it is the whole reason a negative question is in the set.
    expect(r.top1).toBe(4);
  });

  test("contradiction coverage is null on a question with only one side", () => {
    const r = scoreQuestion(question(), [span(100, 120)], 1, false);
    expect(r.contradiction["5"]).toBeNull();
  });

  test("contradiction coverage is true only when both sides land in top-k", () => {
    const q = question({
      kind: "contradiction",
      gold: [
        { video: "aaaaaaaaaaa", start_s: 0, end_s: 20, side: "pro" },
        { video: "bbbbbbbbbbb", start_s: 0, end_s: 20, side: "contra" },
      ],
    });
    const proOnly = scoreQuestion(q, [span(0, 20, 9)], 1, false);
    expect(proOnly.contradiction["5"]).toBe(false);

    const both = scoreQuestion(q, [span(0, 20, 9), span(0, 20, 8, "bbbbbbbbbbb")], 1, false);
    expect(both.contradiction["5"]).toBe(true);
  });

  test("an empty ranking has no top-1 score", () => {
    expect(scoreQuestion(question(), [], 1, false).top1).toBeNull();
  });

  test("every line it writes parses back as a Result", () => {
    const line = scoreQuestion(question(), [span(100, 120)], 2, true);
    expect(Result.parse(JSON.parse(JSON.stringify(line)))).toEqual(line);
    expect(Line.parse(JSON.parse(JSON.stringify(line))).type).toBe("result");
  });
});

describe("repeatDisagreements", () => {
  const of = (repeat: number, ranked: Retrieved[]) => scoreQuestion(question(), ranked, repeat, false);

  test("three identical repeats disagree about nothing", () => {
    const ranked = [span(0, 20, 9), span(40, 60, 8)];
    expect(repeatDisagreements([of(1, ranked), of(2, ranked), of(3, ranked)])).toEqual([]);
  });

  test("a reordered ranking is a disagreement even when the metrics match", () => {
    // Both orderings score recall@5 = 0 and MRR = 0. Comparing metrics would
    // call this deterministic; comparing the ranking is the point.
    const a = [span(0, 20, 5), span(40, 60, 5)];
    const b = [span(40, 60, 5), span(0, 20, 5)];
    expect(repeatDisagreements([of(1, a), of(2, b)])).toEqual(["q"]);
  });

  test("a changed score alone is a disagreement", () => {
    expect(repeatDisagreements([of(1, [span(0, 20, 9)]), of(2, [span(0, 20, 8)])])).toEqual(["q"]);
  });

  test("questions are reported by slug, sorted", () => {
    const one = scoreQuestion(question({ slug: "zebra" }), [span(0, 1, 1)], 1, false);
    const two = scoreQuestion(question({ slug: "zebra" }), [span(0, 1, 2)], 2, false);
    const three = scoreQuestion(question({ slug: "apple" }), [span(0, 1, 1)], 1, false);
    const four = scoreQuestion(question({ slug: "apple" }), [span(0, 1, 2)], 2, false);
    expect(repeatDisagreements([one, two, three, four])).toEqual(["apple", "zebra"]);
  });
});

describe("meanOf", () => {
  test("hand-computed mean over the questions that carry the metric", () => {
    expect(meanOf([1, 0, 0.5])).toBeCloseTo(0.5, 10);
  });

  test("nulls are filtered, not coerced to zero", () => {
    // 1 and 0 average to 0.5. Coercing the two nulls would give 0.25.
    expect(meanOf([1, 0, null, null])).toBe(0.5);
  });

  test("all-null is null, because a mean of no questions is not zero", () => {
    expect(meanOf([null, null])).toBeNull();
    expect(meanOf([])).toBeNull();
  });
});

describe("tally", () => {
  test("counts trues over the questions that carry the metric", () => {
    expect(tally([true, false, true, null])).toEqual({ k: 2, n: 3 });
  });

  test("no measurable questions is 0 of 0, not 0 of 4", () => {
    expect(tally([null, null, null, null])).toEqual({ k: 0, n: 0 });
  });
});

describe("medianThreshold and falsePositives", () => {
  test("the median of five answerable top-1 scores is the middle one", () => {
    // Sorted: 0.1 0.3 0.5 0.7 0.9. Index floor(0.5 * 5) = 2.
    expect(medianThreshold([0.1, 0.5, 0.3, 0.9, 0.7])).toBe(0.5);
  });

  test("hand-computed rate: 2 of 4 negatives at or above the threshold", () => {
    expect(falsePositives([0.9, 0.8, 0.2, 0.1], 0.5)).toBe(0.5);
  });

  test("the threshold is inclusive, matching falsePositiveRate", () => {
    expect(falsePositives([0.5], 0.5)).toBe(1);
  });

  test("an abstention counts in the denominator without inventing a score", () => {
    // The only null in the codebase that is neither filtered nor coerced.
    // Filtering would give 1/1; coercing to 0 gives the right 1/2 but invents
    // a score for a retriever that produced none.
    expect(falsePositives([0.9, null], 0.5)).toBe(0.5);
    expect(falsePositives([null, null], 0.5)).toBe(0);
    expect(falsePositives([0.9, 0.9], 0.5)).toBe(1);
  });

  test("an empty negative set has no rate, distinct from a rate of zero", () => {
    expect(falsePositives([], 0.5)).toBeNull();
  });

  test("no answerable questions means no threshold and no rate", () => {
    expect(medianThreshold([])).toBeNull();
    expect(falsePositives([0.9], null)).toBeNull();
  });
});

describe("discordance", () => {
  const map = (o: Record<string, boolean | null>) => new Map(Object.entries(o));

  test("hand-computed 2x2: 2 both, 1 neither, 1 A only, 1 B only", () => {
    const a = map({ w: true, x: true, y: false, z: false, v: true });
    const b = map({ w: true, x: false, y: true, z: false, v: true });
    expect(discordance(a, b)).toEqual({ both: 2, neither: 1, aOnly: 1, bOnly: 1 });
  });

  test("a question either side could not score is not a disagreement", () => {
    const a = map({ q: true, neg: null });
    const b = map({ q: false, neg: null });
    expect(discordance(a, b)).toEqual({ both: 0, neither: 0, aOnly: 1, bOnly: 0 });
  });

  test("a question one side never ran is skipped rather than counted as a flip", () => {
    expect(discordance(map({ q: true, only_a: true }), map({ q: true })))
      .toEqual({ both: 1, neither: 0, aOnly: 0, bOnly: 0 });
  });

  test("identical configurations produce no discordant pairs", () => {
    const a = map({ w: true, x: false, y: true });
    expect(discordance(a, a)).toEqual({ both: 2, neither: 1, aOnly: 0, bOnly: 0 });
  });
});

describe("foundWithin", () => {
  test("fractional recall becomes a yes/no, because McNemar needs one", () => {
    // A comparative question with two gold spans, one found: recall 0.5. The
    // paired outcome is "found at least one", so true.
    const q = question({
      kind: "comparative",
      gold: [
        { video: "aaaaaaaaaaa", start_s: 0, end_s: 20 },
        { video: "bbbbbbbbbbb", start_s: 0, end_s: 20 },
      ],
    });
    const r = scoreQuestion(q, [span(0, 20, 9)], 1, false);
    expect(r.recall["5"]).toBe(0.5);
    expect(foundWithin(r, 5)).toBe(true);
  });

  test("nothing found is false, and a null recall stays null", () => {
    expect(foundWithin(scoreQuestion(question(), [span(0, 1)], 1, false), 5)).toBe(false);
    expect(foundWithin(scoreQuestion(question({ gold: [] }), [span(0, 1)], 1, null), 5)).toBeNull();
  });

  test("a cut-off the run never recorded is null, not false", () => {
    expect(foundWithin(scoreQuestion(question(), [span(100, 120)], 1, false), 7)).toBeNull();
  });
});

describe("chunkSha", () => {
  test("the same texts in the same order give the same digest", () => {
    expect(chunkSha(["a", "b"])).toBe(chunkSha(["a", "b"]));
  });

  test("reordering changes it", () => {
    expect(chunkSha(["a", "b"])).not.toBe(chunkSha(["b", "a"]));
  });

  test("the separator stops two texts joining into one", () => {
    // Without the delimiter, ["ab"] and ["a","b"] would hash identically.
    expect(chunkSha(["ab"])).not.toBe(chunkSha(["a", "b"]));
  });
});

describe("Header", () => {
  const header: Header = {
    type: "header",
    cell: "fixed-128__bm25__none",
    order: 0,
    chunking: { id: "fixed-128", strategy: "fixed", params: { tokens: 128, overlap: 0 } },
    retrieval: { id: "bm25", kind: "lexical" },
    reranking: { id: "none", kind: "none" },
    embedder: null,
    golden_sha: "abc",
    git_sha: "def",
    chunk_count: 3238,
    questions: 104,
    top_k: 10,
    repeats: 3,
    hit_coverage: 0.5,
    started_at: "2026-09-10T00:00:00.000Z",
  };

  test("round trips through JSON", () => {
    expect(Header.parse(JSON.parse(JSON.stringify(header)))).toEqual(header);
  });

  test("a header missing golden_sha is rejected — invariant 6 turns on it", () => {
    const { golden_sha, ...without } = header;
    expect(() => Header.parse(without)).toThrow();
  });

  test("a header missing git_sha is rejected", () => {
    const { git_sha, ...without } = header;
    expect(() => Header.parse(without)).toThrow();
  });

  test("a header missing its matrix position is rejected", () => {
    // The report sorts rows on it; without one the table falls back to
    // filename order, which puts fixed-1024 before fixed-128.
    const { order, ...without } = header;
    expect(() => Header.parse(without)).toThrow();
  });
});
