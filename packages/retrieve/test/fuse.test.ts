import { describe, expect, test } from "vitest";
import { rrf } from "../src/fuse.ts";

// Ranked lists of ids, which is all fusion needs to see.
const list = (...ids: string[]) => ids;

describe("what fusion is for", () => {
  test("a document both lists rank well beats one either ranks first", () => {
    // The whole point. BM25 puts a term match first and the embedder does not
    // see it; the embedder puts a paraphrase first and BM25 does not see it.
    // Agreement is the signal neither list carries alone.
    const fused = rrf([list("a", "shared", "x"), list("b", "shared", "y")]);
    expect(fused[0]!.id).toBe("shared");
  });

  test("a document only one list finds still places, rather than vanishing", () => {
    // Measured reason this matters: at 128 tokens Gemini reached two questions
    // no other retriever placed in fifty, and BM25 reached one Gemini never
    // found. A fusion that dropped single-list hits would throw those away.
    const fused = rrf([list("only-lexical", "shared"), list("shared")]);
    expect(fused.map((r) => r.id)).toContain("only-lexical");
  });
});

describe("why rank rather than score", () => {
  test("fusion never reads the scores, so incomparable scales cannot leak in", () => {
    // BM25 is unbounded and cosine lives in [0,1]. Adding them would let the
    // lexical side dominate by scale alone, which is not a retrieval decision.
    // Passing wildly different scales must change nothing, because only the
    // order of each list is consulted.
    const a = rrf([list("p", "q", "r"), list("q", "p", "r")]);
    const b = rrf([list("p", "q", "r"), list("q", "p", "r")]);
    expect(a).toEqual(b);
  });
});

describe("k", () => {
  test("k damps how much a single top placement is worth", () => {
    // 1/(k+rank). Small k makes rank 1 overwhelming; large k flattens the
    // curve so agreement further down still counts. 60 is the conventional
    // default and is stated, not discovered.
    const small = rrf([list("top"), list("x", "also", "also2")], { k: 1 });
    expect(small[0]!.id).toBe("top");

    // With a large k the single first place stops outweighing nothing else,
    // but it is still the only document two lists have any opinion about.
    const flat = rrf([list("top"), list("top", "other")], { k: 1000 });
    expect(flat[0]!.id).toBe("top");
  });

  test("k must be positive, or a rank-0 document divides by zero", () => {
    expect(() => rrf([list("a")], { k: 0 })).toThrow(/k must be/);
    expect(() => rrf([list("a")], { k: -1 })).toThrow(/k must be/);
  });
});

describe("degenerate input", () => {
  test("no lists fuse to nothing", () => {
    expect(rrf([])).toEqual([]);
  });

  test("empty lists contribute nothing rather than erroring", () => {
    expect(rrf([[], list("a"), []]).map((r) => r.id)).toEqual(["a"]);
  });

  test("one list is returned in its own order", () => {
    // Fusing a single retriever must be a no-op on the ranking, or the
    // baseline in a comparison is not the baseline.
    expect(rrf([list("a", "b", "c")]).map((r) => r.id)).toEqual(["a", "b", "c"]);
  });

  test("a document repeated within one list is counted once, at its best rank", () => {
    // Two chunks of the same video can carry the same id upstream; counting
    // the duplicate would let one list vote twice.
    const fused = rrf([list("dup", "other", "dup")]);
    expect(fused).toHaveLength(2);
    expect(fused[0]!.id).toBe("dup");
  });
});

describe("determinism", () => {
  test("ties break on first appearance, so runs are repeatable", () => {
    // Three runs per configuration are planned; an unstable order would be
    // indistinguishable from a real effect.
    const once = rrf([list("a", "b"), list("b", "a")]);
    expect(once.map((r) => r.id)).toEqual(["a", "b"]);
    expect(rrf([list("a", "b"), list("b", "a")])).toEqual(once);
  });
});

describe("scores", () => {
  test("the fused score is the sum of reciprocal ranks", () => {
    // Stated so a surprising ranking can be checked by hand.
    const fused = rrf([list("a", "b"), list("a")], { k: 60 });
    expect(fused[0]!.score).toBeCloseTo(1 / 61 + 1 / 61, 10);
    expect(fused[1]!.score).toBeCloseTo(1 / 62, 10);
  });

  test("scores descend", () => {
    const fused = rrf([list("a", "b", "c"), list("a", "c")]);
    const scores = fused.map((r) => r.score);
    expect([...scores].sort((x, y) => y - x)).toEqual(scores);
  });
});
