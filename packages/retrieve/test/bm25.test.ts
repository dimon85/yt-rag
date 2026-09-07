import { describe, expect, test } from "vitest";
import { buildIndex, search, tokenize } from "../src/bm25.ts";

const docs = [
  { id: 1, text: "the rate limit for the paid plan was doubled" },
  { id: 2, text: "context window compaction happens at the threshold" },
  { id: 3, text: "the rate limit is a weekly cap, not an hourly one" },
  { id: 4, text: "nothing relevant here at all whatsoever" },
];

describe("tokenize", () => {
  test("lowercases and strips punctuation", () => {
    expect(tokenize("Rate-limit, DOUBLED!")).toEqual(["rate", "limit", "doubled"]);
  });

  test("drops single characters, which carry no signal and inflate length", () => {
    expect(tokenize("a b ok")).toEqual(["ok"]);
  });
});

describe("search", () => {
  const index = buildIndex(docs);

  test("finds documents sharing query terms, best first", () => {
    const hits = search(index, "rate limit");
    expect(hits.map((h) => h.id).slice(0, 2).sort()).toEqual([1, 3]);
  });

  test("a document sharing nothing is not returned at all", () => {
    expect(search(index, "rate limit").map((h) => h.id)).not.toContain(4);
  });

  test("a query matching nothing returns nothing rather than everything", () => {
    expect(search(index, "kubernetes helm chart")).toEqual([]);
  });

  test("topK truncates", () => {
    expect(search(index, "the rate limit", 1)).toHaveLength(1);
  });

  test("an empty index is not a crash", () => {
    expect(search(buildIndex([]), "anything")).toEqual([]);
  });

  test("results are stable across identical calls", () => {
    // Three runs per configuration are planned; retrieval disagreeing with
    // itself would be indistinguishable from a real effect.
    expect(search(index, "the rate limit")).toEqual(search(index, "the rate limit"));
  });

  test("a term in most documents does not push its holders down", () => {
    // Without the IDF floor, "the" scores negative here and documents
    // containing it rank below documents that do not.
    const common = search(index, "the");
    expect(common.every((h) => h.score >= 0)).toBe(true);
  });

  test("term frequency saturates rather than growing without bound", () => {
    const repeated = buildIndex([
      { id: 1, text: "limit" },
      { id: 2, text: "limit limit limit limit limit limit limit limit" },
    ]);
    const [top, second] = search(repeated, "limit");
    // Eight occurrences are worth more than one, but nowhere near eight times.
    expect(top!.score).toBeLessThan(second!.score * 3);
  });
});
