import { describe, expect, test } from "vitest";
import { join } from "node:path";
import { ROOT } from "../../ingest/src/corpus.ts";
import {
  embeddingRate, loadPrices, Prices, rerankRate, unpricedReason,
} from "../src/prices.ts";

const parse = (over: Record<string, unknown> = {}) =>
  Prices.parse({
    version: 1,
    embedding: { local: { usd_per_million_tokens: 0, as_of: "2026-09-10", source: "cpu" } },
    rerank: { none: { usd_per_1000_searches: 0, as_of: "2026-09-10", source: "local" } },
    ...over,
  });

describe("a price needs provenance or an explicit reason", () => {
  test("a number with a source and a date is accepted", () => {
    expect(() => parse()).not.toThrow();
  });

  test("a bare number is rejected — that is how a guess reaches a README", () => {
    expect(() => parse({ embedding: { gemini: { usd_per_million_tokens: 0.15 } } })).toThrow();
  });

  test("a number with a source but no date is rejected", () => {
    expect(() =>
      parse({ embedding: { gemini: { usd_per_million_tokens: 0.15, source: "the pricing page" } } })
    ).toThrow();
  });

  test("null needs a reason, and with one it is accepted", () => {
    expect(() => parse({ embedding: { gemini: { usd_per_million_tokens: null } } })).toThrow();
    expect(() =>
      parse({ embedding: { gemini: { usd_per_million_tokens: null, reason: "free tier" } } })
    ).not.toThrow();
  });

  test("a typo'd key is rejected rather than ignored", () => {
    // `sources` would leave a priced number with no provenance that reads as
    // though it had some.
    expect(() =>
      parse({ embedding: { gemini: { usd_per_million_tokens: 0.15, as_of: "2026-09-10", sources: "x" } } })
    ).toThrow();
  });

  test("a null price cannot smuggle in a source instead of a reason", () => {
    expect(() =>
      parse({ embedding: { gemini: { usd_per_million_tokens: null, as_of: "2026-09-10", source: "x" } } })
    ).toThrow();
  });
});

describe("rates", () => {
  const prices = parse({
    embedding: {
      local: { usd_per_million_tokens: 0, as_of: "2026-09-10", source: "cpu" },
      gemini: { usd_per_million_tokens: null, reason: "free tier, counted in texts" },
    },
    rerank: { "cohere-rerank": { usd_per_1000_searches: null, reason: "never billed" } },
  });

  test("a lexical cell embeds nothing, which is free rather than unknown", () => {
    expect(embeddingRate(prices, null)).toBe(0);
  });

  test("an unpriced embedder is null, not zero", () => {
    expect(embeddingRate(prices, "gemini")).toBeNull();
  });

  test("an embedder the file does not mention is null too", () => {
    expect(embeddingRate(prices, "some-other-model")).toBeNull();
  });

  test("none and mmr rerank locally, so they cost nothing", () => {
    expect(rerankRate(prices, "none")).toBe(0);
    expect(rerankRate(prices, "mmr")).toBe(0);
  });

  test("the reason travels with the null, in the file's own words", () => {
    expect(unpricedReason(prices, "gemini")).toMatch(/free tier/);
    expect(unpricedReason(prices, "cohere-rerank")).toMatch(/never billed/);
  });

  test("a priced line has no reason to report", () => {
    expect(unpricedReason(prices, "local")).toBeNull();
  });

  test("a vendor absent from the file says so rather than returning null", () => {
    expect(unpricedReason(prices, "nobody")).toMatch(/no entry/);
  });
});

describe("the repository's own prices.yaml", () => {
  // It is read by pnpm report on every invocation, so a schema violation here
  // breaks the report rather than one test.
  const prices = loadPrices(join(ROOT, "prices.yaml"));

  test("parses", () => {
    expect(prices.version).toBe(1);
  });

  test("the local model is priced at zero, with its reasoning", () => {
    expect(embeddingRate(prices, "local")).toBe(0);
  });

  test("gemini and cohere are unpriced, and say why", () => {
    expect(embeddingRate(prices, "gemini")).toBeNull();
    expect(rerankRate(prices, "cohere-rerank")).toBeNull();
    expect(unpricedReason(prices, "gemini")).toMatch(/free tier/i);
    expect(unpricedReason(prices, "cohere-rerank")).toMatch(/never billed/i);
  });
});
