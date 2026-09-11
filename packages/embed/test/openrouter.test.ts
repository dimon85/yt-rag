import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  classifyError, DEFAULT_MODEL, OpenRouterEmbedder, parseEmbeddings,
} from "../src/openrouter.ts";
import { STORAGE_DIM } from "../src/storage.ts";

const dir = () => mkdtempSync(join(tmpdir(), "yt-rag-or-"));
const body = (vectors: number[][]) => ({
  data: vectors.map((embedding, index) => ({ index, embedding })),
  usage: { prompt_tokens: 10, cost: 0.0001 },
});

describe("parseEmbeddings", () => {
  test("returns one vector per input, unit length", () => {
    const [v] = parseEmbeddings(body([[3, 4]]), 1);
    expect(v).toEqual([0.6, 0.8]);
  });

  test("orders by the response's index, not by position", () => {
    // A client that trusts position and is wrong attaches plausible vectors to
    // the wrong texts, which nothing downstream can detect.
    const shuffled = {
      data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }],
      usage: {},
    };
    expect(parseEmbeddings(shuffled, 2)).toEqual([[1, 0], [0, 1]]);
  });

  test("a short data array is an error, not a partial result", () => {
    expect(() => parseEmbeddings(body([[1, 0]]), 2)).toThrow(/asked for 2/);
  });

  test("a missing data array is an error", () => {
    expect(() => parseEmbeddings({ usage: {} }, 1)).toThrow(/no data array/);
  });

  test("an index outside the request is rejected", () => {
    const bad = { data: [{ index: 7, embedding: [1, 0] }], usage: {} };
    expect(() => parseEmbeddings(bad, 1)).toThrow(/index 7/);
  });

  test("a repeated index is rejected rather than overwriting", () => {
    const bad = { data: [{ index: 0, embedding: [1, 0] }, { index: 0, embedding: [0, 1] }], usage: {} };
    expect(() => parseEmbeddings(bad, 2)).toThrow(/repeats index 0/);
  });

  test("a non-numeric or empty embedding is rejected", () => {
    expect(() => parseEmbeddings({ data: [{ index: 0, embedding: [] }], usage: {} }, 1))
      .toThrow(/finite number array/);
    expect(() => parseEmbeddings({ data: [{ index: 0, embedding: ["a"] }], usage: {} }, 1))
      .toThrow(/finite number array/);
    expect(() => parseEmbeddings({ data: [{ index: 0, embedding: [NaN] }], usage: {} }, 1))
      .toThrow(/finite number array/);
  });

  test("a vector wider than the storage width fails here, not at toStorage", () => {
    const wide = [{ index: 0, embedding: new Array(STORAGE_DIM + 1).fill(1) }];
    expect(() => parseEmbeddings({ data: wide, usage: {} }, 1)).toThrow(/storage width/);
  });

  test("a zero vector stays zero rather than becoming NaN", () => {
    expect(parseEmbeddings({ data: [{ index: 0, embedding: [0, 0] }], usage: {} }, 1))
      .toEqual([[0, 0]]);
  });
});

describe("classifyError", () => {
  test("a bad key is not retried", () => {
    expect(classifyError(401, "")).toBe("auth");
    expect(classifyError(403, "")).toBe("auth");
  });

  test("402 is spent credits, which waiting cannot fix", () => {
    expect(classifyError(402, "")).toBe("credits");
  });

  test("429 splits on the body, because it carries both meanings", () => {
    expect(classifyError(429, "rate limit exceeded")).toBe("rate");
    expect(classifyError(429, "insufficient credits")).toBe("credits");
    expect(classifyError(429, "daily quota reached")).toBe("credits");
  });

  test("5xx is transient and worth a retry", () => {
    expect(classifyError(500, "")).toBe("rate");
    expect(classifyError(503, "")).toBe("rate");
  });

  test("anything else is not retried", () => {
    expect(classifyError(400, "JSON parsing failed")).toBe("other");
  });
});

describe("the cache", () => {
  test("a missing key is not a constructor error — --dry-run needs no key", () => {
    const e = new OpenRouterEmbedder(dir(), undefined);
    expect(e.hasKey).toBe(false);
    expect(e.cachedCount(["anything"])).toBe(0);
  });

  test("keys on the model as well as the text", () => {
    // The same text under two models must not share a vector: one is 1,024
    // values and the other 3,072, and a collision would serve the wrong one.
    // The filename separates them too, so this is the second of two locks —
    // and the one that still holds if the files are ever merged.
    const d = dir();
    const a = new OpenRouterEmbedder(d, "k", "model-a");
    const b = new OpenRouterEmbedder(d, "k", "model-b");
    // @ts-expect-error — reaching for the private key is the point of the test
    expect(a.key("same text")).not.toBe(b.key("same text"));
    // @ts-expect-error — and the same model must agree with itself
    expect(a.key("same text")).toBe(new OpenRouterEmbedder(d, "k", "model-a").key("same text"));
  });

  test("a vector cached under one model is invisible to another", async () => {
    const d = dir();
    const a = new OpenRouterEmbedder(d, "k", "model-a");
    // @ts-expect-error — as above
    a.cache[a.key("shared")] = [1, 0];
    a.flush();
    // Keyless on purpose: a cache miss must fail before the request, so this
    // test never touches the network.
    const b = new OpenRouterEmbedder(d, undefined, "model-b");
    expect(b.cachedCount(["shared"])).toBe(0);
    await expect(b.embedAll(["shared"])).rejects.toThrow(/OPENROUTER_API_KEY is not set/);
  });

  test("a cache written by one instance is read by the next", async () => {
    const d = dir();
    const first = new OpenRouterEmbedder(d, "k");
    // @ts-expect-error — reaching into the cache is the point of the test
    first.cache[first.key("hello")] = [1, 0];
    first.flush();

    const second = new OpenRouterEmbedder(d, "k");
    expect(second.cachedCount(["hello"])).toBe(1);
    expect(await second.embedAll(["hello"])).toEqual([[1, 0]]);
    expect(second.usage.calls).toBe(0);
    expect(second.usage.cached).toBe(1);
  });

  test("duplicates in one call are embedded once and returned twice", async () => {
    const d = dir();
    const e = new OpenRouterEmbedder(d, "k");
    // @ts-expect-error — as above
    e.cache[e.key("same")] = [0, 1];
    expect(await e.embedAll(["same", "same"])).toEqual([[0, 1], [0, 1]]);
    expect(e.usage.calls).toBe(0);
  });

  test("embedding without a key fails at the request, not at construction", async () => {
    const e = new OpenRouterEmbedder(dir(), undefined);
    await expect(e.embedAll(["uncached"])).rejects.toThrow(/OPENROUTER_API_KEY is not set/);
  });

  test("the default model is the one prices.yaml carries a measured price for", () => {
    expect(DEFAULT_MODEL).toBe("baai/bge-m3");
  });
});
