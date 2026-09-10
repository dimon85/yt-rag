import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  classifyError, COHERE_MODEL, CohereReranker, MAX_DOCUMENTS, parseRerankResponse,
  rerankKey, retryAfterMs,
} from "../src/cohere.ts";

const CACHE_FILE = `cohere-${COHERE_MODEL}.json`;

const docs = (...texts: string[]) => texts.map((text, id) => ({ id, text }));

/** A cache directory holding one ranking, so nothing here touches the network. */
function seeded(query: string, texts: string[], ranking: { i: number; s: number }[]) {
  const dir = mkdtempSync(join(tmpdir(), "yt-rag-cohere-"));
  writeFileSync(
    join(dir, CACHE_FILE),
    JSON.stringify({ [rerankKey(COHERE_MODEL, query, texts)]: ranking }),
  );
  return dir;
}

describe("classifyError", () => {
  test("a bad key is not retried", () => {
    expect(classifyError(401, "")).toBe("auth");
    expect(classifyError(403, "")).toBe("auth");
  });

  test("a per-minute 429 is waited out and a spent allowance is not", () => {
    // The distinction cost an hour on the embedding side: retrying a finished
    // allowance burns time on something that cannot succeed.
    expect(classifyError(429, "rate limit exceeded")).toBe("minute");
    expect(classifyError(429, "trial key has reached its monthly limit")).toBe("exhausted");
    expect(classifyError(429, "please upgrade your billing plan")).toBe("exhausted");
  });

  test("anything else is neither waited out nor mistaken for a quota", () => {
    expect(classifyError(500, "")).toBe("other");
    expect(classifyError(400, "invalid model")).toBe("other");
  });
});

describe("retryAfterMs", () => {
  test("the header is used when it says something usable", () => {
    expect(retryAfterMs("30")).toBe(30_500);
  });

  test("a missing or nonsense header backs off a minute rather than hammering", () => {
    expect(retryAfterMs(null)).toBe(60_000);
    expect(retryAfterMs("soon")).toBe(60_000);
    expect(retryAfterMs("0")).toBe(60_000);
    expect(retryAfterMs("-5")).toBe(60_000);
  });
});

describe("parseRerankResponse", () => {
  test("reads index and relevance_score", () => {
    const body = { results: [{ index: 2, relevance_score: 0.9 }, { index: 0, relevance_score: 0.1 }] };
    expect(parseRerankResponse(body, 3)).toEqual([{ i: 2, s: 0.9 }, { i: 0, s: 0.1 }]);
  });

  test("a response with no results array is rejected, not read as empty", () => {
    // An empty ranking and a malformed response are different situations, and
    // the second one silently becoming the first is how a cell records a zero
    // that was really a broken call.
    expect(() => parseRerankResponse({}, 3)).toThrow(/no results array/);
    expect(() => parseRerankResponse({ results: null }, 3)).toThrow(/no results array/);
  });

  test("an index outside the document list is rejected", () => {
    expect(() => parseRerankResponse({ results: [{ index: 5, relevance_score: 1 }] }, 3))
      .toThrow(/index 5 for 3 documents/);
    expect(() => parseRerankResponse({ results: [{ index: -1, relevance_score: 1 }] }, 3))
      .toThrow(/index -1/);
  });

  test("a missing or non-numeric score is rejected", () => {
    expect(() => parseRerankResponse({ results: [{ index: 0 }] }, 1)).toThrow(/non-numeric/);
    expect(() => parseRerankResponse({ results: [{ index: 0, relevance_score: "high" }] }, 1))
      .toThrow(/non-numeric/);
  });

  test("an empty ranking is valid", () => {
    expect(parseRerankResponse({ results: [] }, 3)).toEqual([]);
  });
});

describe("rerankKey", () => {
  test("the query is part of the key, because a score is a function of the pair", () => {
    expect(rerankKey("m", "one", ["a"])).not.toBe(rerankKey("m", "two", ["a"]));
  });

  test("the candidate list is part of the key", () => {
    // Keying on the query alone would serve one chunking configuration's
    // ranking to another.
    expect(rerankKey("m", "q", ["a", "b"])).not.toBe(rerankKey("m", "q", ["a", "c"]));
  });

  test("candidate order matters, because the indices are into that order", () => {
    expect(rerankKey("m", "q", ["a", "b"])).not.toBe(rerankKey("m", "q", ["b", "a"]));
  });

  test("the model is part of the key", () => {
    expect(rerankKey("m1", "q", ["a"])).not.toBe(rerankKey("m2", "q", ["a"]));
  });

  test("the separator stops two documents joining into one", () => {
    expect(rerankKey("m", "q", ["ab"])).not.toBe(rerankKey("m", "q", ["a", "b"]));
  });

  test("the same request gives the same key", () => {
    expect(rerankKey("m", "q", ["a", "b"])).toBe(rerankKey("m", "q", ["a", "b"]));
  });
});

describe("reading the cache", () => {
  test("a cached ranking is returned in score order and mapped back onto chunk ids", () => {
    // Document 2 scores highest, then 0, then 1 — so the ids come back 2, 0, 1
    // rather than in request order.
    const dir = seeded("q", ["a", "b", "c"], [{ i: 0, s: 0.5 }, { i: 1, s: 0.1 }, { i: 2, s: 0.9 }]);
    const r = new CohereReranker(dir, "unused-key");

    expect(r.cachedRerank("q", docs("a", "b", "c"), 3)).toEqual([
      { id: 2, score: 0.9 },
      { id: 0, score: 0.5 },
      { id: 1, score: 0.1 },
    ]);
    expect(r.usage.calls).toBe(0);
  });

  test("equal scores break on request position, so two repeats cannot disagree", () => {
    const dir = seeded("q", ["a", "b"], [{ i: 1, s: 0.5 }, { i: 0, s: 0.5 }]);
    expect(new CohereReranker(dir, "k").cachedRerank("q", docs("a", "b"), 2))
      .toEqual([{ id: 0, score: 0.5 }, { id: 1, score: 0.5 }]);
  });

  test("topN truncates the cached ranking rather than needing a second call", () => {
    // The call asks for every candidate, so one cached call serves any top_k
    // the report later wants.
    const dir = seeded("q", ["a", "b", "c"], [{ i: 0, s: 0.9 }, { i: 1, s: 0.5 }, { i: 2, s: 0.1 }]);
    expect(new CohereReranker(dir, "k").cachedRerank("q", docs("a", "b", "c"), 1))
      .toEqual([{ id: 0, score: 0.9 }]);
  });

  test("a miss is null rather than an empty ranking", () => {
    const dir = seeded("q", ["a"], [{ i: 0, s: 1 }]);
    const r = new CohereReranker(dir, "k");
    expect(r.cachedRerank("other query", docs("a"), 1)).toBeNull();
    expect(r.cachedRerank("q", docs("different text"), 1)).toBeNull();
  });

  test("isCached answers without spending and without a key", () => {
    const dir = seeded("q", ["a"], [{ i: 0, s: 1 }]);
    const r = new CohereReranker(dir, undefined);
    expect(r.hasKey).toBe(false);
    expect(r.isCached("q", docs("a"))).toBe(true);
    expect(r.isCached("q", docs("b"))).toBe(false);
  });
});

describe("spending", () => {
  test("a missing key is not a constructor error — --dry-run has to work without one", () => {
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-cohere-"));
    expect(() => new CohereReranker(dir, undefined)).not.toThrow();
    expect(new CohereReranker(dir, undefined).hasKey).toBe(false);
  });

  test("a fresh call without a key fails where it is attempted", async () => {
    const r = new CohereReranker(mkdtempSync(join(tmpdir(), "yt-rag-cohere-")), undefined);
    await expect(r.rerankFresh("q", docs("a"), 1)).rejects.toThrow(/COHERE_API_KEY is not set/);
  });

  test("more documents than the API accepts is refused locally", async () => {
    const r = new CohereReranker(mkdtempSync(join(tmpdir(), "yt-rag-cohere-")), "k");
    const many = docs(...Array.from({ length: MAX_DOCUMENTS + 1 }, (_, i) => `d${i}`));
    await expect(r.rerankFresh("q", many, 10)).rejects.toThrow(/exceeds the API limit/);
  });

  test("no documents needs no call at all", async () => {
    const r = new CohereReranker(mkdtempSync(join(tmpdir(), "yt-rag-cohere-")), "k");
    expect(await r.rerankFresh("q", [], 10)).toEqual([]);
    expect(r.usage.calls).toBe(0);
  });

  test("rerank prefers the cache over a call", async () => {
    // Would throw on a fresh call: there is no key. Returning a ranking proves
    // it never tried.
    const dir = seeded("q", ["a"], [{ i: 0, s: 0.7 }]);
    const r = new CohereReranker(dir, undefined);
    expect(await r.rerank("q", docs("a"), 1)).toEqual([{ id: 0, score: 0.7 }]);
  });
});

describe("the file on disk", () => {
  test("flush writes a parseable file and leaves no temporary behind", () => {
    const dir = seeded("q", ["a"], [{ i: 0, s: 1 }]);
    const r = new CohereReranker(dir, "k");
    r.flush();
    const path = join(dir, CACHE_FILE);
    expect(Object.keys(JSON.parse(readFileSync(path, "utf8")))).toHaveLength(1);
    expect(existsSync(`${path}.tmp`)).toBe(false);
  });

  test("the filename carries the model, so two rerankers cannot share rankings", () => {
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-cohere-"));
    new CohereReranker(dir, "k", "rerank-v3.5").flush();
    new CohereReranker(dir, "k", "rerank-english-v3.0").flush();
    expect(existsSync(join(dir, "cohere-rerank-v3.5.json"))).toBe(true);
    expect(existsSync(join(dir, "cohere-rerank-english-v3.0.json"))).toBe(true);
  });

  test("a missing cache file is not an error", () => {
    const dir = join(mkdtempSync(join(tmpdir(), "yt-rag-cohere-")), "nested");
    expect(() => new CohereReranker(dir, "k")).not.toThrow();
    expect(existsSync(dir)).toBe(true);
  });
});
