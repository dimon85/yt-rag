import { describe, expect, test } from "vitest";
import {
  GEMINI_DIM, GeminiEmbedder, MAX_BATCH, quotaId, quotaKind, retryAfterMs,
} from "../src/gemini.ts";
import { STORAGE_DIM, toStorage } from "../src/storage.ts";

describe("configuration", () => {
  test("the requested width is the storage width, so nothing is padded", () => {
    // The reason chunks.embedding is vector(1536) rather than a model's native
    // size: a hosted model that can be asked for 1536 needs no padding at all.
    expect(GEMINI_DIM).toBe(STORAGE_DIM);
    expect(toStorage(new Array(GEMINI_DIM).fill(0.1), GEMINI_DIM)).toHaveLength(STORAGE_DIM);
  });

  test("a missing key fails at construction, not on the first request", () => {
    // Failing late would mean discovering it after chunking a whole corpus.
    expect(() => new GeminiEmbedder("/tmp/yt-rag-test-cache", "")).toThrow(/GEMINI_API_KEY/);
  });
});

describe("quota handling", () => {
  test("a per-minute quota is waited out, a daily one is not", () => {
    // The distinction the previous harness paid for: five one-minute retries
    // against a daily wall only burn time and leave error rows in the log.
    expect(quotaKind("Quota exceeded for embed_content_free_tier_requests, limit 100")).toBe("minute");
    expect(quotaKind("generativelanguage.googleapis.com/RequestsPerDay")).toBe("day");
    expect(quotaKind("connection reset")).toBe("other");
  });

  test("the wait comes from the error, not from a guess", () => {
    expect(retryAfterMs("Please retry in 29.057856327s.")).toBe(29558);
    expect(retryAfterMs('"retryDelay": "45s"')).toBe(45500);
    expect(retryAfterMs("no hint here")).toBe(30_000);
  });
});

describe("batch limits", () => {
  test("the API ceiling is stated, not discovered at runtime", () => {
    // 250 comes back as "at most 100 requests can be in one batch".
    expect(MAX_BATCH).toBe(100);
  });
});

describe("quotaId", () => {
  test("names the quota from the error, so stopping is checkable", () => {
    expect(quotaId('{"quotaId":"EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier"}'))
      .toMatch(/PerMinute/);
    expect(quotaId("Quota exceeded for metric: generativelanguage.googleapis.com/embed_free_tier"))
      .toBe("generativelanguage.googleapis.com/embed_free_tier");
  });

  test("says so when the error names nothing", () => {
    expect(quotaId("something went wrong")).toBe("unnamed");
  });
});
