import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { CachedLocalEmbedder } from "../src/local-cache.ts";
import { LOCAL_DIM, LOCAL_MODEL } from "../src/local.ts";
import { cosine } from "../src/storage.ts";

const CACHE_FILE = `${LOCAL_MODEL.replace(/\//g, "-")}-${LOCAL_DIM}.json`;

/** The key the embedder uses, recomputed here rather than read off the class. */
const key = (text: string) => createHash("sha256").update(text).digest("hex").slice(0, 24);

/** A distinguishable unit vector, so cosine over two of them is checkable. */
const vector = (at: number): number[] => {
  const v = new Array(LOCAL_DIM).fill(0);
  v[at % LOCAL_DIM] = 1;
  return v;
};

/**
 * A cache directory pre-seeded with vectors for `texts`.
 *
 * Every test here works against a seeded cache on purpose. Letting one call
 * through to the model would download 90MB and run inference, which is the
 * cost this class exists to avoid paying repeatedly — and a unit test that
 * needs a model is not a unit test.
 */
function seeded(texts: string[]) {
  const dir = mkdtempSync(join(tmpdir(), "yt-rag-local-"));
  const cache = Object.fromEntries(texts.map((t, i) => [key(t), vector(i)]));
  writeFileSync(join(dir, CACHE_FILE), JSON.stringify(cache));
  return dir;
}

describe("reading an existing cache", () => {
  test("a fully cached call returns the cached vectors and embeds nothing", async () => {
    const dir = seeded(["alpha", "beta"]);
    const embedder = new CachedLocalEmbedder(dir);

    const out = await embedder.embedAll(["alpha", "beta"]);

    expect(out).toEqual([vector(0), vector(1)]);
    expect(embedder.usage).toEqual({ embedded: 0, cached: 2 });
  });

  test("vectors come back in input order, not cache order", async () => {
    const embedder = new CachedLocalEmbedder(seeded(["alpha", "beta"]));
    expect(await embedder.embedAll(["beta", "alpha"])).toEqual([vector(1), vector(0)]);
  });

  test("a repeated text is served once and returned twice", async () => {
    // What makes the three repeats free: the same query text asked three times
    // is one lookup, not three forward passes.
    const embedder = new CachedLocalEmbedder(seeded(["alpha"]));
    const out = await embedder.embedAll(["alpha", "alpha", "alpha"]);
    expect(out).toEqual([vector(0), vector(0), vector(0)]);
    expect(embedder.usage).toEqual({ embedded: 0, cached: 3 });
  });

  test("the vectors survive the round trip through JSON exactly", async () => {
    // Full precision, not rounded. A repeat served from a rounded cache would
    // score marginally differently from one that recomputed, which would look
    // like non-determinism.
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-local-"));
    const odd = new Array(LOCAL_DIM).fill(0).map((_, i) => (i === 0 ? 0.123456789012345 : 0));
    writeFileSync(join(dir, CACHE_FILE), JSON.stringify({ [key("x")]: odd }));

    const [got] = await new CachedLocalEmbedder(dir).embedAll(["x"]);
    expect(got![0]).toBe(0.123456789012345);
    expect(cosine(got!, odd)).toBeCloseTo(1, 12);
  });

  test("an empty input needs no cache and no model", async () => {
    const embedder = new CachedLocalEmbedder(mkdtempSync(join(tmpdir(), "yt-rag-local-")));
    expect(await embedder.embedAll([])).toEqual([]);
    expect(embedder.usage).toEqual({ embedded: 0, cached: 0 });
  });
});

describe("cachedCount", () => {
  test("counts unique cached texts without embedding anything", async () => {
    // This is what --dry-run calls, so it must never reach the model.
    const embedder = new CachedLocalEmbedder(seeded(["alpha", "beta"]));
    expect(embedder.cachedCount(["alpha", "beta", "gamma"])).toBe(2);
    expect(embedder.usage).toEqual({ embedded: 0, cached: 0 });
  });

  test("a text listed twice counts once", () => {
    const embedder = new CachedLocalEmbedder(seeded(["alpha"]));
    expect(embedder.cachedCount(["alpha", "alpha"])).toBe(1);
  });

  test("an empty cache counts nothing rather than throwing", () => {
    const embedder = new CachedLocalEmbedder(mkdtempSync(join(tmpdir(), "yt-rag-local-")));
    expect(embedder.cachedCount(["alpha"])).toBe(0);
  });
});

describe("the file on disk", () => {
  test("a fresh directory is created and a missing cache is not an error", () => {
    const dir = join(mkdtempSync(join(tmpdir(), "yt-rag-local-")), "nested", "deeper");
    expect(() => new CachedLocalEmbedder(dir)).not.toThrow();
    expect(existsSync(dir)).toBe(true);
  });

  test("flush writes a parseable file and leaves no temporary behind", () => {
    const dir = seeded(["alpha"]);
    const embedder = new CachedLocalEmbedder(dir);
    embedder.flush();

    const path = join(dir, CACHE_FILE);
    expect(Object.keys(JSON.parse(readFileSync(path, "utf8")))).toEqual([key("alpha")]);
    // Written to a temporary name and renamed, so a kill mid-write cannot
    // truncate the file that a two-hour run depends on.
    expect(existsSync(`${path}.tmp`)).toBe(false);
  });

  test("the filename carries the model and the width", () => {
    // A second local model must not read this model's vectors: the vectors are
    // what the name means.
    const dir = seeded([]);
    new CachedLocalEmbedder(dir).flush();
    expect(existsSync(join(dir, CACHE_FILE))).toBe(true);
    expect(CACHE_FILE).toContain(String(LOCAL_DIM));
  });

  test("two embedders over one directory see each other's vectors", async () => {
    // The runner constructs one embedder, but a run that died and was restarted
    // must pick up what the previous one paid for.
    const dir = seeded(["alpha"]);
    new CachedLocalEmbedder(dir).flush();
    expect(await new CachedLocalEmbedder(dir).embedAll(["alpha"])).toEqual([vector(0)]);
  });
});
