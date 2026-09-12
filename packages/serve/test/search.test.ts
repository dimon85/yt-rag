import { describe, expect, test } from "vitest";
import { packVectors, ServeIndex, unpackVectors } from "../src/index-format.ts";
import { EXCERPT_CHARS, excerpt, prepare, retrieve, sourceUrl } from "../src/search.ts";

const index = (over: Partial<ServeIndex> = {}): ServeIndex => ({
  config: {
    chunking: "fixed-512-ov128", retrieval: "hybrid", embedder: "local",
    git_sha: "abc123", built_at: "2026-09-12T00:00:00.000Z",
  },
  dim: 2,
  threshold: 0.03,
  chunks: [
    { video: "aaaaaaaaaaa", start_s: 10.4, end_s: 70, text: "claude code runs subagents in parallel" },
    { video: "bbbbbbbbbbb", start_s: 100, end_s: 160, text: "cursor composer edits several files" },
  ],
  videos: {
    aaaaaaaaaaa: { title: "A talk", channel: "Someone", published_at: "2026-01-01" },
  },
  vectors: packVectors([[1, 0], [0, 1]]),
  ...over,
});

describe("vector packing", () => {
  test("round-trips the values a model returns", () => {
    const v = [[0.5, -0.25, 0.125], [1, 0, 0]];
    const back = unpackVectors(packVectors(v), 3);
    expect([...back[0]!]).toEqual([0.5, -0.25, 0.125]);
    expect([...back[1]!]).toEqual([1, 0, 0]);
  });

  test("float32 is the format, and the loss is bounded", () => {
    // Checked against the real index before this format was chosen: the top-10
    // is identical to the float64 original on all 104 golden questions.
    const [v] = unpackVectors(packVectors([[0.1]]), 1);
    expect(v![0]).toBeCloseTo(0.1, 6);
    expect(v![0]).not.toBe(0.1);
  });
});

describe("excerpt", () => {
  test("short text is returned whole", () => {
    expect(excerpt("a short line")).toBe("a short line");
  });

  test("long text is cut, and says so", () => {
    const long = "word ".repeat(200);
    const out = excerpt(long);
    expect(out.length).toBeLessThanOrEqual(EXCERPT_CHARS + 1);
    expect(out.endsWith("…")).toBe(true);
  });

  test("the cut lands on a word boundary", () => {
    // Three words; the limit falls inside the third, so the third goes
    // entirely rather than appearing as a fragment.
    const out = excerpt(`${"x".repeat(100)} ${"y".repeat(100)} ${"z".repeat(200)}`, 210);
    expect(out).not.toContain("z");
    expect(out).toBe(`${"x".repeat(100)} ${"y".repeat(100)}…`);
  });

  test("a single long word is cut rather than returned whole", () => {
    // No boundary to find; returning the paragraph because it has no spaces
    // would defeat the limit that exists for legal reasons.
    const out = excerpt("z".repeat(500), 100);
    expect(out.length).toBeLessThanOrEqual(101);
  });
});

describe("sourceUrl", () => {
  test("starts the video at the span", () => {
    expect(sourceUrl("aaaaaaaaaaa", 10.4)).toBe("https://www.youtube.com/watch?v=aaaaaaaaaaa&t=10s");
  });

  test("seconds are whole — YouTube ignores fractions", () => {
    expect(sourceUrl("x", 59.999)).toMatch(/t=59s$/);
  });
});

describe("retrieve", () => {
  const ready = prepare(index());

  test("returns spans with a citation, never generated prose", () => {
    const answer = retrieve(ready, [1, 0], "claude code subagents", 2);
    expect(answer.answered).toBe(true);
    if (!answer.answered) return;
    const hit = answer.hits[0]!;
    expect(hit.video).toBe("aaaaaaaaaaa");
    expect(hit.url).toContain("youtube.com/watch?v=aaaaaaaaaaa");
    expect(hit.title).toBe("A talk");
    expect(hit).not.toHaveProperty("answer");
  });

  test("a video without metadata still cites the link", () => {
    // One video in this corpus has no usable metadata file. The timestamp and
    // the link are the citation the spec requires; a title is a courtesy.
    const answer = retrieve(ready, [0, 1], "cursor composer", 1);
    expect(answer.answered).toBe(true);
    if (!answer.answered) return;
    expect(answer.hits[0]!.title).toBe("");
    expect(answer.hits[0]!.url).toContain("bbbbbbbbbbb");
  });

  test("below the threshold it declines, and says what it saw", () => {
    // A plain top-k retriever always returns k results, so without this the
    // false-positive rate is 1 by construction — and 32 golden questions have
    // no answer in the corpus on purpose.
    const strict = prepare(index({ threshold: 0.9 }));
    const answer = retrieve(strict, [1, 0], "anything", 2);
    expect(answer.answered).toBe(false);
    if (answer.answered) return;
    expect(answer.best).toBeLessThan(0.9);
    expect(answer.threshold).toBe(0.9);
  });

  test("the excerpt is never the whole chunk when the chunk is long", () => {
    const long = index({
      chunks: [{ video: "c", start_s: 0, end_s: 60, text: "claude ".repeat(200) }],
      vectors: packVectors([[1, 0]]),
      threshold: 0,
    });
    const answer = retrieve(prepare(long), [1, 0], "claude", 1);
    if (!answer.answered) throw new Error("expected an answer");
    expect(answer.hits[0]!.excerpt.length).toBeLessThan(long.chunks[0]!.text.length);
  });

  test("k is honoured", () => {
    const answer = retrieve(prepare(index({ threshold: 0 })), [1, 0], "claude cursor", 1);
    if (!answer.answered) throw new Error("expected an answer");
    expect(answer.hits).toHaveLength(1);
  });
});
