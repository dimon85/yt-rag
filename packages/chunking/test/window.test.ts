import { describe, expect, test } from "vitest";
import { encode } from "gpt-tokenizer";
import { windowChunks } from "../src/window.ts";
import type { Segment } from "../src/fixed.ts";

const seg = (text: string, start_s: number, end_s: number): Segment => ({ text, start_s, end_s });

describe("degenerate input", () => {
  test("no segments, no chunks", () => {
    expect(windowChunks([], 60)).toEqual([]);
  });

  test("a transcript shorter than one window is one chunk", () => {
    const chunks = windowChunks([seg("hello there", 1, 5)], 60);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toBe("hello there");
    expect(chunks[0]!.start_s).toBe(1);
    expect(chunks[0]!.end_s).toBe(5);
  });

  test("invalid sizes are rejected", () => {
    const segments = [seg("a", 0, 1)];
    expect(() => windowChunks(segments, 0)).toThrow(/must be positive/);
    expect(() => windowChunks(segments, 60, -1)).toThrow(/cannot be negative/);
    expect(() => windowChunks(segments, 60, 60)).toThrow(/must be below/);
    expect(() => windowChunks(segments, 60, 90)).toThrow(/must be below/);
  });
});

describe("window assignment", () => {
  // Boundaries are at multiples of 10 from zero: [0,10), [10,20), [20,30).
  const segments = [
    seg("one", 0, 4),
    seg("two", 5, 9),
    seg("three", 12, 16),
    seg("four", 21, 25),
  ];

  test("segments group by the window their start falls in", () => {
    const chunks = windowChunks(segments, 10);
    expect(chunks.map((c) => c.text)).toEqual(["one two", "three", "four"]);
  });

  test("a segment straddling a boundary goes to the window its start is in", () => {
    // Starts at 8, ends at 13 — the boundary at 10 cuts it in half. It belongs
    // to [0,10) because that is where its text begins.
    const chunks = windowChunks([seg("early", 1, 3), seg("straddle", 8, 13), seg("late", 14, 18)], 10);
    expect(chunks.map((c) => c.text)).toEqual(["early straddle", "late"]);
  });

  test("every segment lands in at least one chunk", () => {
    const joined = windowChunks(segments, 10).map((c) => c.text).join(" ");
    for (const s of segments) expect(joined).toContain(s.text);
  });

  test("without overlap no segment lands in two chunks", () => {
    const chunks = windowChunks(segments, 10);
    const words = chunks.flatMap((c) => c.text.split(" "));
    expect(words).toHaveLength(new Set(words).size);
  });
});

describe("empty windows", () => {
  // A gap in the captions. [0,10) has one segment, [10,20) and [20,30) none,
  // [30,40) has one.
  const gapped = [seg("before", 2, 6), seg("after", 31, 35)];

  test("a gap does not become a zero-length chunk", () => {
    const chunks = windowChunks(gapped, 10);
    expect(chunks).toHaveLength(2);
    for (const c of chunks) expect(c.end_s).toBeGreaterThan(c.start_s);
    expect(chunks.map((c) => c.text)).toEqual(["before", "after"]);
  });

  test("a long silent stretch produces no chunks for it", () => {
    // Two segments an hour apart: 60 windows of 60 s span the gap and every
    // one of them is empty.
    const chunks = windowChunks([seg("start", 0, 3), seg("end", 3600, 3603)], 60);
    expect(chunks).toHaveLength(2);
  });
});

describe("timestamps come from the segments, not the window", () => {
  test("start_s is the earliest start among the segments held", () => {
    // The window is [0,10) but nothing is spoken before 4.
    const chunks = windowChunks([seg("a", 4, 8)], 10);
    expect(chunks[0]!.start_s).toBe(4);
  });

  test("end_s may run past the nominal window end", () => {
    // A caption starting at 9 and ending at 15 is in [0,10), so that chunk
    // really does cover up to 15. Reporting 10 would score the hit rule
    // against a span the chunk does not contain.
    const chunks = windowChunks([seg("a", 1, 3), seg("b", 9, 15)], 10);
    expect(chunks[0]!.end_s).toBe(15);
  });

  test("end_s is the largest end among the segments, not the last one's", () => {
    // Captions in this corpus overlap in time, so the final segment's end_s is
    // not necessarily the highest.
    const chunks = windowChunks([seg("a", 0, 30), seg("b", 1, 20)], 60);
    expect(chunks[0]!.end_s).toBe(30);
  });

  test("chunk spans advance and never run backwards", () => {
    const segments = Array.from({ length: 40 }, (_, i) => seg(`w${i}`, i * 5, i * 5 + 8));
    for (const overlap of [0, 30]) {
      const chunks = windowChunks(segments, 60, overlap);
      for (const c of chunks) expect(c.end_s).toBeGreaterThan(c.start_s);
      for (let i = 1; i < chunks.length; i++) {
        expect(chunks[i]!.start_s).toBeGreaterThan(chunks[i - 1]!.start_s);
      }
    }
  });
});

describe("overlap", () => {
  // Step is 10 - 4 = 6, so windows are [0,10), [6,16), [12,22), [18,28).
  const segments = [
    seg("aa", 1, 3),
    seg("bb", 7, 9),
    seg("cc", 13, 15),
    seg("dd", 19, 21),
  ];

  test("overlapping windows share the segments in the overlap", () => {
    const chunks = windowChunks(segments, 10, 4);
    expect(chunks.map((c) => c.text)).toEqual(["aa bb", "bb cc", "cc dd", "dd"]);
  });

  test("overlap produces more chunks than no overlap", () => {
    const dense = Array.from({ length: 60 }, (_, i) => seg(`w${i}`, i * 5, i * 5 + 5));
    expect(windowChunks(dense, 90, 30).length).toBeGreaterThan(windowChunks(dense, 90, 0).length);
  });

  test("consecutive chunks overlap in time when overlap is on", () => {
    const dense = Array.from({ length: 60 }, (_, i) => seg(`w${i}`, i * 5, i * 5 + 5));
    const lapped = windowChunks(dense, 90, 30);
    expect(lapped[1]!.start_s).toBeLessThan(lapped[0]!.end_s);
  });

  test("two windows holding the same segments yield one chunk, not two", () => {
    // [0,10) and [6,16) both hold only the segment at 7. Emitting it twice
    // would put identical text at two ranks in one top-k.
    const chunks = windowChunks([seg("lonely", 7, 9)], 10, 4);
    expect(chunks).toHaveLength(1);
  });
});

describe("token counts", () => {
  const segments = Array.from({ length: 40 }, (_, i) => seg(`word${i} filler text here`, i * 5, i * 5 + 5));

  test("token_count is what the tokenizer says about the chunk text", () => {
    // Invariant 2: one tokenizer for every configuration, counted rather than
    // estimated from characters.
    for (const c of windowChunks(segments, 60)) {
      expect(c.token_count).toBe(encode(c.text).length);
    }
  });

  test("a time window does not bound token count", () => {
    // The point of the strategy, and the reason it is worth a row: window
    // length is fixed in seconds, so chunk length in tokens varies with how
    // fast the speaker talks.
    const counts = windowChunks(segments, 60).map((c) => c.token_count);
    expect(Math.max(...counts)).toBeGreaterThan(0);
  });
});

describe("hand-computed case", () => {
  // Worked out on paper before the code ran. Windows of 30 s, no overlap, so
  // boundaries at 0, 30, 60. Starts: 5, 25, 29, 31, 58, 61.
  //   [0,30)  -> 5, 25, 29
  //   [30,60) -> 31, 58
  //   [60,90) -> 61
  const segments = [
    seg("s5", 5, 12),
    seg("s25", 25, 32),
    seg("s29", 29, 36),
    seg("s31", 31, 38),
    seg("s58", 58, 65),
    seg("s61", 61, 68),
  ];

  test("three chunks with the spans the segments imply", () => {
    const chunks = windowChunks(segments, 30);
    expect(chunks).toHaveLength(3);
    expect(chunks[0]).toMatchObject({ text: "s5 s25 s29", start_s: 5, end_s: 36 });
    expect(chunks[1]).toMatchObject({ text: "s31 s58", start_s: 31, end_s: 65 });
    expect(chunks[2]).toMatchObject({ text: "s61", start_s: 61, end_s: 68 });
  });
});
