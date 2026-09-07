import { describe, expect, test } from "vitest";
import { encode } from "gpt-tokenizer";
import { fixedChunks, type Segment } from "../src/fixed.ts";

const seg = (text: string, start_s: number, end_s: number): Segment => ({ text, start_s, end_s });

/** Ten tokens of filler, so sizes in these tests are predictable. */
const filler = (n: number) => Array.from({ length: n }, (_, i) => `word${i}`).join(" ");

describe("degenerate input", () => {
  test("no segments, no chunks", () => {
    expect(fixedChunks([])).toEqual([]);
  });

  test("a transcript shorter than the target is one chunk", () => {
    const chunks = fixedChunks([seg("hello there", 0, 5)], 512);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toBe("hello there");
  });

  test("a single segment longer than the target still yields one chunk", () => {
    // Boundaries land between segments, so an oversized caption cannot be
    // split. Taking at least one segment is what stops the loop stalling.
    const chunks = fixedChunks([seg(filler(200), 0, 60)], 10);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.token_count).toBeGreaterThan(10);
  });
});

describe("sizes", () => {
  const segments = Array.from({ length: 30 }, (_, i) => seg(filler(10), i * 5, i * 5 + 5));

  test("no chunk exceeds the target once more than one segment fits", () => {
    for (const c of fixedChunks(segments, 100)) {
      expect(c.token_count).toBeLessThanOrEqual(100);
    }
  });

  test("token_count matches what the tokenizer says about the text", () => {
    for (const c of fixedChunks(segments, 100)) {
      // Rebuilt from the joined text, so this catches an off-by-one in the
      // per-segment accounting rather than trusting the running total.
      expect(Math.abs(encode(c.text).length - c.token_count)).toBeLessThanOrEqual(1);
    }
  });

  test("every segment lands in at least one chunk", () => {
    const joined = fixedChunks(segments, 100).map((c) => c.text).join(" ");
    for (const s of segments) expect(joined).toContain(s.text);
  });
});

describe("overlap", () => {
  const segments = Array.from({ length: 40 }, (_, i) => seg(filler(10), i * 5, i * 5 + 5));

  test("overlap produces more chunks than no overlap", () => {
    const plain = fixedChunks(segments, 100, 0);
    const lapped = fixedChunks(segments, 100, 25);
    expect(lapped.length).toBeGreaterThan(plain.length);
  });

  test("consecutive chunks overlap in time when overlap is on", () => {
    const lapped = fixedChunks(segments, 100, 25);
    expect(lapped[1]!.start_s).toBeLessThan(lapped[0]!.end_s);
  });

  test("no overlap means chunks do not share text", () => {
    const plain = fixedChunks(segments, 100, 0);
    const total = plain.reduce((n, c) => n + c.token_count, 0);
    const whole = encode(segments.map((s) => s.text).join(" ")).length;
    expect(Math.abs(total - whole)).toBeLessThanOrEqual(plain.length);
  });

  test("overlap at or above the target is rejected rather than looping forever", () => {
    expect(() => fixedChunks(segments, 100, 100)).toThrow(/must be below/);
    expect(() => fixedChunks(segments, 100, 200)).toThrow(/must be below/);
  });

  test("invalid sizes are rejected", () => {
    expect(() => fixedChunks(segments, 0)).toThrow(/must be positive/);
    expect(() => fixedChunks(segments, 100, -1)).toThrow(/cannot be negative/);
  });
});

describe("timestamps", () => {
  test("chunk spans advance and never run backwards", () => {
    const segments = Array.from({ length: 30 }, (_, i) => seg(filler(10), i * 5, i * 5 + 5));
    const chunks = fixedChunks(segments, 100, 25);
    for (const c of chunks) expect(c.end_s).toBeGreaterThan(c.start_s);
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i]!.start_s).toBeGreaterThan(chunks[i - 1]!.start_s);
    }
  });

  test("end_s is the largest end among the segments, not the last one's", () => {
    // Captions in this corpus overlap in time, so the final segment's end_s is
    // not necessarily the highest.
    const chunks = fixedChunks([seg("a", 0, 30), seg("b", 10, 20)], 512);
    expect(chunks[0]!.end_s).toBe(30);
  });
});
