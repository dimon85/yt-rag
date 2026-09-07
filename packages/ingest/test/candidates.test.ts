import { describe, expect, test } from "vitest";
import { dedupeOverlapping, score, windows, type Scored, type Window } from "../src/candidates.ts";
import type { Segment } from "../src/text.ts";

const seg = (text: string, start_s: number, end_s: number): Segment => ({ text, start_s, end_s });
const TOOLS = [{ id: "cursor", aliases: ["cursor"] }, { id: "codex", aliases: ["codex"] }];
const TOPICS = [{ id: "cost", keywords: ["rate limit", "expensive"] }];

const win = (text: string): Window => ({
  segments: [seg(text, 0, 30)], start_s: 0, end_s: 30, text,
});

describe("windows", () => {
  test("an empty transcript yields nothing", () => {
    expect(windows([])).toEqual([]);
  });

  test("windows overlap, so a claim on a boundary survives whole somewhere", () => {
    const segs = Array.from({ length: 12 }, (_, i) => seg(`s${i}`, i * 5, i * 5 + 5));
    const ws = windows(segs, 30);
    expect(ws.length).toBeGreaterThan(1);
    // Consecutive windows share segments — that is the point.
    expect(ws[1]!.start_s).toBeLessThan(ws[0]!.end_s);
  });

  test("every window carries its own span and text", () => {
    const ws = windows([seg("hello", 0, 10), seg("world", 10, 20)], 30);
    expect(ws[0]!.text).toBe("hello world");
    expect(ws[0]!.start_s).toBe(0);
    expect(ws[0]!.end_s).toBe(20);
  });
});

describe("score", () => {
  test("narration with nothing in it scores zero", () => {
    const s = score(win("so anyway that is what happened next"), TOOLS, TOPICS);
    expect(s.total).toBe(0);
    expect(s.reasons).toEqual([]);
  });

  test("a number with a unit counts, a bare number does not", () => {
    expect(score(win("it costs 20 dollars"), TOOLS, TOPICS).total)
      .toBeGreaterThan(score(win("it costs 20"), TOOLS, TOPICS).total);
  });

  test("tool mentions are capped so repetition does not dominate", () => {
    const few = score(win("cursor cursor"), TOOLS, TOPICS).total;
    const many = score(win("cursor cursor cursor cursor cursor cursor"), TOOLS, TOPICS).total;
    expect(many).toBe(few + 1);   // 2 -> 3, then capped
  });

  test("reasons name what fired, so a bad ranking is inspectable", () => {
    const s = score(win("cursor is better than codex, 5 hours per day"), TOOLS, TOPICS);
    expect(s.reasons.some((r) => r.startsWith("tool:"))).toBe(true);
    expect(s.reasons.some((r) => r.includes("claim marker"))).toBe(true);
    expect(s.reasons.some((r) => r.includes("number"))).toBe(true);
  });

  test("topics match spoken keywords, not the topic name", () => {
    expect(score(win("the rate limit is brutal"), TOOLS, TOPICS).topics).toEqual(["cost"]);
    expect(score(win("this is about cost"), TOOLS, TOPICS).topics).toEqual([]);
  });

  test("a scarce stance lifts a scoring window but cannot rescue an empty one", () => {
    const empty = score(win("nothing here at all"), TOOLS, TOPICS, { scarceStance: true });
    expect(empty.total).toBe(0);
    const real = score(win("cursor broke on me"), TOOLS, TOPICS, { scarceStance: true });
    expect(real.reasons).toContain("scarce stance");
  });
});

describe("dedupeOverlapping", () => {
  const at = (start_s: number, end_s: number, total: number): Scored => ({
    window: { segments: [], start_s, end_s, text: "" },
    score: { total, reasons: [], tools: [], topics: [] },
  });

  test("keeps the best of a cluster of neighbours", () => {
    // Windows step by half their width, so one strong claim produces three
    // near-identical entries; without this the top of the list repeats itself.
    expect(dedupeOverlapping([at(0, 30, 5), at(15, 45, 9), at(30, 60, 3)]).map((s) => s.score.total))
      .toEqual([9]);
  });

  test("keeps windows that do not touch", () => {
    expect(dedupeOverlapping([at(0, 30, 5), at(100, 130, 3)])).toHaveLength(2);
  });

  test("returns results best first", () => {
    expect(dedupeOverlapping([at(0, 10, 1), at(100, 110, 8), at(200, 210, 4)])
      .map((s) => s.score.total)).toEqual([8, 4, 1]);
  });
});
