import { describe, expect, test } from "vitest";
import {
  AD_PATTERN, clashes, dedupeOverlapping, score, stratify, tile, windows,
  type Scored, type Window,
} from "../src/candidates.ts";
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

describe("tile", () => {
  const segs = Array.from({ length: 20 }, (_, i) => seg(`s${i}`, i * 5, i * 5 + 5));

  test("windows do not overlap", () => {
    const ts = tile(segs, 30);
    for (let i = 1; i < ts.length; i++) {
      expect(ts[i]!.start_s).toBeGreaterThanOrEqual(ts[i - 1]!.end_s - 5);
    }
  });

  test("covers the transcript end to end", () => {
    const ts = tile(segs, 30);
    expect(ts[0]!.start_s).toBe(0);
    expect(ts.at(-1)!.end_s).toBe(100);
  });

  test("an empty transcript yields nothing", () => {
    expect(tile([])).toEqual([]);
  });
});

describe("stratify", () => {
  const rows = Array.from({ length: 40 }, (_, i) => ({
    id: i,
    score: { total: 40 - i, reasons: [], tools: [], topics: [] },
  }));

  test("returns everything when there is nothing to choose between", () => {
    expect(stratify(rows.slice(0, 5), 10)).toHaveLength(5);
  });

  test("does not simply take the top n", () => {
    // The point: the top n are the densest passages, which are the most
    // findable, and a set of those measures the selection rather than the
    // retriever.
    expect(stratify(rows, 8).map((r) => r.id)).not.toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  test("reaches into the weakest band", () => {
    expect(Math.max(...stratify(rows, 8).map((r) => r.id))).toBeGreaterThan(rows.length * 0.6);
  });

  test("returns at most n", () => {
    expect(stratify(rows, 8).length).toBeLessThanOrEqual(8);
  });
});

describe("AD_PATTERN", () => {
  test("catches the phrasings that actually appear", () => {
    // The one that reached a generated batch said neither "sponsor" nor "ad".
    for (const t of [
      "get started with SERP API using 250 free credits",
      "clicking the link in the description",
      "just scan the QR code that you see on the screen",
      "here's a word from our sponsor",
      // Found in a candidate pair, matching none of the forms above.
      "I highly recommend you go and check out firecrawl",
      "they actually provided you 12% off if you want the agent subscription",
    ]) {
      expect(AD_PATTERN.test(t)).toBe(true);
    }
  });

  test("leaves ordinary claims alone", () => {
    for (const t of [
      "the rate limit was doubled for paid plans",
      "it ran for six hours building an iOS app",
      // "check out" alone is ordinary speech and must not trip the filter.
      "let's check out what the agent manager does",
    ]) {
      expect(AD_PATTERN.test(t)).toBe(false);
    }
  });
});

describe("clashes", () => {
  test("adjacent tiles can still overlap, because captions do", () => {
    // Two picks eight seconds apart produced near-duplicate questions in the
    // first generated batch.
    expect(clashes({ start_s: 1316, end_s: 1351 }, [{ start_s: 1287, end_s: 1324 }])).toBe(true);
  });

  test("touching at a point is not a clash", () => {
    expect(clashes({ start_s: 30, end_s: 60 }, [{ start_s: 0, end_s: 30 }])).toBe(false);
  });

  test("nothing taken means no clash", () => {
    expect(clashes({ start_s: 0, end_s: 30 }, [])).toBe(false);
  });
});
