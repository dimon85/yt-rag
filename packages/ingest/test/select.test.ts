import { describe, expect, test } from "vitest";
import {
  decide, normalizeTitle, select, spreadMonths, spreadPick, toolsInTitle,
  type Candidate, type Window,
} from "../src/select.ts";

const W: Window = {
  minDurationS: 480,
  maxDurationS: 3600,
  months: 18,
  now: new Date("2026-09-05T00:00:00Z"),
};

const base: Candidate = {
  youtube_id: "aaaaaaaaaaa",
  channel: "Test",
  title: "Claude Code review",
  duration_s: 900,
  published_at: "2026-03-01",
  has_auto_en: true,
};

describe("decide", () => {
  test("keeps a video inside every bound", () => {
    expect(decide(base, W)).toEqual({ keep: true, transcript_kind: "generated" });
  });

  test("manual subtitles win over auto captions", () => {
    expect(decide({ ...base, has_manual_en: true }, W))
      .toEqual({ keep: true, transcript_kind: "manual" });
  });

  test("no transcript is a normal rejection, not an error", () => {
    expect(decide({ ...base, has_auto_en: false }, W))
      .toEqual({ keep: false, reason: "no-transcript" });
  });

  test.each([
    [479, "too-short"],
    [3601, "too-long"],
  ])("duration %i is rejected as %s", (duration_s, reason) => {
    expect(decide({ ...base, duration_s }, W)).toEqual({ keep: false, reason });
  });

  test.each([[480], [3600]])("duration %i is inclusive", (duration_s) => {
    expect(decide({ ...base, duration_s }, W).keep).toBe(true);
  });

  test("a video older than the window is dropped", () => {
    expect(decide({ ...base, published_at: "2024-01-01" }, W))
      .toEqual({ keep: false, reason: "outside-window" });
  });

  test("duration is checked before the date", () => {
    // Both are wrong; the reported reason must be deterministic, not incidental.
    expect(decide({ ...base, duration_s: 10, published_at: "2000-01-01" }, W))
      .toEqual({ keep: false, reason: "too-short" });
  });
});

describe("normalizeTitle", () => {
  test("strips case, punctuation and emoji", () => {
    expect(normalizeTitle("Claude Code — REVIEW!! 🚀")).toBe("claude code review");
  });

  test("two reuploads with cosmetic differences collapse to one key", () => {
    expect(normalizeTitle("Cursor vs. Copilot (2026)"))
      .toBe(normalizeTitle("cursor vs copilot 2026"));
  });
});

describe("select", () => {
  test("drops the second copy of the same title and says why", () => {
    const r = select(
      [base, { ...base, youtube_id: "bbbbbbbbbbb", title: "claude code review!" }],
      W,
    );
    expect(r.selected).toHaveLength(1);
    expect(r.rejected).toEqual([
      { youtube_id: "bbbbbbbbbbb", title: "claude code review!", reason: "duplicate-title" },
    ]);
  });

  test("every rejected candidate carries a reason", () => {
    const r = select([{ ...base, duration_s: 30 }, { ...base, has_auto_en: false }], W);
    expect(r.selected).toHaveLength(0);
    expect(r.rejected.map((x) => x.reason)).toEqual(["too-short", "no-transcript"]);
  });
});

describe("toolsInTitle", () => {
  const tools = [
    { id: "claude-code", aliases: ["claude code", "cc"] },
    { id: "cursor", aliases: ["cursor", "composer"] },
  ];

  test("matches an alias", () => {
    expect(toolsInTitle("Building with Claude Code", tools)).toEqual(["claude-code"]);
  });

  test("matches several tools in one title", () => {
    expect(toolsInTitle("Cursor vs Claude Code", tools)).toEqual(["claude-code", "cursor"]);
  });

  test("matches on word boundaries, not substrings", () => {
    // "cc" must not fire on "soccer"; the alias list is short and generic on purpose.
    expect(toolsInTitle("soccer highlights", tools)).toEqual([]);
  });
});

describe("spreadMonths", () => {
  test("a year apart is about twelve months", () => {
    expect(spreadMonths(["2025-09-01", "2026-09-01"])).toBeCloseTo(11.99, 1);
  });

  test("a single date spans nothing", () => {
    expect(spreadMonths(["2026-09-01"])).toBe(0);
  });
});

describe("spreadPick", () => {
  const at = (d: string) => ({ d });
  const dateOf = (x: { d: string }) => x.d;

  test("returns everything when there is nothing to choose between", () => {
    const items = [at("2026-01-01"), at("2026-02-01")];
    expect(spreadPick(items, 5, dateOf)).toEqual(items);
  });

  test("takes one from each month before doubling up on any", () => {
    const items = [
      at("2026-01-01"), at("2026-01-02"), at("2026-01-03"),
      at("2026-06-01"), at("2026-09-01"),
    ];
    expect(spreadPick(items, 3, dateOf).map(dateOf))
      .toEqual(["2026-01-01", "2026-06-01", "2026-09-01"]);
  });

  test("keeps both ends of the range", () => {
    const items = Array.from({ length: 12 }, (_, i) =>
      at(`2026-${String(i + 1).padStart(2, "0")}-01`));
    const picked = spreadPick(items, 4, dateOf).map(dateOf);
    expect(picked[0]).toBe("2026-01-01");
    expect(picked.at(-1)).toBe("2026-12-01");
  });

  test("a single pick comes from the middle, not an edge", () => {
    const items = Array.from({ length: 5 }, (_, i) => at(`2026-0${i + 1}-01`));
    expect(spreadPick(items, 1, dateOf).map(dateOf)).toEqual(["2026-03-01"]);
  });

  test("beats slicing the newest n on spread", () => {
    // Twelve monthly videos; picking four should cover most of the year.
    const items = Array.from({ length: 12 }, (_, i) =>
      at(`2026-${String(i + 1).padStart(2, "0")}-01`));
    const picked = spreadPick(items, 4, dateOf).map(dateOf);
    expect(spreadMonths(picked)).toBeGreaterThan(8);
    expect(spreadMonths(items.slice(-4).map(dateOf))).toBeLessThan(4);
  });

  test("falls back gracefully when one month holds everything", () => {
    const items = Array.from({ length: 5 }, (_, i) => at(`2026-03-0${i + 1}`));
    expect(spreadPick(items, 3, dateOf)).toHaveLength(3);
  });
});
