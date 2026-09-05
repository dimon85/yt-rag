import { describe, expect, test } from "vitest";
import { mergeIntoCorpus } from "../src/merge.ts";
import type { Video } from "../src/corpus.ts";

const v = (youtube_id: string, channel: string, over: Partial<Video> = {}): Video => ({
  youtube_id,
  channel,
  title: `t-${youtube_id}`,
  published_at: "2026-01-01",
  duration_s: 900,
  transcript_kind: "generated",
  tools: [],
  topics: [],
  ...over,
});

describe("mergeIntoCorpus", () => {
  test("a single-channel run leaves the other channels alone", () => {
    // The bug this exists for: discover --channel @t3dotgg rebuilt videos:
    // from one channel and deleted the other eleven.
    const existing = [v("aaaaaaaaaaa", "Theo"), v("bbbbbbbbbbb", "Fireship"), v("ccccccccccc", "Cole")];
    const merged = mergeIntoCorpus(existing, [v("ddddddddddd", "Theo")], ["Theo"]);

    expect(merged.map((x) => x.youtube_id).sort())
      .toEqual(["bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"]);
  });

  test("a full run replaces everything it scanned", () => {
    const existing = [v("aaaaaaaaaaa", "Theo"), v("bbbbbbbbbbb", "Fireship")];
    const merged = mergeIntoCorpus(
      existing,
      [v("ccccccccccc", "Theo"), v("ddddddddddd", "Fireship")],
      ["Theo", "Fireship"],
    );
    expect(merged.map((x) => x.youtube_id)).toEqual(["ccccccccccc", "ddddddddddd"]);
  });

  test("hand-written topics survive re-selection", () => {
    // topics is filled in by hand during golden-set work; a later discovery
    // run must not erase hours of annotation.
    const existing = [v("aaaaaaaaaaa", "Theo", { topics: ["reliability", "cost"] })];
    const merged = mergeIntoCorpus(existing, [v("aaaaaaaaaaa", "Theo")], ["Theo"]);
    expect(merged[0]!.topics).toEqual(["reliability", "cost"]);
  });

  test("hand-written tools survive, computed ones fill the gap", () => {
    const existing = [v("aaaaaaaaaaa", "Theo", { tools: ["mcp"] }), v("bbbbbbbbbbb", "Theo")];
    const merged = mergeIntoCorpus(
      existing,
      [v("aaaaaaaaaaa", "Theo", { tools: ["cursor"] }), v("bbbbbbbbbbb", "Theo", { tools: ["codex"] })],
      ["Theo"],
    );
    expect(merged.find((x) => x.youtube_id === "aaaaaaaaaaa")!.tools).toEqual(["mcp"]);
    expect(merged.find((x) => x.youtube_id === "bbbbbbbbbbb")!.tools).toEqual(["codex"]);
  });

  test("fresh metadata wins for everything not hand-written", () => {
    const existing = [v("aaaaaaaaaaa", "Theo", { title: "old", view_count: 1 })];
    const merged = mergeIntoCorpus(
      existing,
      [v("aaaaaaaaaaa", "Theo", { title: "new", view_count: 999 })],
      ["Theo"],
    );
    expect(merged[0]!.title).toBe("new");
    expect(merged[0]!.view_count).toBe(999);
  });

  test("a video that drops out of the selection is removed", () => {
    const existing = [v("aaaaaaaaaaa", "Theo"), v("bbbbbbbbbbb", "Theo")];
    const merged = mergeIntoCorpus(existing, [v("aaaaaaaaaaa", "Theo")], ["Theo"]);
    expect(merged.map((x) => x.youtube_id)).toEqual(["aaaaaaaaaaa"]);
  });
});
