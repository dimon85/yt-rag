import { describe, expect, test } from "vitest";
import { type Annotated, duplicateOf, type Passage } from "../src/clash.ts";

const p = (video: string, start_s: number, end_s: number): Passage =>
  ({ id: "x", video, channel: "Someone", start_s, end_s, text: "" });

const q = (slug: string, spans: [string, number, number][]): Annotated => ({
  slug,
  gold: spans.map(([video, start_s, end_s]) => ({ video, start_s, end_s })),
});

describe("a candidate the set already covers", () => {
  // The case that prompted this: the complement search returned a pair whose
  // two spans were the same two videos as an existing comparative question,
  // shifted by a few seconds. Writing it again would add a question that
  // measures what another already measures, and cost an hour to find out.
  const existing = [q("already-asked", [["vidA", 100, 200], ["vidB", 600, 700]])];

  test("both spans overlapping one question is a duplicate, and it is named", () => {
    expect(duplicateOf(p("vidA", 150, 250), p("vidB", 650, 750), existing)).toBe("already-asked");
  });

  test("a few seconds of shift does not hide it", () => {
    expect(duplicateOf(p("vidA", 195, 300), p("vidB", 699, 800), existing)).toBe("already-asked");
  });

  test("only one span overlapping is not a duplicate", () => {
    // A stretch of video can legitimately answer more than one question. It is
    // the pair that would be redundant, not the passage.
    expect(duplicateOf(p("vidA", 150, 250), p("vidB", 900, 1000), existing)).toBeNull();
  });

  test("the same two videos at unrelated times is not a duplicate", () => {
    expect(duplicateOf(p("vidA", 900, 1000), p("vidB", 900, 1000), existing)).toBeNull();
  });

  test("order does not matter — a and b may arrive either way round", () => {
    expect(duplicateOf(p("vidB", 650, 750), p("vidA", 150, 250), existing)).toBe("already-asked");
  });

  test("touching at an endpoint does not count as overlapping", () => {
    // Half-open intervals, the same convention as the hit rule. Two spans that
    // merely abut cover different seconds.
    expect(duplicateOf(p("vidA", 200, 300), p("vidB", 700, 800), existing)).toBeNull();
  });
});

describe("spread across questions", () => {
  test("two spans matching two different questions is not a duplicate of either", () => {
    // This is the interesting case: each half is annotated, but no existing
    // question joins them, so the pair is new.
    const set = [
      q("one", [["vidA", 100, 200]]),
      q("two", [["vidB", 600, 700]]),
    ];
    expect(duplicateOf(p("vidA", 150, 160), p("vidB", 650, 660), set)).toBeNull();
  });

  test("a question with no gold spans cannot match", () => {
    // Negatives carry `gold: []`.
    expect(duplicateOf(p("vidA", 1, 2), p("vidB", 1, 2), [q("negative", [])])).toBeNull();
  });

  test("an empty set matches nothing", () => {
    expect(duplicateOf(p("vidA", 1, 2), p("vidB", 1, 2), [])).toBeNull();
  });
});
