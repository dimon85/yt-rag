import { describe, expect, test } from "vitest";
import {
  ClashSchema, type Clash, type Passage, POOLED_PROMPT, PROMPT, renderPassages, verify,
} from "../src/clash.ts";

const passage = (id: string, video: string, text = "some words"): Passage => ({
  id, video, channel: "Someone", start_s: 10, end_s: 40, text,
});

const clash = (over: Partial<Clash> = {}): Clash => ({
  claim: "The session limit resets every five hours.",
  pro: "p1",
  contra: "c1",
  why: "one states five hours, the other says there is no fixed reset",
  directness: "flat",
  ...over,
});

describe("guarding against invented citations", () => {
  // The failure this exists for: a model asked for timestamps produces
  // plausible ones. A gold span pointing at the wrong place scores a correct
  // retrieval as a failure for as long as the set lives.
  const offered = [passage("p1", "vidA"), passage("c1", "vidB")];

  test("a clash citing offered passages is kept, with the passages attached", () => {
    const { kept, rejected } = verify([clash()], offered);
    expect(rejected).toEqual([]);
    expect(kept).toHaveLength(1);
    expect(kept[0]!.pro.video).toBe("vidA");
    expect(kept[0]!.contra.video).toBe("vidB");
  });

  test("an unknown id is rejected and named", () => {
    const { kept, rejected } = verify([clash({ contra: "c99" })], offered);
    expect(kept).toEqual([]);
    expect(rejected[0]!.reason).toMatch(/unknown passage c99/);
  });

  test("both ids unknown are both named, so the cause is not guessed at", () => {
    const { rejected } = verify([clash({ pro: "x", contra: "y" })], offered);
    expect(rejected[0]!.reason).toMatch(/x/);
    expect(rejected[0]!.reason).toMatch(/y/);
  });
});

describe("both sides must be different authors", () => {
  test("a clash inside one video is rejected", () => {
    // A speaker qualifying himself is not the contradiction the corpus is
    // supposed to contain, and any retriever returning two adjacent chunks
    // would satisfy it.
    const same = [passage("p1", "vidA"), passage("c1", "vidA")];
    const { kept, rejected } = verify([clash()], same);
    expect(kept).toEqual([]);
    expect(rejected[0]!.reason).toMatch(/one video, vidA/);
  });
});

describe("the claim is a proposition, not a question", () => {
  test("a question is rejected", () => {
    // A question phrased from the passage is found by term matching alone.
    // The question gets written later, from the claim, passage out of view.
    const offered = [passage("p1", "vidA"), passage("c1", "vidB")];
    const { kept, rejected } = verify([clash({ claim: "Does the limit reset every five hours?" })], offered);
    expect(kept).toEqual([]);
    expect(rejected[0]!.reason).toMatch(/question/);
  });

  test("a trailing question mark is caught through surrounding space", () => {
    const offered = [passage("p1", "vidA"), passage("c1", "vidB")];
    expect(verify([clash({ claim: "Is it five hours?  " })], offered).kept).toEqual([]);
  });

  test("a question mark inside a statement is not a question", () => {
    const offered = [passage("p1", "vidA"), passage("c1", "vidB")];
    const claim = 'One side answers "how long?" with five hours, the other with no fixed limit.';
    expect(verify([clash({ claim })], offered).kept).toHaveLength(1);
  });
});

describe("schema", () => {
  test("a claim too short to be a proposition is rejected at parse", () => {
    expect(ClashSchema.safeParse({ clashes: [{ ...clash(), claim: "nope" }] }).success).toBe(false);
  });

  test("directness is one of two stated values", () => {
    expect(ClashSchema.safeParse({ clashes: [{ ...clash(), directness: "sort of" }] }).success).toBe(false);
    expect(ClashSchema.safeParse({ clashes: [clash({ directness: "partial" })] }).success).toBe(true);
  });

  test("an empty list is valid, because most cells contain no clash", () => {
    // 32 vocabulary-matched pairs produced 0 disagreements. A tool that cannot
    // return nothing would invent something.
    expect(ClashSchema.safeParse({ clashes: [] }).success).toBe(true);
  });
});

describe("rendering", () => {
  test("passages carry the id the model must cite, and the channel", () => {
    const out = renderPassages([passage("p1", "vidA", "limit is five hours")], [passage("c1", "vidB", "never ran out")]);
    expect(out).toContain("[p1]");
    expect(out).toContain("[c1]");
    expect(out).toContain("limit is five hours");
    expect(out.indexOf("PRO")).toBeLessThan(out.indexOf("CONTRA"));
  });

  test("one list is rendered without stance labels", () => {
    // Splitting by stance assumes disagreement runs along the hype/skeptic
    // axis. It does not for mcp: hype channels mention it twice across 26
    // videos, practical ones 64 times across 34, so every mcp cell has an
    // empty pro side and the split finds nothing by construction.
    const out = renderPassages([passage("p1", "vidA"), passage("p2", "vidB")]);
    expect(out).toContain("PASSAGES");
    expect(out).not.toContain("PRO");
    expect(out).not.toContain("CONTRA");
    expect(out).toContain("[p2]");
  });

  test("both prompts demand a proposition and forbid inventing ids", () => {
    for (const p of [PROMPT, POOLED_PROMPT]) {
      expect(p).toMatch(/never phrase it as\s+a question/i);
      expect(p).toMatch(/never invent an id/i);
      expect(p).toMatch(/empty list/i);
    }
  });

  test("timestamps are not offered, so they cannot be echoed back wrong", () => {
    // Ids map back to spans locally. Sending times invites the model to cite
    // times it adjusted, and a plausible adjustment is the dangerous case.
    const out = renderPassages([passage("p1", "vidA")], [passage("c1", "vidB")]);
    expect(out).not.toContain("10");
    expect(out).not.toContain("40");
  });
});
