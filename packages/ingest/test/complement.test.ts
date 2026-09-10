import { describe, expect, test } from "vitest";
import {
  COMPLEMENT_PROMPT, ComplementSchema, type Complement, type Passage, verifyComplements,
} from "../src/clash.ts";

const passage = (id: string, video: string, text = "some words"): Passage => ({
  id, video, channel: "Someone", published_at: "2026-01-01", start_s: 10, end_s: 40, text,
});

const pair = (over: Partial<Complement> = {}): Complement => ({
  subject: "The ways of keeping unneeded tool definitions out of a model's context.",
  a: "p1",
  b: "p2",
  a_only: "names progressive disclosure and the file layout it needs",
  b_only: "names deferred loading and the per-tool override",
  ...over,
});

describe("what a comparative pair has to be", () => {
  const offered = [passage("p1", "vidA"), passage("p2", "vidB")];

  test("a pair citing offered passages from two videos is kept", () => {
    const { kept, rejected } = verifyComplements([pair()], offered);
    expect(rejected).toEqual([]);
    expect(kept).toHaveLength(1);
    expect(kept[0]!.a.video).toBe("vidA");
    expect(kept[0]!.b.video).toBe("vidB");
  });

  test("a pair inside one video is rejected", () => {
    // The kind's whole definition: a comparative question requires spans from
    // two different videos. `pnpm golden` enforces it, and a candidate that
    // cannot become a question is worse than no candidate.
    const same = [passage("p1", "vidA"), passage("p2", "vidA")];
    expect(verifyComplements([pair()], same).rejected[0]!.reason).toMatch(/one video, vidA/);
  });

  test("an invented id is rejected and named", () => {
    expect(verifyComplements([pair({ b: "p99" })], offered).rejected[0]!.reason)
      .toMatch(/unknown passage p99/);
  });

  test("a subject phrased as a question is rejected", () => {
    // Same reason as for clashes: the question gets written from the subject
    // afterwards, with the passage out of view. A subject handed over already
    // phrased as a question invites lifting it verbatim.
    const q = pair({ subject: "How do you keep tool definitions out of context?" });
    expect(verifyComplements([q], offered).rejected[0]!.reason).toMatch(/question/);
  });

  test("a pair where one side contributes nothing is rejected", () => {
    // This is the check that separates a comparative pair from two redundant
    // sources. If B adds nothing, the answer is in A alone and the question is
    // factual — and it would be scored against two spans, one of which no
    // retriever needs to find.
    const redundant = pair({ b_only: "" });
    expect(verifyComplements([redundant], offered).rejected[0]!.reason).toMatch(/contributes nothing/);
    expect(verifyComplements([pair({ a_only: "   " })], offered).rejected[0]!.reason)
      .toMatch(/contributes nothing/);
  });
});

describe("schema", () => {
  test("a subject too short to state a topic is rejected at parse", () => {
    expect(ComplementSchema.safeParse({ pairs: [{ ...pair(), subject: "MCP" }] }).success).toBe(false);
  });

  test("an empty list is valid", () => {
    expect(ComplementSchema.safeParse({ pairs: [] }).success).toBe(true);
  });

  test("both contribution fields are required, not optional", () => {
    const { a_only, ...without } = pair();
    expect(ComplementSchema.safeParse({ pairs: [without] }).success).toBe(false);
  });
});

describe("the prompt", () => {
  test("it forbids redundant pairs, which is the whole difficulty", () => {
    expect(COMPLEMENT_PROMPT).toMatch(/neither/i);
    expect(COMPLEMENT_PROMPT).toMatch(/different authors/i);
  });

  test("it carries the same two rules as the clash prompt", () => {
    expect(COMPLEMENT_PROMPT).toMatch(/never invent an id/i);
    expect(COMPLEMENT_PROMPT).toMatch(/empty list/i);
    expect(COMPLEMENT_PROMPT).not.toMatch(/cannot both be true/i);
  });
});
