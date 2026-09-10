import { describe, expect, test } from "vitest";
import {
  type Passage, renderDated, REVISION_PROMPT, type Revision, RevisionSchema, verifyRevisions,
} from "../src/clash.ts";

const passage = (
  id: string, video: string, channel: string, published_at: string, text = "some words",
): Passage => ({ id, video, channel, published_at, start_s: 10, end_s: 40, text });

const rev = (over: Partial<Revision> = {}): Revision => ({
  claim: "The weekly limit applies to every model on the plan.",
  earlier_id: "e1",
  later_id: "l1",
  why: "the later video says the cap is now per-model",
  ...over,
});

describe("what a revision is", () => {
  const offered = [
    passage("e1", "vidOld", "Someone", "2026-01-10"),
    passage("l1", "vidNew", "Someone", "2026-05-02"),
  ];

  test("the same author correcting themselves later is kept", () => {
    // corpus.yaml's selection_rules names outdated claims as the second source
    // of contradictions, and the 12-18 month range was chosen for it. Nothing
    // was mining it until now.
    const { kept, rejected } = verifyRevisions([rev()], offered);
    expect(rejected).toEqual([]);
    expect(kept[0]!.earlier.video).toBe("vidOld");
    expect(kept[0]!.later.video).toBe("vidNew");
  });

  test("two different authors is not a revision", () => {
    // That is what --relation clash is for. Mixing them would let the same
    // pair be found twice under two names.
    const twoPeople = [
      passage("e1", "vidOld", "One", "2026-01-10"),
      passage("l1", "vidNew", "Two", "2026-05-02"),
    ];
    expect(verifyRevisions([rev()], twoPeople).rejected[0]!.reason).toMatch(/different authors/);
  });

  test("the later span must actually be later", () => {
    // The model is told the dates and can still get the direction wrong, and a
    // reversed pair reads as a correction that never happened.
    const backwards = [
      passage("e1", "vidOld", "Someone", "2026-05-02"),
      passage("l1", "vidNew", "Someone", "2026-01-10"),
    ];
    expect(verifyRevisions([rev()], backwards).rejected[0]!.reason).toMatch(/not later/);
  });

  test("two claims in one video are rejected, as everywhere else", () => {
    // Both spans must sit in different videos or contradiction coverage is
    // satisfied by any retriever returning two adjacent chunks.
    const same = [
      passage("e1", "vidOld", "Someone", "2026-01-10"),
      passage("l1", "vidOld", "Someone", "2026-01-10"),
    ];
    expect(verifyRevisions([rev()], same).rejected[0]!.reason).toMatch(/one video/);
  });

  test("an invented id is rejected and named", () => {
    expect(verifyRevisions([rev({ later_id: "l9" })], offered).rejected[0]!.reason)
      .toMatch(/unknown passage l9/);
  });

  test("a claim phrased as a question is flagged, not thrown away", () => {
    const q = rev({ claim: "Does the weekly limit apply to every model?" });
    const { kept, rejected } = verifyRevisions([q], offered);
    expect(rejected).toEqual([]);
    expect(kept[0]!.claimProblem).toMatch(/question/);
  });

  test("same day is not later", () => {
    const sameDay = [
      passage("e1", "vidOld", "Someone", "2026-01-10"),
      passage("l1", "vidNew", "Someone", "2026-01-10"),
    ];
    expect(verifyRevisions([rev()], sameDay).rejected[0]!.reason).toMatch(/not later/);
  });
});

describe("rendering for a revision search", () => {
  test("dates are shown, because the relation is defined by order in time", () => {
    const out = renderDated([
      passage("e1", "vidOld", "Someone", "2026-01-10", "the cap is five hours"),
      passage("l1", "vidNew", "Someone", "2026-05-02", "there is no cap now"),
    ]);
    expect(out).toContain("2026-01-10");
    expect(out).toContain("2026-05-02");
    expect(out).toContain("[e1]");
    expect(out).toContain("the cap is five hours");
  });

  test("passages are listed oldest first, so the order is not a puzzle", () => {
    const out = renderDated([
      passage("l1", "vidNew", "Someone", "2026-05-02"),
      passage("e1", "vidOld", "Someone", "2026-01-10"),
    ]);
    expect(out.indexOf("2026-01-10")).toBeLessThan(out.indexOf("2026-05-02"));
  });
});

describe("schema and prompt", () => {
  test("an empty list is valid", () => {
    expect(RevisionSchema.safeParse({ revisions: [] }).success).toBe(true);
  });

  test("a claim too short to be a proposition is rejected", () => {
    expect(RevisionSchema.safeParse({ revisions: [{ ...rev(), claim: "no" }] }).success).toBe(false);
  });

  test("the prompt asks for one author and carries the shared rules", () => {
    expect(REVISION_PROMPT).toMatch(/same (author|speaker)/i);
    expect(REVISION_PROMPT).toMatch(/never invent an id/i);
    expect(REVISION_PROMPT).toMatch(/empty list/i);
  });
});
