import { describe, expect, test } from "vitest";
import {
  countByKind, goldenSha, lexicalOverlap, validateStructure,
  type CorpusFacts, type GoldenSet, type Question,
} from "../src/golden.ts";

const corpus: CorpusFacts = {
  durations: new Map([["aaaaaaaaaaa", 1200], ["bbbbbbbbbbb", 900], ["ccccccccccc", 600]]),
};

const q = (over: Partial<Question>): Question => ({
  slug: "s", text: "a question long enough to pass", kind: "factual",
  topics: [], tools: [], source: "hand", gold: [], ...over,
});
const set = (...questions: Question[]): GoldenSet => ({ version: 1, questions });
const span = (video: string, start_s = 10, end_s = 40, side?: "pro" | "contra") =>
  ({ video, start_s, end_s, ...(side ? { side } : {}) });

const errors = (s: GoldenSet) =>
  validateStructure(s, corpus).filter((i) => i.level === "error").map((i) => i.message);

describe("span sanity", () => {
  test("a span cannot end before it starts", () => {
    expect(errors(set(q({ gold: [span("aaaaaaaaaaa", 90, 30)] })))[0]).toMatch(/ends before/);
  });

  test("a span cannot run past the end of the video", () => {
    expect(errors(set(q({ gold: [span("ccccccccccc", 10, 2000)] })))[0]).toMatch(/past the end/);
  });

  test("a span must point at a video in the corpus", () => {
    expect(errors(set(q({ gold: [span("zzzzzzzzzzz")] })))[0]).toMatch(/not in corpus/);
  });

  test("duplicate slugs are rejected", () => {
    expect(errors(set(q({ slug: "dup", gold: [span("aaaaaaaaaaa")] }),
                      q({ slug: "dup", gold: [span("bbbbbbbbbbb")] })))).toContain("duplicate slug");
  });
});

describe("factual", () => {
  test("one span is right", () => {
    expect(errors(set(q({ gold: [span("aaaaaaaaaaa")] })))).toEqual([]);
  });

  test("two spans mean it is really comparative", () => {
    // Scoring it as factual credits a retriever that found either half.
    expect(errors(set(q({ gold: [span("aaaaaaaaaaa"), span("bbbbbbbbbbb")] })))[0])
      .toMatch(/exactly 1 gold span/);
  });

  test("no span at all is rejected", () => {
    expect(errors(set(q({ gold: [] })))[0]).toMatch(/exactly 1 gold span/);
  });
});

describe("comparative", () => {
  test("two different videos is right", () => {
    expect(errors(set(q({ kind: "comparative", gold: [span("aaaaaaaaaaa"), span("bbbbbbbbbbb")] }))))
      .toEqual([]);
  });

  test("two spans of the same video is not comparative", () => {
    // It is a factual question with extra words, and nothing forces the
    // retriever to reach a second source.
    expect(errors(set(q({ kind: "comparative", gold: [span("aaaaaaaaaaa", 10, 40), span("aaaaaaaaaaa", 100, 140)] })))[0])
      .toMatch(/2\+ videos/);
  });
});

describe("contradiction", () => {
  test("pro and contra from different videos is right", () => {
    expect(errors(set(q({
      kind: "contradiction",
      gold: [span("aaaaaaaaaaa", 10, 40, "pro"), span("bbbbbbbbbbb", 10, 40, "contra")],
    })))).toEqual([]);
  });

  test("two spans on the same side is not a contradiction", () => {
    // This is what contradiction coverage measures; without both sides the
    // metric reports on a question that has only one.
    expect(errors(set(q({
      kind: "contradiction",
      gold: [span("aaaaaaaaaaa", 10, 40, "pro"), span("bbbbbbbbbbb", 10, 40, "pro")],
    })))).toContain(
      "contradiction needs both a pro and a contra span — that is what coverage measures",
    );
  });

  test("an unlabelled span is rejected", () => {
    expect(errors(set(q({
      kind: "contradiction",
      gold: [span("aaaaaaaaaaa", 10, 40, "pro"), span("bbbbbbbbbbb")],
    })))).toContain("every contradiction span needs side: pro or contra");
  });

  test("both sides from one video is allowed, with a warning", () => {
    // An author contradicting themselves in one video is real. The earlier
    // rule encoded "one side, one video" as a requirement when it was only an
    // assumption.
    const s = set(q({
      kind: "contradiction",
      gold: [span("aaaaaaaaaaa", 10, 40, "pro"), span("aaaaaaaaaaa", 600, 640, "contra")],
    }));
    expect(errors(s)).toEqual([]);
    expect(validateStructure(s, corpus).filter((i) => i.level === "warning")).toHaveLength(1);
  });
});

describe("negative", () => {
  test("no spans is the whole point", () => {
    expect(errors(set(q({ kind: "negative", gold: [] })))).toEqual([]);
  });

  test("a gold span means the answer IS in the corpus", () => {
    // False-positive rate would then be measured on a question that has a
    // right answer, which is the opposite of what it reports.
    expect(errors(set(q({ kind: "negative", gold: [span("aaaaaaaaaaa")] })))[0])
      .toMatch(/must have no gold spans/);
  });
});

describe("side outside contradiction", () => {
  test("is a warning, not an error", () => {
    const issues = validateStructure(set(q({ gold: [span("aaaaaaaaaaa", 10, 40, "pro")] })), corpus);
    expect(issues.filter((i) => i.level === "warning").map((i) => i.message))
      .toContain("side is only meaningful on contradiction questions");
  });
});

describe("lexicalOverlap", () => {
  test("a question copied from the passage scores high", () => {
    const passage = "the default compaction threshold is zero point five of the context window";
    expect(lexicalOverlap("what is the default compaction threshold", passage)).toBe(1);
  });

  test("a paraphrase scores low", () => {
    const passage = "the default compaction threshold is zero point five of the context window";
    expect(lexicalOverlap("when does an agent start dropping earlier messages", passage))
      .toBeLessThan(0.3);
  });

  test("stopwords do not carry a question over the line", () => {
    // "what is the" matching is meaningless; only content words count.
    expect(lexicalOverlap("what is the aardvark", "what is the banana")).toBe(0);
  });

  test("an empty question overlaps with nothing", () => {
    expect(lexicalOverlap("", "anything at all")).toBe(0);
  });
});

describe("countByKind", () => {
  test("counts every kind, including the ones with none", () => {
    expect(countByKind(set(
      q({ slug: "a" }),
      q({ slug: "b" }),
      q({ slug: "c", kind: "negative" }),
    ))).toEqual({ factual: 2, comparative: 0, contradiction: 0, negative: 1 });
  });
});

describe("goldenSha", () => {
  const base = set(
    q({ slug: "b", gold: [span("bbbbbbbbbbb", 10, 40)] }),
    q({ slug: "a", gold: [span("aaaaaaaaaaa", 10, 40)] }),
  );

  test("question order does not change the hash", () => {
    const reordered = set(...[...base.questions].reverse());
    expect(goldenSha(reordered)).toBe(goldenSha(base));
  });

  test("whitespace in the question text does not", () => {
    const respaced = set(...base.questions.map((x) => ({ ...x, text: `  ${x.text}\n  ` })));
    expect(goldenSha(respaced)).toBe(goldenSha(base));
  });

  test("notes and topics do not — they annotate, they do not measure", () => {
    const annotated = set(...base.questions.map((x) => ({ ...x, topics: ["cost"], tools: ["mcp"] })));
    expect(goldenSha(annotated)).toBe(goldenSha(base));
  });

  test("rephrasing a question does — it is a different question", () => {
    const edited = set(...base.questions.map((x) => ({ ...x, text: `${x.text} really` })));
    expect(goldenSha(edited)).not.toBe(goldenSha(base));
  });

  test("moving a gold span does", () => {
    const moved = set(...base.questions.map((x) => ({
      ...x, gold: x.gold.map((g) => ({ ...g, start_s: g.start_s + 5 })),
    })));
    expect(goldenSha(moved)).not.toBe(goldenSha(base));
  });

  test("adding a question does", () => {
    expect(goldenSha(set(...base.questions, q({ slug: "c", gold: [span("ccccccccccc", 10, 40)] }))))
      .not.toBe(goldenSha(base));
  });
});
