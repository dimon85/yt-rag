import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { Header, Result } from "../../evals/src/run.ts";
import {
  byKind, comparable, outcomes, pairings, readCell, readDir, rowLabel, summarise,
  type CellRun,
} from "../src/aggregate.ts";

const header = (over: Partial<Header> = {}): Header => ({
  type: "header",
  cell: "fixed-128__bm25__none",
  order: 0,
  chunking: { id: "fixed-128", strategy: "fixed", params: { tokens: 128, overlap: 0 } },
  retrieval: { id: "bm25", kind: "lexical" },
  reranking: { id: "none", kind: "none" },
  embedder: null,
  golden_sha: "gold1",
  git_sha: "code1",
  chunk_count: 3238,
  questions: 4,
  top_k: 10,
  repeats: 1,
  hit_coverage: 0.5,
  started_at: "2026-09-10T00:00:00.000Z",
  ...over,
});

/** A result line with only the fields a given assertion needs. */
const result = (over: Partial<Result> & { slug: string }): Result => ({
  type: "result",
  repeat: 1,
  kind: "factual",
  lexically_trivial: false,
  spans: [],
  recall: { 1: 0, 3: 0, 5: 0, 10: 0 },
  mrr: 0,
  contradiction: { 5: null, 10: null },
  top1: 1,
  ...over,
});

const cell = (h: Partial<Header>, results: Result[], skipped: string | null = null): CellRun => ({
  file: `/tmp/${h.cell ?? "cell"}.jsonl`,
  header: header(h),
  results,
  skipped: skipped === null ? null : { type: "skipped", reason: skipped },
});

// ─── files ───────────────────────────────────────────────────────────────────

describe("readCell", () => {
  const write = (lines: object[]) => {
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-report-"));
    const file = join(dir, "a.jsonl");
    writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
    return { dir, file };
  };

  test("reads a header and its results", () => {
    const { file } = write([header(), result({ slug: "one" }), result({ slug: "two" })]);
    const read = readCell(file);
    expect(read.header.cell).toBe("fixed-128__bm25__none");
    expect(read.results.map((r) => r.slug)).toEqual(["one", "two"]);
    expect(read.skipped).toBeNull();
  });

  test("a header with a skip line and no results is valid, not an error", () => {
    const { file } = write([header(), { type: "skipped", reason: "not run: no budget" }]);
    const read = readCell(file);
    expect(read.results).toEqual([]);
    expect(read.skipped?.reason).toBe("not run: no budget");
  });

  test("a file that does not start with a header is rejected", () => {
    const { file } = write([result({ slug: "one" }), header()]);
    expect(() => readCell(file)).toThrow(/does not start with a header/);
  });

  test("a malformed line names the file and the line number", () => {
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-report-"));
    const file = join(dir, "a.jsonl");
    writeFileSync(file, `${JSON.stringify(header())}\n{"type":"result"}\n`);
    expect(() => readCell(file)).toThrow(/line 2/);
  });

  test("readDir orders by matrix position, not by filename", () => {
    // Sorted by name, fixed-1024 comes before fixed-128 and the chunk-size
    // trend the table exists to show reads backwards.
    const dir = mkdtempSync(join(tmpdir(), "yt-rag-report-"));
    for (const [name, order] of [["fixed-1024.jsonl", 4], ["fixed-128.jsonl", 0]] as const) {
      writeFileSync(join(dir, name), `${JSON.stringify(header({ cell: name, order }))}\n`);
    }
    expect(readDir(dir).map((c) => c.header.order)).toEqual([0, 4]);
  });

  test("two headers in one file is refused, not silently merged", () => {
    // eval.ts appends, so a second run over the same --out directory extends
    // the file. Taking the first header and every result line would average
    // over a doubled question set and print a plausible number.
    const { file } = write([
      header(), result({ slug: "one" }),
      header(), result({ slug: "one" }),
    ]);
    expect(() => readCell(file)).toThrow(/2 headers/);
  });

  test("an empty directory is an error, not an empty report", () => {
    expect(() => readDir(mkdtempSync(join(tmpdir(), "yt-rag-report-")))).toThrow(/no .jsonl files/);
  });
});

// ─── invariant 6 ─────────────────────────────────────────────────────────────

describe("comparable", () => {
  test("a file from another question set is excluded, not averaged in", () => {
    const good = [
      cell({ cell: "a" }, [result({ slug: "q" })]),
      cell({ cell: "b" }, [result({ slug: "q" })]),
    ];
    const stale = cell({ cell: "c", golden_sha: "gold2" }, [result({ slug: "q" })]);
    const { keep, rejected } = comparable([...good, stale]);
    expect(keep.map((c) => c.header.cell)).toEqual(["a", "b"]);
    expect(rejected).toHaveLength(1);
    expect(rejected[0]!.golden_sha).toBe("gold2");
  });

  test("a file from other code is excluded too", () => {
    const { keep, rejected } = comparable([
      cell({ cell: "a" }, [result({ slug: "q" })]),
      cell({ cell: "b" }, [result({ slug: "q" })]),
      cell({ cell: "c", git_sha: "code2" }, [result({ slug: "q" })]),
    ]);
    expect(keep).toHaveLength(2);
    expect(rejected[0]!.git_sha).toBe("code2");
  });

  test("a uniform directory rejects nothing", () => {
    const { keep, rejected } = comparable([cell({ cell: "a" }, []), cell({ cell: "b" }, [])]);
    expect(keep).toHaveLength(2);
    expect(rejected).toEqual([]);
  });

  test("the tie between two equal groups breaks reproducibly, not on file order", () => {
    const one = cell({ cell: "a", git_sha: "aaa" }, []);
    const two = cell({ cell: "b", git_sha: "zzz" }, []);
    expect(comparable([one, two]).keep.map((c) => c.header.git_sha))
      .toEqual(comparable([two, one]).keep.map((c) => c.header.git_sha));
  });
});

// ─── rows ────────────────────────────────────────────────────────────────────

describe("summarise", () => {
  test("hand-computed means over the answerable questions", () => {
    // Three answerable at recall@5 of 1, 0 and 0.5 → 0.5. One negative,
    // contributing to neither numerator nor denominator (invariant 13).
    const c = cell({}, [
      result({ slug: "a", recall: { 1: 1, 5: 1 }, mrr: 1, top1: 0.9 }),
      result({ slug: "b", recall: { 1: 0, 5: 0 }, mrr: 0, top1: 0.4 }),
      result({ slug: "c", recall: { 1: 0, 5: 0.5 }, mrr: 0.5, top1: 0.6 }),
      result({ slug: "n", kind: "negative", recall: { 1: null, 5: null }, mrr: null, top1: 0.7 }),
    ]);
    const row = summarise(c);
    expect(row.recall[5]).toBeCloseTo(0.5, 10);
    expect(row.recall[1]).toBeCloseTo(1 / 3, 10);
    expect(row.mrr).toBeCloseTo(0.5, 10);
    expect(row.recallPool).toBe(3);
    expect(row.negatives).toBe(1);
  });

  test("the threshold is the median top-1 of answerable, and FP is measured at it", () => {
    // Answerable top-1: 0.4, 0.6, 0.9 → median 0.6. The one negative scores
    // 0.7, which is above it, so the rate is 1 of 1.
    const row = summarise(cell({}, [
      result({ slug: "a", recall: { 1: 1, 5: 1 }, mrr: 1, top1: 0.9 }),
      result({ slug: "b", recall: { 1: 0, 5: 0 }, mrr: 0, top1: 0.4 }),
      result({ slug: "c", recall: { 1: 0, 5: 0 }, mrr: 0, top1: 0.6 }),
      result({ slug: "n", kind: "negative", recall: { 1: null, 5: null }, mrr: null, top1: 0.7 }),
    ]));
    expect(row.threshold).toBe(0.6);
    expect(row.falsePositiveRate).toBe(1);
  });

  test("a negative the retriever abstained on stays in the denominator", () => {
    // A null top-1 means nothing was returned, which on a negative is the
    // CORRECT outcome. Filtering it out would delete a success and make an
    // abstaining retriever look worse for abstaining; the question was still
    // asked, so it counts. Two negatives, one answered above the threshold:
    // 1 of 2, not 1 of 1.
    const row = summarise(cell({}, [
      result({ slug: "a", recall: { 1: 1, 5: 1 }, mrr: 1, top1: 0.5 }),
      result({ slug: "n1", kind: "negative", recall: { 1: null, 5: null }, mrr: null, top1: 0.9 }),
      result({ slug: "n2", kind: "negative", recall: { 1: null, 5: null }, mrr: null, top1: null }),
    ]));
    expect(row.threshold).toBe(0.5);
    expect(row.negatives).toBe(2);
    expect(row.falsePositiveRate).toBe(0.5);
  });

  test("a retriever that abstained on every negative scores no false positives", () => {
    const row = summarise(cell({}, [
      result({ slug: "a", recall: { 1: 1, 5: 1 }, mrr: 1, top1: 0.5 }),
      result({ slug: "n", kind: "negative", recall: { 1: null, 5: null }, mrr: null, top1: null }),
    ]));
    expect(row.falsePositiveRate).toBe(0);
  });

  test("a skipped cell carries its reason and no metrics", () => {
    const row = summarise(cell({}, [], "not run: no budget — 1618 of 1619 chunk vectors"));
    expect(row.notRun).toMatch(/no budget/);
    expect(row.recall).toEqual({});
    expect(row.mrr).toBeNull();
    expect(row.falsePositiveRate).toBeNull();
  });

  test("a header with no results and no skip line is still not a zero", () => {
    expect(summarise(cell({}, [])).notRun).toMatch(/not run/);
  });

  test("later repeats do not move the row", () => {
    // The row shows the first repeat. A second repeat scoring 0 must not pull
    // it to 0.5 — averaging the repeats is how a determinism failure gets
    // hidden instead of reported. The spans differ too, because determinism is
    // judged on the ranking rather than on the score it happened to produce.
    const span = { video: "aaaaaaaaaaa", start_s: 0, end_s: 10, score: 1 };
    const row = summarise(cell({}, [
      result({ slug: "a", spans: [span], recall: { 1: 1, 5: 1 }, mrr: 1 }),
      result({
        slug: "a", repeat: 2, spans: [{ ...span, start_s: 50, end_s: 60 }],
        recall: { 1: 0, 5: 0 }, mrr: 0, top1: 2,
      }),
    ]));
    expect(row.recall[5]).toBe(1);
    expect(row.disagreements).toEqual(["a"]);
  });

  test("the split reports both halves with their own denominators", () => {
    const row = summarise(cell({}, [
      result({ slug: "t1", lexically_trivial: true, recall: { 1: 1, 5: 1 }, mrr: 1 }),
      result({ slug: "t2", lexically_trivial: true, recall: { 1: 0, 5: 1 }, mrr: 1 }),
      result({ slug: "h1", lexically_trivial: false, recall: { 1: 0, 5: 0 }, mrr: 0 }),
    ]));
    expect(row.split.trivial).toEqual({ mean: 1, n: 2 });
    expect(row.split.harder).toEqual({ mean: 0, n: 1 });
  });

  test("contradiction coverage counts only the questions that carry both sides", () => {
    const row = summarise(cell({}, [
      result({ slug: "c1", contradiction: { 5: true, 10: true } }),
      result({ slug: "c2", contradiction: { 5: false, 10: true } }),
      result({ slug: "f1", contradiction: { 5: null, 10: null } }),
    ]));
    expect(row.contradiction).toEqual({ k: 1, n: 2 });
  });
});

describe("rowLabel", () => {
  test("a dense row names its embedder, a lexical one has none to name", () => {
    expect(rowLabel(header({ cell: "c__vector__none", embedder: "local" })))
      .toBe("c__vector__none /local");
    expect(rowLabel(header({ cell: "c__bm25__none" }))).toBe("c__bm25__none");
  });
});

// ─── pairings ────────────────────────────────────────────────────────────────

describe("pairings", () => {
  const a = cell({ cell: "a" }, [
    result({ slug: "w", recall: { 5: 1 } }),
    result({ slug: "x", recall: { 5: 1 } }),
    result({ slug: "y", recall: { 5: 0 } }),
    result({ slug: "z", recall: { 5: 0 } }),
  ]);
  const b = cell({ cell: "b", order: 1 }, [
    result({ slug: "w", recall: { 5: 1 } }),
    result({ slug: "x", recall: { 5: 0 } }),
    result({ slug: "y", recall: { 5: 1 } }),
    result({ slug: "z", recall: { 5: 0 } }),
  ]);

  test("hand-computed 2x2: 1 both, 1 neither, 1 each way", () => {
    expect(pairings([a, b], 5)).toEqual([
      { a: "a", b: "b", k: 5, counts: { both: 1, neither: 1, aOnly: 1, bOnly: 1 }, axesDiffering: 1 },
    ]);
  });

  test("three cells give three pairs", () => {
    expect(pairings([a, b, cell({ cell: "c", order: 2 }, a.results)], 5)).toHaveLength(3);
  });

  test("a skipped cell is not paired with anything", () => {
    expect(pairings([a, b, cell({ cell: "s", order: 3 }, [], "not run: no budget")], 5))
      .toHaveLength(1);
  });

  test("axesDiffering counts the changes, so one-axis ablations are findable", () => {
    // A pair differing in both chunking and retriever is discordant on plenty
    // of questions and says nothing about either change.
    const one = cell({ cell: "fixed-128__bm25__none", order: 0 }, [result({ slug: "w" })]);
    const retrieverOnly = cell({ cell: "fixed-128__vector__none", order: 1, embedder: "local" }, [result({ slug: "w" })]);
    const chunkingOnly = cell({ cell: "fixed-512__bm25__none", order: 2 }, [result({ slug: "w" })]);
    const both = cell({ cell: "fixed-512__vector__none", order: 3, embedder: "local" }, [result({ slug: "w" })]);

    const diff = (x: CellRun, y: CellRun) => pairings([x, y], 5)[0]!.axesDiffering;
    // The retriever change also changes the embedder from none to local, so it
    // is two axes by construction — dense retrieval cannot happen without one.
    expect(diff(one, retrieverOnly)).toBe(2);
    expect(diff(one, chunkingOnly)).toBe(1);
    expect(diff(retrieverOnly, both)).toBe(1);
    expect(diff(one, both)).toBe(3);
  });

  test("fractional recall becomes found / not found", () => {
    // A comparative question at recall@5 = 0.5 counts as found, because
    // McNemar needs a yes/no and "found at least one" is the one used.
    const half = cell({ cell: "h", order: 9 }, [result({ slug: "w", recall: { 5: 0.5 } })]);
    const none = cell({ cell: "n", order: 10 }, [result({ slug: "w", recall: { 5: 0 } })]);
    expect(pairings([half, none], 5)[0]!.counts).toEqual({
      both: 0, neither: 0, aOnly: 1, bOnly: 0,
    });
  });
});

describe("outcomes", () => {
  test("a negative question is null, so it never becomes a discordant pair", () => {
    const c = cell({}, [
      result({ slug: "q", recall: { 5: 1 } }),
      result({ slug: "n", recall: { 5: null } }),
    ]);
    expect(outcomes(c, 5)).toEqual(new Map([["q", true], ["n", null]]));
  });
});

// ─── per kind ────────────────────────────────────────────────────────────────

describe("byKind", () => {
  test("one tally per kind that carries recall", () => {
    const c = cell({}, [
      result({ slug: "f1", kind: "factual", recall: { 5: 1 } }),
      result({ slug: "f2", kind: "factual", recall: { 5: 0 } }),
      result({ slug: "c1", kind: "contradiction", recall: { 5: 1 } }),
    ]);
    expect(byKind(c, 5)).toEqual([
      { kind: "contradiction", k: 1, n: 1 },
      { kind: "factual", k: 1, n: 2 },
    ]);
  });

  test("negatives are absent rather than counted as 0 of n", () => {
    // They carry no recall, so there is nothing to be a proportion of. They
    // are the false-positive column of the main table instead.
    const c = cell({}, [
      result({ slug: "f", kind: "factual", recall: { 5: 1 } }),
      result({ slug: "n", kind: "negative", recall: { 5: null } }),
    ]);
    expect(byKind(c, 5).map((t) => t.kind)).toEqual(["factual"]);
  });
});
