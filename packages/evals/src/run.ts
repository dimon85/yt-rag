// The pieces of a run that are worth testing without running one: the JSONL
// record shapes, the budget pre-flight, and the determinism check over
// repeats. eval.ts is the command; this is what it is made of.
//
// Where the chunk_set_id went. Invariants 1 and 9 are about a `chunks` table
// with a mandatory `chunk_set_id`, and a reader looking for it will look here.
// There is no database in this repo and this runner does not need one:
// everything is derived from cache/transcripts/*.json on each invocation, and
// a JSONL file is one cell of the matrix by construction — its header names
// the chunking, retrieval and reranking config, and every line in it belongs
// to that cell. The `cell` field IS the chunk_set identity for these runs.
// What the column buys in a shared table — being able to tell whose
// embeddings are whose after the fact — the filename and header buy here. If
// a `chunks` table ever lands, the mapping is chunk_set_id ↔ (chunking id,
// embedder), and nothing else in this file changes.
import { createHash } from "node:crypto";
import { z } from "zod";
import type { Question } from "./golden.ts";
import { HIT_COVERAGE } from "./hit.ts";
import {
  contradictionCoverage, falsePositiveRate, mrr, recallAtK, type Retrieved,
} from "./metrics.ts";

/** The cut-offs every run records, so the report never has to re-retrieve. */
export const KS = [1, 3, 5, 10] as const;

// ─── records ─────────────────────────────────────────────────────────────────

/**
 * The first line of every file.
 *
 * `golden_sha` and `git_sha` are here because invariant 6 turns on them: only
 * runs with identical values may be compared, and the report enforces that
 * rather than trusting whoever assembled the directory. `hit_coverage` is
 * recorded for the same reason — the rule in hit.ts is identical across
 * configurations today, and a file that was written under a different
 * threshold should be visible as such rather than silently averaged in.
 */
export const Header = z.object({
  type: z.literal("header"),
  cell: z.string(),
  /**
   * The cell's position in the expanded matrix.
   *
   * Recorded because the report reads a directory, and a directory is sorted
   * by filename — which puts fixed-1024 before fixed-128 and makes the size
   * trend, the one thing the chunking axis is there to show, unreadable. The
   * report must not import configs.yaml to recover the order: it would pull a
   * chunker into a package that is not allowed to have one.
   */
  order: z.number().int().nonnegative(),
  chunking: z.object({ id: z.string(), strategy: z.string(), params: z.record(z.number()) }),
  retrieval: z.object({ id: z.string(), kind: z.string() }),
  reranking: z.object({ id: z.string(), kind: z.string() }),
  /** null for a purely lexical cell, which embeds nothing. */
  embedder: z.string().nullable(),
  golden_sha: z.string(),
  git_sha: z.string(),
  chunk_count: z.number().int().nonnegative(),
  questions: z.number().int().nonnegative(),
  top_k: z.number().int().positive(),
  repeats: z.number().int().positive(),
  hit_coverage: z.number(),
  started_at: z.string(),
});
export type Header = z.infer<typeof Header>;

/** One retrieved span, as it is written down. Scores are kept, not thresholded. */
export const Span = z.object({
  video: z.string(),
  start_s: z.number(),
  end_s: z.number(),
  score: z.number(),
});

/**
 * One question, one repeat.
 *
 * Metrics are computed at write time and the spans are kept alongside them, so
 * the report can aggregate without a retriever and a disagreement between a
 * metric and the spans it came from is checkable after the fact. Nulls are
 * written as nulls: invariant 13 — a negative question has no recall and no
 * MRR, and writing 0 there would make the headline number depend on how many
 * negatives the set holds.
 */
export const Result = z.object({
  type: z.literal("result"),
  repeat: z.number().int().positive(),
  slug: z.string(),
  kind: z.string(),
  /** Set once per run from a plain BM25 pass; used to split the report, never to filter. */
  lexically_trivial: z.boolean().nullable(),
  spans: z.array(Span),
  recall: z.record(z.number().nullable()),
  mrr: z.number().nullable(),
  contradiction: z.record(z.boolean().nullable()),
  /** Top-1 score, the only input false-positive rate needs. */
  top1: z.number().nullable(),
});
export type Result = z.infer<typeof Result>;

/**
 * A cell that was not run, and why.
 *
 * Written as a line rather than left as a missing file. A blank in the table
 * reads as "we tried and got nothing", which is a worse error than an empty
 * cell — and a zero would read as a real result. The reason travels with it so
 * the table can say "not run: no budget" in the row's place.
 */
export const Skipped = z.object({
  type: z.literal("skipped"),
  reason: z.string(),
});
export type Skipped = z.infer<typeof Skipped>;

export const Line = z.discriminatedUnion("type", [Header, Result, Skipped]);
export type Line = z.infer<typeof Line>;

// ─── metrics for one question ────────────────────────────────────────────────

/**
 * Everything a result line records about one ranking.
 *
 * Deliberately not an average of anything: averaging is the report's job, and
 * a per-question line is what makes "31 both got, 13 neither, 4 moved"
 * computable later. The metric functions themselves live in metrics.ts and are
 * called, never reimplemented — the hit rule has one home (invariant 12) and
 * so does everything built on it.
 */
export function scoreQuestion(
  q: Question,
  ranked: Retrieved[],
  repeat: number,
  trivial: boolean | null,
): Result {
  return {
    type: "result",
    repeat,
    slug: q.slug,
    kind: q.kind,
    lexically_trivial: trivial,
    spans: ranked.map((r) => ({
      video: r.video,
      start_s: r.start_s,
      end_s: r.end_s,
      score: r.score,
    })),
    recall: Object.fromEntries(KS.map((k) => [k, recallAtK(q.gold, ranked, k)])),
    mrr: mrr(q.gold, ranked),
    contradiction: Object.fromEntries(
      [5, 10].map((k) => [k, contradictionCoverage(q.gold, ranked, k)]),
    ),
    top1: ranked[0]?.score ?? null,
  };
}

// ─── determinism ─────────────────────────────────────────────────────────────

/**
 * Whether the repeats of a cell agree, question by question.
 *
 * Three repeats are in configs.yaml to CHECK that retrieval is deterministic,
 * not to average noise out of it. So a disagreement is a finding to report,
 * and this returns the questions that disagreed rather than a mean over them.
 *
 * Compared on the ranked span list, not on the metrics: two different rankings
 * can score the same recall@5, and the thing being checked is whether the
 * retriever returned the same thing twice.
 */
export function repeatDisagreements(results: Result[]): string[] {
  const bySlug = new Map<string, Set<string>>();
  for (const r of results) {
    const shape = JSON.stringify(r.spans);
    const seen = bySlug.get(r.slug) ?? new Set<string>();
    seen.add(shape);
    bySlug.set(r.slug, seen);
  }
  return [...bySlug].filter(([, shapes]) => shapes.size > 1).map(([slug]) => slug).sort();
}

// ─── aggregation ─────────────────────────────────────────────────────────────

/**
 * Mean of a metric over the questions that carry it.
 *
 * Nulls are filtered, never coerced. Invariant 13: `recall@k` and MRR are
 * computed over questions that have gold spans, so a negative contributes to
 * neither the numerator nor the denominator. Returns null when nothing
 * carries the metric, because a mean of no questions is not zero.
 */
export function meanOf(values: (number | null)[]): number | null {
  const present = values.filter((v): v is number => v !== null);
  if (present.length === 0) return null;
  return present.reduce((a, b) => a + b, 0) / present.length;
}

/** k successes of n, for a proportion the report prints with an interval. */
export function tally(values: (boolean | null)[]): { k: number; n: number } {
  const present = values.filter((v): v is boolean => v !== null);
  return { k: present.filter(Boolean).length, n: present.length };
}

/**
 * The threshold false-positive rate is reported at: the median top-1 score of
 * the ANSWERABLE questions.
 *
 * Taken from the retriever's own score distribution rather than a fixed grid,
 * because the scales are not comparable — cosine lives in [0,1], BM25 is
 * unbounded, and fused RRF scores collapse onto a handful of values around
 * 0.03. A fixed grid printed "100% at every threshold" for BM25 on the first
 * attempt, its top-1 scores being off the end of it.
 *
 * "Median of answerable" means: a system tuned to answer half the questions it
 * can answer. What share of the unanswerable ones does it answer anyway.
 */
export function medianThreshold(answerableTop1: number[]): number | null {
  if (answerableTop1.length === 0) return null;
  const sorted = [...answerableTop1].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(0.5 * sorted.length))]!;
}

/**
 * Share of the negative questions the system answered anyway.
 *
 * Takes nulls, and this is the only null in the codebase that is neither
 * filtered nor coerced, so it is worth saying why both of those would be wrong.
 *
 * A null top-1 means the retriever returned nothing at all for that question.
 * On a negative that is the CORRECT outcome, and the question was still asked
 * — so it belongs in the denominator as a non-false-positive. Filtering it out
 * would shrink the denominator and delete a success, making a retriever that
 * abstains look worse for abstaining. Coercing it to 0 reaches the right
 * number, which is why the first version did, but it does so by inventing a
 * score for a retriever that produced none, and the next reader has to work
 * out whether 0 is a real score or a stand-in.
 *
 * So the abstentions are counted here and only the real scores reach
 * falsePositiveRate, which stays a function of scores alone.
 *
 * Measured on the run this was written against: 0 of 1,344 negative
 * question-cells abstained. Every retriever in the matrix returns k results
 * whatever the query, which is the finding falsePositiveRate exists to
 * expose — without an abstention rule the rate is 1 by construction.
 */
export function falsePositives(
  negativeTop1: (number | null)[],
  threshold: number | null,
): number | null {
  if (threshold === null || negativeTop1.length === 0) return null;
  const answered = negativeTop1.filter((s): s is number => s !== null);
  const rate = falsePositiveRate(answered, threshold);
  // Every negative abstained: nothing was falsely answered.
  if (rate === null) return 0;
  // rate * answered.length is the integer count of false positives; the
  // denominator is every negative asked, abstentions included.
  return (rate * answered.length) / negativeTop1.length;
}

// ─── paired comparison ───────────────────────────────────────────────────────

/**
 * The 2x2 of a paired comparison, at one metric and one cut-off.
 *
 * `both` and `neither` carry no information about the difference — this is why
 * the design is paired, and why power.ts says power is governed by how many
 * questions MOVE rather than by how many there are. `aOnly` and `bOnly` are
 * McNemar's b and c.
 */
export type Discordance = { both: number; neither: number; aOnly: number; bOnly: number };

/**
 * Pairs two configurations question by question on a binary outcome.
 *
 * Binary, not the fractional recall a multi-span question produces: McNemar is
 * a test on a 2x2 table and needs a yes/no. "Found at least one gold span
 * within k" is the yes/no used, and it is stated here rather than left to be
 * inferred, because fractional recall averaged and then compared is a
 * different — and unpaired — test.
 *
 * Only questions present in both, and carrying the metric in both, are
 * counted. A question one side skipped is not a disagreement.
 */
export function discordance(
  a: Map<string, boolean | null>,
  b: Map<string, boolean | null>,
): Discordance {
  const out: Discordance = { both: 0, neither: 0, aOnly: 0, bOnly: 0 };
  for (const [slug, av] of a) {
    const bv = b.get(slug);
    if (av === null || bv === null || bv === undefined) continue;
    if (av && bv) out.both++;
    else if (!av && !bv) out.neither++;
    else if (av) out.aOnly++;
    else out.bOnly++;
  }
  return out;
}

/** "Found something" as the paired outcome. Null stays null — see invariant 13. */
export const foundWithin = (r: Result, k: number): boolean | null => {
  const v = r.recall[String(k)];
  return v === null || v === undefined ? null : v > 0;
};

// ─── identity ────────────────────────────────────────────────────────────────

/**
 * A short digest of the chunk texts a configuration produced.
 *
 * Not used for comparability — golden_sha and git_sha are what invariant 6
 * turns on — but it is what distinguishes "the same code chunked a changed
 * corpus" from "the corpus is the same". Cheap to record, impossible to
 * reconstruct later.
 */
export function chunkSha(texts: string[]): string {
  const h = createHash("sha256");
  for (const t of texts) h.update(t).update(" ");
  return h.digest("hex").slice(0, 16);
}
