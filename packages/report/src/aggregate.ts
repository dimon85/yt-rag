// Reading JSONL and turning it into numbers. Pure functions over lines that
// are already on disk.
//
// The separation is the point, and it is a convention rather than tidiness:
// this package has no retriever, no embedder and no chunker, so there is no
// path by which a number in the report could have been produced by re-running
// something instead of by reading what a run recorded. It also means the
// numbers are auditable — every figure here traces to a line in a file whose
// header says which code and which question set produced it.
//
// Nothing in this file may import from packages/retrieve, packages/embed or
// packages/chunking. If a figure cannot be computed from the JSONL, the fix is
// to record it in the JSONL.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import {
  Header, Line, Result, Skipped, discordance, foundWithin, meanOf,
  medianThreshold, falsePositives, repeatDisagreements, tally,
  type Discordance,
} from "../../evals/src/run.ts";

export type CellRun = {
  file: string;
  header: Header;
  results: Result[];
  skipped: Skipped | null;
};

/**
 * One cell's file.
 *
 * A file with a header and no results is not an error — a skipped cell is
 * written exactly that way on purpose, so its reason can be printed in the
 * row's place instead of a blank.
 */
export function readCell(file: string): CellRun {
  const lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error(`${file} is empty`);

  const parsed = lines.map((l, i) => {
    try {
      return Line.parse(JSON.parse(l));
    } catch (e) {
      throw new Error(`${file} line ${i + 1}: ${(e as Error).message}`);
    }
  });

  const headers = parsed.filter((p) => p.type === "header");
  const header = parsed[0];
  if (header?.type !== "header") throw new Error(`${file} does not start with a header line`);
  if (headers.length > 1) {
    // Two headers means two runs wrote to one filename — eval.ts appends, so a
    // second run over the same --out directory extends the file instead of
    // replacing it. Taking the first header and every result line would then
    // average over a doubled set of questions and print a plausible number.
    // eval.ts refuses this up front; this is the check that cannot be
    // bypassed, including by concatenating two files by hand.
    throw new Error(
      `${file} holds ${headers.length} headers — two runs wrote to one filename. ` +
      `Its results cover the question set more than once, so aggregating them ` +
      `would average a doubled set. Split the file or re-run the cell with --force.`,
    );
  }

  return {
    file,
    header,
    results: parsed.filter((p): p is Result => p.type === "result"),
    skipped: parsed.find((p): p is Skipped => p.type === "skipped") ?? null,
  };
}

export function readDir(dir: string): CellRun[] {
  const files = readdirSync(dir)
    .filter((f) => f.endsWith(".jsonl"))
    .map((f) => join(dir, f))
    .filter((f) => statSync(f).isFile())
    .sort();
  if (files.length === 0) throw new Error(`no .jsonl files in ${dir}`);
  // Matrix order, not filename order. Sorted by name, fixed-1024 lands before
  // fixed-128 and the chunk-size trend the table exists to show is unreadable.
  return files.map(readCell).sort((a, b) => a.header.order - b.header.order);
}

// ─── invariant 6 ─────────────────────────────────────────────────────────────

export type Comparability = {
  golden_sha: string;
  git_sha: string;
  /** Files that do not match the majority pair, with what they carry instead. */
  rejected: { file: string; golden_sha: string; git_sha: string }[];
};

/**
 * Splits a directory into the comparable set and the rest.
 *
 * Invariant 6: only runs with identical `golden_sha` and `git_sha` may be
 * compared, and mixed comparisons go in the bin. Enforced here rather than
 * trusted to the operator, because the failure is invisible — a table mixing
 * two question sets has two denominators and still prints.
 *
 * The majority pair wins, and the minority is reported rather than silently
 * dropped: a directory holding one stale file is a different situation from
 * one holding two runs of equal size, and the operator has to be able to tell
 * which they have.
 */
export function comparable(cells: CellRun[]): { keep: CellRun[]; rejected: Comparability["rejected"] } {
  const counts = new Map<string, number>();
  for (const c of cells) {
    const key = `${c.header.golden_sha}|${c.header.git_sha}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  // Ties break on the key so the choice is reproducible rather than dependent
  // on directory order.
  const [winner] = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]!;

  const keep: CellRun[] = [];
  const rejected: Comparability["rejected"] = [];
  for (const c of cells) {
    if (`${c.header.golden_sha}|${c.header.git_sha}` === winner) keep.push(c);
    else {
      rejected.push({
        file: c.file,
        golden_sha: c.header.golden_sha,
        git_sha: c.header.git_sha,
      });
    }
  }
  return { keep, rejected };
}

// ─── one row of the table ────────────────────────────────────────────────────

export type Row = {
  label: string;
  cell: string;
  embedder: string | null;
  chunkCount: number;
  /** Set when the cell was not run; every metric below is then null. */
  notRun: string | null;
  recall: Record<number, number | null>;
  mrr: number | null;
  contradiction: { k: number; n: number };
  threshold: number | null;
  falsePositiveRate: number | null;
  negatives: number;
  /** recall@5 split by whether term matching alone answers the question. */
  split: { trivial: { mean: number | null; n: number }; harder: { mean: number | null; n: number } };
  /** Questions whose ranking differed between repeats. Empty is the expected case. */
  disagreements: string[];
  recallPool: number;
};

/**
 * A configuration is its cell plus, for a dense retriever, the embedder.
 *
 * The embedder is not an axis in configs.yaml — it names `vector` and `hybrid`
 * without naming a model — but two runs of `vector` under different models are
 * different configurations, and a table that collapsed them would average a
 * 384-dimension local model with a 1536-dimension hosted one.
 */
export const rowLabel = (h: Header) => (h.embedder ? `${h.cell} /${h.embedder}` : h.cell);

/**
 * Metrics for one cell, over the FIRST repeat only.
 *
 * Not a mean of the three. The repeats exist to check that retrieval is
 * deterministic, so averaging them would be either a no-op (when it is) or a
 * way of hiding a finding (when it is not). Where they disagree, the
 * disagreement is reported and the first repeat is what the row shows —
 * arbitrary, and stated as such, rather than an average nobody asked for.
 */
export function summarise(cell: CellRun): Row {
  const { header } = cell;
  const base = {
    label: rowLabel(header),
    cell: header.cell,
    embedder: header.embedder,
    chunkCount: header.chunk_count,
    negatives: 0,
    recallPool: 0,
  };

  if (cell.skipped || cell.results.length === 0) {
    return {
      ...base,
      notRun: cell.skipped?.reason ?? "not run: the file holds a header and no results",
      recall: {},
      mrr: null,
      contradiction: { k: 0, n: 0 },
      threshold: null,
      falsePositiveRate: null,
      split: { trivial: { mean: null, n: 0 }, harder: { mean: null, n: 0 } },
      disagreements: [],
    };
  }

  const first = cell.results.filter((r) => r.repeat === 1);
  const ks = [...new Set(first.flatMap((r) => Object.keys(r.recall)))].map(Number).sort((a, b) => a - b);

  // Invariant 13: the recall pool is the questions that carry the metric, and
  // negatives are identified by carrying no recall rather than by their kind
  // label — the label is metadata, the null is the fact.
  const answerable = first.filter((r) => r.recall[String(ks[0] ?? 1)] !== null);
  const negatives = first.filter((r) => r.mrr === null);

  const splitOn = (want: boolean) => {
    const subset = answerable.filter((r) => r.lexically_trivial === want);
    return { mean: meanOf(subset.map((r) => r.recall["5"] ?? null)), n: subset.length };
  };

  const threshold = medianThreshold(
    answerable.map((r) => r.top1).filter((s): s is number => s !== null),
  );

  return {
    ...base,
    notRun: null,
    recall: Object.fromEntries(ks.map((k) => [k, meanOf(first.map((r) => r.recall[String(k)] ?? null))])),
    mrr: meanOf(first.map((r) => r.mrr)),
    contradiction: tally(first.map((r) => r.contradiction["5"] ?? null)),
    threshold,
    falsePositiveRate: falsePositives(negatives.map((r) => r.top1), threshold),
    negatives: negatives.length,
    split: { trivial: splitOn(true), harder: splitOn(false) },
    disagreements: repeatDisagreements(cell.results),
    recallPool: answerable.length,
  };
}

// ─── paired comparisons ──────────────────────────────────────────────────────

export type Pairing = {
  a: string;
  b: string;
  k: number;
  counts: Discordance;
  /**
   * How many of the four axes — chunking, retrieval, reranking, embedder —
   * the two configurations differ in.
   *
   * 1 is an ablation: one thing changed and the rest held. Anything higher is
   * two changes at once, and attributing the difference to either of them is
   * not something a paired test can do. The printer leads with the 1s for
   * that reason, not because they are more significant.
   */
  axesDiffering: number;
};

/** chunking, retrieval, reranking, embedder — the four things a cell is. */
const axes = (c: CellRun) => [...c.header.cell.split("__"), c.header.embedder ?? "-"];

function axesDiffering(a: CellRun, b: CellRun): number {
  const x = axes(a), y = axes(b);
  return x.filter((v, i) => v !== y[i]).length;
}

/** "Found at least one gold span within k", per question, for the first repeat. */
export function outcomes(cell: CellRun, k: number): Map<string, boolean | null> {
  return new Map(
    cell.results.filter((r) => r.repeat === 1).map((r) => [r.slug, foundWithin(r, k)]),
  );
}

/**
 * Every pair of run cells, at one cut-off.
 *
 * All pairs rather than a chosen baseline: nothing in the repo compared two
 * runs before this, and which comparison matters is a question about the
 * result, not something to hard-code. The printer sorts and truncates.
 */
export function pairings(cells: CellRun[], k: number): Pairing[] {
  const live = cells.filter((c) => !c.skipped && c.results.length > 0);
  const out: Pairing[] = [];
  for (let i = 0; i < live.length; i++) {
    for (let j = i + 1; j < live.length; j++) {
      out.push({
        a: rowLabel(live[i]!.header),
        b: rowLabel(live[j]!.header),
        k,
        counts: discordance(outcomes(live[i]!, k), outcomes(live[j]!, k)),
        axesDiffering: axesDiffering(live[i]!, live[j]!),
      });
    }
  }
  return out;
}

// ─── per kind ────────────────────────────────────────────────────────────────

export type KindTally = { kind: string; k: number; n: number };

/**
 * Per-`kind` figures, as proportions.
 *
 * Invariant 14: descriptive, never comparative. 18 contradiction and 30
 * negative questions do not support "A beats B on contradictions", so this
 * returns one configuration's counts and the printer attaches a
 * Clopper-Pearson interval instead of a difference.
 *
 * The outcome is "found at least one gold span within k" for the kinds that
 * carry recall. Negatives carry none, so they are absent here and appear as
 * false-positive rate instead — the metric they exist for.
 */
export function byKind(cell: CellRun, k: number): KindTally[] {
  const first = cell.results.filter((r) => r.repeat === 1);
  const kinds = [...new Set(first.map((r) => r.kind))].sort();
  return kinds.flatMap((kind) => {
    const t = tally(first.filter((r) => r.kind === kind).map((r) => foundWithin(r, k)));
    return t.n === 0 ? [] : [{ kind, k: t.k, n: t.n }];
  });
}
