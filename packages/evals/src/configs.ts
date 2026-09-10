// configs.yaml as a typed object, and the ablation matrix it expands into.
//
// Until this file existed, configs.yaml was read by nothing: the six chunking
// rows, three retrievers and three reranking settings lived in the repo as
// prose that the docs quoted and no code consulted. `pnpm ceiling` takes its
// chunk size from a command-line flag instead, which is right for a gate on
// one configuration and is not the ablation.
//
// Schema and validation live in one Zod object, as corpus.ts and golden.ts do
// it, so the config cannot be loaded with validation skipped. The chunking
// list is a DISCRIMINATED UNION on `strategy`, which is the load-bearing part:
// an unknown strategy, or a `fixed` row carrying `seconds`, has to fail here
// and not two hours into a run. `sentence` is in docs/spec.md as configuration
// #6 and deliberately absent from the union — it needs punctuation restoration
// that does not exist yet, and a typo'd `sentance` row should be rejected for
// the same reason a real one would be.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";
import { z } from "zod";
import { fixedChunks, type Chunk, type Segment } from "../../chunking/src/fixed.ts";
import { windowChunks } from "../../chunking/src/window.ts";
// Relative across packages on purpose — the project runs TypeScript directly
// via type stripping, so a package-name import has no resolvable entry point.
import { ROOT } from "../../ingest/src/corpus.ts";

export const CONFIGS_PATH = join(ROOT, "configs.yaml");

// ─── chunking ────────────────────────────────────────────────────────────────

const FixedChunking = z.object({
  id: z.string().min(1),
  strategy: z.literal("fixed"),
  params: z
    .object({
      tokens: z.number().int().positive(),
      overlap: z.number().int().nonnegative().default(0),
    })
    .refine((p) => p.overlap < p.tokens, {
      message: "overlap must be below tokens, or the walk never advances",
    }),
  tests: z.string().optional(),
});

const WindowChunking = z.object({
  id: z.string().min(1),
  strategy: z.literal("time_window"),
  params: z
    .object({
      seconds: z.number().positive(),
      overlap_s: z.number().nonnegative().default(0),
    })
    .refine((p) => p.overlap_s < p.seconds, {
      message: "overlap_s must be below seconds, or the walk never advances",
    }),
  tests: z.string().optional(),
});

export const Chunking = z.discriminatedUnion("strategy", [FixedChunking, WindowChunking]);
export type Chunking = z.infer<typeof Chunking>;

/**
 * The chunker a config names. Exhaustive over the union, so adding a strategy
 * to the schema without wiring it up is a type error rather than a run that
 * silently chunks two of six configurations the same way.
 */
export function chunkerFor(config: Chunking): (segments: Segment[]) => Chunk[] {
  switch (config.strategy) {
    case "fixed":
      return (segments) => fixedChunks(segments, config.params.tokens, config.params.overlap);
    case "time_window":
      return (segments) => windowChunks(segments, config.params.seconds, config.params.overlap_s);
  }
}

// ─── retrieval ───────────────────────────────────────────────────────────────

/**
 * `kind` rather than `id` carries the meaning. The ids in configs.yaml are
 * `bm25`, `vector` and `hybrid`; the kinds say which of them need embeddings,
 * which is the question the dry-run has to answer before spending anything.
 */
export const Retrieval = z.object({
  id: z.string().min(1),
  kind: z.enum(["lexical", "dense", "lexical+dense"]),
  params: z.object({ fusion: z.literal("rrf").optional() }).default({}),
  tests: z.string().optional(),
});
export type Retrieval = z.infer<typeof Retrieval>;

export const needsEmbeddings = (r: Retrieval) => r.kind !== "lexical";

// ─── reranking ───────────────────────────────────────────────────────────────

const NoRerank = z.object({
  id: z.string().min(1),
  kind: z.literal("none"),
  note: z.string().optional(),
});

const MmrRerank = z.object({
  id: z.string().min(1),
  kind: z.literal("mmr"),
  params: z.object({ lambda: z.number().min(0).max(1) }),
  note: z.string().optional(),
});

const ApiRerank = z.object({
  id: z.string().min(1),
  kind: z.literal("api"),
  params: z.object({ model: z.string().optional() }).default({}),
  note: z.string().optional(),
});

/**
 * The `none` row in configs.yaml is written as bare `- id: none`, with no
 * `kind`. Filling it in before the union sees it keeps the file readable and
 * still lets the discriminant do its job on the rows that have one — a
 * `kind: mmr` row missing its lambda is rejected here.
 */
export const Reranking = z.preprocess(
  (v) => (v !== null && typeof v === "object" && !("kind" in v) ? { ...v, kind: "none" } : v),
  z.discriminatedUnion("kind", [NoRerank, MmrRerank, ApiRerank]),
);
export type Reranking = z.infer<typeof Reranking>;

// ─── the file ────────────────────────────────────────────────────────────────

function uniqueIds<T extends { id: string }>(rows: T[], ctx: z.RefinementCtx) {
  const seen = new Set<string>();
  for (const [i, row] of rows.entries()) {
    if (seen.has(row.id)) {
      ctx.addIssue({
        code: "custom",
        // Zod prefixes the array's own key, so the index alone locates the row.
        path: [i, "id"],
        // Run output is keyed by id: a duplicate would make two configurations
        // write to the same file and the report average them together.
        message: `duplicate id ${row.id} — run output is keyed by it`,
      });
    }
    seen.add(row.id);
  }
}

export const AblationConfig = z.object({
  version: z.number().int().positive(),
  defaults: z.object({
    // Invariant 2: one tokenizer for every configuration. A second value here
    // would silently invalidate every comparison in the table, so the only
    // accepted value is the one the chunkers actually use.
    tokenizer: z.literal("gpt-tokenizer"),
    top_k: z.number().int().positive().default(10),
    repeats: z.number().int().positive().default(3),
  }),
  chunking: z.array(Chunking).min(1).superRefine(uniqueIds),
  retrieval: z.array(Retrieval).min(1).superRefine(uniqueIds),
  reranking: z.array(Reranking).min(1).superRefine(uniqueIds),
});
export type AblationConfig = z.infer<typeof AblationConfig>;

export function loadConfigs(path = CONFIGS_PATH): AblationConfig {
  return AblationConfig.parse(YAML.parse(readFileSync(path, "utf8")));
}

// ─── the matrix ──────────────────────────────────────────────────────────────

export type Cell = {
  chunking: Chunking;
  retrieval: Retrieval;
  reranking: Reranking;
};

/** How a cell is named in a filename, a header and a table row. */
export const cellId = (c: Cell) => `${c.chunking.id}__${c.retrieval.id}__${c.reranking.id}`;

export type Selection = {
  chunking?: string[];
  retrieval?: string[];
  reranking?: string[];
};

/**
 * chunking × retrieval × reranking, in the order the file lists them.
 *
 * Order is part of the output: the JSONL filenames and the report's row order
 * come from here, and a run that dies partway should have done the same cells
 * as the last one that did.
 *
 * An id in a filter that matches nothing throws rather than quietly narrowing
 * the matrix — a typo'd `--chunking fixed-129` that ran zero cells and exited
 * 0 is the failure that looks like success.
 */
export function expandMatrix(cfg: AblationConfig, select: Selection = {}): Cell[] {
  const pick = <T extends { id: string }>(rows: T[], want: string[] | undefined, axis: string) => {
    if (!want || want.length === 0) return rows;
    const byId = new Map(rows.map((r) => [r.id, r]));
    return want.map((id) => {
      const row = byId.get(id);
      if (!row) {
        throw new Error(
          `no ${axis} config "${id}" in configs.yaml — have ${rows.map((r) => r.id).join(", ")}`,
        );
      }
      return row;
    });
  };

  const cells: Cell[] = [];
  for (const chunking of pick(cfg.chunking, select.chunking, "chunking")) {
    for (const retrieval of pick(cfg.retrieval, select.retrieval, "retrieval")) {
      for (const reranking of pick(cfg.reranking, select.reranking, "reranking")) {
        cells.push({ chunking, retrieval, reranking });
      }
    }
  }
  return cells;
}
