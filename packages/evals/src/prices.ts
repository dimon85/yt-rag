// prices.yaml as a typed object, and the arithmetic that turns recorded units
// into dollars.
//
// Nothing here imports a chunker, an embedder or a retriever, because
// packages/report has to be able to reach it: the report prices what a run
// wrote down, and `report/test/isolation.test.ts` walks its import graph to
// keep it from reaching anything that could retrieve instead of read.
//
// The schema's one real job is refusing a number with no provenance. A price
// is a claim about a vendor's tariff on a date, it ends up quoted in a README
// table beside measured recall, and a bare figure is indistinguishable there
// from one somebody guessed. So a price is either a number with a `source` and
// an `as_of`, or it is null with a `reason` — and null is a legitimate answer
// rather than missing data, because this project's honest embedding spend was
// a free-tier allowance counted in texts, not dollars.
import { readFileSync } from "node:fs";
import YAML from "yaml";
import { z } from "zod";

/**
 * A priced line, or an explicitly unpriced one.
 *
 * Two shapes rather than optional fields, so there is no way to express
 * "0.15, no idea where from". Both are `.strict()`: a typo'd `sources` should
 * fail here and not become a price with no provenance that reads as one with.
 */
export const EmbeddingPrice = z.union([
  z.object({
    usd_per_million_tokens: z.number().nonnegative(),
    as_of: z.string().min(1),
    source: z.string().min(1),
  }).strict(),
  z.object({
    usd_per_million_tokens: z.null(),
    reason: z.string().min(1),
  }).strict(),
]);

export const RerankPrice = z.union([
  z.object({
    usd_per_1000_searches: z.number().nonnegative(),
    as_of: z.string().min(1),
    source: z.string().min(1),
  }).strict(),
  z.object({
    usd_per_1000_searches: z.null(),
    reason: z.string().min(1),
  }).strict(),
]);

export const Prices = z.object({
  version: z.number().int().positive(),
  embedding: z.record(z.string(), EmbeddingPrice),
  rerank: z.record(z.string(), RerankPrice),
});
export type Prices = z.infer<typeof Prices>;

export function loadPrices(path: string): Prices {
  return Prices.parse(YAML.parse(readFileSync(path, "utf8")));
}

/** The dollar rate for an embedder, or null when the file says it is unpriced. */
export function embeddingRate(prices: Prices, embedder: string | null): number | null {
  if (embedder === null) return 0; // a lexical cell embeds nothing, and that is free, not unknown
  const row = prices.embedding[embedder];
  return row === undefined ? null : row.usd_per_million_tokens;
}

/** The dollar rate for a reranker, or null when unpriced. `none` is free. */
export function rerankRate(prices: Prices, reranker: string): number | null {
  if (reranker === "none" || reranker === "mmr") return 0; // local, no vendor
  const row = prices.rerank[reranker];
  return row === undefined ? null : row.usd_per_1000_searches;
}

/** Why a dollar figure could not be produced, in the file's own words. */
export function unpricedReason(prices: Prices, key: string): string | null {
  const row = prices.embedding[key] ?? prices.rerank[key];
  if (row === undefined) return `prices.yaml has no entry for "${key}"`;
  return "reason" in row ? row.reason.trim() : null;
}
