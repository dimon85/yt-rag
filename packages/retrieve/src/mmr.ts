// Maximal Marginal Relevance. Pure, and generic over what "similar" means, so
// the same code reranks lexical and vector results.
//
// Why it is a configuration here rather than a nicety. On the first real run,
// contradiction coverage@5 was 0 of 2 for every retriever and every chunk
// size: plain top-k returns five chunks arguing the same side of a
// disagreement, because they are all highly similar to the query and to each
// other. Recall scores that as a success. The metric that is supposed to
// distinguish this project cannot move until the ranking is allowed to trade
// some relevance for difference.
//
//   MMR = argmax over remaining d of
//           lambda * relevance(d) - (1 - lambda) * max similarity(d, already picked)
//
// lambda 1 is plain relevance ranking; lambda 0 ignores the query entirely and
// picks the most mutually different documents it can find — including for the
// first selection, where nothing has been picked yet and every candidate
// therefore scores 0. Useful lambdas sit between; 0 is documented rather than
// special-cased so the formula above stays literally true.

export type Candidate<T> = { item: T; relevance: number };

/**
 * Reranks candidates, trading relevance against redundancy.
 *
 * `similarity` must be symmetric and roughly comparable in scale to
 * `relevance`, or lambda stops meaning what it looks like it means. For
 * embeddings both are cosine; for BM25 the relevance scale is unbounded, so
 * the caller normalizes before calling.
 */
export function mmrRerank<T>(
  candidates: Candidate<T>[],
  similarity: (a: T, b: T) => number,
  opts: { lambda?: number; k?: number } = {},
): Candidate<T>[] {
  const lambda = opts.lambda ?? 0.5;
  const k = opts.k ?? candidates.length;
  if (lambda < 0 || lambda > 1) throw new Error(`lambda must be in [0,1], got ${lambda}`);
  if (candidates.length === 0) return [];

  // Index order is the tiebreak throughout, so the same input always produces
  // the same output. Three runs per configuration are planned, and a ranking
  // that disagrees with itself is indistinguishable from a real effect.
  const remaining = candidates.map((c, i) => ({ ...c, i }));
  const picked: (Candidate<T> & { i: number })[] = [];

  while (picked.length < Math.min(k, candidates.length)) {
    let best = -1;
    let bestScore = -Infinity;

    for (let j = 0; j < remaining.length; j++) {
      const c = remaining[j]!;
      const redundancy = picked.length === 0
        ? 0
        : Math.max(...picked.map((p) => similarity(c.item, p.item)));
      const score = lambda * c.relevance - (1 - lambda) * redundancy;
      if (score > bestScore || (score === bestScore && best !== -1 && c.i < remaining[best]!.i)) {
        bestScore = score;
        best = j;
      }
    }

    picked.push(remaining[best]!);
    remaining.splice(best, 1);
  }

  return picked.map(({ item, relevance }) => ({ item, relevance }));
}

/** Overlap of token sets. The cheap stand-in for similarity when there are no vectors. */
export function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let shared = 0;
  for (const x of a) if (b.has(x)) shared++;
  return shared / (a.size + b.size - shared);
}
