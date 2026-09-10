// Reciprocal Rank Fusion. Combines ranked lists by position, never by score.
//
// Why this project needs it, from the numbers rather than from the literature.
// At 128-token chunks Gemini reached two questions that neither BM25 nor the
// local model placed in fifty; BM25 reached one that Gemini never found at
// either chunk size. And the sizes disagree about who wins: BM25 leads at 512
// (56.5% recall@5 against 38.0%), Gemini leads at 128 (55.4% against 50.0%).
// Two retrievers failing on different questions is the precondition for
// fusion being worth anything, and it is measured here, not assumed.
//
// The one question Gemini never finds says why the two failure modes differ in
// kind. Its answer is a single sentence inside two minutes of unrelated setup
// talk: the sentence alone scores 0.759 against the question, which would rank
// it first, while the window containing it scores 0.623 — below the weakest
// genuine hit in the whole set. A dense vector averages its window, so one
// sentence in foreign company is invisible to it and trivial for term
// matching. No amount of tuning either retriever fixes that; it is an argument
// for having both.
//
//   RRF(d) = sum over lists of 1 / (k + rank(d))
//
// Position rather than score is the load-bearing choice. BM25 scores are
// unbounded and cosine lives in [0,1], so any weighted sum of the two is
// decided by scale before it is decided by relevance — and normalizing them
// onto a common scale means inventing an exchange rate between "matched a rare
// term" and "pointed the same way in embedding space". Ranks dodge the
// question: fusion only ever asks each retriever what it preferred, which is
// the one thing both are entitled to an opinion about.
//
// The cost is real, and it turned out larger than "throws away margin"
// suggests. A retriever certain about its first result and unsure about its
// second says so in its scores, and RRF cannot hear it. Measured against the
// 32 negative questions — ones whose answer is not in the corpus at all —
// fusion is the worst configuration tried, not merely a little worse:
//
//                       recall@5   false positives at p25 / p50
//   gemini, 128 tokens     55.4%              9%  /  0%
//   bm25,   128 tokens     50.0%             59%  / 13%
//   hybrid, 128 tokens     60.9%             47%  /  6%
//
// The mechanism is visible in the thresholds themselves. Fused scores are sums
// of a few reciprocals, so they take few distinct values: across the whole
// answerable set the p25 top-1 score is 0.031 and the median is 0.032. There
// is no room left between "found it" and "found nothing", and a threshold
// cannot be placed where the distribution has collapsed.
//
// So fusion buys recall and sells the ability to abstain. Which of those a
// retrieval stage should prefer is not a question this module can answer, and
// the honest use of it is to report both numbers rather than one.

export type Fused = { id: string; score: number };

/**
 * Fuses ranked lists of ids into one ranking.
 *
 * Each list is in the retriever's own order, best first. Ids appearing in more
 * than one list accumulate; ids appearing in only one still place, which is
 * the behaviour the measured asymmetry above depends on.
 *
 * `k` damps how much a single high placement is worth: at small k a first
 * place dominates, at large k the curve flattens and agreement deeper in the
 * lists still counts. 60 is the conventional default from the original paper.
 */
export function rrf(lists: string[][], opts: { k?: number } = {}): Fused[] {
  const k = opts.k ?? 60;
  if (!(k > 0)) throw new Error(`k must be positive, got ${k}`);

  const scores = new Map<string, number>();
  // First appearance across all lists, so ties break deterministically. Three
  // runs per configuration are planned, and a ranking that disagrees with
  // itself is indistinguishable from a real effect.
  const seen = new Map<string, number>();
  let order = 0;

  for (const list of lists) {
    // A duplicate inside one list would otherwise let that retriever vote
    // twice for the same document, at two different ranks.
    const counted = new Set<string>();
    list.forEach((id, i) => {
      if (counted.has(id)) return;
      counted.add(id);
      if (!seen.has(id)) seen.set(id, order++);
      scores.set(id, (scores.get(id) ?? 0) + 1 / (k + i + 1));
    });
  }

  return [...scores]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || seen.get(a.id)! - seen.get(b.id)!);
}
