// BM25. Pure functions, no index on disk, no network — at this corpus size
// (roughly 950 chunks) a full scan is milliseconds, and an approximate index
// would add a variable to the experiment for no gain.
//
// It is here as the lexical baseline. The interesting comparison is not "does
// BM25 win" but whether embeddings beat it at all: if they do not, that is a
// finding about transcripts, not a failure of the setup.
//
// Note what it cannot do, given how the golden set was written. Questions were
// deliberately paraphrased rather than lifted from the passage, so BM25 is
// working against wording it has never seen. It measures the floor.

export type Doc = { id: number; text: string };

export type Bm25Index = {
  docs: { id: number; length: number; terms: Map<string, number> }[];
  df: Map<string, number>;
  avgLength: number;
  k1: number;
  b: number;
};

/** Lowercase, strip punctuation, split. The same normalization for both sides. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .split(" ")
    .filter((w) => w.length > 1);
}

export function buildIndex(docs: Doc[], k1 = 1.2, b = 0.75): Bm25Index {
  const prepared = docs.map((d) => {
    const words = tokenize(d.text);
    const terms = new Map<string, number>();
    for (const w of words) terms.set(w, (terms.get(w) ?? 0) + 1);
    return { id: d.id, length: words.length, terms };
  });

  const df = new Map<string, number>();
  for (const d of prepared) for (const t of d.terms.keys()) df.set(t, (df.get(t) ?? 0) + 1);

  const totalLength = prepared.reduce((n, d) => n + d.length, 0);
  return {
    docs: prepared,
    df,
    avgLength: prepared.length === 0 ? 0 : totalLength / prepared.length,
    k1,
    b,
  };
}

/**
 * Robertson/Sparck-Jones IDF with the +0.5 smoothing.
 *
 * Floored at zero: without the floor a term appearing in more than half the
 * documents scores negative, and a query containing one would push documents
 * that contain it *below* documents that do not.
 */
function idf(index: Bm25Index, term: string): number {
  const n = index.df.get(term) ?? 0;
  const N = index.docs.length;
  return Math.max(0, Math.log(1 + (N - n + 0.5) / (n + 0.5)));
}

export type Scored = { id: number; score: number };

export function search(index: Bm25Index, query: string, topK = 10): Scored[] {
  if (index.docs.length === 0) return [];
  const terms = tokenize(query);
  const scored: Scored[] = [];

  for (const d of index.docs) {
    let score = 0;
    for (const t of terms) {
      const tf = d.terms.get(t);
      if (!tf) continue;
      const norm = 1 - index.b + index.b * (d.length / (index.avgLength || 1));
      score += idf(index, t) * ((tf * (index.k1 + 1)) / (tf + index.k1 * norm));
    }
    if (score > 0) scored.push({ id: d.id, score });
  }

  // Ties broken by id so a run is reproducible. Retrieval is supposed to be
  // deterministic, and an unstable sort would make three runs disagree for
  // reasons that have nothing to do with the configuration.
  scored.sort((a, b) => b.score - a.score || a.id - b.id);
  return scored.slice(0, topK);
}
