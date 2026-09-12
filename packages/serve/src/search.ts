// Retrieval over a prepared index. Pure enough to test without a socket:
// index in, query in, spans out.
//
// This is the configuration the README's headline number describes —
// fixed-512-ov128, hybrid, local embeddings, 58.4% at recall@5 — and it is
// assembled from the same modules the ablation ran, not reimplemented. A
// server that retrieved differently from the harness would make every figure
// in the README a statement about something else.
import { buildIndex, search as bm25Search } from "../../retrieve/src/bm25.ts";
import { rrf } from "../../retrieve/src/fuse.ts";
import { unpackVectors, type ServeIndex } from "./index-format.ts";

/** How much of a chunk a public endpoint may return. See the legal note below. */
export const EXCERPT_CHARS = 320;

export type Hit = {
  video: string;
  title: string;
  channel: string;
  published_at: string;
  start_s: number;
  end_s: number;
  score: number;
  excerpt: string;
  url: string;
};

export type Answer =
  | { answered: true; hits: Hit[]; threshold: number }
  | { answered: false; reason: string; best: number | null; threshold: number };

export type Ready = {
  index: ServeIndex;
  lexical: ReturnType<typeof buildIndex>;
  vectors: Float32Array[];
};

export function prepare(index: ServeIndex): Ready {
  return {
    index,
    lexical: buildIndex(index.chunks.map((c, i) => ({ id: i, text: c.text }))),
    vectors: unpackVectors(index.vectors, index.dim),
  };
}

/**
 * A link that starts playing at the span, which is the citation the spec asks
 * for. Floored to the second because YouTube ignores fractions.
 */
export const sourceUrl = (video: string, startS: number) =>
  `https://www.youtube.com/watch?v=${video}&t=${Math.floor(startS)}s`;

/**
 * A short excerpt, never the paragraph.
 *
 * Transcripts belong to their authors. docs/spec.md is explicit that a public
 * endpoint returns short excerpts with a timestamp and a link, and this is
 * where that is enforced rather than left to a caller: the full chunk text is
 * in the index and never leaves this function intact. Cut on a word boundary
 * so the excerpt reads as a sentence fragment rather than a truncation.
 */
export function excerpt(text: string, limit = EXCERPT_CHARS): string {
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

const cosine = (a: Float32Array, b: number[]): number => {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
};

/**
 * Hybrid retrieval, fused by rank.
 *
 * RRF and not a weighted sum, for the reason the runner gives: BM25 is
 * unbounded and cosine lives in [0,1], so any weighted sum is decided by scale
 * before it is decided by relevance. The fused score is on neither scale,
 * which is why the abstention threshold is measured on it rather than picked.
 */
export function retrieve(ready: Ready, queryVector: number[], query: string, k: number): Answer {
  const { index, lexical, vectors } = ready;
  const pool = Math.max(k, 10);

  const lex = bm25Search(lexical, query, pool).map(({ id }) => String(id));
  const dense = vectors
    .map((v, i) => ({ id: i, score: cosine(v, queryVector) }))
    .sort((a, b) => b.score - a.score || a.id - b.id)
    .slice(0, pool)
    .map(({ id }) => String(id));

  const fused = rrf([lex, dense]).slice(0, k);
  const best = fused[0]?.score ?? null;

  if (best === null || best < index.threshold) {
    // Abstention is a feature here, not an error. A plain top-k retriever
    // always returns k results, so without this the false-positive rate is 1
    // by construction — and 32 of the golden questions have no answer in the
    // corpus on purpose.
    return {
      answered: false,
      reason: "nothing in this corpus scores above the threshold for that question",
      best,
      threshold: index.threshold,
    };
  }

  const hits = fused.map(({ id, score }) => {
    const chunk = index.chunks[Number(id)]!;
    const meta = index.videos[chunk.video];
    return {
      video: chunk.video,
      title: meta?.title ?? "",
      channel: meta?.channel ?? "",
      published_at: meta?.published_at ?? "",
      start_s: chunk.start_s,
      end_s: chunk.end_s,
      score,
      excerpt: excerpt(chunk.text),
      url: sourceUrl(chunk.video, chunk.start_s),
    };
  });

  return { answered: true, hits, threshold: index.threshold };
}
