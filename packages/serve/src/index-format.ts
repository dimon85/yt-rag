// The artifact the server runs on, and the one thing it needs to exist.
//
// Everything else in this repository derives what it needs from cache/ on each
// invocation — but cache/ is 300MB of vectors for seven chunking
// configurations and is not in git, so a container cannot be built from it.
// `pnpm serve:prepare` reduces it to one file holding exactly one
// configuration: its chunks, its vectors, and the video metadata a citation
// needs. The image copies that file and nothing else.
//
// Vectors are stored as float32 in base64 rather than as JSON numbers. The
// ablation keeps full precision on purpose — a repeat served from a rounded
// cache would score differently from one that recomputed — but that argument
// is about comparing a cell against itself across repeats, and this file is
// never compared against a run. What it buys is 1.6MB against 8MB, and the
// ranking it produces was checked against the float64 original before the
// format was chosen, not after.
import { z } from "zod";

export const ServeIndex = z.object({
  /** The configuration this was built from, so a served answer is traceable. */
  config: z.object({
    chunking: z.string(),
    retrieval: z.string(),
    embedder: z.string(),
    /** The commit the vectors were produced by. */
    git_sha: z.string(),
    built_at: z.string(),
  }),
  dim: z.number().int().positive(),
  /**
   * Below this fused score the server declines to answer.
   *
   * The median top-1 score of the answerable golden questions under this exact
   * configuration — the same threshold the report measures false positives at,
   * so the abstention behaviour in production is the behaviour the table
   * describes. A fixed grid would not work here: RRF scores collapse around
   * 0.03 and mean nothing on any other retriever's scale.
   */
  threshold: z.number(),
  chunks: z.array(z.object({
    video: z.string(),
    start_s: z.number(),
    end_s: z.number(),
    text: z.string(),
  })),
  /**
   * Title, channel and date, for the videos that have them.
   *
   * Keyed by video id and deliberately not required to cover every chunk: one
   * video in this corpus has no usable metadata file, and the citation the
   * spec asks for is the timestamp and the link. A missing title costs a
   * courtesy, not the attribution.
   */
  videos: z.record(z.string(), z.object({
    title: z.string(),
    channel: z.string(),
    published_at: z.string(),
  })),
  /** float32, little-endian, chunks.length × dim, base64. */
  vectors: z.string(),
});
export type ServeIndex = z.infer<typeof ServeIndex>;

export function packVectors(vectors: number[][]): string {
  const dim = vectors[0]?.length ?? 0;
  const flat = new Float32Array(vectors.length * dim);
  vectors.forEach((v, i) => flat.set(v, i * dim));
  return Buffer.from(flat.buffer).toString("base64");
}

export function unpackVectors(packed: string, dim: number): Float32Array[] {
  const buf = Buffer.from(packed, "base64");
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const out: Float32Array[] = [];
  for (let i = 0; i < flat.length; i += dim) out.push(flat.subarray(i, i + dim));
  return out;
}
