// Local embeddings through transformers.js. No API key, no network after the
// model is cached, no per-token cost.
//
// This is the model for the ceiling check, not for the results table. It
// answers one question — can this corpus separate one retrieval configuration
// from another, or is the baseline already against the ceiling — and that
// question has to be answered before the golden set is worth finishing.
//
// The width is 384, which is the case the storage decision was made for:
// chunks.embedding is vector(1536) as a storage width, and anything narrower
// is zero-padded up to it. Padding preserves cosine distance exactly.
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

/** 384 dimensions, ~90MB, cached under ~/.cache/huggingface after first use. */
export const LOCAL_MODEL = "Xenova/all-MiniLM-L6-v2";
export const LOCAL_DIM = 384;

let extractor: FeatureExtractionPipeline | null = null;

async function load(model = LOCAL_MODEL): Promise<FeatureExtractionPipeline> {
  extractor ??= await pipeline("feature-extraction", model);
  return extractor;
}

/**
 * Embeds texts in batches, returning unit-length vectors.
 *
 * Normalized here rather than at query time so cosine similarity is a plain
 * dot product downstream, and so a caller cannot forget: an unnormalized
 * vector makes longer chunks score higher for reasons unrelated to meaning.
 */
export async function embedLocal(
  texts: string[],
  opts: { batchSize?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const model = await load();
  const batchSize = opts.batchSize ?? 16;
  const out: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const result = await model(batch, { pooling: "mean", normalize: true });
    const flat = result.data as Float32Array;
    const dim = flat.length / batch.length;
    for (let b = 0; b < batch.length; b++) {
      out.push(Array.from(flat.slice(b * dim, (b + 1) * dim)));
    }
    opts.onProgress?.(Math.min(i + batchSize, texts.length), texts.length);
  }
  return out;
}
