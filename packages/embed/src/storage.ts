// Pads model vectors up to the storage width, and compares them.
//
// chunks.embedding is vector(1536): a storage width, not a model dimension.
// The model's real width lives in chunk_sets.dim and must be <= 1536.
// Everything narrower is zero-padded up here and nowhere else.
//
// Padding preserves cosine distance exactly. cos(a,b) = a·b / (|a|·|b|), and
// appending zeros to the tail of both vectors changes neither the dot product
// nor either norm. Vectors from different models are never compared, because
// every query filters on chunk_set_id.

export const STORAGE_DIM = 1536;

/**
 * Pads a model vector up to the storage width. Chunks and query vectors must
 * both go through this — a query padded differently silently ruins recall.
 */
export function toStorage(vec: number[], expectedDim: number): number[] {
  if (vec.length !== expectedDim) {
    throw new Error(`model returned ${vec.length}, chunk_set says ${expectedDim}`);
  }
  if (expectedDim > STORAGE_DIM) {
    throw new Error(`dim ${expectedDim} exceeds storage width ${STORAGE_DIM}`);
  }
  return [...vec, ...new Array(STORAGE_DIM - expectedDim).fill(0)];
}

/** pg hands `vector` over as a string in both directions. */
export const toPgVector = (v: number[]) => `[${v.join(",")}]`;
export const fromPgVector = (s: string): number[] => JSON.parse(s);

export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

export function norm(v: number[]): number {
  return Math.sqrt(dot(v, v));
}

/** Cosine similarity. Zero for a zero vector rather than NaN. */
export function cosine(a: number[], b: number[]): number {
  const denom = norm(a) * norm(b);
  return denom === 0 ? 0 : dot(a, b) / denom;
}
