// A disk cache for the local embedding model, keyed the way GeminiEmbedder
// keys its own: sha256 of the text, so the cache survives rechunking that
// produces the same text.
//
// embedLocal has no cache, which was fine while it served one gate run over
// one chunk size. The ablation is a different shape: seven chunking
// configurations, and repeats on top. Without a cache the matrix re-embeds
// 10,777 chunk texts on CPU on every invocation, and the three repeats — which
// exist to VERIFY determinism, not to pay for it — would each pay again.
//
// Why the model's own cache is not enough: transformers.js caches the model
// weights, not the inference. The 90MB download happens once; the forward pass
// happens every time.
//
// One file, written whole, exactly as GeminiEmbedder does it — a run that dies
// partway keeps everything it embedded. The one deliberate difference is when
// it writes. Gemini batches 100 texts a request and had 33 requests to make,
// so writing after each one was free; this model batches 16, so 10,777 texts
// is 674 batches and 674 rewrites of an 80MB file. It writes every
// WRITE_EVERY new vectors instead.
//
// Vectors are stored at full precision. Rounding them would shrink the file by
// half and is exactly the wrong economy here: a repeat served from a rounded
// cache would score marginally differently from one that recomputed, and this
// project reads a disagreement between repeats as a finding about the
// retriever.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { embedLocal, LOCAL_DIM, LOCAL_MODEL } from "./local.ts";

export type LocalUsage = { embedded: number; cached: number };

/** New vectors between writes. Chosen so a crash loses seconds, not minutes. */
const WRITE_EVERY = 512;

export class CachedLocalEmbedder {
  private cachePath: string;
  private cache: Record<string, number[]>;
  readonly usage: LocalUsage = { embedded: 0, cached: 0 };

  constructor(cacheDir: string) {
    mkdirSync(cacheDir, { recursive: true });
    // The model name and width are in the filename because they are what the
    // vectors mean. A second local model must not read this file's vectors.
    this.cachePath = join(cacheDir, `${LOCAL_MODEL.replace(/\//g, "-")}-${LOCAL_DIM}.json`);
    this.cache = existsSync(this.cachePath) ? JSON.parse(readFileSync(this.cachePath, "utf8")) : {};
  }

  /** Same key function as GeminiEmbedder's, over the text and nothing else. */
  private key = (text: string) => createHash("sha256").update(text).digest("hex").slice(0, 24);

  /** How many of these are already on disk. Used by --dry-run; embeds nothing. */
  cachedCount(texts: string[]): number {
    return new Set(texts.filter((t) => this.cache[this.key(t)])).size;
  }

  async embedAll(
    texts: string[],
    opts: { onProgress?: (done: number, total: number) => void } = {},
  ): Promise<number[][]> {
    const missing = [...new Set(texts.filter((t) => !this.cache[this.key(t)]))];
    this.usage.cached += texts.length - missing.length;
    this.usage.embedded += missing.length;

    // Sliced here rather than relying on embedLocal's internal batching,
    // because a vector is only in hand once the call resolves: flushing every
    // WRITE_EVERY texts means calling it every WRITE_EVERY texts.
    for (let i = 0; i < missing.length; i += WRITE_EVERY) {
      const slice = missing.slice(i, i + WRITE_EVERY);
      const vectors = await embedLocal(slice, {
        onProgress: (done) => opts.onProgress?.(Math.min(i + done, missing.length), missing.length),
      });
      slice.forEach((t, j) => (this.cache[this.key(t)] = vectors[j]!));
      this.flush();
    }

    return texts.map((t) => this.cache[this.key(t)]!);
  }

  /** Written to a temporary name and renamed, so a kill mid-write cannot truncate it. */
  flush(): void {
    const tmp = `${this.cachePath}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.cache));
    renameSync(tmp, this.cachePath);
  }
}
