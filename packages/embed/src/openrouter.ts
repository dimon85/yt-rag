// Embeddings through OpenRouter's OpenAI-compatible endpoint.
//
// The third embedder, and the one that makes the embedding axis of the
// ablation runnable at all. The hosted model the design named is Gemini, and
// Gemini's free tier spends its allowance per TEXT against a daily cap of
// roughly a thousand — this corpus is 3,238 chunks at 128 tokens, so that axis
// was never going to run on it, at any schedule. Here the same work is billed
// per token: the whole matrix, 3.2M tokens across seven chunking
// configurations, costs about three cents through `baai/bge-m3`.
//
// Four things were measured against the live endpoint before this was written,
// because each one decides something in it.
//
// The response is OpenAI-shaped — `{data: [{index, embedding}], usage}` — and
// `index` comes back in request order. It is still read by `index` rather than
// by position, because a client that assumes order and is wrong produces
// plausible vectors attached to the wrong texts, which no test downstream can
// see.
//
// Vectors arrive unit length already. They are renormalized anyway, for the
// reason GeminiEmbedder gives: relying on a provider's undocumented invariant
// is how a quiet scaling bug gets in.
//
// `usage.cost` carries the real dollars the request spent, not an estimate.
// 11,101 tokens came back as $0.00011101, which is exactly the catalogue's
// $0.01 per million — so this client does not have to infer cost from a price
// table, and prices.yaml can carry a MEASURED figure instead of a list price.
//
// The same text embedded twice in one request does NOT come back identical.
// Measured: cosine 0.999998877, largest element difference 1.6e-4, and not one
// of 1,024 values equal. It is floating-point non-determinism from batching on
// the provider's side, far too small to move a ranking — but it means the disk
// cache is load-bearing for the determinism check, not merely for cost. Three
// repeats agree because they read one cached vector, and what they verify with
// a hosted embedder is that the cache is stable, not that the vendor is. Said
// plainly here because the project reads a repeat disagreement as a finding.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { STORAGE_DIM } from "./storage.ts";

export const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

/**
 * 1,024 dimensions, multilingual, 8,194 tokens of context, $0.01 per million.
 *
 * Chosen over `google/gemini-embedding-2` for the first run on two grounds
 * beyond the twentyfold price difference: 1,024 fits under the 1,536 storage
 * width with no request for a reduced dimensionality, and OpenRouter's
 * pass-through of such a request is not something this repository can verify
 * without spending to find out.
 */
export const DEFAULT_MODEL = "baai/bge-m3";

/** Verified against the live endpoint: 101 texts in one request, 4 seconds. */
export const MAX_BATCH = 100;

export type OpenRouterUsage = {
  calls: number;
  texts: number;
  cached: number;
  tokens: number;
  /** Dollars, as the vendor reported them. Not derived from a price table. */
  usd: number;
  waitedMs: number;
};

/** New vectors between cache writes. A crash loses seconds of work, not minutes. */
const WRITE_EVERY = 500;

/**
 * Which kind of failure this is.
 *
 * The same three-way split GeminiEmbedder and CohereReranker draw, for the
 * same reason: a rate limit is waited out, a spent allowance and a bad key are
 * not, and retrying either of the last two burns time on something that cannot
 * succeed.
 */
export function classifyError(status: number, body: string): "rate" | "credits" | "auth" | "other" {
  if (status === 401 || status === 403) return "auth";
  if (status === 402) return "credits";
  if (status === 429) {
    // OpenRouter returns 429 both for per-minute limits and for a free-tier
    // daily allowance; only the body separates them.
    return /credit|quota|daily|insufficient/i.test(body) ? "credits" : "rate";
  }
  return status >= 500 ? "rate" : "other";
}

export class OpenRouterEmbedder {
  private cachePath: string;
  private cache: Record<string, number[]>;
  private apiKey: string | undefined;
  private pending = 0;
  // Plain fields rather than constructor parameter properties: this project
  // runs TypeScript through type stripping, which rejects those outright.
  readonly model: string;
  readonly usage: OpenRouterUsage =
    { calls: 0, texts: 0, cached: 0, tokens: 0, usd: 0, waitedMs: 0 };

  /**
   * A missing key is not an error here, as it is not in CohereReranker.
   * "How much of this run is already paid for" has to be answerable with no
   * key at all — that is what `--dry-run` asks — and a constructor that throws
   * makes the answer unobtainable. Spending without a key fails at the request.
   */
  constructor(cacheDir: string, apiKey = process.env.OPENROUTER_API_KEY, model = DEFAULT_MODEL) {
    this.apiKey = apiKey;
    this.model = model;
    mkdirSync(cacheDir, { recursive: true });
    // The model is in the filename because it is what the vectors mean. Two
    // models must never read each other's cache, and the id carries a slash.
    this.cachePath = join(cacheDir, `openrouter-${model.replace(/\//g, "-")}.json`);
    this.cache = existsSync(this.cachePath) ? JSON.parse(readFileSync(this.cachePath, "utf8")) : {};
  }

  /**
   * Keyed on the model AND the text, unlike GeminiEmbedder's key.
   *
   * That client serves one model, so the text alone identifies a vector. This
   * one takes a model argument, and a key over the text alone would serve
   * bge-m3's vectors for a gemini-embedding-2 run — 1,024 values where 3,072
   * were expected, or worse, the right count and the wrong meaning. Delimited
   * with "\0" for the reason chunkSha uses it: a NUL cannot occur in either
   * part, so no pair of model and text can collide with another.
   */
  private key(text: string): string {
    return createHash("sha256").update(this.model).update("\0").update(text)
      .digest("hex").slice(0, 24);
  }

  get hasKey(): boolean {
    return Boolean(this.apiKey);
  }

  /** How many of these are already on disk. Used by --dry-run; embeds nothing. */
  cachedCount(texts: string[]): number {
    return new Set(texts.filter((t) => this.cache[this.key(t)])).size;
  }

  /**
   * Embeds everything, from cache where possible, and returns one vector per
   * input in input order — including duplicates, which are embedded once.
   */
  async embedAll(
    texts: string[],
    opts: { batchSize?: number; onProgress?: (done: number, total: number) => void } = {},
  ): Promise<number[][]> {
    const step = Math.min(Math.max(1, opts.batchSize ?? MAX_BATCH), MAX_BATCH);
    const missing = [...new Set(texts.filter((t) => !this.cache[this.key(t)]))];
    this.usage.cached += texts.length - missing.length;

    for (let i = 0; i < missing.length; i += step) {
      const batch = missing.slice(i, i + step);
      const vectors = await this.batch(batch);
      batch.forEach((t, j) => (this.cache[this.key(t)] = vectors[j]!));
      this.pending += batch.length;
      if (this.pending >= WRITE_EVERY) this.flush();
      opts.onProgress?.(Math.min(i + batch.length, missing.length), missing.length);
    }
    if (this.pending > 0) this.flush();

    return texts.map((t) => this.cache[this.key(t)]!);
  }

  /** One billed request. */
  private async batch(texts: string[]): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error("OPENROUTER_API_KEY is not set — put it in .env, see .env.example");
    }

    const body = await this.withRetry(async () => {
      const res = await fetch(OPENROUTER_URL, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: this.model, input: texts }),
      });
      if (!res.ok) throw new HttpError(res.status, await res.text());
      return await res.json() as unknown;
    });

    const vectors = parseEmbeddings(body, texts.length);
    const usage = (body as { usage?: { prompt_tokens?: number; cost?: number } }).usage;
    this.usage.calls++;
    this.usage.texts += texts.length;
    this.usage.tokens += usage?.prompt_tokens ?? 0;
    this.usage.usd += usage?.cost ?? 0;
    return vectors;
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await fn();
      } catch (e) {
        if (!(e instanceof HttpError)) throw e;
        const kind = classifyError(e.status, e.body);
        if (kind === "auth") {
          throw new Error(`OPENROUTER_API_KEY was rejected (${e.status}). Nothing to retry.`);
        }
        if (kind === "credits") {
          throw new Error(
            `OpenRouter credits are spent, nothing to wait for (${e.status}): ` +
            `${e.body.slice(0, 300)}\nVectors already embedded are cached and will not be re-requested.`,
          );
        }
        if (kind !== "rate" || attempt >= 5) {
          throw new Error(`embeddings failed (${e.status}): ${e.body.slice(0, 300)}`);
        }
        // Exponential, because a per-minute limit that is refused twice is not
        // going to clear in the same second it was refused in.
        const ms = 2 ** attempt * 1000;
        this.usage.waitedMs += ms;
        process.stdout.write(`\n  rate limited, waiting ${ms / 1000}s\n`);
        await new Promise((r) => setTimeout(r, ms));
      }
    }
  }

  /** Temporary name then rename, so a kill mid-write cannot truncate the file. */
  flush(): void {
    const tmp = `${this.cachePath}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.cache));
    renameSync(tmp, this.cachePath);
    this.pending = 0;
  }
}

/**
 * Vectors from the response body, in request order, unit length.
 *
 * Exported and separate because it is the part worth testing without a
 * network. A short `data` array, an index outside the request, or a vector
 * wider than the storage width all have to fail loudly: the equivalent mistake
 * on the Gemini side — a string array collapsing to one vector — returned
 * plausible data and was caught only by counting.
 */
export function parseEmbeddings(body: unknown, expected: number): number[][] {
  const data = (body as { data?: unknown })?.data;
  if (!Array.isArray(data)) throw new Error("embeddings response has no data array");
  if (data.length !== expected) {
    throw new Error(`asked for ${expected} embeddings, got ${data.length}`);
  }

  const out: number[][] = new Array(expected);
  for (const row of data) {
    const i = (row as { index?: unknown }).index;
    const vec = (row as { embedding?: unknown }).embedding;
    if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= expected) {
      throw new Error(`embeddings response carries index ${String(i)} for ${expected} inputs`);
    }
    if (!Array.isArray(vec) || vec.length === 0 || !vec.every((x) => typeof x === "number" && Number.isFinite(x))) {
      throw new Error(`embedding ${i} is not a finite number array`);
    }
    if (vec.length > STORAGE_DIM) {
      throw new Error(`model returned ${vec.length} dimensions, above the storage width ${STORAGE_DIM}`);
    }
    if (out[i] !== undefined) throw new Error(`embeddings response repeats index ${i}`);
    out[i] = unit(vec as number[]);
  }
  return out;
}

/** Unit length. A zero vector stays zero rather than becoming NaN. */
function unit(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n === 0 ? v : v.map((x) => x / n);
}

class HttpError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string) {
    super(`HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}
