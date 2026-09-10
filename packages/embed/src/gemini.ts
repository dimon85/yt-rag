// Gemini embeddings. The hosted model, against which the local one is only a
// gate instrument.
//
// 1536 dimensions by request, which is exactly the storage width chosen for
// chunks.embedding — so nothing is padded and that decision costs nothing here.
// Vectors come back at unit length already; they are renormalized anyway,
// because relying on a provider's undocumented invariant is how a quiet
// scaling bug gets in.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GoogleGenAI } from "@google/genai";
import { STORAGE_DIM } from "./storage.ts";

export const GEMINI_MODEL = "gemini-embedding-2";
export const GEMINI_DIM = STORAGE_DIM;

/** The API rejects anything larger: "at most 100 requests can be in one batch". */
export const MAX_BATCH = 100;

/**
 * Batches must be arrays of content objects, never arrays of strings.
 *
 * `contents: ["a", "b", "c"]` is accepted and returns ONE vector: the array is
 * read as parts of a single document and concatenated. Three chunks would
 * share an embedding and nothing would error — only the count of vectors would
 * disagree with the count of inputs, which is why `batch` checks it every time.
 */
const asContents = (texts: string[]) => texts.map((text) => ({ parts: [{ text }] }));

/**
 * Which kind of 429 this is, the distinction the previous harness learned the
 * hard way: a per-minute quota is waited out, a per-day one is not, and
 * treating them alike burns an hour retrying something that cannot succeed.
 */
export function quotaKind(message: string): "minute" | "day" | "other" {
  if (/PerDay|per_day|RequestsPerDay/i.test(message)) return "day";
  if (/429|RESOURCE_EXHAUSTED|quota/i.test(message)) return "minute";
  return "other";
}

/** The quota Google names in the error, so a claim about which one is checkable. */
export function quotaId(message: string): string {
  return message.match(/"quotaId":\s*"([^"]+)"/)?.[1]
    ?? message.match(/for metric:\s*([^\s,"]+)/)?.[1]
    ?? "unnamed";
}

/** Google states the wait in the error; use it rather than guessing. */
export function retryAfterMs(message: string): number {
  const m = message.match(/retry in ([\d.]+)s/i) ?? message.match(/"retryDelay":\s*"(\d+)s"/i);
  return m ? Math.ceil(Number(m[1]) * 1000) + 500 : 30_000;
}

export type EmbedUsage = { calls: number; texts: number; cached: number; waitedMs: number };

export class GeminiEmbedder {
  private ai: GoogleGenAI;
  private cachePath: string;
  private cache: Record<string, number[]>;
  readonly usage: EmbedUsage = { calls: 0, texts: 0, cached: 0, waitedMs: 0 };

  constructor(cacheDir: string, apiKey = process.env.GEMINI_API_KEY) {
    if (!apiKey) throw new Error("GEMINI_API_KEY is not set — put it in .env, see .env.example");
    this.ai = new GoogleGenAI({ apiKey });
    mkdirSync(cacheDir, { recursive: true });
    this.cachePath = join(cacheDir, `${GEMINI_MODEL}-${GEMINI_DIM}.json`);
    this.cache = existsSync(this.cachePath) ? JSON.parse(readFileSync(this.cachePath, "utf8")) : {};
  }

  /**
   * Content hash, so the cache survives rechunking that produces the same text.
   *
   * Nothing but the text goes into the key, which is only safe while one text
   * has one vector. It does today: `taskType` is the parameter that would break
   * it, and this model ignores it. Measured rather than assumed — embedding the
   * same passage as RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, QUESTION_ANSWERING and
   * with no task type at all returns four byte-identical vectors, while an
   * invented value is rejected with a 400. So the parameter is validated and
   * then discarded. Passing it would have looked like asymmetric retrieval and
   * done nothing; a model that honours it would need the task type in this key.
   */
  private key = (text: string) => createHash("sha256").update(text).digest("hex").slice(0, 24);

  /** One request. Returns one vector per input, or throws. */
  async batch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (texts.length > MAX_BATCH) throw new Error(`batch of ${texts.length} exceeds the API limit of ${MAX_BATCH}`);

    const res = await this.ai.models.embedContent({
      model: GEMINI_MODEL,
      contents: asContents(texts),
      config: { outputDimensionality: GEMINI_DIM },
    });
    const out = res.embeddings ?? [];
    if (out.length !== texts.length) {
      throw new Error(
        `asked for ${texts.length} embeddings, got ${out.length}. ` +
        `A string array collapses to one vector — contents must be content objects.`,
      );
    }
    this.usage.calls++;
    this.usage.texts += texts.length;
    return out.map((e) => normalize(e.values ?? []));
  }

  /**
   * Embeds everything, from cache where possible.
   *
   * Quota is spent per text, not per request — a batch of 100 costs 100 units.
   * This is worth stating plainly because the arithmetic that looks right is
   * wrong: 3,238 chunks is 33 requests at the maximum batch size, which sounds
   * comfortable against a limit of 100 a minute, and is in fact 3,238 units
   * against a free daily allowance of roughly a thousand. The free tier cannot
   * embed this corpus at 128 tokens at all, on any schedule.
   *
   * Measured, once the day's budget was nearly gone: a one-text call succeeded
   * while a hundred-text one was refused outright as a per-day exhaustion. A
   * batch is not partially served — it is refused whole when it asks for more
   * than remains, which is why `batchSize` is worth turning down at the end of
   * a budget rather than retrying the same request.
   *
   * The cache is what makes any of this recoverable: a run that dies partway
   * keeps everything it embedded.
   */
  async embedAll(
    texts: string[],
    opts: { sleepMs?: number; batchSize?: number; onProgress?: (done: number, total: number) => void } = {},
  ): Promise<number[][]> {
    const sleepMs = opts.sleepMs ?? 700;
    const step = Math.min(Math.max(1, opts.batchSize ?? MAX_BATCH), MAX_BATCH);
    const missing = [...new Set(texts.filter((t) => !this.cache[this.key(t)]))];
    this.usage.cached = texts.length - missing.length;

    for (let i = 0; i < missing.length; i += step) {
      const slice = missing.slice(i, i + step);
      const vectors = await this.withRetry(() => this.batch(slice));
      slice.forEach((t, j) => (this.cache[this.key(t)] = vectors[j]!));
      writeFileSync(this.cachePath, JSON.stringify(this.cache));
      opts.onProgress?.(Math.min(i + step, missing.length), missing.length);
      if (i + step < missing.length && sleepMs > 0) await wait(sleepMs);
    }
    return texts.map((t) => this.cache[this.key(t)]!);
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await fn();
      } catch (e) {
        const message = String((e as Error).message ?? e);
        const kind = quotaKind(message);
        if (kind === "day") {
          // Naming the quota matters: the first version truncated the message
          // to 200 characters, which cut off the quotaId — the one field that
          // says whether stopping was right.
          throw new Error(
            `daily quota exhausted, nothing to wait for: ${quotaId(message)}\n` +
            `Embeddings already fetched are cached and will not be re-requested.`,
          );
        }
        if (kind !== "minute" || attempt >= 5) throw e;
        const ms = retryAfterMs(message);
        this.usage.waitedMs += ms;
        process.stdout.write(`\n  rate limited, waiting ${Math.round(ms / 1000)}s\n`);
        await wait(ms);
      }
    }
  }
}

/**
 * How many of these texts already have a cached vector, without constructing
 * an embedder.
 *
 * Separate from the class on purpose: the constructor demands GEMINI_API_KEY,
 * and the question "would this run cost anything" has to be answerable when
 * there is no key and no budget left. Reads the cache file and nothing else —
 * no client, no network.
 */
export function geminiCacheCoverage(
  cacheDir: string,
  texts: string[],
): { cached: number; missing: number; total: number } {
  const path = join(cacheDir, `${GEMINI_MODEL}-${GEMINI_DIM}.json`);
  const cache: Record<string, unknown> = existsSync(path)
    ? JSON.parse(readFileSync(path, "utf8"))
    : {};
  const unique = new Set(texts);
  let cached = 0;
  for (const t of unique) {
    if (cache[createHash("sha256").update(t).digest("hex").slice(0, 24)]) cached++;
  }
  return { cached, missing: unique.size - cached, total: unique.size };
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Unit length. A zero vector stays zero rather than becoming NaN. */
function normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n === 0 ? v : v.map((x) => x / n);
}
