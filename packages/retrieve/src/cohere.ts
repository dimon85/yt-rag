// Cohere Rerank. The cross-encoder half of the ablation, and the only one
// available here.
//
// docs/spec.md picks it not as one option among two but as the only one: a
// local cross-encoder in TypeScript means sentence-transformers, so either a
// Python service alongside or an ONNX runtime with a hand-rolled tokenizer,
// and both cost more than this step is worth. Three consequences the spec
// states up front and this file implements.
//
// Reranking is paid and networked. So it is cached on disk exactly as
// GeminiEmbedder is, and for a sharper reason than embeddings: the matrix asks
// each configuration the same 104 questions three times, and the repeats exist
// to VERIFY determinism, not to buy it three times. Without a cache, 14 cells
// × 104 questions × 3 repeats is 4,368 billed calls to establish that a hosted
// model returns the same ranking twice.
//
// The reranker-on/off comparison stays valid but is a comparison against ONE
// reranker. That belongs in the README as a limitation.
//
// Determinism stops being a formality. For a hosted model the three repeats
// check whether the vendor drifts between calls — which is a genuinely
// interesting line — and a cache would hide exactly that. So `rerank` reads the
// cache and `rerankFresh` deliberately does not, and eval.ts uses the second
// for repeats beyond the first when it has been told it may spend.
//
// No SDK. Node 22 has fetch, and adding a dependency for one POST would put a
// package in the lockfile that this repo can otherwise not reach at all.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const COHERE_MODEL = "rerank-v3.5";
export const COHERE_URL = "https://api.cohere.com/v2/rerank";

/** The API rejects more than this in one request. */
export const MAX_DOCUMENTS = 1000;

export type RerankScore = { id: number; score: number };

/** What the cache stores: index into the request's document list, and its score. */
type Cached = { i: number; s: number }[];

export type RerankUsage = { calls: number; documents: number; cached: number; waitedMs: number };

/**
 * Which kind of failure this is.
 *
 * The same distinction GeminiEmbedder draws, because it cost an hour there: a
 * per-minute limit is waited out, a spent monthly allowance and a bad key are
 * not, and retrying either of the last two burns time on something that cannot
 * succeed.
 */
export function classifyError(status: number, body: string): "minute" | "exhausted" | "auth" | "other" {
  if (status === 401 || status === 403) return "auth";
  if (status === 429) {
    // Cohere returns 429 for both the per-minute limit and a finished
    // allowance, and only the body distinguishes them — but not by the words
    // the first version of this looked for. A trial key's RATE limit reads:
    //
    //   "You are using a Trial key, which is limited to 10 API calls / minute.
    //    You can continue to use the Trial key for free or upgrade to a
    //    Production key..."
    //
    // which contains "Trial" and "upgrade", so matching those classified a
    // per-minute limit as a spent allowance and aborted a run that only had to
    // wait six seconds. Measured against the live API, not imagined.
    //
    // The rate limit is the one that names a per-minute rate, so that is what
    // is matched, and everything else at 429 is treated as exhausted. The
    // asymmetry is deliberate: waiting on a spent allowance costs a minute of
    // nothing, while aborting on a rate limit throws away the whole run.
    return /per minute|\/\s*min|calls\s*\/\s*minute|requests? per min/i.test(body)
      ? "minute"
      : /trial|monthly|billing|upgrade|quota|allowance/i.test(body)
      ? "exhausted"
      : "minute";
  }
  return "other";
}

/** Cohere states the wait in a header when it sets one; otherwise back off a minute. */
export function retryAfterMs(header: string | null): number {
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds * 1000) + 500 : 60_000;
}

/**
 * The ranking, from the response body.
 *
 * Separate and exported because it is the part worth testing without a
 * network: a response whose `results` are missing, or short, or carrying
 * indices outside the document list, has to fail loudly. The equivalent
 * mistake on the embedding side — a string array collapsing to one vector —
 * returned plausible-looking data and was caught only by counting.
 */
export function parseRerankResponse(body: unknown, documentCount: number): Cached {
  const results = (body as { results?: unknown })?.results;
  if (!Array.isArray(results)) throw new Error(`rerank response has no results array`);

  return results.map((r) => {
    const i = (r as { index?: unknown }).index;
    const s = (r as { relevance_score?: unknown }).relevance_score;
    if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= documentCount) {
      throw new Error(`rerank returned index ${String(i)} for ${documentCount} documents`);
    }
    if (typeof s !== "number" || !Number.isFinite(s)) {
      throw new Error(`rerank returned a non-numeric relevance_score for index ${i}`);
    }
    return { i, s };
  });
}

/**
 * Cache key over everything that decides the answer.
 *
 * The model, the query and the document TEXTS in order. Wider than
 * GeminiEmbedder's key, which is the text alone, because a rerank score is a
 * function of the pair — the same passage scores differently against a
 * different query, and the same query ranks differently over a different
 * candidate list. Keying on the query alone would serve one chunking
 * configuration's ranking to another.
 *
 * Delimited by "\0" for the reason chunkSha uses it: a space occurs inside
 * every document text and a NUL cannot, so a space would let two different
 * candidate lists collide on one key and serve the wrong ranking. Written as
 * the escape and never as a literal byte — a NUL in the source makes git treat
 * the file as binary and grep skip it without saying so.
 */
export function rerankKey(model: string, query: string, texts: string[]): string {
  const h = createHash("sha256").update(model).update("\0").update(query);
  for (const t of texts) h.update("\0").update(t);
  return h.digest("hex").slice(0, 24);
}

export class CohereReranker {
  private cachePath: string;
  private cache: Record<string, Cached>;
  private apiKey: string | undefined;
  /** Minimum gap between billed requests, and when the last one went out. */
  private minIntervalMs = 0;
  private lastRequestAt = 0;
  // Plain fields, not constructor parameter properties: the project runs
  // TypeScript directly through type stripping, which rejects those outright.
  // Vitest transpiles and would not have caught it.
  readonly model: string;
  readonly usage: RerankUsage = { calls: 0, documents: 0, cached: 0, waitedMs: 0 };

  /**
   * A missing key is NOT an error here, unlike GeminiEmbedder's constructor.
   *
   * The question "how much of this run is already paid for" has to be
   * answerable with no key at all — that is what --dry-run asks — and a
   * constructor that throws makes the answer unobtainable. Spending without a
   * key fails at `rerankFresh` instead, where it is actually attempted.
   */
  constructor(cacheDir: string, apiKey = process.env.COHERE_API_KEY, model = COHERE_MODEL) {
    this.apiKey = apiKey;
    this.model = model;
    mkdirSync(cacheDir, { recursive: true });
    this.cachePath = join(cacheDir, `cohere-${model}.json`);
    this.cache = existsSync(this.cachePath) ? JSON.parse(readFileSync(this.cachePath, "utf8")) : {};
  }

  get hasKey(): boolean {
    return Boolean(this.apiKey);
  }

  /**
   * Paces billed requests to at most `rpm` a minute.
   *
   * Retrying on a 429 is the safety net, not the plan: a trial key allows 10
   * calls a minute, and a client that discovers this by being refused spends
   * its attempts learning the same fact 300 times. Pacing costs the same wall
   * clock — the limit is the limit — and arrives without the refusals.
   */
  pace(rpm: number): void {
    this.minIntervalMs = rpm > 0 ? Math.ceil(60_000 / rpm) : 0;
  }

  /** Whether this exact query-and-candidates request is already on disk. */
  isCached(query: string, docs: { id: number; text: string }[]): boolean {
    return Boolean(this.cache[rerankKey(this.model, query, docs.map((d) => d.text))]);
  }

  /** The cached ranking, or null. Spends nothing and needs no key. */
  cachedRerank(query: string, docs: { id: number; text: string }[], topN: number): RerankScore[] | null {
    const hit = this.cache[rerankKey(this.model, query, docs.map((d) => d.text))];
    if (!hit) return null;
    this.usage.cached++;
    return toScores(hit, docs, topN);
  }

  /** Cache first, then the API. */
  async rerank(query: string, docs: { id: number; text: string }[], topN: number): Promise<RerankScore[]> {
    return this.cachedRerank(query, docs, topN) ?? await this.rerankFresh(query, docs, topN);
  }

  /**
   * One billed call, cache written after it.
   *
   * Used directly for the repeats past the first, where the point is to see
   * whether the vendor's model drifts — a cached second repeat would answer
   * that question with the first repeat's answer.
   */
  async rerankFresh(
    query: string,
    docs: { id: number; text: string }[],
    topN: number,
  ): Promise<RerankScore[]> {
    if (!this.apiKey) {
      throw new Error("COHERE_API_KEY is not set — put it in .env, see .env.example");
    }
    if (docs.length > MAX_DOCUMENTS) {
      throw new Error(`${docs.length} documents exceeds the API limit of ${MAX_DOCUMENTS}`);
    }
    if (docs.length === 0) return [];

    const texts = docs.map((d) => d.text);
    if (this.minIntervalMs > 0) {
      const wait = this.lastRequestAt + this.minIntervalMs - Date.now();
      if (wait > 0) {
        // Counted as waiting, not as service time. eval.ts subtracts it from
        // the latency it records: a p95 that reports our own throttle would
        // describe this client's configuration rather than the vendor's speed.
        this.usage.waitedMs += wait;
        await new Promise((r) => setTimeout(r, wait));
      }
      this.lastRequestAt = Date.now();
    }
    const parsed = await this.withRetry(async () => {
      const res = await fetch(COHERE_URL, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: this.model,
          query,
          documents: texts,
          // Asked for in full and truncated locally, so one cached call serves
          // every top_k the report might want.
          top_n: Math.min(docs.length, MAX_DOCUMENTS),
        }),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new HttpError(res.status, body, res.headers.get("retry-after"));
      }
      this.usage.calls++;
      this.usage.documents += docs.length;
      return parseRerankResponse(await res.json(), docs.length);
    });

    this.cache[rerankKey(this.model, query, texts)] = parsed;
    this.flush();
    return toScores(parsed, docs, topN);
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await fn();
      } catch (e) {
        if (!(e instanceof HttpError)) throw e;
        const kind = classifyError(e.status, e.body);
        if (kind === "auth") {
          throw new Error(`COHERE_API_KEY was rejected (${e.status}). Nothing to retry.`);
        }
        if (kind === "exhausted") {
          throw new Error(
            `Cohere allowance is spent, nothing to wait for (${e.status}): ${e.body.slice(0, 300)}\n` +
            `Rankings already fetched are cached and will not be re-requested.`,
          );
        }
        if (kind !== "minute" || attempt >= 5) {
          throw new Error(`rerank failed (${e.status}): ${e.body.slice(0, 300)}`);
        }
        const ms = retryAfterMs(e.retryAfter);
        this.usage.waitedMs += ms;
        process.stdout.write(`\n  rate limited, waiting ${Math.round(ms / 1000)}s\n`);
        await new Promise((r) => setTimeout(r, ms));
      }
    }
  }

  /** Temporary name then rename, so a kill mid-write cannot truncate the file. */
  flush(): void {
    const tmp = `${this.cachePath}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.cache));
    renameSync(tmp, this.cachePath);
  }
}

class HttpError extends Error {
  readonly status: number;
  readonly body: string;
  readonly retryAfter: string | null;

  constructor(status: number, body: string, retryAfter: string | null) {
    super(`HTTP ${status}`);
    this.status = status;
    this.body = body;
    this.retryAfter = retryAfter;
  }
}

/**
 * Response indices back onto chunk ids, best first.
 *
 * Sorted by score and then by the document's position in the request, which is
 * the same tiebreak every other ranking in this codebase uses. A cross-encoder
 * returning two identical scores is not hypothetical at three significant
 * figures, and without the tiebreak two repeats would differ for a reason that
 * has nothing to do with the vendor drifting.
 */
function toScores(cached: Cached, docs: { id: number; text: string }[], topN: number): RerankScore[] {
  return [...cached]
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .slice(0, topN)
    .map(({ i, s }) => ({ id: docs[i]!.id, score: s }));
}
