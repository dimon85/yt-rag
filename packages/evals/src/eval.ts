// The ablation runner.
//
//   pnpm eval --run-all
//   pnpm eval --chunking fixed-128 --retrieval bm25,vector --reranking none
//   pnpm eval --dry-run --run-all      # print the matrix and the cost, run nothing
//
// This is `pnpm ceiling` generalised over configs.yaml. The gate answers one
// question about one hand-specified configuration; this walks the matrix the
// file declares — chunking × retrieval × reranking, three repeats each — and
// writes one JSONL file per cell, appended as results arrive, so a run that
// dies keeps what it did.
//
// It reads cache/transcripts/*.json and never re-fetches (invariant 5). There
// is no database; see the note at the top of run.ts for where the
// `chunk_set_id` of invariants 1 and 9 lives instead.
//
// Two things it will not do, and both are refusals rather than omissions.
//
// It does not spend money it has not been told it has. Dense cells are checked
// against the embedding cache BEFORE anything runs, and a cell whose vectors
// are not already on disk is recorded as "not run: no budget" — never as a
// zero and never as a blank, because a missing cell that looks like a bad
// result is worse than an empty one.
//
// It does not average the three repeats. They are in configs.yaml to verify
// that retrieval is deterministic, and a disagreement between them is a
// finding printed at the end of the run, not noise to smooth away.
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { encode } from "gpt-tokenizer";
import { CachedLocalEmbedder } from "../../embed/src/local-cache.ts";
import { geminiCacheCoverage, GEMINI_DIM, GeminiEmbedder } from "../../embed/src/gemini.ts";
import { LOCAL_DIM } from "../../embed/src/local.ts";
import { DEFAULT_MODEL as OPENROUTER_MODEL, OpenRouterEmbedder } from "../../embed/src/openrouter.ts";
import { cosine, toStorage } from "../../embed/src/storage.ts";
import { CACHE_DIR, loadCorpus, ROOT } from "../../ingest/src/corpus.ts";
import type { Segment } from "../../ingest/src/text.ts";
import { buildIndex, search, tokenize } from "../../retrieve/src/bm25.ts";
import { rrf } from "../../retrieve/src/fuse.ts";
import { jaccard, mmrRerank } from "../../retrieve/src/mmr.ts";
import { CohereReranker } from "../../retrieve/src/cohere.ts";
import {
  cellId, chunkerFor, expandMatrix, loadConfigs, needsEmbeddings,
  type Cell, type Chunking,
} from "./configs.ts";
import { goldenSha, loadGolden, type Question } from "./golden.ts";
import { embeddingRate, loadPrices } from "./prices.ts";
import { HIT_COVERAGE } from "./hit.ts";
import { lexicallyTrivial, type Retrieved } from "./metrics.ts";
import {
  chunkSha, repeatDisagreements, scoreDrift, scoreQuestion, type Header, type Result,
} from "./run.ts";

const HELP = `
usage: pnpm eval [--run-all | --chunking IDS --retrieval IDS --reranking IDS]
                 [--dry-run] [--embedder local|gemini] [--top-k N] [--repeats N]
                 [--out DIR] [--force] [--spend-rerank]

  --run-all      every cell of chunking × retrieval × reranking in configs.yaml
  --chunking     comma-separated ids; same for --retrieval and --reranking
  --dry-run      print the matrix, the chunk counts and what embedding each
                 dense cell would need. Makes no network call and writes no
                 run files.
  --rerank-rpm N pace billed rerank calls to N a minute. A Cohere trial key
                 allows 10; without pacing the run spends its retries being
                 refused.
  --spend-rerank allow billed Cohere Rerank calls. Without it a reranking
                 config of kind \`api\` runs only from its disk cache, and a cell
                 needing new calls is recorded as not run. Reranking is the one
                 axis where the three repeats really do cost three times: they
                 check whether the vendor drifts, so they must not be cached.
  --force        replace existing cell files in --out. Without it an --out
                 directory that already holds a run is refused, because
                 appending to a cell file gives it two headers.
  --embedder     which model the dense retrievers use. Not an axis in
                 configs.yaml, which names \`vector\` and \`hybrid\` without
                 naming a model, so it is a run-level choice recorded in every
                 header. \`local\` is free and needs no key; \`gemini\` spends
                 quota per text and is skipped unless every vector it needs is
                 already cached; \`openrouter\` bills per token and needs
                 --spend-embed before it will spend anything.
  --embed-model  the model id for --embedder openrouter (default ${OPENROUTER_MODEL}).
                 It is the embedder recorded in the header, because the model
                 and not the gateway is what a vector means.
  --spend-embed  allow billed embedding calls. Without it a cell whose vectors
                 are not all cached is recorded as "not run", with the count.
`.trimStart();

const argv = process.argv.slice(2);
const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        "run-all": { type: "boolean", default: false },
        "dry-run": { type: "boolean", default: false },
        chunking: { type: "string" },
        retrieval: { type: "string" },
        reranking: { type: "string" },
        embedder: { type: "string", default: "local" },
        "embed-model": { type: "string" },
        "spend-embed": { type: "boolean", default: false },
        "top-k": { type: "string" },
        repeats: { type: "string" },
        out: { type: "string" },
        force: { type: "boolean", default: false },
        "spend-rerank": { type: "boolean", default: false },
        "rerank-rpm": { type: "string" },
      },
      allowPositionals: false,
    });
  } catch (e) {
    console.error(`${(e as Error).message}\n\n${HELP}`);
    process.exit(2);
  }
})();

const ids = (s: string | undefined) =>
  s === undefined ? undefined : s.split(",").map((x) => x.trim()).filter(Boolean);

const selection = {
  chunking: ids(values.chunking),
  retrieval: ids(values.retrieval),
  reranking: ids(values.reranking),
};
const narrowed = Object.values(selection).some((v) => v !== undefined);
if (!values["run-all"] && !narrowed) {
  // Neither the whole matrix nor a named subset. Running the whole thing by
  // default would spend an hour on a bare `pnpm eval`.
  console.error(`nothing selected — pass --run-all or name an axis\n\n${HELP}`);
  process.exit(2);
}

const embedderName = values.embedder!;
if (!["local", "gemini", "openrouter"].includes(embedderName)) {
  console.error(`--embedder must be local, gemini or openrouter\n\n${HELP}`);
  process.exit(2);
}
const embedModel = values["embed-model"] ?? OPENROUTER_MODEL;
// The model, not the gateway, is what a vector means: two OpenRouter models
// are two different configurations, and the header, the row label and
// prices.yaml all key on this.
const embedderLabel = embedderName === "openrouter" ? embedModel : embedderName;

const cfg = loadConfigs();
const topK = values["top-k"] ? Number(values["top-k"]) : cfg.defaults.top_k;
const repeats = values.repeats ? Number(values.repeats) : cfg.defaults.repeats;
if (!(topK > 0) || !(repeats > 0)) {
  console.error(`--top-k and --repeats must be positive\n\n${HELP}`);
  process.exit(2);
}

const golden = loadGolden();
if (golden.questions.length === 0) {
  console.error("golden/questions.yaml is empty — nothing to measure.");
  process.exit(2);
}
const corpus = loadCorpus();
const cells = expandMatrix(cfg, selection);

// ─── provenance ──────────────────────────────────────────────────────────────

/**
 * The sha invariant 6 compares runs on, with `-dirty` appended when the tree
 * has uncommitted changes.
 *
 * A dirty tree's HEAD sha describes code that was not the code that ran, so
 * two runs from two different dirty trees would compare as identical. The
 * suffix makes them non-comparable instead, which is the honest answer.
 */
function gitSha(): string {
  try {
    const head = execFileSync("git", ["rev-parse", "--short=12", "HEAD"], { cwd: ROOT })
      .toString().trim();
    const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: ROOT })
      .toString().trim().length > 0;
    return dirty ? `${head}-dirty` : head;
  } catch {
    return "unknown";
  }
}

const GOLDEN_SHA = goldenSha(golden);
const GIT_SHA = gitSha();

// ─── chunk ───────────────────────────────────────────────────────────────────

type IndexedChunk = Retrieved & { text: string; token_count: number };

const segmentsByVideo = new Map<string, Segment[]>();
for (const v of corpus.videos) {
  const path = join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`);
  if (!existsSync(path)) {
    console.error(`no cached transcript for ${v.youtube_id} — run pnpm ingest first`);
    process.exit(2);
  }
  segmentsByVideo.set(v.youtube_id, JSON.parse(readFileSync(path, "utf8")).segments);
}

/**
 * How much wider than top-k a reranker's candidate pool is.
 *
 * Shared by the reranking branch of `retrieve` and by the dry run's call
 * estimate, because a rerank request is cached on its exact candidate list: two
 * different pool sizes are two different requests, and an estimate computed
 * with the wrong one would report cached calls that are not.
 */
const RERANK_POOL = 5;

/** Chunked once per chunking config and reused across every retrieval × reranking cell. */
const chunkCache = new Map<string, IndexedChunk[]>();
function chunksFor(config: Chunking): IndexedChunk[] {
  const hit = chunkCache.get(config.id);
  if (hit) return hit;
  const chunk = chunkerFor(config);
  const out: IndexedChunk[] = [];
  for (const [video, segments] of segmentsByVideo) {
    for (const c of chunk(segments)) {
      out.push({
        video, start_s: c.start_s, end_s: c.end_s,
        text: c.text, token_count: c.token_count, score: 0,
      });
    }
  }
  chunkCache.set(config.id, out);
  return out;
}

// ─── budget pre-flight ───────────────────────────────────────────────────────

// Both the chunk vectors and the QUERY vectors have to be cached for a dense
// cell to run without spending anything. Checking only the chunks is the
// mistake this is written to avoid: the cache covers fixed-128 and fixed-512
// completely, and covers 45 of the 104 golden questions — so a `vector` cell
// on fixed-128 needs 59 new API calls despite every chunk being on disk.
const localEmbedder = new CachedLocalEmbedder(join(CACHE_DIR, "embeddings"));
// Constructed unconditionally: its constructor does not demand a key, so
// --dry-run can ask what a run would cost with no key present at all.
const openrouterEmbedder = new OpenRouterEmbedder(join(CACHE_DIR, "embeddings"), undefined, embedModel);
const reranker = new CohereReranker(join(CACHE_DIR, "rerank"));
// A Cohere trial key allows 10 calls a minute. Pacing to it turns 300
// refusals into 300 requests that are simply spread out — the wall clock is
// the same either way, because the limit is the limit.
if (values["rerank-rpm"]) reranker.pace(Number(values["rerank-rpm"]));
const questionTexts = golden.questions.map((q) => q.text);
// Counted once, with the tokenizer the chunkers use — invariant 2 — so the
// query and index token counts in cost_units are on one scale.
const questionTokens = questionTexts.reduce((n, t) => n + encode(t).length, 0);

/**
 * Billed rerank calls a cell would still make.
 *
 * Counted, not estimated, for the first repeat: whether a request is cached
 * depends on the exact candidate list, so the pool has to be built to know.
 * The repeats past the first are counted as uncached on purpose — they are
 * there to see whether the vendor's model drifts between calls, and a cached
 * repeat would answer that with the first repeat's answer. That makes reranking
 * the one axis where the three repeats genuinely cost three times, and the
 * dry run says so rather than discovering it at the invoice.
 */
function rerankCallsNeeded(cell: Cell): number {
  const chunks = chunksFor(cell.chunking);
  const index = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
  let missing = 0;
  for (const q of golden.questions) {
    const pool = search(index, q.text, topK * RERANK_POOL).map(({ id }) => ({ id, text: chunks[id]!.text }));
    if (!reranker.isCached(q.text, pool)) missing++;
  }
  // The first repeat may be cached; the rest are always fresh.
  return missing + golden.questions.length * (repeats - 1);
}

type Verdict = { run: true } | { run: false; reason: string };

function preflight(cell: Cell): Verdict {
  if (cell.reranking.kind === "api") {
    // A client exists (packages/retrieve/src/cohere.ts). What it needs is a key
    // and permission to spend, and the two are separate reasons: no key is a
    // setup problem, an uncached call without --spend-rerank is a refusal.
    if (!reranker.hasKey) {
      return { run: false, reason: `not run: COHERE_API_KEY is not set (${cell.reranking.id})` };
    }
    const missing = rerankCallsNeeded(cell);
    if (missing > 0 && !values["spend-rerank"]) {
      return {
        run: false,
        reason:
          `not run: ${missing} of ${golden.questions.length * repeats} rerank calls are not cached ` +
          `and --spend-rerank was not passed`,
      };
    }
    // Falls through: a rerank cell still needs its retrieval half checked.
  }
  if (!needsEmbeddings(cell.retrieval)) return { run: true };

  if (embedderName === "local") return { run: true };

  if (embedderName === "openrouter") {
    const chunks = chunksFor(cell.chunking);
    const texts = [...chunks.map((c) => c.text), ...questionTexts];
    const unique = new Set(texts).size;
    const missing = unique - openrouterEmbedder.cachedCount(texts);
    if (missing === 0) return { run: true };
    // Billed per token rather than capped per day, so unlike Gemini this is a
    // decision rather than a wall — and the decision is the operator's, made
    // once with a flag, not implied by running the command.
    if (!openrouterEmbedder.hasKey) {
      return { run: false, reason: `not run: ${missing} vectors are not cached and OPENROUTER_API_KEY is not set` };
    }
    if (!values["spend-embed"]) {
      return {
        run: false,
        reason: `not run: ${missing} of ${unique} vectors are not cached and --spend-embed was not passed`,
      };
    }
    return { run: true };
  }

  const chunks = chunksFor(cell.chunking);
  const forChunks = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), chunks.map((c) => c.text));
  const forQueries = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), questionTexts);
  const missing = forChunks.missing + forQueries.missing;
  if (missing === 0) return { run: true };
  return {
    run: false,
    reason:
      `not run: no budget — ${forChunks.missing} of ${forChunks.total} chunk vectors and ` +
      `${forQueries.missing} of ${forQueries.total} query vectors are not cached, ` +
      `and the Gemini spend cap is exhausted`,
  };
}

// ─── dry run ─────────────────────────────────────────────────────────────────

const pad = (s: string, n: number) => s.padEnd(n);

if (values["dry-run"]) {
  console.log(`matrix: ${cells.length} cells × ${repeats} repeats × ${golden.questions.length} questions`);
  console.log(`golden_sha ${GOLDEN_SHA}   git_sha ${GIT_SHA}   top_k ${topK}   embedder ${embedderName}\n`);

  console.log("chunking configurations:");
  const seen = new Set<string>();
  for (const cell of cells) {
    if (seen.has(cell.chunking.id)) continue;
    seen.add(cell.chunking.id);
    const chunks = chunksFor(cell.chunking);
    const texts = chunks.map((c) => c.text);
    // Counted with gpt-tokenizer by the chunkers themselves — invariant 2, the
    // same tokenizer in every configuration, so these figures are comparable.
    const tokens = chunks.reduce((n, c) => n + c.token_count, 0);
    // Cached-versus-needed for BOTH embedders, because the choice of embedder
    // is what the cost claim depends on and a dry run should not make the
    // reader run it twice to see the other half.
    const g = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), texts);
    const l = localEmbedder.cachedCount(texts);
    console.log(
      `  ${pad(cell.chunking.id, 18)} ${String(chunks.length).padStart(6)} chunks   ` +
      `${(tokens / 1000).toFixed(0)}k tokens, mean ${(tokens / chunks.length).toFixed(0)}   ` +
      `gemini ${g.cached}/${g.total} cached   local ${l}/${new Set(texts).size} cached`,
    );
  }

  const q = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), questionTexts);
  console.log(
    `\nquery vectors (${questionTexts.length} questions): ` +
    `gemini ${q.cached} cached, ${q.missing} would need an API call; ` +
    `local ${localEmbedder.cachedCount(questionTexts)} cached\n`,
  );

  console.log("cells:");
  let runnable = 0;
  for (const cell of cells) {
    const v = preflight(cell);
    if (v.run) runnable++;
    console.log(`  ${v.run ? "run    " : "skip   "}${pad(cellId(cell), 42)}${v.run ? "" : v.reason}`);
  }
  console.log(
    `\n${runnable} of ${cells.length} cells would run under --embedder ${embedderName}.`,
  );

  // The budget picture for the OTHER embedder too. Printed unconditionally
  // because the default is `local`, which costs nothing and would therefore
  // show no budget skips at all — and "what would this cost against the
  // hosted model" is the question a dry run is for. Computed from the cache
  // files alone; still no network call.
  const dense = cells.filter((c) => needsEmbeddings(c.retrieval) && c.reranking.kind !== "api");
  const blocked = dense.filter((cell) => {
    const g = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), chunksFor(cell.chunking).map((c) => c.text));
    return g.missing > 0 || q.missing > 0;
  });
  console.log(
    `\nunder --embedder gemini, ${blocked.length} of those ${dense.length} dense cells would be\n` +
    `skipped as "not run: no budget". The cache covers fixed-128 and fixed-512\n` +
    `completely and ${q.cached} of ${q.total} query vectors, so even a chunking config whose\n` +
    `chunks are all cached still needs ${q.missing} query embeddings — which is why the\n` +
    `pre-flight checks the queries and not only the chunks.`,
  );

  // Cohere: the client exists, so what blocks these cells is a key and
  // permission, and both are cheap to state precisely.
  const api = cells.filter((c) => c.reranking.kind === "api");
  if (api.length > 0) {
    const perCell = golden.questions.length * repeats;
    console.log(
      `\n${api.length} cells use a reranker of kind api. ` +
      (reranker.hasKey
        ? `COHERE_API_KEY is set; each needs up to ${perCell} billed calls\n` +
          `(${golden.questions.length} questions × ${repeats} repeats — the repeats check whether the vendor\n` +
          `drifts, so they are deliberately not served from cache), and they run only\n` +
          `with --spend-rerank.`
        : `COHERE_API_KEY is not set, so they cannot run. With a key each needs up to\n` +
          `${perCell} billed calls (${golden.questions.length} questions × ${repeats} repeats), and --spend-rerank.`),
    );
  }

  // Gemini: what blocks these is 59 query vectors, and that is a number worth
  // printing as a price rather than as a refusal — quota is spent per text, so
  // it is exactly 59 units to unblock every cell whose chunks are already cached.
  if (q.missing > 0) {
    const unblocked = dense.filter((cell) => {
      const g = geminiCacheCoverage(join(CACHE_DIR, "embeddings"), chunksFor(cell.chunking).map((c) => c.text));
      return g.missing === 0;
    });
    console.log(
      `\nUnder --embedder gemini the cheapest thing that changes the picture is the\n` +
      `query vectors: ${q.missing} texts, and quota is spent per text, so ${q.missing} units would\n` +
      `unblock ${unblocked.length} cells whose chunks are already cached in full. The remaining\n` +
      `${blocked.length - unblocked.length} would still need ` +
      `their chunks embedded.`,
    );
  }

  if (embedderName === "openrouter") {
    // What spending would actually buy, before it is spent. Priced from
    // prices.yaml at the same rate the report uses, so the estimate here and
    // the figure printed afterwards cannot drift apart.
    const rate = embeddingRate(loadPrices(join(ROOT, "prices.yaml")), embedModel);
    let tokens = 0;
    const seen = new Set<string>();
    for (const cell of cells) {
      if (seen.has(cell.chunking.id)) continue;
      seen.add(cell.chunking.id);
      for (const c of chunksFor(cell.chunking)) {
        if (!openrouterEmbedder.cachedCount([c.text])) tokens += c.token_count;
      }
    }
    tokens += questionTokens;
    console.log(
      `\nunder --embedder openrouter (${embedModel}): about ${(tokens / 1000).toFixed(0)}k tokens are\n` +
      `not cached` +
      (rate === null
        ? `. prices.yaml carries no rate for this model, so the cost cannot be stated.`
        : `, which at $${rate}/M is about $${(tokens / 1e6 * rate).toFixed(4)}. Billing is per\n` +
          `token rather than a daily cap, so this is a decision rather than a wall — and\n` +
          `--spend-embed is how the decision is made, once, rather than implied by running\n` +
          `the command.`),
    );
  }

  console.log(`\nNo network call was made and no run file was written.`);
  process.exit(0);
}

// ─── retrieval ───────────────────────────────────────────────────────────────

/** Per chunking config: everything a retriever needs, built once. */
type Built = {
  chunks: IndexedChunk[];
  lexical: ReturnType<typeof buildIndex>;
  tokenSets: Set<string>[];
  /** null when no dense cell in this group runs. */
  vectors: number[][] | null;
  queryVectors: Map<string, number[]> | null;
  /** Which questions plain BM25 answers at rank 1. Labels the report; filters nothing. */
  trivial: Map<string, boolean | null>;
  /** Mean ms of one query embedding, over the ones actually computed. null when all cached. */
  queryEmbedMs: number | null;
};

async function build(config: Chunking, wantDense: boolean): Promise<Built> {
  const chunks = chunksFor(config);
  const lexical = buildIndex(chunks.map((c, i) => ({ id: i, text: c.text })));
  const tokenSets = chunks.map((c) => new Set(tokenize(c.text)));

  // Invariant 11 and metrics.ts: computed from a plain BM25 pass over these
  // chunks, used only to split the report. Using a retriever to SELECT
  // questions would tune the set to the retriever; labelling discards nothing.
  const trivial = new Map<string, boolean | null>();
  for (const q of golden.questions) {
    const ranked = search(lexical, q.text, 1).map(({ id, score }) => ({ ...chunks[id]!, score }));
    trivial.set(q.slug, lexicallyTrivial(q.gold, ranked));
  }

  if (!wantDense) {
    return { chunks, lexical, tokenSets, vectors: null, queryVectors: null, trivial, queryEmbedMs: null };
  }

  const embed = async (texts: string[], label: string): Promise<number[][]> => {
    const onProgress = (done: number, total: number) =>
      process.stdout.write(`\r  ${label} ${done}/${total}   `);
    if (embedderName === "local") return localEmbedder.embedAll(texts, { onProgress });
    if (embedderName === "openrouter") return openrouterEmbedder.embedAll(texts, { onProgress });
    // Reached only when preflight found every vector already cached, so this
    // makes no request. The client is constructed lazily for the same reason:
    // its constructor demands a key the cached path does not need.
    const embedder = new GeminiEmbedder(join(CACHE_DIR, "embeddings"));
    return embedder.embedAll(texts);
  };

  const raw = await embed(chunks.map((c) => c.text), `${config.id} chunks`);
  // The width of a hosted model is whatever it returns, and asserting a
  // constant would be a guess. Taken from the first vector; toStorage then
  // rejects any later vector that disagrees, which is the check that matters.
  const dim = embedderName === "local" ? LOCAL_DIM
    : embedderName === "gemini" ? GEMINI_DIM
    : raw[0]?.length ?? 0;
  const vectors = raw.map((v) => toStorage(v, dim));

  // Timed, and divided by the queries that were actually computed rather than
  // by all of them: a run that read 104 vectors from disk in 3 ms would
  // otherwise report 0.03 ms per query embedding and put that in a latency
  // column, which is a measurement of the cache and not of the model.
  const cachedBefore = embedderName === "local"
    ? localEmbedder.cachedCount(questionTexts)
    : embedderName === "openrouter"
    ? openrouterEmbedder.cachedCount(questionTexts)
    : geminiCacheCoverage(join(CACHE_DIR, "embeddings"), questionTexts).cached;
  const computed = new Set(questionTexts).size - cachedBefore;
  const startedEmbedding = performance.now();
  const queries = (await embed(questionTexts, `${config.id} queries`)).map((v) => toStorage(v, dim));
  const queryEmbedMs = computed > 0 ? (performance.now() - startedEmbedding) / computed : null;
  process.stdout.write("\r");

  return {
    chunks,
    lexical,
    tokenSets,
    vectors,
    queryVectors: new Map(golden.questions.map((q, i) => [q.slug, queries[i]!])),
    trivial,
    queryEmbedMs,
  };
}

/**
 * The ranking one cell produces for one question.
 *
 * Every ordering here sorts by score and then by index. Equal scores are
 * common with BM25 — a query term matching one document is worth the same in
 * every document of the same length — and without the tiebreak two repeats
 * would differ for reasons that have nothing to do with the configuration.
 */
async function retrieve(cell: Cell, built: Built, q: Question, repeat: number): Promise<Retrieved[]> {
  const { chunks } = built;
  // A reranker can only promote something it was given, so it reorders a wider
  // pool than it returns. Without this MMR would reshuffle the same top-k and
  // never reach the other side of a disagreement, and a cross-encoder would be
  // asked to improve a ranking it cannot add to.
  const poolK = cell.reranking.kind === "none" ? topK : topK * RERANK_POOL;

  const lexRanked = () => search(built.lexical, q.text, poolK);
  const vecRanked = () => {
    const query = built.queryVectors!.get(q.slug)!;
    return chunks
      .map((_, i) => ({ id: i, score: cosine(query, built.vectors![i]!) }))
      .sort((a, b) => b.score - a.score || a.id - b.id)
      .slice(0, poolK);
  };

  let pool: { id: number; score: number }[];
  switch (cell.retrieval.kind) {
    case "lexical":
      pool = lexRanked();
      break;
    case "dense":
      pool = vecRanked();
      break;
    case "lexical+dense":
      // Fused by RANK, never by score. BM25 is unbounded and cosine lives in
      // [0,1], so any weighted sum of the two is decided by scale before it is
      // decided by relevance. The fused score replaces both and is on neither
      // scale, which is why it is carried through as-is rather than rescaled
      // back onto one of them.
      pool = rrf([
        lexRanked().map(({ id }) => String(id)),
        vecRanked().map(({ id }) => String(id)),
      ]).slice(0, poolK).map(({ id, score }) => ({ id: Number(id), score }));
      break;
  }

  const asRetrieved = (p: { id: number; score: number }) => ({ ...chunks[p.id]!, score: p.score });

  if (cell.reranking.kind === "none") return pool.slice(0, topK).map(asRetrieved);

  if (cell.reranking.kind === "api") {
    const candidates = pool.map((p) => ({ id: p.id, text: chunks[p.id]!.text }));
    // Repeat 1 may read the cache; later repeats must not, or the determinism
    // check would compare the first repeat with itself. The scores that come
    // back are the cross-encoder's own, on its own [0,1] scale — they replace
    // the retrieval score entirely rather than being blended with it, because
    // a weighted sum of a BM25 score and a rerank score is decided by scale.
    const ranked = repeat === 1
      ? await reranker.rerank(q.text, candidates, topK)
      : await reranker.rerankFresh(q.text, candidates, topK);
    return ranked.map(({ id, score }) => ({ ...chunks[id]!, score }));
  }

  // Relevance has to be on the same scale as similarity or lambda stops
  // meaning what it looks like. Cosine already is; BM25 and RRF are not, so
  // they are divided by the pool's top score — the same rescaling ceiling.ts
  // does, for the same reason.
  const top = pool[0]?.score ?? 1;
  const rescale = cell.retrieval.kind === "dense" ? (s: number) => s : (s: number) => (top === 0 ? 0 : s / top);
  // Vector similarity where there are vectors, token overlap where there are
  // not. Both live in [0,1].
  const similar = built.vectors
    ? (a: number, b: number) => cosine(built.vectors![a]!, built.vectors![b]!)
    : (a: number, b: number) => jaccard(built.tokenSets[a]!, built.tokenSets[b]!);

  return mmrRerank(
    pool.map((p) => ({ item: p, relevance: rescale(p.score) })),
    (a, b) => similar(a.id, b.id),
    { lambda: cell.reranking.params.lambda, k: topK },
  ).map((r) => asRetrieved(r.item));
}

// ─── run ─────────────────────────────────────────────────────────────────────

const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outDir = values.out ?? join(ROOT, "runs", runId);
mkdirSync(outDir, { recursive: true });

/**
 * One file per cell, named so that two embedders can share a directory.
 *
 * `cellId` is chunking × retrieval × reranking and deliberately says nothing
 * about the embedder — it is not an axis in configs.yaml. But the embedder IS
 * part of what a run measures, and without it in the filename a second run
 * under a different model silently overwrites the first: same names, same
 * directory, one table. That is precisely the comparison the design asks for
 * ("the best configurations × 2 embedding models"), so it has to be possible
 * to put both in one report — `rowLabel` already tells the rows apart, and
 * `comparable` already refuses to mix runs from different code.
 *
 * A lexical cell embeds nothing and keeps the bare name, so its file is
 * written once no matter which embedder the run names.
 */
const fileFor = (cell: Cell) =>
  join(outDir, needsEmbeddings(cell.retrieval)
    ? `${cellId(cell)}__${embedderLabel.replace(/\//g, "-")}.jsonl`
    : `${cellId(cell)}.jsonl`);

// Lines are appended as they arrive so a run that dies keeps what it did. The
// cost of appending is that a SECOND run writing the same filename does not
// replace the first — it adds a second header and a second set of results to
// the same file, and a reader that takes the first header and every result
// line would then average over a doubled set. The default output directory is
// stamped with a timestamp and hides this; --out into a directory that has
// already been used does not.
//
// So an existing file is refused here, before anything is chunked or embedded,
// rather than detected afterwards. --force truncates instead, which is the
// only honest way to reuse a directory: replace the cell, never extend it.
// packages/report refuses a multi-header file as well; this is the cheap check
// and that is the one that cannot be bypassed.
const clash = cells.map(fileFor).filter((f) => existsSync(f));
if (clash.length > 0 && !values.force) {
  console.error(
    `${clash.length} of ${cells.length} cells already have a file in ${outDir}:\n` +
    `${clash.slice(0, 5).map((f) => `  ${f.split("/").pop()}`).join("\n")}\n` +
    (clash.length > 5 ? `  ... and ${clash.length - 5} more\n` : "") +
    `\nAppending would give each of them a second header and a second set of\n` +
    `results, and a report over that file would average a doubled set. Pass\n` +
    `--force to replace them, or --out a fresh directory.`,
  );
  process.exit(2);
}

/** Truncates the cell's file, then writes its header. */
const startCell = (cell: Cell, line: Header) => {
  writeFileSync(fileFor(cell), `${JSON.stringify(line)}\n`);
};

const write = (file: string, line: object) => appendFileSync(file, `${JSON.stringify(line)}\n`);

const matrixOrder = new Map(cells.map((c, i) => [cellId(c), i]));

/**
 * What one pass over the question set costs this cell, in units.
 *
 * Derived from the configuration, not from the run's own spending: see the
 * note on `cost_units` in run.ts. So a cell whose vectors were all cached
 * reports the same units as the one that first embedded them, which is what
 * makes the column a property of the configuration.
 *
 * Query tokens are counted with gpt-tokenizer — invariant 2, the same
 * tokenizer as the chunkers — so the figure is on one scale with
 * `index_embed_tokens`. It is not the tokenizer the embedding vendor bills by,
 * and it cannot be: their count is not observable from here. Stated rather
 * than smoothed over, because it puts a few percent of slack on the dollar
 * figure and no amount of arithmetic here removes it.
 */
function costUnits(cell: Cell, built: Built): NonNullable<Header["cost_units"]> {
  const dense = needsEmbeddings(cell.retrieval);
  const api = cell.reranking.kind === "api";
  const questions = golden.questions.length;
  // Cohere bills one search per query against up to 100 documents; a wider
  // pool is billed as more than one, so the pool size decides both columns.
  const poolK = cell.reranking.kind === "none" ? topK : topK * RERANK_POOL;
  const documents = Math.min(poolK, built.chunks.length);

  return {
    index_embed_texts: dense ? built.chunks.length : 0,
    index_embed_tokens: dense ? built.chunks.reduce((n, c) => n + c.token_count, 0) : 0,
    query_embed_texts: dense ? questions : 0,
    query_embed_tokens: dense ? questionTokens : 0,
    rerank_searches: api ? questions * Math.ceil(documents / 100) : 0,
    rerank_documents: api ? questions * documents : 0,
  };
}

function header(cell: Cell, chunkCount: number, built: Built | null): Header {
  return {
    type: "header",
    cell: cellId(cell),
    order: matrixOrder.get(cellId(cell))!,
    chunking: { id: cell.chunking.id, strategy: cell.chunking.strategy, params: cell.chunking.params },
    retrieval: { id: cell.retrieval.id, kind: cell.retrieval.kind },
    reranking: { id: cell.reranking.id, kind: cell.reranking.kind },
    embedder: needsEmbeddings(cell.retrieval) ? embedderLabel : null,
    golden_sha: GOLDEN_SHA,
    git_sha: GIT_SHA,
    chunk_count: chunkCount,
    questions: golden.questions.length,
    top_k: topK,
    repeats,
    hit_coverage: HIT_COVERAGE,
    // A skipped cell was never built, so there is nothing measured to record.
    // Absent rather than zeroed: the report prints "not recorded", and a zero
    // would read as a configuration that costs nothing to run.
    cost_units: built ? costUnits(cell, built) : undefined,
    query_embed_ms_mean: built ? built.queryEmbedMs : undefined,
    started_at: new Date().toISOString(),
  };
}

console.log(`runs → ${outDir}`);
console.log(
  `golden_sha ${GOLDEN_SHA}   git_sha ${GIT_SHA}   ` +
  `${cells.length} cells × ${repeats} repeats × ${golden.questions.length} questions   ` +
  `embedder ${embedderName}\n`,
);
if (GIT_SHA.endsWith("-dirty")) {
  console.log(
    "git_sha carries -dirty: the tree has uncommitted changes, so HEAD does not\n" +
    "describe the code that ran. Invariant 6 makes these runs comparable only to\n" +
    "each other, which is what the suffix is for.\n",
  );
}

// Grouped by chunking config so each one is chunked and embedded once, not
// once per retrieval × reranking cell.
const byChunking = new Map<string, Cell[]>();
for (const cell of cells) {
  byChunking.set(cell.chunking.id, [...(byChunking.get(cell.chunking.id) ?? []), cell]);
}

const disagreed: { cell: string; slugs: string[] }[] = [];
const drifted: { cell: string; max: number; questions: number }[] = [];
const skipped: { cell: string; reason: string }[] = [];
let ran = 0;

for (const [chunkingId, group] of byChunking) {
  const verdicts = new Map(group.map((cell) => [cellId(cell), preflight(cell)]));

  for (const cell of group) {
    const v = verdicts.get(cellId(cell))!;
    if (v.run) continue;
    // A skipped cell still gets a file with a header, so the report can print
    // its reason in the row's place rather than leaving a blank.
    startCell(cell, header(cell, chunksFor(cell.chunking).length, null));
    write(fileFor(cell), { type: "skipped", reason: v.reason });
    skipped.push({ cell: cellId(cell), reason: v.reason });
    console.log(`skip  ${pad(cellId(cell), 42)} ${v.reason}`);
  }

  const live = group.filter((cell) => verdicts.get(cellId(cell))!.run);
  if (live.length === 0) continue;

  const wantDense = live.some((cell) => needsEmbeddings(cell.retrieval));
  const built = await build(group[0]!.chunking, wantDense);
  console.log(
    `${pad(chunkingId, 18)} ${String(built.chunks.length).padStart(6)} chunks   ` +
    `sha ${chunkSha(built.chunks.map((c) => c.text))}`,
  );

  for (const cell of live) {
    const file = fileFor(cell);
    startCell(cell, header(cell, built.chunks.length, built));

    const results: Result[] = [];
    for (let repeat = 1; repeat <= repeats; repeat++) {
      for (const q of golden.questions) {
        const started = performance.now();
        const waitedBefore = reranker.usage.waitedMs;
        const ranked = await retrieve(cell, built, q, repeat);
        // Minus whatever the rerank client spent waiting — its own pacing and
        // any rate-limit back-off. Those are properties of a trial key and of
        // this client's configuration, not of how fast the service answers,
        // and a p95 of 66 seconds measured through a 10-per-minute throttle
        // would be a number about the throttle.
        const latencyMs = performance.now() - started - (reranker.usage.waitedMs - waitedBefore);
        const line = {
          ...scoreQuestion(q, ranked, repeat, built.trivial.get(q.slug) ?? null),
          latency_ms: latencyMs,
        };
        // Appended as it arrives, not batched at the end of the cell.
        write(file, line);
        results.push(line);
      }
    }

    const slugs = repeatDisagreements(results);
    if (slugs.length > 0) disagreed.push({ cell: cellId(cell), slugs });
    const drift = scoreDrift(results);
    if (drift.max > 0) drifted.push({ cell: cellId(cell), ...drift });
    ran++;
    console.log(
      `  ran ${pad(cellId(cell), 42)} ${results.length} lines` +
      (slugs.length > 0 ? `   ${slugs.length} questions disagreed between repeats` : ""),
    );
  }
}

// ─── what the run found out about itself ─────────────────────────────────────

console.log(`\n${"─".repeat(72)}`);
console.log(`${ran} cells ran, ${skipped.length} skipped. Files in ${outDir}`);

if (openrouterEmbedder.usage.calls > 0 || openrouterEmbedder.usage.cached > 0) {
  const u = openrouterEmbedder.usage;
  console.log(
    `openrouter (${embedModel}): ${u.texts} texts embedded in ${u.calls} requests, ` +
    `${u.cached} from the disk cache, ${u.tokens} tokens, ` +
    // The vendor's own figure from usage.cost, not a price table multiplied by
    // a token count — so it is what was actually billed.
    `$${u.usd.toFixed(6)} billed`,
  );
}

if (localEmbedder.usage.embedded > 0 || localEmbedder.usage.cached > 0) {
  console.log(
    `local embeddings: ${localEmbedder.usage.embedded} computed, ` +
    `${localEmbedder.usage.cached} from the disk cache`,
  );
}

if (skipped.length > 0) {
  console.log(
    `\n${skipped.length} cells were not run. They are recorded in their files as\n` +
    `"not run", with the reason, rather than as a zero or a blank — a missing cell\n` +
    `that looks like a bad result is worse than an empty one.`,
  );
}

if (drifted.length > 0) {
  // Reported next to determinism and not as part of it: the ordering is what
  // every metric consumes, and a score that moves below the resolution of any
  // decision made from it is a different fact about a different thing.
  console.log(
    `\nScore drift: ${drifted.length} cell${drifted.length === 1 ? "" : "s"} returned identical\n` +
    `rankings with scores that were not identical. That is what a hosted model\n` +
    `looks like when it is stable in the way that matters and not in the way that\n` +
    `does not.`,
  );
  for (const d of drifted) {
    console.log(`  ${pad(d.cell, 42)} up to ${d.max.toExponential(2)} on ${d.questions} questions`);
  }
}

if (disagreed.length === 0) {
  console.log(
    `\nDeterminism: all ${repeats} repeats of every cell returned identical rankings.\n` +
    `That is what the repeats are for. It also means noise = 0 in the power\n` +
    `calculation, which is the assumption mdePaired is called with — verified\n` +
    `here rather than assumed.`,
  );
} else {
  console.log(
    `\nDeterminism: FAILED on ${disagreed.length} cells. This is a finding, not noise to\n` +
    `average away — a ranking that disagrees with itself is indistinguishable from\n` +
    `a real effect, and every paired comparison in the report assumes it does not\n` +
    `happen.`,
  );
  for (const d of disagreed) {
    console.log(`  ${pad(d.cell, 42)} ${d.slugs.length} questions: ${d.slugs.slice(0, 5).join(", ")}`);
  }
}

console.log(`\nNext: pnpm report --dir ${outDir}`);
