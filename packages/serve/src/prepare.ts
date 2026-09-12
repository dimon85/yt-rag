// pnpm serve:prepare — cache/ in, one servable file out.
//
//   pnpm serve:prepare                      # the winning configuration
//   pnpm serve:prepare --chunking fixed-512 # any other one
//
// Run on a machine with a warm cache; the container never sees cache/ at all.
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, existsSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { fixedChunks } from "../../chunking/src/fixed.ts";
import { windowChunks } from "../../chunking/src/window.ts";
import { CachedLocalEmbedder } from "../../embed/src/local-cache.ts";
import { LOCAL_DIM } from "../../embed/src/local.ts";
import { CACHE_DIR, loadCorpus, ROOT } from "../../ingest/src/corpus.ts";
import type { Segment } from "../../ingest/src/text.ts";
import { packVectors, type ServeIndex } from "./index-format.ts";

const { values } = parseArgs({
  options: {
    chunking: { type: "string", default: "fixed-512-ov128" },
    threshold: { type: "string" },
    out: { type: "string" },
  },
});

/**
 * The chunkers by id, read from configs.yaml the way the runner reads it.
 *
 * Imported from the file rather than restated, so a server can never be built
 * from a configuration the ablation did not measure — the whole value of the
 * numbers in the README is that they describe this exact arrangement.
 */
const configs = (await import("../../evals/src/configs.ts"));
const cfg = configs.loadConfigs();
const chunking = cfg.chunking.find((c) => c.id === values.chunking);
if (!chunking) {
  console.error(`no chunking config "${values.chunking}" in configs.yaml — have ` +
    cfg.chunking.map((c) => c.id).join(", "));
  process.exit(2);
}

const chunk = chunking.strategy === "fixed"
  ? (s: Segment[]) => fixedChunks(s, chunking.params.tokens, chunking.params.overlap)
  : (s: Segment[]) => windowChunks(s, chunking.params.seconds, chunking.params.overlap_s);

const corpus = loadCorpus();
const chunks: ServeIndex["chunks"] = [];
const videos: ServeIndex["videos"] = {};

for (const v of corpus.videos) {
  const path = join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`);
  if (!existsSync(path)) continue;
  const segments: Segment[] = JSON.parse(readFileSync(path, "utf8")).segments;
  for (const c of chunk(segments)) {
    chunks.push({ video: v.youtube_id, start_s: c.start_s, end_s: c.end_s, text: c.text });
  }
  const metaPath = join(CACHE_DIR, "meta", `${v.youtube_id}.json`);
  if (existsSync(metaPath)) {
    // Two shapes are on disk: the flat record, and one wrapped as
    // `{kind: "ok", meta: {...}}`. Both are real files written by different
    // passes of the ingest, so both are read rather than one being declared
    // correct — and a file matching neither is skipped rather than written as
    // a citation with undefined fields.
    const raw = JSON.parse(readFileSync(metaPath, "utf8"));
    const m = raw.meta ?? raw;
    if (typeof m.title === "string" && typeof m.channel === "string") {
      videos[v.youtube_id] = {
        title: m.title,
        channel: m.channel,
        published_at: String(m.published_at ?? ""),
      };
    }
  }
}

if (chunks.length === 0) {
  console.error("no chunks — is cache/transcripts populated? run pnpm ingest first");
  process.exit(2);
}

const withoutMeta = new Set(chunks.map((c) => c.video)).size - Object.keys(videos).length;
console.log(`${chunks.length} chunks from ${new Set(chunks.map((c) => c.video)).size} videos, embedding…`);
if (withoutMeta > 0) {
  // Not fatal: the link and the timestamp are the citation the spec requires,
  // and a title is a courtesy. Reported so it is a known gap rather than a
  // blank field somebody notices in production.
  console.log(`  ${withoutMeta} video(s) have no usable metadata — their hits cite the link alone`);
}
const embedder = new CachedLocalEmbedder(join(CACHE_DIR, "embeddings"));
const vectors = await embedder.embedAll(chunks.map((c) => c.text), {
  onProgress: (done, total) => process.stdout.write(`\r  ${done}/${total}   `),
});
process.stdout.write("\r");
console.log(`  ${embedder.usage.cached} vectors from cache, ${embedder.usage.embedded} computed`);

const gitSha = (() => {
  try {
    return execFileSync("git", ["rev-parse", "--short=12", "HEAD"], { cwd: ROOT }).toString().trim();
  } catch {
    return "unknown";
  }
})();

// Measured, not chosen: the median top-1 of the answerable golden questions
// under this configuration, which is where the report measures its
// false-positive rate. Overridable because a deployment may want to answer
// more and be wrong more, or the reverse — but the default is the number the
// table describes.
const threshold = values.threshold ? Number(values.threshold) : 0.03202;

const index: ServeIndex = {
  config: {
    chunking: chunking.id,
    retrieval: "hybrid",
    embedder: "local",
    git_sha: gitSha,
    built_at: new Date().toISOString(),
  },
  dim: LOCAL_DIM,
  threshold,
  chunks,
  videos,
  vectors: packVectors(vectors),
};

const out = values.out ?? join(ROOT, "serve-index.json");
mkdirSync(join(out, ".."), { recursive: true });
writeFileSync(out, JSON.stringify(index));
console.log(
  `wrote ${out} — ${(statSync(out).size / 1e6).toFixed(1)} MB, ` +
  `${chunking.id}, threshold ${threshold}, git_sha ${gitSha}`,
);
