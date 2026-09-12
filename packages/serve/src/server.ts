// The public endpoint. Spans with timestamps and links; no generated prose.
//
//   pnpm serve                    # port 8080, serve-index.json
//   PORT=3000 INDEX=/data/i.json pnpm serve
//
//   GET /health   → what is loaded and which commit produced it
//   GET /search?q=…&k=5
//
// It does not answer questions, and that is the product decision the whole
// project rests on: retrieval quality is what is measured, and an LLM writing
// prose on top would hide exactly the thing being measured. So the response is
// the evidence — excerpt, timestamp, link — and the reader decides.
//
// No framework. Node has an HTTP server and this has three routes; a
// dependency here would be a line in the lockfile to save twenty lines of
// code.
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { embedLocal } from "../../embed/src/local.ts";
import { ServeIndex } from "./index-format.ts";
import { prepare, retrieve } from "./search.ts";

const PORT = Number(process.env.PORT ?? 8080);
const INDEX_PATH = process.env.INDEX ?? "serve-index.json";
const MAX_QUERY_CHARS = 500;

console.log(`loading ${INDEX_PATH}…`);
const index = ServeIndex.parse(JSON.parse(readFileSync(INDEX_PATH, "utf8")));
const ready = prepare(index);
console.log(
  `${index.chunks.length} chunks, ${Object.keys(index.videos).length} videos, ` +
  `${index.config.chunking} / ${index.config.retrieval} / ${index.config.embedder}, ` +
  `git_sha ${index.config.git_sha}`,
);

// Warmed at boot, not on the first request: transformers.js loads ~90MB the
// first time it embeds anything, and paying that inside a user's request makes
// the first query look like a ten-second retrieval.
console.log("warming the embedding model…");
const warmStart = performance.now();
await embedLocal(["warm"]);
console.log(`  ready in ${((performance.now() - warmStart) / 1000).toFixed(1)}s`);

const json = (res: import("node:http").ServerResponse, status: number, body: unknown) => {
  const payload = JSON.stringify(body, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
    // The corpus is third-party content served as short excerpts; nothing here
    // should end up in someone else's page as if it were theirs.
    "x-content-type-options": "nosniff",
  });
  res.end(payload);
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);

  if (req.method !== "GET") return json(res, 405, { error: "only GET" });

  if (url.pathname === "/health") {
    return json(res, 200, {
      ok: true,
      chunks: index.chunks.length,
      videos: Object.keys(index.videos).length,
      config: index.config,
      threshold: index.threshold,
    });
  }

  if (url.pathname !== "/search") return json(res, 404, { error: "try /search?q=…" });

  const q = (url.searchParams.get("q") ?? "").trim();
  if (!q) return json(res, 400, { error: "q is required" });
  if (q.length > MAX_QUERY_CHARS) {
    return json(res, 400, { error: `q is longer than ${MAX_QUERY_CHARS} characters` });
  }
  const k = Math.min(Math.max(Number(url.searchParams.get("k") ?? 5) || 5, 1), 20);

  const started = performance.now();
  try {
    const [queryVector] = await embedLocal([q]);
    const answer = retrieve(ready, queryVector!, q, k);
    return json(res, 200, {
      query: q,
      took_ms: Math.round(performance.now() - started),
      ...answer,
      // Every response carries what produced it. A span quoted from this
      // endpoint can be traced to a configuration and a commit, which is the
      // same property the JSONL headers give the report.
      config: index.config,
    });
  } catch (e) {
    console.error(`search failed for ${JSON.stringify(q)}:`, e);
    return json(res, 500, { error: "search failed" });
  }
});

server.listen(PORT, () => console.log(`listening on http://localhost:${PORT}`));

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    console.log(`\n${signal} — closing`);
    server.close(() => process.exit(0));
  });
}
