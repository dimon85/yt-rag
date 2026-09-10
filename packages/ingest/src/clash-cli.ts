// Passages that disagree, for annotation.
//
//   pnpm clashes --cells                     # material per cell, before spending anything
//   pnpm clashes --tool claude-code --topic limits
//   pnpm clashes --tool cursor --contra skeptical+practical
//
// Prints candidate clashes with both spans and the disputed claim. It does not
// write questions and does not touch questions.yaml: the claim is a seed to
// write one from later, with the passage out of view. See clash.ts for why.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { GoogleGenAI } from "@google/genai";
import {
  ClashSchema, type Passage, POOLED_PROMPT, PROMPT, renderPassages, verify,
} from "./clash.ts";
import { hasPhrase, score, tile } from "./candidates.ts";
import { CACHE_DIR, loadCorpus, type Stance } from "./corpus.ts";
import { normalizeTranscript, type Segment } from "./text.ts";

const HELP = `
usage: pnpm clashes --tool ID [--topic ID] [--sides S] [--contra S]
                    [--width S] [--limit N] [--model ID]
       pnpm clashes --cells [--sides S]

  --tool ID     tool the disagreement must be about (required)
  --topic ID    restrict to passages matching the topic's spoken keywords
  --sides S     stance | any (default stance)
                stance splits passages by channel stance and tells the model
                which side is which. any pools them and lets it find
                disagreement anywhere — the right mode wherever one stance
                barely covers the tool, which is measurably the case for mcp
  --contra S    with --sides stance, which stances form the contra side
                (default skeptical) — skeptical | skeptical+practical
  --width S     passage width in seconds (default 45)
  --limit N     passages per side (default 40)
  --model ID    generation model (default gemini-3.1-pro-preview)
  --cells       material per tool x topic per side, and spend nothing
`.trimStart();

function usage(message: string): never {
  console.error(`${message}\n\n${HELP}`);
  process.exit(2);
}

const argv = process.argv.slice(2)
  .filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--")));
const { values } = (() => {
  try {
    return parseArgs({
      args: argv,
      options: {
        tool: { type: "string" },
        topic: { type: "string" },
        sides: { type: "string", default: "stance" },
        contra: { type: "string", default: "skeptical" },
        width: { type: "string", default: "45" },
        limit: { type: "string", default: "40" },
        model: { type: "string", default: "gemini-3.1-pro-preview" },
        cells: { type: "boolean", default: false },
      },
      allowPositionals: false,
    });
  } catch (e) {
    usage((e as Error).message);
  }
})();

const cfg = loadCorpus();
const stanceOf = new Map<string, Stance>(cfg.channels.map((c) => [c.name, c.stance]));

const CONTRA_SETS: Record<string, Stance[]> = {
  skeptical: ["skeptical"],
  "skeptical+practical": ["skeptical", "practical"],
};
const contraStances = CONTRA_SETS[values.contra!]
  ?? usage(`--contra must be one of ${Object.keys(CONTRA_SETS).join(" | ")}`);

const pooled = values.sides === "any";
if (!["stance", "any"].includes(values.sides!)) usage("--sides must be stance or any");

const widthS = Number(values.width);
const perSide = Number(values.limit);

// ─── passages ────────────────────────────────────────────────────────────────

/** A passage plus the segments it came from, which `score` needs and the model never sees. */
type Sourced = Passage & { segments: Segment[] };

/**
 * Every window mentioning the tool, tiled rather than sliding.
 *
 * Tiling matters here for the same reason it did for sourcing gold spans: a
 * half-stepped window plus greedy deduplication keeps only the densest passage
 * in each neighbourhood, and the densest passage is the one every retriever
 * already finds. A clash is more likely to sit in an ordinary aside.
 */
function passagesFor(tool: string, topic?: string): { pro: Sourced[]; contra: Sourced[] } {
  const toolDef = cfg.tools.find((t) => t.id === tool) ?? usage(`unknown tool ${tool}`);
  const topicDef = topic
    ? cfg.topics.find((t) => t.id === topic) ?? usage(`unknown topic ${topic}`)
    : undefined;

  const pro: Sourced[] = [];
  const contra: Sourced[] = [];

  for (const v of cfg.videos) {
    const stance = stanceOf.get(v.channel);
    if (!stance) continue;
    // Pooled mode puts everything in one list, so `pro` is simply "the list".
    const side = pooled
      ? pro
      : stance === "hype" ? pro : contraStances.includes(stance) ? contra : null;
    if (!side) continue;

    const path = join(CACHE_DIR, "transcripts", `${v.youtube_id}.json`);
    const segments: Segment[] = JSON.parse(readFileSync(path, "utf8")).segments;

    for (const w of tile(segments, widthS)) {
      const normalized = normalizeTranscript(w.segments);
      if (!toolDef.aliases.some((a) => hasPhrase(normalized, a))) continue;
      if (topicDef && !topicDef.keywords.some((k) => hasPhrase(normalized, k))) continue;
      side.push({
        id: `${side === pro ? "p" : "c"}${side.length + 1}`,
        video: v.youtube_id,
        channel: v.channel,
        start_s: w.start_s,
        end_s: w.end_s,
        text: w.text,
        segments: w.segments,
      });
    }
  }
  return { pro, contra };
}

/**
 * Highest-scoring passages first, so a truncated list keeps the best of it.
 *
 * `score` reads tool and topic matches out of the window's segments, not its
 * text, so the segments have to be carried this far — passing an empty array
 * would silently zero both and leave the ranking to number and claim markers
 * alone.
 */
function rank(ps: Sourced[]): Sourced[] {
  const tools = cfg.tools.map((t) => ({ id: t.id, aliases: t.aliases }));
  const topics = cfg.topics.map((t) => ({ id: t.id, keywords: t.keywords }));
  return [...ps]
    .map((p) => ({ p, s: score(p, tools, topics).total }))
    .sort((a, b) => b.s - a.s || a.p.id.localeCompare(b.p.id))
    .map(({ p }) => p);
}

// ─── cells ───────────────────────────────────────────────────────────────────

if (values.cells) {
  console.log(
    pooled
      ? `passages per cell, ${widthS}s windows, all stances pooled\n`
      : `passages per side, ${widthS}s windows, contra = ${contraStances.join("+")}\n`,
  );
  console.log("tool".padEnd(14), "topic".padEnd(16), "pro".padStart(5), "contra".padStart(7));
  for (const t of cfg.tools) {
    for (const topic of cfg.topics) {
      const { pro, contra } = passagesFor(t.id, topic.id);
      if (pro.length === 0 && contra.length === 0) continue;
      // Pooled needs two passages from different videos; split needs one each.
      const usable = pooled
        ? new Set(pro.map((p) => p.video)).size > 1
        : pro.length > 0 && contra.length > 0;
      console.log(
        t.id.padEnd(14), topic.id.padEnd(16),
        String(pro.length).padStart(5), String(contra.length).padStart(7),
        usable ? "" : pooled ? "  ← fewer than two authors" : "  ← one side empty, nothing to compare",
      );
    }
  }
  process.exit(0);
}

// ─── find ────────────────────────────────────────────────────────────────────

if (!values.tool) usage("--tool is required");

const all = passagesFor(values.tool, values.topic);
const pro = rank(all.pro).slice(0, perSide);
const contra = rank(all.contra).slice(0, perSide);

console.log(
  `${values.tool}${values.topic ? ` / ${values.topic}` : ""}: ` +
  (pooled
    ? `${pro.length} passages of ${all.pro.length}, pooled`
    : `${pro.length} pro of ${all.pro.length}, ${contra.length} contra of ${all.contra.length}`),
);
const authors = new Set([...pro, ...contra].map((p) => p.video)).size;
if (authors < 2) {
  console.log(`only ${authors} video contributes — nothing to compare, and no request made.`);
  process.exit(0);
}
if (!pooled && (pro.length === 0 || contra.length === 0)) {
  console.log(
    "one side is empty — nothing to compare, and no request made.\n" +
    "Try --sides any: the stance split finds nothing where one stance barely covers the tool.",
  );
  process.exit(0);
}

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) usage("GEMINI_API_KEY is not set — put it in .env, see .env.example");

const ai = new GoogleGenAI({ apiKey });
const res = await ai.models.generateContent({
  model: values.model!,
  contents: pooled
    ? `${POOLED_PROMPT}\n\n${renderPassages(pro)}`
    : `${PROMPT}\n\n${renderPassages(pro, contra)}`,
  config: {
    responseMimeType: "application/json",
    responseSchema: {
      type: "object",
      properties: {
        clashes: {
          type: "array",
          items: {
            type: "object",
            properties: {
              claim: { type: "string" },
              pro: { type: "string" },
              contra: { type: "string" },
              why: { type: "string" },
              directness: { type: "string", enum: ["flat", "partial"] },
            },
            required: ["claim", "pro", "contra", "why", "directness"],
          },
        },
      },
      required: ["clashes"],
    },
  },
});

const parsed = ClashSchema.safeParse(JSON.parse(res.text ?? "{}"));
if (!parsed.success) {
  console.error("the model returned something the schema rejects:");
  console.error(parsed.error.issues.map((i) => `  ${i.path.join(".")}: ${i.message}`).join("\n"));
  console.error(`\nraw:\n${res.text}`);
  process.exit(1);
}

const { kept, rejected } = verify(parsed.data.clashes, [...pro, ...contra]);

console.log(`\n${parsed.data.clashes.length} proposed, ${kept.length} verified\n`);
for (const c of kept) {
  console.log(`${c.directness.toUpperCase()}  ${c.claim}`);
  console.log(`  ${c.why}`);
  console.log(`  pro     ${c.pro.channel} — youtu.be/${c.pro.video}?t=${Math.floor(c.pro.start_s)}` +
    `  (${c.pro.start_s.toFixed(2)}–${c.pro.end_s.toFixed(2)})`);
  console.log(`  contra  ${c.contra.channel} — youtu.be/${c.contra.video}?t=${Math.floor(c.contra.start_s)}` +
    `  (${c.contra.start_s.toFixed(2)}–${c.contra.end_s.toFixed(2)})\n`);
}

if (rejected.length > 0) {
  console.log(`rejected ${rejected.length}:`);
  for (const { clash, reason } of rejected) console.log(`  ${reason} — ${clash.claim}`);
}

console.log(
  "\nBoth spans need watching before either becomes gold. The claim is a seed:\n" +
  "write the question from it with the passage out of view, or term matching\n" +
  "answers it and the question stops separating configurations.",
);
