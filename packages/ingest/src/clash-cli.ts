// Pairs of passages by different authors, for annotation.
//
//   pnpm pairs --cells                       # material per cell, before spending anything
//   pnpm pairs --tool claude-code --relation clash --sides any
//   pnpm pairs --tool mcp --relation complement
//
// Prints candidates with both spans. It does not write questions and does not
// touch questions.yaml: what comes back is a seed to write one from later,
// with the passage out of view. See clash.ts for why.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { parse as parseYaml } from "yaml";
import { GoogleGenAI } from "@google/genai";
import {
  type Annotated, ClashSchema, COMPLEMENT_PROMPT, ComplementSchema, duplicateOf,
  type Passage, POOLED_PROMPT, PROMPT, renderDated, renderPassages, REVISION_PROMPT,
  RevisionSchema, verify, verifyComplements, verifyRevisions,
} from "./clash.ts";
import { AD_PATTERN, hasPhrase, score, tile } from "./candidates.ts";
import { CACHE_DIR, loadCorpus, ROOT, type Stance } from "./corpus.ts";
import { normalizeTranscript, type Segment } from "./text.ts";

const HELP = `
usage: pnpm pairs --tool ID [--relation R] [--topic ID] [--sides S] [--contra S]
                  [--width S] [--limit N] [--model ID]
       pnpm pairs --cells [--sides S]

  --relation R  clash | complement | revision (default clash)
                clash finds passages that cannot both be true, for a
                contradiction question. complement finds passages that each
                cover part of one subject where neither covers it alone, for a
                comparative question — the cheaper of the two, since two
                authors need only cover different parts of a thing rather than
                disagree about it. revision finds one author contradicting his
                own earlier video, which corpus.yaml names as the second source
                of contradictions and nothing was mining; it needs --channel.
  --channel N   restrict to one channel, exactly as named in corpus.yaml
  --tool ID     tool the pair must be about. Required unless --channel or
                --topic narrows the search instead
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
        relation: { type: "string", default: "clash" },
        channel: { type: "string" },
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

if (!["clash", "complement", "revision"].includes(values.relation!)) {
  usage("--relation must be clash, complement or revision");
}
const complement = values.relation === "complement";
const revision = values.relation === "revision";
// A complement is defined across authors, not across stances, so the stance
// split has nothing to say about it.
const pooled = values.sides === "any" || complement || revision;
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
function passagesFor(tool: string | undefined, topic?: string): { pro: Sourced[]; contra: Sourced[] } {
  // Optional, so a revision search can read everything one author said. The
  // tool filter is what made the first pass miss most of a channel: a claim
  // and its later retraction need not both name the tool they are about.
  const toolDef = tool
    ? cfg.tools.find((t) => t.id === tool) ?? usage(`unknown tool ${tool}`)
    : undefined;
  const topicDef = topic
    ? cfg.topics.find((t) => t.id === topic) ?? usage(`unknown topic ${topic}`)
    : undefined;

  const pro: Sourced[] = [];
  const contra: Sourced[] = [];

  for (const v of cfg.videos) {
    if (values.channel && v.channel !== values.channel) continue;
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
      if (toolDef && !toolDef.aliases.some((a) => hasPhrase(normalized, a))) continue;
      if (topicDef && !topicDef.keywords.some((k) => hasPhrase(normalized, k))) continue;
      side.push({
        id: `${side === pro ? "p" : "c"}${side.length + 1}`,
        video: v.youtube_id,
        channel: v.channel,
        published_at: v.published_at,
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

if (!values.tool && !values.channel && !values.topic) {
  usage("--tool is required, unless --channel or --topic narrows the search");
}

if (revision && !values.channel) {
  // A revision is defined within one author, so pooling every channel would
  // only invite the model to pair two people and call it a change of mind.
  console.error("--relation revision needs --channel. Channels in the corpus:\n");
  const counts = new Map<string, number>();
  for (const v of cfg.videos) counts.set(v.channel, (counts.get(v.channel) ?? 0) + 1);
  for (const [name, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    console.error(`  ${String(n).padStart(2)} videos  ${name}`);
  }
  process.exit(2);
}

const all = passagesFor(values.tool, values.topic);
const pro = rank(all.pro).slice(0, perSide);
const contra = rank(all.contra).slice(0, perSide);

console.log(
  `${values.tool ?? "all tools"}${values.topic ? ` / ${values.topic}` : ""}` +
  `${values.channel ? ` / ${values.channel}` : ""}: ` +
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
/**
 * One retry loop around the request.
 *
 * A run died on `SocketError: other side closed` after assembling its
 * passages, which threw away the assembly and the request alike. The embedder
 * has waited on transient failures since its own first run; this had nothing,
 * for no better reason than that it was written later.
 *
 * A quota refusal is not retried here: `quotaKind` in the embedder exists
 * because retrying a daily limit only burns time, and the same holds for a
 * generation quota.
 */
async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  for (let i = 1; ; i++) {
    try {
      return await fn();
    } catch (e) {
      const message = String((e as Error).message ?? e);
      if (i >= attempts || /quota|RESOURCE_EXHAUSTED|NOT_FOUND|API key/i.test(message)) throw e;
      const waitMs = 2000 * i;
      console.error(`  request failed (${message.slice(0, 80)}), retrying in ${waitMs / 1000}s`);
      await new Promise((r) => setTimeout(r, waitMs));
    }
  }
}

const res = await withRetry(() => ai.models.generateContent({
  model: values.model!,
  contents: revision
    ? `${REVISION_PROMPT}\n\n${renderDated(pro)}`
    : complement
    ? `${COMPLEMENT_PROMPT}\n\n${renderPassages(pro)}`
    : pooled
      ? `${POOLED_PROMPT}\n\n${renderPassages(pro)}`
      : `${PROMPT}\n\n${renderPassages(pro, contra)}`,
  config: {
    responseMimeType: "application/json",
    responseSchema: revision
      ? {
        type: "object",
        properties: {
          revisions: {
            type: "array",
            items: {
              type: "object",
              properties: {
                claim: { type: "string" },
                earlier_id: { type: "string" },
                later_id: { type: "string" },
                why: { type: "string" },
              },
              required: ["claim", "earlier_id", "later_id", "why"],
            },
          },
        },
        required: ["revisions"],
      }
      : complement
      ? {
        type: "object",
        properties: {
          pairs: {
            type: "array",
            items: {
              type: "object",
              properties: {
                subject: { type: "string" },
                a: { type: "string" },
                b: { type: "string" },
                a_only: { type: "string" },
                b_only: { type: "string" },
              },
              required: ["subject", "a", "b", "a_only", "b_only"],
            },
          },
        },
        required: ["pairs"],
      }
      : {
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
}));

const raw = JSON.parse(res.text ?? "{}");
const parsed = revision
  ? RevisionSchema.safeParse(raw)
  : complement
    ? ComplementSchema.safeParse(raw)
    : ClashSchema.safeParse(raw);
if (!parsed.success) {
  console.error("the model returned something the schema rejects:");
  console.error(parsed.error.issues.map((i) => `  ${i.path.join(".")}: ${i.message}`).join("\n"));
  console.error(`\nraw:\n${res.text}`);
  process.exit(1);
}

const offered = [...pro, ...contra];

// What the set already covers. Read straight from the YAML rather than through
// the golden loader: that lives in packages/evals, which already imports from
// here, and only the slug and the spans are needed.
const annotated: Annotated[] = (
  parseYaml(readFileSync(join(ROOT, "golden", "questions.yaml"), "utf8")).questions ?? []
).map((q: any) => ({ slug: q.slug, gold: q.gold ?? [] }));

const dupe = (a: Passage, b: Passage) => duplicateOf(a, b, annotated);
const where = (p: Passage) =>
  `${p.channel} — youtu.be/${p.video}?t=${Math.floor(p.start_s)}` +
  `  (${p.start_s.toFixed(2)}–${p.end_s.toFixed(2)})` +
  // Marked, not dropped: a 90-second window can hold an ad read and a real
  // claim, and narrowing the span is a judgement only a reader can make.
  (AD_PATTERN.test(p.text) ? "  [contains an ad read — check the span]" : "");

let proposed: number;
let verified: number;
const rejections: { reason: string; claim: string }[] = [];

if (revision) {
  const { revisions } = parsed.data as { revisions: import("./clash.ts").Revision[] };
  const { kept, rejected } = verifyRevisions(revisions, offered);
  proposed = revisions.length;
  verified = kept.length;
  console.log(`\n${proposed} proposed, ${verified} verified\n`);
  for (const r of kept) {
    const already = dupe(r.earlier, r.later);
    console.log(r.claim);
    if (r.claimProblem) console.log(`  RESTATE: ${r.claimProblem}`);
    if (already) console.log(`  ALREADY ASKED as ${already}`);
    console.log(`  ${r.why}`);
    console.log(`  earlier ${r.earlier.published_at}  ${where(r.earlier)}`);
    console.log(`  later   ${r.later.published_at}  ${where(r.later)}\n`);
  }
  rejections.push(...rejected.map(({ revision: v, reason }) => ({ reason, claim: v.claim })));
} else if (complement) {
  const { pairs } = parsed.data as { pairs: import("./clash.ts").Complement[] };
  const { kept, rejected } = verifyComplements(pairs, offered);
  proposed = pairs.length;
  verified = kept.length;
  console.log(`\n${proposed} proposed, ${verified} verified\n`);
  for (const c of kept) {
    const already = dupe(c.a, c.b);
    console.log(c.subject);
    if (c.claimProblem) console.log(`  RESTATE: ${c.claimProblem}`);
    if (already) console.log(`  ALREADY ASKED as ${already}`);
    console.log(`  a only  ${c.a_only}`);
    console.log(`  b only  ${c.b_only}`);
    console.log(`  a       ${where(c.a)}`);
    console.log(`  b       ${where(c.b)}\n`);
  }
  rejections.push(...rejected.map(({ pair, reason }) => ({ reason, claim: pair.subject })));
} else {
  const { clashes } = parsed.data as { clashes: import("./clash.ts").Clash[] };
  const { kept, rejected } = verify(clashes, offered);
  proposed = clashes.length;
  verified = kept.length;
  console.log(`\n${proposed} proposed, ${verified} verified\n`);
  for (const c of kept) {
    const already = dupe(c.pro, c.contra);
    console.log(`${c.directness.toUpperCase()}  ${c.claim}`);
    if (c.claimProblem) console.log(`  RESTATE: ${c.claimProblem}`);
    if (already) console.log(`  ALREADY ASKED as ${already}`);
    console.log(`  ${c.why}`);
    console.log(`  pro     ${where(c.pro)}`);
    console.log(`  contra  ${where(c.contra)}\n`);
  }
  rejections.push(...rejected.map(({ clash, reason }) => ({ reason, claim: clash.claim })));
}

if (rejections.length > 0) {
  console.log(`rejected ${rejections.length}:`);
  for (const r of rejections) console.log(`  ${r.reason} — ${r.claim}`);
}

console.log(
  "\nBoth spans need watching before either becomes gold. What the model returns\n" +
  "is a seed, not a question: write the question from it with the passage out of\n" +
  "view, or term matching answers it and it stops separating configurations.",
);
