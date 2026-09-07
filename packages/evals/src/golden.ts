// The golden set: schema, validation, and the hash that makes a run
// comparable. Everything here is about one thing — a question set that quietly
// says something other than what it looks like is worse than no question set,
// because the numbers still come out and still look reasonable.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";
import { z } from "zod";
// Relative across packages on purpose: the project runs TypeScript directly
// via type stripping, with no build step, so a package-name import has no
// resolvable entry point. corpus.ts and text.ts are shared infrastructure
// rather than ingest-specific — when a third consumer appears they should
// move to their own package and these imports become ordinary ones.
import { ROOT } from "../../ingest/src/corpus.ts";

export const GOLDEN_PATH = join(ROOT, "golden", "questions.yaml");

export const Kind = z.enum(["factual", "comparative", "contradiction", "negative"]);
export type Kind = z.infer<typeof Kind>;

/** One annotated span: where the answer actually is. */
export const Gold = z.object({
  video: z.string().length(11),
  start_s: z.number().nonnegative(),
  end_s: z.number().positive(),
  /** Required for contradiction questions, meaningless elsewhere. */
  side: z.enum(["pro", "contra"]).optional(),
  note: z.string().optional(),
});
export type Gold = z.infer<typeof Gold>;

export const Question = z.object({
  slug: z.string().regex(/^[a-z0-9]+(-[a-z0-9]+)*$/, "slug must be kebab-case"),
  text: z.string().min(10),
  kind: Kind,
  topics: z.array(z.string()).default([]),
  tools: z.array(z.string()).default([]),
  gold: z.array(Gold).default([]),
});
export type Question = z.infer<typeof Question>;

export const GoldenSet = z.object({
  version: z.number().int().positive(),
  questions: z.array(Question).default([]),
});
export type GoldenSet = z.infer<typeof GoldenSet>;

/**
 * The hash recorded in `runs.golden_sha`.
 *
 * Over a canonical projection, not the raw bytes. Hashing the file would make
 * a reindent or a reordered list invalidate every earlier run, which is a
 * false negative about comparability — the questions did not change.
 *
 * What is in it: slug, text, kind, and the gold spans, each sorted so that
 * moving a question up the file changes nothing.
 *
 * What is deliberately out: `note`, which is a comment to the annotator, and
 * `topics` / `tools`, which slice the report rather than decide any retrieval
 * result. `text` IS in: rephrasing a question makes a different question, and
 * the old numbers do not carry over.
 */
export function goldenSha(set: GoldenSet): string {
  const canonical = set.questions
    .map((q) => ({
      slug: q.slug,
      text: q.text.trim().replace(/\s+/g, " "),
      kind: q.kind,
      gold: [...q.gold]
        .map((g) => ({
          video: g.video,
          start_s: Number(g.start_s.toFixed(2)),
          end_s: Number(g.end_s.toFixed(2)),
          side: g.side ?? null,
        }))
        .sort((a, b) => a.video.localeCompare(b.video) || a.start_s - b.start_s),
    }))
    .sort((a, b) => a.slug.localeCompare(b.slug));

  return createHash("sha256").update(JSON.stringify(canonical)).digest("hex").slice(0, 16);
}

export function loadGolden(path = GOLDEN_PATH): GoldenSet {
  return GoldenSet.parse(YAML.parse(readFileSync(path, "utf8")));
}

// ─── validation ──────────────────────────────────────────────────────────────

export type Issue = { level: "error" | "warning"; slug: string; message: string };

export type CorpusFacts = {
  /** youtube_id -> duration in seconds */
  durations: Map<string, number>;
};

/**
 * Structural rules. Each one exists because breaking it makes a metric mean
 * something other than what it is called.
 */
export function validateStructure(set: GoldenSet, corpus: CorpusFacts): Issue[] {
  const issues: Issue[] = [];
  const err = (slug: string, message: string) => issues.push({ level: "error", slug, message });
  const warn = (slug: string, message: string) => issues.push({ level: "warning", slug, message });

  const seen = new Set<string>();
  for (const q of set.questions) {
    if (seen.has(q.slug)) err(q.slug, "duplicate slug");
    seen.add(q.slug);

    for (const g of q.gold) {
      if (g.end_s <= g.start_s) err(q.slug, `span ends before it starts (${g.start_s}..${g.end_s})`);
      const duration = corpus.durations.get(g.video);
      if (duration === undefined) err(q.slug, `video ${g.video} is not in corpus.yaml`);
      else if (g.end_s > duration + 1) {
        err(q.slug, `span ${g.end_s}s runs past the end of ${g.video} (${duration}s)`);
      }
    }

    const videos = new Set(q.gold.map((g) => g.video));

    switch (q.kind) {
      case "factual":
        // "The answer sits in one specific place in one video." More than one
        // span means it is really a comparative question, and scoring it as
        // factual credits a retriever that found either half.
        if (q.gold.length !== 1) err(q.slug, `factual needs exactly 1 gold span, has ${q.gold.length}`);
        break;

      case "comparative":
        // Needs two different videos, not two spans of the same one. A single
        // video answering it makes the question factual with extra words.
        if (videos.size < 2) err(q.slug, `comparative needs spans from 2+ videos, has ${videos.size}`);
        break;

      case "contradiction": {
        const sides = new Set(q.gold.map((g) => g.side));
        if (q.gold.some((g) => !g.side)) err(q.slug, "every contradiction span needs side: pro or contra");
        if (!(sides.has("pro") && sides.has("contra"))) {
          err(q.slug, "contradiction needs both a pro and a contra span — that is what coverage measures");
        }
        // Not an error: an author who contradicts themselves inside one video
        // is a real case, and one of the more interesting ones. The rule
        // "one side, one video" was an assumption, not a requirement — but
        // both sides landing in a single chunk would make coverage trivially
        // satisfied, so it is worth a second look.
        if (videos.size < 2) {
          warn(q.slug, "both sides come from one video — check they are far enough apart to land in different chunks");
        }
        break;
      }

      case "negative":
        // The correct result is empty. A gold span here means the answer IS in
        // the corpus, and the false-positive rate would be measured against a
        // question that has a right answer.
        if (q.gold.length > 0) err(q.slug, `negative must have no gold spans, has ${q.gold.length}`);
        break;
    }

    if (q.kind !== "contradiction" && q.gold.some((g) => g.side)) {
      warn(q.slug, "side is only meaningful on contradiction questions");
    }
  }
  return issues;
}

/**
 * The design the power calculation assumes.
 *
 * 76 of these carry `recall@k` — factual, comparative and contradiction. That
 * is the number the 10.5 pp detectable difference comes from, and negatives do
 * not change it however many there are.
 *
 * Negatives are 30 rather than the 14 a proportional split would give. The
 * proportions were scaled from an earlier 40-question design, which is
 * inherited arithmetic rather than a decision: the four kinds differ in what
 * they cost and in what they buy. Negatives are the cheapest question in the
 * set — no transcript to read, no span to pin down, `gold: []` — and they are
 * the sole input to false-positive rate, one of the two metrics this project
 * is built around.
 *
 * At 14, a perfect result reads as "somewhere between 0% and 23%". At 30 it
 * reads as "0% to 12%". The extra hour of writing halves the interval on a
 * headline number.
 */
export const TARGET: Record<Kind, number> = {
  factual: 36,
  comparative: 22,
  contradiction: 18,
  negative: 30,
};

/**
 * The kinds that carry `recall@k`, and so the denominator the power
 * calculation is about. Derived from TARGET rather than written out, because
 * the two numbers have to move together: a report that prints a live numerator
 * against a stale 76 is wrong in the direction that looks fine.
 */
export const RECALL_POOL_TARGET = TARGET.factual + TARGET.comparative + TARGET.contradiction;

export function countByKind(set: GoldenSet): Record<Kind, number> {
  const out: Record<Kind, number> = { factual: 0, comparative: 0, contradiction: 0, negative: 0 };
  for (const q of set.questions) out[q.kind]++;
  return out;
}

// ─── lexical overlap ─────────────────────────────────────────────────────────

/**
 * How much of a question is lifted verbatim from the passage it is annotated
 * against.
 *
 * This is the quiet way a golden set stops measuring anything. A question
 * phrased in the transcript's own words is found by exact term matching, so
 * every retriever scores well on it and the ablation compares configurations on
 * questions none of them can get wrong. It pushes the baseline into the ceiling
 * by construction, and no amount of extra questions fixes it.
 *
 * The number is not a verdict — a question about "the context window" has to
 * say "context window". It flags questions worth rereading.
 */
export function lexicalOverlap(question: string, spanText: string): number {
  const q = contentWords(question);
  if (q.size === 0) return 0;
  const s = contentWords(spanText);
  let shared = 0;
  for (const w of q) if (s.has(w)) shared++;
  return shared / q.size;
}

const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
  "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
  "do", "does", "did", "have", "has", "had", "it", "its", "this", "that",
  "these", "those", "what", "which", "who", "whom", "when", "where", "why",
  "how", "you", "your", "they", "their", "he", "she", "we", "i", "not", "no",
  "can", "could", "should", "would", "will", "about", "says", "say", "said",
]);

function contentWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .split(" ")
      .filter((w) => w.length > 2 && !STOPWORDS.has(w)),
  );
}

/** Above this, the question is mostly the passage restated. */
export const OVERLAP_WARN = 0.75;

/**
 * Below this, the question and its own answer share almost no content words.
 *
 * The number is deliberately low, and deliberately not the one the data
 * suggests. A pilot run put mean recall@5 at 0.28 for questions under 30%
 * overlap against 0.79 above it — but that was measured with BM25, which *is*
 * term matching, so the relationship is partly tautological, and choosing a
 * threshold from a retriever's output tunes the question set to the retriever.
 *
 * What the number cannot tell you is which of two very different things it has
 * found, and this matters more than the threshold:
 *
 *   1. A question the corpus does not answer. The annotation is wrong.
 *   2. A perfectly good question phrased in entirely different words from the
 *      passage that answers it.
 *
 * The second is not a defect. It is the most valuable question type in the set,
 * because it is the only kind that separates lexical retrieval from semantic
 * retrieval — and separating those is the point of the ablation. One question
 * here sits at 10% overlap, is never found by BM25 at any chunk size, and is
 * found by local embeddings at 128 tokens. Raising its overlap would delete
 * exactly the signal it carries, and it was briefly raised by mistake.
 *
 * So this is a prompt to check that the answer really is in the span, not a
 * prompt to rewrite. Only case 1 should ever be edited.
 */
export const OVERLAP_FLOOR = 0.15;
