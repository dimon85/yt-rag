// corpus.yaml as a typed, self-validating object.
//
// Schema and balance check live in one Zod object on purpose: there is no way to
// load the config while skipping validation, so an unbalanced plan cannot reach
// ingest. See docs/spec.md, "Config validation".
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";
import { z } from "zod";

export const ROOT = join(import.meta.dirname, "..", "..", "..");
export const CORPUS_PATH = join(ROOT, "corpus.yaml");
export const CACHE_DIR = join(ROOT, "cache");

const Stance = z.enum(["hype", "practical", "skeptical"]);
export type Stance = z.infer<typeof Stance>;

const Channel = z.object({
  name: z.string(),
  handle: z.string().startsWith("@"),
  channel_id: z.string().regex(/^UC[A-Za-z0-9_-]{22}$/, "not a YouTube channel id"),
  stance: Stance,
  note: z.string().optional(),
  target_videos: z.number().int().positive(),
});
export type Channel = z.infer<typeof Channel>;

const Tool = z.object({
  id: z.string(),
  name: z.string(),
  aliases: z.array(z.string()).min(1),
  target_videos: z.number().int().positive(),
});
export type Tool = z.infer<typeof Tool>;

/** One selected video. Written by `discover`, annotated with topics by hand. */
export const Video = z.object({
  youtube_id: z.string().length(11),
  channel: z.string(),
  title: z.string(),
  published_at: z.string(),          // ISO date, YYYY-MM-DD
  duration_s: z.number().int().positive(),
  view_count: z.number().int().nonnegative().optional(),
  transcript_kind: z.enum(["manual", "generated"]),
  tools: z.array(z.string()),
  topics: z.array(z.string()).default([]),
});
export type Video = z.infer<typeof Video>;

export const CorpusConfig = z
  .object({
    version: z.number().int(),
    tools: z.array(Tool).min(1),
    tools_deliberately_absent: z.array(z.object({ id: z.string(), name: z.string() })),
    channels: z.array(Channel).min(1),
    topics: z.array(
      z.object({
        id: z.string(),
        name: z.string(),
        /** How the topic is spoken about, not what it is called. */
        keywords: z.array(z.string()).default([]),
      }).passthrough(),
    ),
    selection_rules: z.array(z.string()),
    targets: z.object({
      total_videos: z.number().int().positive(),
      min_videos: z.number().int().positive(),
      stance_share: z.record(Stance, z.number()),
      tolerance: z.number().default(0.03),
      min_bucket_videos: z.number().int().default(15),
      duration_s: z.tuple([z.number().int(), z.number().int()]).default([480, 3600]),
      window_months: z.number().int().default(18),
    }).passthrough(),
    videos: z.array(Video).default([]),
  })
  .superRefine((cfg, ctx) => {
    const planned = new Map<Stance, number>();
    for (const ch of cfg.channels) {
      planned.set(ch.stance, (planned.get(ch.stance) ?? 0) + ch.target_videos);
    }
    const total = [...planned.values()].reduce((s, n) => s + n, 0);

    for (const [stance, want] of Object.entries(cfg.targets.stance_share)) {
      const count = planned.get(stance as Stance) ?? 0;
      const got = count / total;
      if (Math.abs(got - want) > cfg.targets.tolerance) {
        ctx.addIssue({
          code: "custom",
          path: ["targets", "stance_share", stance],
          message: `plan gives ${got.toFixed(3)}, target ${want.toFixed(3)}`,
        });
      }
      if (count < cfg.targets.min_bucket_videos) {
        ctx.addIssue({
          code: "custom",
          path: ["channels"],
          message: `stance ${stance}: only ${count} videos planned`,
        });
      }
    }

    const ids = cfg.channels.map((c) => c.channel_id);
    if (new Set(ids).size !== ids.length) {
      ctx.addIssue({ code: "custom", path: ["channels"], message: "duplicate channel_id" });
    }
  });

export type CorpusConfig = z.infer<typeof CorpusConfig>;

export function loadCorpus(path = CORPUS_PATH): CorpusConfig {
  return CorpusConfig.parse(YAML.parse(readFileSync(path, "utf8")));
}

/**
 * Writes the `videos:` list back, leaving every comment in the file intact.
 *
 * A plain YAML.stringify of the parsed object would drop every comment in
 * corpus.yaml — and the comments there carry the reasoning for the whole corpus
 * design. So the document is edited in place instead.
 */
export function writeVideos(videos: Video[], path = CORPUS_PATH): void {
  const doc = YAML.parseDocument(readFileSync(path, "utf8"));
  doc.set("videos", doc.createNode(videos));
  writeFileSync(path, String(doc));
}
