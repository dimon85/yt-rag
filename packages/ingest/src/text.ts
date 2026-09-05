// Pure text functions used after transcripts land on disk. No network, no
// filesystem — the same reason packages/chunking is pure: this is where silent
// corpus damage would come from, so it stays trivially testable.
import { createHash } from "node:crypto";

export type Segment = { text: string; start_s: number; end_s: number };

/**
 * Normalized transcript text, used as the identity of a video.
 *
 * Deduplication cannot be by `youtube_id`: the same talk gets reuploaded on
 * several channels, often under a different title, and each upload has its own
 * id. Titles are the cheap first pass in select.ts; this is the authoritative
 * one, and it can only run once the transcript exists.
 */
export function normalizeTranscript(segments: Segment[]): string {
  return segments
    .map((s) => s.text)
    .join(" ")
    .toLowerCase()
    .replace(/\[[^\]]*\]/g, " ")        // [Music], [Applause] and friends
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** Stable short identity for a normalized transcript. */
export function fingerprint(normalized: string): string {
  return createHash("sha256").update(normalized).digest("hex").slice(0, 16);
}

/**
 * How many times each tool is mentioned, matched on the aliases in corpus.yaml.
 *
 * Counts rather than a boolean on purpose. A video that says "Cursor" once in
 * passing is not a video about Cursor, and tagging it as one would corrupt the
 * "at least 3 channels per tool" check with videos that carry no claims about
 * the tool at all.
 */
export function toolMentions(
  normalized: string,
  tools: { id: string; aliases: string[] }[],
): Map<string, number> {
  const hay = ` ${normalized} `;
  const out = new Map<string, number>();

  for (const t of tools) {
    // Spans, not a running total. Aliases of one tool overlap by design —
    // ["cursor", "cursor composer", "composer"] all fire on the phrase "cursor
    // composer" — so counting each alias separately turns one utterance into
    // three mentions, and a single passing phrase clears the tagging threshold
    // on its own.
    const spans: [number, number][] = [];
    for (const alias of t.aliases) {
      const needle = ` ${normalizeAlias(alias)} `;
      if (needle.trim() === "") continue;
      let from = 0;
      for (;;) {
        const at = hay.indexOf(needle, from);
        if (at === -1) break;
        spans.push([at, at + needle.length]);
        from = at + needle.length - 1;   // needles share their padding space
      }
    }
    const n = countMerged(spans);
    if (n > 0) out.set(t.id, n);
  }
  return out;
}

const normalizeAlias = (alias: string) =>
  alias.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

/** How many distinct stretches of text the spans cover once overlaps merge. */
function countMerged(spans: [number, number][]): number {
  if (spans.length === 0) return 0;
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let merged = 1;
  let end = spans[0]![1];
  for (const [s, e] of spans.slice(1)) {
    // Touching at the shared padding space is not an overlap: " cursor " and
    // " cursor " in "cursor cursor" abut at one character and are two mentions.
    if (s < end - 1) end = Math.max(end, e);
    else {
      merged++;
      end = e;
    }
  }
  return merged;
}

/** Tool ids mentioned at least `min` times, in corpus order. */
export function toolsInText(
  normalized: string,
  tools: { id: string; aliases: string[] }[],
  min = 3,
): string[] {
  const counts = toolMentions(normalized, tools);
  return tools.filter((t) => (counts.get(t.id) ?? 0) >= min).map((t) => t.id);
}

/** Total spoken seconds, for sanity-checking a transcript against its video. */
export function coverageSeconds(segments: Segment[]): number {
  if (segments.length === 0) return 0;
  return Math.max(...segments.map((s) => s.end_s)) - Math.min(...segments.map((s) => s.start_s));
}

/**
 * Decodes the HTML entities YouTube leaves in caption text.
 *
 * The transcript XML arrives with apostrophes as `&#39;`, quotes as `&quot;`
 * and so on: 13,182 occurrences across all 78 videos of this corpus, every one
 * of them affected. Left alone this is not cosmetic — `&#39;` costs more
 * tokens than `'`, and "512 tokens" has to mean the same thing in every
 * configuration for the ablation to compare anything.
 *
 * One pass, deliberately. Decoding repeatedly would turn a literal `&amp;#39;`
 * — text that genuinely contains an ampersand — into an apostrophe.
 */
export function decodeEntities(text: string): string {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (whole, body: string) => {
    if (body.startsWith("#")) {
      const code = body[1] === "x" || body[1] === "X"
        ? Number.parseInt(body.slice(2), 16)
        : Number.parseInt(body.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
    }
    return NAMED[body.toLowerCase()] ?? whole;
  });
}

const NAMED: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
};
