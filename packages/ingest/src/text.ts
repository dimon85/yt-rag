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
    let n = 0;
    for (const alias of t.aliases) {
      const needle = ` ${alias.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim()} `;
      if (needle.trim() === "") continue;
      let from = 0;
      for (;;) {
        // Overlapping matches are impossible here because every needle is
        // padded with spaces, so indexOf from the space before is correct.
        const at = hay.indexOf(needle, from);
        if (at === -1) break;
        n++;
        from = at + needle.length - 1;
      }
    }
    if (n > 0) out.set(t.id, n);
  }
  return out;
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
