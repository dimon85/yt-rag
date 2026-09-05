import type { Video } from "./corpus.ts";

/**
 * Folds a discovery run's results back into the existing corpus.
 *
 * Two things this prevents, both of which a plain `writeVideos(kept)` did:
 *
 *   - `pnpm discover --channel @t3dotgg` rebuilt the whole `videos:` list from
 *     one channel's results, deleting the other eleven channels from the
 *     corpus. The command reads like it touches one channel; it wiped 70 of 78.
 *   - Re-selecting a video discarded the annotation attached to it. `topics` is
 *     filled in by hand during golden-set work, and hours of that would vanish
 *     on the next discovery run.
 *
 * Only the channels that were actually scanned are replaced. Everything else is
 * carried through untouched, and hand-added fields survive re-selection.
 */
export function mergeIntoCorpus(
  existing: Video[],
  discovered: Video[],
  scannedChannels: string[],
): Video[] {
  const scanned = new Set(scannedChannels);
  const prior = new Map(existing.map((v) => [v.youtube_id, v]));

  const carried = existing.filter((v) => !scanned.has(v.channel));
  const refreshed = discovered.map((v) => {
    const before = prior.get(v.youtube_id);
    if (!before) return v;
    return {
      ...v,
      // Hand-written annotation wins over anything discovery computes.
      topics: before.topics.length > 0 ? before.topics : v.topics,
      tools: before.tools.length > 0 ? before.tools : v.tools,
    };
  });

  return [...carried, ...refreshed];
}
