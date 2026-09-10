// Time-window chunking. Pure, like fixed.ts: segments in, chunks out, no I/O.
//
// The strategy configs.yaml calls `time_window`. It exists to test whether a
// transcript's own timeline is a better boundary than a token count — a speaker
// finishing a thought is a time event, not a length event.
//
// Three things about this corpus decide the implementation, and all three are
// visible in the cached captions rather than assumed.
//
// Captions overlap in time. Auto-generated captions here arrive as a rolling
// window: the second caption starts roughly at the midpoint of the first
// (0.16-4.24, then 2.32-6.00, then 4.24-7.76). 13,232 segment pairs overlap.
// So a window boundary is never a clean cut — at any instant two captions are
// live — and "which window does this segment belong to" has to be answered
// explicitly. The answer here is: the window its START falls in.
//
//   Why the start and not the midpoint, or the larger share of its duration.
//   Start-based assignment is the only one of the three that keeps segments in
//   transcript order. Under a majority-overlap rule two consecutive captions
//   can be assigned to windows in the opposite order to the order they were
//   spoken, because the second may lie more inside the earlier window than the
//   first does; the chunk text then reads backwards. It is also the only rule
//   that is a partition without arithmetic: with no overlap every segment
//   lands in exactly one window, so nothing is duplicated and nothing is lost.
//   The cost is that a chunk's text runs slightly past its nominal window,
//   which the reported span records honestly rather than hides.
//
// A chunk's span comes from the segments it holds, not from the window. The
// nominal window is [t, t+seconds); the chunk covers min(start_s) to
// max(end_s) of what actually landed in it, which can start late (silence at
// the front) and end past t+seconds (a caption straddling the far boundary).
// Reporting the nominal window instead would score the hit rule in hit.ts
// against seconds the chunk never contained.
//
// Empty windows are dropped. Captions have gaps — sponsor reads, music, a
// stretch with no speech — and one video in this corpus has two segments an
// hour apart. An empty window emitted as a chunk would have start_s == end_s,
// coverage() returns 0 for a zero-length interval, and the run would look like
// a retrieval failure at every rank rather than like a hole in the captions.
import { encode } from "gpt-tokenizer";
import type { Chunk, Segment } from "./fixed.ts";

/**
 * Splits a transcript into windows of `windowSeconds`, stepping forward by
 * `windowSeconds - overlapSeconds`.
 *
 * The grid is anchored at zero rather than at the first segment, so the same
 * timestamp falls in the same window in every video and in every run. Windows
 * before the first caption and after the last are never visited.
 */
export function windowChunks(
  segments: Segment[],
  windowSeconds = 60,
  overlapSeconds = 0,
): Chunk[] {
  if (!(windowSeconds > 0)) throw new Error(`windowSeconds must be positive, got ${windowSeconds}`);
  if (overlapSeconds < 0) throw new Error(`overlapSeconds cannot be negative, got ${overlapSeconds}`);
  if (overlapSeconds >= windowSeconds) {
    // Step would be zero or negative and the walk would never advance.
    throw new Error(`overlapSeconds (${overlapSeconds}) must be below windowSeconds (${windowSeconds})`);
  }
  if (segments.length === 0) return [];

  const step = windowSeconds - overlapSeconds;
  // Transcript order, by start. Cached captions already arrive sorted, but the
  // chunk text is only readable if that holds, so it is not left to trust.
  const sorted = [...segments].sort((a, b) => a.start_s - b.start_s);
  const firstStart = sorted[0]!.start_s;
  const lastStart = sorted[sorted.length - 1]!.start_s;

  // The earliest window on the zero-anchored grid that can still reach the
  // first segment. With overlap, several windows contain it; this is the first.
  const firstIndex = Math.max(0, Math.ceil((firstStart - windowSeconds + 1e-9) / step));

  const chunks: Chunk[] = [];
  let previous: { from: number; to: number } | null = null;

  for (let i = firstIndex; ; i++) {
    const from = i * step;
    if (from > lastStart) break;
    const to = from + windowSeconds;

    // Half-open [from, to): a segment starting exactly on a boundary belongs to
    // the window that opens there, so no segment can fall in both of two
    // adjacent non-overlapping windows and none can fall in neither. Sorted by
    // start, so the segments held are a contiguous index range.
    let lo = 0;
    while (lo < sorted.length && sorted[lo]!.start_s < from) lo++;
    let hi = lo;
    while (hi < sorted.length && sorted[hi]!.start_s < to) hi++;
    if (hi === lo) continue;

    // Sparse captions can put the same segments in two consecutive overlapping
    // windows. Emitting both would place identical text at two ranks of one
    // top-k, which moves recall@k for a reason that has nothing to do with the
    // configuration being measured.
    if (previous && previous.from === lo && previous.to === hi) continue;
    previous = { from: lo, to: hi };

    const held = sorted.slice(lo, hi);

    const text = held.map((s) => s.text).join(" ");
    chunks.push({
      text,
      start_s: Math.min(...held.map((s) => s.start_s)),
      // Not the last segment's end: overlapping captions mean the highest end
      // is not necessarily the last one's, and it may exceed `to`.
      end_s: Math.max(...held.map((s) => s.end_s)),
      // Counted with gpt-tokenizer over the joined text — invariant 2, one
      // tokenizer for every configuration. A time window does not bound this,
      // which is the whole difference between this strategy and fixed.
      token_count: encode(text).length,
    });
  }

  return chunks;
}
