// Fixed-token chunking. Pure functions: segments in, chunks out. No database,
// no network — this is where bugs hide and where they are cheapest to catch.
//
// Two things the spec insists on and this file implements literally.
//
// Segments are merged before splitting. They arrive as 2-8 second captions, so
// splitting them directly would make a "chunk" half a sentence.
//
// One tokenizer for every configuration. "512 tokens" has to mean the same
// thing in every run or the comparison is invalid, so sizes are counted with
// gpt-tokenizer and never estimated from character counts.
import { encode } from "gpt-tokenizer";

export type Segment = { text: string; start_s: number; end_s: number };

export type Chunk = {
  text: string;
  start_s: number;
  end_s: number;
  token_count: number;
};

/**
 * Splits a transcript into chunks of at most `targetTokens`, with
 * `overlapTokens` carried from the end of each chunk into the next.
 *
 * Boundaries land between segments rather than inside them: a caption is the
 * smallest unit with a timestamp, and cutting one would leave a chunk whose
 * start_s and end_s no longer describe its text. The cost is that chunks come
 * out slightly under target rather than exactly on it.
 */
export function fixedChunks(
  segments: Segment[],
  targetTokens = 512,
  overlapTokens = 0,
): Chunk[] {
  if (targetTokens < 1) throw new Error(`targetTokens must be positive, got ${targetTokens}`);
  if (overlapTokens < 0) throw new Error(`overlapTokens cannot be negative, got ${overlapTokens}`);
  if (overlapTokens >= targetTokens) {
    // Otherwise each chunk would carry over at least as much as it holds and
    // the walk would never advance.
    throw new Error(`overlapTokens (${overlapTokens}) must be below targetTokens (${targetTokens})`);
  }
  if (segments.length === 0) return [];

  const counts = segments.map((s, i) => encode(`${i > 0 ? " " : ""}${s.text}`).length);

  const chunks: Chunk[] = [];
  let from = 0;

  while (from < segments.length) {
    let to = from;
    let tokens = 0;
    // Always take at least one segment: a single caption longer than the target
    // would otherwise produce an empty chunk and stall the loop.
    while (to < segments.length && (to === from || tokens + counts[to]! <= targetTokens)) {
      tokens += counts[to]!;
      to++;
    }

    chunks.push(build(segments.slice(from, to), tokens));

    if (to >= segments.length) break;

    if (overlapTokens === 0) {
      from = to;
      continue;
    }
    // Step back over enough segments to carry roughly overlapTokens forward,
    // then guarantee forward progress.
    let back = to;
    let carried = 0;
    while (back > from + 1 && carried + counts[back - 1]! <= overlapTokens) {
      back--;
      carried += counts[back]!;
    }
    from = Math.max(from + 1, back);
  }

  return chunks;
}

function build(segments: Segment[], token_count: number): Chunk {
  return {
    text: segments.map((s) => s.text).join(" "),
    start_s: segments[0]!.start_s,
    // Captions in this corpus overlap in time — 13,232 of the segment pairs do
    // — so the last segment's end_s is not necessarily the largest.
    end_s: Math.max(...segments.map((s) => s.end_s)),
    token_count,
  };
}
