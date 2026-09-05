// Transcript fetching. Same shape as ytdlp.ts — ok / gone / throttled — because
// the lesson is the same one: a temporary refusal must never be cached as a
// verdict about the video.
//
// This library throws typed errors, so the classification is exact rather than
// a regex over stderr. One wrinkle: the error classes are exported at runtime
// but not re-exported from the package's index.d.ts, so `import { ... }` fails
// to typecheck even though the values are there. They are looked up on the
// module namespace instead, which keeps `instanceof` real — matching on
// `constructor.name` would break the day the package ships minified.
// transcripts.test.ts asserts the classes still exist, so an upgrade that
// renames them fails a test instead of silently reclassifying every error.
import * as ytp from "youtube-transcript-plus";
import type { Segment } from "./text.ts";

/** What the library hands back per caption line: seconds, plus a duration. */
type RawSegment = { text: string; offset: number; duration: number };

type ErrorClass = new (...args: never[]) => Error;
const errorClasses = ytp as unknown as Record<string, ErrorClass | undefined>;

export const ERROR_NAMES = {
  throttled: ["YoutubeTranscriptTooManyRequestError"],
  gone: [
    "YoutubeTranscriptVideoUnavailableError",
    "YoutubeTranscriptDisabledError",
    "YoutubeTranscriptNotAvailableError",
    "YoutubeTranscriptNotAvailableLanguageError",
  ],
} as const;

export const missingErrorClasses = (): string[] =>
  [...ERROR_NAMES.throttled, ...ERROR_NAMES.gone]
    .filter((n) => typeof errorClasses[n] !== "function");

const isOneOf = (e: unknown, names: readonly string[]): boolean =>
  names.some((n) => {
    const cls = errorClasses[n];
    return typeof cls === "function" && e instanceof cls;
  });

export type TranscriptResult =
  | { kind: "ok"; segments: Segment[] }
  | { kind: "gone"; reason: string }
  | { kind: "throttled"; reason: string };

export async function fetchTranscript(youtubeId: string): Promise<TranscriptResult> {
  try {
    // retries: 0 — backoff belongs to the caller, so every wait is logged in one
    // place and behaves the same way it does in discover.
    const segs: RawSegment[] = await ytp.fetchTranscript(youtubeId, { lang: "en", retries: 0 });
    return {
      kind: "ok",
      segments: segs.map((s): Segment => ({
        text: s.text,
        start_s: round2(s.offset),
        end_s: round2(s.offset + s.duration),
      })),
    };
  } catch (e) {
    if (isOneOf(e, ERROR_NAMES.throttled)) return { kind: "throttled", reason: "429 from YouTube" };
    // Each of these is a settled fact about the video: captions switched off,
    // none produced, no English track, or the video itself is gone.
    if (isOneOf(e, ERROR_NAMES.gone)) {
      return { kind: "gone", reason: (e as Error).constructor.name };
    }
    // Network blips, parser surprises, anything unrecognized: transient by
    // default. Being wrong that way costs a retry; the other way poisons cache.
    return { kind: "throttled", reason: String((e as Error).message ?? e).slice(0, 120) };
  }
}

const round2 = (n: number) => Math.round(n * 100) / 100;
