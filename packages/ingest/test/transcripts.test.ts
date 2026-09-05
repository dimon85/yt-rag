import { expect, test } from "vitest";
import { ERROR_NAMES, missingErrorClasses } from "../src/transcripts.ts";

test("the error classes the classifier depends on still exist", () => {
  // If youtube-transcript-plus renames or drops one of these, every error of
  // that kind falls through to the transient branch. A 'gone' video would then
  // be retried five times with backoff on every run, forever. Better to fail
  // here than to discover it as a mysteriously slow ingest.
  expect(missingErrorClasses()).toEqual([]);
});

test("no error name is classified as both transient and settled", () => {
  const both = ERROR_NAMES.throttled
    .filter((n) => (ERROR_NAMES.gone as readonly string[]).includes(n));
  expect(both).toEqual([]);
});
