import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { describe, expect, test } from "vitest";

// packages/report aggregates; it never re-runs anything. That is a convention,
// and it is also what makes the numbers auditable — there is no path by which a
// figure in the report could have been produced by retrieving again instead of
// by reading what a run recorded.
//
// Checked mechanically rather than by review, because the failure is a single
// convenient import away and would not look like a mistake: pulling in
// `search` to "just recompute the trivial labels" would work, produce
// plausible numbers, and quietly make the report a second retriever.
const FORBIDDEN = ["packages/retrieve", "packages/embed", "packages/chunking"];

const SRC = resolve(import.meta.dirname, "..", "src");

/** Every module reachable from the report's entry points, followed by hand. */
function reachable(entry: string): string[] {
  const seen = new Set<string>();
  const queue = [entry];
  while (queue.length > 0) {
    const file = queue.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    const source = readFileSync(file, "utf8");
    for (const m of source.matchAll(/from\s+"([^"]+)"/g)) {
      const spec = m[1]!;
      if (!spec.startsWith(".")) continue;
      queue.push(resolve(dirname(file), spec));
    }
  }
  return [...seen];
}

describe("packages/report cannot retrieve anything", () => {
  const entries = readdirSync(SRC).filter((f) => f.endsWith(".ts")).map((f) => join(SRC, f));

  test("there are entry points to check", () => {
    expect(entries.length).toBeGreaterThan(0);
  });

  for (const entry of entries) {
    test(`${entry.split("/").pop()} reaches no retriever, embedder or chunker`, () => {
      const offenders = reachable(entry)
        .map((f) => f.replaceAll("\\", "/"))
        .filter((f) => FORBIDDEN.some((bad) => f.includes(`/${bad}/`)));
      expect(offenders).toEqual([]);
    });
  }
});
