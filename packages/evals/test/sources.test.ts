import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, test } from "vitest";

// Two hash keys in this repo delimit their parts with a NUL — chunkSha here
// and rerankKey in packages/retrieve — because a space occurs inside every
// chunk text and a NUL cannot. The byte is correct; writing it into the source
// as a literal is not, and that is what this checks.
//
// Checked mechanically because the failure is invisible in exactly the places
// that would catch it. A literal NUL early in a file makes git call the file
// binary, so the diff prints "Bin 11197 bytes" and the review never happens;
// grep and ripgrep skip such a file and report no matches rather than an
// error, so a search for the symbol comes back empty and looks like an answer.
// It is also fragile: an editor that strips the byte would silently change
// every cache key derived from it.
const SRC = resolve(import.meta.dirname, "..", "..");

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    if (entry === "node_modules") return [];
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return sources(path);
    return path.endsWith(".ts") ? [path] : [];
  });
}

describe("no source file carries a literal NUL byte", () => {
  const files = sources(SRC);

  test("there are sources to check", () => {
    expect(files.length).toBeGreaterThan(0);
  });

  test('a NUL delimiter is written as the escape "\\0"', () => {
    const offenders = files
      .filter((f) => readFileSync(f, "utf8").includes("\0"))
      .map((f) => f.slice(SRC.length + 1));
    expect(offenders).toEqual([]);
  });
});
