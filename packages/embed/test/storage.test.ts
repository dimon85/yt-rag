import { describe, expect, test } from "vitest";
import {
  cosine, fromPgVector, STORAGE_DIM, toPgVector, toStorage,
} from "../src/storage.ts";

const rand = (n: number) => Array.from({ length: n }, () => Math.random() - 0.5);

describe("toStorage", () => {
  test("pads up to the storage width", () => {
    expect(toStorage(rand(384), 384)).toHaveLength(STORAGE_DIM);
  });

  test("a vector already at the width is unchanged in length", () => {
    expect(toStorage(rand(STORAGE_DIM), STORAGE_DIM)).toHaveLength(STORAGE_DIM);
  });

  test("padding is zeros, not noise", () => {
    expect(toStorage([1, 2, 3], 3).slice(3).every((x) => x === 0)).toBe(true);
  });

  test("a model returning the wrong width fails immediately", () => {
    // If a provider quietly changes its default, ingest dies here rather than
    // producing a corrupted metrics table discovered a week later.
    expect(() => toStorage(rand(512), 384)).toThrow(/model returned 512/);
  });

  test("a model wider than storage is refused", () => {
    expect(() => toStorage(rand(3072), 3072)).toThrow(/exceeds storage width/);
  });
});

describe("padding preserves cosine", () => {
  test("identical before and after, to 12 decimal places", () => {
    // The invariant the whole storage-width decision rests on.
    const a = rand(1024);
    const b = rand(1024);
    expect(cosine(toStorage(a, 1024), toStorage(b, 1024))).toBeCloseTo(cosine(a, b), 12);
  });

  test("holds for vectors of different real widths padded to the same width", () => {
    const a = rand(384);
    const b = rand(384);
    expect(cosine(toStorage(a, 384), toStorage(b, 384))).toBeCloseTo(cosine(a, b), 12);
  });
});

describe("cosine", () => {
  test("a vector with itself is 1", () => {
    const v = rand(64);
    expect(cosine(v, v)).toBeCloseTo(1, 12);
  });

  test("opposite vectors are -1", () => {
    const v = rand(64);
    expect(cosine(v, v.map((x) => -x))).toBeCloseTo(-1, 12);
  });

  test("a zero vector is 0, not NaN", () => {
    expect(cosine(new Array(64).fill(0), rand(64))).toBe(0);
  });

  test("mismatched lengths are an error, not a silent truncation", () => {
    expect(() => cosine(rand(10), rand(20))).toThrow(/length mismatch/);
  });
});

describe("pg round-trip", () => {
  test("a padded vector survives the string codec", () => {
    // pg does not know the vector type and hands it over as a string. Passing
    // a JS array straight in serializes as {0.1,0.2} — Postgres array syntax —
    // which casts without complaint and stores garbage.
    const v = toStorage(rand(384), 384);
    expect(fromPgVector(toPgVector(v))).toEqual(v);
  });

  test("the encoded form is bracket syntax, not brace", () => {
    expect(toPgVector([0.1, 0.2])).toBe("[0.1,0.2]");
  });
});
