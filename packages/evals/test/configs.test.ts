import { describe, expect, test } from "vitest";
import YAML from "yaml";
import {
  AblationConfig, cellId, chunkerFor, expandMatrix, loadConfigs, needsEmbeddings,
} from "../src/configs.ts";

/** A minimal valid file, so each test can break exactly one thing. */
const base = () => ({
  version: 1,
  defaults: { tokenizer: "gpt-tokenizer", top_k: 10, repeats: 3 },
  chunking: [
    { id: "fixed-128", strategy: "fixed", params: { tokens: 128, overlap: 0 } },
    { id: "window-60", strategy: "time_window", params: { seconds: 60, overlap_s: 0 } },
  ],
  retrieval: [
    { id: "bm25", kind: "lexical" },
    { id: "vector", kind: "dense" },
  ],
  reranking: [{ id: "none" }, { id: "mmr-0.7", kind: "mmr", params: { lambda: 0.7 } }],
});

describe("the real configs.yaml", () => {
  const cfg = loadConfigs();

  test("parses", () => {
    expect(cfg.version).toBe(1);
    expect(cfg.defaults.repeats).toBe(3);
    expect(cfg.defaults.top_k).toBe(10);
  });

  test("declares both time_window configurations the strategy exists for", () => {
    const windows = cfg.chunking.filter((c) => c.strategy === "time_window");
    expect(windows.map((c) => c.id)).toEqual(["window-60", "window-90-ov30"]);
  });

  test("every chunking id maps to a chunker", () => {
    const segments = Array.from({ length: 200 }, (_, i) => ({
      text: `word${i} some filler text`,
      start_s: i * 5,
      end_s: i * 5 + 5,
    }));
    for (const c of cfg.chunking) {
      expect(chunkerFor(c)(segments).length).toBeGreaterThan(0);
    }
  });

  test("only bm25 avoids embeddings", () => {
    expect(cfg.retrieval.filter((r) => !needsEmbeddings(r)).map((r) => r.id)).toEqual(["bm25"]);
  });
});

describe("the discriminated union", () => {
  test("an unknown strategy is rejected at load, not mid-run", () => {
    const bad = base();
    bad.chunking.push({ id: "sent", strategy: "sentence", params: { tokens: 512 } } as never);
    expect(() => AblationConfig.parse(bad)).toThrow();
  });

  test("a fixed row carrying time_window params is rejected", () => {
    const bad = base();
    bad.chunking[0] = { id: "fixed-128", strategy: "fixed", params: { seconds: 60 } } as never;
    expect(() => AblationConfig.parse(bad)).toThrow();
  });

  test("overlap at or above the chunk size is rejected", () => {
    const bad = base();
    bad.chunking[0]!.params = { tokens: 128, overlap: 128 };
    expect(() => AblationConfig.parse(bad)).toThrow(/overlap must be below tokens/);

    const worse = base();
    worse.chunking[1]!.params = { seconds: 60, overlap_s: 60 } as never;
    expect(() => AblationConfig.parse(worse)).toThrow(/overlap_s must be below seconds/);
  });

  test("a bare `- id: none` reranking row parses as kind none", () => {
    const cfg = AblationConfig.parse(base());
    expect(cfg.reranking[0]).toEqual({ id: "none", kind: "none" });
  });

  test("an mmr row without a lambda is rejected", () => {
    const bad = base();
    bad.reranking[1] = { id: "mmr", kind: "mmr" } as never;
    expect(() => AblationConfig.parse(bad)).toThrow();
  });

  test("a lambda outside [0,1] is rejected", () => {
    const bad = base();
    bad.reranking[1] = { id: "mmr", kind: "mmr", params: { lambda: 1.5 } };
    expect(() => AblationConfig.parse(bad)).toThrow();
  });
});

describe("invariant 2: one tokenizer", () => {
  test("a second tokenizer is rejected rather than silently invalidating the table", () => {
    const bad = base();
    bad.defaults.tokenizer = "js-tiktoken";
    expect(() => AblationConfig.parse(bad)).toThrow();
  });
});

describe("unique ids", () => {
  test("a duplicate chunking id is rejected — run output is keyed by it", () => {
    const bad = base();
    bad.chunking.push({ ...bad.chunking[0]! });
    expect(() => AblationConfig.parse(bad)).toThrow(/duplicate id fixed-128/);
  });

  test("a duplicate retrieval id is rejected", () => {
    const bad = base();
    bad.retrieval.push({ id: "bm25", kind: "dense" });
    expect(() => AblationConfig.parse(bad)).toThrow(/duplicate id bm25/);
  });

  test("a duplicate reranking id is rejected", () => {
    const bad = base();
    bad.reranking.push({ id: "none" });
    expect(() => AblationConfig.parse(bad)).toThrow(/duplicate id none/);
  });
});

describe("expandMatrix", () => {
  const cfg = AblationConfig.parse(base());

  test("2 x 2 x 2 is 8 cells", () => {
    expect(expandMatrix(cfg)).toHaveLength(8);
  });

  test("cells come out in file order", () => {
    expect(expandMatrix(cfg).map(cellId)).toEqual([
      "fixed-128__bm25__none",
      "fixed-128__bm25__mmr-0.7",
      "fixed-128__vector__none",
      "fixed-128__vector__mmr-0.7",
      "window-60__bm25__none",
      "window-60__bm25__mmr-0.7",
      "window-60__vector__none",
      "window-60__vector__mmr-0.7",
    ]);
  });

  test("a filter narrows one axis and leaves the others whole", () => {
    const cells = expandMatrix(cfg, { retrieval: ["bm25"] });
    expect(cells).toHaveLength(4);
    expect(new Set(cells.map((c) => c.retrieval.id))).toEqual(new Set(["bm25"]));
  });

  test("an unknown id throws rather than running zero cells and exiting 0", () => {
    expect(() => expandMatrix(cfg, { chunking: ["fixed-129"] })).toThrow(/no chunking config/);
    expect(() => expandMatrix(cfg, { retrieval: ["bm52"] })).toThrow(/no retrieval config/);
  });

  test("an empty filter list means the whole axis", () => {
    expect(expandMatrix(cfg, { chunking: [] })).toHaveLength(8);
  });

  test("the real matrix is 7 x 3 x 3", () => {
    // Worth asserting rather than deriving: docs/spec.md still describes six
    // chunking rows and configs.yaml now has seven, and the number of cells is
    // what the dry-run's cost claim is built on.
    const real = loadConfigs();
    expect(real.chunking).toHaveLength(7);
    expect(expandMatrix(real)).toHaveLength(63);
  });
});

describe("YAML round trip", () => {
  test("parses from text the same way it parses from an object", () => {
    const text = YAML.stringify(base());
    expect(AblationConfig.parse(YAML.parse(text))).toEqual(AblationConfig.parse(base()));
  });
});
