// What the question set can currently see, and what changing it would buy.
//
//   pnpm power
//   pnpm power --contradictions 5    # if the corpus only supports five
//
// Reads the set and computes; it runs no retrieval and needs no API key. The
// point of computing this before a run rather than after is that "every
// configuration looks the same" and "this set cannot tell them apart" are
// indistinguishable in the output and completely different as conclusions.
import { parseArgs } from "node:util";
import { loadGolden } from "./golden.ts";
import { clopperPearson, mdePaired, powerPaired } from "./power.ts";

const { values } = parseArgs({
  args: process.argv.slice(2)
    .filter((a, i, all) => !(a === "--" && all.slice(0, i).every((x) => x === "--"))),
  options: { contradictions: { type: "string" } },
  allowPositionals: false,
});

// The design in docs/spec.md, "Why 90 and not 40".
const DESIGN = { factual: 36, comparative: 22, contradiction: 18, negative: 30 };

const set = loadGolden();
const have = { factual: 0, comparative: 0, contradiction: 0, negative: 0 } as Record<string, number>;
for (const q of set.questions) have[q.kind] = (have[q.kind] ?? 0) + 1;

const pp = (x: number | null) =>
  x === null ? "nothing in range" : `${(x * 100).toFixed(1)} pp`;
const pool = (k: Record<string, number>) => k.factual! + k.comparative! + k.contradiction!;

console.log(`golden/questions.yaml — version ${set.version}, ${set.questions.length} questions\n`);
console.log("kind            have  design");
for (const kind of Object.keys(DESIGN) as (keyof typeof DESIGN)[]) {
  console.log(`${kind.padEnd(15)} ${String(have[kind]).padStart(4)}  ${String(DESIGN[kind]).padStart(6)}`);
}

// ─── the recall pool ─────────────────────────────────────────────────────────

// Negatives carry no gold spans, so they contribute nothing to recall@k. The
// pool that carries the configuration comparison is the other three kinds.
console.log("\n─── what a paired comparison can detect ───");
console.log("power 0.8, alpha 0.05, deterministic retrieval\n");
console.log("  recall pool                    MDE");
const scenarios: [string, number][] = [
  [`now (${have.factual}+${have.comparative}+${have.contradiction})`, pool(have)],
  [`design (36+22+18)`, pool(DESIGN)],
];
if (values.contradictions !== undefined) {
  const c = Number(values.contradictions);
  scenarios.push([`36+22+${c} contradictions`, 36 + 22 + c]);
  // Restoring the pool by writing comparative questions instead: they need
  // only two videos on one subject, which the corpus has in quantity, where a
  // flat disagreement between authors it evidently does not.
  const need = pool(DESIGN) - (36 + c);
  scenarios.push([`36+${need}+${c}, pool restored`, 36 + need + c]);
}
for (const [label, n] of scenarios) {
  console.log(`  ${label.padEnd(30)} ${String(n).padStart(3)} → ${pp(mdePaired(n, 0))}`);
}
console.log(
  "\n  Differences between chunking strategies realistically live in 5-15 pp.\n" +
  "  An MDE above that range means a null result says nothing about chunking.",
);

// ─── the per-kind metrics ────────────────────────────────────────────────────

// These two are what the project is for, and neither is a comparison. Said
// plainly because the temptation is to read a difference in them anyway.
console.log("\n─── contradiction coverage and false positives ───");
console.log("Reported as a proportion with an interval, never as a comparison.");
console.log("The numbers below are why:\n");

console.log("  contradiction coverage, if every configuration were compared");
for (const n of [have.contradiction!, DESIGN.contradiction]) {
  console.log(
    `    n=${String(n).padStart(2)}  MDE ${pp(mdePaired(n, 0))}` +
    `   power at a 20 pp difference: ${(powerPaired(n, 0.2, 0) * 100).toFixed(0)}%`,
  );
}
console.log("\n  and as a proportion, which is how it is actually read");
const cn = have.contradiction!;
for (const k of [0, Math.floor(cn / 2), cn]) {
  const [lo, hi] = clopperPearson(k, cn);
  console.log(`    ${k}/${cn} → ${(lo * 100).toFixed(0)}% to ${(hi * 100).toFixed(0)}%`);
}

console.log("\n  false positives on negatives, at a perfect result");
for (const n of [14, have.negative!, DESIGN.negative]) {
  const [, hi] = clopperPearson(0, n);
  console.log(`    0/${String(n).padStart(2)} → 0% to ${(hi * 100).toFixed(0)}%`);
}
