// What difference this question set is capable of seeing.
//
// The table in docs/spec.md ("Why 90 and not 40") was computed once and
// written down, so nothing in the repo could check it or recompute it when the
// set changed. In a project whose whole subject is measurement that is the
// wrong way round, and it mattered as soon as the corpus turned out to contain
// four distinct contradictions rather than eighteen: the question "so what
// does that cost" had no runnable answer.
//
// Ported from a previous project's implementation (ai-roadmap/lib/power.ts and
// stats.ts), with its reasoning kept — the comments below about NaN at
// pb + pc = 1 and about the normal approximation record failures that were
// paid for there, not here.
//
// Two designs, and the difference between them is large:
//
//   unpaired  configuration A on some questions, B on others. Question-level
//             noise goes into the difference whole.
//   paired    both configurations on the SAME questions. Questions both get
//             right, or both get wrong, drop out of the difference entirely:
//             only DISCORDANT pairs carry information.
//
// This project is paired by construction — every configuration runs the same
// frozen set — and the consequence is the non-obvious part: power is governed
// not by how many questions there are but by how many MOVE. A set of "31
// always found + 13 never found + 5 that depend on the configuration" has 49
// questions and a moving mass of 5, and it is the 5 that decide what it can
// see.
//
// Everything here is exact. No normal approximation: at n=5 it produces
// nonsense, and n=5 is precisely the case this file exists to answer.

// Log-factorial, memoized as it grows. Computing the binomial coefficient in a
// loop per call made powerPaired O(n^3), and the search for a required n into
// the hundreds did not terminate.
const LOG_FACTORIAL: number[] = [0, 0];
const lg = (x: number) => {
  while (LOG_FACTORIAL.length <= x) {
    LOG_FACTORIAL.push(LOG_FACTORIAL[LOG_FACTORIAL.length - 1]! + Math.log(LOG_FACTORIAL.length));
  }
  return LOG_FACTORIAL[x]!;
};

/** Binomial density, O(1) through the log-factorial table. */
export function binomPmf(n: number, k: number, p: number): number {
  if (k < 0 || k > n) return 0;
  if (p === 0) return k === 0 ? 1 : 0;
  if (p === 1) return k === n ? 1 : 0;
  return Math.exp(lg(n) - lg(k) - lg(n - k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
}

/**
 * Exact two-sided binomial p-value for k of n against p0.
 *
 * "Two-sided" here is the minimum-likelihood method: every outcome no more
 * likely than the observed one is summed. This is the same test McNemar runs
 * on the discordant pairs.
 */
export function binomTest(k: number, n: number, p0 = 0.5): number {
  if (n === 0) return 1;
  const observed = binomPmf(n, k, p0);
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const d = binomPmf(n, i, p0);
    if (d <= observed * (1 + 1e-9)) sum += d;
  }
  return Math.min(1, sum);
}

// Rejection table for McNemar: is H0 rejected at b successes out of d discordant.
const rejectCache = new Map<string, boolean[]>();
function rejectTable(d: number, alpha: number): boolean[] {
  const key = `${d}|${alpha}`;
  const hit = rejectCache.get(key);
  if (hit) return hit;
  const table = Array.from({ length: d + 1 }, (_, b) => binomTest(b, d, 0.5) <= alpha);
  rejectCache.set(key, table);
  return table;
}

/**
 * Power of a PAIRED comparison (McNemar), exact and NOT conditioned on d.
 *
 * The model: a question is discordant either through noise or through the
 * effect.
 *   noise — the share of questions that flip on their own, with no change of
 *           configuration. Measured by repeating a run; it is symmetric, so it
 *           adds to both b and c. Retrieval here is deterministic, so it is 0
 *           — and the parameter exists so that "deterministic" is an argument
 *           rather than an assumption.
 *   delta — the true difference in accuracy, as a fraction. It adds
 *           discordance in ONE direction.
 * So p(b) = noise/2 + delta and p(c) = noise/2.
 *
 * What this model does NOT distinguish: a set where the effect moves the same
 * questions the noise moves from one where it moves different ones. In the
 * first case the real power is lower than computed here.
 */
export function powerPaired(n: number, delta: number, noise: number, alpha = 0.05): number {
  const pc = noise / 2;
  // pb + pc cannot exceed 1: these are three exclusive states of one question.
  // An earlier version clamped only pb, so at delta = 1.0 it got pb = 1 with
  // pc > 0, the conditional pc/(1 - pb) divided by zero, and power came back
  // NaN. Found by asking what happens at a perfect effect — which is exactly
  // where the answer is most wanted.
  const pb = Math.min(noise / 2 + delta, 1 - pc);
  let power = 0;
  for (let b = 0; b <= n; b++) {
    const fb = binomPmf(n, b, pb);
    if (fb < 1e-12) continue;
    for (let c = 0; c + b <= n; c++) {
      // Clamped into [0,1]: at pb + pc = 1 rounding gives 1.0000000000000021,
      // and log(1 - p) of a negative argument is a NaN that silently eats the
      // whole power figure.
      const conditional = Math.max(0, Math.min(1, pc / (1 - pb) || 0));
      const fc = binomPmf(n - b, c, conditional);
      if (fc < 1e-12) continue;
      const d = b + c;
      if (d === 0) continue;
      if (rejectTable(d, alpha)[b]) power += fb * fc;
    }
  }
  return power;
}

/**
 * The smallest effect the set can detect, as a fraction — not percentage points.
 *
 * Returns null when no effect in range reaches the target power. On small n
 * that is the normal outcome, and printing a large number instead would be a
 * lie in the direction of "we can measure something after all".
 */
export function mde(
  power: (delta: number) => number,
  target = 0.8,
  max = 1,
  step = 0.005,
): number | null {
  for (let d = step; d <= max + 1e-9; d += step) {
    if (power(d) >= target) return Math.round(d * 1000) / 1000;
  }
  return null;
}

export const mdePaired = (n: number, noise: number, alpha = 0.05, target = 0.8) =>
  mde((d) => powerPaired(n, d, noise, alpha), target, Math.max(0, 1 - noise));

// ─── interval ────────────────────────────────────────────────────────────────

function choose(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

function betaCdf(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  let sum = 0;
  const n = Math.round(a + b - 1);
  for (let k = Math.round(a); k <= n; k++) sum += choose(n, k) * x ** k * (1 - x) ** (n - k);
  return sum;
}

function betaInv(p: number, a: number, b: number): number {
  let lo = 0, hi = 1;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    betaCdf(mid, a, b) < p ? (lo = mid) : (hi = mid);
  }
  return (lo + hi) / 2;
}

/**
 * Exact 95% Clopper-Pearson interval for k successes of n.
 *
 * Exact rather than normal because of the case this project reports most
 * often: on 0 of 23 the normal approximation gives [0, 0] — "cannot happen" —
 * which the data does not say. Contradiction coverage and false-positive rate
 * are both reported this way, as a proportion with an interval rather than as
 * a comparison between configurations.
 */
export function clopperPearson(k: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const lo = k === 0 ? 0 : betaInv(0.025, k, n - k + 1);
  const hi = k === n ? 1 : betaInv(0.975, k + 1, n - k);
  return [lo, hi];
}
