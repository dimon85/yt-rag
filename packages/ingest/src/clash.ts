// Finding passages that disagree, after matching by vocabulary failed.
//
// The first attempt paired a skeptical passage with the hype passage most
// similar to it by shared words, on the assumption that two people arguing
// about the same thing use the same terms. Of 32 pairs produced that way, 0
// disagreed about anything. The same structural fact sank MMR as a way to
// raise contradiction coverage: disagreement is not lexical similarity. Two
// passages can share every noun and be about different questions, and two can
// contradict each other flatly while sharing almost no wording — "you'll hit
// the limit in an hour" against "I've never run out in a full day".
//
// So this asks a model to read one stance's passages against another's and
// name the proposition they differ on, which is a judgement about meaning
// rather than about surface form.
//
// Two rules constrain the output, and both exist because breaking them would
// quietly corrupt the measurement rather than fail loudly:
//
//  1. It returns the disputed claim as a neutral one-line proposition, never a
//     question. A question phrased from a passage is answered by term matching
//     alone, and the set already has 16 of 46 questions in that state. The
//     question gets written afterwards, from the proposition, without the
//     passage in view.
//
//  2. Every span it cites must be one that was sent to it. A model asked for
//     timestamps will invent plausible ones, and a gold span pointing at the
//     wrong place is worse than a missing question: it scores a correct
//     retrieval as a failure forever.
import { z } from "zod";

/** A passage offered to the model, with the id it must cite. */
export type Passage = {
  id: string;
  video: string;
  channel: string;
  /** ISO date of the video. Only the revision relation uses it, and it is the relation. */
  published_at: string;
  start_s: number;
  end_s: number;
  text: string;
};

export const ClashSchema = z.object({
  clashes: z.array(z.object({
    /**
     * The disputed claim, stated neutrally, taking neither side. This is the
     * seed a question is written from later — so it must not be phrased as a
     * question, and must not borrow the passages' wording.
     */
    claim: z.string().min(10),
    /** Id of the passage asserting the claim. */
    pro: z.string(),
    /** Id of the passage denying it. */
    contra: z.string(),
    /** Why these two are held to be in conflict, in one line. */
    why: z.string(),
    /**
     * How direct the conflict is. `flat` means the two statements cannot both
     * be true; `partial` means they conflict under some reading, usually
     * because one is about a case the other does not cover.
     */
    directness: z.enum(["flat", "partial"]),
  })),
});
export type Clash = z.infer<typeof ClashSchema>["clashes"][number];

const RULES = `
This is hard and most pairs do not qualify. Do NOT report:
- two passages that merely discuss the same topic
- differing enthusiasm, tone, or preference about the same facts
- one passage praising a feature the other does not mention
- claims about different tools, versions, or plans

For each real clash, state the disputed claim as a neutral one-line
proposition: a statement of fact that one side asserts and the other denies,
taking neither side and not reusing the passages' phrasing. Never phrase it as
a question. Put the id of the passage asserting it in "pro" and the id of the
passage denying it in "contra".

Cite passages ONLY by the ids given. Never invent an id or a timestamp.
Return an empty list if nothing genuinely conflicts.
`.trim();

export const PROMPT = `
You are given two sets of passages transcribed from YouTube tutorials about AI
coding tools. Passages in PRO come from channels that are enthusiastic about
these tools; passages in CONTRA come from channels that are skeptical or
strictly practical.

Find pairs where the two passages make claims that CANNOT BOTH BE TRUE about
the same specific thing — a limit, a price, a capability, a behaviour, a
recommendation.

Report EVERY pair that qualifies, not only the clearest one. Different pairs
may dispute different claims, and the same passage may appear in more than one
pair. Work through the passages systematically rather than stopping at the
first conflict you find.

${RULES}
`.trim();

/**
 * One pooled list, no stance labels.
 *
 * Splitting passages by channel stance assumes disagreement runs along the
 * hype/skeptic axis. Measured on this corpus, it does not always: hype
 * channels mention MCP twice across 26 videos while practical channels mention
 * it 64 times across 34, so every MCP cell has an empty pro side and the split
 * finds nothing by construction. Two practitioners contradicting each other
 * about a technical fact is still a contradiction, and it is the larger pool.
 */
export const POOLED_PROMPT = `
You are given passages transcribed from YouTube tutorials about AI coding
tools, by different authors.

Find pairs of passages that make claims which CANNOT BOTH BE TRUE about the
same specific thing — a limit, a price, a capability, a behaviour, a
recommendation. The two passages must be by different authors.

Report EVERY pair that qualifies, not only the clearest one. Different pairs
may dispute different claims, and the same passage may appear in more than one
pair. Work through the passages systematically rather than stopping at the
first conflict you find.

${RULES}
`.trim();

/**
 * The passages, formatted for the prompt.
 *
 * With one list the model looks for disagreement anywhere in it; with two it
 * is told which side is which. Which of those is right is a property of the
 * cell, not a preference — see POOLED_PROMPT.
 */
export function renderPassages(pro: Passage[], contra: Passage[] = []): string {
  const block = (label: string, ps: Passage[]) =>
    `${label}:\n${ps.map((p) => `[${p.id}] (${p.channel}) ${p.text}`).join("\n\n")}`;
  return contra.length === 0
    ? block("PASSAGES", pro)
    : `${block("PRO", pro)}\n\n${block("CONTRA", contra)}`;
}

/**
 * A clash whose citations resolved. `pro` and `contra` are the passages
 * themselves rather than the ids — the ids are the model's language, and
 * everything downstream wants the span.
 */
export type Verified = Omit<Clash, "pro" | "contra"> &
  { pro: Passage; contra: Passage; claimProblem: string | null };

/**
 * The checks both relations need, as one place rather than two.
 *
 * Returns the reason a pair cannot be used, or null when it can.
 */
/**
 * Ids as the model wrote them, which is not always as they were given.
 *
 * Passages are rendered as `[p12] (channel) text...`, and a model asked to cite
 * p12 will sometimes cite `[p12]` — the form it saw. One search returned its
 * only candidate that way and the whole thing was thrown out as an invented
 * id. Stripping the brackets it copied is not the same as accepting an id that
 * was never offered: an unknown id still fails, and the rejection still prints
 * what the model actually said.
 */
export const normalizeId = (id: string) => id.trim().replace(/^\[|\]$/g, "").trim();

function rejectionReason(
  claim: string,
  ids: [string, string],
  resolved: [Passage | undefined, Passage | undefined],
): string | null {
  const [a, b] = resolved;
  if (!a || !b) {
    // Truncated: a model that writes a paragraph into an id field turns this
    // line into an unreadable wall, and the first 40 characters are enough to
    // see what it did.
    const short = (id: string) => (id.length > 40 ? `${id.slice(0, 40)}...` : id);
    const missing = [!a && short(ids[0]), !b && short(ids[1])].filter(Boolean).join(", ");
    return `cites unknown passage ${missing}`;
  }
  if (a.video === b.video) return `both passages come from one video, ${a.video}`;
  return null;
}

/**
 * What is wrong with how the claim is worded, or null.
 *
 * Kept apart from `rejectionReason` after a candidate was thrown away for it.
 * A claim phrased badly is not a bad pair: the spans, the authors and the dates
 * can all be right while the sentence is a description of the evidence rather
 * than the disputed fact. One search returned exactly that — an author saying
 * a large context window removes the need for compression, against the same
 * author later saying it did not solve the problem — and the whole candidate
 * disappeared over the sentence, which is the one part a person rewrites
 * anyway before the question is written.
 *
 * So these are reported next to the candidate instead of deleting it.
 */
export function claimProblem(claim: string): string | null {
  if (/\?\s*$/.test(claim)) return "claim is phrased as a question";
  // "The earlier passage asserts X, while the later passage..." — a report
  // about the evidence. There is nothing to ask from it, only a note that two
  // people said things.
  if (/\b(?:the (?:earlier|later|first|second) passage|passage p?\d|the author (?:claims|asserts|states))\b/i.test(claim)) {
    return "claim describes the passages instead of stating the disputed fact";
  }
  return null;
}

/**
 * Keeps only clashes whose cited passages were actually offered, and whose two
 * sides come from different videos.
 *
 * The same-video check is not pedantry. A speaker who says "the limit is five
 * hours" and later "well, it depends" has not contradicted anyone — the
 * contradiction the corpus is supposed to contain is between authors, and
 * `contradiction coverage` measures whether retrieval returns both of them.
 * A pair inside one video would be satisfied by any retriever that returns two
 * adjacent chunks, which measures nothing.
 */
export function verify(
  clashes: Clash[],
  offered: Passage[],
): { kept: Verified[]; rejected: { clash: Clash; reason: string }[] } {
  const byId = new Map(offered.map((p) => [p.id, p]));
  const find = (id: string) => byId.get(normalizeId(id));
  const kept: Verified[] = [];
  const rejected: { clash: Clash; reason: string }[] = [];

  for (const c of clashes) {
    const pro = find(c.pro);
    const contra = find(c.contra);
    const reason = rejectionReason(c.claim, [c.pro, c.contra], [pro, contra]);
    if (reason) rejected.push({ clash: c, reason });
    else kept.push({ ...c, pro: pro!, contra: contra!, claimProblem: claimProblem(c.claim) });
  }
  return { kept, rejected };
}

// ─── the other relation ──────────────────────────────────────────────────────

/**
 * Two passages that each cover part of one subject, where neither covers it
 * alone. That is what a `comparative` question needs: `pnpm golden` requires
 * spans from two different videos, and the point of the kind is that returning
 * either one is not enough.
 *
 * It is the same machinery as the clash search and a different relation, which
 * is why it lives here. It is also the cheaper half of the shortfall: the set
 * is 14 comparative questions short and 13 contradictions short, and a
 * disagreement needs two authors to conflict while this needs only two authors
 * to cover different parts of one thing. The corpus has the second in
 * quantity and, measurably, not much of the first.
 */
export const ComplementSchema = z.object({
  pairs: z.array(z.object({
    /** The subject both passages speak to, stated neutrally and not as a question. */
    subject: z.string().min(10),
    a: z.string(),
    b: z.string(),
    /** What A covers that B does not. Empty means the pair is redundant. */
    a_only: z.string(),
    /** What B covers that A does not. */
    b_only: z.string(),
  })),
});
export type Complement = z.infer<typeof ComplementSchema>["pairs"][number];

export const COMPLEMENT_PROMPT = `
You are given passages transcribed from YouTube tutorials about AI coding
tools, by different authors.

Find pairs of passages where BOTH are needed to answer one question about a
specific subject — a mechanism, a trade-off, a workflow, a cost. The two
passages must be by different authors.

The requirement that makes this hard: NEITHER passage may answer the subject on
its own. Each must carry something the other does not. Do NOT report:
- two passages that say the same thing in different words
- a pair where one passage already covers everything the other does
- two passages about the same tool but different subjects
- a passage that merely mentions the subject in passing

For each pair, state the subject as a neutral one-line proposition — not a
question, and not reusing the passages' phrasing — and say concretely what each
passage covers that the other does not.

Cite passages ONLY by the ids given. Never invent an id or a timestamp.
Return an empty list if no pair genuinely needs both halves.
`.trim();

export type VerifiedComplement = Omit<Complement, "a" | "b"> &
  { a: Passage; b: Passage; claimProblem: string | null };

/**
 * As `verify`, plus the check that separates a comparative pair from two
 * redundant sources: both sides must contribute something.
 *
 * Without it the pair becomes a factual question scored against two spans, one
 * of which no retriever needs to find — which quietly lowers recall for every
 * configuration equally and measures nothing.
 */
export function verifyComplements(
  pairs: Complement[],
  offered: Passage[],
): { kept: VerifiedComplement[]; rejected: { pair: Complement; reason: string }[] } {
  const byId = new Map(offered.map((p) => [p.id, p]));
  const find = (id: string) => byId.get(normalizeId(id));
  const kept: VerifiedComplement[] = [];
  const rejected: { pair: Complement; reason: string }[] = [];

  for (const p of pairs) {
    const a = find(p.a);
    const b = find(p.b);
    const reason = rejectionReason(p.subject, [p.a, p.b], [a, b])
      ?? (p.a_only.trim() === "" || p.b_only.trim() === ""
        ? "one side contributes nothing the other does not"
        : null);
    if (reason) rejected.push({ pair: p, reason });
    else kept.push({ ...p, a: a!, b: b!, claimProblem: claimProblem(p.subject) });
  }
  return { kept, rejected };
}

// ─── what the set already covers ─────────────────────────────────────────────

/** The little of an existing question this needs to know. */
export type Annotated = {
  slug: string;
  gold: { video: string; start_s: number; end_s: number }[];
};

const overlaps = (
  a: { video: string; start_s: number; end_s: number },
  b: { video: string; start_s: number; end_s: number },
) => a.video === b.video && a.end_s > b.start_s && b.end_s > a.start_s;

/**
 * The slug of an existing question this pair would duplicate, or null.
 *
 * A pair counts as a duplicate only when BOTH of its spans overlap the gold
 * spans of one question. One overlapping span is not enough: a stretch of
 * video can legitimately answer more than one question, and it is the pair
 * that would be redundant rather than the passage.
 *
 * Added after the complement search returned a pair whose two spans were the
 * same two videos as `skills-versus-tool-discovery`, shifted by a few seconds.
 * That is the expensive kind of duplicate — it looks new until the spans are
 * read side by side, by which point an hour is gone.
 *
 * Intervals are half-open, the same convention as the hit rule: two spans that
 * merely touch at an endpoint cover different seconds.
 */
export function duplicateOf(
  a: Passage,
  b: Passage,
  questions: Annotated[],
): string | null {
  for (const q of questions) {
    const hitsA = q.gold.some((g) => overlaps(g, a));
    const hitsB = q.gold.some((g) => overlaps(g, b));
    if (hitsA && hitsB) return q.slug;
  }
  return null;
}


// ─── the third relation: one author, later ───────────────────────────────────

/**
 * A claim an author made, and the same author saying otherwise later.
 *
 * `selection_rules` in corpus.yaml names outdated claims as the second source
 * of contradictions, and the 12-18 month range the corpus was selected over
 * exists for it — but nothing was mining it. The clash search cannot: it
 * requires two different authors, because a speaker who qualifies himself
 * inside one video has not contradicted anyone.
 *
 * The constraint that keeps this honest is that the two spans must still be in
 * two different videos. Same author, different video, later date. A same-video
 * pair would be satisfied by any retriever returning two adjacent chunks,
 * which is why every relation here rejects one — and the reason applies
 * exactly as much when the author is arguing with his past self.
 */
export const RevisionSchema = z.object({
  revisions: z.array(z.object({
    /** The disputed claim, stated neutrally and not as a question. */
    claim: z.string().min(10),
    // Named with the _id suffix after a run put prose in both: "earlier" on
    // its own reads as "the earlier claim" as easily as "the earlier id", and
    // three candidates were lost to that reading.
    /** Id of the passage making the claim first. */
    earlier_id: z.string(),
    /** Id of the passage from the same author that later says otherwise. */
    later_id: z.string(),
    why: z.string(),
  })),
});
export type Revision = z.infer<typeof RevisionSchema>["revisions"][number];

export const REVISION_PROMPT = `
You are given passages from several videos by the SAME author, listed oldest
first with the publication date of each.

Find pairs where the author states something in an earlier video and states
something incompatible with it in a later one — a limit that changed, a
recommendation reversed, a capability that arrived or was withdrawn, a claim
retracted.

This is hard and most pairs do not qualify. Do NOT report:
- the same claim restated in different words
- a later video simply covering more ground than an earlier one
- a topic mentioned twice without an incompatible assertion
- claims about different tools, versions, or plans

For each real revision, state the disputed claim as a neutral one-line
proposition: what the earlier passage asserts and the later one denies, taking
neither side and not reusing the passages' phrasing. Never phrase it as a
question.

"earlier_id" and "later_id" take an id and nothing else — "p12", not a
description of the passage.

Cite passages ONLY by the ids given. Never invent an id or a timestamp.
Return an empty list if the author never changed position.
`.trim();

/** The passages with their dates, oldest first, for a revision search. */
export function renderDated(passages: Passage[]): string {
  const ordered = [...passages].sort((a, b) => a.published_at.localeCompare(b.published_at));
  return `PASSAGES:\n${
    ordered.map((p) => `[${p.id}] (${p.published_at}) ${p.text}`).join("\n\n")
  }`;
}

export type VerifiedRevision = Omit<Revision, "earlier_id" | "later_id"> &
  { earlier: Passage; later: Passage; claimProblem: string | null };

/**
 * As the other two, plus the two checks this relation is made of: one author,
 * and the correction genuinely after the claim.
 *
 * The direction is checked rather than trusted. The model is given the dates
 * and can still return the pair the wrong way round, and a reversed pair reads
 * as a correction that never happened — which would put the `pro` and `contra`
 * labels on the wrong spans and quietly invert the question.
 */
export function verifyRevisions(
  revisions: Revision[],
  offered: Passage[],
): { kept: VerifiedRevision[]; rejected: { revision: Revision; reason: string }[] } {
  const byId = new Map(offered.map((p) => [p.id, p]));
  const find = (id: string) => byId.get(normalizeId(id));
  const kept: VerifiedRevision[] = [];
  const rejected: { revision: Revision; reason: string }[] = [];

  for (const r of revisions) {
    const earlier = find(r.earlier_id);
    const later = find(r.later_id);
    let reason = rejectionReason(r.claim, [r.earlier_id, r.later_id], [earlier, later]);
    if (!reason && earlier && later) {
      if (earlier.channel !== later.channel) {
        reason = `different authors, ${earlier.channel} and ${later.channel} — that is a clash, not a revision`;
      } else if (later.published_at <= earlier.published_at) {
        reason = `the cited later passage is not later: ${later.published_at} against ${earlier.published_at}`;
      }
    }
    if (reason) rejected.push({ revision: r, reason });
    else kept.push({ ...r, earlier: earlier!, later: later!, claimProblem: claimProblem(r.claim) });
  }
  return { kept, rejected };
}
