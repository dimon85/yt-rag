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
export type Verified = Omit<Clash, "pro" | "contra"> & { pro: Passage; contra: Passage };

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
  const kept: Verified[] = [];
  const rejected: { clash: Clash; reason: string }[] = [];

  for (const c of clashes) {
    const pro = byId.get(c.pro);
    const contra = byId.get(c.contra);
    if (!pro || !contra) {
      const missing = [!pro && c.pro, !contra && c.contra].filter(Boolean).join(", ");
      rejected.push({ clash: c, reason: `cites unknown passage ${missing}` });
      continue;
    }
    if (pro.video === contra.video) {
      rejected.push({ clash: c, reason: `both sides are in video ${pro.video}` });
      continue;
    }
    if (/\?\s*$/.test(c.claim)) {
      rejected.push({ clash: c, reason: "claim is phrased as a question" });
      continue;
    }
    kept.push({ ...c, pro, contra });
  }
  return { kept, rejected };
}
