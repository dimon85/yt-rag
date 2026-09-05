import { describe, expect, test } from "vitest";
import {
  coverageSeconds, decodeEntities, fingerprint, normalizeTranscript, toolMentions,
  toolsInText, type Segment,
} from "../src/text.ts";

const seg = (text: string, start_s: number, end_s: number): Segment => ({ text, start_s, end_s });

const TOOLS = [
  { id: "claude-code", aliases: ["claude code", "cc"] },
  { id: "cursor", aliases: ["cursor", "composer"] },
];

describe("normalizeTranscript", () => {
  test("joins segments and strips case and punctuation", () => {
    expect(normalizeTranscript([seg("Hello, World!", 0, 1), seg("It's fine.", 1, 2)]))
      .toBe("hello world it s fine");
  });

  test("drops bracketed caption artefacts", () => {
    // [Music] and [Applause] appear in auto-captions and differ between
    // uploads of the same talk, so they must not affect identity.
    expect(normalizeTranscript([seg("[Music] hello there [Applause]", 0, 1)]))
      .toBe("hello there");
  });
});

describe("normalizeTranscript as identity", () => {
  test("two uploads of the same talk collapse to one fingerprint", () => {
    const a = [seg("[Music]", 0, 1), seg("Welcome back, folks!", 1, 4)];
    const b = [seg("welcome back folks", 0, 3)];
    expect(fingerprint(normalizeTranscript(a))).toBe(fingerprint(normalizeTranscript(b)));
  });

  test("different talks do not", () => {
    const a = [seg("welcome back", 0, 1)];
    const b = [seg("goodbye now", 0, 1)];
    expect(fingerprint(normalizeTranscript(a))).not.toBe(fingerprint(normalizeTranscript(b)));
  });

  test("fingerprint is stable across calls", () => {
    const n = normalizeTranscript([seg("same text", 0, 1)]);
    expect(fingerprint(n)).toBe(fingerprint(n));
  });
});

describe("toolMentions", () => {
  test("counts every mention, not just the first", () => {
    const n = normalizeTranscript([seg("cursor is fine, cursor again, and cursor once more", 0, 1)]);
    expect(toolMentions(n, TOOLS).get("cursor")).toBe(3);
  });

  test("counts across aliases of the same tool", () => {
    const n = normalizeTranscript([seg("claude code and cc are the same thing", 0, 1)]);
    expect(toolMentions(n, TOOLS).get("claude-code")).toBe(2);
  });

  test("overlapping aliases count the phrase once", () => {
    // ["cursor", "cursor composer", "composer"] all fire on "cursor composer".
    // Counting each alias made one utterance into three mentions, enough to
    // clear the tagging threshold on its own.
    const tools = [{ id: "cursor", aliases: ["cursor", "cursor composer", "cursor ide", "composer"] }];
    const n = normalizeTranscript([seg("i use cursor composer daily", 0, 1)]);
    expect(toolMentions(n, tools).get("cursor")).toBe(1);
  });

  test("separate occurrences still count separately", () => {
    const tools = [{ id: "cursor", aliases: ["cursor", "composer"] }];
    const n = normalizeTranscript([seg("cursor and composer are different", 0, 1)]);
    expect(toolMentions(n, tools).get("cursor")).toBe(2);
  });

  test("adjacent repeats are not merged", () => {
    const tools = [{ id: "cursor", aliases: ["cursor"] }];
    const n = normalizeTranscript([seg("cursor cursor cursor", 0, 1)]);
    expect(toolMentions(n, tools).get("cursor")).toBe(3);
  });

  test("matches whole words only", () => {
    const n = normalizeTranscript([seg("soccer and accordion", 0, 1)]);
    expect(toolMentions(n, TOOLS).get("claude-code")).toBeUndefined();
  });

  test("a tool never mentioned is absent, not zero", () => {
    const n = normalizeTranscript([seg("nothing relevant here", 0, 1)]);
    expect(toolMentions(n, TOOLS).size).toBe(0);
  });
});

describe("toolsInText", () => {
  test("one passing mention does not tag the video", () => {
    // The whole point of counting: a video that says 'Cursor' once is not a
    // video about Cursor, and tagging it corrupts the per-tool channel check.
    const n = normalizeTranscript([seg("we also looked at cursor briefly", 0, 1)]);
    expect(toolsInText(n, TOOLS, 3)).toEqual([]);
  });

  test("sustained discussion does", () => {
    const n = normalizeTranscript([seg("cursor cursor cursor cursor", 0, 1)]);
    expect(toolsInText(n, TOOLS, 3)).toEqual(["cursor"]);
  });

  test("results follow corpus order, not mention order", () => {
    const n = normalizeTranscript([
      seg("cursor cursor cursor and claude code claude code claude code", 0, 1),
    ]);
    expect(toolsInText(n, TOOLS, 3)).toEqual(["claude-code", "cursor"]);
  });
});

describe("coverageSeconds", () => {
  test("spans first start to last end", () => {
    expect(coverageSeconds([seg("a", 10, 12), seg("b", 300, 305)])).toBe(295);
  });

  test("an empty transcript covers nothing", () => {
    expect(coverageSeconds([])).toBe(0);
  });
});

describe("decodeEntities", () => {
  test("decodes the entities YouTube actually emits", () => {
    // &#39; alone occurs 11,709 times in this corpus.
    expect(decodeEntities("That&#39;s 5 &gt; 3 &amp; &quot;fine&quot;"))
      .toBe(`That's 5 > 3 & "fine"`);
  });

  test("handles hex and named forms", () => {
    expect(decodeEntities("&#x27;a&apos;b&nbsp;c")).toBe("'a'b c");
  });

  test("leaves unknown entities alone rather than guessing", () => {
    expect(decodeEntities("a &notareal; b")).toBe("a &notareal; b");
  });

  test("decodes once, so a literal ampersand survives", () => {
    // Text that genuinely contains "&#39;" arrives as "&amp;#39;". Decoding
    // twice would silently turn it into an apostrophe.
    expect(decodeEntities("write &amp;#39; to escape")).toBe("write &#39; to escape");
  });

  test("text without entities is untouched", () => {
    expect(decodeEntities("plain text, nothing to do")).toBe("plain text, nothing to do");
  });
});
