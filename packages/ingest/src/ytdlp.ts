// yt-dlp wrappers. Every call goes through here so the two-stage shape stays
// visible: one cheap request per channel, then one full request per surviving
// candidate.
//
// Why two stages. `--flat-playlist` lists a whole channel in one request and
// returns id, title, duration and view_count — but no date at all (`timestamp`
// comes back null). The date filter therefore needs a per-video request. That
// request also reports subtitle availability, so `transcript_kind` is decided
// in the same call rather than by a separate probe.
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);
const MAX_BUFFER = 64 * 1024 * 1024;

export type FlatEntry = {
  youtube_id: string;
  title: string;
  duration_s: number | null;
  view_count: number | null;
};

export async function listChannel(channelId: string, limit: number): Promise<FlatEntry[]> {
  const url = `https://www.youtube.com/channel/${channelId}/videos`;
  const { stdout } = await run(
    "yt-dlp",
    [
      "--flat-playlist", "--no-warnings", "--ignore-errors",
      "--playlist-end", String(limit),
      "--print", "%(id)s\t%(title)s\t%(duration)s\t%(view_count)s",
      url,
    ],
    { maxBuffer: MAX_BUFFER },
  );

  return stdout.split("\n").filter(Boolean).map((line) => {
    const [id, title, duration, views] = line.split("\t");
    return {
      youtube_id: id ?? "",
      title: title ?? "",
      duration_s: num(duration),
      view_count: num(views),
    };
  });
}

export type VideoMeta = {
  youtube_id: string;
  title: string;
  channel: string;
  published_at: string | null;   // ISO date
  duration_s: number | null;
  view_count: number | null;
  has_manual_en: boolean;
  has_auto_en: boolean;
};

/**
 * A failed fetch is two different things and they need opposite handling.
 *
 * `gone` is a property of the video: private, removed, members-only,
 * region-blocked. It will be just as gone tomorrow, so it is a normal outcome —
 * record it and move on.
 *
 * `throttled` is a property of the moment: YouTube's "Sign in to confirm you're
 * not a bot", or a 429. The video is fine; we are the problem. Treating this
 * like `gone` is how a run silently loses a channel — on the first full run,
 * 87 videos were recorded as unavailable when they were nothing of the sort,
 * and the last two channels came out empty because of it.
 */
export type FetchResult =
  | { kind: "ok"; meta: VideoMeta }
  | { kind: "gone"; reason: string }
  | { kind: "throttled"; reason: string };

const THROTTLED = /confirm you.?re not a bot|HTTP Error 429|Too Many Requests|rate.?limit/i;
const GONE = /Video unavailable|Private video|has been removed|members-only|account associated .* has been terminated|violat|not available in your country|age.?restricted/i;

export async function fetchMeta(youtubeId: string, cookiesFrom?: string): Promise<FetchResult> {
  const args = ["--skip-download", "--no-warnings", "-J"];
  if (cookiesFrom) args.push("--cookies-from-browser", cookiesFrom);
  args.push(`https://www.youtube.com/watch?v=${youtubeId}`);

  try {
    const { stdout } = await run("yt-dlp", args, { maxBuffer: MAX_BUFFER });
    const d = JSON.parse(stdout);
    return {
      kind: "ok",
      meta: {
        youtube_id: youtubeId,
        title: d.title ?? "",
        channel: d.channel ?? d.uploader ?? "",
        published_at: isoDate(d.upload_date),
        duration_s: typeof d.duration === "number" ? d.duration : null,
        view_count: typeof d.view_count === "number" ? d.view_count : null,
        has_manual_en: hasEnglish(d.subtitles),
        has_auto_en: hasEnglish(d.automatic_captions),
      },
    };
  } catch (e) {
    const message = String((e as { stderr?: string; message?: string }).stderr
      ?? (e as Error).message ?? e).trim().split("\n").at(-1) ?? "";
    if (THROTTLED.test(message)) return { kind: "throttled", reason: message };
    if (GONE.test(message)) return { kind: "gone", reason: message };
    // Unrecognized failures are treated as transient. Being wrong that way
    // costs a retry; being wrong the other way poisons the cache permanently.
    return { kind: "throttled", reason: message };
  }
}

const num = (s: string | undefined): number | null => {
  const n = Number(s);
  return s && s !== "NA" && Number.isFinite(n) ? n : null;
};

/** yt-dlp returns YYYYMMDD; everything downstream wants YYYY-MM-DD. */
const isoDate = (d: unknown): string | null =>
  typeof d === "string" && /^\d{8}$/.test(d)
    ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
    : null;

/**
 * `automatic_captions` lists ~150 languages because YouTube auto-translates
 * from one source track. Only an actual `en*` key means an English transcript
 * we can fetch.
 */
const hasEnglish = (track: unknown): boolean =>
  !!track && typeof track === "object" &&
  Object.keys(track as object).some((k) => k === "en" || k.startsWith("en-"));
