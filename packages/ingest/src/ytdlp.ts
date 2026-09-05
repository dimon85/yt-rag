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

export async function fetchMeta(youtubeId: string): Promise<VideoMeta | null> {
  try {
    const { stdout } = await run(
      "yt-dlp",
      ["--skip-download", "--no-warnings", "-J", `https://www.youtube.com/watch?v=${youtubeId}`],
      { maxBuffer: MAX_BUFFER },
    );
    const d = JSON.parse(stdout);
    return {
      youtube_id: youtubeId,
      title: d.title ?? "",
      channel: d.channel ?? d.uploader ?? "",
      published_at: isoDate(d.upload_date),
      duration_s: typeof d.duration === "number" ? d.duration : null,
      view_count: typeof d.view_count === "number" ? d.view_count : null,
      has_manual_en: hasEnglish(d.subtitles),
      has_auto_en: hasEnglish(d.automatic_captions),
    };
  } catch {
    // Private, removed, region-blocked, age-gated: all normal outcomes here.
    // The caller records the id as unavailable rather than failing the run.
    return null;
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
