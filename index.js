#!/usr/bin/env node
//
// Transcript — pull the spoken words out of a YouTube or TikTok URL, over MCP.
//
// The same integration as the compiled `leo-transcript` package, reachable as a
// package the hub installs at runtime instead of one it is rebuilt for. One
// tool, `transcript`, that turns a video URL into timestamped text. It stops
// there deliberately: it does not summarise, classify, or interpret. Whatever
// asked for the transcript decides what it means.
//
// ## Why the timestamps are not a detail
//
// The output carries `[h:mm:ss]` markers inline, not just as structured
// segments, because the consumer is almost always a language model and the
// failure mode is almost always the same one: asked to extract something from a
// long, disfluent transcript, a model produces a clean and plausible answer
// whether or not the video supports it. A consumer can only defend against that
// by making the model cite a timestamp and then checking the citation — and it
// cannot check a citation the model was never given the means to make. So the
// timestamps ride along even though nothing here reads them.
//
// ## Captions first, then speech-to-text
//
// This prefers caption tracks that already exist. Most YouTube has machine
// captions; a lot of TikTok has none at all. When there is no track the audio is
// transcribed locally rather than giving up — otherwise "is this video readable"
// is really the question "did somebody upload subtitles", which silently decides
// what a consumer is allowed to research. It never falls back to the video
// description: a description is not a transcript, and a consumer that received
// one silently would have no way to tell.
//
// ## A machine transcript is different evidence, and it says so
//
// Whisper's characteristic error is *fluent, grammatical text for audio that
// contains no such speech* — so a consumer that defends itself by checking
// quotes against the transcript is fully satisfied by a hallucination, because
// the words genuinely are in the transcript. So an ASR result is marked
// `caption_source: "asr"`, carries the decoder's own confidence in an `asr`
// block, and sets `asr.confident: false` when the decoder was unsure or heard
// mostly non-speech.
//
// The language is DETECTED, never forced. Forcing a language does not fail on a
// video that is in a different one — Whisper quietly *translates*, and the
// result reads as a confident transcript of words nobody said.
//
// ## Shelling out to yt-dlp is the deliberate choice
//
// YouTube changes caption delivery constantly; yt-dlp exists because keeping up
// with that is a full-time job. Every subprocess is spawned with `execFile` and
// an argument ARRAY — never a shell string — so a URL carrying `;` or `$( )` is
// one argument and not a command.
//
// Plain JavaScript on purpose. This ships as a git tarball rather than an npm
// package, and npm does not reliably run build steps for a tarball URL — a
// TypeScript source tree would install and then fail to start, at the far end,
// on somebody else's machine.

import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// Leo hands an entitled setting to the subprocess under its *settings key*,
// verbatim and lower-case — see `Entitlements::resolve_env_secrets`. These are
// the keys the package's descriptor declares in `settings_read`, so the two
// spellings have to agree or the setting silently never arrives. Both are
// optional: the resolver below finds either binary in every ordinary location,
// and these are the escape hatch for a machine where they live somewhere else.
const YT_DLP_SETTING = "yt_dlp_path";
const WHISPER_SETTING = "whisper_path";

/** yt-dlp resolves a URL, then downloads one caption track. Long enough for a
 *  slow network and a long video; short enough that a wedged fetch does not pin
 *  a tool call forever. */
const FETCH_TIMEOUT_SECS = 120;

/** Transcription is far slower than a caption download — it decodes the whole
 *  audio track on CPU. Generous for a long talk, bounded so a wedged decode
 *  cannot pin a tool call forever. */
const ASR_TIMEOUT_SECS = 900;

/** The decoder's own per-segment confidence. On clean speech `avg_logprob` sits
 *  around -0.1 to -0.4; sustained values below this are the regime where Whisper
 *  emits fluent text for audio that does not contain it. */
const ASR_LOW_CONFIDENCE_LOGPROB = -1.0;

/** Above this, the decoder itself thinks the audio is not speech — music, or
 *  silence it filled in anyway. */
const ASR_MAX_NO_SPEECH_PROB = 0.6;

/** Share of low-confidence segments above which the transcript is not evidence
 *  at all. */
const ASR_MAX_LOW_CONFIDENCE_SHARE = 0.3;

/** Small is the honest default for a fallback: a few seconds of CPU on a short
 *  clip, and good enough that the confidence gate — not the model size — is what
 *  decides trust. Override per call with `asr_model`. */
const DEFAULT_ASR_MODEL = "small";

// ── Locating the binaries ───────────────────────────────────────────────────

/**
 * Where Leo puts binaries it installed itself. The service sets `LEO_DATA_DIR`;
 * `~/.leo/data` is the documented default for a hub that does not.
 */
export function dataDir() {
  return process.env.LEO_DATA_DIR || path.join(os.homedir(), ".leo", "data");
}

/**
 * Find `binary`, or `null` if it is not installed anywhere Leo knows to look.
 *
 * The search order carries a lesson `leo-audible` paid for: **Leo runs as a
 * service, and a service PATH is typically just `/usr/local/bin:/usr/bin`.** It
 * does not include `~/.local/bin`, where pipx and `pip install --user` put
 * things — so a bare `execFile("yt-dlp", …)` fails with ENOENT on a machine
 * where the owner's own shell runs it perfectly, which is a maddening bug to
 * diagnose from the outside.
 *
 * An explicitly configured path wins outright and is NOT fallen back on: a
 * setting that points at nothing must read as "your path is wrong", not
 * silently resolve to some other copy that behaves differently.
 */
export function resolveBinary(binary, configured) {
  const explicit = (configured ?? "").trim();
  if (explicit) return isFile(explicit) ? explicit : null;

  // Leo's own directory first: if Leo installed it, that is the copy Leo should
  // use, even when an older one is earlier on PATH.
  const owned = path.join(dataDir(), "bin", binary);
  if (isFile(owned)) return owned;

  for (const dir of (process.env.PATH ?? "").split(path.delimiter)) {
    if (!dir) continue;
    const cand = path.join(dir, binary);
    if (isFile(cand)) return cand;
  }

  const home = os.homedir();
  const fallbacks = [
    path.join(home, ".local", "bin", binary),
    path.join(home, ".local", "pipx", "venvs", binary, "bin", binary),
    `/usr/local/bin/${binary}`,
    `/usr/bin/${binary}`,
  ];
  return fallbacks.find(isFile) ?? null;
}

function isFile(p) {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

// ── The URL ─────────────────────────────────────────────────────────────────

/**
 * Accept only something yt-dlp will treat as a URL.
 *
 * `execFile` with an argument array closes command injection — a URL containing
 * `;` or backticks is one argument and never reaches a shell. What it does NOT
 * close is *argument* injection into yt-dlp itself: a "URL" of
 * `--config-location=/tmp/x` or `--exec=curl …` is a perfectly ordinary array
 * element that yt-dlp parses as an option and obeys. So the scheme is checked
 * before the URL is ever placed in an argv.
 *
 * Deliberately not restricted to YouTube and TikTok. yt-dlp reads a thousand
 * sites, `platform_of` already reports `other` for the rest, and refusing them
 * here would remove behaviour the compiled package has.
 */
export function checkUrl(url) {
  const s = (url ?? "").trim();
  if (!s) return "no URL was given";
  // Strictly redundant — a WHATWG scheme must start with an ASCII letter, so
  // anything beginning with "-" fails the parse below too. Kept for the
  // message: "yt-dlp would read this as an option" is the sentence that tells
  // the caller what they actually did.
  if (s.startsWith("-")) {
    return `"${s.slice(0, 40)}" starts with "-", so yt-dlp would read it as an option, not a URL`;
  }
  let parsed;
  try {
    parsed = new URL(s);
  } catch {
    return `"${s.slice(0, 60)}" is not a URL`;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return `"${parsed.protocol}" is not a fetchable scheme — give an http or https video URL`;
  }
  return null;
}

/**
 * Which site this is, from the URL alone.
 *
 * Reported in the payload so a consumer can weight the result: a TikTok
 * transcript is far more often machine-heard than a YouTube one.
 */
export function platformOf(url) {
  const u = String(url ?? "").toLowerCase();
  if (u.includes("tiktok.")) return "tiktok";
  if (u.includes("youtube.") || u.includes("youtu.be")) return "youtube";
  return "other";
}

// ── Subprocess argument construction ────────────────────────────────────────

/**
 * The one yt-dlp invocation that writes both the metadata and the caption file,
 * so there is a single network round trip and the two can never describe
 * different videos.
 *
 * `--sub-langs` is scoped to the requested language rather than `all`: YouTube
 * auto-translates captions into ~100 languages, and asking for all of them
 * downloads a hundred files to use one. `-live_chat` subtracts the chat replay,
 * which is a subtitle track by yt-dlp's reckoning and is not speech.
 *
 * Exported because these arguments are the part fixtures cannot check any other
 * way: every one of them fails *quietly* if it is wrong — a dropped
 * `--write-info-json` loses the title and channel, a dropped `--write-auto-subs`
 * makes most of YouTube look captionless and silently routes it through ASR.
 */
export function captionArgs(base, url, lang) {
  return [
    "--skip-download",
    "--no-playlist",
    "--no-warnings",
    "--write-info-json",
    "--write-subs",
    "--write-auto-subs",
    "--sub-format",
    "json3/vtt/best",
    "--sub-langs",
    `${lang}.*,${lang},-live_chat`,
    "-o",
    `${base}.%(ext)s`,
    url,
  ];
}

/**
 * Audio only. The video stream is never fetched, and no format conversion is
 * requested, because conversion is the step that would drag in an ffmpeg binary
 * — the decoder reads whatever container yt-dlp hands back.
 */
export function audioArgs(audioBase, url) {
  return [
    "--no-playlist",
    "--no-warnings",
    "-f",
    "bestaudio/best",
    "-o",
    `${audioBase}.%(ext)s`,
    url,
  ];
}

/**
 * The decoder invocation.
 *
 * There is no `--language` here and there must never be one. Forcing a language
 * does not fail on a video that is in a different one — Whisper quietly
 * *translates*, and the output reads as a confident transcript of words nobody
 * said. `--vad_filter` is what stops music and silence being decoded into lyrics
 * and filler.
 */
export function asrArgs(model, dir, audioPath) {
  return [
    "--model",
    model,
    "--output_format",
    "json",
    "--output_dir",
    dir,
    "--vad_filter",
    "True",
    audioPath,
  ];
}

// ── Running one ─────────────────────────────────────────────────────────────

function run(bin, args, timeoutSecs) {
  return new Promise((resolve) => {
    execFile(
      bin,
      args,
      {
        timeout: timeoutSecs * 1000,
        // yt-dlp's info JSON goes to a file, but its progress and warnings do
        // not; the 1MB default would kill a long fetch as a "failure".
        maxBuffer: 32 * 1024 * 1024,
        encoding: "utf8",
      },
      (error, stdout, stderr) => {
        resolve({
          ok: !error,
          timedOut: Boolean(error?.killed || error?.signal),
          spawnFailed: Boolean(error?.code === "ENOENT" || error?.code === "EACCES"),
          message: error ? String(error.message) : "",
          stdout: stdout ?? "",
          stderr: stderr ?? "",
        });
      },
    );
  });
}

/**
 * yt-dlp's own message is the useful one ("Video unavailable", "Your IP address
 * is blocked from accessing this post"). Pass it through rather than flattening
 * every distinct failure into one unhelpful sentence.
 */
export function ytDlpError(stderr) {
  const lines = String(stderr ?? "").split("\n");
  const named = lines.find((l) => l.includes("ERROR"));
  return (named ?? String(stderr ?? "")).trim();
}

// ── Caption parsing ─────────────────────────────────────────────────────────

/** Collapse all runs of whitespace, the way `str::split_whitespace` does. */
export function clean(s) {
  return String(s ?? "")
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .join(" ");
}

/** Drop `<c.colorE5E5E5>`-style inline cue markup, which YouTube emits inside
 *  VTT and which no consumer wants in a quote. */
export function stripTags(s) {
  let out = "";
  let depth = 0;
  for (const c of String(s ?? "")) {
    if (c === "<") depth += 1;
    else if (c === ">") depth = depth > 0 ? depth - 1 : 0;
    else if (depth === 0) out += c;
  }
  return out;
}

/**
 * `[h:mm:ss]`, with the hour dropped when it is zero.
 *
 * The whole point of the tool is that a consumer can make a model cite a
 * timestamp and then go and check it, so an off-by-anything here is a citation
 * that points at the wrong moment while still looking checkable.
 */
export function hhmmss(seconds) {
  const n = Number(seconds);
  const total = Math.floor(Number.isFinite(n) && n > 0 ? n : 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (v) => String(v).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/**
 * Seconds from the start side of a `00:00:01.000 --> 00:00:03.000` cue header.
 *
 * Returns null for anything that is not a cue header, which is how the VTT
 * parser tells a timing line from a line of speech. `Number("")` is 0 in
 * JavaScript, so the fields are matched against a digits-only pattern rather
 * than coerced — otherwise a malformed header would silently time-stamp a cue
 * at the start of the video.
 */
export function parseVttTimestamp(line) {
  const text = String(line ?? "");
  if (!text.includes("-->")) return null;
  const head = text.split("-->")[0].trim();

  const dot = head.search(/[.,]/);
  const secsPart = dot === -1 ? head : head.slice(0, dot);
  const frac = dot === -1 ? "0" : head.slice(dot + 1);

  const parts = secsPart.split(":");
  const num = (v) => (/^\d+(?:\.\d+)?$/.test(v) ? Number(v) : null);
  let h;
  let m;
  let s;
  if (parts.length === 3) [h, m, s] = parts.map(num);
  else if (parts.length === 2) [h, m, s] = [0, ...parts.map(num)];
  else return null;
  if (h === null || m === null || s === null) return null;

  const millis = /^\d+$/.test(frac) ? Number(`0.${frac}`) : 0;
  return h * 3600 + m * 60 + s + millis;
}

/** YouTube's json3 format. Timings are explicit per cue, so this is the
 *  preferred path. */
export function parseJson3(raw) {
  let doc;
  try {
    doc = JSON.parse(typeof raw === "string" ? raw : Buffer.from(raw).toString("utf8"));
  } catch (e) {
    throw new Error(`could not parse json3 captions: ${e.message}`);
  }
  const out = [];
  for (const ev of doc?.events ?? []) {
    const text = clean((ev?.segs ?? []).map((s) => s?.utf8 ?? "").join(""));
    if (text) out.push([Number(ev?.tStartMs ?? 0) / 1000, text]);
  }
  return out;
}

/**
 * WebVTT / SRT — the fallback, and what TikTok generally serves.
 *
 * YouTube's auto-captions repeat the previous cue as a scrolling window, so a
 * naive parse triples the transcript and makes every line appear several times.
 * Consumers that verify quotes by substring would still pass, but a model
 * reading it sees the same sentence three times and weights it accordingly — so
 * a cue identical to the one before it is dropped.
 */
export function parseVtt(raw) {
  const out = [];
  let start = null;
  let buf = [];

  const flush = () => {
    if (start !== null) {
      const text = clean(buf.join(" "));
      const last = out.length ? out[out.length - 1][1] : null;
      if (text && text !== last) out.push([start, text]);
      start = null;
    }
    buf = [];
  };

  for (const line of String(raw ?? "").split("\n")) {
    const t = parseVttTimestamp(line);
    if (t !== null) {
      flush();
      start = t;
      continue;
    }
    const s = line.trim();
    const isHeader =
      s === "" ||
      s.startsWith("WEBVTT") ||
      s.startsWith("NOTE") ||
      s.startsWith("Kind:") ||
      s.startsWith("Language:") ||
      /^[0-9]+$/.test(s);
    if (isHeader) {
      if (s === "") flush();
      continue;
    }
    buf.push(stripTags(s));
  }
  flush();
  return out;
}

// ── Picking the caption file ────────────────────────────────────────────────

/**
 * Pick the caption file yt-dlp wrote, deterministically, from a list of names.
 *
 * A single request routinely produces several files: asking for `en` on one
 * video wrote both `t.en.json3` and `t.en-orig.json3`. Directory order is not
 * defined, so choosing by "first one seen" picks a different track run to run —
 * and on a Spanish video with English auto-translation those two files are
 * *different languages*, one of them not the one that was asked for. So
 * candidates are scored and sorted:
 *
 *   1. the exact requested language beats a variant (`en` over `en-orig`, `en-US`)
 *   2. json3 beats VTT — its timings are explicit per cue, where VTT timings
 *      have to be parsed out of a text format several tools spell differently
 *   3. filename, purely so ties break the same way every time
 *
 * Takes names rather than reading a directory so the rule can be exercised
 * without a filesystem; `findSubFile` is the thin wrapper that reads one.
 */
export function pickSubName(names, wantLang) {
  const candidates = [];
  for (const name of names) {
    // `t.<lang>.<ext>` — anything else here is the info json or a stray.
    if (!name.startsWith("t.") || name.endsWith(".info.json")) continue;
    const isJson3 = name.endsWith(".json3");
    if (!isJson3 && !name.endsWith(".vtt") && !name.endsWith(".srt")) continue;

    let stem = name;
    while (stem.startsWith("t.")) stem = stem.slice(2);
    const cut = stem.lastIndexOf(".");
    const lang = cut === -1 ? "" : stem.slice(0, cut);

    candidates.push({
      langRank: lang === wantLang ? 0 : 1,
      fmtRank: isJson3 ? 0 : 1,
      name,
      lang,
      isJson3,
    });
  }
  candidates.sort(
    (a, b) =>
      a.langRank - b.langRank ||
      a.fmtRank - b.fmtRank ||
      (a.name < b.name ? -1 : a.name > b.name ? 1 : 0),
  );
  return candidates[0] ?? null;
}

function findSubFile(dir, wantLang) {
  let names;
  try {
    names = fs.readdirSync(dir);
  } catch {
    return null;
  }
  const picked = pickSubName(names, wantLang);
  return picked ? { ...picked, path: path.join(dir, picked.name) } : null;
}

// ── yt-dlp metadata ─────────────────────────────────────────────────────────

/**
 * Whether the chosen track was written by a human or by a machine.
 *
 * Read from yt-dlp's own metadata rather than guessed from the filename. The
 * filename does not encode it: an auto-captioned video writes plain
 * `t.en.json3`, identical in shape to a hand-authored track, so any guess is
 * wrong roughly half the time. It matters because machine captions mangle
 * exactly the jargon a consumer is usually looking for, and a consumer that
 * knows the source can weight a quote accordingly.
 */
export function captionSource(info, lang) {
  // `en-orig` is the file yt-dlp writes for the original track behind an `en`
  // request, so match on the base code as well as the exact tag.
  const base = String(lang ?? "").split("-")[0];
  const subs = info?.subtitles ?? {};
  const manual =
    Object.prototype.hasOwnProperty.call(subs, lang) ||
    Object.prototype.hasOwnProperty.call(subs, base);
  return manual ? "subtitles" : "automatic_captions";
}

/**
 * Languages the video actually offers — quoted back when the requested one is
 * missing, because "no en captions" is not actionable and "available: es, pt"
 * is.
 */
export function availableLangs(info) {
  const keys = [
    ...Object.keys(info?.subtitles ?? {}),
    ...Object.keys(info?.automatic_captions ?? {}),
  ].filter((k) => k !== "live_chat");
  return [...new Set(keys)].sort().slice(0, 12);
}

// ── Speech-to-text ──────────────────────────────────────────────────────────

/** Rust's `f64::round` is half-away-from-zero; JavaScript's `Math.round` is
 *  half-up, which differs on the negative values every mean log-probability
 *  is. */
function round4(v) {
  const scaled = v * 10000;
  const rounded = Math.sign(scaled) * Math.round(Math.abs(scaled));
  return rounded / 10000;
}

/**
 * Whisper's JSON output -> segments plus a confidence report.
 *
 * Pure, and separate from the process plumbing, because the confidence rule is
 * the part that decides whether a machine transcript counts as evidence at all
 * — and a rule that can only be exercised by downloading a model and a video is
 * a rule that does not get tested.
 *
 * Throws when the decode produced no speech. Returning an empty transcript
 * instead would let a consumer extract confidently from a video it never heard.
 */
export function parseAsrJson(raw, model) {
  let doc;
  try {
    doc = JSON.parse(raw);
  } catch (e) {
    throw new Error(`could not parse the speech-to-text output: ${e.message}`);
  }

  const segments = [];
  const logprobs = [];
  let low = 0;
  for (const s of doc?.segments ?? []) {
    const text = clean(s?.text ?? "");
    if (!text) continue;
    const logprob = Number(s?.avg_logprob ?? 0);
    const noSpeech = Number(s?.no_speech_prob ?? 0);
    logprobs.push(logprob);
    if (logprob < ASR_LOW_CONFIDENCE_LOGPROB || noSpeech > ASR_MAX_NO_SPEECH_PROB) low += 1;
    segments.push([Number(s?.start ?? 0), text]);
  }

  if (segments.length === 0) {
    throw new Error(
      "speech-to-text produced no speech — the audio is probably music or silence",
    );
  }

  const n = segments.length;
  const share = low / n;
  const mean = logprobs.reduce((a, b) => a + b, 0) / logprobs.length;
  const report = {
    model,
    language: typeof doc?.language === "string" ? doc.language : "",
    n_segments: n,
    mean_avg_logprob: round4(mean),
    low_confidence_segments: low,
    low_confidence_share: round4(share),
    // The one field a consumer must branch on. False means the transcript is not
    // evidence — not that it is a slightly worse transcript.
    confident: share <= ASR_MAX_LOW_CONFIDENCE_SHARE,
    caveat:
      "Machine transcript. Whisper's characteristic error is fluent text for " +
      "audio containing no such speech, which survives any check that matches " +
      "quotes against this transcript.",
  };
  return { segments, report };
}

/** Download the audio and transcribe it locally. */
async function transcribe(asr, ytDlp, url, dir) {
  const audioBase = path.join(dir, "audio");
  const dl = await run(ytDlp, audioArgs(audioBase, url), FETCH_TIMEOUT_SECS);
  if (dl.timedOut) throw new Error(`audio download timed out after ${FETCH_TIMEOUT_SECS}s`);
  if (dl.spawnFailed) throw new Error(`could not run yt-dlp for audio: ${dl.message}`);
  if (!dl.ok) throw new Error(`could not download audio for ${url}: ${ytDlpError(dl.stderr)}`);

  // yt-dlp names the file by the format it chose, so the extension is not known
  // in advance. Largest wins: a partial or a thumbnail sitting alongside is
  // always the smaller file.
  const audio = fs
    .readdirSync(dir)
    .filter((n) => n.startsWith("audio."))
    .map((n) => {
      const p = path.join(dir, n);
      let size = 0;
      try {
        size = fs.statSync(p).size;
      } catch {
        size = 0;
      }
      return { p, size };
    })
    .sort((a, b) => b.size - a.size)[0];
  if (!audio) throw new Error(`no audio file was downloaded for ${url}`);

  const decode = await run(asr.bin, asrArgs(asr.model, dir, audio.p), ASR_TIMEOUT_SECS);
  if (decode.timedOut) throw new Error(`speech-to-text timed out after ${ASR_TIMEOUT_SECS}s`);
  if (decode.spawnFailed) throw new Error(`could not run speech-to-text: ${decode.message}`);
  if (!decode.ok) {
    const lines = decode.stderr.split("\n").filter((l) => l.trim());
    throw new Error(
      `speech-to-text failed for ${url}: ${(lines[lines.length - 1] ?? "(no output)").trim()}`,
    );
  }

  // whisper-ctranslate2 writes `<stem>.json` beside the audio it was given.
  const jsonPath = audio.p.replace(/\.[^.]*$/, "") + ".json";
  let raw;
  try {
    raw = fs.readFileSync(jsonPath, "utf8");
  } catch (e) {
    throw new Error(`speech-to-text wrote no transcript for ${url}: ${e.message}`);
  }
  return parseAsrJson(raw, asr.model);
}

// ── Fetch ───────────────────────────────────────────────────────────────────

async function fetchTranscript(bin, url, lang, asr) {
  const dir = path.join(os.tmpdir(), `leo-transcript-${crypto.randomUUID()}`);
  fs.mkdirSync(dir, { recursive: true });
  try {
    const base = path.join(dir, "t");
    const out = await run(bin, captionArgs(base, url, lang), FETCH_TIMEOUT_SECS);
    if (out.timedOut) throw new Error(`yt-dlp timed out after ${FETCH_TIMEOUT_SECS}s`);
    if (out.spawnFailed) throw new Error(`could not run yt-dlp: ${out.message}`);
    if (!out.ok) throw new Error(`yt-dlp failed for ${url}: ${ytDlpError(out.stderr)}`);

    let info;
    try {
      info = JSON.parse(fs.readFileSync(path.join(dir, "t.info.json"), "utf8"));
    } catch (e) {
      throw new Error(`yt-dlp wrote no usable metadata for ${url}: ${e.message}`);
    }

    // No caption track is the common TikTok case, not an exotic one —
    // transcribe the audio rather than declaring the video unreadable.
    const sub = findSubFile(dir, lang);
    let segments;
    let source;
    let outLang;
    let asrReport = null;

    if (sub) {
      const raw = fs.readFileSync(sub.path);
      segments = sub.isJson3 ? parseJson3(raw) : parseVtt(raw.toString("utf8"));
      if (segments.length === 0) {
        throw new Error(
          `the caption track for ${url} parsed to nothing — it may be empty or in ` +
            "an unexpected format",
        );
      }
      source = captionSource(info, sub.lang);
      outLang = sub.lang;
    } else {
      if (!asr) {
        const offered = availableLangs(info);
        const detail = offered.length
          ? `available languages: ${offered.join(", ")}`
          : "the video has no caption track at all";
        throw new Error(
          `no ${lang} captions for ${url} — ${detail}, and speech-to-text is ` +
            "unavailable or disabled. Retry with an available `lang` if one fits, " +
            "or install the speech-to-text engine for this package.",
        );
      }
      const result = await transcribe(asr, bin, url, dir);
      segments = result.segments;
      asrReport = result.report;
      source = "asr";
      outLang = result.report.language ?? "";
    }

    const text = segments.map(([t, s]) => `[${hhmmss(t)}] ${s}`).join("\n");

    const payload = {
      url,
      platform: platformOf(url),
      title: info.title ?? "",
      // `uploader` first and `channel` only when it is absent — matching the
      // compiled package, where an empty `uploader` still counts as present.
      channel: info.uploader ?? info.channel ?? "",
      duration_s: info.duration ?? 0,
      upload_date: info.upload_date ?? "",
      caption_source: source,
      lang: outLang,
      n_segments: segments.length,
      // Bytes, not UTF-16 units — the compiled package reports `text.len()`, and
      // for a Japanese transcript the two differ by a factor of three.
      chars: Buffer.byteLength(text, "utf8"),
      segments: segments.map(([t, s]) => ({ t, text: s })),
      text,
    };

    // Only present on the ASR path, so its absence means "this came from real
    // captions" rather than "the decoder happened not to report". A consumer
    // can branch on the key.
    if (asrReport) payload.asr = asrReport;
    return payload;
  } finally {
    // A failed fetch must not leak a scratch directory of audio.
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// ── Tool ────────────────────────────────────────────────────────────────────

const TOOLS = {
  transcript: {
    description:
      "Fetch the transcript of a YouTube or TikTok video from its URL. Returns " +
      "timestamped text plus title, channel and duration. Prefers the video's own " +
      "caption track; when there is none it transcribes the audio locally and sets " +
      '`caption_source` to "asr". CHECK THAT FIELD: an "asr" transcript is what ' +
      "a machine heard, not what the speaker is known to have said, and its `asr` " +
      "block carries the decoder's confidence — when `asr.confident` is false, treat " +
      "the transcript as unreliable rather than as slightly worse. Use the timestamps " +
      "when quoting, and quote rather than paraphrase when the exact wording matters.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "YouTube or TikTok video URL" },
        lang: {
          type: "string",
          description:
            "Preferred caption language code. Default: en. Ignored when falling " +
            "back to speech-to-text, which detects the language rather than being " +
            "told it.",
        },
        asr: {
          type: "boolean",
          description:
            "Allow speech-to-text when there is no caption track. Default true. " +
            "Set false to get an error instead of a machine transcript.",
        },
        asr_model: {
          type: "string",
          description:
            "Whisper model for the fallback. Default small. Larger is more " +
            "accurate and much slower: base/small/medium/large-v3/large-v3-turbo.",
        },
      },
      required: ["url"],
    },
    run: async (a) => {
      const url = String(a.url).trim();
      const bad = checkUrl(url);
      if (bad) throw new Error(bad);

      const rawLang = String(a.lang ?? "").trim();
      const lang = rawLang || "en";
      const allowAsr = a.asr === undefined || a.asr === null ? true : Boolean(a.asr);
      const model = String(a.asr_model ?? "").trim() || DEFAULT_ASR_MODEL;

      const bin = resolveBinary("yt-dlp", process.env[YT_DLP_SETTING]);
      if (!bin) {
        // Named precisely, because the failure the owner will actually hit is a
        // binary that is installed somewhere this could not see — not a broken
        // server.
        throw new Error(
          "yt-dlp is not installed, or is somewhere Leo cannot see. Install it " +
            `(\`pipx install yt-dlp\`) or set ${YT_DLP_SETTING} in the Transcript ` +
            "package settings.",
        );
      }

      const whisper = allowAsr
        ? resolveBinary("whisper-ctranslate2", process.env[WHISPER_SETTING])
        : null;
      const asr = whisper ? { bin: whisper, model } : null;

      const payload = await fetchTranscript(bin, url, lang, asr);
      return JSON.stringify(payload, null, 2);
    },
  },
};

/** Every tool declares its own required parameters; this enforces them. */
export function missingRequired(name, args) {
  const required = TOOLS[name]?.inputSchema?.required ?? [];
  return required.filter((key) => {
    const v = args?.[key];
    return v === undefined || v === null || v === "";
  });
}

const server = new Server(
  { name: "leo-transcript-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: Object.entries(TOOLS).map(([name, t]) => ({
    name,
    description: t.description,
    inputSchema: t.inputSchema,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const tool = TOOLS[name];
  if (!tool) {
    return {
      isError: true,
      content: [{ type: "text", text: `Unknown tool: ${name}` }],
    };
  }
  const missing = missingRequired(name, args);
  if (missing.length > 0) {
    // Answered here rather than sent to a subprocess, so a forgotten parameter
    // costs nothing and reads as what it is.
    return {
      isError: true,
      content: [{ type: "text", text: `${name} requires: ${missing.join(", ")}` }],
    };
  }
  try {
    return { content: [{ type: "text", text: await tool.run(args ?? {}) }] };
  } catch (error) {
    // `isError` is what the hub reads to tell "the video has no captions" from
    // "the fetch did not run" — the bridge turns this into a real error rather
    // than an empty transcript, which is the one outcome a consumer must never
    // mistake for silence.
    return {
      isError: true,
      content: [{ type: "text", text: String(error?.message ?? error) }],
    };
  }
});

// Importing this file for its helpers must not also start a server on stdio.
if (process.env.LEO_TRANSCRIPT_MCP_NO_SERVE !== "1") {
  await server.connect(new StdioServerTransport());
}
