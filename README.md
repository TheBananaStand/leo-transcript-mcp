# leo-transcript-mcp

Pull the spoken words out of a YouTube or TikTok URL, as a Leo package over MCP
— timestamped text plus title, channel and duration.

This is the same integration as the compiled `leo-transcript` package, reachable
as a package the hub **installs at runtime** rather than one it has to be
rebuilt for. Install it from the Store; there is nothing to compile.

It stops at the transcript, deliberately: it does not summarise, classify, or
interpret. Whatever asked for the transcript decides what it means.

## Tools

| Tool | Requires | Also takes |
|---|---|---|
| `transcript` | `url` | `lang` (default `en`), `asr` (default `true`), `asr_model` (default `small`) |

One tool, because there is one thing to do. The result is a JSON object:

```
url, platform, title, channel, duration_s, upload_date,
caption_source, lang, n_segments, chars, segments[], text
```

`text` is the whole transcript with `[h:mm:ss]` markers inline; `segments` is the
same thing as `{ t, text }` pairs.

## Why the timestamps are not a detail

The consumer is almost always a language model, and the failure mode is almost
always the same one: asked to extract something from a long, disfluent
transcript, a model produces a clean and plausible answer whether or not the
video supports it. A consumer can only defend against that by making the model
cite a timestamp and then checking the citation — and it cannot check a citation
the model was never given the means to make. So the markers ride along in the
prose even though nothing here reads them.

## Captions first, then speech-to-text

Most YouTube has machine captions; a lot of TikTok has none at all. When there
is no caption track the audio is transcribed **on this machine** rather than
giving up — otherwise "is this video readable" is really the question "did
somebody upload subtitles", which silently decides what a consumer is allowed to
research. Audio never leaves the box.

It never falls back to the video description. A description is not a transcript,
and a consumer that received one silently would have no way to tell.

## A machine transcript is different evidence, and it says so

Whisper's characteristic error is **fluent, grammatical text for audio that
contains no such speech**. A consumer that defends itself by checking quotes
against the transcript is fully satisfied by that — the words genuinely are in
the transcript. Quote-checking pins the consumer's own model; it cannot pin this
one.

So a speech-to-text result is marked `caption_source: "asr"` and carries an
`asr` block with the model, the detected language, the decoder's mean
log-probability and:

```json
"confident": false
```

when the decoder was unsure or heard mostly non-speech. **Branch on that field.**
A low-confidence transcript is not-evidence, not a slightly worse transcript.
The block is absent entirely on the caption path, so its absence means "this came
from real captions" rather than "the decoder happened not to report".

The language is **detected, never forced**. Forcing one does not fail on a video
in another language — Whisper quietly *translates*, and the result reads as a
confident transcript of words nobody said.

## Requirements

Two binaries on the box: **yt-dlp** (required) and **whisper-ctranslate2** (only
for the captionless path). A CTranslate2 build rather than openai-whisper: it
runs acceptably on CPU and needs no ffmpeg binary and no GPU, so the fallback
works on a plain machine rather than only where someone has provisioned
inference.

```bash
pipx install yt-dlp whisper-ctranslate2
```

Prefer that over a distro package. Distro builds of yt-dlp lag, and a stale
yt-dlp is the characteristic failure of this tool: YouTube changes caption
delivery every few months and an old build returns nothing rather than erroring,
so the transcript comes back empty and nothing says why.

Both are found automatically. The search looks in Leo's own `bin` directory
first, then `PATH`, then `~/.local/bin` and `~/.local/pipx/venvs` — because Leo
runs as a service and a service `PATH` is typically just
`/usr/local/bin:/usr/bin`, which is how a binary the owner's own shell runs
perfectly comes back "not installed".

## Configuration

Two settings, both **optional**: `yt_dlp_path` and `whisper_path`. They are the
escape hatch for a machine where those binaries live somewhere the search does
not reach. Leo hands an entitled setting to this process under its settings key
**verbatim and lower-case**, so the descriptor's `settings_read` and
`process.env.yt_dlp_path` have to agree or the setting silently never arrives.

A configured path that points at nothing is an error, not a fallback: it must
read as "your path is wrong" rather than quietly resolving to some other copy
that behaves differently.

Without either binary the server still starts and still lists its tool; a call
answers with the binary named and how to install it. A server that refused to
launch would show up as a broken package rather than an unconfigured one.

## What it sends, and where

The video URL, to the site it names — and nothing else. Fetching the page and
its caption track is an ordinary anonymous request: no account, no conversation,
no file contents are attached. Audio, when there is no caption track, is
downloaded to a scratch directory that removes itself and transcribed locally;
it is never uploaded anywhere.

Every subprocess is spawned with an argument **array**, never a shell string, so
a URL containing `;` or `$( )` is one argument and not a command. The URL is
additionally required to be `http`/`https` before it reaches an argv — an array
closes command injection but not *argument* injection, and a "URL" of
`--config-location=…` or `--exec=…` is an ordinary array element that yt-dlp
would parse as an option and obey.

## Development

```bash
npm install
node test.js        # no network, no subprocesses
```

The test covers only what fails *quietly* — yt-dlp and Whisper fail loudly and
are somebody else's to get right:

- **URL classification and refusal** — `platform` is how a consumer weights the
  result, and a yt-dlp option dressed as a URL is obeyed rather than fetched.
- **Timestamps.** `Number("")` is `0` in JavaScript, so a cue header the parser
  half-understood would come back as a confident `0` and stamp every following
  line at the start of the video — a citation that looks checkable and is not.
- **Which caption file is chosen.** One request routinely writes several;
  directory order is undefined, and on a Spanish video with English
  auto-translation `t.en.json3` and `t.en-orig.json3` are *different languages*.
  The wrong pick is a fluent, well-formed transcript of the wrong thing.
- **Absent captions versus an empty transcript.** One means transcribe the
  audio; the other has to be an error. A transcript with no words in it reads as
  "the speaker said nothing".
- **The confidence gate**, including its boundary and the music case (a healthy
  log-probability with a high `no_speech_prob`), because it is the only thing
  standing between a hallucination and a quotable transcript.
- **The argv handed to each subprocess.** A dropped `--write-auto-subs` makes
  most of YouTube look captionless and silently routes it through minutes of CPU
  for a weaker result; a `--language` added to the decoder turns it into a
  translator.

## Publishing

```bash
./store/publish.sh          # live
./store/publish.sh draft    # stage for review at admin.leoconnect.io
```

Needs a Cloudflare login with `D1:Edit` on the `leo-store` database. The script
refuses to publish unless the commit pinned in `store/registry-entry.json` is
both real and pushed — a SHA that resolves nowhere installs cleanly and then
fails on every hub at first launch, which is the one failure it can prevent and
nothing downstream can.
