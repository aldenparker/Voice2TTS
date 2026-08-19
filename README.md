# Voice2TTS

Speak into your microphone; a synthetic voice says the same words into a virtual
microphone that Discord, Zoom, OBS or any other app can pick up.

    your mic → Silero VAD → Whisper (STT) → Piper (TTS) → N output devices
                                                          ├─ CABLE Input → Discord
                                                          └─ your headphones (monitor)

Everything runs locally. No audio leaves the machine.

## Install

Run `Voice2TTS-Setup-<version>.exe`. It installs per-user, so no administrator
prompt is needed.

Windows will show **"Windows protected your PC"** because the installer is not
code-signed — choose *More info* → *Run anyway*. Code signing certificates require a
registered business identity, which this project doesn't have.

The app works the moment it finishes installing: three voices and a CPU speech
recognition model ship in the installer, with nothing else to download.

### First run

A wizard offers two optional extras:

**A virtual microphone.** Discord can only hear Voice2TTS through a virtual audio
device, and Windows only lets a signed kernel driver create one. The wizard finds
any device you already have and skips this step if so:

| Product | Devices |
|---|---|
| VB-CABLE | `CABLE Input` ↔ `CABLE Output` |
| VB-CABLE A+B / C+D | `CABLE-A` … `CABLE-D` |
| VB-Audio Matrix | `VBMatrix In 1–8` ↔ `VBMatrix Out 1–8` |
| VoiceMeeter / Banana / Potato | VAIO, AUX VAIO, VAIO3 |
| VB-Audio Hi-Fi Cable | `Hi-Fi Cable Input` ↔ `Output` |
| Virtual Audio Cable (VAC) | `Line 1` … |
| Synchronous Audio Router | — |

Multi-channel products are all offered, so with Matrix you pick which of the eight
channels to use. Detection pairs the two halves using the driver name Windows shows
in parentheses, which is identical on both sides — so `VBMatrix In 3` correctly maps
to `VBMatrix Out 3`, and it keeps working when a vendor renames the friendly parts
(VoiceMeeter's 2024 driver renamed the capture side from `VoiceMeeter Output` to
`VoiceMeeter Out B1`).

If nothing is installed, the wizard downloads VB-CABLE from VB-Audio and runs their
installer — Windows will ask for administrator permission, and a restart is needed
afterwards.

VB-CABLE is [donationware by VB-Audio](https://vb-audio.com/Services/licensing.htm).
It is not bundled here; it's fetched from their servers at your request. If you find
it useful, pay them for a licence.

**GPU acceleration.** With an NVIDIA card, ~1.3 GB of CUDA libraries makes
transcription about 20× faster and enables a more accurate model. Entirely optional
— the CPU path is genuinely usable. Downloaded from NVIDIA's official packages into
`%LOCALAPPDATA%\Voice2TTS\cuda`, removable at any time from Settings.

Both can be skipped and done later from Settings or **Setup wizard** in the tray menu.

### Point Discord at it

In **Discord → Settings → Voice & Video**:

- **Input Device**: `CABLE Output (VB-Audio Virtual Cable)`
- **Input Mode**: Voice Activity, sensitivity **manual**, slider well to the left
- **Noise Suppression**: **Off** — Krisp classifies synthesized speech as noise and
  gates it out. This is the single most common reason "Discord can't hear it".
- **Echo Cancellation**: Off
- **Automatic Gain Control**: Off

## Using it

Settings is organised by what you are doing:

| Tab | What is in it |
|---|---|
| **Normal** | The voice, and how recognition works. Everything ordinary speech needs. |
| **Translate** | The language pair, the models, and every reason a combination will not work. |
| **Studio** | Recording, training and designing a voice of your own. |
| **Add-ons** | Optional downloads: GPU acceleration, Japanese voices, Studio training. |
| **Misc** | Audio devices, triggers, the voice library, word rules, history, updates, status. |

The tray icon's colour is the current state:

| Colour | Meaning |
|---|---|
| grey | stopped |
| amber | loading models / transcribing |
| blue | ready |
| green | listening |
| red | speaking |

Right-click for Start/Stop, mode, **Speak text…**, **Settings** and **Setup wizard**.

### Modes

- **Push to talk** — hold the hotkey (default `Ctrl+Alt+V`) while speaking. Most
  predictable; nothing is transcribed unless you ask for it.
- **Automatic (VAD)** — Silero detects speech and endpoints it for you.
- **Both** — VAD runs continuously, and the hotkey still works.

The hotkey is **not** suppressed, so it still reaches whatever app has focus. Pick a
combo you don't otherwise use, especially in games.

### When it speaks

Settings → **Normal** → Recognition offers two modes. They are genuinely different jobs,
not a fast setting and a slow one, so pick by what you are doing:

| | Wait for a sentence | Speak while talking |
|---|---|---|
| Speaks | the whole utterance, after you pause | each phrase as it settles |
| Delay | however long you keep talking | ~0.5 s to have the words, plus however long the voice takes to say them |
| Intonation | natural — a whole sentence at once | flatter, a phrase at a time |
| Cost while speaking | nothing | ~half a processor core |
| Works with push-to-talk | yes | no — it needs automatic detection |

**Wait for a sentence** is the default and the right choice for most use. You
say something, you pause, the far end hears it properly phrased.

**Speak while talking** suits a long uninterrupted stretch — presenting,
explaining, reading something out — where waiting for the end would leave the
other side in silence for half a minute.

How it works: the recogniser re-reads the last few seconds about once a second,
and whatever two consecutive readings agree on is treated as settled and spoken.
Agreement is the safeguard. Whisper genuinely does change its mind — reading
four seconds of *"the tests are still failing"* it produced an obscenity, then
corrected itself a second later once more context arrived. That word was never
spoken, because the two readings never agreed on it.

Three things worth knowing before you choose it:

- **It keeps listening while it speaks**, so *Mute microphone while speaking*
  does not apply. It cannot: pausing the recording either cuts a sentence in
  half or ends the phrase early, and both mangle the words. Use headphones, or
  send only to the virtual cable, or it will hear itself.
- **It cannot speak faster than you talk.** Saying a sentence takes about as
  long as saying it did, so any delay it picks up it keeps. If it drifts
  further behind it says so and suggests raising the speech speed.

- **A GPU does not make it cheaper.** Measured at 0.51× realtime on CUDA against
  0.49× on CPU. Decoding a long transcript is a chain of small dependent steps,
  so it is waiting on latency rather than on arithmetic and there is nothing for
  a GPU to speed up.
- **If your machine cannot keep up, it says so and stops.** It first spreads the
  readings further apart; past the point where that would be slower than simply
  waiting for a pause, it switches to sentence mode and tells you. It does not
  quietly keep pretending.

With translation on, it holds each sentence until the full stop rather than
speaking part of it — a translator given half a clause produces something
fluent and wrong.

### Multiple outputs

Settings → Misc → Audio holds a list of outputs. **Outputs** sets how many — one
by default, which is all you need for Discord. Each has its own enable toggle
and gain. A two-output setup:

| Device | Gain | Why |
|---|---|---|
| `CABLE Input` | 1.0 | what Discord hears |
| `Headphones` | 0.7 | so you can hear yourself |

Each device gets its own stream, resampler and gain, because devices run on
independent clocks and often differ in sample rate.

If you monitor through **speakers** rather than headphones, leave *Mute microphone
while speaking* enabled or the app will hear its own output and re-speak it.

### Voices

Three ship with the app: `lessac-medium`, `amy-medium` and `ryan-high`. Settings →
**Voice library** browses the full [Piper
catalogue](https://huggingface.co/rhasspy/piper-voices) — 100+ voices across many
languages — and downloads them into `%APPDATA%\Voice2TTS\voices`. Bundled voices
can't be deleted; downloaded ones can.

### Translating what you say

Settings → **Translate**. Speak English, have the far end hear German. It runs on
this machine like everything else — nothing is sent anywhere.

Two ways to do it, and the tab explains which applies:

| | Downloaded models | The recogniser |
|---|---|---|
| Languages | any pair, either direction | **to English only** |
| Download | ~60 MB per direction | none |
| Quality | good | usable from `small` up |
| Cost | ~50–130 ms | ~1.1 s on CPU at `small` |

**Downloaded models** are OPUS-MT, converted to CTranslate2 and published as
release assets. Pick the pair, press Download, and it appears in the list. Pairs
nobody publishes directly are routed through English automatically, at the cost
of a second hop.

> **The models have to be published once per repository.** They are not built by
> the app release, because converting sixteen pairs takes about an hour and a
> new language does not need a new build. If the Translate tab says no models
> are published, someone with write access runs it once:
>
> ```
> git tag models-1 && git push origin models-1
> ```
>
> or starts the **Translation models** action from the Actions tab. The tag has
> to match `MODELS_TAG` in `voice2tts/translate.py`, which a test enforces.

**The recogniser** is Whisper translating as it listens — one setting and no
download, but it only ever produces English. It needs a multilingual model
(`small` or better; `base` returns *"can you be nice to me?"* for *"kannst du
mich hören?"*).

Three things have to line up, and the tab says so in plain words when they do
not:

- **The recognition model must hear the language you speak.** The models ending
  in `.en` hear English only. Settings → Normal → Recognition has the multilingual ones
  and a spoken-language picker.
- **The voice must speak the target language.** A German sentence read by an
  English voice is confident gibberish. The tab names a voice that fits and
  switches to it with one button.
- **Words rules come in two lists.** Settings → Misc → Words has a *Rules for* switch:
  *What I said* fixes what the recogniser misheard and runs **before**
  translation; *What is spoken* fixes what the voice pronounces badly and runs
  **after** it. Applying an English pronunciation list to German output would be
  nonsense, which is why they are separate.

**Japanese needs one extra download.** Japanese voices are built on a different
phonemizer, which Piper does not carry — without it a Japanese voice cannot
speak at all. Settings → **Add-ons** fetches it: about 100 MB, 330 MB on disk.
It is not bundled because that is a lot to add to every installer for one
language. Voices needing it are marked *needs add-on* in the voice library.

Models are Helsinki-NLP's work, used under CC-BY-4.0. Each download carries a
`LICENSE` file with the attribution that licence requires.

## Making your own voice

Settings → **Studio**. Two ways to get a voice nobody else has, with very
different costs.

### Design one — minutes, no training

**Studio → Design.** Some Piper voices contain many speakers rather than one:
`en_GB-vctk-medium` has 109, `en_US-libritts-high` has 904. Every point *between*
those speakers is also a voice, and the Design tab lets you go there.

The speakers are laid out on a map by similarity. Click anywhere to blend the
ones around that point, press **Listen**, and adjust. Scroll to zoom and
right-drag to pan — with 904 speakers the dots sit about 2.6 pixels apart, so
zooming is how you pick one out rather than a nearby crowd. Six controls shape the
result — Size, Warmth, Brightness, Breathiness, Dynamics and Space. Size moves
pitch and formants together, so a larger voice is deeper without being slower.

**Add to my voices** builds it into a real voice file. It then behaves like any
other voice: pick it on the Voice tab, use it in profiles, hear it in Discord.

You need a multi-speaker voice installed first. Tick **Multi-speaker only** in
the Voice library to find one — the Speakers column shows how many each has.

**Save recipe** writes a `.v2tvoice` file, a couple of hundred bytes of TOML:

```toml
schema = 1
name = "Narrator"
base_voice = "en_GB-vctk-medium"

[speakers]
p294 = 0.498
p288 = 0.224

[design]
size = 0.35
warmth = 0.3
```

It names the base voice rather than containing it, so it is small enough to
paste into a chat message and carries none of anyone's model weights. Whoever
opens it needs the same base voice, from the same place you got it.

### Train one from your own voice — hours, needs an NVIDIA GPU

**Studio → Record**, then **Train**. This fine-tunes a real model on recordings
of you.

*Setup* checks the machine and installs the training environment — PyTorch and
the Piper trainer, several GB, kept separate from the app and removable from the
same tab. The hardware check is advice, not a refusal: under-spec hardware runs
out of memory rather than breaking anything, so there is an override.

*Record* shows one sentence at a time from a phonetically balanced script and
tracks how much usable audio you have banked. Each take is checked immediately —
too quiet, clipping, background noise, cut off — because a fault found now costs
thirty seconds and a fault found later costs the whole session. Existing audio
files can be imported instead, behind a confirmation that the speaker agreed to
it.

*Train* picks a voice to start from, downloads its training checkpoint (about
850 MB, resumable, with its licence shown first, since your voice inherits it),
and runs. Expect a few hours on a good card. It can be stopped and picked up
later — progress is saved every epoch — and **Listen** plays the current
checkpoint mid-run, so a run that is going nowhere can be heard rather than
waited out. **Export voice** installs the result.

## Updates

Updates work out of the box — the repository is baked into the build, so there is
nothing to configure. The app checks on startup (at most once every 24 hours,
adjustable) and offers a one-click update when a newer release exists.

Settings → Misc → **Updates** shows the repository if you want to point a fork somewhere
else, and clearing that field disables update checking entirely. **Use default**
puts it back.

Installing an update downloads the installer, verifies its size and SHA-256 against
the published `.sha256` asset, runs it silently, and relaunches. No UAC prompt — the
install is per-user — and no SmartScreen prompt either, because a programmatically
downloaded file carries no mark-of-the-web.

**Your settings survive an update.** Config lives in `%APPDATA%\Voice2TTS`, and
downloaded voices and the GPU pack in `%LOCALAPPDATA%\Voice2TTS`; the installer only
replaces the program directory. The config loader also ignores unknown keys and
fills in missing ones, so a config written by a different version still loads.

Checking contacts `api.github.com` and nothing else. Set the interval to 0, or clear
the repository, to disable it entirely.

### Publishing a release

```powershell
.\scripts\release.ps1 -Bump 0.4.0 -Notes "What changed."
```

Rewrites `voice2tts/__init__.py`, builds, tags, pushes, and creates the GitHub
release with the installer and checksum attached. `-Draft` stages it for review
first; the updater ignores drafts. Requires `gh auth login`.

`voice2tts/__init__.py` is the single source of truth for the version — `build.ps1`
and the Inno script both read it, and the updater compares against it. Don't set a
version anywhere else.

Pushing a tag is enough on its own; the Release workflow lints, runs both test
suites, builds and publishes. It refuses tags it does not recognise, so a typo
stops the release rather than shipping under the wrong version.

### Beta releases

Tag `vX.Y.Z-beta-N` to publish a pre-release:

```bash
git tag v0.6.0-beta-1 && git push origin v0.6.0-beta-1
```

`N` starts at 1 and goes up with each beta of that version. `__version__` stays
at the release being worked towards — `0.6.0` — and the build stamps the full
`0.6.0-beta-1` in, so an installed beta reports exactly which one it is.

Betas are published as GitHub **pre-releases**. That is what keeps them away
from everyone else: the in-app updater asks for `/releases/latest`, which skips
pre-releases, so people on the stable build are never offered one. When `v0.6.0`
proper is tagged, beta testers are offered it as an upgrade — pre-releases sort
below the release they lead to.

Release notes come from the `[X.Y.Z]` CHANGELOG section if there is one, falling
back to `[Unreleased]`.

To receive betas in the app, tick **Include beta versions** under Settings →
Updates. It is off by default, and the two channels read different endpoints:
off asks for `/releases/latest`, which excludes pre-releases outright; on asks
for the full listing and picks the newest installable release from it. Turning
it off later does not roll anything back — install the latest normal release
over the top to return to stable.

## Everyday use

| | |
|---|---|
| `Ctrl+Alt+V` | Push to talk (hold) |
| `Ctrl+Alt+C` | Speak whatever is on the clipboard |
| `Ctrl+Alt+X` | Stop speaking |

**Type to speak** (tray menu) opens a small window that stays put: Enter speaks,
Shift+Enter adds a line, Up/Down recalls what you said before.

**Words** rewrites text between recognition and speech — fixes names the recogniser
mishears, expands abbreviations, and corrects words the voice says badly. Whole-word
matching by default, with a live preview.

**History** lists recent utterances so you can say one again when someone missed it.
Settings → Misc → Status also has a theme picker: `native` (the default, Windows' own
appearance) plus `light` and `dark` if you want a dark window at night.

**Review before speaking** (History tab) shows each transcript for approval first.
Slower, but nothing unreviewed reaches a call. It discards on timeout rather than
speaking.

**Profiles** save situational settings — mode, voice, speed, detection tuning — so a
meeting and a game are one dropdown apart. Devices and models stay global, since
those describe the machine rather than the situation.

## Checking it reaches Discord

Settings → Misc → Audio → **Test the Discord path** plays a tone into the virtual cable and
listens on the recording side, confirming the exact route Discord will use. If it
fails, **Find the right device** plays one tone and reports which recording device
actually receives the audio.

That second button matters for **routers**. VB-CABLE is a fixed loop in the driver
and works with nothing running. VB-Audio Matrix and VoiceMeeter are mixers: their
endpoints are ports, so audio sent to `VBMatrix In 1` only reaches `VBMatrix Out 1`
if the application is running and routing it there. Voice2TTS says so rather than
naming a device that may carry nothing.

## Tuning

| Symptom | Fix |
|---|---|
| Cuts you off mid-sentence | Trigger → raise *End-of-speech silence* |
| Waits too long after you stop | Lower *End-of-speech silence* |
| Triggers on background noise | Raise *Sensitivity threshold* |
| Misses the first word | Raise *Pre-roll kept* |
| Speaks random phrases in silence | Raise threshold; the hallucination filter in `stt.drop_phrases` catches the common ones |
| Too slow | Install the GPU pack, or Recognition → smaller model |
| Voice too fast/slow | Voice → *Speed* |

Config lives at `%APPDATA%\Voice2TTS\config.toml` and can be edited directly.
Logs are next to it in `voice2tts.log`.

## Performance

Measured on an RTX 5080 with `small.en` and `en_US-lessac-medium`:

| Stage | Time |
|---|---|
| Whisper transcribe (8.6 s of audio) | 235 ms |
| Piper time to first chunk | 60 ms |
| **Utterance → first audio out** (measured end-to-end) | ~300 ms |

Add the VAD endpoint delay (`min_silence_ms`, 600 ms by default) for the wall-clock
gap in automatic mode. Push-to-talk has no such delay.

On CPU with the bundled `base.en`, expect roughly a second more per utterance.

The very first CUDA inference on a machine costs ~6.7 s of cuDNN kernel autotune;
later launches reuse the driver's compute cache and warm up in ~0.2 s. The app runs
a warmup pass at startup either way, so this never lands on your first utterance.

## Troubleshooting

**"no usable output devices"** — every output in your config is disabled, or none
could be opened. Loading the config repairs this by enabling the system default, but
you can also tick one in Settings → Misc → Audio, or delete
`%APPDATA%\Voice2TTS\config.toml` to regenerate defaults.

**"Discord can't hear anything"** — noise suppression. Turn Krisp off. Then check in
Windows → Sound → Recording that `CABLE Output` shows a moving level bar. If that bar
moves, the problem is Discord's settings; if it doesn't, the problem is this app.

**The cable installer failed** — install it yourself from
<https://vb-audio.com/Cable/>: unzip, right-click `VBCABLE_Setup_x64.exe` → *Run as
administrator*, reboot. The wizard's Re-check button will find it.

**Whisper falls back to CPU** — Settings → **Add-ons** shows the GPU pack state.
Re-download it there if it reports missing libraries.

**"Library cublas64_12.dll is not found"** — the CUDA DLLs live in a directory
Windows doesn't search. `voice2tts/cuda.py` handles this by preloading them by
absolute path; note that `os.add_dll_directory()` and PATH changes both look like
they should work and both silently fail.

**Hotkey does nothing in a game** — some anti-cheat blocks low-level keyboard hooks.
Try a different combo, or run as administrator.

**Choppy audio** — the log reports underruns. Raise `output_blocksize` in the config
(480 → 960) to trade latency for stability.

**Reporting a problem** — Settings → Status → **Copy diagnostics** puts version,
devices, capabilities, configuration and the last 60 log lines on your clipboard,
with your Windows user name redacted. Paste that into the issue. Note the log
records transcribed text at INFO level, so skim it before posting publicly.

**Voice2TTS says it's already running** — only one copy can run at a time, since two
would both claim the hotkey and the microphone. The existing window is brought to
the front instead; look for the tray icon.

## Developing

```powershell
.\setup.ps1          # venv, dependencies, models
.\run.ps1            # tray app
.\run.ps1 -Cli       # headless, console logging
.\run.ps1 -Check     # verify models, CUDA, virtual cable
.\run.ps1 -Devices   # list audio devices
```

The virtualenv is created at `%USERPROFILE%\.venvs\voice2tts`, deliberately outside
the project — it usually lives in OneDrive, and syncing a multi-gigabyte venv causes
file locks and long stalls.

### Building the installer

```powershell
.\build.ps1
```

Fetches bundle assets, generates the icon, runs PyInstaller, smoke-tests the frozen
build, then compiles the installer with [Inno Setup](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`). Output lands in `dist\installer\`.

`-SkipInstaller` stops after PyInstaller. `-BundleCuda` produces a fat offline build
with CUDA included (~1.9 GB) instead of downloading it on demand.

### Tests

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\selftest.py
```

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\guitest.py
```

`selftest.py` covers config, hotkey parsing, VAD segmentation, transcription, output
streams, the packaging modules, and a full pipeline run. It opens real output devices
but writes silence, so it's safe with headphones on. `--no-network` skips the live
catalogue checks; `--no-e2e` skips the pipeline run.

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\bare_machine.py
```

Runs both suites against a machine with no GPU and no audio devices, which is
what CI has. Three separate CI failures have come from tests that quietly
assert the *developer's* hardware — a microphone, an NVIDIA card, more devices
than WASAPI exposes. They all pass locally and fail on every runner, and none
of them shows up until a push. Run this before touching anything that reads
hardware. `--keep-outputs` simulates the other awkward case, a desktop with
speakers and no microphone.

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\fresh_machine.py
```

Runs the interface tests with every optional download absent — no extra voices,
no translation models, neither pack — and a small screen. The same class of
problem as above, one layer up: a developer machine accumulates all of those,
and a test that quietly depends on one is green here and red on every runner.
Two of them shipped this way, including a window-size check that read the wrong
measurement on a window nobody had mapped. Run this before touching the
settings window.

## Project layout

    voice2tts/
      __main__.py     entry point and CLI
      app.py          tray shell (Tk main thread + pystray worker thread)
      wizard.py       first-run setup flow
      gui.py          settings window
      pipeline.py     orchestrator; capture → VAD/PTT → STT → TTS → outputs
      capture.py      mic capture, resampled to 16 kHz mono
      vad.py          Silero VAD + endpointing state machine
      stt.py          faster-whisper wrapper
      tts.py          Piper wrapper
      streaming.py    recognising while someone is still speaking
      translate.py    OPUS-MT models: catalogue, install, translation
      plan.py         what the app is about to do, and what is wrong with it
      output.py       multi-device fan-out
      hotkey.py       global hotkey with press/release
      cable.py        virtual cable detection and assisted install
      voices.py       Piper catalogue and downloader
      gpupack.py      on-demand CUDA download
      jppack.py       on-demand Japanese phonemizer
      studioui.py     Voice Studio panels: record, train, design
      studiopack.py   training environment install and hardware gate
      recorder.py     clip capture at the microphone's own rate
      prompts.py      the recording script and time estimates
      dataset.py      clip quality checks and Piper dataset layout
      training.py     runs piper.train, exports, auditions
      checkpoints.py  base checkpoints to fine-tune from
      designer.py     speaker blending, projection, baking
      dsp.py          the designer's effects chain
      v2tvoice.py     the .v2tvoice recipe format
      net.py          one User-Agent, resumable downloads, checksums
      updater.py      GitHub release checks and one-click install
      cuda.py         Windows CUDA DLL preloading
      config.py       TOML-backed settings
      devices.py      device discovery by name
      paths.py        install / config / cache locations
    installer/        Inno Setup script and assets
    scripts/          model fetch, icon generation, tests
    spike/            standalone experiments and measured findings

## Licence

Voice2TTS is **GPL-3.0-or-later**. It bundles Piper (`piper-tts`), which is GPL
because it links eSpeak NG, so the combined work must be GPL too. Releases before
0.4.0 incorrectly declared MIT.

Full text in [COPYING](COPYING); the third-party component list, including the
things downloaded rather than bundled, is in [LICENSE.txt](LICENSE.txt).

See also [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
[CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).
