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
any device you already have (VB-CABLE, VoiceMeeter, Virtual Audio Cable, Hi-Fi
Cable, Synchronous Audio Router) and skips this step if so. Otherwise it downloads
VB-CABLE from VB-Audio and runs their installer — Windows will ask for administrator
permission, and a restart is needed afterwards.

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

### Multiple outputs

Settings → Audio holds a list of outputs. Add as many as you like; each has its own
enable toggle and gain. Typical setup:

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

## Updates

Updates work out of the box — the repository is baked into the build, so there is
nothing to configure. The app checks on startup (at most once every 24 hours,
adjustable) and offers a one-click update when a newer release exists.

Settings → **Updates** shows the repository if you want to point a fork somewhere
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
you can also tick one in Settings → Audio, or delete
`%APPDATA%\Voice2TTS\config.toml` to regenerate defaults.

**"Discord can't hear anything"** — noise suppression. Turn Krisp off. Then check in
Windows → Sound → Recording that `CABLE Output` shows a moving level bar. If that bar
moves, the problem is Discord's settings; if it doesn't, the problem is this app.

**The cable installer failed** — install it yourself from
<https://vb-audio.com/Cable/>: unzip, right-click `VBCABLE_Setup_x64.exe` → *Run as
administrator*, reboot. The wizard's Re-check button will find it.

**Whisper falls back to CPU** — Settings → Recognition shows the GPU pack state.
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
      output.py       multi-device fan-out
      hotkey.py       global hotkey with press/release
      cable.py        virtual cable detection and assisted install
      voices.py       Piper catalogue and downloader
      gpupack.py      on-demand CUDA download
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

See also [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and
[CHANGELOG.md](CHANGELOG.md).
