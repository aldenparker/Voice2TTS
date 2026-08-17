# Changelog

All notable changes to Voice2TTS. Format follows [Keep a Changelog](https://keepachangelog.com/),
versioning follows [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-08-16

### Fixed — virtual device detection

- **VB-Audio Matrix was not detected at all**, while a second, inconsistent detector
  matched it by substring. The wizard reported "no virtual microphone found" at the
  same moment the default config wired up `VBMatrix In 8`. There is now one
  implementation; `devices.find_cable_output()` delegates to it.
- **The Discord-side device was reported wrongly for multi-channel products.**
  Pairing assumed an `Input` → `Output` rename, so `VBMatrix In 8` resolved to
  `VBMatrix Out 2` — whichever endpoint enumerated first. Pairing is now done on the
  driver name Windows shows in parentheses, which is identical on both halves of a
  cable, plus the channel number. Verified against all eight Matrix channels.
  This also fixes VoiceMeeter's 2024 driver, where the capture side was renamed from
  `VoiceMeeter Output` to `VoiceMeeter Out B1` and the old rename no longer worked.
- Every device was matched across all four host APIs, so the "exactly one match"
  pairing test never passed and correct pairings were all flagged uncertain.
  Detection is now WASAPI-only on both sides — MME also truncates names to 31
  characters, destroying the driver tag.
- Changing the selected channel appended a duplicate output row instead of updating
  the existing one, because the cable row was found by looking for "cable" in the
  name and a Matrix channel is called `VBMatrix In 1`.

### Added

- Support for the full VB-Audio range: Matrix (8 channels), CABLE A+B and C+D,
  Hi-Fi Cable, VoiceMeeter/Banana/Potato, plus a generic fallback so an unrecognised
  VB-Audio product is still usable.
- A channel picker in the setup wizard when more than one virtual device exists.
- `--devices` and Settings now name the exact recording device to select in Discord,
  and say so explicitly when the pairing had to be inferred rather than confirmed.
- Render-only virtual devices (NVIDIA's virtual audio, game-streaming sinks) are
  rejected: without a capture endpoint they would silently swallow speech.

## [0.4.0] - 2026-08-16

### Changed — licence

- **Voice2TTS is now GPL-3.0-or-later.** Earlier releases incorrectly declared MIT.
  Piper (`piper-tts`), which is bundled, is GPL-3.0-or-later because it links
  eSpeak NG, so the combined work must be GPL. `LICENSE.txt` has been corrected and
  the full licence text added as `COPYING`.

### Fixed

- **GPU acceleration never worked in the installed build.** `gpupack` writes the
  CUDA libraries to `cuda/nvidia/<pkg>/bin`, but the loader globbed `<root>/*/bin`
  — one level too shallow — so the downloaded pack was never found. The bug was
  invisible during development because pip-installed `nvidia-*` wheels sit at
  exactly the depth the glob expected, masking it in a virtualenv. Found by running
  the frozen build's own `--check`.
- Error messages never appeared in five failure paths (voice download, catalogue
  fetch, GPU install, update check, cable removal). Each did
  `except Exception as exc:` then referenced `exc` inside a lambda run later by Tk;
  Python deletes the name when the block ends, so those raised `NameError` instead
  of showing the problem.
- Automatic mode could stop working silently: an exception inside the Silero VAD
  killed the segmenter thread while the tray icon still showed "ready" and nothing
  reached the log. Errors are now caught, reported, and retried, falling back to
  push-to-talk after repeated failures.
- Unhandled exceptions vanished entirely in the windowed build, which has no
  stderr. `sys.excepthook`, `threading.excepthook` and Tk's callback hook now route
  to the log file.
- A disconnected microphone (unplugging a USB mic) left the app running and deaf
  with no indication. Capture failures are now detected and surfaced.
- Launching Voice2TTS twice produced two tray icons, two hotkey listeners both
  firing on one keypress, and two claims on the microphone. A second launch now
  surfaces the existing window instead.
- Update checks against a private repository reported "no releases found", which is
  indistinguishable from the real cause. The message now explains both, and rate
  limiting is reported separately.
- `build.ps1` stopped any process named `Voice2TTS*`, which would kill an installed
  copy the user was running. It is now scoped to processes started from `dist\`.

### Added

- Per-monitor DPI awareness, so the UI is not bitmap-stretched on scaled displays.
- **Copy diagnostics** button: version, devices, capabilities, config and the log
  tail, with the Windows user name redacted.
- In-app **Start with Windows** toggle (previously installer-only).
- Virtual cable **removal** from Settings → Audio.
- Warning when a non-English voice is paired with an English-only Whisper model —
  previously this silently produced confident nonsense.
- `schema_version` in the config, giving future versions a migration hook.
- GitHub Actions CI running lint, the self-test suite and a packaging build on
  every push, plus a release workflow that builds and publishes the installer when
  a `v*` tag is pushed. The release job refuses to run if the tag and
  `voice2tts/__init__.py` disagree, since the updater compares against the latter.
- The self-test synthesizes its own speech sample instead of relying on a leftover
  from `spike/02_tts.py`, which is gitignored and so absent on a clean checkout —
  the VAD and STT checks failed on the first CI run for exactly that reason.
- The update repository is now prefilled with this build's own repository, so a
  normal install updates itself without anyone having to find and type a repo name.
  Clearing the field still disables checking, and forks override it by changing
  `DEFAULT_UPDATE_REPO` in `voice2tts/__init__.py`. Config schema bumped to 2: a
  schema-1 file with a blank repository adopts the default, while one the user
  chose — or deliberately cleared at schema 2 — is left alone.
- `pyproject.toml`, ruff configuration, `.editorconfig`, `CONTRIBUTING.md`,
  `SECURITY.md`.

## [0.3.0] - 2026-08-16

### Added

- Update system: checks GitHub Releases on start (throttled, opt-out), verifies the
  download by size and SHA-256, installs silently and relaunches.
- `scripts/release.ps1` to build, tag, push and publish a release.
- `.gitignore` covering build output and the ~380 MB of models.

### Fixed

- Version was declared in two places and disagreed (`__init__.py` said 0.1.0 while
  the installer said 0.2.0). `voice2tts/__init__.py` is now the single source.
- Update repository validation accepted anything containing a slash, so a pasted
  URL reached the API and produced a confusing network error.

## [0.2.0] - 2026-08-16

### Added

- Windows installer (Inno Setup), per-user, with Start Menu entries and uninstaller.
- First-run wizard: virtual cable install, GPU acceleration, devices, and a test
  phrase that actually plays.
- Assisted VB-CABLE installation — detects 12 virtual-audio products before
  suggesting anything, then downloads from VB-Audio and runs their installer.
- On-demand GPU acceleration (~1.3 GB) downloaded from NVIDIA's packages on PyPI.
- Three bundled voices plus a library browsing the full 174-voice Piper catalogue.

### Fixed

- Frozen builds would not start: freezing `voice2tts/__main__.py` made it top-level
  `__main__`, breaking its relative imports. Added `launcher.py`.
- A crashed windowed build hung forever on a modal traceback dialog, holding its
  files and breaking the next build.
- `--cli` from the Start Menu printed nothing, because a windowed bootloader has no
  stdout. A separate console executable now ships alongside.
- The GPU pack did not take effect until a full restart, because the CUDA probe
  result was cached from startup.

## [0.1.0] - 2026-08-16

Initial working pipeline: microphone → Silero VAD → Whisper → Piper → multiple
output devices, with push-to-talk and automatic modes and a tray UI.

### Fixed during development

- Silero VAD returned near-zero probabilities for obviously loud speech: v5 expects
  the previous window's last 64 samples prepended (576, not 512), and the ONNX input
  shape is dynamic so a bare 512 raises no error.
- The underrun counter fired once per utterance on the normal drain tail, making it
  useless as a diagnostic.
- A fresh config had every output disabled, so the app loaded, listened,
  transcribed, synthesized — and had nowhere to send audio.
