# Changelog

All notable changes to Voice2TTS. Format follows [Keep a Changelog](https://keepachangelog.com/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Two recognition modes**, chosen in Settings → Recognition, because they
  serve different purposes rather than being a fast setting and a slow one:

  - **Wait for a sentence** — the default and what the app has always done.
    Waits for a pause, then speaks the whole utterance with its natural
    intonation. Costs nothing while you are quiet.
  - **Speak while talking** — recognises as you go and speaks each phrase once
    it settles, so a long stretch does not leave the other side in silence.
    Typically runs about 2 s behind you, and costs roughly half a processor core
    for as long as anyone is speaking.

  A phrase is only spoken once two consecutive readings agree on it. That
  matters: reading four seconds of "the tests are still failing", Whisper
  produced an obscenity and corrected itself a second later. The word was never
  spoken, because the two readings never agreed.

  Measured, and against the obvious assumption: **a GPU does not make streaming
  cheaper** (0.51x realtime on CUDA against 0.49x on CPU). If a machine cannot
  keep up, the readings are spread further apart first, and past the point where
  that beats simply waiting, it switches to sentence mode and says so.

  Streaming keeps listening while it speaks, so *Mute microphone while speaking*
  does not apply to it — the interface says so when both are set. And because
  speaking a sentence takes about as long as saying it did, it warns rather than
  drifting silently further behind.

- **Live translation.** Speak one language, have the far end hear another,
  entirely on this machine. Settings → **Translate** picks the pair, downloads
  the models and says up front why a combination will not work.

  Models are Helsinki-NLP's OPUS-MT, converted to CTranslate2 and published as
  release assets. Converting them ourselves rather than using a pre-converted
  third-party set settles the licence question — those state terms for their
  software and nothing about the weights — and turned out smaller and faster
  anyway: **63 MB against 151, and 19 ms against 37** on a short sentence.
  Each download carries the CC-BY-4.0 attribution the licence requires.

- **Whisper's own translation**, as a second method: one setting, no download,
  English only. Measured on synthesized German — `base` is not good enough
  (*"can you be nice to me?"* for *"kannst du mich hören?"*), `small` is, and
  `medium` costs three times as much for nothing. The tab says so.

- **Multilingual recognition.** The Whisper models without `.en` are now
  offered, with a spoken-language picker and `auto` detection — what
  translating *from* anything but English needs.

- **Two pronunciation lists.** Settings → Words has a *Rules for* switch.
  Source rules fix what the recogniser misheard and run before translation;
  target rules fix what the voice says badly and run after it.

- The Translate tab names a voice that speaks the target language and switches
  to it with one button, rather than only warning about the mismatch.

### Removed

- **The soft endpoint that split continuous speech into segments** (0.6.1). In
  use it cut mid-thought and the app talked over itself. Waiting for a real
  pause is back, and is now one of the two modes above -- which is what that
  change was reaching for and the wrong way to get it.

### Fixed

- **The language pickers only offered languages you already had models for**, so
  you could not choose German until you had German and could not get German
  without choosing it. They now list every language, with names.
- **Trying the recogniser method overwrote the saved target language.** An
  "English to German" pair became "English to English", which validation treats
  as a no-op and switches translation off — so ticking the box afterwards
  appeared to do nothing. The method no longer touches the language pair, and a
  pair that would do nothing now says so instead of silently unticking.
- **With no model for the pair, the tab still offered to switch to a voice for
  the target language.** The text stays in the source language when there is
  nothing to translate it, so that produced English read in a German accent. The
  voice now follows what will actually be spoken.
- **Translation being unavailable was a quiet warning**; it is an error now,
  because the far end hears a language nobody asked for.
- **The models release can be published by pushing a `models-*` tag**, not only
  from the Actions tab. The first build shipped a download button with nothing
  behind it because the workflow had simply never been run.
- Target substitution rules were written to the config but loaded back as plain
  dictionaries, so a saved rule would have broken the substituter.
- Four modules announced themselves to servers as `Voice2TTS/0.2` long after
  0.2 shipped. There is one User-Agent now, and it reports the real version.
- A model whose tokenizer pair was half-downloaded was treated as usable, and
  would have decoded output with the source tokenizer — producing text with raw
  tokenizer marks in it rather than an error.


## [0.6.2]

### Fixed

- **The app could fall silent and sit on "thinking" until restarted.** 0.6.1
  made this common by splitting speech into several segments instead of one, so
  the state changed roughly five times as often.

  The cause was that the pipeline told the tray and the settings window about
  every change *from the thread doing the work*. Those observers marshal onto
  the interface thread with `root.after()`, and a cross-thread Tk call
  serialises on the Tcl interpreter lock — so a busy interface blocked the
  worker inside the "thinking" transition, before it ever reached synthesis.
  Nothing downstream ran, so the app went quiet, and only a restart cleared it.

  Observers now run on their own thread. A stalled interface can fall behind;
  it can no longer stop audio. Verified by wedging an observer permanently: the
  old code hung outright at the first state change, the new one carries on.

- **Speech captured just before playback started was thrown away.** While the
  app speaks it stops listening, so it does not transcribe its own voice — but
  it did that by resetting the segmenter, which also discarded audio already
  captured, including the tail that a mid-speech split deliberately carries
  forward. It now hands that audio on to be spoken instead.

- **Utterances are no longer dropped when the pipeline falls behind.** The queue
  was capped at eight. Speaking sixty seconds of speech takes sixty seconds, so
  anyone talking without pauses is always ahead of playback; the cap silently
  lost words that had actually been said.

- **Changing a detection setting mid-session** no longer discards what is being
  captured or races with the thread reading it.

### Added

- **A stall watchdog.** If the pipeline sits in a working state for a minute it
  logs a stack dump of every thread and says so in the interface, so "it got
  stuck" becomes "it is stuck here".


## [0.6.1]

### Fixed

- **Speaking quickly meant nothing was said until you stopped.** The endpoint
  rule waited for 600 ms of silence, and fast speech never leaves that much: on
  34 seconds of continuous speech the first audio came out at 30 seconds — the
  hard cap, which cut mid-word. It now comes out at 5.

  Lowering "End-of-speech silence" could not have fixed this. On that audio the
  speech probability sits at 1.0, only 7% of windows fall below the threshold at
  all, and *no* gap reaches 600 ms; even at 300 ms there are zero usable cut
  points. The pauses in fast speech are 100–250 ms.

  So after a few seconds the requirement is eased down, and if no pause arrives
  the segment is cut at the quietest moment of the last second — between two
  words rather than mid-vowel — with the remainder carried forward as the start
  of the next segment. Nothing is lost: 33.7 of 33.9 seconds survived the test.

  Short utterances followed by a real pause behave exactly as before. Two new
  sliders in Settings → Detection tuning control it, and setting "Start
  splitting after" to 0 restores the old behaviour entirely.


### Added — Voice Studio (0.6.0, in progress)

- **Record your own voice.** A Studio → Record tab shows one sentence at a time
  from the CMU ARCTIC prompt list, records it, checks it and keeps a running
  total against a target. Prompts are shuffled, so stopping early still leaves
  broad phonetic coverage; the remaining estimate is based on how fast you
  actually read rather than a fixed sentence count.
- **Clip quality checks at recording time.** Too quiet, clipping, too short,
  noisy, mostly silence, and "shorter than the prompt — was it cut off?". Noise
  is judged as signal-to-noise ratio rather than an absolute level, so a quiet
  clean take passes and a loud hissy one does not.
- **Importing existing recordings**, through the same checks, behind a
  confirmation that the speaker consented.
- **Training** — `piper.train` in the studio pack's own interpreter, with live
  progress, a stop button, and resume from `last.ckpt` after a crash or reboot.
- **Base checkpoints** downloaded from `rhasspy/piper-checkpoints` (~850 MB,
  resumable). The model card's licence is shown and agreed to first, since a
  trained voice inherits it.
- **Auditioning mid-run**, so a run that is going nowhere can be heard rather
  than waited out.
- **Export** to `.onnx` + `.onnx.json` in the user voices directory, with a
  provenance sidecar recording the base checkpoint, dataset size and epochs.

### Added — Voice Designer (0.6.0, in progress)

- **Design a voice without training or recording.** A Studio → Design tab lays
  the speakers of a multi-speaker voice out on a map by similarity; click
  anywhere to blend the speakers around that point, listen, and adjust.
  `en_GB-vctk-medium` has 109 speakers, `en_US-libritts-high` has 904.
- **A zoomable map.** Scroll to zoom, right-drag to pan. `en_US-libritts-high`
  puts 904 speakers in a small square, roughly 2.6 pixels apart; at 12x they are
  21 pixels apart and individually selectable.
- **Six shape controls** — Size, Warmth, Brightness, Breathiness, Dynamics and
  Space — over a fixed effects chain. Size shifts pitch and formants together
  and puts the timing back, so a larger voice is deeper without being slower.
- **Designed voices are ordinary voices.** The blend is baked into the model, so
  the result appears in the voice picker and works with profiles, previews and
  everything else with no special handling.
- **`.v2tvoice` recipes** — a few hundred bytes of TOML naming the base voice and
  the blend. Shareable and diffable, and it distributes a pointer rather than
  anybody's weights.

Neither tier has been run end to end on real hardware — see ROADMAP.md.

## [0.5.2] - 2026-08-16

### Fixed

- **No dropdown in the application would open in 0.5.1.** Resetting the combobox
  list colours used `option_add(..., "")`, which stores an empty string rather than
  unsetting the option; Tk then could not parse it as a colour and the list never
  appeared. Every picker was affected — microphone, outputs, voice, profile, theme,
  recognition model, cable channel. Native mode now writes the real Windows system
  colours (`SystemWindow` and friends), so the list also follows a high-contrast or
  custom system theme.

  Only 0.5.1 was affected: earlier releases defaulted to a repainted theme, which
  always wrote real colours.

  Nothing caught this because constructing a combobox never builds its dropdown —
  the list is created when the arrow is clicked. The suite now opens one for real
  in every theme mode.

## [0.5.1] - 2026-08-16

### Fixed — release gating

- **A tag could publish a release over failing tests, and did.** v0.5.0 shipped
  from a commit whose CI run had failed: `ci.yml` skips tags, and `release.yml`
  ran no checks of its own, so nothing stood between a tag push and a published
  installer. A comment in `ci.yml` claimed release handled it; that was never
  true. `release.yml` now has a `verify` job — tag/version check, lint, self-test —
  that the publishing job depends on, and a test asserts the dependency so it
  cannot quietly disappear again.
- **CI had been failing on every push since the first commit.** `test_device_lists`
  asserted the trimmed device list was shorter than the raw one, which is false on
  a machine with no audio hardware — as every CI runner is. It now checks that
  enumeration returns empty cleanly instead.
- The GUI smoke test now runs in CI, having previously not run there at all. It is
  non-blocking until Tk on a headless runner is proven reliable.

### Changed

- **The interface uses Windows' native widget appearance again.** 0.5.0 repainted
  it, which suited a functional utility less well than the platform's own look.
  Light and dark remain available in Settings → Status for anyone who wants a dark
  window, but `native` is the default and nothing is restyled in that mode.
  Existing configs on the old default migrate across; an explicit light or dark
  choice is left alone.

## [0.5.0] - 2026-08-16

### Added — quality of life

- **Device pickers now list real endpoints only.** They showed 54 inputs and 54
  outputs on a machine with 12, because PortAudio exposes every device through four
  host APIs. A "Show every host API" checkbox restores the old view. Devices that
  share a name (two identical monitors) get a `#2` ordinal, without which the
  second was unselectable.
- **Verify the path to Discord** without opening Discord: plays a tone into the
  cable and measures it on the recording side. "Find the right device" plays one
  tone and reports which recording device actually receives it — the only reliable
  answer for router products. Plus live output level meters.
- **Speak clipboard** and **stop speaking** hotkeys. All hotkeys now share one
  keyboard hook rather than one per binding, and conflicting combinations are
  reported instead of silently leaving the second dead.
- **Pronunciation dictionary** applied between recognition and speech, with
  whole-word matching, optional regex, a live preview, and a set of common
  abbreviations.
- **History tab** of recent utterances with say-again and copy, kept in memory only.
- **Review before speaking** (optional): check and edit the transcript first. Times
  out to discard rather than speak.
- **Automatic microphone recovery** — unplugging a USB mic no longer needs a
  restart.
- **Profiles**: named snapshots of situational settings, with optional per-app
  switching. Devices and models stay global.
- **Voice preview** before downloading, **a persistent type-to-speak window** with
  recent-entry recall, optional **light/dark theming** (reverted to native by
  default in 0.5.1), an **in-app log viewer**, and **winget manifests**.

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
