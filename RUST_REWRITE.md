# Voice2TTS in Rust — a thought experiment

Status: **not a plan of record.** This is an honest working-through of what a
Rust rewrite would involve, written while 0.7.0-beta-9 was going out. It exists
to make the decision arguable rather than vibes-based. The verdict is at the
bottom and it is not a straightforward yes.

---

## 1. What Python actually cost

Not "Python is slow" — it isn't, the hot path is C++ anyway. The cost was
specific, and this release measured it. Every one of these shipped:

| The bug | Why the language allowed it |
|---|---|
| `stt.task="translate"` and `translation.enabled=true` both set — Whisper translated to English, then a model chain translated that again | Two fields that must not both be true. Nothing but a comment said so. |
| `plan.TRANSLATE == "translate"` and `stt.task == "translate"`, same literal, **opposite meanings** | Modes were strings. Strings compare equal across concepts. |
| A Japanese voice reported as "not an English voice" while translating *into* Japanese | Five modules each re-derived the languages; the checker could not see they disagreed. |
| Repeated VAD failure announced "switching to push-to-talk" and never bound the key. The app went deaf. | An assignment where a method call was needed. Nothing typed the difference. |
| A failing review hook **spoke the unreviewed text**; the timeout path discarded it | Two paths, one feature, opposite outcomes. No total function to be exhaustive over. |
| `review_timeout_s = 0` meant "wait forever" | An unconstrained `float` in a field whose docstring promised otherwise. |
| The Japanese pack shipped without `pydantic` | Runtime dependency resolution, on a machine that already had it. |
| `status()` imported the phonemizer, locking its `.pyd` files, so the pack could never be repaired | A query with an invisible side effect at a distance. |
| ~40 places that failed and said nothing | `except: pass` and `return ""` are one keystroke each. |

The fixes were good — `StrEnum` + `assert_never` + mypy, `plan.build()` as sole
owner, `Repair` returned rather than logged, `probe.rs`-style proving. But every
one of them is **a convention held in place by a linter**, retrofitted after the
bug shipped. In Rust most of that table is a compile error, and the rest is a
type you cannot construct wrongly.

That is the whole argument. Everything below is about whether the argument
survives contact with the dependency stack.

---

## 2. What Rust changes — and what it does not

**Genuinely fixed by the language:**

- Sum types with exhaustive `match`. Adding a mode breaks the build at every
  site that ignores it. No `assert_never`, no lint, no CI step to forget.
- `Result<T, E>` with `#[must_use]`. `except: pass` has no equivalent — you
  write `?`, `.expect()`, or an explicit `match`. The forty-silent-failures
  class largely evaporates.
- Newtypes. `Heard(LangCode)` and `Spoken(LangCode)` cannot be swapped, which
  is the exact confusion behind the Japanese-voice bug.
- `Send`/`Sync` and ownership. The Tk thread-affinity rules that `_later()`
  exists to enforce become compiler-checked. The `_streaming_started` snapshot
  vs live `streaming_mode` race is a borrow error.
- Typestate. `Pipeline<Stopped>` has no `submit()` method. The illegal state
  transitions the `_ALLOWED` table now logs at runtime stop compiling.
- Static linking. **The pydantic bug cannot happen** — there is no runtime
  dependency resolution to get wrong.

**Not fixed by the language, and worth being clear-eyed about:**

- *Testing against your own machine.* Rust will not stop you writing a test that
  depends on a model you happen to have downloaded. `nothing_installed.py` and
  `isolated_pack.py` exist because that bit me three times this release; their
  equivalents are still needed. (Cargo does help: a test binary has no ambient
  site-packages, so the specific pydantic shape is much harder to hit.)
- *Windows file locking.* Loaded DLLs are still unreplaceable. Mostly moot —
  static linking means almost nothing is dynamically loaded — but CUDA and
  ONNX Runtime still are.
- *Audio device flakiness.* Unplugged speakers, PortAudio-equivalent silence,
  device enumeration races. Same problems, different API.
- *Model quality.* Identical models, identical output. `base` still mistranslates.
- *Domain mistakes.* Nothing stops `plan()` from encoding the wrong rule; it only
  stops the rule from being encoded in five places that disagree.

---

## 3. The dependency map

The instruction was "same dependencies". Mostly achievable — the heavy lifting
is C++ in both worlds, and Rust binds it directly instead of through CPython.

| Today | Rust | Same artefact? | Confidence |
|---|---|---|---|
| faster-whisper (CTranslate2) | [`ct2rs`](https://crates.io/crates/ct2rs), `whisper` feature | **Yes** — same CTranslate2, same converted models | Medium-high; binding is real and covers Whisper + Marian |
| OPUS-MT via CTranslate2 | `ct2rs` `Translator` | **Yes** — Marian-MT is explicitly supported | Medium-high |
| sentencepiece | `ct2rs` bundled tokenizers, or `sentencepiece` crate | Yes | High |
| Piper (VITS + ONNX) | [`piper-rs`](https://crates.io/crates/piper-rs) or `ort` directly | **Yes** — same `.onnx` + `.onnx.json` voices | Medium; several competing crates, may end up owning this |
| onnxruntime | [`ort`](https://crates.io/crates/ort) | Yes | High — mature, well maintained |
| Silero VAD | `ort` + the same `silero_vad.onnx` | **Yes** — identical model | High |
| espeak-ng (via Piper) | `espeak-rs` (FFI) | Yes | Medium |
| **pyopenjtalk + sudachi (340 MB pack)** | [`jpreprocess`](https://crates.io/crates/jpreprocess) + `jpreprocess-naist-jdic` | **No — a rewrite of OpenJTalk in Rust, dictionary bundled into the binary** | Medium-high |
| sounddevice / PortAudio | [`cpal`](https://crates.io/crates/cpal) — WASAPI, ALSA, PipeWire | Substituted | High |
| soxr | `rubato` | Substituted | High |
| pynput (hotkeys) | `global-hotkey`, plus a portal path on Wayland | Substituted | **Low on Linux** — see §5 |
| pystray | [`tray-icon`](https://lib.rs/crates/tray-icon) | Substituted | Medium-high |
| Tkinter | Slint (see §4.4) | Substituted | Medium |
| PyTorch (Voice Studio training) | **stays Python, out of process** | Unchanged | High |

Two entries deserve emphasis.

**`jpreprocess` deletes an entire subsystem.** It is a Rust reimplementation of
OpenJTalk's text processor, and `jpreprocess-naist-jdic` bundles the dictionary
into the executable. That means: no 340 MB download, no wheel resolution, no
transitive `pydantic`, no `.pyd` locking, no Add-ons row, no `probe.rs` for
Japanese, no "restart to finish installing". `jppack.py`, its 200 lines of
hand-rolled resolver, and both scripts written to test it — all gone. The single
worst-behaved component in the codebase stops existing. That is the strongest
single argument in this document.

**`ct2rs` is the load-bearing risk.** Everything about recognition and
translation goes through one third-party binding to a C++ library that needs
CMake, and optionally CUDA. If it does not build cleanly for
`x86_64-pc-windows-msvc` *and* `x86_64-unknown-linux-gnu`, with and without
CUDA, the project stalls on day one. **Spike this before anything else** (§7).

---

## 4. Architecture

### 4.1 Crate layout

Workspace, with the dependency arrows pointing one way only. The layering is
what makes "only one place decides" structural rather than aspirational.

```
voice2tts/
  core/          no I/O, no threads, no ML. Types, modes, the plan, config
                 parsing, substitution rules. 100% unit-testable, instant.
  audio/         cpal capture and playback, resampling, device enumeration.
  models/        ct2rs + ort wrappers. Whisper, Piper, Marian, Silero.
  engine/        the pipeline: threads, channels, the state machine.
  platform/      the two-implementation seam. Hotkeys, tray, autostart,
                 virtual-device setup, paths. One trait per capability,
                 windows.rs and linux.rs behind it.
  ui/            Slint. Depends on core + engine, never on models directly.
  app/           the binary. Wiring, logging, CLI, single-instance.
  studio/        subprocess control for the Python trainer. Unchanged in kind.
```

`core` must not depend on `models` or `audio`. That constraint alone would have
prevented `plan.py` growing an import of `translate`, which is how the plan
became something you needed models installed to test.

### 4.2 Parse, don't validate

The single biggest structural change, and it retires `Config.validate()`
entirely.

```rust
/// Exactly what is in the file. Serde-shaped, permissive, never used directly.
#[derive(Deserialize)]
struct RawConfig { /* Option<String>, Option<f32>, unknown fields tolerated */ }

/// Checked once, at the boundary. Every field is already legal.
pub struct Config {
    pub trigger: TriggerMode,
    pub translation: Translation,   // see 4.3
    pub review_timeout: ReviewTimeout,  // newtype, 1..=600s, no zero
    /* ... */
}

impl RawConfig {
    /// The only way to get a Config. Repairs are RETURNED, never logged and
    /// forgotten -- the lesson from `Repair` in 0.7.0, made structural.
    fn parse(self) -> (Config, Vec<Repair>) { /* ... */ }
}
```

Downstream code takes `&Config` and never re-checks anything, because there is
no path by which an unchecked one exists. `review_timeout_s = 0` is not a bug to
find; `ReviewTimeout::new(0)` returns `Err`.

### 4.3 Illegal states, unrepresentable

0.7.0 collapsed two config fields into `TranslationMode` because they could
contradict each other. Rust takes it further — the *data each mode needs* lives
inside the variant, so an irrelevant field cannot even be read:

```rust
pub enum Translation {
    Off,
    /// A downloaded chain. The route is resolved at load, so `Models` cannot
    /// exist without one -- the "translation is on but no model" state is gone.
    Models { route: Route, target: Language },
    /// Whisper's own task. English only, so there is no target field to ignore.
    Recogniser { source: Language },
}

impl Translation {
    pub fn whisper_task(&self) -> WhisperTask {
        match self {                       // exhaustive; no default arm
            Translation::Recogniser { .. } => WhisperTask::Translate,
            Translation::Off | Translation::Models { .. } => WhisperTask::Transcribe,
        }
    }
}
```

Compare with today: `Recogniser` still carries a `target` that `plan.build()`
must remember to warn is ignored. Here there is nothing to warn about.

### 4.4 The GUI decision

This is the one place where "correctness by construction" and "works for the
users" pull in different directions, and the tie-breaker is Japanese.

| | Architecture | Native look | IME | Accessibility |
|---|---|---|---|---|
| **Iced** | Best in class — Elm messages map exactly onto our state machine | No | **Reported missing/aspirational** | Aspirational |
| **egui** | Immediate mode; state lives wherever you put it | No | **Known problems** | AccessKit, good |
| **Slint** | Declarative DSL, decent separation | **Targets native** | **Handles it** | Windows Narrator supported |

Iced is the intellectually right answer: its `Message` enum and `update()`
function *are* the discipline this release spent a week retrofitting, and an
unhandled message is a compile error. I wanted to recommend it.

**Recommend Slint anyway.** The flagship 0.7.0 feature is translation, including
Japanese, and there is a "type to speak" box and a review-before-speaking
dialog. A toolkit that mishandles IME composition cannot take Japanese input at
all — the app would be unable to accept text in the language it exists to
produce. That is disqualifying, and no amount of architectural elegance
compensates. Slint also gets closer to the native Windows appearance this
project has deliberately kept.

The Elm discipline can be imposed by hand: put an `enum Message` and a single
`fn update(&mut self, msg: Message)` in `ui`, make every callback send a message
and nothing else. You lose the compiler's help on wiring; you keep it on
exhaustiveness. Revisit if Iced's IME story lands.

### 4.5 Threads and the state machine

Keep today's shape — it is sound and hard-won — and let the types enforce it:

```
cpal callback  ──► SPSC ring ──►  segmenter task ──► mpsc ──► worker task
   realtime,                       VAD / PTT gating          Whisper → chain
   allocation-free                                           → Piper → sinks
```

- The audio callback owns no allocation and no lock. `rtrb` or similar.
- Everything else on a `tokio` runtime, or plain threads plus `crossbeam`.
  Async buys little here — the work is CPU-bound FFI — so **plain threads plus
  channels is the honest choice**, and it keeps stack traces readable.
- One task owns each model. No mutex on the engines, because ownership says so
  rather than a comment saying so.
- State transitions as a typestate where the lifecycle allows
  (`Pipeline<Stopped>` → `Pipeline<Running>`), and as a checked enum transition
  where it does not (IDLE ↔ LISTENING ↔ THINKING…). The `_ALLOWED` table becomes
  a `const` matrix checked in one `set_state`, same as now, but the *lifecycle*
  half stops being checkable at all because it stops compiling.

### 4.6 The observer boundary

`_notify` exists because touching Tk off-thread crashes it. In Rust that rule is
`!Send` on the UI handle — the compiler refuses the cross-thread call, and the
channel is the only way through. The bug 0.6.2 fixed becomes unwritable.

---

## 5. Cross-platform, honestly

Four platform problems. Two are easy, one is better on Linux, one is genuinely
unsolved.

**Paths, autostart, single instance.** Routine. `directories` crate for
XDG/Known Folders; a `.desktop` file in `~/.config/autostart` vs the Run key; an
abstract socket vs a named mutex. Behind one trait each in `platform`.

**The virtual microphone — better on Linux.** Windows needs a third-party kernel
driver (VB-CABLE), which is why `cable.py` and the wizard exist. On Linux,
PipeWire lets the *application create its own* virtual source at runtime — a
null sink plus `module-remap-source`, or a loopback module, no install, no
elevation, no wizard step. The cross-platform trait is therefore asymmetric on
purpose:

```rust
trait VirtualMic {
    fn detect(&self) -> Option<Cable>;
    /// None on Windows: only the user can install a driver.
    fn create(&self) -> Option<Result<Cable, Error>>;
}
```

Do **not** paper over this. On Linux the first-run wizard should just make the
device; on Windows it must keep pointing at the VB-CABLE download.

**Global hotkeys — the real problem.** X11 is fine. Wayland is not: shortcuts go
through the `org.freedesktop.portal.GlobalShortcuts` portal, KDE implements it,
GNOME's support is partway, and **wlroots-based compositors (Sway, Niri) ship no
implementation at all**. On those, push-to-talk cannot work through supported
APIs. Options, in order of preference:

1. Portal where available. Ask at first run; the portal shows its own consent UI.
2. `evdev` fallback, requiring the user to be in the `input` group. Documented,
   opt-in, and honest about what it means (it reads all keyboard input).
3. **Say so and offer VAD.** Automatic detection needs no hotkey and is already
   a first-class mode. On an unsupported compositor the wizard should recommend
   it rather than offering a key binding that silently never fires.

Option 3 is the one that matters. The app has a mode that works without global
hotkeys; the failure here is a *product* answer, not a technical one.

**Packaging.** Windows: same Inno Setup, one static binary instead of a
PyInstaller tree. Linux: `.deb` + `.rpm` + AppImage. **Avoid Flatpak** — the
sandbox fights both PipeWire device creation and global shortcuts, and debugging
that costs more than it earns.

---

## 6. Correctness by construction, bug by bug

Taking this release's actual bugs and showing what stops them:

```rust
// "translate twice" -- unrepresentable. There is no pair to contradict.
match cfg.translation { Off => .., Models { .. } => .., Recogniser { .. } => .. }

// The VAD fallback that never bound the key. `set_mode` consumes and returns
// the trigger state, so an assignment does not typecheck.
self.trigger = self.trigger.switch_to(TriggerMode::Ptt)?;   // rebinds, or fails

// "return ''" meaning four things.
enum Heard { Speech(Transcript), TooShort, Silence, Noise(String) }
// The caller must match. There is no falsy string to drop on the floor.

// Swapping heard and spoken.
fn check_voice(voice: &Voice, spoken: Spoken) -> Option<Problem>
// check_voice(v, heard) -- mismatched types, does not compile.

// The review hook whose two failure paths disagreed.
fn review(&self, text: &str) -> Result<Option<String>, ReviewError>
// Err and Ok(None) both flow to one `?`; there is no second path to forget.

// A pack that is present but broken.
enum Addon { Missing, Ready(Verified), Broken { reason: String } }
// The boolean that produced a Download button on an installed pack is gone.
```

The `Verified` above is the point of `probe.py`, made into a type: you cannot
obtain one without having actually loaded the thing, so "checked it exists" and
"proved it works" stop being confusable.

---

## 7. How the migration would actually go

Strangler, not big-bang. The Python app keeps shipping throughout.

**Phase 0 — the spike that decides it (1–2 weeks).** Before any architecture.
One throwaway binary that: loads a converted Whisper model through `ct2rs` and
transcribes a WAV; loads an OPUS-MT pair and translates a sentence; synthesises
through `piper-rs`; phonemises Japanese through `jpreprocess`; opens a `cpal`
capture and playback stream. Built for **both** targets, in CI, with and without
CUDA on Windows.

If `ct2rs` will not build cross-platform, or CUDA needs a toolchain nobody can
reproduce, **stop here.** That is a real possible outcome and it costs two
weeks to find out instead of six months.

**Phase 1 — `core`, ported not rewritten (2–3 weeks).** Config parsing, modes,
`plan`, substitutions, language routing. No I/O. This is where the design work
already done pays: `plan.py` and `modes.py` translate almost line for line, and
they arrive better because the compiler enforces what mypy currently asks for.
Port the mode matrix test as-is — it is 972 cases and it is the specification.

**Phase 2 — differential testing against Python (1 week, then continuous).**
A harness that feeds the same inputs to both implementations and diffs:
- the same WAV → transcripts must match exactly (same model, same runtime)
- the same text → translations must match
- the same text → synthesised audio within a tight tolerance
- the same 972 config combinations → the same plan, problem for problem

This is what makes the rewrite safe rather than hopeful. **Do not skip it.** It
also catches the substituted dependencies: `rubato` vs `soxr`, `cpal` vs
PortAudio, where a subtle difference would otherwise show up as "the new one
sounds slightly worse" with no way to pin it down.

**Phase 3 — the engine (3–4 weeks).** Capture, VAD, the worker, output sinks,
streaming with LocalAgreement-2. Headless first — a CLI that does everything
except the tray and settings window. Re-measure the streaming numbers here
(0.5 s interval, 0.6 cost target); they were measured against Python's overhead
and the right values may differ.

**Phase 4 — the interface (4–6 weeks).** Slint. This is the long pole and the
least interesting work. Feature-match the current settings window, which is
2900 lines of Tk.

**Phase 5 — platform (2–3 weeks).** Tray, hotkeys (both paths), autostart,
virtual-device setup, installers for three package formats.

**Phase 6 — Studio (1 week).** Least work: it is already a subprocess boundary.
The Rust side gains a typed protocol instead of parsing stdout.

Realistically **4–6 months part-time**, and the interface is half of it. Anyone
quoting less has not counted Phase 4.

---

## 8. Testing

Keep what works. The four-suite structure is good and the discipline is the
point, not the language.

- `core` gets ordinary `#[test]`s. They run in milliseconds and need nothing
  installed — which is most of what `nothing_installed.py` currently proves.
- The mode matrix becomes a table test, and `proptest` can push it further:
  generate arbitrary configs and assert the invariants (`serious ⟹ has a
  location`, `spoken ≠ voice language ⟹ some serious problem`) rather than
  enumerating.
- `insta` snapshots for the plan's rendered output — the settings window, the
  log line and the diagnostics report all render one `SpeechPlan`, so one
  snapshot covers all three and they cannot drift apart.
- The differential harness from Phase 2 stays in CI for the whole migration.
- **Keep an `isolated`-style check.** Static linking kills the pydantic class,
  but ONNX Runtime and CUDA are still dynamic. A test that runs the built binary
  on a machine with no toolchain, no CUDA, and an empty cache is still the only
  thing that proves what ships works.
- Mutation testing (`cargo-mutants`) instead of the by-hand mutation runs this
  release used. Same idea, automated: it reintroduces bugs and tells you which
  ones no test noticed.

---

## 9. Risks, ranked

1. **`ct2rs` does not build cross-platform, or CUDA is unreproducible.** Kills
   the project. Phase 0 exists solely to find this out in two weeks.
2. **Phase 4 never ends.** GUI work always overruns, the current window is
   large, and Slint is less familiar than Tk. Mitigation: ship the headless CLI
   at the end of Phase 3 and dogfood it; if the interface stalls, that is still
   a usable artefact.
3. **A substituted dependency is subtly worse.** `cpal` device enumeration,
   `rubato` resampling quality, `espeak-rs` phoneme differences. Mitigation:
   Phase 2's differential harness, which catches these as diffs rather than as
   vague complaints months later.
4. **Two Linux audio stacks.** PipeWire is the future, PulseAudio and bare ALSA
   are the present on older distros. Mitigation: target PipeWire, degrade to
   Pulse, document that bare ALSA gets no virtual mic.
5. **Wayland push-to-talk.** Unsolvable on some compositors. Mitigation: VAD
   mode and honesty (§5).
6. **Losing a working product to a rewrite.** The classic failure. Mitigation:
   the Python app ships the whole time, and the Rust one has to beat it on the
   differential harness before anyone switches.

---

## 10. Verdict

**The case for is stronger than I expected, and it is not mainly about types.**

If it were only "Rust would have caught the mode bugs" — that is true, but 0.7.0
already caught them, and mypy plus `StrEnum` plus the mode matrix now stop them
recurring. Redoing 20,000 lines to replace a linter with a compiler is not a
good trade on its own.

What tips it is the **deployment model**. Nearly every bug that reached you
rather than CI came from the gap between a development machine and an installed
one: a pack that resolved dependencies at runtime and missed one; a `.pyd`
locked by an import; tests that quietly used models the author happened to have.
A single statically-linked binary with `jpreprocess` compiled in has no add-on
pack, no wheel resolution, no runtime import to lock a file, and no venv to
diverge from. **Three of the last four bug reports would have been structurally
impossible**, and that is a different kind of argument from "the types are nicer".

Cross-platform is the second real gain: Tkinter plus PyInstaller plus per-OS
audio would make Linux support painful in Python, and it is routine in Rust.

**What I would actually do:**

1. **Ship 0.7.0 properly first.** Finish translation, let it settle. A rewrite
   started mid-feature inherits half-finished ideas.
2. **Run Phase 0.** Two weeks, one throwaway binary, both platforms. It is cheap
   and it answers the only question that can kill the idea.
3. **Then decide with real information.** If `ct2rs` and `jpreprocess` behave,
   the rest is work rather than risk, and the argument in §10 holds. If they do
   not, this document was still worth two weeks to avoid six months.

And regardless of the outcome: **the design work from 0.7.0 is the asset.**
`modes`, `plan`, the mode matrix, parse-don't-validate, "proved not present" —
those are the specification. They port in an afternoon each. The rewrite is
mostly a matter of restating decisions that have already been made, in a
language that will hold them for you.

---

## Sources

- [ct2rs — Rust bindings for CTranslate2](https://crates.io/crates/ct2rs) ·
  [docs](https://docs.rs/ct2rs/latest/ct2rs/) ·
  [repo](https://github.com/jkawamoto/ctranslate2-rs)
- [jpreprocess — OpenJTalk rewritten in Rust](https://github.com/jpreprocess/jpreprocess) ·
  [naist-jdic bundling](https://crates.io/crates/jpreprocess-naist-jdic)
- [piper-rs](https://lib.rs/crates/piper-rs) · [ort (ONNX Runtime)](https://crates.io/crates/ort)
- [cpal](https://crates.io/crates/cpal) · [tray-icon](https://lib.rs/crates/tray-icon)
- [iced](https://iced.rs/) · [egui](https://github.com/emilk/egui) ·
  [2025 survey of Rust GUI libraries](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) ·
  [Windows GUI in Rust, 2026](https://rust-pc.github.io/rust-windows-gui.html)
- [XDG GlobalShortcuts portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.GlobalShortcuts.html) ·
  [Are We GlobalShortcuts Yet?](https://areweglobalshortcutsyet.github.io/)
- [PipeWire virtual devices](https://wiki.archlinux.org/title/PipeWire/Examples)
