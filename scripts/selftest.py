"""Offline self-test: exercises config, VAD, STT and the output sink.

Opens real output streams but writes silence, so it is safe to run with headphones
on. Does not touch the microphone.

    python scripts/selftest.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice2tts.config import Config, OutputTarget, load_config  # noqa: E402
from voice2tts.logging_setup import setup_logging  # noqa: E402

SAMPLE = ROOT / "spike" / "out" / "tts_sample.wav"

# Kept in sync with what test_stt asserts on; "pipeline" must survive recognition.
SAMPLE_TEXT = (
    "This is a test of the voice to text to speech pipeline. "
    "It should split into a few sentences. "
    "Each one gets synthesized separately so playback can start early."
)

passed = failed = 0


def ensure_sample() -> Path | None:
    """Return a speech sample, synthesizing one if it is not already there.

    The sample used to be a leftover from spike/02_tts.py, which meant CI failed on
    a clean checkout -- spike/out is gitignored. Generating it with Piper keeps the
    suite self-contained and costs about a second.
    """
    if SAMPLE.exists():
        return SAMPLE
    try:
        from voice2tts.config import TtsConfig
        from voice2tts.tts import PiperEngine

        engine = PiperEngine(TtsConfig())
        audio = engine.synth(SAMPLE_TEXT)
        if not len(audio):
            return None
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(SAMPLE), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(engine.rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print(f"  (generated {SAMPLE.name}: {len(audio) / engine.rate:.2f}s)")
        return SAMPLE
    except Exception as exc:  # noqa: BLE001 - reported by the caller's check()
        print(f"  (could not generate sample: {exc})")
        return None


def load_sample_16k() -> np.ndarray | None:
    """The speech sample as 16 kHz mono float32, or None if unavailable."""
    path = ensure_sample()
    if path is None:
        return None
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if rate != 16000:
        import soxr

        audio = soxr.resample(audio, rate, 16000)
    return audio


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def test_config() -> None:
    print("\n[config]")
    cfg = Config()
    cfg.audio.outputs = [
        OutputTarget(match="CABLE Input", gain=0.8, enabled=True),
        OutputTarget(match="", gain=0.5, enabled=False),
    ]
    cfg.trigger.hotkey = "ctrl+alt+v"
    tmp = ROOT / "spike" / "out" / "roundtrip.toml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(tmp)

    import tomllib

    with tmp.open("rb") as fh:
        back = Config.from_dict(tomllib.load(fh))
    check("toml round-trip preserves outputs", len(back.audio.outputs) == 2)
    check("output fields survive",
          back.audio.outputs[0].match == "CABLE Input"
          and abs(back.audio.outputs[0].gain - 0.8) < 1e-6
          and back.audio.outputs[0].enabled is True)
    check("second output disabled", back.audio.outputs[1].enabled is False)

    partial = Config.from_dict({"trigger": {"mode": "nonsense"}, "junk": 1})
    check("unknown mode falls back to ptt", partial.trigger.mode == "ptt")
    check("unknown keys ignored", isinstance(partial, Config))

    # An all-disabled config makes the app silent with a confusing error; loading
    # one must repair it rather than leave the user with nowhere for audio to go.
    dead = Config()
    dead.audio.outputs = [
        OutputTarget(match="CABLE Input", gain=1.0, enabled=False),
        OutputTarget(match="", gain=0.7, enabled=False),
    ]
    check("all-disabled outputs detected", dead.ensure_usable_output())
    check("system default enabled by repair",
          any(t.enabled and not t.match for t in dead.audio.outputs))
    check("repair is idempotent", not dead.ensure_usable_output())

    healthy = Config()
    healthy.audio.outputs = [OutputTarget(match="CABLE Input", enabled=True)]
    check("healthy config untouched", not healthy.ensure_usable_output())

    empty = Config()
    empty.audio.outputs = []
    check("empty output list repaired", empty.ensure_usable_output()
          and len(empty.audio.outputs) == 1)
    tmp.unlink(missing_ok=True)


def test_hotkey() -> None:
    print("\n[hotkey]")
    from voice2tts.hotkey import HotkeySpec, describe

    check("parses ctrl+alt+v", describe("ctrl+alt+v") == "")
    check("parses f8", describe("f8") == "")
    check("rejects gibberish", describe("ctrl+nope") != "")
    spec = HotkeySpec("ctrl+shift+k")
    check("modifiers extracted", spec.modifiers == {"ctrl", "shift"})


def test_hotkey_manager() -> None:
    """Several bindings over one hook, and the conflict check."""
    print("\n[hotkey manager]")
    from voice2tts.hotkey import HotkeyManager

    fired: list[str] = []
    mgr = HotkeyManager()
    mgr.bind("ptt", "ctrl+alt+v", lambda: fired.append("ptt-down"),
             lambda: fired.append("ptt-up"))
    mgr.bind("clipboard", "ctrl+alt+c", lambda: fired.append("clip"))
    mgr.bind("stop", "ctrl+alt+x", lambda: fired.append("stop"))
    check("three bindings registered", mgr.bound == ["clipboard", "ptt", "stop"],
          str(mgr.bound))
    check("distinct combos do not conflict", not mgr.conflicts())

    mgr.bind("duplicate", "ctrl+alt+v", lambda: None)
    clashes = mgr.conflicts()
    check("identical combos are reported",
          any({"ptt", "duplicate"} == set(pair) for pair in clashes), str(clashes))
    mgr.unbind("duplicate")

    # Rebinding must replace, not accumulate.
    mgr.bind("clipboard", "ctrl+alt+b", lambda: fired.append("clip2"))
    check("rebinding replaces", len(mgr.bound) == 3, str(mgr.bound))

    try:
        mgr.bind("bad", "ctrl+nonsense", lambda: None)
        check("invalid combo rejected", False, "no error raised")
    except ValueError:
        check("invalid combo rejected", True)
    check("failed bind leaves state untouched", "bad" not in mgr.bound)

    mgr.clear()
    check("clear removes everything", not mgr.bound)
    mgr.stop()  # must be safe without ever having started


def test_clipboard() -> None:
    print("\n[clipboard]")
    from voice2tts import clipboard

    text = clipboard.get_text()
    check("clipboard read without raising", isinstance(text, str),
          f"{len(text)} chars")
    speakable = clipboard.get_speakable_text()
    check("speakable form collapses whitespace",
          "\n" not in speakable and "  " not in speakable,
          repr(speakable[:40]))
    check("truncation limit is sane", clipboard.MAX_CHARS >= 1000)


def test_substitutions() -> None:
    print("\n[substitutions]")
    from voice2tts.substitutions import STARTER_RULES, Rule, Substituter, preview

    sub = Substituter([Rule("brb", "be right back"), Rule("gg", "good game")])
    check("simple replacement", sub.apply("brb") == "be right back")
    check("case-insensitive by default", sub.apply("BRB") == "be right back")
    check("multiple rules in one pass",
          sub.apply("gg, brb") == "good game, be right back")
    check("untouched text passes through", sub.apply("hello there") == "hello there")
    check("empty input is safe", sub.apply("") == "")

    # Whole-word matching is the whole point: a substring rule would maul ordinary
    # words containing the pattern.
    check("whole word does not match inside a word",
          Substituter([Rule("al", "ALPHA")]).apply("also always metal")
          == "also always metal")
    check("partial matching still available when asked for",
          Substituter([Rule("al", "X", whole_word=False)]).apply("metal") == "metX")

    check("punctuation-adjacent still matches",
          sub.apply("brb, back soon") == "be right back, back soon")

    case_rule = Substituter([Rule("IT", "information technology",
                                  case_sensitive=True)])
    check("case-sensitive rule respects case",
          case_rule.apply("IT is it") == "information technology is it")

    regex = Substituter([Rule(r"\bv(\d+)\b", r"version \1", regex=True)])
    check("regex with backreference", regex.apply("v2 release") == "version 2 release",
          regex.apply("v2 release"))
    # ...but a backslash typed into a PLAIN rule must stay literal rather than
    # becoming a capture reference (or raising "invalid group reference").
    literal = Substituter([Rule("path", r"C:\1\temp")])
    check("backslashes literal in a plain rule",
          literal.apply("path") == r"C:\1\temp", literal.apply("path"))

    bad = Rule("(unclosed", "x", regex=True)
    check("invalid regex reported", "invalid regular expression" in bad.describe_error())
    check("invalid regex does not crash the substituter",
          Substituter([bad]).apply("(unclosed") == "(unclosed")
    check("empty pattern rejected", "empty" in Rule("", "x").describe_error())

    check("disabled rules are skipped",
          Substituter([Rule("brb", "nope", enabled=False)]).apply("brb") == "brb")

    # Rules that rewrite each other must terminate rather than spin.
    looping = Substituter([Rule("a", "b"), Rule("b", "a")])
    result = looping.apply("a")
    check("looping rules terminate", result in ("a", "b"), repr(result))

    check("starter rules are usable",
          preview(list(STARTER_RULES), "brb afk") == "be right back away from keyboard")

    # Live wiring: the pipeline must actually apply them.
    from voice2tts.config import Config, SubstitutionRule
    from voice2tts.pipeline import Pipeline

    cfg = Config()
    cfg.text.substitutions = [SubstitutionRule("aiden", "Aidan")]
    p = Pipeline(cfg)
    check("pipeline compiles configured rules", p.apply_text_changes() == 1)
    check("pipeline applies them", p.substituter.apply("tell aiden") == "tell Aidan")
    cfg.text.substitutions_enabled = False
    check("disabling switches them all off", p.apply_text_changes() == 0)
    check("nothing rewritten when disabled",
          p.substituter.apply("tell aiden") == "tell aiden")


def test_device_recovery() -> None:
    """A microphone that goes away must be retried, not written off."""
    print("\n[device recovery]")
    from voice2tts import devices as devices_mod
    from voice2tts.capture import MicCapture
    from voice2tts.config import Config
    from voice2tts.pipeline import Pipeline

    device = devices_mod.default_input() or next(iter(devices_mod.list_inputs()), None)
    if device is None:
        check("an input device exists to test with", False, "none found")
        return

    cap = MicCapture(device)
    check("an unstarted capture is not 'failed'", cap.check_alive() and not cap.failed)

    cap.start()
    check("starts healthy", cap.check_alive() and not cap.failed)
    # Simulate the callback silently stopping, which is how an unplugged USB device
    # usually presents -- PortAudio does not always raise.
    cap._last_callback -= 99
    check("silence is detected as failure", not cap.check_alive())
    check("failure reason recorded", bool(cap.failure_reason), cap.failure_reason)
    cap.stop()

    cap2 = MicCapture(device)
    cap2._mark_failed("test")
    check("failure is sticky until reopened", not cap2.check_alive())

    p = Pipeline(Config())
    p.capture = cap2
    recovered = p._try_recover_capture()
    check("recovery reopens the device", recovered)
    check("a live capture replaces the dead one",
          p.capture is not cap2 and not p.capture.failed)
    check("recovered capture reports alive", p.capture.check_alive())
    p.capture.stop()


def test_theme() -> None:
    print("\n[theme]")
    import tkinter as tk

    from voice2tts import theme

    light, dark, native = theme.LIGHT, theme.DARK, theme.NATIVE
    check("native is the first offered mode", theme.MODES[0] == "native",
          str(theme.MODES))
    check("light and dark differ", light.bg != dark.bg)
    check("dark palette identifies as dark", dark.is_dark and not light.is_dark)
    check("native is never treated as dark", not native.is_dark)
    check("explicit modes ignore the system setting",
          theme.resolve("dark") is dark and theme.resolve("light") is light)
    check("native mode resolves to the native palette",
          theme.resolve("native") is native)
    check("legacy 'system' still follows Windows",
          theme.resolve("system") in (light, dark))
    check("unknown mode falls back to native", theme.resolve("banana") is native)
    check("windows preference readable",
          isinstance(theme.windows_prefers_dark(), bool))

    # Every semantic colour must be set, or status text goes invisible.
    for name, palette in (("light", light), ("dark", dark), ("native", native)):
        missing = [f for f in ("bg", "text", "muted", "ok", "warn", "error")
                   if not getattr(palette, f).startswith("#")]
        check(f"{name} palette complete", not missing, str(missing))

    root = tk.Tk()
    root.withdraw()
    try:
        from tkinter import ttk

        style = ttk.Style(root)
        platform_default = style.theme_use()

        applied = theme.apply(root, "dark")
        check("applying returns the palette in use", applied is dark)
        text = tk.Text(root)
        theme.style_text_widget(text, applied)
        check("text widget takes the palette background",
              text.cget("background") == dark.field, text.cget("background"))

        # The point of native mode: nothing is repainted, and it undoes a previous
        # light/dark apply() rather than leaving clam behind.
        native_applied = theme.apply(root, "native")
        check("native returns the native palette", native_applied is native)
        check("native restores the platform ttk theme",
              style.theme_use() == platform_default,
              f"{style.theme_use()} vs {platform_default}")

        plain = tk.Text(root)
        before = plain.cget("background")
        theme.style_text_widget(plain, native_applied)
        check("native leaves text widgets untouched",
              plain.cget("background") == before, plain.cget("background"))

        theme.apply(root, "light")
        check("theme can be switched at runtime", True)
    finally:
        root.destroy()

    # Every mode must leave comboboxes usable. The dropdown is a plain Tk listbox
    # coloured only through the option database, and a bad value there does not
    # raise at apply() time -- it fails when the user clicks the arrow, which no
    # amount of widget-construction testing catches. Post one for real.
    def dropdown_opens(mode: str) -> str:
        root = tk.Tk()
        root.withdraw()
        try:
            theme.apply(root, mode)
            win = tk.Toplevel(root)
            combo = ttk.Combobox(win, values=["one", "two"])
            combo.pack()
            root.update()
            try:
                root.tk.call("ttk::combobox::Post", str(combo))
                root.update()
                return ""
            except tk.TclError as exc:
                return str(exc)
        finally:
            root.destroy()

    for mode in theme.MODES:
        error = dropdown_opens(mode)
        check(f"dropdowns open in {mode} mode", not error, error)

    # Switching away from a repainted theme must restore usable colours, not
    # leave the empty strings that broke every picker in 0.5.1.
    root = tk.Tk()
    root.withdraw()
    try:
        theme.apply(root, "dark")
        theme.apply(root, "native")
        win = tk.Toplevel(root)
        combo = ttk.Combobox(win, values=["one", "two"])
        combo.pack()
        root.update()
        try:
            root.tk.call("ttk::combobox::Post", str(combo))
            root.update()
            switched = ""
        except tk.TclError as exc:
            switched = str(exc)
    finally:
        root.destroy()
    check("dropdowns still open after switching back to native",
          not switched, switched)


def test_studio_gate() -> None:
    """The hardware gate, and that the override is a real escape hatch."""
    print("\n[studio gate]")
    from voice2tts import studiopack
    from voice2tts.studiopack import Hardware

    good = Hardware(gpu_name="RTX 5080", vram_gb=16.0, free_disk_gb=200.0,
                    cuda_pack_installed=True)
    check("capable machine passes", studiopack.gate(hardware=good).ok)
    check("a passing machine has no blockers",
          not studiopack.gate(hardware=good).blockers)

    no_gpu = Hardware(gpu_name="", vram_gb=0.0, free_disk_gb=200.0)
    result = studiopack.gate(hardware=no_gpu)
    check("no GPU is blocked", not result.ok)
    check("and says why", "NVIDIA" in " ".join(result.blockers), str(result.blockers))

    small = Hardware(gpu_name="GTX 1660", vram_gb=6.0, free_disk_gb=200.0)
    result = studiopack.gate(hardware=small)
    check("under-spec VRAM is blocked", not result.ok)
    check("but the message says it may still work",
          "smaller batch" in " ".join(result.blockers), str(result.blockers))

    full = Hardware(gpu_name="RTX 5080", vram_gb=16.0, free_disk_gb=5.0)
    result = studiopack.gate(hardware=full)
    check("insufficient disk is blocked", not result.ok)
    check("disk message names the shortfall",
          "free" in " ".join(result.blockers).lower(), str(result.blockers))

    # The override is the point of the design: an OOM costs time, not damage.
    overridden = studiopack.gate(hardware=small, override=True)
    check("override lets a weak machine proceed", overridden.ok)
    check("override is recorded, not hidden", overridden.overridden)
    check("blockers are still reported when overridden",
          bool(overridden.blockers))
    check("override does not fabricate a pass on capable hardware",
          not studiopack.gate(hardware=good, override=True).overridden)

    # A missing GPU pack is advice, not an obstacle: training brings its own CUDA.
    no_cuda = Hardware(gpu_name="RTX 5080", vram_gb=16.0, free_disk_gb=200.0,
                       cuda_pack_installed=False)
    result = studiopack.gate(hardware=no_cuda)
    check("missing GPU pack warns but does not block",
          result.ok and result.warnings, str(result.warnings))

    check("probe runs on this machine", isinstance(studiopack.probe(), Hardware))
    live = studiopack.probe()
    check("probe finds this GPU", live.has_gpu, f"{live.gpu_name} {live.vram_gb} GB")
    check("probe reads free disk", live.free_disk_gb > 0,
          f"{live.free_disk_gb} GB")

    state = studiopack.status()
    check("pack status readable", isinstance(state.installed, bool),
          f"installed={state.installed} {state.size_gb:.1f} GB")
    check("studio interpreter path is isolated from ours",
          "studio" in str(studiopack.python_exe()).lower(),
          str(studiopack.python_exe()))

    env = studiopack.environment_for_training()
    check("training env drops PYTHONPATH",
          "PYTHONPATH" not in env and "PYTHONHOME" not in env)

    check("torch pin targets a Blackwell-capable CUDA build",
          "cu128" in studiopack.TORCH_SPEC, studiopack.TORCH_SPEC)


def test_release_is_gated() -> None:
    """A tag must not be able to publish without the checks having run.

    v0.5.0 shipped from a commit whose CI run had failed: ci.yml ignores tags, and
    release.yml ran no tests, so nothing stood between a tag push and a published
    installer. A comment claimed otherwise, which is why this is now asserted.
    """
    print("\n[release gating]")
    import yaml

    wf = ROOT / ".github" / "workflows"
    release = yaml.safe_load((wf / "release.yml").read_text(encoding="utf-8"))
    ci = yaml.safe_load((wf / "ci.yml").read_text(encoding="utf-8"))

    jobs = release.get("jobs", {})
    check("release.yml has a verify job", "verify" in jobs, str(list(jobs)))

    publisher = jobs.get("release", {})
    needs = publisher.get("needs") or []
    if isinstance(needs, str):
        needs = [needs]
    check("the publishing job depends on verify", "verify" in needs, str(needs))

    def step_names(job):
        return [str(s.get("name", "")) for s in job.get("steps", [])]

    verify_steps = " ".join(step_names(jobs.get("verify", {}))).lower()
    check("verify runs the self-test", "self-test" in verify_steps, verify_steps)
    check("verify runs the linter", "lint" in verify_steps, verify_steps)
    check("verify checks the tag against __version__",
          "resolve version" in verify_steps, verify_steps)

    # The publishing job must not run any of that itself -- if it did, a failure
    # there would happen after the build rather than before it.
    publish_steps = " ".join(step_names(publisher)).lower()
    check("publishing happens only after verify",
          "self-test" not in publish_steps, publish_steps)

    # ci.yml skipping tags is only safe because release.yml verifies them.
    triggers = ci.get(True) or ci.get("on") or {}
    push = triggers.get("push", {}) if isinstance(triggers, dict) else {}
    if "tags-ignore" in push:
        check("tags skipped in CI are covered by release verify",
              "verify" in jobs and "self-test" in verify_steps,
              "ci.yml ignores tags")
    else:
        check("tags skipped in CI are covered by release verify", True,
              "ci.yml also runs on tags")


def test_winget_manifests() -> None:
    print("\n[winget]")
    import yaml

    folder = ROOT / "installer" / "winget"
    files = sorted(folder.glob("*.yaml"))
    check("three manifests present", len(files) == 3, str([f.name for f in files]))

    ids, versions, types = set(), set(), set()
    for path in files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            check(f"{path.name} parses", False, str(exc))
            continue
        ids.add(data.get("PackageIdentifier"))
        versions.add(str(data.get("PackageVersion")))
        types.add(data.get("ManifestType"))

    check("identifier consistent across manifests", len(ids) == 1, str(ids))
    check("version consistent across manifests", len(versions) == 1, str(versions))
    check("all three manifest types present",
          types == {"version", "installer", "defaultLocale"}, str(types))

    import voice2tts

    check("manifest version matches the package",
          versions == {voice2tts.__version__},
          f"{versions} vs {voice2tts.__version__}")

    locale = yaml.safe_load(
        (folder / "Voice2TTS.Voice2TTS.locale.en-US.yaml").read_text(encoding="utf-8"))
    check("licence declared as GPL", "GPL-3.0" in str(locale.get("License")),
          str(locale.get("License")))
    check("short description within winget's limit",
          len(locale.get("ShortDescription", "")) <= 256)


def test_profiles() -> None:
    print("\n[profiles]")
    from voice2tts import profiles
    from voice2tts.config import Config

    cfg = Config()
    cfg.trigger.mode = "vad"
    cfg.tts.voice = "en_US-amy-medium"
    cfg.tts.length_scale = 1.4
    cfg.audio.input_match = "some microphone"

    snapshot = profiles.capture(cfg, "meeting")
    check("profile captures the profiled fields",
          snapshot.values["trigger.mode"] == "vad"
          and snapshot.values["tts.voice"] == "en_US-amy-medium")
    # Device and model choices describe the machine, not the situation; a profile
    # that silently changed your microphone would be a nasty surprise.
    check("device selection is not captured",
          "audio.input_match" not in snapshot.values)
    check("whisper model is not captured", "stt.model" not in snapshot.values)

    cfg.trigger.mode = "ptt"
    cfg.tts.voice = "en_US-ryan-high"
    cfg.audio.input_match = "another microphone"
    changed = profiles.apply(cfg, snapshot)
    check("applying restores captured fields",
          cfg.trigger.mode == "vad" and cfg.tts.voice == "en_US-amy-medium")
    check("changed fields reported", set(changed) >= {"trigger.mode", "tts.voice"},
          str(changed))
    check("unprofiled fields left alone",
          cfg.audio.input_match == "another microphone")
    check("re-applying reports no change", not profiles.apply(cfg, snapshot))

    round_tripped = profiles.from_dict(profiles.to_dict(snapshot))
    check("profile survives a round-trip",
          round_tripped.values == snapshot.values
          and round_tripped.name == "meeting")

    # A config written by a newer version may name fields we do not know.
    hostile = profiles.from_dict(
        {"name": "x", "values": {"trigger.mode": "vad", "evil.field": 1}})
    check("unknown fields dropped on load", "evil.field" not in hostile.values)

    tagged = profiles.Profile("gaming", {}, match_apps=["discord.exe"])
    check("app matching finds a profile",
          profiles.find_for_app([tagged], "C:\\x\\Discord.exe") is tagged)
    check("app matching ignores others",
          profiles.find_for_app([tagged], "notepad.exe") is None)
    check("blank executable matches nothing",
          profiles.find_for_app([tagged], "") is None)

    check("foreground lookup returns a string",
          isinstance(profiles.foreground_executable(), str))


def test_history_and_review() -> None:
    print("\n[history + review]")
    from voice2tts.config import Config
    from voice2tts.pipeline import Pipeline

    cfg = Config()
    cfg.text.history_size = 3
    p = Pipeline(cfg)

    p.record("heard one", "spoken one")
    p.record("heard two", "spoken two", "clipboard")
    check("entries recorded", len(p.history) == 2)
    check("heard and spoken kept separately", p.history[0].heard == "heard one"
          and p.history[0].spoken == "spoken one")
    check("source recorded", p.history[1].source == "clipboard")
    check("edited flag set when they differ", p.history[0].edited)

    for i in range(5):
        p.record(f"x{i}", f"x{i}")
    check("history bounded by history_size", len(p.history) == 3, str(len(p.history)))
    check("oldest dropped first", p.history[-1].heard == "x4")

    p.clear_history()
    check("history clears", not p.history)

    # Review: the hook decides what gets spoken.
    cfg.text.review_before_speaking = True
    seen: list[str] = []

    p.review_hook = lambda text: (seen.append(text), text.upper())[1]
    check("hook receives the text and can rewrite it",
          p._review("hello") == "HELLO" and seen == ["hello"])

    p.review_hook = lambda _text: None
    check("returning None discards", p._review("hello") is None)

    # A broken hook must not swallow the utterance silently.
    def boom(_text):
        raise RuntimeError("hook exploded")

    p.review_hook = boom
    check("a failing hook falls back to speaking", p._review("hello") == "hello")


def test_vad() -> None:
    print("\n[vad]")
    from voice2tts.config import VadConfig
    from voice2tts.vad import WINDOW, SileroVad, VadSegmenter

    vad = SileroVad()
    silence = np.zeros(WINDOW, dtype=np.float32)
    check("silence scores low", vad(silence) < 0.3, f"p={vad(silence):.3f}")

    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return

    seg = VadSegmenter(VadConfig(), preroll_ms=300, max_utterance_s=30)
    utterances = []
    # Pad with silence so the final utterance gets its end-of-speech trigger.
    padded = np.concatenate([np.zeros(16000, np.float32), audio, np.zeros(24000, np.float32)])
    for i in range(0, len(padded) - WINDOW, WINDOW):
        got = seg.process(padded[i:i + WINDOW])
        if got is not None:
            utterances.append(got)
    tail = seg.flush()
    if tail is not None:
        utterances.append(tail)

    total = sum(len(u) for u in utterances) / 16000
    check("segmented speech from silence", len(utterances) >= 1,
          f"{len(utterances)} utterance(s), {total:.2f}s of {len(audio)/16000:.2f}s")


def test_stt() -> None:
    print("\n[stt]")
    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return
    from voice2tts.config import SttConfig
    from voice2tts.stt import WhisperEngine

    engine = WhisperEngine(SttConfig())
    warm = engine.warmup()
    check("engine loaded", True, f"{engine.device}/{engine.compute_type}")
    check("warmup completed", warm > 0, f"{warm:.2f}s")

    t0 = time.perf_counter()
    text = engine.transcribe(audio)
    elapsed = time.perf_counter() - t0
    check("transcribes sample", "pipeline" in text.lower(), f"{elapsed*1000:.0f} ms")
    check("hallucination filter drops 'Thank you.'", engine._is_noise("Thank you."))
    check("real text survives filter", not engine._is_noise("hello there friend"))


def test_tts_and_sink() -> None:
    print("\n[tts + output]")
    from voice2tts.config import TtsConfig
    from voice2tts.output import OutputSink
    from voice2tts.tts import PiperEngine

    cfg = load_config()
    tts = PiperEngine(TtsConfig())
    check("piper loaded", tts.rate > 0, f"{tts.rate} Hz")

    audio = tts.synth("Self test.")
    check("synthesized audio", len(audio) > 1000, f"{len(audio)/tts.rate:.2f}s")

    # Open the real default device but push silence: verifies stream setup and the
    # resampler path without making noise.
    sink = OutputSink(cfg.audio)
    failures = sink.configure([OutputTarget(match="", gain=1.0, enabled=True)], tts.rate)
    check("default output opened", len(sink.targets) == 1,
          sink.targets[0].name if sink.targets else str(failures))
    if sink.targets:
        t = sink.targets[0]
        check("resampler engaged where needed",
              (t._resampler is not None) == (t.rate != tts.rate),
              f"{tts.rate} -> {t.rate}")
        sink.begin_utterance()
        sink.write(np.zeros(tts.rate, dtype=np.float32))
        check("audio queued", sink.active)
        sink.end_utterance()
        drained = sink.wait_drain(timeout=5)
        check("drains cleanly", drained)
        check("no underruns", t.underruns == 0, f"{t.underruns}")
    sink.close()


def test_packaging_bits() -> None:
    """Offline checks for the cable, voices and GPU-pack modules."""
    print("\n[packaging]")
    import zipfile

    from voice2tts import cable, gpupack, voices
    from voice2tts.paths import bundled_whisper, cache_dir, cuda_dir

    # --- paths
    check("cache dir is LOCALAPPDATA", "Local" in str(cache_dir()), str(cache_dir()))
    check("cuda dir under cache", cuda_dir().parent == cache_dir())
    check("base.en bundled for offline install", bundled_whisper("base.en") is not None,
          str(bundled_whisper("base.en")))
    check("absent model reports None", bundled_whisper("nonexistent-model") is None)

    # --- voices
    installed = voices.installed_keys()
    missing = [v for v in voices.BUNDLED if v not in installed]
    check("all three voices bundled", not missing, f"missing={missing}" if missing
          else ", ".join(voices.BUNDLED))
    check("bundled voices are not removable",
          not any(voices.is_removable(v) for v in voices.BUNDLED))
    check("unknown voice resolves to None", voices.installed_path("no-such-voice") is None)

    # --- cable detection
    found = cable.detect()
    check("cable detection runs", True, found.label if found else "none installed")
    check("installed() agrees with detect()", cable.installed() == (found is not None))
    check("product table is non-trivial", len(cable._PRODUCTS) >= 10,
          f"{len(cable._PRODUCTS)} patterns")

    # The single source of truth: devices.find_cable_output() used to disagree with
    # cable.detect() -- one found a Matrix channel by substring while the other
    # reported nothing, so the wizard said "not installed" while the config wired
    # up a device anyway.
    from voice2tts import devices as devices_mod

    legacy = devices_mod.find_cable_output()
    check("device helper agrees with cable.detect()",
          (legacy is None) == (found is None)
          and (found is None or legacy.name == found.output_name),
          f"{getattr(legacy, 'name', None)} vs {found.output_name if found else None}")

    check("driver tag extracted",
          cable._driver_tag("VBMatrix In 8 (VB-Audio Matrix VAIO)")
          == "vb-audio matrix vaio")
    check("channel extracted", cable._channel("VBMatrix In 8 (VB-Audio Matrix VAIO)") == 8)
    check("no channel when unnumbered",
          cable._channel("CABLE Input (VB-Audio Virtual Cable)") is None)

    products = {
        "VBMatrix In 1 (VB-Audio Matrix VAIO)": "VB-Audio Matrix",
        "CABLE Input (VB-Audio Virtual Cable)": "VB-CABLE",
        "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)": "VoiceMeeter",
        "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)": "VoiceMeeter Banana",
        "VoiceMeeter VAIO3 Input (VB-Audio VoiceMeeter VAIO3)": "VoiceMeeter Potato",
        "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)": "VB-Audio Hi-Fi Cable",
    }
    misnamed = {n: cable._identify(n) for n, want in products.items()
                if (cable._identify(n) or ("", 0))[0] != want}
    check("products identified from the driver tag", not misnamed, str(misnamed))
    check("real hardware is not mistaken for a cable",
          cable._identify("Headphones (MOMENTUM 3)") is None
          and cable._identify("Microphone (fifine Microphone)") is None)
    check("NVIDIA virtual audio is not treated as a cable",
          cable._identify("NVIDIA Virtual Audio Device (Wave Extensible) (WDM)") is None,
          str(cable._identify("NVIDIA Virtual Audio Device (Wave Extensible) (WDM)")))

    # Pairing, against a synthetic multi-channel device set. This is the case that
    # broke on real hardware: "In 8" must pair with "Out 8", not whichever endpoint
    # happened to be enumerated first.
    Dev = devices_mod.Device
    fake_inputs = [
        Dev(index=i, name=f"VBMatrix Out {i} (VB-Audio Matrix VAIO)",
            hostapi=devices_mod.WASAPI, rate=48000, max_in=2, max_out=0)
        for i in range(1, 9)
    ]
    mismatches = []
    for n in range(1, 9):
        out = Dev(index=100 + n, name=f"VBMatrix In {n} (VB-Audio Matrix VAIO)",
                  hostapi=devices_mod.WASAPI, rate=48000, max_in=0, max_out=2)
        partner, certain = cable._pair_input(out, fake_inputs)
        if partner != f"VBMatrix Out {n} (VB-Audio Matrix VAIO)" or not certain:
            mismatches.append((n, partner, certain))
    check("multi-channel pairing matches by number", not mismatches, str(mismatches))

    # Single-cable pairing has no numbers to match on; the driver tag alone must do.
    cable_out = Dev(index=1, name="CABLE Input (VB-Audio Virtual Cable)",
                    hostapi=devices_mod.WASAPI, rate=48000, max_in=0, max_out=2)
    cable_in = [Dev(index=2, name="CABLE Output (VB-Audio Virtual Cable)",
                    hostapi=devices_mod.WASAPI, rate=48000, max_in=2, max_out=0)]
    check("single cable pairs by driver tag",
          cable._pair_input(cable_out, cable_in) ==
          ("CABLE Output (VB-Audio Virtual Cable)", True))

    # VoiceMeeter's 2024 driver renamed the capture side to "Out B1", so the old
    # Input->Output rename no longer works; the driver tag still does.
    vm_out = Dev(index=3, name="VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
                 hostapi=devices_mod.WASAPI, rate=48000, max_in=0, max_out=2)
    vm_in = [Dev(index=4, name="VoiceMeeter Out B1 (VB-Audio VoiceMeeter VAIO)",
                 hostapi=devices_mod.WASAPI, rate=48000, max_in=2, max_out=0)]
    check("VoiceMeeter 2024 naming pairs correctly",
          cable._pair_input(vm_out, vm_in) ==
          ("VoiceMeeter Out B1 (VB-Audio VoiceMeeter VAIO)", True))

    # A render-only virtual device must be rejected: without a capture side it
    # swallows speech Discord can never hear.
    render_only = Dev(index=90, name="Some Renderer (VB-Audio Thing)",
                      hostapi=devices_mod.WASAPI, rate=48000, max_in=0, max_out=2)
    check("render-only virtual device finds no partner",
          cable._pair_input(render_only, [])[0] == "")

    check("config fragments recognised as virtual",
          all(cable.is_virtual_device(x) for x in
              ("CABLE Input", "VBMatrix In 3", "VoiceMeeter Aux Input")))
    check("real devices not recognised as virtual",
          not any(cable.is_virtual_device(x) for x in
                  ("Headphones (MOMENTUM 3)", "Speakers (Realtek)", "")))

    # The scraper regex is the fragile part; pin its behaviour on sample markup.
    sample = ('<a href="https://download.vb-audio.com/Download_CABLE/'
              'VBCABLE_Driver_Pack45.zip">Download</a>')
    m = cable._ZIP_LINK.search(sample)
    check("scraper finds a driver pack link", m is not None and m.group(0).endswith(".zip"),
          m.group(0) if m else "no match")
    check("scraper ignores unrelated zips",
          cable._ZIP_LINK.search('<a href="https://x.test/other.zip">') is None)

    # --- archive safety
    bad = ROOT / "spike" / "out" / "evil.zip"
    bad.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("VBCABLE_Setup_x64.exe", b"stub")
        zf.writestr("../../escaped.txt", b"nope")
    try:
        cable.extract(bad, ROOT / "spike" / "out" / "unpack")
        check("traversal path rejected", False, "extract() accepted ../ member")
    except RuntimeError as exc:
        check("traversal path rejected", "unsafe path" in str(exc), str(exc))
    finally:
        bad.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "spike" / "out" / "unpack", ignore_errors=True)

    # --- gpu pack
    pack = gpupack.status()
    check("gpu pack status readable", isinstance(pack.installed, bool),
          f"installed={pack.installed} usable={pack.usable} {pack.size_mb:.0f} MB")
    check("nvidia gpu probe runs", isinstance(gpupack.gpu_present(), bool),
          f"gpu_present={gpupack.gpu_present()}")

    # gpupack writes nvidia/<pkg>/bin INSIDE its own directory, one level deeper
    # than the pip layout. A single-level glob in cuda.py missed it entirely, and
    # the bug was invisible in a venv because the pip-installed wheels sit at the
    # depth the glob expected. Build the layout in a temp dir so nothing else can
    # mask a regression.
    import tempfile
    from unittest import mock

    from voice2tts import cuda as cuda_mod

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "cuda"
        for pkg, dll in (("cublas", "cublas64_12.dll"), ("cudnn", "cudnn64_9.dll")):
            d = fake / "nvidia" / pkg / "bin"
            d.mkdir(parents=True)
            (d / dll).write_bytes(b"MZ stub")
        with mock.patch("voice2tts.paths.cuda_dir", return_value=fake):
            found = cuda_mod._search_dirs()
        names = {p.name for d in found for p in d.glob("*.dll")}
        check("cuda search finds the gpupack layout",
              "cublas64_12.dll" in names and "cudnn64_9.dll" in names,
              f"{len(found)} dirs")

    if pack.usable:
        # If the pack is installed for real, the loader must actually see it.
        dirs = cuda_mod._search_dirs()
        has_cublas = any((d / "cublas64_12.dll").exists() for d in dirs)
        check("installed gpu pack is on the search path", has_cublas,
              f"{len(dirs)} dirs searched")


def test_updates() -> None:
    """Version comparison, throttling, and the refusal to self-install from source."""
    print("\n[updates]")
    import voice2tts
    from voice2tts import updater
    from voice2tts.config import Config

    check("version is a valid triple",
          len(updater.parse_version(voice2tts.__version__)) >= 2,
          voice2tts.__version__)

    cases = [
        ("0.3.0", "0.2.0", True),
        ("0.2.1", "0.2.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.2.0", "0.2.0", False),
        ("0.1.0", "0.2.0", False),
        ("0.10.0", "0.9.0", True),      # numeric, not lexicographic
        ("v0.3.0", "0.2.0", True),      # tolerates a v prefix
        ("0.3.0-beta", "0.2.0", True),  # tolerates a pre-release suffix
    ]
    wrong = [(a, b, e) for a, b, e in cases if updater.is_newer(a, b) != e]
    check("version comparison", not wrong, f"failed: {wrong}" if wrong else f"{len(cases)} cases")

    check("throttle blocks a recent check",
          not updater.should_check(time.time(), 24))
    check("throttle allows an old check",
          updater.should_check(time.time() - 25 * 3600, 24))
    check("interval 0 disables checking",
          not updater.should_check(0, 0))

    # Bad repo strings must fail loudly rather than build a nonsense URL.
    for bad in ("", "not-a-repo", "https://example.test"):
        try:
            updater.check(bad)
            check(f"rejects repo {bad!r}", False, "no error raised")
        except ValueError:
            check(f"rejects repo {bad!r}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"rejects repo {bad!r}", False, f"wrong error: {type(exc).__name__}")

    # A source checkout must never run an installer over itself.
    try:
        updater.apply(ROOT / "does-not-exist.exe")
        check("refuses to self-install from source", False, "no error raised")
    except RuntimeError as exc:
        check("refuses to self-install from source", "source checkout" in str(exc))
    except Exception as exc:  # noqa: BLE001
        check("refuses to self-install from source", False, str(exc))

    # The repository must be prefilled, or every user has to find and type it.
    import voice2tts as pkg
    from voice2tts.config import CURRENT_SCHEMA

    fresh = Config()
    check("new config prefills the update repo",
          fresh.updates.repo == pkg.DEFAULT_UPDATE_REPO, fresh.updates.repo)
    check("default repo passes the same validation as user input",
          bool(updater._REPO_RE.match(pkg.DEFAULT_UPDATE_REPO)),
          pkg.DEFAULT_UPDATE_REPO)
    check("new config is written at the current schema",
          fresh.schema_version == CURRENT_SCHEMA, str(fresh.schema_version))

    # A schema-1 file predates the prefill, so a blank repo there means "never set"
    # and should adopt the default.
    old = Config.from_dict({"schema_version": 1, "updates": {"repo": ""}})
    notes = old.migrate()
    check("schema 1 with a blank repo adopts the default",
          old.updates.repo == pkg.DEFAULT_UPDATE_REPO, str(notes))
    check("migration advances the schema", old.schema_version == CURRENT_SCHEMA)

    # ...but a repo the user chose must survive untouched.
    kept = Config.from_dict({"schema_version": 1, "updates": {"repo": "someone/fork"}})
    kept.migrate()
    check("migration keeps a user-chosen repo", kept.updates.repo == "someone/fork",
          kept.updates.repo)

    # From schema 2 on, blank means the user deliberately turned checking off.
    off = Config.from_dict({"schema_version": CURRENT_SCHEMA, "updates": {"repo": ""}})
    off.migrate()
    check("clearing the repo stays cleared at current schema", off.updates.repo == "",
          repr(off.updates.repo))

    # Schema 3: the briefly-shipped repainted interface goes back to native, but a
    # deliberate light or dark choice is not overridden.
    themed = Config.from_dict({"schema_version": 2, "theme": "system"})
    themed.migrate()
    check("old default theme migrates to native", themed.theme == "native",
          themed.theme)
    chosen = Config.from_dict({"schema_version": 2, "theme": "dark"})
    chosen.migrate()
    check("an explicit theme choice survives migration", chosen.theme == "dark")
    check("new configs default to native", Config().theme == "native")

    # Config normalisation: a pasted URL should become owner/name.
    cfg = Config()
    cfg.updates.repo = "https://github.com/someone/Voice2TTS/"
    cfg.validate()
    check("normalises a pasted repo URL", cfg.updates.repo == "someone/Voice2TTS",
          cfg.updates.repo)
    cfg.updates.interval_hours = -5
    cfg.validate()
    check("clamps negative interval", cfg.updates.interval_hours == 0)

    # Settings must survive an upgrade: everything user-owned lives outside {app}.
    from voice2tts.paths import cache_dir, config_path, user_data_dir

    app_dir = Path(sys.prefix)
    for name, path in (("config", config_path()), ("voices", user_data_dir()),
                       ("cache", cache_dir())):
        check(f"{name} stored outside the install dir",
              app_dir not in path.parents and path != app_dir, str(path))


def test_network() -> None:
    """Live checks against VB-Audio and HuggingFace. Needs internet."""
    print("\n[network]")
    from voice2tts import cable, voices

    try:
        url, source = cable.resolve_download_url()
        check("cable download URL resolves",
              url.lower().endswith(".zip"), f"{source}: {url.rsplit('/', 1)[-1]}")
    except Exception as exc:  # noqa: BLE001
        check("cable download URL resolves", False, str(exc))

    try:
        catalogue = voices.fetch_catalogue()
        check("voice catalogue fetched", len(catalogue) > 50, f"{len(catalogue)} voices")
        english = voices.filter_catalogue(catalogue, language_prefix="en_US")
        check("catalogue filters by language", len(english) > 5, f"{len(english)} en_US")
        keys = {e.key for e in catalogue}
        missing = [v for v in voices.BUNDLED if v not in keys]
        check("bundled voices exist in catalogue", not missing, str(missing))
        sized = [e for e in catalogue if e.size_mb > 0]
        check("catalogue reports sizes", len(sized) > len(catalogue) // 2,
              f"{len(sized)}/{len(catalogue)}")
    except Exception as exc:  # noqa: BLE001
        check("voice catalogue fetched", False, str(exc))


def test_pipeline_end_to_end() -> None:
    """Drive the real Pipeline threads with a recorded utterance.

    Outputs are all disabled so this makes no sound; the microphone is opened (the
    pipeline needs an input device) but nothing captured from it is used, because
    push-to-talk mode leaves VAD idle and we inject the utterance directly.
    """
    print("\n[pipeline end-to-end]")
    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return

    import threading

    from voice2tts.pipeline import Pipeline, State

    cfg = load_config()
    cfg.trigger.mode = "ptt"
    for target in cfg.audio.outputs:
        target.enabled = False  # silent run

    seen: list[tuple[str, str]] = []
    spoke = threading.Event()

    def on_state(state: State) -> None:
        if state is State.SPEAKING:
            spoke.set()

    pipeline = Pipeline(cfg, on_state=on_state, on_event=lambda k, m: seen.append((k, m)))
    try:
        pipeline.start()
        check("pipeline started", pipeline.running)
        check("state is idle", pipeline.state is State.IDLE, pipeline.state.value)

        pipeline._submit(audio)

        deadline = time.time() + 30
        while time.time() < deadline and not pipeline.last_transcript:
            time.sleep(0.05)

        check("produced a transcript", bool(pipeline.last_transcript),
              pipeline.last_transcript[:60])
        check("reached SPEAKING state", spoke.wait(timeout=10))

        deadline = time.time() + 15
        while time.time() < deadline and pipeline.state is not State.IDLE:
            time.sleep(0.05)
        check("returned to idle", pipeline.state is State.IDLE, pipeline.state.value)

        kinds = {k for k, _ in seen}
        check("emitted transcript event", "transcript" in kinds, str(sorted(kinds)))
        check("emitted latency event", "latency" in kinds,
              next((m for k, m in seen if k == "latency"), "-"))

        status = pipeline.status()
        check("no dropped capture windows", status["dropped"] == 0, str(status["dropped"]))
    finally:
        pipeline.shutdown()
    check("shut down cleanly", not pipeline.running)


def test_device_lists() -> None:
    """The pickers must show one row per real endpoint, not one per host API."""
    print("\n[device lists]")
    from voice2tts import devices

    trimmed_in = devices.list_inputs()
    trimmed_out = devices.list_outputs()
    all_in = devices.list_inputs(all_apis=True)
    all_out = devices.list_outputs(all_apis=True)

    # A CI runner has no audio hardware at all, so there is nothing to trim and
    # "fewer than before" is not a meaningful claim. Assert what still holds --
    # enumeration works and returns nothing -- rather than failing the build.
    if not all_in and not all_out:
        check("no-hardware enumeration is empty, not broken",
              trimmed_in == [] and trimmed_out == []
              and devices.default_input() is None
              and devices.resolve_input("") is None,
              "no audio devices present")
    else:
        check("trimmed list is smaller than the raw one",
              len(trimmed_in) < len(all_in) and len(trimmed_out) < len(all_out),
              f"{len(trimmed_in)}/{len(all_in)} in, "
              f"{len(trimmed_out)}/{len(all_out)} out")
    check("trimmed never exceeds the raw list",
          len(trimmed_in) <= len(all_in) and len(trimmed_out) <= len(all_out))
    check("trimmed lists are WASAPI only",
          all(d.hostapi == devices.WASAPI for d in trimmed_in + trimmed_out))
    # Devices CAN legitimately share a name -- two identical monitors on one GPU.
    # What must be unique is the dropdown label, or the second is unselectable.
    for listing in (trimmed_in, trimmed_out):
        labels = devices.annotate(listing)
        if len(set(labels)) != len(labels):
            dupes = [x for x in labels if labels.count(x) > 1]
            check("dropdown labels are unique", False, str(sorted(set(dupes))))
            break
    else:
        check("dropdown labels are unique", True)

    # And each ordinal must resolve to a different device.
    dup_names = [n for n in {d.name for d in trimmed_out}
                 if sum(1 for d in trimmed_out if d.name == n) > 1]
    if dup_names:
        name = dup_names[0]
        first = devices.resolve_output(f"{name}   #1")
        second = devices.resolve_output(f"{name}   #2")
        check("same-named devices resolve separately",
              first is not None and second is not None and first.index != second.index,
              f"{name[:32]}: #{getattr(first,'index','?')} vs #{getattr(second,'index','?')}")
    else:
        check("same-named devices resolve separately", True, "none present")

    check("ordinal parsing", devices.split_ordinal("Speakers   #3") == ("Speakers", 3)
          and devices.split_ordinal("Speakers") == ("Speakers", 1))

    pseudo = [d.name for d in trimmed_in + trimmed_out
              if devices.is_pseudo_device(d.name)]
    check("routing aggregates hidden", not pseudo, str(pseudo))
    check("pseudo-device detection works",
          devices.is_pseudo_device("Microsoft Sound Mapper - Input")
          and devices.is_pseudo_device("Primary Sound Driver")
          and devices.is_pseudo_device(
              "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0)")
          and not devices.is_pseudo_device("Headphones (MOMENTUM 3)"))

    # Display annotations must round-trip, or a saved config keeps a suffix that
    # stops the device resolving.
    for raw in ("Headphones (MOMENTUM 3)", "VBMatrix In 1 (VB-Audio Matrix VAIO)"):
        for tag in ("", devices.DEFAULT_TAG, devices.VIRTUAL_TAG):
            if devices.strip_display(raw + tag) != raw:
                check("display tags strip cleanly", False, repr(raw + tag))
                break
    else:
        check("display tags strip cleanly", True)

    # A device only reachable through a hidden host API must still resolve, so an
    # older config does not silently fall back to the default microphone.
    hidden = next((d for d in all_in if d.hostapi != devices.WASAPI
                   and not any(t.name == d.name for t in trimmed_in)), None)
    if hidden is not None:
        found = devices.resolve_input(hidden.name)
        check("hidden-API device still resolves", found is not None, hidden.name[:40])
    else:
        check("hidden-API device still resolves", True, "no such device to test")


def test_loopback() -> None:
    """Tone detection maths, and the cable-vs-router distinction."""
    print("\n[loopback]")
    from voice2tts import cable, loopback

    rate = loopback.RATE
    t = np.arange(rate, dtype=np.float32) / rate

    tone = 0.25 * np.sin(2 * np.pi * loopback.TONE_HZ * t)
    silence = np.zeros(rate, dtype=np.float32)
    noise = (np.random.randn(rate) * 0.05).astype(np.float32)
    other = 0.25 * np.sin(2 * np.pi * 300.0 * t)

    tone_db = loopback._tone_level_db(tone, rate, loopback.TONE_HZ)
    silence_db = loopback._tone_level_db(silence, rate, loopback.TONE_HZ)
    noise_db = loopback._tone_level_db(noise, rate, loopback.TONE_HZ)
    other_db = loopback._tone_level_db(other, rate, loopback.TONE_HZ)

    check("tone detected loudly", tone_db > -20, f"{tone_db:.0f} dB")
    check("silence reads as nothing", silence_db < -100, f"{silence_db:.0f} dB")
    check("broadband noise does not look like the tone",
          tone_db - noise_db > loopback.DETECT_MARGIN_DB,
          f"tone {tone_db:.0f} vs noise {noise_db:.0f} dB")
    check("a different frequency is rejected",
          tone_db - other_db > loopback.DETECT_MARGIN_DB,
          f"997Hz {tone_db:.0f} vs 300Hz {other_db:.0f} dB")

    # Quiet speech must not be mistaken for the test signal.
    speech = load_sample_16k()
    if speech is not None:
        import soxr

        resampled = soxr.resample(speech, 16000, rate).astype(np.float32)
        speech_db = loopback._tone_level_db(resampled, rate, loopback.TONE_HZ)
        check("speech is not mistaken for the tone",
              tone_db - speech_db > loopback.DETECT_MARGIN_DB,
              f"tone {tone_db:.0f} vs speech {speech_db:.0f} dB")

    check("missing device reported, not raised",
          not loopback.verify("no-such-output", "no-such-input").ok)

    # Cable vs router: a router's endpoints are mixer ports, so its pairing can
    # never be reported as certain and its caveat must mention the application.
    kinds = {p: k for _, p, _, k in cable._PRODUCTS}
    check("VB-CABLE is a hardwired cable", kinds["VB-CABLE"] == cable.CABLE)
    check("Matrix is a router", kinds["VB-Audio Matrix"] == cable.ROUTER)
    check("VoiceMeeter is a router", kinds["VoiceMeeter"] == cable.ROUTER)

    router = cable.CableInfo(product="VB-Audio Matrix", output_name="VBMatrix In 1",
                             input_name="VBMatrix Out 1", channel=1, certain=True,
                             kind=cable.ROUTER)
    check("router pairing carries a caveat", bool(router.caveat))
    check("caveat names the application", "VB-Audio Matrix" in router.caveat)

    plain = cable.CableInfo(product="VB-CABLE", output_name="CABLE Input",
                            input_name="CABLE Output", certain=True)
    check("a real cable has no caveat", not plain.caveat)
    check("a real cable needs no application running", plain.app_running)

    found = cable.detect()
    if found is not None and found.is_router:
        check("router state reported", isinstance(found.app_running, bool),
              f"{found.product} running={found.app_running}")


def test_platform() -> None:
    """Windows integration bits that do not need audio hardware."""
    print("\n[platform]")
    from voice2tts import platform_win
    from voice2tts.diagnostics import diagnostics

    guard = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("first instance acquires the mutex", guard.acquire())
    second = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("second instance is refused", not second.acquire())
    check("second instance reports why", second.already_running)
    second.release()
    guard.release()

    after = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("mutex is reusable once released", after.acquire())
    after.release()

    check("run_at_login readable", isinstance(platform_win.run_at_login(), bool))
    # Source checkouts have no exe to register, so this must decline rather than
    # write a broken registry entry pointing at python.exe.
    check("run_at_login refuses from source", not platform_win.set_run_at_login(True))

    report = diagnostics(load_config(), None)
    check("diagnostics report generated", len(report.splitlines()) > 20,
          f"{len(report.splitlines())} lines")
    user = os.environ.get("USERNAME") or ""
    if len(user) > 2:
        check("diagnostics redacts the user name", user not in report)
    check("diagnostics has no transcript text", "transcript" not in report.lower()
          or "last transcript" not in report.lower())


def test_language_guard() -> None:
    print("\n[language guard]")
    from voice2tts import voices

    check("english voice + english model is fine",
          not voices.language_mismatch("en_US-amy-medium", "base.en"))
    warning = voices.language_mismatch("de_DE-thorsten-medium", "small.en")
    check("german voice + english model warns", bool(warning))
    check("warning names both sides",
          "de_DE-thorsten-medium" in warning and "small.en" in warning)
    check("multilingual model accepts any voice",
          not voices.language_mismatch("de_DE-thorsten-medium", "large-v3"))
    check("blank inputs do not warn", not voices.language_mismatch("", "base.en"))


def main() -> int:
    setup_logging("WARNING")
    no_audio = "--no-audio" in sys.argv
    print("Voice2TTS self-test" + (" (no audio hardware)" if no_audio else ""))
    test_config()
    test_hotkey()
    test_hotkey_manager()
    test_clipboard()
    test_substitutions()
    if not no_audio:
        test_device_recovery()
    test_theme()
    test_studio_gate()
    test_release_is_gated()
    test_winget_manifests()
    test_profiles()
    test_history_and_review()
    test_vad()
    test_stt()
    test_device_lists()
    if not no_audio:
        test_loopback()
    test_platform()
    test_language_guard()
    if not no_audio:
        test_tts_and_sink()
    test_packaging_bits()
    test_updates()
    if "--no-network" not in sys.argv:
        test_network()
    if not no_audio and "--no-e2e" not in sys.argv:
        test_pipeline_end_to_end()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
