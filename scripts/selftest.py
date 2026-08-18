"""Offline self-test: exercises config, VAD, STT and the output sink.

Opens real output streams but writes silence, so it is safe to run with headphones
on. Does not touch the microphone.

    python scripts/selftest.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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


def speech_like(seconds: float, rate: int, room_tone: float = 0.001) -> np.ndarray:
    """A signal shaped like speech: word-length bursts separated by gaps.

    A continuous tone is not a stand-in for speech. The quality checks work by
    comparing the loud and quiet parts of a clip, and a tone has neither -- it
    reads as both "noisy" (never quiet) and "silent" (never above its own level),
    which tells us nothing about whether the checks are right.
    """
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    voiced = (np.sin(2 * np.pi * 130 * t)
              + 0.5 * np.sin(2 * np.pi * 260 * t)
              + 0.25 * np.sin(2 * np.pi * 390 * t))
    voiced *= 0.3 / np.abs(voiced).max()

    envelope = np.zeros_like(t)
    pos = 0.0
    while pos < seconds:                      # ~0.35 s words, ~0.12 s gaps
        envelope[int(pos * rate):int(min(pos + 0.35, seconds) * rate)] = 1.0
        pos += 0.47
    ramp = np.hanning(max(2, int(rate * 0.02)))
    envelope = np.convolve(envelope, ramp / ramp.sum(), mode="same")

    tone = np.random.default_rng(0).standard_normal(len(t)) * room_tone
    return (voiced * envelope + tone).astype(np.float32)


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

    # Two rule sets, because translation goes between them. Source rules fix
    # what the recogniser misheard and must run while the text is still in the
    # language that was spoken; target rules fix what the VOICE says badly,
    # which is a property of the output language.
    cfg = Config()
    cfg.text.substitutions = [SubstitutionRule("aiden", "Aidan")]
    cfg.text.target_substitutions = [SubstitutionRule("Aidan", "AY-dan")]
    p = Pipeline(cfg)
    check("both rule sets compile", p.apply_text_changes() == 2)
    check("source rules fix what was misheard",
          p.substituter.apply("tell aiden") == "tell Aidan")
    check("target rules fix what is said badly",
          p.target_substituter.apply("tell Aidan") == "tell AY-dan")
    check("with no translation between them they run back to back",
          p.target_substituter.apply(p.substituter.apply("tell aiden"))
          == "tell AY-dan")

    # An existing configuration has one list and must behave exactly as before.
    legacy = Config()
    legacy.text.substitutions = [SubstitutionRule("aiden", "Aidan")]
    old = Pipeline(legacy)
    check("a config with no target rules is unchanged",
          old.apply_text_changes() == 1
          and old.target_substituter.apply("tell Aidan") == "tell Aidan",
          "the target list is empty, so it is a passthrough")

    cfg.text.substitutions_enabled = False
    check("the switch turns off both sets", p.apply_text_changes() == 0)
    check("and neither rewrites anything",
          p.substituter.apply("tell aiden") == "tell aiden"
          and p.target_substituter.apply("tell Aidan") == "tell Aidan")


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


def test_prompts() -> None:
    print("\n[prompts]")
    from voice2tts import prompts

    corpus = prompts.load()
    check("corpus loads", len(corpus) > 100, f"{len(corpus)} prompts")
    check("prompts have text", all(p.text for p in corpus))
    check("prompts have keys", len({p.key for p in corpus}) == len(corpus),
          "keys unique")

    parsed = prompts.parse('( arctic_a0001 "Author of the danger trail." )')
    check("festival format parsed",
          len(parsed) == 1 and parsed[0].key == "arctic_a0001"
          and parsed[0].text == "Author of the danger trail.", str(parsed))
    plain = prompts.parse("Just a plain sentence.\n; a comment\n\nAnother one.")
    check("plain text accepted, comments skipped", len(plain) == 2, str(plain))

    p = prompts.Prompt("k", "one two three four five six")
    check("word count", p.words == 6)
    check("estimate scales with rate",
          p.estimated_seconds(100) > p.estimated_seconds(200))

    # The estimate must be driven by measured speech, not a sentence count:
    # a slow reader needs far fewer sentences for the same audio.
    check("rate defaults until there is data",
          prompts.measured_wpm(10, 5.0) == prompts.DEFAULT_WPM)
    fast = prompts.measured_wpm(600, 120.0)
    slow = prompts.measured_wpm(200, 120.0)
    check("measured rate reflects the speaker", fast > slow, f"{fast:.0f} vs {slow:.0f}")
    check("absurd rates are clamped",
          60.0 <= prompts.measured_wpm(100000, 60.0) <= 320.0)

    count_slow, secs_slow = prompts.remaining_estimate(corpus, set(), 600.0, wpm=90)
    count_fast, secs_fast = prompts.remaining_estimate(corpus, set(), 600.0, wpm=220)
    check("a slower reader needs fewer sentences", count_slow < count_fast,
          f"{count_slow} vs {count_fast} for 10 minutes")
    check("estimate reaches the target", secs_slow >= 600.0 and secs_fast >= 600.0)
    check("nothing needed when the target is met",
          prompts.remaining_estimate(corpus, set(), 0.0) == (0, 0.0))

    done = {p.key for p in corpus[:50]}
    offered = prompts.next_prompts(corpus, done, 600.0)
    check("recorded prompts are not offered again",
          not any(p.key in done for p in offered),
          f"{len(offered)} offered, {len(done)} already done")
    check("the next prompt follows what was recorded",
          offered and offered[0].key == corpus[50].key,
          offered[0].key if offered else "none")
    # Skipping prompts must not shorten the session: the target is an amount of
    # audio, so the same time still has to be read either way.
    _, secs_done = prompts.remaining_estimate(corpus, done, 600.0)
    check("skipping still reaches the target", secs_done >= 600.0, f"{secs_done:.0f}s")

    # Reading in file order means the first session is all one author's prose.
    order_a = [p.key for p in prompts.shuffled(corpus, seed=1)[:10]]
    order_b = [p.key for p in prompts.shuffled(corpus, seed=1)[:10]]
    check("shuffle is deterministic", order_a == order_b)
    check("shuffle actually reorders",
          order_a != [p.key for p in corpus[:10]])


def test_dataset() -> None:
    print("\n[dataset]")
    import tempfile

    from voice2tts import dataset

    rate = dataset.TARGET_RATE
    speech = speech_like(2.0, rate)

    stats, issues = dataset.analyse(speech, rate, "one two three four five")
    check("a clean clip passes", not issues, str(issues))
    check("speech is measured as mostly speech",
          stats["speech_fraction"] > 0.5, f"{stats['speech_fraction']:.2f}")

    _s, issues = dataset.analyse(speech * 0.001, rate)
    check("silence is rejected", any("quiet" in i for i in issues), str(issues))

    _s, issues = dataset.analyse(np.ones(rate, dtype=np.float32), rate)
    check("clipping is caught", any("clipping" in i for i in issues), str(issues))

    _s, issues = dataset.analyse(speech[:1000], rate)
    check("a too-short clip is caught", any("short" in i for i in issues), str(issues))

    noisy = speech + (np.random.randn(len(speech)) * 0.08).astype(np.float32)
    _s, issues = dataset.analyse(noisy, rate)
    check("background noise is caught", any("noisy" in i for i in issues), str(issues))

    mostly_silence = np.concatenate(
        [np.zeros(int(rate * 3), dtype=np.float32), speech_like(1.0, rate)])
    _s, issues = dataset.analyse(mostly_silence, rate)
    check("mostly-silence is caught", any("silence" in i for i in issues), str(issues))

    # Quiet-but-clean must pass where an absolute noise floor would have failed it,
    # and loud-but-hissy must fail where an absolute floor would have passed it.
    _s, issues = dataset.analyse(speech * 0.15, rate)
    check("a quiet clean clip is accepted", not issues, str(issues))
    loud_hiss = speech_like(2.0, rate, room_tone=0.05) * 3.0
    _s, issues = dataset.analyse(np.clip(loud_hiss, -0.99, 0.99), rate)
    check("a loud hissy clip is rejected", any("noisy" in i for i in issues),
          str(issues))

    _s, issues = dataset.analyse(speech, rate, " ".join(["word"] * 40))
    check("a clip far shorter than its prompt is caught",
          any("cut off" in i for i in issues), str(issues))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "voice"
        session = dataset.RecordingSession("Test Voice", root=root,
                                           target_minutes=1.0)
        session.add("p1", "First sentence here.", speech, rate)
        session.add("p2", "Second sentence here.", speech, rate)
        check("clips stored", len(session.usable) == 2, session.summary())
        check("audio written", len(list(session.audio_dir.glob("*.wav"))) == 2)
        check("duration accumulates", session.seconds > 3.5, f"{session.seconds:.1f}s")

        # Recording a prompt again should replace it, not train on it twice.
        session.add("p1", "First sentence here.", speech, rate)
        check("re-recording replaces the take", len(session.clips) == 2,
              f"{len(session.clips)} clips")

        session.add("bad", "A rejected clip.", speech * 0.0005, rate)
        check("unusable clips are kept but excluded",
              len(session.clips) == 3 and len(session.usable) == 2,
              session.summary())

        # Resuming must survive closing the app mid-session.
        reloaded = dataset.RecordingSession.load(root)
        check("session reloads", len(reloaded.clips) == 3
              and abs(reloaded.seconds - session.seconds) < 0.01)
        check("target survives the round-trip",
              abs(reloaded.target_seconds - 60.0) < 0.01)

        csv_path = session.prepare()
        rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
        check("metadata written", len(rows) == 2, f"{len(rows)} rows")
        check("pipe-delimited wav|text",
              all(r.count("|") == 1 and r.startswith("wav/") for r in rows),
              rows[0] if rows else "")
        check("only usable clips exported",
              "bad" not in csv_path.read_text(encoding="utf-8"))
        exported = list((csv_path.parent / "wav").glob("*.wav"))
        check("audio copied beside the csv", len(exported) == 2, str(len(exported)))

        _audio, out_rate = dataset.read_wav(exported[0])
        check("exported audio is at the training rate", out_rate == dataset.TARGET_RATE,
              f"{out_rate} Hz")

        # Imported files take the same path and the same checks.
        source = Path(tmp) / "imported.wav"
        dataset.write_wav(source, speech, rate)
        clip = session.import_file(source, "An imported line.")
        check("import works", clip.ok and clip.source == "imported", str(clip.issues))

        empty = dataset.RecordingSession("Empty", root=Path(tmp) / "empty")
        try:
            empty.prepare()
            check("preparing an empty session is refused", False, "no error")
        except RuntimeError:
            check("preparing an empty session is refused", True)

    # The checks above all use a signal I built to satisfy them, which proves only
    # that they are self-consistent. Real speech is the thing they have to accept.
    real = load_sample_16k()
    if real is None:
        print("  SKIP  real speech passes the quality checks (no sample available)")
    else:
        import soxr

        resampled = soxr.resample(real, 16000, rate, quality="VHQ").astype(np.float32)
        stats, issues = dataset.analyse(resampled, rate)
        check("real speech passes the quality checks", not issues,
              f"{issues} {stats}")


def test_dsp() -> None:
    """The designer's effects chain, measured rather than assumed.

    Every macro is checked for the direction it claims. That is not pedantry:
    `size` shipped inverted in the first draft -- "larger" made the voice
    brighter and shorter -- and nothing but a measurement catches that.
    """
    print("\n[dsp]")
    from voice2tts import dsp

    rate = 22050
    real = load_sample_16k()
    if real is None:
        print("  SKIP  dsp checks (no speech sample available)")
        return
    import soxr

    audio = soxr.resample(real, 16000, rate, quality="VHQ").astype(np.float32)

    def centroid(x: np.ndarray) -> float:
        frame = 1024
        usable = len(x) // frame * frame
        if not usable:
            return 0.0
        frames = x[:usable].reshape(-1, frame) * np.hanning(frame)
        mag = np.abs(np.fft.rfft(frames, axis=1))
        freqs = np.fft.rfftfreq(frame, 1.0 / rate)
        total = mag.sum(axis=1)
        live = total > 1e-9
        return float(((mag * freqs).sum(axis=1)[live] / total[live]).mean())

    def crest(x: np.ndarray) -> float:
        rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
        return float(np.abs(x).max()) / rms if rms > 0 else 0.0

    base_centroid, base_crest = centroid(audio), crest(audio)

    neutral = dsp.apply(audio, rate, dsp.Design())
    check("a neutral design changes nothing at all",
          np.array_equal(neutral, audio))
    check("neutral is recognised as neutral", dsp.Design().is_neutral)
    check("any macro makes it non-neutral", not dsp.Design(warmth=0.1).is_neutral)

    # SIZE. A larger speaker is lower and longer, not higher and shorter.
    big = dsp.apply(audio, rate, dsp.Design(size=1.0))
    small = dsp.apply(audio, rate, dsp.Design(size=-1.0))
    check("a larger size lowers the voice", centroid(big) < base_centroid,
          f"{centroid(big):.0f} < {base_centroid:.0f} Hz")
    check("a smaller size raises it", centroid(small) > base_centroid,
          f"{centroid(small):.0f} > {base_centroid:.0f} Hz")
    # ...and takes just as long to say. Resampling alone would make a deeper
    # voice a slower one; the stretch puts the timing back.
    for label, shaped in (("larger", big), ("smaller", small)):
        drift = abs(len(shaped) - len(audio)) / len(audio)
        check(f"a {label} voice still takes the same time to speak", drift < 0.02,
              f"{drift * 100:.1f}% drift ({len(shaped)} vs {len(audio)})")

    ratio = dsp.Design(size=1.0).size_ratio
    check("a larger voice resamples downward", ratio < 1.0, f"{ratio:.4f}")
    check("size stays within a fifth either way",
          0.79 < ratio < 1.0 and 1.0 < dsp.Design(size=-1.0).size_ratio < 1.27,
          f"{ratio:.3f} / {dsp.Design(size=-1.0).size_ratio:.3f}")

    # Pitch is the thing size is actually for, so measure it rather than infer
    # it from the spectral centroid.
    def pitch(x: np.ndarray) -> float:
        frame, hop = 2048, 512
        low, high = int(rate / 350), int(rate / 70)
        found = []
        for start in range(0, len(x) - frame, hop):
            seg = x[start:start + frame].astype(np.float64)
            if np.sqrt((seg ** 2).mean()) < 0.02:
                continue
            seg = seg - seg.mean()
            auto = np.correlate(seg, seg, mode="full")[frame - 1:]
            if auto[0] <= 0:
                continue
            auto = auto / auto[0]
            band = auto[low:high]
            if not len(band):
                continue
            peak = int(np.argmax(band)) + low
            if auto[peak] > 0.3:
                found.append(rate / peak)
        return float(np.median(found)) if found else 0.0

    base_pitch = pitch(audio)
    check("the sample has a measurable pitch", 60 < base_pitch < 350,
          f"{base_pitch:.0f} Hz")
    for label, shaped, expect in (("larger", big, base_pitch * ratio),
                                  ("smaller", small,
                                   base_pitch * dsp.Design(size=-1.0).size_ratio)):
        got = pitch(shaped)
        check(f"a {label} voice shifts pitch by the resampling ratio",
              abs(got - expect) / expect < 0.05,
              f"{got:.0f} Hz, expected {expect:.0f}")

    # The stretcher on its own.
    stretched = dsp.time_stretch(audio, rate, 1.25)
    check("time stretching lengthens by the factor",
          abs(len(stretched) / len(audio) - 1.25) < 0.05,
          f"x{len(stretched) / len(audio):.3f}")
    check("and leaves the pitch alone",
          abs(pitch(stretched) - base_pitch) / base_pitch < 0.05,
          f"{pitch(stretched):.0f} vs {base_pitch:.0f} Hz")
    check("a factor of one is a passthrough",
          np.array_equal(dsp.time_stretch(audio, rate, 1.0), audio))
    check("a fragment too short to splice still comes back",
          len(dsp.time_stretch(np.zeros(64, dtype=np.float32), rate, 1.2)) > 0)

    # TONE.
    warm = dsp.apply(audio, rate, dsp.Design(warmth=1.0))
    bright = dsp.apply(audio, rate, dsp.Design(brightness=1.0))
    check("warmth darkens", centroid(warm) < base_centroid,
          f"{centroid(warm):.0f} < {base_centroid:.0f} Hz")
    check("brightness brightens", centroid(bright) > base_centroid,
          f"{centroid(bright):.0f} > {base_centroid:.0f} Hz")
    check("tone does not change the length",
          len(warm) == len(bright) == len(audio))

    # BREATH. Some lift in the top end, but breath rather than hiss: the first
    # attempt took the centroid from 2330 Hz to 6300, which is not a voice.
    breathy = dsp.apply(audio, rate, dsp.Design(breathiness=1.0))
    lift = centroid(breathy) / base_centroid
    check("breathiness adds air", lift > 1.15, f"x{lift:.2f}")
    check("but does not drown the voice in hiss", lift < 2.0, f"x{lift:.2f}")

    # DYNAMICS. Evenness means a lower crest factor. An RMS detector actually
    # made this worse, which is why the compressor follows peaks.
    even = dsp.apply(audio, rate, dsp.Design(dynamics=1.0))
    check("dynamics evens the level out", crest(even) < base_crest * 0.8,
          f"crest {crest(even):.2f} from {base_crest:.2f}")
    check("and does not just turn it down",
          abs(float(np.abs(even).max()) - float(np.abs(audio).max())) < 0.05,
          f"peak {float(np.abs(even).max()):.3f} vs {float(np.abs(audio).max()):.3f}")

    # SPACE adds a tail.
    roomy = dsp.apply(audio, rate, dsp.Design(space=1.0))
    check("space adds a tail", len(roomy) > len(audio),
          f"+{len(roomy) - len(audio)} samples")

    # Nothing in the chain may leave the signal outside full scale.
    hot = dsp.apply(audio * 0.99, rate,
                    dsp.Design(size=0.5, warmth=1.0, brightness=1.0,
                               breathiness=1.0, dynamics=1.0, space=1.0))
    check("the chain never clips", float(np.abs(hot).max()) <= 1.0,
          f"peak {float(np.abs(hot).max()):.4f}")

    # This runs on the live call path, so it has to be cheap.
    full = dsp.Design(size=0.5, warmth=0.4, brightness=0.3, breathiness=0.3,
                      dynamics=0.5, space=0.3)
    start = time.perf_counter()
    dsp.apply(audio, rate, full)
    cost = (time.perf_counter() - start) / (len(audio) / rate)
    check("the whole chain runs far faster than realtime", cost < 0.1,
          f"{cost * 100:.1f}% of realtime")

    check("macros are clamped to their range",
          dsp.Design(warmth=5.0).clamped().warmth == 1.0
          and dsp.Design(size=-9.0).clamped().size == -1.0)
    check("empty audio survives the chain",
          len(dsp.apply(np.zeros(0, dtype=np.float32), rate,
                        dsp.Design(space=1.0))) == 0)


def test_designer() -> None:
    """Blending, projection and baking. Offline: no model needed."""
    print("\n[designer]")
    from voice2tts import designer

    rng = np.random.default_rng(7)
    table = rng.standard_normal((40, 16)).astype(np.float32)

    mid = designer.blend(table, {0: 1.0, 1: 1.0})
    check("an even blend is the midpoint",
          np.allclose(mid, (table[0] + table[1]) / 2, atol=1e-6))
    check("weights are normalised, not summed",
          np.allclose(designer.blend(table, {0: 5.0, 1: 5.0}), mid, atol=1e-6),
          "doubling every weight must not move the result")
    check("a single speaker blends to itself",
          np.allclose(designer.blend(table, {3: 0.7}), table[3], atol=1e-6))
    check("blend width matches the table", designer.blend(table, {0: 1.0}).size == 16)

    for weights, why in (({}, "no speakers"), ({0: 0.0}, "all-zero weights")):
        try:
            designer.blend(table, weights)
            check(f"a blend with {why} is refused", False, "no error")
        except ValueError:
            check(f"a blend with {why} is refused", True)
    try:
        designer.blend(table, {99: 1.0})
        check("a speaker outside the model is refused", False, "no error")
    except IndexError:
        check("a speaker outside the model is refused", True)

    coords = designer.project(table)
    check("projection is 2D, one point per speaker", coords.shape == (40, 2),
          str(coords.shape))
    check("projection fits the unit square", float(np.abs(coords).max()) <= 1.0001,
          f"{float(np.abs(coords).max()):.3f}")
    check("projection is deterministic",
          np.array_equal(coords, designer.project(table)),
          "the same model must always give the same map")
    check("projection spreads the speakers out",
          float(coords[:, 0].std()) > 0.05 and float(coords[:, 1].std()) > 0.05,
          f"sd {coords[:, 0].std():.3f}, {coords[:, 1].std():.3f}")

    # Clicking exactly on a speaker gives that speaker, not a near-miss blend
    # weighted by a division by almost zero.
    x, y = float(coords[5][0]), float(coords[5][1])
    vector, weights = designer.from_map(table, coords, x, y)
    check("clicking a speaker selects it exactly", weights == {5: 1.0}, str(weights))
    check("and returns that speaker's own vector",
          np.allclose(vector, table[5], atol=1e-6))

    between = (coords[5] + coords[6]) / 2
    _v, weights = designer.from_map(table, coords, float(between[0]),
                                    float(between[1]), count=3)
    check("clicking between speakers blends several", len(weights) == 3, str(weights))
    check("nearer speakers weigh more",
          max(weights.values()) > min(weights.values()))

    found = designer.nearest(table, table[9], count=3)
    check("nearest finds the speaker itself first",
          found[0][0] == 9 and found[0][1] < 1e-5, str(found[:1]))
    check("and returns them in order",
          [d for _i, d in found] == sorted(d for _i, d in found))

    recipe = designer.Recipe(base_voice="x", weights={"a": 3.0, "b": 1.0})
    check("a recipe normalises to one",
          abs(sum(recipe.normalised().values()) - 1.0) < 1e-9)
    check("recipe keeps the proportions",
          abs(recipe.normalised()["a"] - 0.75) < 1e-9)
    check("an all-zero recipe is empty rather than a division by zero",
          designer.Recipe(weights={"a": 0.0}).is_empty)


def _toy_multispeaker(path: Path, speakers: int = 4, width: int = 8) -> np.ndarray:
    """A minimal ONNX model shaped like the part of VITS the designer touches.

    Real Piper voices are 70-120 MB and only downloaded on demand, so baking
    could not otherwise be tested anywhere the models are absent -- which
    includes CI. This has the same three features that matter: an `emb_g` table,
    a Gather indexing it with `sid`, and something downstream consuming the
    result.
    """
    import json

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    table = (np.arange(speakers * width, dtype=np.float32)
             .reshape(speakers, width) / 10.0)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gather", ["emb_g.weight", "sid"], ["gathered"],
                             name="lookup", axis=0),
            helper.make_node("Identity", ["gathered"], ["output"], name="out"),
        ],
        name="toy",
        inputs=[helper.make_tensor_value_info("sid", TensorProto.INT64, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                               [1, width])],
        initializer=[numpy_helper.from_array(table, name="emb_g.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    path.with_suffix(".onnx.json").write_text(
        json.dumps({"num_speakers": speakers, "sample_rate": 22050,
                    "speaker_id_map": {f"p{i}": i for i in range(speakers)}}),
        encoding="utf-8")
    return table


def test_baking() -> None:
    """Freezing a blend into a model, which is what makes a designed voice."""
    print("\n[baking]")
    import json
    import tempfile

    import onnxruntime as ort

    from voice2tts import designer

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.onnx"
        table = _toy_multispeaker(base)

        read_back = designer.speaker_table(base)
        check("the speaker table is readable from a model",
              np.allclose(read_back, table), str(read_back.shape))
        check("speaker names come from the config",
              designer.speaker_names(base.with_suffix(".onnx.json"), 4)
              == ["p0", "p1", "p2", "p3"])
        check("multi-speaker models are recognised",
              designer.is_multi_speaker(base.with_suffix(".onnx.json")))

        vector = designer.blend(table, {0: 1.0, 2: 1.0})
        baked = designer.bake(base, base.with_suffix(".onnx.json"), vector,
                              Path(tmp) / "designed.onnx", name="Test")

        check("baked model exists", baked.is_file())
        check("its config exists too", baked.with_suffix(".onnx.json").is_file(),
              "a voice without its config loads and then speaks nonsense")
        config = json.loads(baked.with_suffix(".onnx.json").read_text(encoding="utf-8"))
        check("the baked voice reports one speaker", config["num_speakers"] == 1,
              str(config["num_speakers"]))
        check("and carries no speaker map", config["speaker_id_map"] == {})

        session = ort.InferenceSession(str(baked),
                                       providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        check("sid is no longer an input", "sid" not in names, str(names))
        check("nothing replaced it", names == [], str(names))

        out = session.run(None, {})[0].reshape(-1)
        check("the baked vector is what comes out",
              np.allclose(out, vector, atol=1e-6), f"{out[:3]} vs {vector[:3]}")

        # Baking a speaker must give that speaker back, or every designed voice
        # is subtly not the thing that was auditioned.
        for speaker in range(4):
            solo = designer.bake(base, base.with_suffix(".onnx.json"),
                                 table[speaker], Path(tmp) / f"s{speaker}.onnx")
            got = ort.InferenceSession(
                str(solo), providers=["CPUExecutionProvider"]).run(None, {})[0]
            if not np.allclose(got.reshape(-1), table[speaker], atol=1e-6):
                check("baking a speaker reproduces that speaker", False,
                      f"speaker {speaker}")
                break
        else:
            check("baking a speaker reproduces that speaker", True,
                  "all 4 exact")

        # The table is dead weight once a single vector is frozen in.
        check("the speaker table is dropped from the baked model",
              baked.stat().st_size < base.stat().st_size,
              f"{baked.stat().st_size} < {base.stat().st_size} bytes")

        try:
            designer.bake(base, base.with_suffix(".onnx.json"),
                          np.zeros(3, dtype=np.float32), Path(tmp) / "bad.onnx")
            check("a wrongly sized embedding is refused", False, "no error")
        except ValueError:
            check("a wrongly sized embedding is refused", True)

        # Single-speaker models have no space to move through.
        plain = Path(tmp) / "plain.onnx"
        plain.write_bytes(baked.read_bytes())
        try:
            designer.speaker_table(plain)
            check("a single-speaker model is rejected", False, "no error")
        except designer.NotMultiSpeaker:
            check("a single-speaker model is rejected", True)

        # The effects sidecar rides beside the model, not inside Piper's config.
        from voice2tts.dsp import Design

        check("a plain voice has no design", designer.read_design(baked) is None)
        designer.write_design(baked, Design(warmth=0.4, size=-0.2), "Test")
        loaded = designer.read_design(baked)
        check("the design sidecar round-trips",
              loaded and abs(loaded.warmth - 0.4) < 1e-4
              and abs(loaded.size + 0.2) < 1e-4, str(loaded))
        check("the sidecar sits outside Piper's config",
              "warmth" not in baked.with_suffix(".onnx.json").read_text(
                  encoding="utf-8"))

        designer.design_path(baked).write_text("{ broken", encoding="utf-8")
        check("a corrupt sidecar is ignored rather than fatal",
              designer.read_design(baked) is None,
              "a voice that speaks without effects beats one that will not speak")

        check("removing a designed voice takes its files with it",
              designer.remove_designed(baked) and not baked.exists()
              and not baked.with_suffix(".onnx.json").exists())

    # A voice is its model plus whatever the Studio wrote beside it. Nothing
    # owned that list, so deleting a voice left its effects sidecar behind and
    # the NEXT voice of the same name inherited it -- silently, with nothing in
    # the interface to explain why a clean voice sounded processed.
    with tempfile.TemporaryDirectory() as tmp:
        from voice2tts import voices as voices_mod
        from voice2tts.dsp import Design

        base = Path(tmp) / "base.onnx"
        table = _toy_multispeaker(base)
        dest = Path(tmp) / "narrator.onnx"

        designer.bake(base, base.with_suffix(".onnx.json"), table[0], dest)
        designer.write_design(dest, Design(warmth=0.9), "narrator")
        check("a designed voice has an effects sidecar",
              designer.read_design(dest) is not None)

        # Re-baking the same name with no effects must not keep the old ones.
        designer.bake(base, base.with_suffix(".onnx.json"), table[1], dest)
        check("re-baking a name clears the previous effects",
              designer.read_design(dest) is None,
              "otherwise a neutral voice keeps the old warmth")

        designer.write_design(dest, Design(space=0.5), "narrator")
        listed = voices_mod.voice_files(dest)
        check("the voice file list covers model, config and sidecars",
              len(listed) == 5 and listed[0] == dest,
              str([p.name for p in listed]))
        cleared = voices_mod.clear_sidecars(dest)
        check("clearing removes the sidecars but not the model",
              dest.exists() and designer.read_design(dest) is None
              and len(cleared) >= 1, str([p.name for p in cleared]))

        # And an exported trained voice lands on the same guarantee.
        designer.write_design(dest, Design(warmth=0.9), "narrator")
        voices_mod.clear_sidecars(dest)
        check("a trained export would not inherit either",
              designer.read_design(dest) is None)


def test_v2tvoice() -> None:
    """The recipe format: round trip, and refusing what it cannot read."""
    print("\n[v2tvoice]")
    import tempfile

    from voice2tts import v2tvoice
    from voice2tts.dsp import Design

    voice = v2tvoice.DesignedVoice(
        name="Narrator", base_voice="en_GB-vctk-medium",
        speakers={"p225": 0.6, "p243": 0.4},
        design=Design(size=0.2, warmth=0.35),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = voice.save(Path(tmp) / "narrator")
        check("saved with the right suffix", path.suffix == ".v2tvoice", path.name)
        # The whole point of a recipe is that it is small enough to paste.
        check("a recipe is tiny", path.stat().st_size < 1024,
              f"{path.stat().st_size} bytes")
        text = path.read_text(encoding="utf-8")
        check("it is readable text naming the base voice",
              "en_GB-vctk-medium" in text and "p225" in text)
        check("it does not embed the model",
              "onnx" not in text.lower() and path.stat().st_size < 4096)

        back = v2tvoice.load(path)
        check("round trip keeps the name", back.name == "Narrator")
        check("round trip keeps the speakers", back.speakers == voice.speakers,
              str(back.speakers))

        # Weights are rounded so the file stays readable. That is fine, but the
        # rounding has to be far below anything audible.
        precise = v2tvoice.DesignedVoice(
            base_voice="b", speakers={"a": 0.5123456789, "b": 0.4876543211})
        again = v2tvoice.load(precise.save(Path(tmp) / "precise"))
        drift = max(abs(again.speakers[k] - v) for k, v in precise.speakers.items())
        check("rounding a recipe stays well below audible", drift < 1e-6,
              f"largest drift {drift:.2e}")
        check("round trip keeps the macros",
              abs(back.design.size - 0.2) < 1e-4
              and abs(back.design.warmth - 0.35) < 1e-4)

        # A plain blend should not write six zeroes it does not need.
        plain = v2tvoice.DesignedVoice(base_voice="b", speakers={"0": 1.0})
        plain_path = plain.save(Path(tmp) / "plain")
        check("a design with no effects writes no design section",
              "[design]" not in plain_path.read_text(encoding="utf-8"))
        check("and still loads", v2tvoice.load(plain_path).design.is_neutral)

    for data, why in (
        ({"name": "x", "base_voice": "b", "speakers": {"0": 1}}, "no schema"),
        ({"schema": 99, "base_voice": "b", "speakers": {"0": 1}}, "a future schema"),
        ({"schema": 1, "speakers": {"0": 1}}, "no base voice"),
        ({"schema": 1, "base_voice": "b"}, "no speakers"),
        ({"schema": 1, "base_voice": "b", "speakers": {}}, "an empty speaker list"),
        ({"schema": 1, "base_voice": "b", "speakers": {"0": 0}}, "all-zero weights"),
        ({"schema": 1, "base_voice": "b", "speakers": {"0": "loud"}},
         "a non-numeric weight"),
    ):
        try:
            v2tvoice.from_dict(data)
            check(f"a recipe with {why} is refused", False, "no error")
        except v2tvoice.UnreadableVoice:
            check(f"a recipe with {why} is refused", True)

    # These files are meant to be hand-edited, so a typo has to come back as
    # "this recipe is wrong" rather than as a raw ValueError.
    try:
        v2tvoice.from_dict({"schema": 1, "base_voice": "b", "speakers": {"0": 1.0},
                            "design": {"warmth": "hot"}})
        check("a non-numeric macro is refused as a recipe error", False, "no error")
    except v2tvoice.UnreadableVoice:
        check("a non-numeric macro is refused as a recipe error", True)
    except ValueError as exc:
        check("a non-numeric macro is refused as a recipe error", False,
              f"leaked a raw {type(exc).__name__}")

    # Unknown macros from a newer build are ignored, not fatal.
    forward = v2tvoice.from_dict({
        "schema": 1, "base_voice": "b", "speakers": {"0": 1.0},
        "design": {"warmth": 0.5, "rasp": 0.9},
    })
    check("an unknown macro does not stop the voice loading",
          abs(forward.design.warmth - 0.5) < 1e-6)

    names = ["p225", "p226", "p243"]
    resolved = v2tvoice.resolve_speakers(
        v2tvoice.DesignedVoice(base_voice="b", speakers={"p225": 0.6, "p243": 0.4}),
        names)
    check("speaker labels map to indices", resolved == {0: 0.6, 2: 0.4},
          str(resolved))
    numeric = v2tvoice.resolve_speakers(
        v2tvoice.DesignedVoice(base_voice="b", speakers={"1": 1.0}), names)
    check("bare indices still work", numeric == {1: 1.0}, str(numeric))
    try:
        v2tvoice.resolve_speakers(
            v2tvoice.DesignedVoice(base_voice="b", speakers={"nobody": 1.0}), names)
        check("a speaker the model lacks is refused, not dropped", False, "no error")
    except v2tvoice.UnreadableVoice:
        # Dropping it would silently change the voice into something the author
        # never heard.
        check("a speaker the model lacks is refused, not dropped", True)


def test_checkpoints() -> None:
    """Locating base checkpoints. Offline: listings are real captured responses."""
    print("\n[checkpoints]")
    from voice2tts import checkpoints as ck

    check("voice key maps to the repo layout",
          ck.voice_directory("en_US-lessac-medium") == "en/en_US/lessac/medium",
          ck.voice_directory("en_US-lessac-medium"))
    # The name may contain underscores; only dashes separate the three fields.
    check("underscored names survive",
          ck.voice_directory("en_GB-northern_english_male-medium")
          == "en/en_GB/northern_english_male/medium")
    for bad in ("lessac", "en_US-lessac-medium-extra", ""):
        try:
            ck.voice_directory(bad)
            check(f"malformed key {bad!r} rejected", False, "no error")
        except ValueError:
            check(f"malformed key {bad!r} rejected", True)

    # Exactly what the HuggingFace API returned for lessac/medium.
    listing = [
        {"type": "file", "path": "en/en_US/lessac/medium/.gitattributes", "size": 1},
        {"type": "file", "path": "en/en_US/lessac/medium/MODEL_CARD", "size": 400},
        {"type": "file", "path": "en/en_US/lessac/medium/config.json", "size": 4000},
        {"type": "file", "path": "en/en_US/lessac/medium/dataset.jsonl.gz",
         "size": 3_000_000},
        {"type": "file",
         "path": "en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt",
         "size": 845_898_328},
        {"type": "file", "path": "en/en_US/lessac/medium/train.sh", "size": 300},
    ]
    found = ck.pick_checkpoint(listing)
    check("the checkpoint is picked out of the directory",
          found and found.filename == "epoch=2164-step=1355540.ckpt", str(found))
    check("size carried through", found.size == 845_898_328)
    check("size reported in GB", abs(found.size_gb - 0.846) < 0.01,
          f"{found.size_gb:.3f}")
    check("config.json is not mistaken for a checkpoint",
          not found.filename.endswith(".json"))
    check("directory derived from the path",
          found.directory == "en/en_US/lessac/medium")
    # The "=" in the filename has to survive into a usable URL.
    check("download url quotes the epoch marker",
          "epoch%3D2164" in found.url and found.url.startswith("https://"),
          found.url)
    check("a directory with no checkpoint yields nothing",
          ck.pick_checkpoint([{"type": "file", "path": "x/MODEL_CARD"}]) is None)
    check("an empty listing yields nothing", ck.pick_checkpoint([]) is None)

    # Verbatim from the lessac/medium MODEL_CARD.
    card = (
        "# Model card for lessac (medium)\n\n"
        "* Language: en_US (English, United States)\n"
        "* Speakers: 1\n"
        "* Quality: medium\n"
        "* Samplerate: 22,050Hz\n\n"
        "## Dataset\n\n"
        "* URL: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/\n"
        "* License: https://www.cstr.ed.ac.uk/projects/blizzard/2013/"
        "lessac_blizzard2013/license.html\n"
    )
    check("licence read from the model card",
          ck.licence_from_card(card).endswith("license.html"),
          ck.licence_from_card(card))
    check("dataset url read from the model card",
          ck.dataset_from_card(card).startswith("https://"),
          ck.dataset_from_card(card))
    # A voice with unknown terms must read as unknown, never as permissive.
    check("a card with no licence line reports nothing, not a guess",
          ck.licence_from_card("# Model card\n\nNothing useful here.\n") == "")


def test_recorder() -> None:
    """The clip recorder's buffering, without opening a real microphone.

    _callback is driven directly, which is the whole audio path minus PortAudio.
    """
    print("\n[recorder]")
    from voice2tts import recorder
    from voice2tts.devices import Device

    mic = Device(index=1, name="Test Mic", hostapi="WASAPI", rate=48000,
                 max_in=2, max_out=0)
    rec = recorder.ClipRecorder(mic)
    check("captures at the device rate", rec.rate == 48000, f"{rec.rate}")

    # The reason this class exists: recording through the recognition path would
    # resample to 16 kHz in the callback and silently cap the trained voice.
    from voice2tts import capture
    check("recognition path is 16 kHz, recording is not",
          capture.SAMPLE_RATE == 16000 and rec.rate != capture.SAMPLE_RATE,
          f"{capture.SAMPLE_RATE} vs {rec.rate}")

    stereo = np.zeros((4800, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = 0.1
    rec._callback(stereo, 4800, None, None)
    check("stereo is downmixed to mono", rec.seconds == 0.1, f"{rec.seconds}")
    audio, rate = rec.stop()
    check("returned at the device rate, unresampled", rate == 48000, f"{rate}")
    check("channels averaged", abs(float(audio[0]) - 0.3) < 1e-6, str(audio[0]))
    check("length preserved", len(audio) == 4800, str(len(audio)))

    rec2 = recorder.ClipRecorder(mic)
    check("a fresh recorder has nothing", rec2.stop()[0].size == 0)

    loud = np.full((480, 1), 1.0, dtype=np.float32)
    rec2._callback(loud, 480, None, None)
    check("clipping is flagged live", rec2.clipped)
    quiet = np.full((480, 1), 0.2, dtype=np.float32)
    rec2._callback(quiet, 480, None, None)
    check("clipping stays flagged once it happened", rec2.clipped)
    check("peak follows the latest block", abs(rec2.peak - 0.2) < 1e-6, str(rec2.peak))

    # An unattended recorder must not grow without bound.
    capped = recorder.ClipRecorder(mic, max_seconds=0.5)
    block = np.zeros((4800, 1), dtype=np.float32)
    for _ in range(20):
        capped._callback(block, 4800, None, None)
    check("recording stops itself at the cap", capped.overran)
    check("and keeps only up to the cap", capped.seconds <= 0.6,
          f"{capped.seconds:.2f}s")

    # A second take must not inherit the first one's audio.
    reused = recorder.ClipRecorder(mic)
    reused._callback(np.full((4800, 1), 0.4, dtype=np.float32), 4800, None, None)
    reused.stop()
    reused._callback(np.full((480, 1), 0.4, dtype=np.float32), 480, None, None)
    check("takes do not accumulate", reused.stop()[0].size == 480)


def test_training() -> None:
    """The training command line, progress parsing, and export guards.

    Training itself is hours of GPU time and is not run here. What is checked is
    everything that decides whether those hours are wasted: the arguments, which
    checkpoint gets picked up, and that a half-finished export cannot be
    mistaken for a working voice.
    """
    print("\n[training]")
    import json
    import os
    import tempfile

    from voice2tts import training

    cfg = training.TrainingConfig(
        voice_name="my-voice",
        dataset_csv=Path("D:/ds/metadata.csv"),
        work_dir=Path("D:/work"),
        base_checkpoint=Path("D:/base/lessac.ckpt"),
    )
    fresh = training.build_command(cfg)
    joined = " ".join(fresh)
    check("command invokes the studio interpreter",
          fresh[1:4] == ["-m", "piper.train", "fit"], str(fresh[1:4]))
    for flag in ("--data.csv_path", "--data.audio_dir", "--data.config_path",
                 "--data.espeak_voice", "--data.voice_name", "--model.sample_rate",
                 "--trainer.default_root_dir"):
        check(f"passes {flag}", flag in fresh)
    check("audio dir is the csv's own directory",
          fresh[fresh.index("--data.audio_dir") + 1] == str(Path("D:/ds")))
    # sample_rate is linked model -> data by the CLI; setting both is an error.
    check("sample rate is set once, on the model", "--data.sample_rate" not in fresh)

    # The distinction that costs hours if inverted.
    check("a base voice warm-starts", "--model.warmstart_ckpt" in fresh, joined)
    check("a base voice does not resume its trainer state",
          "--ckpt_path" not in fresh, joined)
    resumed = training.build_command(cfg, resume_from=Path("D:/work/last.ckpt"))
    check("resuming our own run restores trainer state",
          "--ckpt_path" in resumed, " ".join(resumed))
    check("resuming does not also warm-start",
          "--model.warmstart_ckpt" not in resumed, " ".join(resumed))

    sizes = [training.suggest_batch_size(v) for v in (6, 8, 12, 16, 24, 80)]
    check("batch size grows with VRAM", sizes == sorted(sizes), str(sizes))
    check("a weak card still gets a usable batch", sizes[0] >= 4, str(sizes[0]))

    # Progress parsing, against Lightning's actual bar format.
    bar = ("Epoch 3:  45%|####5     | 45/100 [00:12<00:15,  3.55it/s, "
           "v_num=0, loss=42.125]")
    p = training.parse_progress(bar)
    check("epoch parsed", p and p.epoch == 3, str(p))
    check("steps parsed", p and (p.step, p.total_steps) == (45, 100), str(p))
    check("loss parsed", p and abs(p.loss - 42.125) < 1e-6, str(p))
    check("fraction computed", p and abs(p.fraction - 0.45) < 1e-6, str(p.fraction))
    check("unrelated output is ignored",
          training.parse_progress("INFO: seeding everything with 1234") is None)
    check("a bar without an epoch keeps the last one",
          training.parse_progress("60/100 [00:20<00:10]", p).epoch == 3)
    check("fraction is zero before any steps",
          training.Progress().fraction == 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        nested = work / "lightning_logs" / "version_1" / "checkpoints"
        nested.mkdir(parents=True)
        check("no checkpoints yet", training.best_checkpoint(work) is None)
        check("nothing to resume from", training.resume_point(work) is None)

        for name in ("epoch=4-val_mel=0.4000.ckpt", "epoch=9-val_mel=0.2100.ckpt",
                     "epoch=9-val_mos=3.9000.ckpt", "last.ckpt"):
            (nested / name).write_bytes(b"x")

        check("checkpoints found in Lightning's nested layout",
              len(training.checkpoints(work)) == 4,
              str(len(training.checkpoints(work))))
        best = training.best_checkpoint(work)
        check("best is the lowest val_mel, not the newest file",
              best and best.name == "epoch=9-val_mel=0.2100.ckpt", str(best))
        check("resume uses last.ckpt",
              training.resume_point(work).name == "last.ckpt")

        # An interrupted run has only last.ckpt, and must still be exportable.
        bare = Path(tmp) / "bare"
        (bare / "ckpt").mkdir(parents=True)
        (bare / "ckpt" / "last.ckpt").write_bytes(b"x")
        check("an unscored run still offers something to export",
              training.best_checkpoint(bare).name == "last.ckpt")

        # Export guards. Neither of these should reach the subprocess.
        try:
            training.export(Path(tmp) / "missing.ckpt", Path(tmp) / "config.json",
                            Path(tmp) / "out", "v")
            check("a missing checkpoint is refused", False, "no error")
        except FileNotFoundError:
            check("a missing checkpoint is refused", True)

        ckpt = Path(tmp) / "real.ckpt"
        ckpt.write_bytes(b"x")
        try:
            training.export(ckpt, Path(tmp) / "config.json", Path(tmp) / "out", "v")
            check("a missing voice config is refused", False, "no error")
        except FileNotFoundError as exc:
            # A voice exported without its config loads and then speaks nonsense,
            # so the message has to say where the config comes from.
            check("a missing voice config is refused",
                  "written by training" in str(exc), str(exc))

        # Auditioning has the same guards, since it reads the same two files.
        try:
            training.audition(Path(tmp) / "missing.ckpt", Path(tmp) / "config.json",
                              Path(tmp) / "audio")
            check("audition refuses a missing checkpoint", False, "no error")
        except FileNotFoundError:
            check("audition refuses a missing checkpoint", True)

    check("the audition line exercises more than one sound",
          len(set(training.AUDITION_TEXT.lower()) & set("aeiou")) >= 5
          and len(training.AUDITION_TEXT.split()) > 8, training.AUDITION_TEXT)

    # A run outlives the settings window, which is destroyed on close. Without
    # this record the next panel would offer to start a second trainer on the
    # same directory, and both would write the same checkpoints.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        check("nothing running in a fresh directory",
              training.running_elsewhere(work) is None)

        training.mark_running(work, os.getpid())
        # This process is not the studio interpreter, so the record is stale by
        # image path even though the pid is alive.
        check("a pid that is not the trainer is treated as stale",
              training.running_elsewhere(work) is None)
        check("and the stale record is cleared, not left to block forever",
              not (work / training.PID_FILE).exists())

        training.mark_running(work, 999_999_999)
        check("a dead pid is stale", training.running_elsewhere(work) is None)

        (work / training.PID_FILE).write_text("not a number", encoding="utf-8")
        check("a corrupt pid file is survivable",
              training.running_elsewhere(work) is None)

        training.mark_running(work, os.getpid())
        training.clear_running(work)
        check("clearing removes the record",
              not (work / training.PID_FILE).exists())
        check("clearing twice is harmless",
              training.clear_running(work) is None)

    prov = training.Provenance(voice_name="mine", dataset_clips=120,
                               dataset_seconds=1800.0, epochs=400,
                               base_checkpoint="lessac.ckpt")
    data = json.loads(json.dumps(prov.to_dict()))
    check("provenance survives json", data["voice_name"] == "mine")
    check("provenance records where it came from",
          data["base_checkpoint"] == "lessac.ckpt" and data["dataset_clips"] == 120)
    check("provenance stamps the app version", bool(data["app_version"]))


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

    # The GPU identity is cached because the Studio tab asks repeatedly, but a
    # cached answer to "is the GPU pack installed" or "how much disk is free"
    # would be wrong rather than merely slow.
    first = studiopack.probe()
    start = time.perf_counter()
    again = studiopack.probe()
    cached_ms = (time.perf_counter() - start) * 1000
    check("a repeat probe skips nvidia-smi", cached_ms < 40, f"{cached_ms:.0f} ms")
    check("cached probe agrees on the GPU",
          (again.gpu_name, again.vram_gb) == (first.gpu_name, first.vram_gb))
    check("disk and pack state are still re-read, not cached",
          "free_disk_gb" in vars(studiopack.probe()))
    forced = studiopack.probe(force=True)
    check("forcing re-reads the hardware", forced.gpu_name == first.gpu_name)

    check("probe runs on this machine", isinstance(studiopack.probe(), Hardware))
    live = studiopack.probe()
    # NOT "this machine has a GPU" -- that asserts the developer's hardware and
    # fails on any runner, which is exactly how this test broke CI. What is
    # actually worth checking is that the probe agrees with itself: a named GPU
    # comes with memory, and no GPU comes with none.
    consistent = (live.has_gpu and live.gpu_name and live.vram_gb > 0) or (
        not live.has_gpu and live.vram_gb == 0)
    check("probe reports the GPU consistently", bool(consistent),
          f"{live.gpu_name or 'no NVIDIA GPU'}, {live.vram_gb} GB")
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
    # ci.yml ignores tags, so every suite has to be named here or it never runs
    # against the commit that actually ships. 0.5.1 shipped with every dropdown
    # broken because the GUI suite was not part of the gate.
    check("verify runs the GUI test", "gui test" in verify_steps, verify_steps)
    # Betas. The tag filter already matches them, so the danger is not that they
    # fail to build -- it is that they build and publish as the latest stable
    # release, and every user is offered one.
    verify_job = jobs.get("verify", {})
    resolve = next((s for s in verify_job.get("steps", [])
                    if "version" in str(s.get("name", "")).lower()), {})
    resolve_run = str(resolve.get("run", ""))
    check("the tag shape is validated, not just trimmed",
          "beta" in resolve_run and "not vX.Y.Z" in resolve_run,
          "a typo like v0.6.0-beta1 must be rejected, not shipped as stable")
    check("verify reports whether the tag is a pre-release",
          "prerelease=" in resolve_run)

    publish = " ".join(str(s.get("run", "")) for s in publisher.get("steps", []))
    check("betas are published as pre-releases", "--prerelease" in publish,
          "the updater reads /releases/latest, which skips pre-releases")
    check("the pre-release flag is conditional, not always on",
          "outputs.prerelease" in publish,
          "a stable release must not be marked as a pre-release")
    check("the build stamps the tag version into the code",
          "__version__" in publish,
          "otherwise an installed beta reports itself as the plain release")

    ci_steps = " ".join(name for job in ci.get("jobs", {}).values()
                        for name in step_names(job)).lower()
    for suite in ("self-test", "gui test"):
        check(f"every suite CI runs is also in the release gate ({suite})",
              suite not in ci_steps or suite in verify_steps,
              f"ci={suite in ci_steps} verify={suite in verify_steps}")
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

    # -- the soft endpoint ---------------------------------------------------
    # Someone speaking quickly never leaves a 600 ms gap, so the endpoint rule
    # never fired and nothing was spoken until they stopped. Measured on this
    # sample: probability sits at 1.0, no gap reaches 600 ms, and lowering
    # min_silence_ms to 300 still yields zero cut points.
    def segment(cfg, source):
        seg = VadSegmenter(cfg, preroll_ms=300, max_utterance_s=30)
        pieces, times = [], []
        for i in range(0, len(source) - WINDOW, WINDOW):
            got = seg.process(source[i:i + WINDOW])
            if got is not None:
                pieces.append(got)
                times.append((i + WINDOW) / 16000)
        rest = seg.flush()
        if rest is not None:
            pieces.append(rest)
            times.append(len(source) / 16000)
        return pieces, times

    continuous = np.tile(audio, 4)
    off = VadConfig(soft_endpoint_s=0.0)
    on = VadConfig()

    was, was_at = segment(off, continuous)
    now, now_at = segment(on, continuous)

    check("without it, continuous speech waits for the hard cap",
          was_at[0] > 20.0, f"first audio at {was_at[0]:.1f}s")
    check("the soft endpoint cuts continuous speech into segments",
          len(now) > len(was), f"{len(now)} vs {len(was)}")
    check("and gets audio out far sooner", now_at[0] < was_at[0] / 3,
          f"{now_at[0]:.1f}s vs {was_at[0]:.1f}s")
    check("no segment outruns the ceiling",
          max(len(p) for p in now) / 16000 <= on.max_segment_s + 1.0,
          f"longest {max(len(p) for p in now) / 16000:.1f}s "
          f"(ceiling {on.max_segment_s}s)")

    # Cutting must CARRY THE REST FORWARD, not drop it -- the speaker has not
    # stopped, so anything after the cut is still speech they expect to be said.
    kept = sum(len(p) for p in now) / 16000
    check("cutting loses no speech", kept > len(continuous) / 16000 * 0.97,
          f"{kept:.1f}s of {len(continuous) / 16000:.1f}s")

    # A short utterance followed by a real pause must be untouched: the whole
    # point is that ordinary speech behaves as before.
    gap = np.zeros(int(16000 * 0.9), dtype=np.float32)
    short = audio[:16000 * 2]
    polite = np.concatenate([np.zeros(8000, np.float32), short, gap, short, gap])
    before, before_at = segment(off, polite)
    after, after_at = segment(on, polite)
    check("short utterances with real pauses are unaffected",
          [len(p) for p in before] == [len(p) for p in after]
          and before_at == after_at,
          f"{[round(t, 2) for t in before_at]} vs {[round(t, 2) for t in after_at]}")

    check("turning it off restores the old behaviour exactly",
          segment(VadConfig(soft_endpoint_s=0.0), continuous)[1] == was_at)

    # The relaxation itself: constant, then easing down, never below the floor.
    probe = VadSegmenter(on, preroll_ms=300, max_utterance_s=30)
    from voice2tts.vad import WINDOW_MS

    required = []
    for seconds in (0, 1, 2, 3, 4, 5, 6, 8):
        probe._buf = [np.zeros(WINDOW, np.float32)] * int(seconds * 1000 / WINDOW_MS)
        required.append(probe._required_silence())
    check("the requirement never rises", required == sorted(required, reverse=True),
          str(required))
    check("it starts at the normal rule",
          required[0] == on.min_silence_ms // WINDOW_MS, str(required[:2]))
    check("and never drops below the floor",
          min(required) >= on.min_silence_floor_ms // WINDOW_MS,
          f"min {min(required)} windows")


def _fake_model(root: Path, code: str) -> Path:
    """A directory shaped like a translation model, without the 164 MB.

    is_usable() only asks whether both halves are present, so the parts that
    manage models can be tested everywhere -- CI has no model and never will.
    """
    directory = root / code
    (directory / "model").mkdir(parents=True, exist_ok=True)
    (directory / "model" / "model.bin").write_bytes(b"not really a model")
    (directory / "sentencepiece.model").write_bytes(b"not really a tokenizer")
    return directory


def test_translate() -> None:
    """Managing translation models. Translating itself needs one installed."""
    print("\n[translation]")
    import tempfile
    import zipfile

    from voice2tts import translate

    check("sentences split on terminal punctuation",
          translate.split_sentences("First one. Then a second! And a third?")
          == ["First one.", "Then a second!", "And a third?"])
    check("a single sentence stays whole",
          translate.split_sentences("No punctuation here") == ["No punctuation here"])
    check("blank text yields nothing", translate.split_sentences("   ") == [])

    real_dir = translate.models_dir
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        translate.models_dir = lambda: root
        try:
            check("nothing installed to begin with", translate.installed_pairs() == [])

            _fake_model(root, "en_de")
            _fake_model(root, "de_en")
            pairs = translate.installed_pairs()
            check("installed pairs are found",
                  [p.code for p in pairs] == ["de_en", "en_de"],
                  str([p.code for p in pairs]))
            check("a pair knows its direction",
                  pairs[1].source == "en" and pairs[1].target == "de")
            check("find_pair matches a direction",
                  translate.find_pair("en", "de") is not None
                  and translate.find_pair("en", "fr") is None)

            # A model without its tokenizer produces gibberish rather than an
            # error, so half a model must not count as installed.
            half = root / "en_fr"
            (half / "model").mkdir(parents=True)
            (half / "model" / "model.bin").write_bytes(b"x")
            check("a model with no tokenizer is not offered",
                  translate.find_pair("en", "fr") is None,
                  "half a model would translate to nonsense")

            check("a direct route is one hop",
                  [p.code for p in translate.route("en", "de")] == ["en_de"])
            check("no route to a language nobody installed",
                  translate.route("en", "es") == [])
            check("translating to the same language is a no-op",
                  translate.route("en", "en") == [])

            # Pivoting: de -> en -> ... needs the second half to exist too.
            check("a pivot needs both halves", translate.route("de", "es") == [])
            _fake_model(root, "en_es")
            route = translate.route("de", "es")
            check("a pivot is used when there is no direct model",
                  [p.code for p in route] == ["de_en", "en_es"],
                  str([p.code for p in route]))
            # ...but a direct model always wins, because a pivot compounds errors.
            _fake_model(root, "de_es")
            check("a direct model beats a pivot",
                  [p.code for p in translate.route("de", "es")] == ["de_es"])

            # Installing from a package: whatever shape it arrives in, only the
            # model and its tokenizer are kept.
            archive = root / "package.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("translate-en_it/model/model.bin", "weights")
                zf.writestr("translate-en_it/model/config.json", "{}")
                zf.writestr("translate-en_it/sentencepiece.model", "tokens")
                zf.writestr("translate-en_it/README.md", "hello")
                # A sentence splitter ships alongside in some packages. It is a
                # torch checkpoint, and unpacking it would drag torch in.
                zf.writestr("translate-en_it/stanza/en/tokenize/ewt.pt", "torch!")
            installed = translate.install_package(archive, "en", "it")
            check("a package installs", translate.is_usable(installed))
            check("and is normalised to model + tokenizer",
                  sorted(p.name for p in installed.iterdir())
                  == ["model", "sentencepiece.model"],
                  str(sorted(p.name for p in installed.iterdir())))
            check("the sentence splitter is dropped",
                  not (installed / "stanza").exists(),
                  "it is a torch checkpoint we do not need")
            check("no staging directory is left behind",
                  not any(p.name.startswith(".") for p in root.iterdir()),
                  str([p.name for p in root.iterdir() if p.name.startswith(".")]))

            bad = root / "bad.zip"
            with zipfile.ZipFile(bad, "w") as zf:
                zf.writestr("something/README.md", "no model here")
            try:
                translate.install_package(bad, "en", "pt")
                check("a package with no model is refused", False, "no error")
            except translate.TranslationUnavailable:
                check("a package with no model is refused", True)

            check("removing a pair works", translate.remove_pair("en", "it"))
            check("and it is gone", translate.find_pair("en", "it") is None)
            check("removing what is not there is not an error",
                  translate.remove_pair("zz", "yy") is False)

            try:
                translate.Chain([])
                check("a chain with no route is refused", False, "no error")
            except translate.TranslationUnavailable:
                check("a chain with no route is refused", True)
        finally:
            translate.models_dir = real_dir

    # The real thing, if a model happens to be installed on this machine.
    live = translate.installed_pairs()
    if not live:
        print("  SKIP  live translation (no model installed)")
        return
    pair = live[0]
    chain = translate.Chain([pair])
    start = time.perf_counter()
    out = chain.translate("Hello, can you hear me?")
    elapsed = time.perf_counter() - start
    check(f"translates {pair.code}", bool(out.strip()),
          f"{elapsed * 1000:.0f} ms -> {out[:40]!r}")
    check("stays inside the latency budget", elapsed < 0.3,
          f"{elapsed * 1000:.0f} ms")
    check("empty input is safe", chain.translate("") == "")
    chain.close()


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


def _powershell() -> str | None:
    """pwsh if present (what Actions uses), else Windows PowerShell, else None."""
    return shutil.which("pwsh") or shutil.which("powershell")


def _run_ps(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    shell = _powershell()
    full = dict(os.environ, **(env or {}))
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=full,
    )


_EXPRESSION = re.compile(r"\$\{\{[^}]*\}\}")


def test_release_powershell() -> None:
    """The workflows' PowerShell must parse, and tag resolution must be right.

    Both halves exist because of how this fails otherwise: a workflow is not
    compiled until it runs, and it only runs when a release is tagged. A stray
    quote or a wrong regex is therefore discovered at the worst possible moment,
    with a tag already pushed.
    """
    print("\n[release powershell]")
    import tempfile

    shell = _powershell()
    if shell is None:
        print("  SKIP  no PowerShell on this machine")
        return
    # Actions always uses pwsh (PowerShell 7). Falling back to Windows
    # PowerShell 5.1 still catches syntax and logic errors, but the two differ
    # in places -- notably "-Encoding utf8", which writes a BOM in 5.1 and none
    # in 7 -- so it is worth knowing which one ran.
    print(f"  (using {Path(shell).name})")

    import yaml

    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    check("workflows found", bool(workflows), str([w.name for w in workflows]))

    # -- every pwsh block must parse -----------------------------------------
    blocks: list[tuple[str, str, str]] = []
    for path in workflows:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps", []):
                if step.get("shell") == "pwsh" and step.get("run"):
                    blocks.append((path.name, f"{job_name}/{step.get('name', '?')}",
                                   step["run"]))
    check("pwsh blocks found to check", len(blocks) >= 3, f"{len(blocks)} blocks")

    broken = []
    for source, where, body in blocks:
        # ${{ }} is substituted by Actions before PowerShell ever sees it. A
        # placeholder keeps the surrounding syntax intact while making the
        # result parseable.
        script = _EXPRESSION.sub("PLACEHOLDER", body)
        probe = (
            "$errors = $null; $tokens = $null;\n"
            "[void][System.Management.Automation.Language.Parser]::ParseInput("
            "$env:V2T_SCRIPT, [ref]$tokens, [ref]$errors)\n"
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }\n"
        )
        result = _run_ps(probe, {"V2T_SCRIPT": script})
        if result.returncode != 0:
            broken.append(f"{source}:{where}: {result.stdout.strip()[:160]}")
    check("every pwsh block parses", not broken,
          "; ".join(broken) if broken else f"{len(blocks)} blocks")

    # -- tag resolution actually works ---------------------------------------
    release = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    resolve = next(
        (s for s in release["jobs"]["verify"]["steps"]
         if "version" in str(s.get("name", "")).lower() and s.get("run")), None)
    check("the version resolution step is findable", resolve is not None)
    if resolve is None:
        return

    import voice2tts

    declared = voice2tts.__version__

    def resolve_tag(tag: str) -> tuple[bool, dict]:
        """Run the real step with a given tag. Returns (succeeded, outputs)."""
        script = resolve["run"].replace(
            "${{ github.event.inputs.tag || github.ref_name }}", tag)
        script = _EXPRESSION.sub("", script)
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "out.txt"
            out_file.touch()
            result = _run_ps(
                f"$ErrorActionPreference='Stop'\nSet-Location '{ROOT}'\n{script}",
                {"GITHUB_OUTPUT": str(out_file)})
            values = {}
            # utf-8-sig: under Windows PowerShell 5.1 the first append writes a
            # BOM, which would otherwise turn the key "version" into "
            # version". Actions runs pwsh, where it does not happen; this is
            # about the local fallback, not about the workflow.
            for line in out_file.read_text(encoding="utf-8-sig").splitlines():
                key, _, value = line.partition("=")
                if key:
                    values[key.strip()] = value.strip()
            return result.returncode == 0, values

    ok, outputs = resolve_tag(f"v{declared}")
    check("a plain release tag resolves", ok and outputs.get("version") == declared,
          str(outputs))
    check("and is not marked as a pre-release",
          outputs.get("prerelease") == "false", str(outputs))

    ok, outputs = resolve_tag(f"v{declared}-beta-1")
    check("a beta tag resolves", ok, str(outputs))
    check("the beta keeps its full version",
          outputs.get("version") == f"{declared}-beta-1", str(outputs))
    check("the beta reports the release it targets",
          outputs.get("base") == declared, str(outputs))
    check("and IS marked as a pre-release",
          outputs.get("prerelease") == "true",
          "otherwise it becomes the latest stable release for every user")

    ok, outputs = resolve_tag(f"v{declared}-beta-12")
    check("beta numbers past nine work",
          ok and outputs.get("version") == f"{declared}-beta-12", str(outputs))

    # -- the build stamps the tag version into the code ----------------------
    stamp = next((s for s in release["jobs"]["release"]["steps"]
                  if "stamp" in str(s.get("name", "")).lower() and s.get("run")), None)
    check("the stamping step is findable", stamp is not None,
          "without it an installed beta reports the plain release version")
    if stamp is not None:
        def run_stamp(version: str, base: str) -> str:
            script = (stamp["run"]
                      .replace("${{ needs.verify.outputs.version }}", version)
                      .replace("${{ needs.verify.outputs.base }}", base))
            with tempfile.TemporaryDirectory() as tmp:
                package = Path(tmp) / "voice2tts"
                package.mkdir()
                target = package / "__init__.py"
                target.write_text(
                    f'"""Doc."""\n\n__version__ = "{base}"\n'
                    'DEFAULT_UPDATE_REPO = "aldenparker/Voice2TTS"\n',
                    encoding="utf-8")
                out = _run_ps(
                    f"$ErrorActionPreference='Stop'\nSet-Location '{tmp}'\n{script}")
                if out.returncode != 0:
                    return f"FAILED: {(out.stderr or out.stdout).strip()[:200]}"
                return target.read_text(encoding="utf-8-sig")

        stamped = run_stamp(f"{declared}-beta-3", declared)
        check("stamping writes the beta version",
              f'__version__ = "{declared}-beta-3"' in stamped,
              stamped.strip().replace("\n", " | ")[:120])
        check("and leaves the rest of the file alone",
              "DEFAULT_UPDATE_REPO" in stamped and stamped.count("__version__") == 1,
              "a greedy replace would take the repo line with it")
        plain = run_stamp(declared, declared)
        check("stamping a plain release is a no-op it accepts",
              f'__version__ = "{declared}"' in plain, plain[:80])

    # Typos must stop the release, not quietly ship as stable.
    for bad, why in (
        (f"v{declared}-beta1", "missing the separator"),
        (f"v{declared}-rc-1", "a channel that does not exist"),
        (f"v{declared}-beta-0", "beta numbering starts at 1"),
        (f"v{declared}-beta", "no beta number"),
        ("v1.2", "not three components"),
        ("v9.9.9", "does not match __version__"),
    ):
        ok, _out = resolve_tag(bad)
        check(f"tag {bad!r} is rejected ({why})", not ok)


def test_perf_sampling() -> None:
    """The CPU sampler, which exists to answer "what is eating the CPU?"."""
    print("\n[cpu sampling]")
    import threading as _threading

    from voice2tts import perf

    snap = perf.sample(0.3)
    if not snap.supported:
        print("  SKIP  CPU sampling is Windows-only")
        return

    check("sampling finds this process's threads", len(snap.threads) >= 1,
          f"{len(snap.threads)} threads")
    check("the sampled window is about what was asked for",
          0.25 <= snap.seconds <= 1.0, f"{snap.seconds:.2f}s")
    check("an idle process is not pegged", snap.busy < 2.0,
          f"{snap.busy * 100:.0f}% of one core")
    check("the report is human-readable", any("Process CPU" in line
                                              for line in snap.report()))

    # The point of the tool: a runaway thread must be NAMED, not just counted.
    stop = _threading.Event()

    def burn():
        data = np.random.rand(200, 200)
        while not stop.is_set():
            data @ data

    worker = _threading.Thread(target=burn, name="v2t-test-burner", daemon=True)
    worker.start()
    try:
        time.sleep(0.2)
        hot = perf.sample(0.5)
    finally:
        stop.set()
        worker.join(timeout=5)

    check("a busy process is reported as busy", hot.busy > snap.busy,
          f"{hot.busy * 100:.0f}% vs {snap.busy * 100:.0f}% of one core")

    # NOT "the burner ranks first". numpy hands the multiplication to BLAS
    # workers, so the thread that dispatches can legitimately use less CPU than
    # any of them -- asserting a rank tested BLAS's scheduler, and failed about
    # one run in eight. What the sampler promises is attribution BY NAME.
    burner = next((share for name, share in hot.threads
                   if "v2t-test-burner" in name), None)
    check("a named thread's own CPU is attributed to it",
          burner is not None and burner > 0.01,
          f"{(burner or 0) / hot.seconds * 100:.0f}% of a core")
    check("busy threads appear in the report",
          any("of a core" in line for line in hot.report()))

    idle = perf.Snapshot(seconds=1.0, process_cpu=0.0, threads=[("MainThread", 0.0)])
    check("an idle report says so plainly",
          any("idle" in line for line in idle.report()), str(idle.report()))


def test_packaging_bits() -> None:
    """Offline checks for the cable, voices and GPU-pack modules."""
    print("\n[packaging]")
    import re
    import zipfile

    from voice2tts import cable, gpupack, voices
    from voice2tts.paths import bundled_whisper, cache_dir, cuda_dir

    # --- paths
    check("cache dir is LOCALAPPDATA", "Local" in str(cache_dir()), str(cache_dir()))
    check("cuda dir under cache", cuda_dir().parent == cache_dir())
    check("base.en bundled for offline install", bundled_whisper("base.en") is not None,
          str(bundled_whisper("base.en")))
    check("absent model reports None", bundled_whisper("nonexistent-model") is None)

    # --- the spec must name what Analysis cannot see
    # PyInstaller walks module-level imports reliably; imports tucked inside a
    # function are the ones that can go missing from a frozen build, and the
    # failure only shows up when a user reaches the feature that needs them.
    #
    # Checking gui.py alone was not enough: updater, clipboard, loopback and perf
    # are all reachable ONLY through a function-level import, from other modules.
    # A missing voice2tts.updater would stop updates working for everyone, in the
    # shipped build only.
    spec = (ROOT / "Voice2TTS.spec").read_text(encoding="utf-8")
    package = ROOT / "voice2tts"
    own = {p.stem for p in package.glob("*.py")} - {"__init__", "__main__"}

    top_level = re.compile(r"^(?:from \.(\w+) import|from \. import ([\w, ]+))")
    inside_a_function = re.compile(r"^\s+(?:from \.(\w+) import|from \. import ([\w, ]+))")

    def collect(pattern) -> set[str]:
        found: set[str] = set()
        for source in package.glob("*.py"):
            for line in source.read_text(encoding="utf-8").splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                if match.group(1):
                    found.add(match.group(1))
                elif match.group(2):
                    found.update(name.strip().split(" as ")[0]
                                 for name in match.group(2).split(","))
        return found & own

    lazy_only = sorted(collect(inside_a_function) - collect(top_level))
    check("some modules really are only imported lazily", bool(lazy_only),
          ", ".join(lazy_only) or "none -- has the import style changed?")
    undeclared = sorted(m for m in lazy_only if f'"voice2tts.{m}"' not in spec)
    check("every lazily-reachable module is declared in the spec",
          not undeclared,
          f"missing from hiddenimports: {undeclared}" if undeclared
          else ", ".join(lazy_only))

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
        ("0.2.0", "0.2", False),        # 0.2 and 0.2.0 are the same version
        ("1.2.3+build7", "1.2.3", False),   # build metadata is not a version
    ]
    wrong = [(a, b, e) for a, b, e in cases if updater.is_newer(a, b) != e]
    check("version comparison", not wrong, f"failed: {wrong}" if wrong else f"{len(cases)} cases")

    # Betas must order BELOW the release they lead to. Getting this wrong is
    # invisible until a beta tester is silently stranded on the beta, because
    # the app cannot tell it is behind the finished release.
    beta_cases = [
        ("0.6.0", "0.6.0-beta-1", True),        # the release supersedes its betas
        ("0.6.0-beta-2", "0.6.0-beta-1", True),  # betas advance
        ("0.6.0-beta-1", "0.6.0-beta-2", False),
        ("0.6.0-beta-1", "0.6.0", False),       # a beta never supersedes a release
        ("0.6.0-beta-1", "0.5.2", True),        # but it is ahead of the last one
        ("0.6.0-beta-1", "0.6.0-beta-1", False),
        ("0.6.0-beta-10", "0.6.0-beta-9", True),  # numeric, not lexicographic
    ]
    wrong = [(a, b, e) for a, b, e in beta_cases if updater.is_newer(a, b) != e]
    check("betas order below the release they lead to", not wrong,
          f"failed: {wrong}" if wrong else f"{len(beta_cases)} cases")
    check("a full release outranks its own beta",
          updater.parse_version("0.6.0") > updater.parse_version("0.6.0-beta-99"),
          "no number of betas reaches the release")

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

    # -- the beta channel ----------------------------------------------------
    # Selection from a full listing, which is only consulted when someone opts
    # in. Offline: these are the shapes the GitHub API returns.
    def entry(tag, *, draft=False, asset=True, prerelease=False):
        payload = {"tag_name": tag, "draft": draft, "prerelease": prerelease,
                   "body": "notes", "html_url": f"https://example.invalid/{tag}",
                   "assets": []}
        if asset:
            payload["assets"] = [
                {"name": f"Voice2TTS-Setup-{tag.lstrip('v')}.exe", "size": 123,
                 "browser_download_url": f"https://example.invalid/{tag}.exe"},
                {"name": f"Voice2TTS-Setup-{tag.lstrip('v')}.exe.sha256",
                 "browser_download_url": f"https://example.invalid/{tag}.sha256"},
            ]
        return payload

    listing = [entry("v0.6.0-beta-1", prerelease=True),
               entry("v0.5.2"),
               entry("v0.6.0-beta-2", prerelease=True)]
    picked = updater.pick_newest(listing)
    check("the newest beta is picked from a listing",
          picked and picked["tag_name"] == "v0.6.0-beta-2",
          str(picked and picked["tag_name"]))

    # GitHub orders by creation date. A patch to an older series published after
    # a newer release would come back first and must not win.
    out_of_order = [entry("v0.5.3"), entry("v0.9.0"), entry("v0.5.4")]
    check("selection goes by version, not by the order GitHub returns",
          updater.pick_newest(out_of_order)["tag_name"] == "v0.9.0",
          updater.pick_newest(out_of_order)["tag_name"])

    check("drafts are never offered",
          updater.pick_newest([entry("v9.9.9", draft=True), entry("v0.5.2")]
                              )["tag_name"] == "v0.5.2")
    # A release with no installer cannot be installed, so falling back to one
    # that can beats failing the whole check.
    check("a release with no installer is skipped, not fatal",
          updater.pick_newest([entry("v9.9.9", asset=False), entry("v0.5.2")]
                              )["tag_name"] == "v0.5.2")
    check("an empty listing yields nothing", updater.pick_newest([]) is None)
    check("a listing of only drafts yields nothing",
          updater.pick_newest([entry("v9.9.9", draft=True)]) is None)

    # The two channels must read different endpoints; that difference is the
    # whole mechanism keeping betas away from people who did not ask.
    check("the stable channel asks for the latest release only",
          updater.API_TEMPLATE.endswith("/releases/latest"),
          updater.API_TEMPLATE)
    check("the beta channel asks for the full listing",
          "/releases?" in updater.RELEASES_TEMPLATE
          and "latest" not in updater.RELEASES_TEMPLATE,
          updater.RELEASES_TEMPLATE)

    check("opting in is off by default", Config().updates.include_prereleases is False,
          "a beta is something you choose, not something you drift into")

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


def _reach(what: str, call, attempts: int = 3, pause: float = 2.0):
    """Run a live network call, retrying transient failures.

    Returns (value, unreachable_reason). These checks exist to confirm that OUR
    parsing still matches what the service actually publishes -- a reset socket
    says nothing about that, so a connection that never comes up is reported as
    unreachable and skipped rather than failing the run. A release must not
    hinge on somebody else's uptime, and CI lost a build to
    "[WinError 10054] An existing connection was forcibly closed by the remote
    host".

    A 4xx other than 429 is NOT treated as transient: that means the URL we
    hardcode is wrong, which is our bug and worth failing over.
    """
    import urllib.error

    last = ""
    for attempt in range(attempts):
        try:
            return call(), None
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            last = f"HTTP {exc.code}"
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        if attempt + 1 < attempts:
            time.sleep(pause * (attempt + 1))
    return None, f"could not reach {what} after {attempts} tries: {last}"


def test_network() -> None:
    """Live checks against VB-Audio and HuggingFace. Needs internet."""
    print("\n[network]")
    # A non-transient HTTP error is our bug -- the address we ship is wrong --
    # so it fails rather than skips. Caught here rather than left to propagate,
    # because an exception out of a test takes the whole run down with it and a
    # readable FAIL is more use than a traceback.
    import urllib.error

    from voice2tts import cable, voices

    try:
        resolved, unreachable = _reach("the VB-Audio download page",
                                       cable.resolve_download_url)
    except urllib.error.HTTPError as exc:
        check("cable download URL resolves", False,
              f"HTTP {exc.code} -- the address in cable.py may be wrong")
        resolved, unreachable = None, None
    if unreachable:
        print(f"  SKIP  cable download URL ({unreachable})")
    elif resolved is not None:
        url, source = resolved
        check("cable download URL resolves",
              url.lower().endswith(".zip"), f"{source}: {url.rsplit('/', 1)[-1]}")

    try:
        catalogue, unreachable = _reach("the voice catalogue",
                                        voices.fetch_catalogue)
    except urllib.error.HTTPError as exc:
        check("voice catalogue fetched", False,
              f"HTTP {exc.code} -- CATALOGUE_URL in voices.py may be wrong")
        return
    if unreachable:
        print(f"  SKIP  voice catalogue ({unreachable})")
        return
    check("voice catalogue fetched", len(catalogue) > 50, f"{len(catalogue)} voices")
    english = voices.filter_catalogue(catalogue, language_prefix="en_US")
    check("catalogue filters by language", len(english) > 5, f"{len(english)} en_US")
    keys = {e.key for e in catalogue}
    missing = [v for v in voices.BUNDLED if v not in keys]
    check("bundled voices exist in catalogue", not missing, str(missing))
    sized = [e for e in catalogue if e.size_mb > 0]
    check("catalogue reports sizes", len(sized) > len(catalogue) // 2,
          f"{len(sized)}/{len(catalogue)}")


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

    # "Fewer than before" is only meaningful for a kind of device that exists.
    # Requiring it of BOTH kinds fails on a machine with speakers and no
    # microphone -- an ordinary desktop, and every CI runner.
    if not all_in and not all_out:
        check("no-hardware enumeration is empty, not broken",
              trimmed_in == [] and trimmed_out == []
              and devices.default_input() is None
              and devices.resolve_input("") is None,
              "no audio devices present")
    else:
        for kind, trimmed, raw in (("input", trimmed_in, all_in),
                                   ("output", trimmed_out, all_out)):
            if not raw:
                check(f"no {kind}s to trim, and none invented",
                      trimmed == [], f"0 {kind}s on this machine")
            else:
                check(f"the {kind} list is trimmed",
                      len(trimmed) < len(raw), f"{len(trimmed)}/{len(raw)}")
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

    # A voice built in the Studio is named by its author -- "narator", "my
    # voice" -- and carries no language in the filename. Reading the name told
    # people their own English voice was not English.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vdir = Path(tmp)
        real_list, real_path = voices.list_voices, voices.installed_path
        voices.list_voices = lambda: sorted(vdir.glob("*.onnx"))
        voices.installed_path = lambda k: (vdir / f"{k}.onnx"
                                           if (vdir / f"{k}.onnx").exists() else None)
        try:
            def make(name, language):
                (vdir / f"{name}.onnx").write_bytes(b"x")
                (vdir / f"{name}.onnx.json").write_text(
                    json.dumps({"language": language, "espeak": {"voice": "en-us"}}),
                    encoding="utf-8")

            make("narator", {"code": "en_US", "family": "en"})
            check("a Studio-named English voice is recognised as English",
                  voices.voice_language("narator") == "en"
                  and voices.is_english("narator"),
                  voices.voice_language("narator"))
            check("and does not warn against an English model",
                  not voices.language_mismatch("narator", "base.en"),
                  "the name says nothing; the config says English")

            # The config is authoritative, not the name.
            make("mein-sprecher", {"code": "de_DE", "family": "de"})
            check("a German voice with an English-looking name still warns",
                  bool(voices.language_mismatch("mein-sprecher", "base.en")),
                  voices.voice_language("mein-sprecher"))

            # Config missing or unreadable, and the name says nothing either.
            (vdir / "mystery.onnx").write_bytes(b"x")
            check("an unknowable language is reported as unknown",
                  voices.voice_language("mystery") == "")
            check("and warns about nothing, rather than guessing",
                  not voices.language_mismatch("mystery", "base.en"),
                  "a wrong warning is worse than none")

            # Older configs carry only the espeak voice.
            (vdir / "old.onnx").write_bytes(b"x")
            (vdir / "old.onnx.json").write_text(
                json.dumps({"espeak": {"voice": "fr-fr"}}), encoding="utf-8")
            check("espeak's voice is used when there is no language block",
                  voices.voice_language("old") == "fr", voices.voice_language("old"))
        finally:
            voices.list_voices, voices.installed_path = real_list, real_path

    check("a catalogue key still resolves without being installed",
          voices.voice_language("nl_BE-nathalie-medium") == "nl")


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
    test_prompts()
    test_dataset()
    test_recorder()
    test_dsp()
    test_designer()
    test_baking()
    test_v2tvoice()
    test_checkpoints()
    test_training()
    test_studio_gate()
    test_release_is_gated()
    test_release_powershell()
    test_winget_manifests()
    test_profiles()
    test_history_and_review()
    test_vad()
    test_translate()
    test_stt()
    test_device_lists()
    if not no_audio:
        test_loopback()
    test_platform()
    test_language_guard()
    if not no_audio:
        test_tts_and_sink()
    test_perf_sampling()
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
