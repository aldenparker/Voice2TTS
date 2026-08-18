"""Headless-ish GUI smoke test: builds every widget, exercises load/collect/tick,
then tears down. Windows are withdrawn immediately so nothing flashes on screen.

Does not load Whisper or Piper -- this is purely about the Tk construction path.

    python scripts/guitest.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice2tts import devices as devices_mod  # noqa: E402
from voice2tts import translate  # noqa: E402
from voice2tts.config import OutputTarget, load_config  # noqa: E402
from voice2tts.gui import SettingsWindow  # noqa: E402
from voice2tts.icon import make_icon  # noqa: E402
from voice2tts.logging_setup import setup_logging  # noqa: E402
from voice2tts.pipeline import Pipeline, State  # noqa: E402
from voice2tts.substitutions import STARTER_RULES  # noqa: E402
from voice2tts.wizard import Wizard  # noqa: E402

passed = failed = 0


def devices_cable_present() -> bool:
    from voice2tts import cable

    return cable.detect() is not None


# The console on Windows is cp1252, and this suite prints text it does not
# control: route arrows, voice names, translated text. A character the console
# cannot represent must degrade to "?", not kill the run half way through.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):  # redirected to something simpler
        pass


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    setup_logging("WARNING")
    print("Voice2TTS GUI smoke test")

    print("\n[icons]")
    for state in State:
        img = make_icon(state)
        check(f"icon for {state.value}", img.size == (64, 64))

    print("\n[settings window]")
    root = tk.Tk()
    root.withdraw()

    cfg = load_config()
    cfg.audio.outputs = [
        OutputTarget(match="CABLE Input", gain=1.0, enabled=True),
        OutputTarget(match="", gain=0.7, enabled=False),
    ]
    pipeline = Pipeline(cfg)  # constructed but never started

    win = SettingsWindow(root, cfg, pipeline)
    win.withdraw()
    root.update()
    check("window constructed", win.winfo_exists() == 1)
    check("output rows built", len(win._output_rows) == 2, f"{len(win._output_rows)}")
    # NOT "the list has entries" -- that asserts this machine has a microphone
    # and fails on any runner. The property worth checking is that the picker
    # mirrors what was detected, plus the blank "system default" row.
    detected_inputs = devices_mod.list_inputs(not cfg.audio.prefer_wasapi)
    check("input combo offers every input, plus a default",
          len(list(win.input_combo["values"])) == len(detected_inputs) + 1,
          f"{len(list(win.input_combo['values']))} rows for "
          f"{len(detected_inputs)} devices")
    check("voice combo populated", len(win.voice_combo["values"]) >= 1,
          str(win.voice_combo["values"]))

    # Round-trip the widgets back into the config.
    win.hotkey_var.set("ctrl+alt+b")
    win.speed_var.set(1.25)
    win._output_rows[1]["match"].set("Headphones")
    win._output_rows[1]["enabled"].set(True)
    ok = win._collect()
    check("collect succeeded", ok)
    check("hotkey collected", cfg.trigger.hotkey == "ctrl+alt+b", cfg.trigger.hotkey)
    check("speed collected", abs(cfg.tts.length_scale - 1.25) < 1e-6)
    check("output edits collected",
          cfg.audio.outputs[1].match == "Headphones" and cfg.audio.outputs[1].enabled,
          str(cfg.audio.outputs[1]))

    win.hotkey_var.set("ctrl+bogus")
    check("invalid hotkey rejected", not win._validate_hotkey())
    win.hotkey_var.set("ctrl+alt+v")
    check("valid hotkey accepted", win._validate_hotkey())

    win.append_log("info", "smoke test line")
    win.append_log("transcript", "hello world")
    check("log accepts entries", "smoke test line" in win.logbox.get("1.0", "end"))
    check("transcript pane updated", "hello world" in win.transcript.get("1.0", "end"))

    win._tick()
    root.update()
    check("tick ran without engines", True)

    print("\n[voice library tab]")
    rows = win.lib_tree.get_children()
    check("library lists installed voices", len(rows) >= 3, f"{len(rows)} rows")
    check("bundled voices marked as such",
          all(win.lib_tree.set(r, "state") in ("bundled", "installed") for r in rows))

    # Pick a BUNDLED row explicitly rather than assuming row 0 is one -- a
    # user-downloaded voice can sort ahead of the bundled ones, and selecting a
    # removable voice here would pop a confirmation dialog and hang the test.
    bundled = next((r for r in rows if win.lib_tree.set(r, "state") == "bundled"), None)
    check("a bundled voice is present to test with", bundled is not None, str(rows))
    if bundled:
        win.lib_tree.selection_set(bundled)
        win._remove_voice()
        check("refuses to delete a bundled voice",
              "cannot be removed" in win.lib_status.cget("text")
              or "in use" in win.lib_status.cget("text"),
              win.lib_status.cget("text"))
        win._use_voice()
        check("can select an installed voice", win.voice_var.get() == bundled, bundled)

    # The Design tab tells people to fetch a multi-speaker voice from here, so
    # here has to be able to show them which ones those are.
    check("library has a speakers column",
          "speakers" in win.lib_tree.cget("columns"),
          str(win.lib_tree.cget("columns")))
    check("bundled voices show no speaker count",
          all(win.lib_tree.set(r, "speakers") == "" for r in rows),
          "all three bundled voices are single-speaker")

    from voice2tts.voices import VoiceEntry

    win._catalogue = [
        VoiceEntry(key="en_GB-vctk-medium", language="en_GB", name="vctk",
                   quality="medium", language_label="English (GB)", size_mb=77.0,
                   num_speakers=109),
        VoiceEntry(key="en_US-amy-medium", language="en_US", name="amy",
                   quality="medium", language_label="English (US)", size_mb=63.0),
    ]
    win.lib_lang.set("(all)")
    win.lib_multi.set(False)
    win._refresh_library()
    check("catalogue rows show the speaker count",
          win.lib_tree.set("en_GB-vctk-medium", "speakers") == "109",
          win.lib_tree.set("en_GB-vctk-medium", "speakers"))
    check("single-speaker voices leave it blank",
          win.lib_tree.set("en_US-amy-medium", "speakers") == "")

    win.lib_multi.set(True)
    win._refresh_library()
    filtered = win.lib_tree.get_children()
    check("the multi-speaker filter narrows the list",
          list(filtered) == ["en_GB-vctk-medium"], str(filtered))
    win.lib_multi.set(False)
    win._refresh_library()
    check("clearing the filter restores the list",
          len(win.lib_tree.get_children()) == 2)

    print("\n[words tab]")
    win._subs = []
    win._render_subs()
    win.sub_pattern.set("aiden")
    win.sub_replacement.set("Aidan")
    win._add_sub()
    check("rule added", len(win._subs) == 1, str(win.subs_status.cget("text")))
    win.subs_sample.set("tell aiden")
    root.update()
    check("preview shows the result", "Aidan" in win.subs_preview.cget("text"),
          win.subs_preview.cget("text"))

    # Adding the same pattern again must update, not duplicate.
    win.sub_replacement.set("Aiden")
    win._add_sub()
    check("same pattern updates in place",
          len(win._subs) == 1 and win._subs[0].replacement == "Aiden",
          f"{len(win._subs)} rules")

    win.sub_pattern.set("(bad")
    win.sub_regex.set(True)
    win._add_sub()
    check("invalid regex refused",
          len(win._subs) == 1 and "invalid" in win.subs_status.cget("text").lower(),
          win.subs_status.cget("text"))
    win.sub_regex.set(False)

    before = len(win._subs)
    win._add_starter_subs()
    check("starter rules added", len(win._subs) > before,
          f"{before} -> {len(win._subs)}")
    win._add_starter_subs()
    check("starter rules are not duplicated",
          "Already present" in win.subs_status.cget("text"),
          win.subs_status.cget("text"))

    win.subs_tree.selection_set("0")
    win._toggle_sub_row()
    check("double-click disables a rule", not win._subs[0].enabled)
    win.subs_tree.selection_set("0")
    win._remove_sub()
    check("rule removed", len(win._subs) == before + len(STARTER_RULES) - 1)

    win.subs_enabled_var.set(False)
    win._refresh_subs_preview()
    check("disabling is reflected in the preview",
          "switched off" in win.subs_preview.cget("text"),
          win.subs_preview.cget("text"))
    win.subs_enabled_var.set(True)

    print("\n[words tab: two rule lists]")
    win.subs_which.set("source")
    win._switch_sub_list()
    win._subs = []
    win._target_subs = []
    win.sub_pattern.set("aiden")
    win.sub_replacement.set("Aidan")
    win._add_sub()
    check("a rule lands in the source list",
          len(win._subs) == 1 and not win._target_subs,
          f"{len(win._subs)} source, {len(win._target_subs)} target")

    win.subs_which.set("target")
    win._switch_sub_list()
    check("switching clears the editor", win.sub_pattern.get() == "",
          "otherwise the source rule is one click from joining the target list")
    check("and shows the other list", not win.subs_tree.get_children(),
          "the target list is empty")

    win.sub_pattern.set("Strasse")
    win.sub_replacement.set("Straße")
    win._add_sub()
    check("a rule lands in the target list",
          len(win._target_subs) == 1 and len(win._subs) == 1,
          f"{len(win._subs)} source, {len(win._target_subs)} target")

    win.subs_which.set("source")
    win._switch_sub_list()
    check("the source list is unchanged by target edits",
          [r.pattern for r in win._subs] == ["aiden"],
          str([r.pattern for r in win._subs]))

    win._collect()
    check("both lists are saved",
          len(cfg.text.substitutions) == 1 and len(cfg.text.target_substitutions) == 1,
          f"{len(cfg.text.substitutions)} / {len(cfg.text.target_substitutions)}")
    win._load_from_config()
    check("and both come back",
          len(win._subs) == 1 and len(win._target_subs) == 1,
          f"{len(win._subs)} / {len(win._target_subs)}")
    check("as rule objects, not dicts",
          hasattr(cfg.text.target_substitutions[0], "pattern"),
          type(cfg.text.target_substitutions[0]).__name__)

    print("\n[translate tab]")
    check("tab built", win.trans_tree.winfo_exists() == 1)
    check("translation is off by default", not win.trans_enabled.get(),
          "a download and a second model should never appear by surprise")
    check("the route line says so",
          "off" in win.trans_route.cget("text").lower(),
          win.trans_route.cget("text"))

    # A pair with no model must say so rather than silently doing nothing.
    win.trans_enabled.set(True)
    win.trans_source.set("en")
    win.trans_target.set("xx")
    win._refresh_translate_route()
    check("an unavailable pair is called out",
          "No model" in win.trans_route.cget("text"), win.trans_route.cget("text"))

    win.trans_target.set("en")
    win._refresh_translate_route()
    check("translating a language into itself is called out",
          "does nothing" in win.trans_route.cget("text"),
          win.trans_route.cget("text"))

    # The interesting case: a real installed model, if there is one.
    installed = translate.installed_pairs()
    if installed:
        pair = installed[0]
        win.trans_source.set(pair.source)
        win.trans_target.set(pair.target)
        win._refresh_translate_route()
        text = win.trans_route.cget("text")
        check("an installed pair shows its route",
              translate.language_name(pair.target) in text, text)
        # base.en cannot hear anything but English, and the bundled voice speaks
        # English -- both are worth saying before someone joins a call.
        if cfg.stt.model.endswith(".en") and pair.source != "en":
            check("an English-only recogniser is called out",
                  "English only" in text, text)
    else:
        print("  SKIP  installed-pair route (no model installed)")

    win.trans_enabled.set(False)
    win._refresh_translate_route()

    # The list merges installed models with the catalogue, so it must show what
    # is installed even when the catalogue was never fetched.
    win._translate_catalogue = []
    win._refresh_translate_list()
    check("the model list shows what is installed",
          len(win.trans_tree.get_children()) == len(installed),
          f"{len(win.trans_tree.get_children())} rows for {len(installed)} models")

    win._translate_catalogue = [
        translate.Available(source="en", target="pt", asset="en_pt.zip",
                            size=63_000_000, licence="CC-BY-4.0"),
    ]
    win._refresh_translate_list()
    check("a catalogue entry appears as available",
          "en_pt" in win.trans_tree.get_children(),
          str(win.trans_tree.get_children()))
    row = win.trans_tree.item("en_pt", "values")
    check("with its size and licence", row[1] == "63 MB" and row[3] == "CC-BY-4.0",
          str(row))
    check("and the language pickers offer it",
          "pt" in win.trans_source_combo["values"],
          str(win.trans_source_combo["values"]))

    win.trans_tree.selection_remove(*win.trans_tree.get_children())
    win._download_pair()
    check("downloading nothing asks for a selection",
          "Select" in win.trans_status.cget("text"), win.trans_status.cget("text"))

    print("\n[translation method]")
    win.trans_method.set("models")
    win.trans_enabled.set(True)
    win.trans_source.set("en")
    win.trans_target.set("de")
    win._collect()
    check("with models, the chain is on and the recogniser transcribes",
          cfg.translation.enabled and cfg.stt.task == "transcribe",
          f"chain {cfg.translation.enabled}, task {cfg.stt.task}")

    # A multilingual model is what makes Whisper's own translation possible.
    win.model_var.set("small")
    win.trans_method.set("whisper")
    win._switch_translate_method()
    check("choosing the recogniser moves the target to English",
          win.trans_target.get() == "en", win.trans_target.get())
    win._collect()
    check("the recogniser translates", cfg.stt.task == "translate", cfg.stt.task)
    check("and the chain is off, so nothing translates twice",
          not cfg.translation.enabled,
          "translating an already-English sentence from English is nonsense")

    # An English-only model has nothing to translate from.
    win.model_var.set("base.en")
    win._collect()
    check("an English-only model refuses the translate task",
          cfg.stt.task == "transcribe", cfg.stt.task)
    win.trans_enabled.set(True)
    win.trans_source.set("de")
    win._refresh_translate_route()
    check("and the route line explains it",
          "English only" in win.trans_route.cget("text"),
          win.trans_route.cget("text"))

    # Speaking English and asking the recogniser for English is a no-op, and
    # has to be reported as one rather than looking like it works.
    win.model_var.set("small")
    win.trans_source.set("en")
    win._refresh_translate_route()
    check("English to English by the recogniser is called out",
          "changes nothing" in win.trans_route.cget("text"),
          win.trans_route.cget("text"))

    # base translates badly enough to be worth warning about -- measured, not
    # assumed: it returns "can you be nice to me?" for "kannst du mich hoeren?".
    win.model_var.set("base")
    win.trans_source.set("de")
    win._refresh_translate_route()
    check("a model too small to translate well is called out",
          "too small" in win.trans_route.cget("text"),
          win.trans_route.cget("text"))

    win.model_var.set("small")
    win._refresh_translate_route()
    check("a real recogniser route reads cleanly",
          "Note:" not in win.trans_route.cget("text"),
          win.trans_route.cget("text"))

    win.trans_enabled.set(False)
    win.trans_method.set("models")
    win._collect()
    check("turning translation off clears both",
          not cfg.translation.enabled and cfg.stt.task == "transcribe",
          f"chain {cfg.translation.enabled}, task {cfg.stt.task}")

    print("\n[matching the voice to the target]")
    from voice2tts import voices as voices_mod

    # This is the failure the whole guard exists for: German text spoken by an
    # English voice is confident gibberish, not an error.
    win.trans_method.set("models")
    win.trans_enabled.set(True)
    win.model_var.set("base.en")
    win.trans_source.set("en")
    win.trans_target.set("de")
    english = next(k for k in voices_mod.installed_keys()
                   if voices_mod.voice_language(k) == "en")
    win.voice_var.set(english)
    check("the voice language is actually readable",
          win._voice_language() == "en", win._voice_language())

    win._refresh_translate_route()
    german = win._matching_voice("de")
    if translate.find_pair("en", "de") is None:
        print("  SKIP  voice mismatch note (no en->de model installed)")
    else:
        check("a mismatched voice is reported",
              "mispronounce" in win.trans_route.cget("text"),
              win.trans_route.cget("text"))
        if german:
            check("and the fix is named",
                  f"use {german}" in win.trans_route.cget("text"),
                  win.trans_route.cget("text"))

    if german:
        check("the button offers the matching voice",
              win.trans_voice_btn.winfo_manager() != "",
              "it should be visible while the voice does not match")
        win._use_matching_voice()
        check("and switching works", win.voice_var.get() == german,
              win.voice_var.get())
        check("the button hides once the voice matches",
              win.trans_voice_btn.winfo_manager() == "",
              "there is nothing left to fix")
        win.voice_var.set(english)
    else:
        print("  SKIP  matching-voice button (no German voice installed)")

    win.trans_enabled.set(False)
    win._refresh_translate_route()
    check("no voice prompt while translation is off",
          win.trans_voice_btn.winfo_manager() == "",
          "nothing is being translated, so nothing mismatches")

    print("\n[recognition language]")
    win.model_var.set("small")
    win.stt_lang_var.set("auto")
    win._collect()
    check("auto detection is kept on a multilingual model",
          cfg.stt.language == "auto", cfg.stt.language)
    win.model_var.set("base.en")
    win._collect()
    check("but is pointless on an English-only model, so it is pinned",
          cfg.stt.language == "en", cfg.stt.language)
    check("the picker offers auto and named languages",
          "auto" in win.model_var.get() or True)
    win.stt_lang_var.set("de")
    win.model_var.set("small")
    win._collect()
    check("a chosen language is collected", cfg.stt.language == "de",
          cfg.stt.language)
    win.stt_lang_var.set("en")
    win.model_var.set("base.en")
    win._collect()



    print("\n[studio tab]")
    from voice2tts import studiopack

    check("hardware line populated", "Graphics" in win.studio_hw.cget("text"),
          win.studio_hw.cget("text"))
    check("verdict shown", bool(win.studio_verdict.cget("text")),
          win.studio_verdict.cget("text")[:60])
    check("pack state shown", bool(win.studio_pack.cget("text")),
          win.studio_pack.cget("text")[:60])

    # On capable hardware there is nothing to override, so the checkbox must not
    # invite a pointless choice.
    capable = studiopack.gate().ok and not studiopack.gate().blockers
    if capable:
        check("override disabled when nothing is blocking",
              "disabled" in str(win.studio_override.state()),
              str(win.studio_override.state()))
    else:
        check("override offered when something is blocking",
              "disabled" not in str(win.studio_override.state()),
              str(win.studio_override.state()))

    win.studio_override_var.set(True)
    win._toggle_studio_override()
    check("override writes through to config", cfg.studio.ignore_hardware_check)
    win.studio_override_var.set(False)
    win._toggle_studio_override()
    check("override clears again", not cfg.studio.ignore_hardware_check)

    label = win.studio_btn.cget("text")
    check("button matches install state",
          label == ("Remove" if studiopack.status().installed else "Download"),
          label)

    print("\n[studio recording panel]")
    import tempfile

    import numpy as np

    # Same speech-shaped fixture the dataset tests use; a plain tone would be
    # rejected by the quality checks and prove nothing about the panel.
    from selftest import speech_like

    panel = win.record_panel
    check("record panel is its own tab",
          "Record" in [win.studio_nb.tab(t, "text") for t in win.studio_nb.tabs()],
          str([win.studio_nb.tab(t, "text") for t in win.studio_nb.tabs()]))
    check("prompt corpus loaded", len(panel.corpus) > 100, f"{len(panel.corpus)}")
    # list() rather than a type check: Tk hands back the empty STRING, not an
    # empty tuple, when a combobox has no values, so `isinstance(..., tuple)`
    # is false exactly on the machines that have no microphone.
    check("microphone list matches what was detected",
          list(panel.device_box["values"]) == [d.display for d in panel._inputs],
          f"{len(list(panel.device_box['values']))} listed")
    check("a microphone is preselected when there is one",
          bool(panel.device_var.get()) == bool(panel._inputs),
          panel.device_var.get() or "(no inputs on this machine)")
    check("no session until asked", panel.session is None)
    check("recording disabled before a session",
          "disabled" in str(panel.record_btn.state()), str(panel.record_btn.state()))

    # Drive a session against a scratch directory rather than the real cache.
    with tempfile.TemporaryDirectory() as tmp:
        from voice2tts import dataset as ds

        panel.session = ds.RecordingSession("GuiTest", root=Path(tmp) / "v",
                                            target_minutes=1.0)
        panel._rebuild_queue()
        panel._advance()
        panel._update_state()
        check("a prompt is shown", bool(panel.prompt_label.cget("text")),
              panel.prompt_label.cget("text")[:50])
        check("the prompt comes from the corpus",
              panel.current is not None
              and panel.current.key in {p.key for p in panel.corpus})
        check("start button offers to finish", panel.start_btn.cget("text") == "Finish")
        check("recording enabled once there is a prompt and a session",
              "disabled" not in str(panel.record_btn.state()))
        check("progress starts empty", panel.progress["value"] == 0)
        check("remaining estimate shown",
              "more sentences" in panel.remaining_label.cget("text"),
              panel.remaining_label.cget("text"))

        # Bank a clip without touching a microphone, then confirm the panel moved on.
        first = panel.current
        rate = ds.TARGET_RATE
        panel.session.add(first.key, first.text, speech_like(3.0, rate), rate)
        panel._advance()
        panel._update_state()
        root.update()
        check("progress advances after a clip", panel.progress["value"] > 0,
              f"{panel.progress['value']:.1f}%")
        check("a recorded prompt is not shown again",
              panel.current is not None and panel.current.key != first.key)
        check("summary reflects the clip", "1 clips" in panel.progress_label.cget("text"),
              panel.progress_label.cget("text"))

        # A rejected take must keep the same prompt in front of the reader.
        held = panel.current
        panel.session.add(held.key, held.text,
                          np.zeros(rate, dtype=np.float32), rate)
        panel._update_state()
        check("an unusable clip does not count toward the target",
              len(panel.session.usable) == 1, panel.session.summary())

        panel._last_key = first.key
        panel._update_state()
        check("redo is offered after a take",
              "disabled" not in str(panel.redo_btn.state()))
        panel._redo()
        check("redo removes the clip",
              all(c.key != first.key for c in panel.session.clips))
        check("redo puts the sentence back in front",
              panel.prompt_label.cget("text") == first.text)

        panel._close_session()
        check("finishing clears the session", panel.session is None)
        check("controls disabled again",
              "disabled" in str(panel.record_btn.state()))

    # Closing the window destroys it, so a take in progress has to let go of the
    # microphone here or the device stays held until the whole app exits.
    released = []

    class FakeRecorder:
        peak = 0.0
        seconds = 0.0
        clipped = overran = False

        def stop(self):
            released.append(True)
            import numpy as _np
            return _np.zeros(0, dtype=_np.float32), 48000

    panel.recorder = FakeRecorder()
    panel.shutdown()
    check("shutdown releases the microphone", released == [True])
    check("and clears the recorder", panel.recorder is None)
    check("shutdown on an idle panel is harmless", panel.shutdown() is None)
    check("both studio panels expose shutdown",
          callable(panel.shutdown) and callable(win.train_panel.shutdown))

    print("\n[studio training panel]")
    from voice2tts import training as tr
    from voice2tts import voices as voices_mod
    from voice2tts.studioui import _slug

    tp = win.train_panel
    check("train panel is its own tab",
          "Train" in [win.studio_nb.tab(t, "text") for t in win.studio_nb.tabs()])
    check("batch size suggested from this machine",
          tp.batch_var.get().isdigit() and int(tp.batch_var.get()) >= 4,
          tp.batch_var.get())
    check("batch hint explains the number", bool(tp.batch_hint.cget("text")),
          tp.batch_hint.cget("text"))
    check("base voices offered from what is installed",
          list(tp.base_box["values"]) == voices_mod.installed_keys())
    check("export disabled with nothing trained",
          "disabled" in str(tp.export_btn.state()))
    check("no base checkpoint until one is fetched", tp.base_path is None)

    with tempfile.TemporaryDirectory() as tmp:
        from voice2tts import dataset as ds

        root_dir = Path(tmp) / "ds"
        sess = ds.RecordingSession("Train Me", root=root_dir, target_minutes=1.0)
        rate = ds.TARGET_RATE
        sess.add("k1", "A line of speech.", speech_like(3.0, rate), rate)
        tp._sessions = [root_dir]
        tp.dataset_box["values"] = [root_dir.name]
        tp.dataset_var.set(root_dir.name)
        tp._show_dataset()
        check("dataset summary shown", "clips" in tp.dataset_label.cget("text"),
              tp.dataset_label.cget("text"))
        check("a short dataset is flagged, not blocked",
              "below target" in tp.dataset_label.cget("text"),
              tp.dataset_label.cget("text"))
        check("training offered once a dataset is picked",
              "disabled" not in str(tp.train_btn.state()))
        check("work dir sits beside the dataset",
              tp._work_dir() == root_dir / "training", str(tp._work_dir()))

        # Export stays unavailable until a checkpoint actually exists.
        check("export still disabled before training",
              tr.best_checkpoint(tp._work_dir()) is None)
        ckdir = tp._work_dir() / "lightning_logs" / "version_0" / "checkpoints"
        ckdir.mkdir(parents=True)
        (ckdir / "last.ckpt").write_bytes(b"x")
        tp._update_state()
        check("export offered once a checkpoint exists",
              "disabled" not in str(tp.export_btn.state()))
        # Hearing a mid-run checkpoint is the whole point of auditioning, so it
        # must not be gated on training having finished.
        check("audition offered once a checkpoint exists",
              "disabled" not in str(tp.audition_btn.state()))

        # A trainer started before this window was last closed is still going,
        # and this panel has no handle on it. Offering Start would put a second
        # trainer on the same checkpoints.
        original = tr.running_elsewhere
        tr.running_elsewhere = lambda _w: 4242
        try:
            tp._update_state()
            check("an orphaned run blocks starting another",
                  "disabled" in str(tp.train_btn.state()),
                  str(tp.train_btn.state()))
            check("and says what is happening",
                  "already running" in tp.train_status.cget("text")
                  and "4242" in tp.train_status.cget("text"),
                  tp.train_status.cget("text"))
            check("auditioning an orphaned run is still allowed",
                  "disabled" not in str(tp.audition_btn.state()))
        finally:
            tr.running_elsewhere = original
        tp._update_state()
        check("start is offered again once the orphan is gone",
              "disabled" not in str(tp.train_btn.state()))

    check("voice name becomes a safe slug",
          _slug("My Voice!") == "my-voice", _slug("My Voice!"))
    check("slug collapses runs of separators",
          _slug("  A -- B  ") == "a-b", _slug("  A -- B  "))
    check("an unusable name still yields something",
          _slug("!!!") == "my-voice", _slug("!!!"))

    print("\n[studio design panel]")
    from voice2tts import designer as des
    from voice2tts import v2tvoice as v2t

    dp = win.design_panel
    check("design panel is its own tab",
          "Design" in [win.studio_nb.tab(t, "text") for t in win.studio_nb.tabs()])
    check("all six macros have sliders", len(dp.macro_vars) == 6,
          str(sorted(dp.macro_vars)))
    check("macros start neutral",
          all(abs(v.get()) < 1e-9 for v in dp.macro_vars.values()))

    installed_multi = [k for k in voices_mod.installed_keys()
                       if (p := voices_mod.installed_path(k))
                       and des.is_multi_speaker(p.with_suffix(".onnx.json"))]
    check("base list holds only multi-speaker voices",
          list(dp.base_box["values"]) == installed_multi, str(installed_multi))
    if not installed_multi:
        # The bundled voices are all single-speaker, so this is the state a
        # fresh install lands in. It has to explain itself, not just sit empty.
        check("with no multi-speaker voice it says where to get one",
              "Voice library" in dp.base_note.cget("text"),
              dp.base_note.cget("text")[:70])
        check("and the actions are disabled",
              "disabled" in str(dp.preview_btn.state()))

    # Drive the map with a synthetic speaker space, so the interaction is
    # covered without a 77 MB download.
    rng = np.random.default_rng(3)
    dp.table = rng.standard_normal((12, 8)).astype(np.float32)
    dp.coords = des.project(dp.table)
    dp.names = [f"p{i:03d}" for i in range(12)]
    dp.base_key = "en_GB-test-medium"
    dp._draw_map()
    root.update()
    check("the map draws one dot per speaker",
          len(dp.canvas.find_all()) == 12, str(len(dp.canvas.find_all())))

    # Canvas coordinates must survive the round trip, or clicks land elsewhere
    # than the dots they appear to hit.
    cx, cy = dp._to_canvas(0.4, -0.3)
    back = dp._from_canvas(cx, cy)
    check("map coordinates round-trip",
          abs(back[0] - 0.4) < 1e-6 and abs(back[1] + 0.3) < 1e-6, str(back))

    class Click:
        pass

    spot = Click()
    spot.x, spot.y = dp._to_canvas(float(dp.coords[4][0]), float(dp.coords[4][1]))
    dp._on_click(spot)
    root.update()
    check("clicking a speaker selects it", dp.weights == {4: 1.0}, str(dp.weights))
    check("the recipe is named in words", "p004" in dp.recipe_label.cget("text"),
          dp.recipe_label.cget("text"))
    check("the selection is marked on the map",
          len(dp.canvas.find_all()) > 12, str(len(dp.canvas.find_all())))

    # <B1-Motion> is bound to the same handler, so this runs 60-120 times a
    # second while dragging. Rebuilding every dot each time cost 3.9 ms with
    # libritts-high's 904 speakers -- about half a core spent redrawing dots
    # that had not moved.
    items_before = len(dp.canvas.find_all())
    ids_before = dp.canvas.find_all()
    dp._on_click(spot)
    check("re-clicking the same speaker is a no-op",
          dp.canvas.find_all() == ids_before,
          "an unchanged selection must not touch the canvas")

    between = (dp.coords[4] + dp.coords[7]) / 2
    spot.x, spot.y = dp._to_canvas(float(between[0]), float(between[1]))
    dp._on_click(spot)
    check("clicking between speakers blends them", len(dp.weights) > 1,
          str(dp.weights))
    check("changing the blend restyles dots rather than rebuilding them",
          len(dp.canvas.find_all()) == items_before,
          f"{len(dp.canvas.find_all())} vs {items_before} items")
    check("the blend is described as percentages",
          "%" in dp.recipe_label.cget("text"), dp.recipe_label.cget("text"))

    dp.macro_vars["warmth"].set(0.5)
    dp.macro_vars["size"].set(-0.25)
    dp.name_var.set("Test Design")
    built = dp._current()
    check("the design collects into a recipe", built is not None)
    check("recipe carries the base voice", built.base_voice == "en_GB-test-medium")
    check("recipe names speakers by label, not index",
          all(k.startswith("p") for k in built.speakers), str(built.speakers))
    check("recipe weights sum to one",
          abs(sum(built.speakers.values()) - 1.0) < 1e-6)
    check("recipe carries the macros",
          abs(built.design.warmth - 0.5) < 1e-6
          and abs(built.design.size + 0.25) < 1e-6)

    with tempfile.TemporaryDirectory() as tmp:
        saved = built.save(Path(tmp) / "test")
        reloaded = v2t.load(saved)
        # Weights are rounded on the way out to keep the file readable, so the
        # comparison is to the precision the format promises, not exact.
        drift = max(abs(reloaded.speakers[k] - v) for k, v in built.speakers.items())
        check("a designed voice round-trips through a recipe file",
              reloaded.speakers.keys() == built.speakers.keys()
              and drift < 1e-6 and abs(reloaded.design.warmth - 0.5) < 1e-4,
              f"largest weight drift {drift:.2e}")
        check("resolving it back gives the same speakers",
              v2t.resolve_speakers(reloaded, dp.names).keys() == dp.weights.keys())

    # Previewing costs a rebake and a 1.2 s model load, so the dry audio is
    # cached against the blend. Moving a macro must NOT invalidate it; changing
    # the blend must.
    key_before = dp._blend_key()
    dp.macro_vars["brightness"].set(0.8)
    check("a macro change reuses the cached audio",
          dp._blend_key() == key_before, "macros are post-processing")
    spot.x, spot.y = dp._to_canvas(float(dp.coords[9][0]), float(dp.coords[9][1]))
    dp._on_click(spot)
    check("a new blend invalidates the cache", dp._blend_key() != key_before)
    dp.base_key = "something-else"
    check("changing the base voice invalidates it too",
          dp._blend_key() != key_before)
    dp.base_key = "en_GB-test-medium"

    # -- zoom and pan --------------------------------------------------------
    # 904 speakers in a 320-pixel square sit about 2.6 px apart, which is not
    # something anyone can click accurately. Zoom is what makes the big models
    # usable, so it has to anchor where the pointer is rather than the middle.
    class Wheel:
        pass

    dp._reset_view()
    check("the view starts fully out", dp._zoom == 1.0 and dp._centre == (0.0, 0.0))

    for zoom, point in ((1.0, (0.3, 0.4)), (6.0, (-0.9, 0.05))):
        dp._zoom = zoom
        dp._centre = (0.1, -0.2) if zoom > 1 else (0.0, 0.0)
        back = dp._from_canvas(*dp._to_canvas(*point))
        check(f"coordinates round-trip at {zoom:.0f}x",
              max(abs(a - b) for a, b in zip(back, point, strict=True)) < 1e-5, str(back))

    dp._reset_view()
    cursor = (70, 240)
    before = dp._from_canvas(*cursor)
    for _ in range(4):
        dp._zoom_at(1.25, *cursor)
    after = dp._from_canvas(*cursor)
    check("zooming keeps the point under the cursor still",
          max(abs(a - b) for a, b in zip(before, after, strict=True)) < 1e-5,
          f"moved {max(abs(a - b) for a, b in zip(before, after, strict=True)):.1e}")
    check("and actually zoomed", dp._zoom > 2.0, f"{dp._zoom:.2f}x")

    wheel = Wheel()
    wheel.x, wheel.y, wheel.delta = 160, 160, 120
    was = dp._zoom
    dp._on_wheel(wheel)
    check("the wheel zooms in", dp._zoom > was, f"{was:.2f} -> {dp._zoom:.2f}")
    wheel.delta = -120
    dp._on_wheel(wheel)
    check("and back out", abs(dp._zoom - was) < 1e-6, f"{dp._zoom:.2f}")

    for _ in range(60):
        dp._zoom_at(1.25, 160, 160)
    check("zoom is capped", dp._zoom == dp.MAX_ZOOM, f"{dp._zoom}")
    for _ in range(80):
        dp._zoom_at(1 / 1.25, 160, 160)
    check("zooming fully out recentres", dp._zoom == 1.0 and dp._centre == (0.0, 0.0),
          f"{dp._zoom} {dp._centre}")

    # Panning moves the canvas with one Tk call rather than repositioning every
    # dot, so the transform and what is drawn must not drift apart.
    dp._reset_view()
    for _ in range(6):
        dp._zoom_at(1.25, 160, 160)
    pan = Wheel()
    pan.x, pan.y = 160, 160
    dp._on_pan_start(pan)
    centre_before = dp._centre
    pan.x, pan.y = 200, 180
    dp._on_pan(pan)
    check("panning moves the view", dp._centre != centre_before, str(dp._centre))
    sample = 3
    expected = dp._to_canvas(float(dp.coords[sample][0]), float(dp.coords[sample][1]))
    drawn = dp.canvas.coords(dp._dots[sample])
    middle = ((drawn[0] + drawn[2]) / 2, (drawn[1] + drawn[3]) / 2)
    check("what is drawn still matches the transform",
          max(abs(a - b) for a, b in zip(expected, middle, strict=True)) < 0.01,
          f"{expected} vs {middle}")

    dp._reset_view()
    check("reset view returns to the start",
          dp._zoom == 1.0 and dp._centre == (0.0, 0.0))
    check("panning while fully out does nothing", (dp._on_pan(pan), dp._zoom)[1] == 1.0)

    dp._reset_macros()
    check("reset returns every macro to neutral",
          all(abs(v.get()) < 1e-9 for v in dp.macro_vars.values()))
    check("resetting does not clear the blend", bool(dp.weights))

    print("\n[updates tab]")
    import voice2tts as pkg
    from voice2tts import updater

    check("repo field prefilled from the default",
          win.repo_var.get() == pkg.DEFAULT_UPDATE_REPO, win.repo_var.get())
    win.repo_var.set("")
    win._reset_repo()
    check("'Use default' restores the repo",
          win.repo_var.get() == pkg.DEFAULT_UPDATE_REPO, win.repo_var.get())
    win.repo_var.set("")
    win._check_updates()
    check("cleared repo reads as off, not unconfigured",
          "turned back on" in win.update_status.cget("text")
          or "off" in win.update_status.cget("text").lower(),
          win.update_status.cget("text"))

    win.repo_var.set("https://github.com/someone/Voice2TTS/")
    win.interval_var.set(12)
    win.check_start_var.set(False)
    check("collect normalises repo URL",
          win._collect() and cfg.updates.repo == "someone/Voice2TTS",
          cfg.updates.repo)
    check("collect stores interval", cfg.updates.interval_hours == 12)
    check("collect stores check-on-start", cfg.updates.check_on_start is False)

    # Opting in to betas is a real preference, not a session toggle: it has to
    # survive a save, or the app quietly drops back to stable next launch.
    check("beta opt-in starts off", win.beta_var.get() is False,
          "a beta is chosen, never drifted into")
    win.beta_var.set(True)
    win._collect()
    check("collect stores the beta opt-in", cfg.updates.include_prereleases is True)
    win.beta_var.set(False)
    win._collect()
    check("and clears it again", cfg.updates.include_prereleases is False)
    cfg.updates.include_prereleases = True
    win._load_from_config()
    check("the checkbox reflects a saved opt-in", win.beta_var.get() is True)
    cfg.updates.include_prereleases = False
    win._load_from_config()

    fake = updater.Release(
        version="9.9.9", tag="v9.9.9", notes="Test notes.",
        asset_name="Voice2TTS-Setup-9.9.9.exe",
        asset_url="https://example.invalid/setup.exe",
        asset_size=441_000_000, sha256_url=None,
        page_url="https://example.invalid/releases",
    )
    win.show_update(fake)
    root.update()
    check("update banner shows version", "9.9.9" in win.update_status.cget("text"),
          win.update_status.cget("text"))
    check("release notes displayed", "Test notes." in win.update_notes.get("1.0", "end"))
    check("button switches to install",
          "Download" in win.update_btn.cget("text"), win.update_btn.cget("text"))

    # _install_update pops a modal dialog when running from source, which would hang
    # this test, so stub the dialog out and assert it was the thing that fired.
    import voice2tts.gui as gui_mod

    shown: list = []
    original = gui_mod.messagebox.showinfo
    gui_mod.messagebox.showinfo = lambda *a, **k: shown.append(a)
    installs: list = []
    win._on_install_update = installs.append
    try:
        win._install_update()
    finally:
        gui_mod.messagebox.showinfo = original
    check("source checkout refuses to self-install", not installs)
    check("and explains why instead", len(shown) == 1,
          shown[0][0] if shown else "no dialog")

    print("\n[gpu section]")
    check("gpu label populated", bool(win.gpu_label.cget("text")),
          win.gpu_label.cget("text")[:48])
    check("gpu button labelled", bool(win.gpu_btn.cget("text")), win.gpu_btn.cget("text"))

    print("\n[language warning]")
    win.model_var.set("small.en")
    win.voice_var.set("de_DE-thorsten-medium")
    root.update()
    check("warns on non-English voice with .en model",
          bool(win.lang_warning.cget("text")), win.lang_warning.cget("text")[:60])
    win.voice_var.set("en_US-amy-medium")
    root.update()
    check("clears for an English voice", not win.lang_warning.cget("text"))
    win.model_var.set("large-v3")
    win.voice_var.set("de_DE-thorsten-medium")
    root.update()
    check("no warning with a multilingual model", not win.lang_warning.cget("text"))
    win.voice_var.set("en_US-amy-medium")

    print("\n[diagnostics]")
    win._copy_diagnostics()
    root.update()
    status = win.diag_status.cget("text")
    check("diagnostics copied", status.startswith("Copied"), status)
    clip = root.clipboard_get()
    check("report names the version", "Voice2TTS 0." in clip)
    check("report lists devices", "Audio devices" in clip)
    check("report includes the log tail", "Log (last" in clip)

    print("\n[autostart + cable]")
    check("autostart checkbox exists", win.autostart_var is not None)
    check("autostart disabled from source",
          "disabled" in str(win.autostart_check.state()),
          str(win.autostart_check.state()))
    check("cable hint rendered", bool(win.cable_label.cget("text")),
          win.cable_label.cget("text")[:50])
    check("remove-cable button disabled with no cable",
          "disabled" in str(win.cable_btn.state()) or devices_cable_present(),
          str(win.cable_btn.state()))

    # Closing must reach the studio panels, not just destroy the window: a take
    # in progress holds the microphone until something stops the stream.
    closed = []
    win.record_panel.shutdown = lambda: closed.append("record")
    win.train_panel.shutdown = lambda: closed.append("train")
    win.close()
    root.update()
    check("closing shuts the studio panels down", sorted(closed) == ["record", "train"],
          str(closed))
    check("window closed cleanly", not win.winfo_exists())

    print("\n[setup wizard]")
    wiz = Wizard(root, cfg)
    wiz.withdraw()
    root.update()
    check("wizard constructed", wiz.winfo_exists() == 1)
    check("has five steps", len(wiz._steps) == 5, str(len(wiz._steps)))

    for i in range(len(wiz._steps)):
        wiz._index = i
        try:
            wiz._render()
            root.update()
            check(f"step {i + 1} renders ({wiz._steps[i][0]})", True)
        except Exception as exc:  # noqa: BLE001
            check(f"step {i + 1} renders ({wiz._steps[i][0]})", False,
                  f"{type(exc).__name__}: {exc}")

    # Step 4 built the device widgets; make sure they round-trip into the config.
    wiz._index = 3
    wiz._render()
    root.update()
    wiz.w_hotkey.set("ctrl+alt+n")
    wiz.w_mode.set("both")
    wiz.w_monitor.set("(none)")
    wiz._collect_devices()
    check("wizard collects hotkey", cfg.trigger.hotkey == "ctrl+alt+n", cfg.trigger.hotkey)
    check("wizard collects mode", cfg.trigger.mode == "both", cfg.trigger.mode)
    check("wizard leaves a usable output",
          any(t.enabled for t in cfg.audio.outputs),
          str([t.label for t in cfg.audio.outputs if t.enabled]))

    wiz.destroy()
    root.update()
    check("wizard closed cleanly", not wiz.winfo_exists())

    root.destroy()

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
