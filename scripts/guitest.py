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
    check("input combo populated", len(win.input_combo["values"]) > 1)
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
    check("microphone list populated or machine has none",
          isinstance(panel.device_box["values"], (tuple, list)))
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
