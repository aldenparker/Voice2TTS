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
    win.lib_tree.selection_set(rows[0])
    win._remove_voice()
    check("refuses to delete a bundled voice",
          "cannot be removed" in win.lib_status.cget("text")
          or "in use" in win.lib_status.cget("text"),
          win.lib_status.cget("text"))
    win._use_voice()
    check("can select an installed voice", win.voice_var.get() == rows[0], rows[0])

    print("\n[updates tab]")
    from voice2tts import updater

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

    win.close()
    root.update()
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
