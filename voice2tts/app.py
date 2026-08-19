"""Tray application shell.

Tk insists on owning the main thread, and pystray's Win32 backend is happy on a
secondary one, so the split is: Tk mainloop on the main thread, tray icon on a
daemon thread. Everything the tray does is marshalled back onto the Tk thread with
`root.after`, because touching widgets from another thread crashes Tk.
"""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import pystray

from . import theme, updater
from .config import Config, Repair, load_config
from .gui import SettingsWindow
from .icon import make_icon
from .modes import TriggerMode
from .pipeline import Pipeline, State
from .platform_win import apply_tk_scaling, listen_for_activation
from .speakbox import SpeakBox
from .wizard import Wizard

log = logging.getLogger(__name__)

_TOOLTIP = {
    State.STOPPED: "Voice2TTS — stopped",
    State.LOADING: "Voice2TTS — loading models",
    State.IDLE: "Voice2TTS — ready",
    State.LISTENING: "Voice2TTS — listening",
    State.THINKING: "Voice2TTS — transcribing",
    State.REVIEWING: "Voice2TTS — waiting for you to approve",
    State.SPEAKING: "Voice2TTS — speaking",
}


class TrayApp:
    def __init__(
        self,
        cfg: Config | None = None,
        autostart: bool = True,
        instance_guard=None,
    ):
        self.cfg = cfg or load_config()
        self.instance_guard = instance_guard
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Voice2TTS")
        apply_tk_scaling(self.root)
        # Applied to the hidden root so every Toplevel inherits it.
        self.palette = theme.apply(self.root, self.cfg.theme)

        self.events: deque[tuple[str, str]] = deque(maxlen=300)
        self.settings: SettingsWindow | None = None
        self.wizard: Wizard | None = None
        self.speakbox: SpeakBox | None = None
        self.pending_update: updater.Release | None = None
        self._quitting = False

        self._review_window: tk.Toplevel | None = None
        self.pipeline = Pipeline(self.cfg, on_state=self._on_state, on_event=self._on_event)
        self.pipeline.review_hook = self._review_hook
        self.icon = pystray.Icon(
            "voice2tts", make_icon(State.STOPPED), "Voice2TTS", self._menu()
        )
        self._autostart = autostart

    # -- menu -----------------------------------------------------------------

    def _menu(self) -> pystray.Menu:
        def mode_item(label: str, mode: TriggerMode) -> pystray.MenuItem:
            return pystray.MenuItem(
                label,
                lambda _i, _it: self._post(self._set_mode, mode),
                checked=lambda _it, m=mode: self.cfg.trigger.mode is m,
                radio=True,
            )

        return pystray.Menu(
            pystray.MenuItem(
                lambda _i: "Stop" if self.pipeline.running else "Start",
                lambda _i, _it: self._post(self._toggle_running),
                default=True,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Mode",
                pystray.Menu(
                    mode_item("Push to talk", TriggerMode.PTT),
                    mode_item("Automatic (VAD)", TriggerMode.VAD),
                    mode_item("Both", TriggerMode.BOTH),
                ),
            ),
            pystray.MenuItem("Type to speak...",
                             lambda _i, _it: self._post(self._speak_prompt)),
            pystray.MenuItem("Speak clipboard",
                             lambda _i, _it: self._post(self._speak_clipboard)),
            pystray.MenuItem(
                "Stop speaking",
                lambda _i, _it: self._post(self._stop_speaking),
                enabled=lambda _it: self.pipeline.state is State.SPEAKING,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", lambda _i, _it: self._post(self._open_settings)),
            pystray.MenuItem("Setup wizard...", lambda _i, _it: self._post(self._open_wizard)),
            pystray.MenuItem(
                lambda _i: (
                    f"Update to {self.pending_update.version}..."
                    if self.pending_update else "Check for updates..."
                ),
                lambda _i, _it: self._post(self._update_menu_clicked),
            ),
            pystray.MenuItem("Quit", lambda _i, _it: self._post(self.quit)),
        )

    # -- thread marshalling ---------------------------------------------------

    def _post(self, fn, *args) -> None:
        """Run `fn` on the Tk thread."""
        if not self._quitting:
            self.root.after(0, lambda: fn(*args))

    def _on_state(self, state: State) -> None:
        def apply() -> None:
            try:
                self.icon.icon = make_icon(state)
                self.icon.title = _TOOLTIP.get(state, "Voice2TTS")
                self.icon.update_menu()
            except Exception:  # noqa: BLE001 - icon may be gone during shutdown
                pass

        self._post(apply)

    def _on_event(self, kind: str, message: str) -> None:
        self.events.append((kind, message))

        def apply() -> None:
            if self.settings is not None and self.settings.winfo_exists():
                self.settings.append_log(kind, message)

        self._post(apply)

    # -- actions --------------------------------------------------------------

    def _toggle_running(self) -> None:
        if self.pipeline.running:
            self.pipeline.stop()
        else:
            threading.Thread(target=self._start_pipeline, daemon=True).start()
        self.icon.update_menu()

    def _start_pipeline(self) -> None:
        try:
            self.pipeline.start()
        except Exception as exc:  # noqa: BLE001 - report instead of dying silently
            self._post(
                messagebox.showerror, "Voice2TTS", f"Could not start:\n\n{exc}"
            )
        self._post(self.icon.update_menu)

    def _set_mode(self, mode: TriggerMode) -> None:
        self.pipeline.set_mode(mode)   # validates, and says what it had to fix
        self._show_repairs(self.cfg.repairs)
        self.cfg.save()
        self.icon.update_menu()

    def _show_repairs(self, repairs: list[Repair]) -> None:
        """Tell the user what the app could not do with their settings.

        These used to go to the log and nowhere else, which meant a config whose
        translation had been switched off looked exactly like one that was
        working -- right down to the tick still being in the box. They are rare
        by construction: a repair means something was genuinely wrong.
        """
        if not repairs:
            return
        for repair in repairs:
            self._on_event("warning", str(repair))
        repairs = list(repairs)
        self.cfg.repairs = []
        self._post(
            messagebox.showwarning, "Voice2TTS settings",
            "Some settings could not be used as saved:\n\n"
            + "\n\n".join(f"\u2022 {r}" for r in repairs))

    def _speak_prompt(self) -> None:
        """Open the type-to-speak box, or bring it forward if already open."""
        if self.speakbox is not None and self.speakbox.winfo_exists():
            self.speakbox.deiconify()
            self.speakbox.lift()
            self.speakbox.focus_force()
            return
        self.speakbox = SpeakBox(self.root, self.pipeline,
                                 on_close=self._speakbox_closed)

    def _speakbox_closed(self) -> None:
        self.speakbox = None

    # -- review before speaking ----------------------------------------------

    def _review_hook(self, text: str) -> str | None:
        """Called on the worker thread; blocks until the user decides.

        The dialog has to be built on the Tk thread, so the decision is passed back
        through an Event. A timeout is essential: a dialog hidden behind a game
        would otherwise wedge the pipeline forever.
        """
        decision: dict[str, str | None] = {"text": None}
        done = threading.Event()
        self._post(self._show_review, text, decision, done)

        # No `or None`: a zero here means Event.wait() never times out, so a
        # review window lost behind a game held the pipeline open forever. The
        # config cannot hold a zero any more, and this must not reintroduce one.
        timeout = max(1.0, float(self.cfg.text.review_timeout_s))
        if not done.wait(timeout):
            log.info("review timed out after %ss; discarding", timeout)
            self._post(self._close_review)
            return None
        return decision["text"]

    def _show_review(self, text: str, decision: dict, done: threading.Event) -> None:
        self._close_review()
        win = tk.Toplevel(self.root)
        self._review_window = win
        win.title("Speak this?")
        win.geometry("520x220")
        win.attributes("-topmost", True)  # useless behind a fullscreen game otherwise

        tk.Label(win, text="Check the text, edit if needed, then speak it.",
                 anchor="w").pack(fill="x", padx=12, pady=(12, 4))
        entry = tk.Text(win, height=5, wrap="word")
        entry.pack(fill="both", expand=True, padx=12)
        entry.insert("1.0", text)
        entry.focus_force()
        # Select all, so retyping replaces rather than appends to a bad transcript.
        entry.tag_add("sel", "1.0", "end-1c")

        def finish(approved: bool) -> None:
            if not done.is_set():
                decision["text"] = entry.get("1.0", "end-1c").strip() if approved else None
                done.set()
            self._close_review()

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=12, pady=10)
        ttk.Button(bar, text="Speak", command=lambda: finish(True)).pack(side="right")
        ttk.Button(bar, text="Discard",
                   command=lambda: finish(False)).pack(side="right", padx=6)
        ttk.Label(bar, text=f"Discards automatically after "
                            f"{self.cfg.text.review_timeout_s:.0f}s",
                  foreground="#666").pack(side="left")

        win.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        win.bind("<Escape>", lambda _e: finish(False))
        # Ctrl+Enter speaks; plain Enter inserts a newline, since this is editable.
        win.bind("<Control-Return>", lambda _e: finish(True))

    def _close_review(self) -> None:
        win, self._review_window = getattr(self, "_review_window", None), None
        if win is not None:
            try:
                win.destroy()
            except Exception:  # noqa: BLE001 - already gone
                pass

    def _speak_clipboard(self) -> None:
        if not self.pipeline.running:
            messagebox.showinfo("Voice2TTS", "Start the pipeline first.")
            return
        if not self.pipeline.speak_clipboard():
            messagebox.showinfo(
                "Voice2TTS", "The clipboard is empty, or holds something that is "
                "not text."
            )

    def _stop_speaking(self) -> None:
        if self.pipeline.running:
            self.pipeline.stop_speaking()

    def _open_settings(self) -> None:
        if self.settings is not None and self.settings.winfo_exists():
            self.settings.deiconify()
            self.settings.lift()
            self.settings.focus_force()
            return
        self.settings = SettingsWindow(
            self.root,
            self.cfg,
            self.pipeline,
            on_close=self._settings_closed,
            on_install_update=self._offer_update,
        )
        if self.pending_update is not None:
            self.settings.show_update(self.pending_update)
        for kind, message in self.events:
            self.settings.append_log(kind, message)

    def _settings_closed(self) -> None:
        self.settings = None

    # -- updates --------------------------------------------------------------

    def _startup_update_check(self) -> None:
        """Silent background check, throttled by config. Never blocks startup."""
        from . import updater

        cfg = self.cfg.updates
        if not cfg.repo or not cfg.check_on_start:
            return
        if not updater.should_check(cfg.last_check, cfg.interval_hours):
            return
        try:
            release = updater.check(cfg.repo,
                                    include_prereleases=cfg.include_prereleases)
        except Exception as exc:  # noqa: BLE001 - a failed check is not worth a dialog
            log.info("update check failed: %s", exc)
            return
        finally:
            cfg.last_check = time.time()
            self.cfg.save()

        if release is None or release.version == cfg.skipped_version:
            return
        self._post(self._announce_update, release)

    def _announce_update(self, release) -> None:
        self.pending_update = release
        self.icon.update_menu()
        try:
            self.icon.notify(
                f"Version {release.version} is available ({release.size_mb:.0f} MB).",
                "Voice2TTS update",
            )
        except Exception:  # noqa: BLE001 - balloon tips are best-effort
            pass
        if self.settings is not None and self.settings.winfo_exists():
            self.settings.show_update(release)

    def _update_menu_clicked(self) -> None:
        if self.pending_update is not None:
            self._offer_update(self.pending_update)
        else:
            self._manual_update_check()

    def _manual_update_check(self) -> None:
        from . import updater

        if not self.cfg.updates.repo:
            # Only reachable if the user deliberately cleared it; the default is
            # baked in, so this is "you turned it off", not "you forgot to set it".
            messagebox.showinfo(
                "Voice2TTS",
                "Update checking is turned off.\n\n"
                "Settings -> Updates -> 'Use default' turns it back on.",
            )
            return

        def work() -> None:
            try:
                release = updater.check(
                    self.cfg.updates.repo,
                    include_prereleases=self.cfg.updates.include_prereleases)
            except Exception as exc:  # noqa: BLE001
                self._post(messagebox.showerror, "Update check failed", str(exc))
                return
            self.cfg.updates.last_check = time.time()
            self.cfg.save()
            if release is None:
                self._post(
                    messagebox.showinfo, "Voice2TTS",
                    f"You are on the latest version ({updater.current_version()}).",
                )
            else:
                self._post(self._announce_update, release)

        threading.Thread(target=work, daemon=True).start()

    def _offer_update(self, release) -> None:
        from . import updater

        notes = release.notes[:600] + ("..." if len(release.notes) > 600 else "")
        message = (
            f"Version {release.version} is available.\n"
            f"You have {updater.current_version()}.\n\n"
            f"Download size: {release.size_mb:.0f} MB\n"
        )
        if notes:
            message += f"\n{notes}\n"
        message += "\nDownload and install now? Voice2TTS will restart."

        if not messagebox.askyesno("Voice2TTS update", message):
            return

        progress = tk.Toplevel(self.root)
        progress.title("Updating Voice2TTS")
        progress.geometry("420x120")
        progress.resizable(False, False)
        label = tk.Label(progress, text="Starting download...", anchor="w")
        label.pack(fill="x", padx=16, pady=(16, 6))
        bar = ttk.Progressbar(progress, maximum=100, length=380)
        bar.pack(padx=16)
        progress.protocol("WM_DELETE_WINDOW", lambda: None)  # no cancel mid-swap

        def on_progress(got: int, total: int) -> None:
            pct = (got / total * 100) if total else 0
            self._post(lambda: (
                bar.configure(value=pct),
                label.configure(
                    text=f"Downloading... {got / 1e6:.0f} of {total / 1e6:.0f} MB"
                ),
            ))

        def work() -> None:
            try:
                path = updater.download(release, progress=on_progress)
                self._post(lambda: label.configure(text="Installing and restarting..."))
                updater.apply(path)
            except Exception as exc:  # noqa: BLE001
                self._post(progress.destroy)
                self._post(messagebox.showerror, "Update failed", str(exc))
                return
            # Must exit promptly: the installer cannot replace files we hold open.
            self._post(self.quit)

        threading.Thread(target=work, daemon=True).start()

    def _open_wizard(self) -> None:
        from .wizard import Wizard

        if self.wizard is not None and self.wizard.winfo_exists():
            self.wizard.lift()
            self.wizard.focus_force()
            return
        self.wizard = Wizard(self.root, self.cfg, on_finish=self._wizard_finished)

    def _wizard_finished(self, completed: bool) -> None:
        self.wizard = None
        if not completed:
            return
        # Devices and voice may both have changed; adopt them without a restart.
        if self.pipeline.running:
            self.pipeline.apply_audio_changes()
            self.pipeline.apply_tts_changes()
            self.pipeline.set_mode(self.cfg.trigger.mode)
            self._show_repairs(self.cfg.repairs)
        else:
            threading.Thread(target=self._start_pipeline, daemon=True).start()

    # -- lifecycle ------------------------------------------------------------

    def run(self) -> None:
        threading.Thread(target=self.icon.run, name="v2t-tray", daemon=True).start()
        # A second launch signals this event rather than starting a rival instance.
        threading.Thread(
            target=listen_for_activation,
            args=(lambda: self._post(self._open_settings),),
            name="v2t-activate",
            daemon=True,
        ).start()
        if not self.cfg.first_run_complete:
            # Hold off on loading models: the wizard may change the device or the
            # voice, and it starts the pipeline itself once the user is done.
            self._post(self._open_wizard)
        else:
            if self._autostart:
                threading.Thread(target=self._start_pipeline, daemon=True).start()
            if not self.cfg.start_minimized:
                self._post(self._open_settings)
            threading.Thread(
                target=self._startup_update_check, name="v2t-update", daemon=True
            ).start()
        try:
            self.root.mainloop()
        finally:
            self._shutdown()

    def quit(self) -> None:
        self._quitting = True
        self.root.quit()

    def _shutdown(self) -> None:
        try:
            self.pipeline.shutdown()
        except Exception:
            log.exception("pipeline shutdown failed")
        try:
            self.icon.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass
