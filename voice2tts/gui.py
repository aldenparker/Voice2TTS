"""Tkinter settings window.

Owns no state of its own: every widget reads from and writes to the live Config
object, and pressing Apply asks the pipeline to adopt the changes. All widget access
happens on the Tk main thread; background callbacks marshal through `root.after`.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

from . import DEFAULT_UPDATE_REPO, cable, devices, gpupack, updater, voices
from .config import MODES, WHISPER_MODELS, Config, OutputTarget
from .diagnostics import diagnostics
from .hotkey import describe
from .paths import config_path, is_frozen, list_voices, log_path
from .pipeline import Pipeline
from .platform_win import run_at_login, set_run_at_login

log = logging.getLogger(__name__)

CABLE_URL = "https://vb-audio.com/Cable/"

_TK_MODIFIERS = {
    "Control_L": "ctrl", "Control_R": "ctrl",
    "Alt_L": "alt", "Alt_R": "alt",
    "Shift_L": "shift", "Shift_R": "shift",
    "Super_L": "win", "Super_R": "win",
}


class SettingsWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Tk,
        cfg: Config,
        pipeline: Pipeline,
        on_close=None,
        on_install_update=None,
    ):
        super().__init__(master)
        self.cfg = cfg
        self.pipeline = pipeline
        self._on_close = on_close
        # The tray app performs the actual update: it owns the progress window and
        # the app shutdown the installer needs.
        self._on_install_update = on_install_update
        self._pending_release = None
        self._output_rows: list[dict] = []
        self._closing = False

        self.title("Voice2TTS Settings")
        self.geometry("640x620")
        self.minsize(560, 520)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._build()
        self._load_from_config()
        self._tick()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self._build_audio(nb)
        self._build_trigger(nb)
        self._build_voice(nb)
        self._build_voice_library(nb)
        self._build_recognition(nb)
        self._build_updates(nb)
        self._build_status(nb)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=8)
        self.state_label = ttk.Label(bar, text="stopped")
        self.state_label.pack(side="left")
        ttk.Button(bar, text="Close", command=self.close).pack(side="right")
        ttk.Button(bar, text="Save", command=self._save).pack(side="right", padx=4)
        ttk.Button(bar, text="Apply", command=self._apply).pack(side="right")

    # -- audio tab ------------------------------------------------------------

    def _build_audio(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Audio")

        ttk.Label(tab, text="Microphone").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(tab, textvariable=self.input_var, width=52)
        self.input_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        header = ttk.Frame(tab)
        header.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Outputs", font=("", 9, "bold")).pack(side="left")
        ttk.Button(header, text="Refresh devices", command=self._refresh_devices).pack(
            side="right"
        )
        ttk.Label(
            tab,
            text="Send speech to any number of devices at once. Point one at your "
            "virtual cable\nfor Discord; add your headphones to hear yourself.",
            foreground="#555",
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 6))

        self.outputs_frame = ttk.Frame(tab)
        self.outputs_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        ttk.Button(tab, text="Add output", command=self._add_output_row).grid(
            row=5, column=0, sticky="w", pady=6
        )

        cable_row = ttk.Frame(tab)
        cable_row.grid(row=6, column=0, columnspan=2, sticky="w")
        self.cable_label = ttk.Label(cable_row, text="", cursor="hand2")
        self.cable_label.pack(side="left")
        self.cable_label.bind("<Button-1>", lambda _e: webbrowser.open(CABLE_URL))
        self.cable_btn = ttk.Button(cable_row, text="Remove virtual cable",
                                    command=self._remove_cable)
        self.cable_btn.pack(side="left", padx=8)

        self.mute_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab,
            text="Mute microphone while speaking (prevents the app hearing itself)",
            variable=self.mute_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.autostart_var = tk.BooleanVar(value=run_at_login())
        self.autostart_check = ttk.Checkbutton(
            tab,
            text="Start Voice2TTS when I sign in to Windows",
            variable=self.autostart_var,
            command=self._toggle_autostart,
        )
        self.autostart_check.grid(row=8, column=0, columnspan=2, sticky="w")
        if not is_frozen():
            self.autostart_check.state(["disabled"])
            ttk.Label(tab, text="(only available in an installed build)",
                      foreground="#666").grid(row=9, column=0, sticky="w")

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

    def _toggle_autostart(self) -> None:
        wanted = self.autostart_var.get()
        if not set_run_at_login(wanted):
            self.autostart_var.set(run_at_login())  # snap back to reality
            messagebox.showerror(
                "Start with Windows",
                "Could not update the startup entry. See the log for details.",
                parent=self,
            )
            return
        self.cfg.run_at_login = self.autostart_var.get()

    def _remove_cable(self) -> None:
        found = cable.detect()
        if found is None:
            return
        if "vb-cable" not in found.product.lower():
            messagebox.showinfo(
                "Remove virtual cable",
                f"{found.product} was not installed by Voice2TTS, so it is not "
                "removed from here. Uninstall it the way you installed it.",
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "Remove VB-CABLE",
            f"Run VB-Audio's uninstaller for {found.product}?\n\n"
            "Windows will ask for administrator permission, and a restart may be "
            "needed. Discord will lose this input device.",
            parent=self,
        ):
            return

        def work() -> None:
            try:
                _needs_reboot, message = cable.uninstall_flow()
            except Exception as exc:  # noqa: BLE001
                # Bind the text now: Python deletes `exc` when this block ends, so
                # a lambda closing over it would raise NameError when Tk runs it.
                failure = str(exc)
                self.after(0, lambda: messagebox.showerror(
                    "Uninstall failed", failure, parent=self))
                return
            self.after(0, lambda: (self._update_cable_hint(),
                                   messagebox.showinfo("Virtual cable", message,
                                                       parent=self)))

        threading.Thread(target=work, daemon=True).start()

    def _add_output_row(self, target: OutputTarget | None = None) -> None:
        target = target or OutputTarget(match="", gain=1.0, enabled=True)
        row = ttk.Frame(self.outputs_frame)
        row.pack(fill="x", pady=2)

        enabled = tk.BooleanVar(value=target.enabled)
        ttk.Checkbutton(row, variable=enabled, width=2).pack(side="left")

        match = tk.StringVar(value=target.match)
        combo = ttk.Combobox(row, textvariable=match, width=34)
        combo["values"] = [""] + [d.name for d in devices.list_outputs()]
        combo.pack(side="left", padx=(0, 6))

        gain = tk.DoubleVar(value=target.gain)
        ttk.Scale(row, from_=0.0, to=2.0, variable=gain, length=90).pack(side="left")
        gain_label = ttk.Label(row, width=5, text=f"{target.gain:.2f}")
        gain_label.pack(side="left", padx=(4, 6))
        gain.trace_add("write", lambda *_: gain_label.config(text=f"{gain.get():.2f}"))

        entry = {"frame": row, "enabled": enabled, "match": match, "gain": gain}
        ttk.Button(row, text="X", width=3,
                   command=lambda: self._remove_output_row(entry)).pack(side="left")
        self._output_rows.append(entry)

    def _remove_output_row(self, entry: dict) -> None:
        entry["frame"].destroy()
        self._output_rows.remove(entry)

    def _refresh_devices(self) -> None:
        devices.refresh()
        inputs = [""] + [d.name for d in devices.list_inputs()]
        outputs = [""] + [d.name for d in devices.list_outputs()]
        self.input_combo["values"] = inputs
        for row in self._output_rows:
            for child in row["frame"].winfo_children():
                if isinstance(child, ttk.Combobox):
                    child["values"] = outputs
        self._update_cable_hint()

    def _update_cable_hint(self) -> None:
        found = cable.detect()
        if found is not None:
            extra = ""
            others = len(cable.list_devices()) - 1
            if others > 0:
                extra = f" (+{others} more channel{'s' if others > 1 else ''})"
            self.cable_label.config(
                text=f"Virtual cable: {found.label}{extra} — "
                     f"Discord input: {found.discord_input}",
                foreground="#2a7",
            )
            # Only VB-CABLE has an uninstaller we know how to drive.
            self.cable_btn.state(
                ["!disabled"] if "vb-cable" in found.product.lower() else ["disabled"]
            )
        else:
            self.cable_label.config(
                text="No virtual audio cable detected — click here to install "
                     "VB-CABLE (needs admin + reboot).",
                foreground="#a33",
            )
            self.cable_btn.state(["disabled"])

    # -- trigger tab ----------------------------------------------------------

    def _build_trigger(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Trigger")

        ttk.Label(tab, text="How speech is captured", font=("", 9, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        self.mode_var = tk.StringVar()
        labels = {
            "ptt": "Push to talk only",
            "vad": "Automatic (voice activity detection)",
            "both": "Both — automatic, plus the hotkey",
        }
        for i, mode in enumerate(MODES):
            ttk.Radiobutton(
                tab, text=labels[mode], value=mode, variable=self.mode_var,
                command=self._on_mode_change,
            ).grid(row=1 + i, column=0, columnspan=3, sticky="w")

        ttk.Separator(tab, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=10
        )

        ttk.Label(tab, text="Hotkey").grid(row=5, column=0, sticky="w")
        self.hotkey_var = tk.StringVar()
        self.hotkey_entry = ttk.Entry(tab, textvariable=self.hotkey_var, width=24)
        self.hotkey_entry.grid(row=5, column=1, sticky="w", padx=6)
        self.record_btn = ttk.Button(tab, text="Record", command=self._record_hotkey)
        self.record_btn.grid(row=5, column=2, sticky="w")
        self.hotkey_error = ttk.Label(tab, text="", foreground="#a33")
        self.hotkey_error.grid(row=6, column=0, columnspan=3, sticky="w")
        self.hotkey_var.trace_add("write", lambda *_: self._validate_hotkey())

        self.latch_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab, text="Tap to toggle instead of holding", variable=self.latch_var
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 10))

        self.vad_header = ttk.Label(tab, text="Detection tuning", font=("", 9, "bold"))
        self.vad_header.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 4))

        # Tracked so they can be greyed out in push-to-talk mode, where VAD is unused.
        self._vad_widgets: list[tk.Widget] = []
        self.vad_threshold = self._slider(
            tab, 9, "Sensitivity threshold", 0.05, 0.95, "{:.2f}", vad_only=True
        )
        self.vad_silence = self._slider(
            tab, 10, "End-of-speech silence (ms)", 200, 2000, "{:.0f}", vad_only=True
        )
        self.vad_min_speech = self._slider(
            tab, 11, "Minimum speech (ms)", 50, 1000, "{:.0f}", vad_only=True
        )
        self.preroll = self._slider(tab, 12, "Pre-roll kept (ms)", 0, 1000, "{:.0f}")
        tab.columnconfigure(1, weight=1)

    def _slider(self, parent, row, label, lo, hi, fmt, vad_only: bool = False) -> tk.DoubleVar:
        name = ttk.Label(parent, text=label)
        name.grid(row=row, column=0, sticky="w", pady=2)
        var = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var)
        scale.grid(row=row, column=1, sticky="ew", padx=6)
        out = ttk.Label(parent, width=7, text="")
        out.grid(row=row, column=2, sticky="w")
        var.trace_add("write", lambda *_: out.config(text=fmt.format(var.get())))
        if vad_only:
            self._vad_widgets += [name, scale, out]
        return var

    def _on_mode_change(self) -> None:
        """Grey out the VAD tuning when the mode does not use it."""
        active = self.mode_var.get() in ("vad", "both")
        state = "normal" if active else "disabled"
        self.vad_header.config(foreground="" if active else "#999")
        for w in self._vad_widgets:
            try:
                w.config(state=state)
            except tk.TclError:
                pass  # plain ttk.Label has no state option on some themes

    def _validate_hotkey(self) -> bool:
        err = describe(self.hotkey_var.get())
        self.hotkey_error.config(text=err)
        return not err

    def _record_hotkey(self) -> None:
        self.record_btn.config(text="Press keys...")
        held: list[str] = []

        def on_key(event: tk.Event) -> str:
            name = event.keysym
            mod = _TK_MODIFIERS.get(name)
            if mod:
                if mod not in held:
                    held.append(mod)
                return "break"
            self.hotkey_var.set("+".join([*held, name.lower()]))
            finish()
            return "break"

        def finish(_event: tk.Event | None = None) -> None:
            self.unbind("<KeyPress>")
            self.unbind("<FocusOut>")
            self.record_btn.config(text="Record")
            self._validate_hotkey()

        self.bind("<KeyPress>", on_key)
        self.bind("<FocusOut>", finish)
        self.focus_force()

    # -- voice tab ------------------------------------------------------------

    def _build_voice(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Voice")

        ttk.Label(tab, text="Piper voice").grid(row=0, column=0, sticky="w")
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(tab, textvariable=self.voice_var, width=36)
        self.voice_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6)

        ttk.Label(
            tab,
            text="Drop extra .onnx voices into the voices folder beside the config "
                 "to add them.",
            foreground="#555",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 4))

        # A non-English voice paired with an English-only Whisper model produces
        # confident nonsense rather than an error, so say so where it is chosen.
        self.lang_warning = ttk.Label(tab, text="", foreground="#a33",
                                      justify="left", wraplength=520)
        self.lang_warning.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.voice_var.trace_add("write", lambda *_: self._check_language())

        self.speed_var = self._slider(tab, 3, "Speed (lower = faster)", 0.5, 2.0, "{:.2f}")
        self.volume_var = self._slider(tab, 4, "Volume", 0.0, 2.0, "{:.2f}")

        ttk.Separator(tab, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=10
        )
        ttk.Label(tab, text="Test phrase").grid(row=6, column=0, sticky="w")
        self.test_var = tk.StringVar(value="Testing, one two three.")
        ttk.Entry(tab, textvariable=self.test_var).grid(
            row=6, column=1, sticky="ew", padx=6
        )
        ttk.Button(tab, text="Speak", command=self._speak_test).grid(row=6, column=2)
        tab.columnconfigure(1, weight=1)

    def _check_language(self) -> None:
        """Surface a voice/recognition-model language mismatch."""
        if not hasattr(self, "lang_warning") or not hasattr(self, "model_var"):
            return
        warning = voices.language_mismatch(
            self.voice_var.get().strip(), self.model_var.get().strip()
        )
        self.lang_warning.config(text=warning.split("\n\n")[0] if warning else "")

    def _speak_test(self) -> None:
        if not self.pipeline.running:
            messagebox.showinfo("Not running", "Start the pipeline first.", parent=self)
            return
        self._apply()
        self.pipeline.say_text(self.test_var.get())

    # -- voice library tab ----------------------------------------------------

    def _build_voice_library(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Voice library")

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Button(top, text="Load catalogue", command=self._load_catalogue).pack(side="left")
        ttk.Label(top, text="Language").pack(side="left", padx=(12, 4))
        self.lib_lang = tk.StringVar(value="en")
        self.lib_lang_combo = ttk.Combobox(top, textvariable=self.lib_lang, width=8,
                                           state="readonly")
        self.lib_lang_combo.pack(side="left")
        self.lib_lang_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_library())
        ttk.Label(top, text="Search").pack(side="left", padx=(12, 4))
        self.lib_query = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.lib_query, width=18)
        entry.pack(side="left")
        self.lib_query.trace_add("write", lambda *_: self._refresh_library())

        cols = ("voice", "language", "quality", "size", "state")
        self.lib_tree = ttk.Treeview(tab, columns=cols, show="headings", height=13)
        for col, width in zip(cols, (170, 150, 70, 70, 90), strict=True):
            self.lib_tree.heading(col, text=col.title())
            self.lib_tree.column(col, width=width, anchor="w")
        self.lib_tree.grid(row=1, column=0, columnspan=4, sticky="nsew")
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.lib_tree.yview)
        sb.grid(row=1, column=4, sticky="ns")
        self.lib_tree.configure(yscrollcommand=sb.set)

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Button(actions, text="Download", command=self._download_voice).pack(side="left")
        ttk.Button(actions, text="Remove", command=self._remove_voice).pack(side="left", padx=6)
        ttk.Button(actions, text="Use selected", command=self._use_voice).pack(side="left")
        self.lib_status = ttk.Label(actions, text="", foreground="#666")
        self.lib_status.pack(side="left", padx=12)

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        self._catalogue: list = []
        self._show_installed_only()

    def _show_installed_only(self) -> None:
        """Before the catalogue is fetched, at least list what is already here."""
        self.lib_tree.delete(*self.lib_tree.get_children())
        for key in voices.installed_keys():
            state = "bundled" if key in voices.BUNDLED else "installed"
            self.lib_tree.insert("", "end", iid=key, values=(key, "-", "-", "-", state))
        self.lib_status.config(text="Load the catalogue to browse more voices.")

    def _load_catalogue(self) -> None:
        self.lib_status.config(text="Fetching catalogue...")

        def work() -> None:
            try:
                entries = voices.fetch_catalogue()
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self.after(0, lambda: self.lib_status.config(text=f"Failed: {failure}"))
                return

            def apply() -> None:
                self._catalogue = entries
                langs = voices.languages(entries)
                self.lib_lang_combo["values"] = ["(all)", *langs]
                if self.lib_lang.get() not in langs:
                    self.lib_lang.set("en_US" if "en_US" in langs else "(all)")
                self._refresh_library()

            self.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def _refresh_library(self) -> None:
        if not self._catalogue:
            return
        lang = self.lib_lang.get()
        entries = voices.filter_catalogue(
            self._catalogue,
            language_prefix="" if lang in ("", "(all)") else lang,
            query=self.lib_query.get().strip(),
        )
        self.lib_tree.delete(*self.lib_tree.get_children())
        for e in entries[:600]:  # the full catalogue is long; keep the widget usable
            if e.bundled:
                state = "bundled"
            elif e.installed:
                state = "installed"
            else:
                state = ""
            self.lib_tree.insert(
                "", "end", iid=e.key,
                values=(e.key, e.language_label, e.quality, f"{e.size_mb:.0f} MB", state),
            )
        shown = min(len(entries), 600)
        self.lib_status.config(
            text=f"{shown} of {len(self._catalogue)} voices"
            + (" (list truncated)" if len(entries) > 600 else "")
        )

    def _selected_voice(self) -> str | None:
        sel = self.lib_tree.selection()
        if not sel:
            self.lib_status.config(text="Select a voice first.")
            return None
        return sel[0]

    def _download_voice(self) -> None:
        key = self._selected_voice()
        if key is None:
            return
        if voices.installed_path(key) is not None:
            self.lib_status.config(text=f"{key} is already installed.")
            return
        self.lib_status.config(text=f"Downloading {key}...")

        def work() -> None:
            try:
                voices.download_voice(key)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self.after(0, lambda: self.lib_status.config(text=f"Failed: {failure}"))
                return
            self.after(0, lambda: self._after_library_change(f"Installed {key}"))

        threading.Thread(target=work, daemon=True).start()

    def _remove_voice(self) -> None:
        key = self._selected_voice()
        if key is None:
            return
        if not voices.is_removable(key):
            self.lib_status.config(text=f"{key} ships with the app and cannot be removed.")
            return
        if key == self.cfg.tts.voice:
            self.lib_status.config(text="That voice is in use. Pick another one first.")
            return
        if not messagebox.askyesno("Remove voice", f"Delete {key}?", parent=self):
            return
        voices.remove_voice(key)
        self._after_library_change(f"Removed {key}")

    def _use_voice(self) -> None:
        key = self._selected_voice()
        if key is None:
            return
        if voices.installed_path(key) is None:
            self.lib_status.config(text="Download it first.")
            return
        self.voice_var.set(key)
        self.cfg.tts.voice = key
        self.pipeline.apply_tts_changes()
        self.lib_status.config(text=f"Now using {key}")

    def _after_library_change(self, message: str) -> None:
        installed = voices.installed_keys()
        self.voice_combo["values"] = installed
        if self._catalogue:
            self._refresh_library()
        else:
            self._show_installed_only()
        self.lib_status.config(text=message)

    # -- recognition tab ------------------------------------------------------

    def _build_recognition(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Recognition")

        ttk.Label(tab, text="Whisper model").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        ttk.Combobox(
            tab, textvariable=self.model_var, values=list(WHISPER_MODELS), width=24
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(tab, text="Compute device").grid(row=1, column=0, sticky="w")
        self.stt_device_var = tk.StringVar()
        ttk.Combobox(
            tab, textvariable=self.stt_device_var, values=["auto", "cuda", "cpu"],
            width=24, state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(tab, text="Beam size").grid(row=2, column=0, sticky="w")
        self.beam_var = tk.IntVar()
        ttk.Spinbox(tab, from_=1, to=5, textvariable=self.beam_var, width=6).grid(
            row=2, column=1, sticky="w", padx=6, pady=2
        )

        ttk.Label(
            tab,
            text="Model and device changes take effect after restarting the pipeline\n"
                 "(tray menu → Stop, then Start).",
            foreground="#555",
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Separator(tab, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=12
        )
        ttk.Label(tab, text="GPU acceleration", font=("", 9, "bold")).grid(
            row=5, column=0, columnspan=2, sticky="w"
        )
        self.gpu_label = ttk.Label(tab, text="", justify="left", foreground="#444")
        self.gpu_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 6))
        gpu_row = ttk.Frame(tab)
        gpu_row.grid(row=7, column=0, columnspan=2, sticky="w")
        self.gpu_btn = ttk.Button(gpu_row, text="", command=self._toggle_gpu_pack)
        self.gpu_btn.pack(side="left")
        self.gpu_progress = ttk.Progressbar(gpu_row, mode="indeterminate", length=200)
        self.gpu_note = ttk.Label(tab, text="", foreground="#666")
        self.gpu_note.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        tab.columnconfigure(1, weight=1)
        self._refresh_gpu()

    def _refresh_gpu(self) -> None:
        pack = gpupack.status()
        if pack.usable:
            self.gpu_label.config(
                text=f"Installed — {pack.dll_count} libraries, {pack.size_mb:.0f} MB."
            )
            self.gpu_btn.config(text="Remove GPU pack")
        elif gpupack.gpu_present():
            self.gpu_label.config(
                text="An NVIDIA GPU was found but the CUDA libraries are not installed.\n"
                     "Downloading them makes transcription roughly 20x faster (~1.3 GB)."
            )
            self.gpu_btn.config(text="Download GPU acceleration")
        else:
            self.gpu_label.config(
                text="No NVIDIA GPU detected. Transcription runs on the CPU."
            )
            self.gpu_btn.config(text="Download anyway", state="disabled")

    def _toggle_gpu_pack(self) -> None:
        if gpupack.status().usable:
            if not messagebox.askyesno(
                "Remove GPU pack",
                "Delete the downloaded CUDA libraries? Transcription falls back to CPU.",
                parent=self,
            ):
                return
            gpupack.uninstall()
            self.cfg.stt.model = "base.en"
            self.model_var.set("base.en")
            self._refresh_gpu()
            return

        self.gpu_btn.config(state="disabled")
        self.gpu_progress.pack(side="left", padx=8)
        self.gpu_progress.start(12)

        def report(msg: str) -> None:
            self.after(0, lambda: self.gpu_note.config(text=msg))

        def work() -> None:
            try:
                gpupack.install(progress=report)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self.after(0, lambda: self._gpu_done(failure))
                return
            self.after(0, lambda: self._gpu_done(None))

        threading.Thread(target=work, daemon=True).start()

    def _gpu_done(self, error: str | None) -> None:
        self.gpu_progress.stop()
        self.gpu_progress.pack_forget()
        self.gpu_btn.config(state="normal")
        if error:
            self.gpu_note.config(text="")
            messagebox.showerror("Download failed", error, parent=self)
            return
        self.cfg.stt.model = gpupack.GPU_WHISPER_MODEL
        self.model_var.set(gpupack.GPU_WHISPER_MODEL)
        self.gpu_note.config(text="Ready. Restart the pipeline to use it.")
        self._refresh_gpu()

    # -- updates tab ----------------------------------------------------------

    def _build_updates(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Updates")

        ttk.Label(tab, text=f"Installed version: {updater.current_version()}",
                  font=("", 9, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        if not is_frozen():
            ttk.Label(
                tab,
                text="Running from source — updates are applied with git, not here.",
                foreground="#a60",
            ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

        ttk.Label(tab, text="GitHub repository").grid(row=2, column=0, sticky="w",
                                                      pady=(12, 2))
        self.repo_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.repo_var, width=34).grid(
            row=2, column=1, sticky="w", padx=6, pady=(12, 2)
        )
        ttk.Button(tab, text="Use default", command=self._reset_repo).grid(
            row=2, column=2, sticky="w"
        )
        ttk.Label(
            tab,
            text=f"Prefilled with {DEFAULT_UPDATE_REPO}. Clear it to stop checking.",
            foreground="#666",
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=6)

        self.check_start_var = tk.BooleanVar()
        ttk.Checkbutton(tab, text="Check for updates when Voice2TTS starts",
                        variable=self.check_start_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(tab, text="Check at most every").grid(row=5, column=0, sticky="w",
                                                        pady=2)
        self.interval_var = tk.IntVar()
        ttk.Spinbox(tab, from_=0, to=720, textvariable=self.interval_var,
                    width=6).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Label(tab, text="hours (0 = never)", foreground="#666").grid(
            row=5, column=2, sticky="w"
        )

        ttk.Label(
            tab,
            text="Checking contacts api.github.com. Nothing else is sent.",
            foreground="#666",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Separator(tab, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=12
        )

        row = ttk.Frame(tab)
        row.grid(row=8, column=0, columnspan=3, sticky="w")
        self.update_btn = ttk.Button(row, text="Check now", command=self._check_updates)
        self.update_btn.pack(side="left")
        self.update_status = ttk.Label(row, text="", foreground="#666")
        self.update_status.pack(side="left", padx=10)

        self.update_notes = tk.Text(tab, height=8, wrap="word")
        self.update_notes.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.update_notes.configure(state="disabled")

        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(9, weight=1)

    def _reset_repo(self) -> None:
        self.repo_var.set(DEFAULT_UPDATE_REPO)
        self.update_status.config(text="")

    def _check_updates(self) -> None:
        repo = self.repo_var.get().strip()
        if not repo:
            self.update_status.config(
                text="Update checking is off. Press 'Use default' to turn it back on."
            )
            return
        self.cfg.updates.repo = repo
        self.cfg.validate()
        self.update_btn.config(state="disabled")
        self.update_status.config(text="Checking...")

        def work() -> None:
            try:
                release = updater.check(self.cfg.updates.repo)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self.after(0, lambda: self._check_done(None, failure))
                return
            self.after(0, lambda: self._check_done(release, None))

        threading.Thread(target=work, daemon=True).start()

    def _check_done(self, release, error: str | None) -> None:
        if not self.winfo_exists():
            return
        self.update_btn.config(state="normal")
        if error:
            self.update_status.config(text=f"Failed: {error}")
            return
        if release is None:
            self.update_status.config(
                text=f"Up to date ({updater.current_version()})."
            )
            self._set_notes("")
            return
        self.show_update(release)

    def show_update(self, release) -> None:
        """Display an available release; also called when the tray finds one."""
        if not self.winfo_exists():
            return
        self._pending_release = release
        self.update_status.config(
            text=f"Version {release.version} available ({release.size_mb:.0f} MB)"
        )
        self.update_btn.config(text="Download and install",
                               command=self._install_update, state="normal")
        self._set_notes(release.notes or "(no release notes)")

    def _set_notes(self, text: str) -> None:
        self.update_notes.configure(state="normal")
        self.update_notes.delete("1.0", "end")
        self.update_notes.insert("1.0", text)
        self.update_notes.configure(state="disabled")

    def _install_update(self) -> None:
        release = getattr(self, "_pending_release", None)
        if release is None:
            return
        if not is_frozen():
            messagebox.showinfo(
                "Running from source",
                "This is a source checkout. Update it with git rather than by "
                "running the installer over it.",
                parent=self,
            )
            return
        # The tray app owns the download UI and the shutdown that must follow it.
        if self._on_install_update is not None:
            self._on_install_update(release)

    # -- status tab -----------------------------------------------------------

    def _build_status(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Status")

        ttk.Label(tab, text="Input level").grid(row=0, column=0, sticky="w")
        self.level = ttk.Progressbar(tab, maximum=100, length=260)
        self.level.grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(tab, text="Last transcript").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        self.transcript = tk.Text(tab, height=3, wrap="word")
        self.transcript.grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        self.transcript.configure(state="disabled")

        ttk.Label(tab, text="Log").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        self.logbox = tk.Text(tab, height=14, wrap="word")
        self.logbox.grid(row=2, column=1, sticky="nsew", padx=6, pady=(8, 0))
        self.logbox.configure(state="disabled")

        actions = ttk.Frame(tab)
        actions.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(actions, text="Copy diagnostics",
                   command=self._copy_diagnostics).pack(side="left")
        ttk.Button(actions, text="Open log folder",
                   command=self._open_log_folder).pack(side="left", padx=6)
        self.diag_status = ttk.Label(actions, text="", foreground="#666")
        self.diag_status.pack(side="left", padx=8)

        ttk.Label(tab, text=f"Config: {config_path()}", foreground="#555").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

    def _copy_diagnostics(self) -> None:
        """Put a support-ready summary on the clipboard.

        Beats talking someone through finding a log file in AppData.
        """
        try:
            report = diagnostics(self.cfg, self.pipeline)
        except Exception as exc:  # noqa: BLE001
            self.diag_status.config(text=f"Failed: {exc}")
            return
        self.clipboard_clear()
        self.clipboard_append(report)
        self.update_idletasks()  # keep the clipboard alive after this window closes
        self.diag_status.config(text=f"Copied ({len(report.splitlines())} lines)")

    def _open_log_folder(self) -> None:
        import subprocess

        try:
            subprocess.Popen(["explorer", "/select,", str(log_path())])
        except Exception as exc:  # noqa: BLE001
            self.diag_status.config(text=f"Could not open: {exc}")

    def append_log(self, kind: str, message: str) -> None:
        if self._closing or not self.winfo_exists():
            return
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"[{kind}] {message}\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")
        if kind == "transcript":
            self.transcript.configure(state="normal")
            self.transcript.delete("1.0", "end")
            self.transcript.insert("1.0", message)
            self.transcript.configure(state="disabled")

    # -- config binding -------------------------------------------------------

    def _load_from_config(self) -> None:
        c = self.cfg
        self.input_combo["values"] = [""] + [d.name for d in devices.list_inputs()]
        self.input_var.set(c.audio.input_match)
        self.mute_var.set(c.audio.mute_mic_during_playback)
        for target in c.audio.outputs:
            self._add_output_row(target)
        if not c.audio.outputs:
            self._add_output_row()
        self._update_cable_hint()

        self.mode_var.set(c.trigger.mode)
        self.hotkey_var.set(c.trigger.hotkey)
        self.latch_var.set(c.trigger.ptt_latch)
        self.vad_threshold.set(c.vad.threshold)
        self.vad_silence.set(c.vad.min_silence_ms)
        self.vad_min_speech.set(c.vad.min_speech_ms)
        self.preroll.set(c.trigger.preroll_ms)
        self._on_mode_change()

        voices = [p.stem for p in list_voices()]
        self.voice_combo["values"] = voices
        self.voice_var.set(c.tts.voice)
        self.speed_var.set(c.tts.length_scale)
        self.volume_var.set(c.tts.volume)

        self.model_var.set(c.stt.model)
        self.stt_device_var.set(c.stt.device)
        self.beam_var.set(c.stt.beam_size)

        self.repo_var.set(c.updates.repo)
        self.check_start_var.set(c.updates.check_on_start)
        self.interval_var.set(c.updates.interval_hours)

        # Both halves of the pairing are loaded now, so the warning can be evaluated.
        self._check_language()

    def _collect(self) -> bool:
        if not self._validate_hotkey():
            messagebox.showerror("Invalid hotkey", "Fix the hotkey first.", parent=self)
            return False
        c = self.cfg
        c.audio.input_match = self.input_var.get().strip()
        c.audio.mute_mic_during_playback = self.mute_var.get()
        c.audio.outputs = [
            OutputTarget(
                match=r["match"].get().strip(),
                gain=round(float(r["gain"].get()), 3),
                enabled=bool(r["enabled"].get()),
            )
            for r in self._output_rows
        ]

        c.trigger.mode = self.mode_var.get()
        c.trigger.hotkey = self.hotkey_var.get().strip()
        c.trigger.ptt_latch = self.latch_var.get()
        c.trigger.preroll_ms = int(self.preroll.get())

        c.vad.threshold = round(float(self.vad_threshold.get()), 3)
        c.vad.min_silence_ms = int(self.vad_silence.get())
        c.vad.min_speech_ms = int(self.vad_min_speech.get())

        c.tts.voice = self.voice_var.get().strip()
        c.tts.length_scale = round(float(self.speed_var.get()), 3)
        c.tts.volume = round(float(self.volume_var.get()), 3)

        c.stt.model = self.model_var.get().strip()
        c.stt.device = self.stt_device_var.get()
        c.stt.beam_size = int(self.beam_var.get())

        c.updates.repo = self.repo_var.get().strip()
        c.updates.check_on_start = self.check_start_var.get()
        c.updates.interval_hours = int(self.interval_var.get())
        c.validate()
        self.repo_var.set(c.updates.repo)  # reflect any normalisation back to the UI
        return True

    def _apply(self) -> None:
        if not self._collect():
            return
        self.pipeline.apply_audio_changes()
        self.pipeline.apply_tts_changes()
        self.pipeline.apply_vad_changes()
        self.pipeline.set_mode(self.cfg.trigger.mode)
        if self.pipeline.hotkey is not None:
            try:
                self.pipeline.hotkey.set_hotkey(self.cfg.trigger.hotkey)
            except ValueError as exc:
                messagebox.showerror("Hotkey", str(exc), parent=self)

    def _save(self) -> None:
        if not self._collect():
            return
        self._apply()
        self.cfg.save()
        self.append_log("info", f"Saved to {config_path()}")

    # -- periodic refresh -----------------------------------------------------

    def _tick(self) -> None:
        if self._closing or not self.winfo_exists():
            return
        status = self.pipeline.status()
        self.state_label.config(text=f"{status['state']}  —  {status['stt']}")
        cap = self.pipeline.capture
        self.level["value"] = min(100.0, (cap.peak if cap else 0.0) * 140)
        self.after(100, self._tick)

    def close(self) -> None:
        self._closing = True
        if self._on_close:
            self._on_close()
        self.destroy()
