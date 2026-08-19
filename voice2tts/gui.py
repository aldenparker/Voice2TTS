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
from functools import partial
from tkinter import messagebox, simpledialog, ttk

from . import (
    DEFAULT_UPDATE_REPO,
    cable,
    devices,
    gpupack,
    loopback,
    profiles,
    studiopack,
    substitutions,
    theme,
    updater,
    voices,
)
from .config import (
    WHISPER_MODELS,
    Config,
    OutputTarget,
    ProfileEntry,
    SubstitutionRule,
)
from .diagnostics import diagnostics
from .hotkey import describe
from .modes import (
    AddonState,
    RecognitionMode,
    SttDevice,
    Theme,
    TranslationMode,
    TriggerMode,
)
from .paths import config_path, is_frozen, list_voices, log_path
from .pipeline import Pipeline
from .platform_win import run_at_login, set_run_at_login
from .substitutions import STARTER_RULES, Rule
from .translate import LANGUAGE_NAMES

log = logging.getLogger(__name__)

CABLE_URL = "https://vb-audio.com/Cable/"

_TK_MODIFIERS = {
    "Control_L": "ctrl", "Control_R": "ctrl",
    "Alt_L": "alt", "Alt_R": "alt",
    "Shift_L": "shift", "Shift_R": "shift",
    "Super_L": "win", "Super_R": "win",
}


# One output is all most setups need: the virtual cable Discord listens
# to. A second is for hearing yourself.
_DEFAULT_OUTPUTS = 1
_MAX_OUTPUTS = 8

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 620
_MIN_WIDTH = 560
_MIN_HEIGHT = 520


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
        # Applied by the tray app to the shared root; kept here so status text can
        # use semantic colours that work in both light and dark.
        self.palette = theme.resolve(cfg.theme)

        self.title("Voice2TTS Settings")
        # Size is set from the built content in _size_to_content().
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._build()
        # Classic Tk text widgets are outside ttk styling, so they stay white in
        # dark mode unless coloured explicitly.
        for widget in (self.logbox, self.transcript, self.update_notes):
            theme.style_text_widget(widget, self.palette)
        self._load_from_config()
        self._tick()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        nb = ttk.Notebook(self)
        self.nb = nb
        # NOT packed yet. Pack gives space in packing order, so a notebook
        # packed first with expand=True claims the whole window and squeezes
        # out whatever comes after it -- which is how Save/Apply/Close ended up
        # off the bottom edge until the window was dragged bigger. The bar is
        # packed first, against the bottom; the notebook then takes what is
        # left, however little that is.
        # Top level is one tab per thing you might be doing, plus the optional
        # downloads. Everything else is a detail of one of those, and lives
        # nested under Misc -- the way Studio has always nested its own panels.
        # Before this, making translation work meant Translate for the pair,
        # Voice for the voice, Recognition for the model and Add-ons for the
        # download, with nothing saying so.
        self._build_normal(nb)
        self._build_translate(nb)
        self._build_studio(nb)
        self._build_addons(nb)

        misc_outer = ttk.Frame(nb)
        nb.add(misc_outer, text="Misc")
        misc = ttk.Notebook(misc_outer)
        misc.pack(fill="both", expand=True, padx=6, pady=6)
        self.misc_nb = misc
        self._build_audio(misc)
        self._build_trigger(misc)
        self._build_voice_library(misc)
        self._build_words(misc)
        self._build_history(misc)
        self._build_updates(misc)
        self._build_status(misc)

        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", padx=8, pady=8)
        nb.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 0))
        self.state_label = ttk.Label(bar, text="stopped")
        self.state_label.pack(side="left")

        # Profiles live on the button bar rather than in a tab: switching is a thing
        # you do while working, not a thing you configure once.
        ttk.Label(bar, text="  Profile").pack(side="left")
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(bar, textvariable=self.profile_var, width=16,
                                          state="readonly")
        self.profile_combo.pack(side="left", padx=4)
        self.profile_combo.bind("<<ComboboxSelected>>",
                                lambda _e: self._switch_profile())
        ttk.Button(bar, text="Save as...", width=9,
                   command=self._save_profile).pack(side="left")
        ttk.Button(bar, text="Close", command=self.close).pack(side="right")
        ttk.Button(bar, text="Save", command=self._save).pack(side="right", padx=4)
        ttk.Button(bar, text="Apply", command=self._apply).pack(side="right")

        # Open big enough for whatever the tabs actually need. Hardcoding a size
        # goes stale every time a tab grows, which is what happened here.
        self._size_to_content()

    def _size_to_content(self) -> None:
        """Open at the size the content asks for, within the screen."""
        self.update_idletasks()
        # Never smaller than the old default, never bigger than the screen.
        width = min(max(_DEFAULT_WIDTH, self.winfo_reqwidth()),
                    self.winfo_screenwidth() - 80)
        height = min(max(_DEFAULT_HEIGHT, self.winfo_reqheight()),
                     self.winfo_screenheight() - 120)
        self.geometry(f"{width}x{height}")
        # The bar is packed against the bottom now, so it survives any size --
        # but there is no reason to allow one so small nothing is readable.
        self.minsize(min(_MIN_WIDTH, width), min(_MIN_HEIGHT, height))

    # -- audio tab ------------------------------------------------------------

    def _build_audio(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Audio")

        ttk.Label(tab, text="Microphone").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(tab, textvariable=self.input_var, width=52,
                                        state="readonly")
        self.input_combo.grid(row=0, column=1, sticky="ew", padx=6)

        opts = ttk.Frame(tab)
        opts.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.all_apis_var = tk.BooleanVar()
        ttk.Checkbutton(
            opts, text="Show every host API (advanced)", variable=self.all_apis_var,
            command=self._toggle_all_apis,
        ).pack(side="left")
        self.device_count = ttk.Label(opts, text="", foreground="#666")
        self.device_count.pack(side="left", padx=10)

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

        count_row = ttk.Frame(tab)
        count_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Label(count_row, text="Outputs").pack(side="left")
        self.output_count = tk.IntVar(value=_DEFAULT_OUTPUTS)
        ttk.Spinbox(count_row, from_=1, to=_MAX_OUTPUTS, width=4,
                    textvariable=self.output_count,
                    command=self._set_output_count).pack(side="left", padx=(6, 0))
        ttk.Label(count_row,
                  text="One is enough for Discord. Add a second to hear "
                       "yourself through headphones.",
                  foreground=self.palette.muted).pack(side="left", padx=(10, 0))

        cable_row = ttk.Frame(tab)
        cable_row.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.cable_label = ttk.Label(cable_row, text="", cursor="hand2",
                                     wraplength=560, justify="left")
        self.cable_label.pack(anchor="w")
        self.cable_label.bind("<Button-1>", lambda _e: webbrowser.open(CABLE_URL))

        verify_row = ttk.Frame(tab)
        verify_row.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.verify_btn = ttk.Button(verify_row, text="Test the Discord path",
                                     command=self._verify_path)
        self.verify_btn.pack(side="left")
        self.scan_btn = ttk.Button(verify_row, text="Find the right device",
                                   command=self._scan_path)
        self.scan_btn.pack(side="left", padx=6)
        self.cable_btn = ttk.Button(verify_row, text="Remove virtual cable",
                                    command=self._remove_cable)
        self.cable_btn.pack(side="left")
        self.verify_status = ttk.Label(tab, text="", foreground="#666",
                                       wraplength=580, justify="left")
        self.verify_status.grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.mute_var = tk.BooleanVar()
        self.mute_check = ttk.Checkbutton(
            tab,
            text="Mute microphone while speaking (prevents the app hearing itself)",
            variable=self.mute_var,
            # Streaming cannot honour this -- pausing the recording mangles the
            # words -- and the note that says so lives on another tab. Without a
            # command the note only updated when something else refreshed it,
            # so the box and the explanation beside it disagreed.
            command=self._refresh_stt_mode,
        )
        self.mute_check.grid(row=9, column=0, columnspan=2, sticky="w",
                             pady=(10, 0))

        self.autostart_var = tk.BooleanVar(value=run_at_login())
        self.autostart_check = ttk.Checkbutton(
            tab,
            text="Start Voice2TTS when I sign in to Windows",
            variable=self.autostart_var,
            command=self._toggle_autostart,
        )
        self.autostart_check.grid(row=10, column=0, columnspan=2, sticky="w")
        if not is_frozen():
            self.autostart_check.state(["disabled"])
            ttk.Label(tab, text="(only available in an installed build)",
                      foreground="#666").grid(row=11, column=0, sticky="w")

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

    # -- verifying the path to Discord ---------------------------------------

    def _selected_cable(self):
        """The configured cable target, falling back to whatever is detected."""
        for target in self.cfg.audio.outputs:
            if target.enabled and cable.is_virtual_device(target.match):
                for info in cable.list_devices():
                    if info.output_name == target.match or target.match in info.output_name:
                        return info
        return cable.detect()

    def _verify_path(self) -> None:
        info = self._selected_cable()
        if info is None:
            self.verify_status.config(
                text="No virtual cable is configured, so there is no path to test.")
            return
        self.verify_btn.state(["disabled"])
        self.verify_status.config(text="Testing...")

        def work() -> None:
            try:
                result = loopback.verify_cable(
                    info, progress=lambda m: self.after(
                        0, partial(self.verify_status.config, text=m))
                )
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
                self._later(lambda: self._verify_done(None, failure))
                return
            self._later(lambda: self._verify_done(result, None))

        threading.Thread(target=work, daemon=True).start()

    def _verify_done(self, result, error: str | None) -> None:
        if not self.winfo_exists():
            return
        self.verify_btn.state(["!disabled"])
        if error:
            self.verify_status.config(text=f"Test failed: {error}", foreground="#a33")
            return
        self.verify_status.config(
            text=("PASS  " if result.ok else "") + result.message,
            foreground="#2a7" if result.ok else "#a33",
        )
        log.info("loopback %s: %s", "ok" if result.ok else "failed", result.detail)

    def _scan_path(self) -> None:
        """Find which recording device actually receives our audio.

        For a router the naming tells you nothing, so measuring is the only way to
        answer 'what do I pick in Discord?'.
        """
        info = self._selected_cable()
        if info is None:
            self.verify_status.config(text="No virtual cable is configured.")
            return
        self.scan_btn.state(["disabled"])
        self.verify_status.config(text="Scanning every recording device...",
                                  foreground="#666")

        def work() -> None:
            try:
                hits = loopback.scan(
                    info.output_name,
                    progress=lambda m: self.after(
                        0, partial(self.verify_status.config, text=m)),
                )
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
                self._later(lambda: self._scan_done(None, failure, info))
                return
            self._later(lambda: self._scan_done(hits, None, info))

        threading.Thread(target=work, daemon=True).start()

    def _scan_done(self, hits, error: str | None, info) -> None:
        if not self.winfo_exists():
            return
        self.scan_btn.state(["!disabled"])
        if error:
            self.verify_status.config(text=f"Scan failed: {error}", foreground="#a33")
            return
        if not hits:
            extra = ""
            if info.is_router and not info.app_running:
                extra = f"  {info.product} is not running, which would explain it."
            self.verify_status.config(
                text=f"Nothing received audio played into {info.output_name}."
                     f"{extra}",
                foreground="#a33",
            )
            return
        best = hits[0]
        others = f"  (+{len(hits) - 1} more)" if len(hits) > 1 else ""
        self.verify_status.config(
            text=f"PASS  Select this in Discord: {best.input_name}{others}",
            foreground="#2a7",
        )

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
                self._later(lambda: messagebox.showerror(
                    "Uninstall failed", failure, parent=self))
                return
            def removed() -> None:
                self._update_cable_hint()
                messagebox.showinfo("Virtual cable", message, parent=self)

            self._later(removed)

        threading.Thread(target=work, daemon=True).start()

    def _add_output_row(self, target: OutputTarget | None = None) -> None:
        target = target or OutputTarget(match="", gain=1.0, enabled=True)
        row = ttk.Frame(self.outputs_frame)
        row.pack(fill="x", pady=2)

        enabled = tk.BooleanVar(value=target.enabled)
        ttk.Checkbutton(row, variable=enabled, width=2).pack(side="left")

        match = tk.StringVar(value=target.match)
        combo = ttk.Combobox(row, textvariable=match, width=34)
        combo["values"] = ["", *self._device_names("out")]
        combo.pack(side="left", padx=(0, 6))

        gain = tk.DoubleVar(value=target.gain)
        ttk.Scale(row, from_=0.0, to=2.0, variable=gain, length=90).pack(side="left")
        gain_label = ttk.Label(row, width=5, text=f"{target.gain:.2f}")
        gain_label.pack(side="left", padx=(4, 6))
        gain.trace_add("write", lambda *_: gain_label.config(text=f"{gain.get():.2f}"))

        # Live level, so you can see audio actually reaching this device rather
        # than inferring it from silence somewhere downstream.
        meter = ttk.Progressbar(row, maximum=100, length=70)
        meter.pack(side="left", padx=(0, 6))

        entry = {"frame": row, "enabled": enabled, "match": match, "gain": gain,
                 "meter": meter}
        ttk.Button(row, text="X", width=3,
                   command=lambda: self._remove_output_row(entry)).pack(side="left")
        self._output_rows.append(entry)

    def _remove_output_row(self, entry: dict) -> None:
        entry["frame"].destroy()
        self._output_rows.remove(entry)
        # Keep the counter honest: removing a row by hand is the same thing as
        # turning the number down.
        if not self._output_rows:
            self._add_output_row()
        self.output_count.set(len(self._output_rows))

    def _clear_output_rows(self) -> None:
        for entry in self._output_rows:
            entry["frame"].destroy()
        self._output_rows.clear()

    def _set_output_count(self) -> None:
        """Grow or shrink the list to the chosen number.

        Rows are removed from the end, so the one someone set up first survives.
        """
        try:
            wanted = int(self.output_count.get())
        except (tk.TclError, ValueError):
            return
        wanted = max(1, min(_MAX_OUTPUTS, wanted))
        while len(self._output_rows) > wanted:
            entry = self._output_rows.pop()
            entry["frame"].destroy()
        while len(self._output_rows) < wanted:
            self._add_output_row()
        self.output_count.set(wanted)

    def _device_names(self, kind: str) -> list[str]:
        """Names for a picker, virtual cables tagged so they are easy to spot."""
        all_apis = not self.cfg.audio.prefer_wasapi
        listing = (devices.list_inputs(all_apis) if kind == "in"
                   else devices.list_outputs(all_apis))
        # annotate() handles the system-default marker and disambiguates devices
        # that share a name; the virtual-cable tag is added on top of that.
        labels = devices.annotate(listing)
        return [
            label if d.default or not cable.is_virtual_device(d.name)
            else label + devices.VIRTUAL_TAG
            for d, label in zip(listing, labels, strict=True)
        ]

    @staticmethod
    def _strip_tag(value: str) -> str:
        return devices.strip_display(value)

    def _refresh_devices(self) -> None:
        devices.refresh()
        inputs = ["", *self._device_names("in")]
        outputs = ["", *self._device_names("out")]
        self.input_combo["values"] = inputs
        for row in self._output_rows:
            for child in row["frame"].winfo_children():
                if isinstance(child, ttk.Combobox):
                    child["values"] = outputs
        self._update_cable_hint()
        self._update_device_count()

    def _update_device_count(self) -> None:
        all_apis = not self.cfg.audio.prefer_wasapi
        n_in = len(devices.list_inputs(all_apis))
        n_out = len(devices.list_outputs(all_apis))
        extra = "" if all_apis else "  (duplicates across host APIs hidden)"
        self.device_count.config(text=f"{n_in} inputs, {n_out} outputs{extra}")

    def _toggle_all_apis(self) -> None:
        # Inverted: the checkbox offers the advanced view, the config stores the
        # normal one.
        self.cfg.audio.prefer_wasapi = not self.all_apis_var.get()
        self._refresh_devices()

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
            TriggerMode.PTT: "Push to talk only",
            TriggerMode.VAD: "Automatic (voice activity detection)",
            TriggerMode.BOTH: "Both — automatic, plus the hotkey",
        }
        # Over the enum, not a copy of its values: a mode added to modes.py that
        # nobody offered here is how the list came to be written out four times.
        for i, mode in enumerate(TriggerMode):
            ttk.Radiobutton(
                tab, text=labels[mode], value=mode.value, variable=self.mode_var,
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
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 6))

        ttk.Label(tab, text="Speak clipboard").grid(row=8, column=0, sticky="w")
        self.clip_hotkey_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.clip_hotkey_var, width=24).grid(
            row=8, column=1, sticky="w", padx=6, pady=2
        )
        ttk.Label(tab, text="blank = off", foreground="#666").grid(
            row=8, column=2, sticky="w")

        ttk.Label(tab, text="Stop speaking").grid(row=9, column=0, sticky="w")
        self.stop_hotkey_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.stop_hotkey_var, width=24).grid(
            row=9, column=1, sticky="w", padx=6, pady=2
        )
        ttk.Label(tab, text="blank = off", foreground="#666").grid(
            row=9, column=2, sticky="w")

        self.extra_hotkey_error = ttk.Label(tab, text="", foreground="#a33")
        self.extra_hotkey_error.grid(row=10, column=0, columnspan=3, sticky="w",
                                     pady=(0, 8))
        for var in (self.clip_hotkey_var, self.stop_hotkey_var):
            var.trace_add("write", lambda *_: self._validate_hotkey())

        self.vad_header = ttk.Label(tab, text="Detection tuning", font=("", 9, "bold"))
        self.vad_header.grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 4))

        # Tracked so they can be greyed out in push-to-talk mode, where VAD is unused.
        self._vad_widgets: list[tk.Widget] = []
        self.vad_threshold = self._slider(
            tab, 12, "Sensitivity threshold", 0.05, 0.95, "{:.2f}", vad_only=True
        )
        self.vad_silence = self._slider(
            tab, 13, "End-of-speech silence (ms)", 200, 2000, "{:.0f}", vad_only=True
        )
        self.vad_min_speech = self._slider(
            tab, 14, "Minimum speech (ms)", 50, 1000, "{:.0f}", vad_only=True
        )
        self.preroll = self._slider(tab, 15, "Pre-roll kept (ms)", 0, 1000, "{:.0f}")

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
                w.config(state=state)  # type: ignore[call-arg]  # Misc in stubs
            except tk.TclError:
                pass  # plain ttk.Label has no state option on some themes

    def _validate_hotkey(self) -> bool:
        err = describe(self.hotkey_var.get())
        self.hotkey_error.config(text=err)

        problems = []
        combos = {"push-to-talk": self.hotkey_var.get().strip()}
        for label, var in (("speak clipboard", self.clip_hotkey_var),
                           ("stop speaking", self.stop_hotkey_var)):
            value = var.get().strip()
            if not value:
                continue  # blank means disabled, not invalid
            extra = describe(value)
            if extra:
                problems.append(f"{label}: {extra}")
            combos[label] = value

        # Two actions on one combination means the second silently never fires.
        seen: dict[str, str] = {}
        for label, combo in combos.items():
            key = combo.lower()
            if key and key in seen:
                problems.append(f"{seen[key]} and {label} use the same combination")
            seen[key] = label

        self.extra_hotkey_error.config(text="; ".join(problems))
        return not err and not problems

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

    def _build_normal(self, nb: ttk.Notebook) -> None:
        """Everything ordinary speech needs, in the order it matters."""
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Normal")
        self._build_voice(tab)
        self._build_recognition(tab)

    def _build_voice(self, parent: ttk.Frame) -> None:
        tab = ttk.LabelFrame(parent, text="Voice", padding=10)
        tab.pack(fill="x", pady=(0, 10))

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

    def _trans_mode(self) -> TranslationMode:
        """The one setting the tick box and the two radio buttons stand for.

        Two widgets are the right interface -- "translate" and "how" are
        separate questions to a person -- but they are one fact, and this is the
        only place that turns them back into it. They used to be turned into two
        config fields that could contradict each other.
        """
        if not hasattr(self, "trans_enabled") or not self.trans_enabled.get():
            return TranslationMode.OFF
        return (TranslationMode.parse(self.trans_method.get())
                or TranslationMode.MODELS)

    def _show_trans_mode(self, mode: TranslationMode) -> None:
        """Point the two widgets at one mode. The inverse of _trans_mode."""
        self.trans_enabled.set(mode.translating)
        if mode.translating:
            self.trans_method.set(mode.value)

    def _current_plan(self):
        """What the app would do with the settings as they are shown.

        Built from the WIDGETS, not the saved config, so every warning answers
        "what happens if I press Apply" rather than "what happened last time".
        """
        from dataclasses import replace

        from . import plan as plan_mod

        cfg = self.cfg
        stt, translation = cfg.stt, cfg.translation
        if hasattr(self, "model_var"):
            stt = replace(
                stt,
                model=self.model_var.get().strip() or stt.model,
                language=self.stt_lang_var.get().strip() or stt.language,
            )
        if hasattr(self, "trans_enabled"):
            translation = replace(
                translation,
                mode=self._trans_mode(),
                source=self._source_code(),
                target=self._target_code(),
            )
        snapshot = replace(cfg, stt=stt, translation=translation,
                           tts=replace(cfg.tts,
                                       voice=self.voice_var.get().strip()))
        return plan_mod.build(snapshot)

    def _check_language(self) -> None:
        """Show whatever is wrong with the plan, worked out in one place.

        This used to compare the voice against the RECOGNITION model, which is
        the wrong question the moment translation is on: a Japanese voice is
        exactly right for English-to-Japanese, and was reported as "not an
        English voice" because the check knew nothing about translation.
        """
        if not hasattr(self, "lang_warning") or not hasattr(self, "model_var"):
            return
        current = self._current_plan()
        trouble = current.serious or current.problems
        self.lang_warning.config(
            text=str(trouble[0]).split("\n\n")[0] if trouble else "")

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

        self.lib_multi = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Multi-speaker only",
                        variable=self.lib_multi,
                        command=self._refresh_library).pack(side="left", padx=(12, 0))

        # "Speakers" is here for the Voice Designer, which needs a voice with
        # more than one. Without it the Design tab can only say "download a
        # multi-speaker voice" and leave you guessing which of 174 those are.
        cols = ("voice", "language", "quality", "size", "speakers", "state")
        self.lib_tree = ttk.Treeview(tab, columns=cols, show="headings", height=13)
        for col, width in zip(cols, (170, 140, 65, 65, 70, 80), strict=True):
            self.lib_tree.heading(col, text=col.title())
            self.lib_tree.column(col, width=width, anchor="w")
        self.lib_tree.grid(row=1, column=0, columnspan=4, sticky="nsew")
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.lib_tree.yview)
        sb.grid(row=1, column=4, sticky="ns")
        self.lib_tree.configure(yscrollcommand=sb.set)

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Button(actions, text="Preview", command=self._preview_voice).pack(side="left")
        ttk.Button(actions, text="Download", command=self._download_voice).pack(
            side="left", padx=6)
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
            # The speaker count is only known from the catalogue, so it stays
            # blank until that is loaded. One value per column, or every field
            # after the gap shows the wrong thing.
            if state != "bundled" and voices.missing_phonemizer(key):
                state = "needs add-on"
            self.lib_tree.insert("", "end", iid=key,
                                 values=(key, "-", "-", "-", "", state))
        self.lib_status.config(text="Load the catalogue to browse more voices.")

    def _load_catalogue(self) -> None:
        self.lib_status.config(text="Fetching catalogue...")

        def work() -> None:
            try:
                entries = voices.fetch_catalogue()
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self._later(lambda: self.lib_status.config(text=f"Failed: {failure}"))
                return

            def apply() -> None:
                self._catalogue = entries
                langs = voices.languages(entries)
                self.lib_lang_combo["values"] = ["(all)", *langs]
                if self.lib_lang.get() not in langs:
                    self.lib_lang.set("en_US" if "en_US" in langs else "(all)")
                self._refresh_library()

            self._later(apply)

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
        if self.lib_multi.get():
            entries = [e for e in entries if e.multi_speaker]
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
                values=(e.key, e.language_label, e.quality, f"{e.size_mb:.0f} MB",
                        str(e.num_speakers) if e.multi_speaker else "", state),
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

    def _preview_voice(self) -> None:
        """Hear a voice before spending 60 MB on it."""
        key = self._selected_voice()
        if key is None:
            return
        entry = next((e for e in self._catalogue if e.key == key), None)
        if entry is None:
            self.lib_status.config(
                text="Load the catalogue first — previews come from there.")
            return
        self.lib_status.config(text=f"Fetching a sample of {key}...")

        def work() -> None:
            try:
                seconds = voices.play_sample(entry)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
                self._later(lambda: self.lib_status.config(
                    text=f"No preview available: {failure}"))
                return
            self._later(lambda: self.lib_status.config(
                text=f"Playing {key} ({seconds:.1f}s)"))

        threading.Thread(target=work, daemon=True).start()

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
                self._later(lambda: self.lib_status.config(text=f"Failed: {failure}"))
                return
            self._later(lambda: self._after_library_change(f"Installed {key}"))

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

    # -- translate tab --------------------------------------------------------

    def _build_translate(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Translate")

        ttk.Label(
            tab,
            text="Speak one language, have the far end hear another. Runs on this "
                 "machine;\nnothing is sent anywhere. Models are downloaded once, "
                 "about 60 MB per direction.",
            foreground="#555", justify="left",
        ).grid(row=0, column=0, columnspan=5, sticky="w")

        self.trans_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Translate what I say", variable=self.trans_enabled,
                        command=self._refresh_translation_views).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(6, 4))

        method = ttk.Frame(tab)
        method.grid(row=1, column=1, columnspan=4, sticky="e", pady=(6, 4))
        # Whisper can translate to English itself, for free, as part of
        # recognition. It is strictly worse than a dedicated model for quality,
        # and cannot do anything but English -- but it needs no download, which
        # makes it the right default for the one case it covers.
        self.trans_method = tk.StringVar(value=TranslationMode.MODELS.value)
        ttk.Label(method, text="Using:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(method, text="Downloaded models",
                        value=TranslationMode.MODELS.value,
                        variable=self.trans_method,
                        command=self._switch_translate_method).pack(side="left")
        ttk.Radiobutton(method, text="The recogniser (English only)",
                        value=TranslationMode.RECOGNISER.value,
                        variable=self.trans_method,
                        command=self._switch_translate_method).pack(
            side="left", padx=(8, 0))

        picker = ttk.Frame(tab)
        picker.grid(row=2, column=0, columnspan=5, sticky="w")
        ttk.Label(picker, text="I speak").pack(side="left")
        self.trans_source = tk.StringVar(value="en")
        self.trans_source_combo = ttk.Combobox(picker, textvariable=self.trans_source,
                                               width=14, state="readonly")
        self.trans_source_combo.pack(side="left", padx=4)
        ttk.Label(picker, text="they hear").pack(side="left", padx=(10, 0))
        self.trans_target = tk.StringVar(value="de")
        self.trans_target_combo = ttk.Combobox(picker, textvariable=self.trans_target,
                                               width=14, state="readonly")
        self.trans_target_combo.pack(side="left", padx=4)
        for combo in (self.trans_source_combo, self.trans_target_combo):
            combo.bind("<<ComboboxSelected>>",
                       lambda _e: self._refresh_translation_views())

        # The route line is where every reason this will not work gets said: no
        # model, a pivot that will compound errors, an English-only recogniser,
        # or a voice that speaks a different language than the output.
        self.trans_route = ttk.Label(tab, text="", justify="left", foreground="#555")
        self.trans_route.grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 4))

        # Warning about a mismatched voice is not much use if fixing it means
        # working out which of the installed voices speaks German. When one
        # does, this switches to it; when none does, it says where to get one.
        self.trans_voice_btn = ttk.Button(tab, text="Use a matching voice",
                                          command=self._use_matching_voice)
        self.trans_voice_btn.grid(row=3, column=4, sticky="e", padx=(8, 0))

        top = ttk.Frame(tab)
        top.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(4, 6))
        ttk.Button(top, text="Load catalogue",
                   command=self._load_translate_catalogue).pack(side="left")
        ttk.Button(top, text="Download", command=self._download_pair).pack(
            side="left", padx=6)
        ttk.Button(top, text="Remove", command=self._remove_pair).pack(side="left")
        self.trans_status = ttk.Label(top, text="", foreground="#666")
        self.trans_status.pack(side="left", padx=12)

        cols = ("pair", "size", "state", "licence")
        self.trans_tree = ttk.Treeview(tab, columns=cols, show="headings", height=9)
        for col, width, heading in zip(cols, (220, 80, 90, 110),
                                       ("Direction", "Size", "State", "Licence"),
                                       strict=True):
            self.trans_tree.heading(col, text=heading)
            self.trans_tree.column(col, width=width, anchor="w")
        self.trans_tree.grid(row=5, column=0, columnspan=5, sticky="nsew")
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.trans_tree.yview)
        sb.grid(row=5, column=5, sticky="ns")
        self.trans_tree.configure(yscrollcommand=sb.set)

        ttk.Label(
            tab,
            text="Models are converted from Helsinki-NLP's OPUS-MT and used under "
                 "CC-BY-4.0.\nThe attribution travels with each one, in a LICENSE "
                 "file beside it.",
            foreground="#777", justify="left",
        ).grid(row=6, column=0, columnspan=5, sticky="w", pady=(6, 0))

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(5, weight=1)
        self._translate_catalogue: list = []
        self._refresh_translate_list()

    def _refresh_translate_list(self) -> None:
        """Show what is installed, merged with the catalogue if it was fetched."""
        from . import translate

        self.trans_tree.delete(*self.trans_tree.get_children())
        rows: dict[str, tuple] = {}
        # Installed first, so a model that is here but not in the catalogue --
        # an older release, or one the network cannot confirm -- still shows.
        for pair in translate.installed_pairs():
            label = (f"{translate.language_name(pair.source)} → "
                     f"{translate.language_name(pair.target)}")
            rows[pair.code] = (label, "-", "installed", "CC-BY-4.0")
        for entry in self._translate_catalogue:
            size = f"{entry.size / 1e6:.0f} MB" if entry.size else "-"
            rows[entry.code] = (entry.label, size,
                                "installed" if entry.installed else "available",
                                entry.licence or "unknown")
        for code in sorted(rows):
            self.trans_tree.insert("", "end", iid=code, values=rows[code])

        # Every language we can name, not only the installed ones. Offering
        # just what is installed made the pickers a chicken and egg: you could
        # not choose German until you had the German model, and you could not
        # get the German model without choosing German.
        codes = set(translate.LANGUAGE_NAMES)
        codes |= {part for code in rows for part in code.split("_")}
        codes |= {self._source_code(), self._target_code(), "en"}
        listed = [self._language_label(c) for c in sorted(codes)]
        self.trans_source_combo["values"] = listed
        self.trans_target_combo["values"] = listed
        self._refresh_translate_route()

    @staticmethod
    def _language_label(code: str) -> str:
        from . import translate

        name = translate.language_name(code)
        return code if name == code else f"{code} ({name})"

    @staticmethod
    def _language_code(display: str) -> str:
        """The code out of a "de (German)" entry, or whatever was typed."""
        return (display or "").strip().split(" ", 1)[0].strip().lower()

    def _source_code(self) -> str:
        return self._language_code(self.trans_source.get())

    def _target_code(self) -> str:
        return self._language_code(self.trans_target.get())

    def _switch_translate_method(self) -> None:
        """Switch method WITHOUT touching the language pair.

        This used to move the target to English, on the reasoning that the
        recogniser can only produce English. It also saved that -- so trying the
        recogniser once turned a saved "English to German" into "English to
        English", which validate() then treats as a no-op and switches
        translation off. Ticking the box afterwards appeared to do nothing.

        The route line says the output will be English; the stored target is
        left exactly as the user set it.
        """
        self._refresh_translate_route()
        self._check_language()

    def _refresh_translation_views(self) -> None:
        """Both views of the plan, refreshed together.

        Leaving each caller to remember the route line AND the warning is how
        they came to show different things at the same moment.
        """
        self._refresh_translate_route()
        self._check_language()

    def _refresh_translate_route(self) -> None:
        """Redraw the route line AND the voice warning.

        One entry point, because they are two views of the same question and
        leaving each caller to remember both is what left "not an English
        voice" on screen while translating INTO that language. The route line
        has several early returns; wrapping it means every one of them still
        updates the warning.
        """
        self._render_route()
        self._check_language()

    def _render_route(self) -> None:
        """Say what will happen, including every reason it might not work.

        Renders the plan; it does not work one out. This used to be ninety lines
        of its own reasoning about models, routes, pivots and voices, running
        beside plan.build()'s -- and the two disagreed, which is the whole
        reason this sweep happened. Everything it used to decide now lives in
        plan.build() and is decided once.
        """
        current = self._current_plan()
        self._update_voice_button(current.spoken if current.translating else "")

        if not current.translating:
            self.trans_route.config(
                text="Translation is off; your own words are spoken as recognised.",
                foreground=self.palette.muted)
            return

        notes = [problem.text for problem in current.problems]
        text = current.summary + ("\nNote: " + " ".join(notes) if notes else "")
        self.trans_route.config(
            text=text,
            foreground=("#a00" if current.serious
                        else "#a60" if notes else "#070"))

    def _matching_voice(self, language: str) -> str | None:
        """An installed voice that speaks `language`, or None."""
        if not language:
            return None
        return next((key for key in voices.installed_keys()
                     if voices.voice_language(key) == language), None)

    def _update_voice_button(self, wanted: str) -> None:
        """Offer the fix only when there is one to offer."""
        if not hasattr(self, "trans_voice_btn"):
            return
        match = self._matching_voice(wanted)
        if match and self.voice_var.get().strip() != match:
            self.trans_voice_btn.grid()
            self.trans_voice_btn.config(text=f"Use {match}")
        else:
            self.trans_voice_btn.grid_remove()

    def _use_matching_voice(self) -> None:
        """Switch to a voice that speaks the target language."""
        # plan.spoken, not the target: with translation on but no model for
        # the pair, the text stays in the SOURCE language, and offering a German
        # voice there produced exactly what it sounds like -- English read with
        # a German accent.
        wanted = self._current_plan().spoken
        match = self._matching_voice(wanted)
        if match is None:
            self.trans_status.config(
                text=f"No installed voice speaks {wanted}. "
                     "The Voice library tab can fetch one.")
            return
        self.voice_var.set(match)
        self.cfg.tts.voice = match
        self.pipeline.apply_tts_changes()
        self.trans_status.config(text=f"Now speaking with {match}")
        self._refresh_translate_route()
        self._check_language()

    def _voice_language(self) -> str:
        """The language of the selected voice, or "" if it cannot be told.

        No try/except: voice_language already returns "" for anything it cannot
        work out, and a broad catch here hid the fact that this was being passed
        a Path when it wants a voice key -- so the mismatch was never reported.
        """
        return voices.voice_language(self.voice_var.get().strip())

    def _load_translate_catalogue(self) -> None:
        from . import translate

        self.trans_status.config(text="Fetching catalogue...")
        repo = self.cfg.updates.repo

        def work() -> None:
            entries = translate.fetch_catalogue(repo)

            def apply() -> None:
                self._translate_catalogue = entries
                self._refresh_translate_list()
                if entries:
                    self.trans_status.config(
                        text=f"{len(entries)} direction(s) available")
                else:
                    # "Could not reach the catalogue" blamed the network for
                    # what is usually a repository that has never published
                    # any, which sends people looking in the wrong place.
                    self.trans_status.config(
                        text=f"No models published at {repo} "
                             f"({translate.MODELS_TAG}), or no connection. "
                             "Publishing needs the 'Translation models' action "
                             f"-- or a {translate.MODELS_TAG} tag -- to be run "
                             "once on that repository.")

            self._later(apply)

        threading.Thread(target=work, daemon=True).start()

    def _selected_pair_code(self) -> str | None:
        selection = self.trans_tree.selection()
        if not selection:
            self.trans_status.config(text="Select a direction first.")
            return None
        return selection[0]

    def _download_pair(self) -> None:
        code = self._selected_pair_code()
        if code is None:
            return
        entry = next((e for e in self._translate_catalogue if e.code == code), None)
        if entry is None:
            self.trans_status.config(text="Load the catalogue first.")
            return
        if entry.installed:
            self.trans_status.config(text=f"{entry.label} is already installed.")
            return

        from . import translate

        repo = self.cfg.updates.repo
        label = entry.label

        def report(done: int, total: int) -> None:
            share = f"{done / max(total, 1) * 100:.0f}%"
            self._later(lambda: self.trans_status.config(
                text=f"Downloading {label}... {share}"))

        def work() -> None:
            try:
                translate.download_pair(entry, repo, progress=report)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self._later(lambda: self.trans_status.config(
                    text=f"Failed: {failure}"))
                return

            def done() -> None:
                self._refresh_translate_list()
                self.trans_status.config(text=f"Installed {label}")
                # A model that arrives while translation is already on should
                # start working without a restart.
                if self.cfg.translation.mode is TranslationMode.MODELS:
                    self.pipeline.apply_translation_changes()

            self._later(done)

        self.trans_status.config(text=f"Downloading {label}...")
        threading.Thread(target=work, daemon=True).start()

    def _remove_pair(self) -> None:
        code = self._selected_pair_code()
        if code is None:
            return
        source, _, target = code.partition("_")

        from . import translate

        label = (f"{translate.language_name(source)} to "
                 f"{translate.language_name(target)}")
        if not messagebox.askyesno("Remove model", f"Delete the {label} model?",
                                   parent=self):
            return
        if translate.remove_pair(source, target):
            self.trans_status.config(text=f"Removed {label}")
            # If the running chain was using it, it has to let go.
            if self.cfg.translation.mode is TranslationMode.MODELS:
                self.pipeline.apply_translation_changes()
        else:
            self.trans_status.config(text="It was not installed.")
        self._refresh_translate_list()

    # -- add-ons tab ----------------------------------------------------------

    @staticmethod
    def _pack_line(status, describe):
        """(AddonState, what to show). Present and broken is its own answer.

        This returned a boolean, so a pack that was unpacked but would not load
        had to be reported as one or the other -- and as "not installed" it
        offered to download something already downloaded, which is exactly what
        the user had been doing.
        """
        note = describe(status)
        problem = getattr(status, "problem", "")
        if problem and problem != "missing":
            return AddonState.BROKEN, f"{note} -- not working: {problem}"
        if status.installed and status.usable:
            return AddonState.READY, note
        return AddonState.MISSING, note

    def _addon_specs(self) -> list[dict]:
        """Every optional download, described in one place.

        A table rather than three hand-written panels: they differ only in what
        they cost and what they unlock, and writing that three times is how the
        GPU pack ended up explained on the Recognition tab while the Studio one
        was explained inside Studio and neither mentioned the other.
        """
        from . import jppack

        def gpu_available() -> tuple[bool, str]:
            if gpupack.gpu_present():
                return True, ""
            return False, "No NVIDIA graphics card was found."

        def studio_available() -> tuple[bool, str]:
            verdict = studiopack.gate(self.cfg.studio.ignore_hardware_check)
            return verdict.ok, "" if verdict.ok else verdict.summary

        return [
            {
                "key": "gpu",
                "title": "GPU acceleration",
                "blurb": ("Runs recognition on an NVIDIA card instead of the "
                          "processor, and upgrades the recognition model to "
                          "small.en."),
                "size": "about 1.9 GB",
                "status": lambda: self._pack_line(
                    gpupack.status(),
                    lambda s: f"{s.dll_count} libraries, {s.size_mb:.0f} MB"),
                "available": gpu_available,
                "install": lambda report: gpupack.install(progress=report),
                "uninstall": gpupack.uninstall,
                "removed": self._after_gpu_removed,
            },
            {
                "key": "japanese",
                "title": "Japanese voices",
                "blurb": ("Japanese voices are built on a different phonemizer, "
                          "which Piper does not carry. Without it a Japanese "
                          "voice cannot speak at all."),
                "size": f"about {jppack.APPROX_DOWNLOAD_MB} MB "
                        f"({jppack.APPROX_INSTALLED_MB} MB on disk)",
                "status": lambda: self._pack_line(
                    jppack.status(), lambda s: f"{s.size_mb:.0f} MB"),
                "available": lambda: (True, ""),
                "install": lambda report: jppack.install(progress=report),
                "uninstall": jppack.uninstall,
                "removed": None,
            },
            {
                "key": "studio",
                "title": "Voice Studio training",
                "blurb": ("Python, PyTorch and the Piper training code, needed "
                          "to train a voice of your own. Designing a voice from "
                          "existing ones does not need it."),
                "size": f"about {studiopack.APPROX_DOWNLOAD_GB:.0f} GB",
                "status": lambda: (studiopack.status().usable,
                                   f"{studiopack.status().size_gb:.1f} GB"),
                "available": studio_available,
                "install": lambda report: studiopack.install(progress=report),
                "uninstall": studiopack.uninstall,
                "removed": None,
            },
        ]

    def _build_addons(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Add-ons")
        self.addons_tab = tab

        ttk.Label(
            tab,
            text="Optional downloads. Nothing here is needed for ordinary use, "
                 "and nothing is fetched\nuntil you ask for it.",
            foreground=self.palette.muted, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self._addon_rows: dict[str, dict] = {}
        for spec in self._addon_specs():
            self._build_addon_row(tab, spec)

        self._refresh_addons()

    def _build_addon_row(self, parent: ttk.Frame, spec: dict) -> None:
        frame = ttk.LabelFrame(parent, text=spec["title"], padding=10)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text=spec["blurb"], foreground=self.palette.muted,
                  justify="left", wraplength=620).grid(
            row=0, column=0, columnspan=2, sticky="w")

        state = ttk.Label(frame, text="", justify="left")
        state.grid(row=1, column=0, sticky="w", pady=(6, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=1, column=1, sticky="e", pady=(6, 0))
        button = ttk.Button(actions, text="Download",
                            command=partial(self._toggle_addon, str(spec["key"])))
        button.pack(side="left")
        progress = ttk.Progressbar(actions, mode="indeterminate", length=140)

        note = ttk.Label(frame, text="", foreground=self.palette.muted,
                         justify="left", wraplength=620)
        note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        frame.columnconfigure(0, weight=1)
        self._addon_rows[spec["key"]] = {
            "spec": spec, "state": state, "button": button,
            "progress": progress, "note": note, "busy": False,
        }

    def _refresh_addons(self) -> None:
        """Say what each add-on costs, and whether it is here."""
        if not hasattr(self, "_addon_rows"):
            return
        for row in self._addon_rows.values():
            if row["busy"]:
                continue
            spec = row["spec"]
            try:
                state, detail = spec["status"]()
            except Exception as exc:  # noqa: BLE001 - a probe must not break the tab
                log.debug("could not read %s status: %s", spec["key"], exc)
                state, detail = AddonState.MISSING, ""
            allowed, why_not = spec["available"]()
            row["addon_state"] = state

            match state:
                case AddonState.READY:
                    row["state"].config(text=f"Installed — {detail}")
                    row["button"].config(text="Remove", state="normal")
                    row["note"].config(text="")
                case AddonState.BROKEN:
                    # Repair, not Download: the files are there, and fetching
                    # them again is what the user has already tried.
                    row["state"].config(text=f"Installed — {detail}")
                    row["button"].config(text="Repair", state="normal")
                    row["note"].config(
                        text="Repair downloads it again, including anything "
                             "that was missing. Remove it from the log tab if "
                             "you would rather start over.")
                case _ if not allowed:
                    row["state"].config(text="Not available on this machine")
                    row["button"].config(text="Download", state="disabled")
                    row["note"].config(text=why_not)
                case _:
                    row["state"].config(text=f"Not installed — {spec['size']}")
                    row["button"].config(text="Download", state="normal")
                    row["note"].config(text="")

    def _toggle_addon(self, key: str) -> None:
        row = self._addon_rows[key]
        spec = row["spec"]
        state, _ = spec["status"]()

        # A broken pack falls through to the install path below: downloading it
        # again is the repair, and it is the only button that can help.
        if state is AddonState.READY:
            if not messagebox.askyesno(
                f"Remove {spec['title']}",
                f"Delete the downloaded files for {spec['title']}?",
                parent=self,
            ):
                return
            spec["uninstall"]()
            if spec["removed"]:
                spec["removed"]()
            self._refresh_addons()
            self._refresh_addon_dependents()
            return

        row["busy"] = True
        row["button"].config(state="disabled")
        row["progress"].pack(side="left", padx=8)
        row["progress"].start(12)

        def report(message: str) -> None:
            self._later(lambda: row["note"].config(text=message))

        def work() -> None:
            try:
                spec["install"](report)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self._later(lambda: self._addon_done(key, failure))
                return
            self._later(lambda: self._addon_done(key, None))

        threading.Thread(target=work, daemon=True).start()

    def _addon_done(self, key: str, error: str | None) -> None:
        if not self.winfo_exists():
            return
        row = self._addon_rows[key]
        row["busy"] = False
        row["progress"].stop()
        row["progress"].pack_forget()
        row["button"].config(state="normal")
        if error:
            row["note"].config(text=f"Failed: {error}")
        self._refresh_addons()
        self._refresh_addon_dependents()

    def _refresh_addon_dependents(self) -> None:
        """Anything whose advice depends on what is now installed.

        Installing the Japanese pack makes voices speakable that were not a
        moment ago, so every place that says so has to be asked again -- and
        the cached "does this import" answer has to be thrown away first, or it
        keeps reporting the state from before the download.
        """
        voices.forget_import_checks()
        self._check_language()
        if hasattr(self, "trans_enabled"):
            self._refresh_translate_route()
        if hasattr(self, "studio_setup_note"):
            self._refresh_studio()

    def _after_gpu_removed(self) -> None:
        """The GPU model is gone with the pack, so stop asking for it."""
        self.cfg.stt.model = "base.en"
        if hasattr(self, "model_var"):
            self.model_var.set("base.en")

    # -- words tab ------------------------------------------------------------

    def _build_words(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Words")

        ttk.Label(
            tab,
            text="Rewrite text between recognition and speech. Fixes names the "
                 "recogniser mishears,\nexpands abbreviations, and corrects words "
                 "the voice pronounces badly.",
            foreground="#555", justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        self.subs_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Apply these rules", variable=self.subs_enabled_var,
                        command=self._refresh_subs_preview).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 4))

        which = ttk.Frame(tab)
        which.grid(row=1, column=1, columnspan=3, sticky="e", pady=(6, 4))
        self.subs_which = tk.StringVar(value="source")
        ttk.Label(which, text="Rules for:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(which, text="What I said", value="source",
                        variable=self.subs_which,
                        command=self._switch_sub_list).pack(side="left")
        ttk.Radiobutton(which, text="What is spoken", value="target",
                        variable=self.subs_which,
                        command=self._switch_sub_list).pack(side="left", padx=(8, 0))

        cols = ("pattern", "replacement", "opts")
        self.subs_tree = ttk.Treeview(tab, columns=cols, show="headings", height=10)
        for col, width, text in zip(cols, (170, 230, 110),
                                    ("Heard", "Spoken as", "Options"), strict=True):
            self.subs_tree.heading(col, text=text)
            self.subs_tree.column(col, width=width, anchor="w")
        self.subs_tree.grid(row=2, column=0, columnspan=4, sticky="nsew")
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.subs_tree.yview)
        sb.grid(row=2, column=4, sticky="ns")
        self.subs_tree.configure(yscrollcommand=sb.set)
        self.subs_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_sub_row())
        self.subs_tree.bind("<Double-1>", lambda _e: self._toggle_sub_row())

        editor = ttk.Frame(tab)
        editor.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        ttk.Label(editor, text="Heard").grid(row=0, column=0, sticky="w")
        self.sub_pattern = tk.StringVar()
        ttk.Entry(editor, textvariable=self.sub_pattern, width=24).grid(
            row=0, column=1, padx=4)
        ttk.Label(editor, text="Spoken as").grid(row=0, column=2, sticky="w",
                                                 padx=(10, 0))
        self.sub_replacement = tk.StringVar()
        ttk.Entry(editor, textvariable=self.sub_replacement, width=28).grid(
            row=0, column=3, padx=4)

        flags = ttk.Frame(tab)
        flags.grid(row=4, column=0, columnspan=4, sticky="w")
        self.sub_whole = tk.BooleanVar(value=True)
        self.sub_regex = tk.BooleanVar(value=False)
        self.sub_case = tk.BooleanVar(value=False)
        ttk.Checkbutton(flags, text="Whole word", variable=self.sub_whole).pack(
            side="left")
        ttk.Checkbutton(flags, text="Regular expression", variable=self.sub_regex).pack(
            side="left", padx=8)
        ttk.Checkbutton(flags, text="Match case", variable=self.sub_case).pack(
            side="left")

        buttons = ttk.Frame(tab)
        buttons.grid(row=5, column=0, columnspan=4, sticky="w", pady=6)
        ttk.Button(buttons, text="Add / update", command=self._add_sub).pack(side="left")
        ttk.Button(buttons, text="Remove", command=self._remove_sub).pack(
            side="left", padx=6)
        ttk.Button(buttons, text="Add common abbreviations",
                   command=self._add_starter_subs).pack(side="left")
        self.subs_status = ttk.Label(buttons, text="", foreground="#666")
        self.subs_status.pack(side="left", padx=10)

        ttk.Separator(tab, orient="horizontal").grid(
            row=6, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Label(tab, text="Try it").grid(row=7, column=0, sticky="w")
        self.subs_sample = tk.StringVar(value="brb, tell Aiden gg")
        ttk.Entry(tab, textvariable=self.subs_sample).grid(
            row=7, column=1, columnspan=3, sticky="ew", padx=4)
        self.subs_sample.trace_add("write", lambda *_: self._refresh_subs_preview())
        self.subs_preview = ttk.Label(tab, text="", foreground="#2a7",
                                      wraplength=560, justify="left")
        self.subs_preview.grid(row=8, column=0, columnspan=4, sticky="w", pady=(4, 0))

        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)
        self._subs: list[SubstitutionRule] = []
        self._target_subs: list[SubstitutionRule] = []

    def _active_subs(self) -> list[SubstitutionRule]:
        """Whichever list the radio buttons are pointing at.

        Two lists rather than one because translation sits between them: source
        rules fix what the recogniser misheard and have to run before the text
        is translated, target rules fix what the voice says badly and are a
        property of the output language.
        """
        return (self._target_subs if self.subs_which.get() == "target"
                else self._subs)

    def _switch_sub_list(self) -> None:
        """Show the other list. The editor is cleared, because leaving a rule
        from the source list in the fields makes it far too easy to add it to
        the target list by accident."""
        self.sub_pattern.set("")
        self.sub_replacement.set("")
        self._render_subs()
        self._refresh_subs_preview()

    def _render_subs(self) -> None:
        rules = self._active_subs()
        self.subs_tree.delete(*self.subs_tree.get_children())
        for i, rule in enumerate(rules):
            opts = []
            if not rule.enabled:
                opts.append("off")
            if rule.regex:
                opts.append("regex")
            if rule.case_sensitive:
                opts.append("case")
            if not rule.whole_word:
                opts.append("partial")
            self.subs_tree.insert("", "end", iid=str(i),
                                  values=(rule.pattern, rule.replacement,
                                          ", ".join(opts)))
        self._refresh_subs_preview()

    def _selected_sub(self) -> int | None:
        sel = self.subs_tree.selection()
        return int(sel[0]) if sel else None

    def _load_sub_row(self) -> None:
        rules = self._active_subs()
        index = self._selected_sub()
        if index is None or index >= len(rules):
            return
        rule = rules[index]
        self.sub_pattern.set(rule.pattern)
        self.sub_replacement.set(rule.replacement)
        self.sub_whole.set(rule.whole_word)
        self.sub_regex.set(rule.regex)
        self.sub_case.set(rule.case_sensitive)

    def _toggle_sub_row(self) -> None:
        rules = self._active_subs()
        index = self._selected_sub()
        if index is None or index >= len(rules):
            return
        rules[index].enabled = not rules[index].enabled
        self._render_subs()

    def _add_sub(self) -> None:
        rules = self._active_subs()
        pattern = self.sub_pattern.get().strip()
        if not pattern:
            self.subs_status.config(text="Enter what is heard first.")
            return
        rule = SubstitutionRule(
            pattern=pattern,
            replacement=self.sub_replacement.get(),
            whole_word=self.sub_whole.get(),
            regex=self.sub_regex.get(),
            case_sensitive=self.sub_case.get(),
        )
        error = Rule(rule.pattern, rule.replacement, regex=rule.regex).describe_error()
        if error:
            self.subs_status.config(text=error)
            return
        existing = next((i for i, r in enumerate(rules)
                         if r.pattern.lower() == pattern.lower()), None)
        if existing is None:
            rules.append(rule)
            self.subs_status.config(text=f"Added {pattern}")
        else:
            rule.enabled = rules[existing].enabled
            rules[existing] = rule
            self.subs_status.config(text=f"Updated {pattern}")
        self._render_subs()

    def _remove_sub(self) -> None:
        rules = self._active_subs()
        index = self._selected_sub()
        if index is None or index >= len(rules):
            self.subs_status.config(text="Select a rule first.")
            return
        removed = rules.pop(index)
        self.subs_status.config(text=f"Removed {removed.pattern}")
        self._render_subs()

    def _add_starter_subs(self) -> None:
        rules = self._active_subs()
        have = {r.pattern.lower() for r in rules}
        added = 0
        for starter in STARTER_RULES:
            if starter.pattern.lower() in have:
                continue
            rules.append(SubstitutionRule(
                pattern=starter.pattern, replacement=starter.replacement))
            added += 1
        self.subs_status.config(text=f"Added {added}" if added else "Already present")
        self._render_subs()

    def _refresh_subs_preview(self) -> None:
        if not hasattr(self, "subs_preview"):
            return
        rules = self._active_subs()
        sample = self.subs_sample.get()
        if not self.subs_enabled_var.get():
            self.subs_preview.config(text="(rules are switched off)",
                                     foreground="#666")
            return
        compiled = [Rule(r.pattern, r.replacement, r.enabled, r.whole_word,
                         r.regex, r.case_sensitive) for r in rules]
        result = substitutions.preview(compiled, sample)
        changed = result != sample
        self.subs_preview.config(
            text=f"Spoken as:  {result}" if changed else "(no rule matches)",
            foreground="#2a7" if changed else "#666",
        )

    # -- profiles -------------------------------------------------------------

    def _refresh_profiles(self) -> None:
        names = [p.name for p in self.cfg.profiles.entries]
        self.profile_combo["values"] = ["(none)", *names]
        active = self.cfg.profiles.active
        self.profile_var.set(active if active in names else "(none)")

    def _switch_profile(self) -> None:
        name = self.profile_var.get()
        if name == "(none)":
            self.cfg.profiles.active = ""
            return
        entry = next((p for p in self.cfg.profiles.entries if p.name == name), None)
        if entry is None:
            return
        changed = profiles.apply(
            self.cfg, profiles.Profile(entry.name, entry.values, entry.match_apps))
        self.cfg.profiles.active = name
        # Reload every widget: a profile can touch several tabs at once.
        self._load_from_config()
        self.profile_var.set(name)
        self._apply(collect=False)
        self.state_label.config(text=f"Profile: {name} ({len(changed)} changed)")

    def _save_profile(self) -> None:
        if not self._collect():
            return
        name = simpledialog.askstring(
            "Save profile", "Name for this profile:", parent=self,
            initialvalue=self.profile_var.get() if self.profile_var.get() != "(none)"
            else "",
        )
        if not name:
            return
        name = name.strip()
        snapshot = profiles.capture(self.cfg, name)
        existing = next((p for p in self.cfg.profiles.entries if p.name == name), None)
        if existing is not None:
            existing.values = snapshot.values
        else:
            self.cfg.profiles.entries.append(
                ProfileEntry(name=name, values=snapshot.values))
        self.cfg.profiles.active = name
        self._refresh_profiles()
        self.profile_var.set(name)
        self.cfg.save()
        self.state_label.config(text=f"Saved profile: {name}")

    # -- history tab ----------------------------------------------------------

    def _build_history(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="History")

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.review_var = tk.BooleanVar()
        ttk.Checkbutton(
            top, text="Check each transcript before speaking it",
            variable=self.review_var,
        ).pack(side="left")
        ttk.Label(top, text="  discard after").pack(side="left")
        self.review_timeout_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(top, from_=5, to=300, increment=5, width=5,
                    textvariable=self.review_timeout_var).pack(side="left", padx=4)
        ttk.Label(top, text="s", foreground="#666").pack(side="left")

        ttk.Label(
            tab,
            text="Recent utterances. Kept in memory only — never written to disk.",
            foreground="#555",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 4))

        cols = ("time", "heard", "spoken")
        self.hist_tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        for col, width, text in zip(cols, (70, 240, 240),
                                    ("Time", "Heard", "Spoken"), strict=True):
            self.hist_tree.heading(col, text=text)
            self.hist_tree.column(col, width=width, anchor="w")
        self.hist_tree.grid(row=2, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(tab, orient="vertical", command=self.hist_tree.yview)
        sb.grid(row=2, column=3, sticky="ns")
        self.hist_tree.configure(yscrollcommand=sb.set)
        self.hist_tree.bind("<Double-1>", lambda _e: self._respeak())

        actions = ttk.Frame(tab)
        actions.grid(row=3, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Button(actions, text="Say again", command=self._respeak).pack(side="left")
        ttk.Button(actions, text="Copy", command=self._copy_history).pack(
            side="left", padx=6)
        ttk.Button(actions, text="Clear", command=self._clear_history).pack(side="left")
        self.hist_status = ttk.Label(actions, text="", foreground="#666")
        self.hist_status.pack(side="left", padx=10)

        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)
        self._history_shown = 0

    def _refresh_history(self) -> None:
        entries = list(self.pipeline.history)
        if len(entries) == self._history_shown:
            return  # nothing new; do not fight the user's selection every 100 ms
        self._history_shown = len(entries)
        self.hist_tree.delete(*self.hist_tree.get_children())
        for i, entry in enumerate(reversed(entries)):
            heard = entry.heard if entry.source == "speech" else f"({entry.source})"
            self.hist_tree.insert("", "end", iid=str(i),
                                  values=(entry.clock, heard, entry.spoken))

    def _selected_history(self):
        sel = self.hist_tree.selection()
        if not sel:
            self.hist_status.config(text="Select something first.")
            return None
        entries = list(reversed(self.pipeline.history))
        index = int(sel[0])
        return entries[index] if index < len(entries) else None

    def _respeak(self) -> None:
        entry = self._selected_history()
        if entry is None:
            return
        if not self.pipeline.running:
            self.hist_status.config(text="Start the pipeline first.")
            return
        self.pipeline.say_text(entry.spoken, source="repeat")
        self.hist_status.config(text=f"Saying: {entry.spoken[:40]}")

    def _copy_history(self) -> None:
        entry = self._selected_history()
        if entry is None:
            return
        self.clipboard_clear()
        self.clipboard_append(entry.spoken)
        self.update_idletasks()
        self.hist_status.config(text="Copied")

    def _clear_history(self) -> None:
        self.pipeline.clear_history()
        self._history_shown = -1  # force a redraw even though the count is 0
        self._refresh_history()
        self.hist_status.config(text="Cleared")

    # -- studio tab -----------------------------------------------------------

    def _build_studio(self, nb: ttk.Notebook) -> None:
        # A sub-notebook: checking hardware and reading 1132 sentences are
        # different sittings, and stacking both into one pane made it unusable.
        outer = ttk.Frame(nb)
        nb.add(outer, text="Studio")
        self.studio_nb = ttk.Notebook(outer)
        self.studio_nb.pack(fill="both", expand=True)

        tab = ttk.Frame(self.studio_nb, padding=10)
        self.studio_nb.add(tab, text="Setup")

        ttk.Label(tab, text="Make your own voice", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(
            tab,
            text="Record yourself reading for a while, then train a voice from it.\n"
                 "The result installs like any other voice and works everywhere.",
            foreground=self.palette.muted, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 10))

        ttk.Label(tab, text="This machine", font=("", 9, "bold")).grid(
            row=2, column=0, columnspan=3, sticky="w")
        self.studio_hw = ttk.Label(tab, text="Checking...", justify="left",
                                   wraplength=560)
        self.studio_hw.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))

        self.studio_verdict = ttk.Label(tab, text="", justify="left", wraplength=560)
        self.studio_verdict.grid(row=4, column=0, columnspan=3, sticky="w")

        self.studio_override_var = tk.BooleanVar()
        self.studio_override = ttk.Checkbutton(
            tab,
            text="I know what I am doing — train anyway",
            variable=self.studio_override_var,
            command=self._toggle_studio_override,
        )
        self.studio_override.grid(row=5, column=0, columnspan=3, sticky="w",
                                  pady=(4, 0))
        ttk.Label(
            tab,
            text="Under-spec hardware runs out of memory rather than breaking "
                 "anything;\nthe cost is the time since the last checkpoint.",
            foreground=self.palette.muted, justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Separator(tab, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=4)

        ttk.Label(tab, text="Training environment", font=("", 9, "bold")).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.studio_pack = ttk.Label(tab, text="", justify="left", wraplength=560)
        self.studio_pack.grid(row=9, column=0, columnspan=3, sticky="w")

        row = ttk.Frame(tab)
        row.grid(row=10, column=0, columnspan=3, sticky="w", pady=6)
        self.studio_btn = ttk.Button(row, text="Download",
                                     command=self._toggle_studio_pack)
        self.studio_btn.pack(side="left")
        ttk.Button(row, text="Re-check",
                   command=lambda: self._refresh_studio(force=True)).pack(
            side="left", padx=6)
        self.studio_progress = ttk.Progressbar(row, mode="indeterminate", length=180)

        self.studio_log = ttk.Label(tab, text="", foreground=self.palette.muted,
                                    wraplength=580, justify="left")
        self.studio_log.grid(row=11, column=0, columnspan=3, sticky="w")

        tab.columnconfigure(2, weight=1)
        self._refresh_studio()

        from .studioui import DesignPanel, RecordingPanel, TrainingPanel

        self.record_panel = RecordingPanel(
            self.studio_nb, self.palette,
            all_apis=not self.cfg.audio.prefer_wasapi)
        self.studio_nb.add(self.record_panel, text="Record")

        self.train_panel = TrainingPanel(self.studio_nb, self.palette)
        self.studio_nb.add(self.train_panel, text="Train")

        self.design_panel = DesignPanel(self.studio_nb, self.palette)
        self.studio_nb.add(self.design_panel, text="Design")

        # The Design tab parses the whole base model to read its speaker table.
        # That is 453 ms for en_US-libritts-high, and it was happening on every
        # settings window open whether or not anyone opened the tab.
        self.studio_nb.bind("<<NotebookTabChanged>>", self._studio_tab_changed)

    def _studio_tab_changed(self, _event=None) -> None:
        chosen = self.studio_nb.select()
        if chosen and self.studio_nb.tab(chosen, "text") == "Design":
            self.design_panel.activate()

    def _refresh_studio(self, force: bool = False) -> None:
        # Cached unless asked: the hardware does not change while the app is
        # open, and this runs several times as the tab is built.
        hw = studiopack.probe(force=force)
        gpu = f"{hw.gpu_name} ({hw.vram_gb:.0f} GB)" if hw.has_gpu else "no NVIDIA GPU"
        self.studio_hw.config(
            text=f"Graphics: {gpu}          Free disk: {hw.free_disk_gb:.0f} GB")

        result = studiopack.gate(
            override=self.cfg.studio.ignore_hardware_check, hardware=hw)
        if result.blockers:
            self.studio_verdict.config(
                text="\n".join(result.blockers),
                foreground=self.palette.warn if result.ok else self.palette.error)
            self.studio_override.state(["!disabled"])
        else:
            self.studio_verdict.config(
                text="This machine can train a voice."
                     + ("\n" + "\n".join(result.warnings) if result.warnings else ""),
                foreground=self.palette.ok)
            # Nothing to override, so the checkbox would only invite confusion.
            self.studio_override.state(["disabled"])

        state = studiopack.status()
        if not state.installed:
            self.studio_pack.config(
                text=f"Not installed. About {studiopack.APPROX_DOWNLOAD_GB:.0f} GB, "
                     "downloaded only when you ask for it.",
                foreground=self.palette.muted)
            self.studio_btn.config(text="Download")
        else:
            detail = f"Installed — {state.size_gb:.1f} GB"
            if not state.usable:
                detail += "  (incomplete; remove and download again)"
            self.studio_pack.config(text=detail, foreground=self.palette.ok
                                    if state.usable else self.palette.error)
            self.studio_btn.config(text="Remove")

        # Training is possible only when the gate allows it and the pack is there.
        allowed = result.ok
        self.studio_btn.state(["!disabled"] if allowed or state.installed
                              else ["disabled"])

    def _toggle_studio_override(self) -> None:
        self.cfg.studio.ignore_hardware_check = self.studio_override_var.get()
        self._refresh_studio()

    def _toggle_studio_pack(self) -> None:
        if studiopack.status().installed:
            if not messagebox.askyesno(
                "Remove training environment",
                "Delete the training environment? Voices you have already trained "
                "are kept.",
                parent=self,
            ):
                return
            studiopack.uninstall()
            self.studio_log.config(text="Removed.")
            self._refresh_studio()
            return

        gb = studiopack.APPROX_DOWNLOAD_GB
        if not messagebox.askokcancel(
            "Download training environment",
            f"This downloads about {gb:.0f} GB (PyTorch and the Piper trainer) into\n"
            f"{studiopack.studio_dir()}\n\n"
            "It runs separately from Voice2TTS and can be removed at any time.\n\n"
            "Continue?",
            parent=self,
        ):
            return

        self.studio_btn.state(["disabled"])
        self.studio_progress.pack(side="left", padx=8)
        self.studio_progress.start(12)

        def report(msg: str) -> None:
            self._later(lambda: self.studio_log.config(text=msg))

        def work() -> None:
            try:
                studiopack.install(progress=report)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
                self._later(lambda: self._studio_done(failure))
                return
            self._later(lambda: self._studio_done(None))

        threading.Thread(target=work, daemon=True).start()

    def _studio_done(self, error: str | None) -> None:
        if not self.winfo_exists():
            return
        self.studio_progress.stop()
        self.studio_progress.pack_forget()
        self.studio_btn.state(["!disabled"])
        if error:
            self.studio_log.config(text="", foreground=self.palette.muted)
            messagebox.showerror("Download failed", error, parent=self)
            self._refresh_studio()
            return
        state = studiopack.status(deep=True)
        cuda = ("CUDA available" if state.cuda_available
                else "CUDA NOT available — training will be very slow")
        self.studio_log.config(
            text=f"Ready. torch {state.torch_version}, {cuda}.",
            foreground=self.palette.ok if state.cuda_available else self.palette.warn)
        self._refresh_studio()

    # -- recognition tab ------------------------------------------------------

    def _build_recognition(self, parent: ttk.Frame) -> None:
        tab = ttk.LabelFrame(parent, text="Recognition", padding=10)
        tab.pack(fill="both", expand=True)

        ttk.Label(tab, text="When to speak", font=("", 9, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.stt_mode_var = tk.StringVar(value="sentence")
        ttk.Radiobutton(
            tab, text="Wait for me to finish a sentence", value="sentence",
            variable=self.stt_mode_var, command=self._refresh_stt_mode,
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            tab, text="Speak while I am still talking", value="streaming",
            variable=self.stt_mode_var, command=self._refresh_stt_mode,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self.stt_mode_note = ttk.Label(tab, text="", foreground="#555",
                                       justify="left", wraplength=560)
        self.stt_mode_note.grid(row=3, column=0, columnspan=2, sticky="w",
                                pady=(2, 10))

        ttk.Separator(tab, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(tab, text="Whisper model").grid(row=5, column=0, sticky="w")
        self.model_var = tk.StringVar()
        ttk.Combobox(
            tab, textvariable=self.model_var, values=list(WHISPER_MODELS), width=24
        ).grid(row=5, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(tab, text="Spoken language").grid(row=6, column=0, sticky="w")
        self.stt_lang_var = tk.StringVar()
        ttk.Combobox(
            tab, textvariable=self.stt_lang_var,
            values=["auto", *sorted(LANGUAGE_NAMES)], width=24,
        ).grid(row=6, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(
            tab,
            text="Only the models without \".en\" can hear anything but English. "
                 "\"auto\" detects\nper utterance, which costs a little accuracy.",
            foreground="#555", justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # The voice warning depends on the model and the spoken language as well
        # as the voice, so all three have to re-evaluate it. Watching only the
        # voice left a stale warning on screen after changing the language.
        for watched in (self.model_var, self.stt_lang_var):
            watched.trace_add("write", lambda *_: self._check_language())

        ttk.Label(tab, text="Compute device").grid(row=8, column=0, sticky="w")
        self.stt_device_var = tk.StringVar()
        ttk.Combobox(
            tab, textvariable=self.stt_device_var, values=["auto", "cuda", "cpu"],
            width=24, state="readonly",
        ).grid(row=8, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(tab, text="Beam size").grid(row=9, column=0, sticky="w")
        self.beam_var = tk.IntVar()
        ttk.Spinbox(tab, from_=1, to=5, textvariable=self.beam_var, width=6).grid(
            row=9, column=1, sticky="w", padx=6, pady=2
        )

        ttk.Label(
            tab,
            text="Model and device changes take effect after restarting the pipeline\n"
                 "(tray menu → Stop, then Start).",
            foreground="#555",
            justify="left",
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Separator(tab, orient="horizontal").grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=12
        )
        # The pack itself lives on the Add-ons tab, with everything else that is
        # downloaded on demand. This says whether it is here and where to get it.
        self.gpu_label = ttk.Label(tab, text="", justify="left",
                                   foreground=self.palette.muted, wraplength=560)
        self.gpu_label.grid(row=12, column=0, columnspan=2, sticky="w")

        tab.columnconfigure(1, weight=1)
        self._refresh_stt_mode()
        self._refresh_gpu()

    def _refresh_stt_mode(self) -> None:
        """Say what each mode costs, in the terms the choice actually turns on.

        Both numbers here are measured (spike/09_streaming.py), not estimated,
        and the CPU/GPU line is worth stating because the obvious assumption --
        that streaming is a GPU feature -- is wrong.
        """
        if self.stt_mode_var.get() == "streaming":
            text = ("Recognises as you talk and speaks each phrase once it "
                    "settles, so a long sentence does not sit silent until you "
                    "stop.\nCosts about half a processor core for as long as "
                    "anyone is speaking — the same on CPU and GPU — "
                    "and delivers speech a phrase at a time rather than a "
                    "sentence at a time. Needs automatic detection; with "
                    "push-to-talk it behaves as below.")
            # It has to keep listening while it speaks: pausing the microphone
            # either cuts the recording mid-sentence or ends the phrase early,
            # and both were measured to mangle the words. So say what that
            # means rather than letting someone find out during a call.
            if self.mute_var.get():
                text += ("\nIt keeps listening while it speaks, so “Mute "
                         "microphone while speaking” does not apply here "
                         "— use headphones, or the virtual cable alone, "
                         "or it will hear itself.")
        else:
            text = ("Waits for a pause, then speaks the whole utterance. Best "
                    "wording and natural intonation, and it costs nothing "
                    "while you are quiet.\nThe delay is however long you keep "
                    "talking.")
        self.stt_mode_note.config(text=text)

    def _refresh_gpu(self) -> None:
        """Say where recognition runs, and point at Add-ons if it could be faster."""
        pack = gpupack.status()
        if pack.usable:
            self.gpu_label.config(
                text=f"GPU acceleration is installed ({pack.size_mb:.0f} MB). "
                     "Manage it on the Add-ons tab.")
        elif gpupack.gpu_present():
            self.gpu_label.config(
                text="An NVIDIA card was found, but the CUDA libraries are not "
                     "installed, so recognition is running on the processor. "
                     "The Add-ons tab can download them.")
        else:
            self.gpu_label.config(
                text="No NVIDIA card detected; recognition runs on the "
                     "processor.")

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

        self.beta_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab,
            text="Include beta versions",
            variable=self.beta_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(
            tab,
            text="Betas arrive earlier and break more. You can go back by "
                 "installing the latest normal release over the top — turning "
                 "this off does not undo one.",
            foreground="#666", justify="left", wraplength=560,
        ).grid(row=7, column=0, columnspan=3, sticky="w")

        ttk.Label(
            tab,
            text="Checking contacts api.github.com. Nothing else is sent.",
            foreground="#666",
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Separator(tab, orient="horizontal").grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=12
        )

        row = ttk.Frame(tab)
        row.grid(row=10, column=0, columnspan=3, sticky="w")
        self.update_btn = ttk.Button(row, text="Check now", command=self._check_updates)
        self.update_btn.pack(side="left")
        self.update_status = ttk.Label(row, text="", foreground="#666")
        self.update_status.pack(side="left", padx=10)

        self.update_notes = tk.Text(tab, height=8, wrap="word")
        self.update_notes.grid(row=11, column=0, columnspan=3, sticky="nsew",
                               pady=(10, 0))
        self.update_notes.configure(state="disabled")

        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(11, weight=1)

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
        # Take the beta opt-in from the checkbox, not from the saved config.
        # The repository was already read from its widget, so reading this one
        # from the config meant ticking "include pre-releases" and pressing
        # Check did nothing until you also pressed Apply -- which looks exactly
        # like beta checking being broken.
        self.cfg.updates.include_prereleases = bool(self.beta_var.get())
        self.cfg.validate()
        self.update_btn.config(state="disabled")
        self.update_status.config(text="Checking...")

        def work() -> None:
            try:
                release = updater.check(
                    self.cfg.updates.repo,
                    include_prereleases=self.cfg.updates.include_prereleases)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)  # `exc` is deleted when this block ends
                self._later(lambda: self._check_done(None, failure))
                return
            self._later(lambda: self._check_done(release, None))

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
        ttk.Button(actions, text="Show log", command=self._show_log).pack(
            side="left", padx=6)
        ttk.Button(actions, text="Open log folder",
                   command=self._open_log_folder).pack(side="left")
        self.diag_status = ttk.Label(actions, text="", foreground=self.palette.muted)
        self.diag_status.pack(side="left", padx=8)

        ttk.Label(actions, text="  Theme").pack(side="left")
        self.theme_var = tk.StringVar()
        theme_combo = ttk.Combobox(actions, textvariable=self.theme_var, width=8,
                                   state="readonly", values=list(theme.MODES))
        theme_combo.pack(side="left", padx=4)
        theme_combo.bind("<<ComboboxSelected>>", lambda _e: self._change_theme())

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

    def _show_log(self) -> None:
        """Show the log tail in-app, rather than sending people into AppData."""
        win = tk.Toplevel(self)
        win.title(f"Log — {log_path()}")
        win.geometry("900x520")

        text = tk.Text(win, wrap="none")
        text.pack(fill="both", expand=True, side="left", padx=(8, 0), pady=8)
        theme.style_text_widget(text, self.palette)
        sb = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        sb.pack(fill="y", side="right", padx=(0, 8), pady=8)
        text.configure(yscrollcommand=sb.set)

        # Colour by level so an error is findable without reading every line.
        for level, colour in (("ERROR", self.palette.error),
                              ("CRITICAL", self.palette.error),
                              ("WARNING", self.palette.warn)):
            text.tag_configure(level, foreground=colour)

        try:
            lines = log_path().read_text(encoding="utf-8",
                                         errors="replace").splitlines()[-2000:]
        except OSError as exc:
            lines = [f"Could not read the log: {exc}"]

        for line in lines:
            start = text.index("end-1c")
            text.insert("end", line + "\n")
            for level in ("CRITICAL", "ERROR", "WARNING"):
                if f" {level} " in line:
                    text.tag_add(level, start, f"{start} lineend")
                    break
        text.see("end")
        text.configure(state="disabled")

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")
        ttk.Label(bar, text=f"{len(lines)} lines (most recent last)",
                  foreground=self.palette.muted).pack(side="left")

    def _change_theme(self) -> None:
        self.cfg.theme = Theme.parse(self.theme_var.get()) or Theme.NATIVE
        self.palette = theme.apply(self.winfo_toplevel(), self.cfg.theme)
        for widget in (self.logbox, self.transcript, self.update_notes):
            theme.style_text_widget(widget, self.palette)
        self.diag_status.config(
            text="Theme applied. Reopen Settings for a full refresh.")

    def _later(self, callback) -> None:
        """Run something on the interface thread, if there is still one.

        Background workers hand their results back with `after`. Closing the
        window while one is in flight destroys the interpreter underneath it,
        and `after` itself raises "main thread is not in main loop" from the
        worker -- before any winfo_exists() guard inside the callback gets a
        chance to run.
        """
        try:
            # winfo_exists() is itself a call into Tk, so it raises the same way
            # `after` does once the interpreter is gone. The guard has to be
            # inside the try, not in front of it.
            if self._closing or not self.winfo_exists():
                return
            self.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass  # the window went away while this worker was running

    def append_log(self, kind: str, message: str) -> None:
        if self._closing or not self.winfo_exists():
            return
        if kind == "partial":
            # Streaming's unsettled text: it arrives several times a second and
            # changes as Whisper makes up its mind, so it belongs in the
            # transcript pane as a preview, not as log lines nobody can read.
            self._show_partial(message)
            return
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"[{kind}] {message}\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")
        if kind == "transcript":
            self._settled_transcript = message
            self._show_transcript(message)

    def _show_partial(self, message: str) -> None:
        """Preview what is still being heard, after what has been settled."""
        settled = getattr(self, "_settled_transcript", "")
        self._show_transcript(f"{settled} {message}".strip() if settled
                              else message)

    def _show_transcript(self, message: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("1.0", message)
        self.transcript.configure(state="disabled")

    # -- config binding -------------------------------------------------------

    def _load_from_config(self) -> None:
        c = self.cfg
        self.all_apis_var.set(not c.audio.prefer_wasapi)
        self.input_combo["values"] = ["", *self._device_names("in")]
        self.input_var.set(c.audio.input_match)
        self.mute_var.set(c.audio.mute_mic_during_playback)
        self._update_device_count()
        # Cleared first. This runs again whenever a profile is switched, and
        # appending to what was already there turned two outputs into four.
        self._clear_output_rows()
        for target in c.audio.outputs:
            self._add_output_row(target)
        if not self._output_rows:
            self._add_output_row()
        self.output_count.set(len(self._output_rows))
        self._update_cable_hint()

        self.mode_var.set(c.trigger.mode.value)
        self.hotkey_var.set(c.trigger.hotkey)
        self.clip_hotkey_var.set(c.trigger.clipboard_hotkey)
        self.stop_hotkey_var.set(c.trigger.stop_hotkey)
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
        self.stt_device_var.set(c.stt.device.value)
        self.beam_var.set(c.stt.beam_size)

        self.review_var.set(c.text.review_before_speaking)
        self.review_timeout_var.set(c.text.review_timeout_s)
        self.subs_enabled_var.set(c.text.substitutions_enabled)
        def copied(entries):
            return [
                SubstitutionRule(r.pattern, r.replacement, r.enabled, r.whole_word,
                                 r.regex, r.case_sensitive)
                for r in entries
            ]

        # Copies, so Cancel really does discard the edits.
        self._subs = copied(c.text.substitutions)
        self._target_subs = copied(c.text.target_substitutions)
        self._render_subs()

        self.stt_lang_var.set(c.stt.language)
        self.stt_mode_var.set(c.stt.mode.value)
        self._refresh_stt_mode()
        self._show_trans_mode(c.translation.mode)
        self.trans_source.set(self._language_label(c.translation.source))
        self.trans_target.set(self._language_label(c.translation.target))
        self._refresh_translate_list()

        self.studio_override_var.set(c.studio.ignore_hardware_check)
        self._refresh_studio()

        self._refresh_profiles()
        self.theme_var.set(c.theme)
        self.repo_var.set(c.updates.repo)
        self.check_start_var.set(c.updates.check_on_start)
        self.interval_var.set(c.updates.interval_hours)
        self.beta_var.set(c.updates.include_prereleases)

        # Both halves of the pairing are loaded now, so the warning can be evaluated.
        self._check_language()

    def _collect(self) -> bool:
        if not self._validate_hotkey():
            messagebox.showerror("Invalid hotkey", "Fix the hotkey first.", parent=self)
            return False
        c = self.cfg
        c.audio.input_match = self._strip_tag(self.input_var.get())
        c.audio.mute_mic_during_playback = self.mute_var.get()
        c.audio.prefer_wasapi = not self.all_apis_var.get()
        c.audio.outputs = [
            OutputTarget(
                match=self._strip_tag(r["match"].get()),
                gain=round(float(r["gain"].get()), 3),
                enabled=bool(r["enabled"].get()),
            )
            for r in self._output_rows
        ]

        c.trigger.mode = TriggerMode.parse(self.mode_var.get()) or TriggerMode.PTT
        c.trigger.hotkey = self.hotkey_var.get().strip()
        c.trigger.clipboard_hotkey = self.clip_hotkey_var.get().strip()
        c.trigger.stop_hotkey = self.stop_hotkey_var.get().strip()
        c.trigger.ptt_latch = self.latch_var.get()
        c.trigger.preroll_ms = int(self.preroll.get())

        c.vad.threshold = round(float(self.vad_threshold.get()), 3)
        c.vad.min_silence_ms = int(self.vad_silence.get())
        c.vad.min_speech_ms = int(self.vad_min_speech.get())

        c.tts.voice = self.voice_var.get().strip()
        c.tts.length_scale = round(float(self.speed_var.get()), 3)
        c.tts.volume = round(float(self.volume_var.get()), 3)

        c.stt.model = self.model_var.get().strip()
        c.stt.device = SttDevice.parse(self.stt_device_var.get()) or SttDevice.AUTO
        c.stt.beam_size = int(self.beam_var.get())

        c.text.substitutions_enabled = self.subs_enabled_var.get()
        c.text.substitutions = list(self._subs)
        c.text.target_substitutions = list(self._target_subs)
        c.text.review_before_speaking = self.review_var.get()
        c.text.review_timeout_s = max(5.0, float(self.review_timeout_var.get()))

        c.stt.language = self.stt_lang_var.get().strip() or "en"
        c.stt.mode = (RecognitionMode.parse(self.stt_mode_var.get())
                      or RecognitionMode.SENTENCE)
        # One field, so "translate twice" is not a state this can produce. It
        # used to write two, guarded only here, and a config edited by hand or
        # written by a profile got Whisper translating to English and then a
        # model chain translating that again.
        c.translation.mode = self._trans_mode()
        c.translation.source = self._source_code()
        c.translation.target = self._target_code()

        c.updates.repo = self.repo_var.get().strip()
        c.updates.check_on_start = self.check_start_var.get()
        c.updates.interval_hours = int(self.interval_var.get())
        c.updates.include_prereleases = bool(self.beta_var.get())
        self._repairs = c.validate()
        self.repo_var.set(c.updates.repo)  # reflect any normalisation back to the UI
        return True

    def _apply(self, collect: bool = True) -> None:
        # collect=False when a profile has just written the config directly; the
        # widgets have been reloaded from it and re-collecting would be a no-op.
        if collect and not self._collect():
            return
        self.pipeline.apply_audio_changes()
        self.pipeline.apply_tts_changes()
        self.pipeline.apply_vad_changes()
        active = self.pipeline.apply_text_changes()
        dropped = self.pipeline.dropped_rules
        self.subs_status.config(
            text=f"{active} rule(s) active"
                 + (f"; {len(dropped)} unusable: {dropped[0]}" if dropped else ""))
        self.pipeline.apply_translation_changes()
        # validate() can refuse what was asked for -- a no-op language pair, an
        # English-only model with nothing to translate from. Point the widgets
        # at what it actually settled on and say what it changed, rather than
        # unticking a box without a word and looking broken.
        self._show_trans_mode(self.cfg.translation.mode)
        self._show_repairs()
        self._refresh_translate_route()
        # set_mode rebinds every hotkey from the config, so the clipboard and stop
        # combos are picked up along with the push-to-talk one.
        self.pipeline.set_mode(self.cfg.trigger.mode)

    def _show_repairs(self) -> None:
        """Say what validate() had to change, in the window that changed it."""
        repairs = getattr(self, "_repairs", [])
        for repair in repairs:
            self.append_log("warning", str(repair))
        if repairs:
            self.trans_status.config(text=str(repairs[0]))
        self._repairs = []

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

        sink = self.pipeline.sink
        levels = sink.levels() if sink is not None else {}
        for row in self._output_rows:
            match = devices.strip_display(row["match"].get())
            value = 0.0
            if match:
                value = next((v for name, v in levels.items() if match in name), 0.0)
            elif levels:
                value = next(iter(levels.values()))
            row["meter"]["value"] = min(100.0, value * 140)

        self._refresh_history()
        # Remembered so close() can cancel it. A tick left scheduled fires
        # after the window is destroyed and Tk complains about an "invalid
        # command name" for a callback nobody can see.
        self._tick_id: str | None = self.after(100, self._tick)

    def close(self) -> None:
        self._closing = True
        tick = getattr(self, "_tick_id", None)
        if tick is not None:
            try:
                self.after_cancel(tick)
            except (tk.TclError, ValueError):
                pass
            self._tick_id = None
        # This window is destroyed on close and rebuilt on reopen, so anything
        # holding a device has to let go of it here.
        for panel in ("record_panel", "train_panel"):
            widget = getattr(self, panel, None)
            if widget is not None:
                try:
                    widget.shutdown()
                except Exception as exc:  # noqa: BLE001 - closing must not fail
                    log.warning("%s did not shut down cleanly: %s", panel, exc)
        if self._on_close:
            self._on_close()
        self.destroy()
