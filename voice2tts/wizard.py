"""First-run setup wizard.

Walks the user through the two things that cannot be automated away (the virtual
cable, which needs admin and a reboot) and the one that shouldn't be automatic (a
1.3 GB GPU download), then ends with a test that produces actual sound.

That last step matters more than it looks: without it "it doesn't work" is a
mystery, and with it the user knows exactly which step failed.

All slow work runs on worker threads and marshals back through `after`, because Tk
widgets may only be touched from the thread running the mainloop.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
import webbrowser
from functools import partial
from tkinter import messagebox, ttk

from . import cable, devices, gpupack, voices
from .config import Config, OutputTarget
from .hotkey import describe
from .modes import SttDevice, TriggerMode

log = logging.getLogger(__name__)

PAD = 16


class Wizard(tk.Toplevel):
    def __init__(self, master: tk.Tk, cfg: Config, on_finish=None):
        super().__init__(master)
        self.cfg = cfg
        self._on_finish = on_finish
        self._busy = False
        self._cancelled = False

        self.title("Voice2TTS Setup")
        self.geometry("620x520")
        self.minsize(560, 480)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._steps = [
            ("Welcome", self._build_welcome),
            ("Virtual microphone", self._build_cable),
            ("Speed", self._build_gpu),
            ("Devices & voice", self._build_devices),
            ("Test", self._build_test),
        ]
        self._index = 0

        self.header = ttk.Label(self, text="", font=("", 13, "bold"))
        self.header.pack(anchor="w", padx=PAD, pady=(PAD, 0))
        self.progress_label = ttk.Label(self, text="", foreground="#666")
        self.progress_label.pack(anchor="w", padx=PAD)

        self.body = ttk.Frame(self, padding=(PAD, 12))
        self.body.pack(fill="both", expand=True)

        bar = ttk.Frame(self, padding=(PAD, 0, PAD, PAD))
        bar.pack(fill="x")
        self.skip_btn = ttk.Button(bar, text="Skip setup", command=self._on_close)
        self.skip_btn.pack(side="left")
        self.next_btn = ttk.Button(bar, text="Next", command=self._next)
        self.next_btn.pack(side="right")
        self.back_btn = ttk.Button(bar, text="Back", command=self._back)
        self.back_btn.pack(side="right", padx=6)

        self._render()

    # -- frame plumbing -------------------------------------------------------

    def _render(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        title, builder = self._steps[self._index]
        self.header.config(text=title)
        self.progress_label.config(text=f"Step {self._index + 1} of {len(self._steps)}")
        self.back_btn.config(state="normal" if self._index else "disabled")
        self.next_btn.config(
            text="Finish" if self._index == len(self._steps) - 1 else "Next"
        )
        builder(self.body)

    def _next(self) -> None:
        if self._busy:
            return
        if self._index == len(self._steps) - 1:
            self._finish()
            return
        self._index += 1
        self._render()

    def _back(self) -> None:
        if self._busy or self._index == 0:
            return
        self._index -= 1
        self._render()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.next_btn.config(state=state)
        self.back_btn.config(state="disabled" if busy or not self._index else "normal")

    def _run_async(self, work, done) -> None:
        """Run `work()` off-thread; call `done(result, error)` back on the Tk thread."""
        self._set_busy(True)

        def runner() -> None:
            result, error = None, None
            try:
                result = work()
            except Exception as exc:
                log.exception("wizard step failed")
                error = exc
            self.after(0, lambda: self._finish_async(result, error, done))

        threading.Thread(target=runner, daemon=True).start()

    def _finish_async(self, result, error, done) -> None:
        self._set_busy(False)
        if self.winfo_exists():
            done(result, error)

    # -- step 1: welcome ------------------------------------------------------

    def _build_welcome(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="This sets up Voice2TTS in four short steps.",
            font=("", 10),
        ).pack(anchor="w")
        ttk.Label(
            parent,
            text=(
                "\nVoice2TTS listens to your microphone, transcribes what you say, and "
                "speaks it back\nin a synthetic voice through a virtual microphone that "
                "Discord and other apps can use.\n\n"
                "Everything runs on your own machine. No audio is sent anywhere.\n\n"
                "You can skip this and configure things manually in Settings at any time."
            ),
            justify="left",
            foreground="#444",
        ).pack(anchor="w")

    # -- step 2: virtual cable ------------------------------------------------

    def _build_cable(self, parent: ttk.Frame) -> None:
        self.cable_status = ttk.Label(parent, text="Checking...", font=("", 10, "bold"))
        self.cable_status.pack(anchor="w")
        self.cable_detail = ttk.Label(parent, text="", justify="left", foreground="#444")
        self.cable_detail.pack(anchor="w", pady=(4, 10))

        self.cable_actions = ttk.Frame(parent)
        self.cable_actions.pack(anchor="w", fill="x")

        self.cable_log = ttk.Label(parent, text="", foreground="#666", justify="left")
        self.cable_log.pack(anchor="w", pady=(10, 0))

        self._refresh_cable()

    def _refresh_cable(self) -> None:
        for child in self.cable_actions.winfo_children():
            child.destroy()

        candidates = cable.list_devices()
        if candidates:
            self._cable_choices = candidates
            chosen = self._cable_choice(candidates)
            colour = "#2a7" if not (chosen.is_router and not chosen.app_running) else "#c80"
            self.cable_status.config(text=f"Found: {chosen.product}", foreground=colour)
            self._describe_cable(chosen)
            self._apply_cable_output(chosen.output_name)

            # Matrix exposes eight channels and VoiceMeeter several VAIOs; without a
            # picker the user is stuck with whichever we guessed.
            if len(candidates) > 1:
                ttk.Label(self.cable_actions, text="Channel").pack(side="left",
                                                                  padx=(0, 4))
                self.cable_var = tk.StringVar(value=chosen.output_name)
                combo = ttk.Combobox(
                    self.cable_actions, textvariable=self.cable_var, width=38,
                    state="readonly",
                    values=[c.output_name for c in candidates],
                )
                combo.pack(side="left", padx=(0, 6))
                combo.bind("<<ComboboxSelected>>", lambda _e: self._on_cable_selected())
            ttk.Button(self.cable_actions, text="Re-check",
                       command=self._refresh_cable).pack(side="left")
            return

        self.cable_status.config(text="No virtual microphone found", foreground="#a33")
        self.cable_detail.config(
            text=(
                "Discord can only hear Voice2TTS through a virtual audio device.\n"
                "Windows requires a signed driver for this, so we use VB-CABLE by "
                "VB-Audio.\n\n"
                "Installing it needs your approval at a Windows prompt, and a restart."
            )
        )
        ttk.Button(self.cable_actions, text="Download and install VB-CABLE",
                   command=self._install_cable).pack(side="left")
        ttk.Button(self.cable_actions, text="Open download page",
                   command=lambda: webbrowser.open(cable.CABLE_PAGE)).pack(side="left", padx=6)
        ttk.Button(self.cable_actions, text="Re-check",
                   command=self._refresh_cable).pack(side="left")
        ttk.Label(
            self.cable_actions,
            text="VB-CABLE is donationware by VB-Audio.",
            foreground="#666",
        ).pack(side="left", padx=(12, 0))

    def _install_cable(self) -> None:
        if not messagebox.askokcancel(
            "Install VB-CABLE",
            "This will download VB-CABLE from vb-audio.com and run VB-Audio's "
            "installer.\n\nWindows will ask for administrator permission, and a "
            "restart is needed afterwards.\n\nContinue?",
            parent=self,
        ):
            return

        def progress(msg: str) -> None:
            self.after(0, lambda: self.cable_log.config(text=msg))

        def done(result, error) -> None:
            if error is not None:
                self.cable_log.config(text="")
                messagebox.showerror(
                    "Install failed",
                    f"{error}\n\nYou can install it manually from {cable.CABLE_PAGE}",
                    parent=self,
                )
                webbrowser.open(cable.CABLE_PAGE)
                return
            needs_reboot, message = result
            self.cable_log.config(text=message)
            self._refresh_cable()
            if needs_reboot and messagebox.askyesno(
                "Restart needed",
                f"{message}\n\nRestart now? Voice2TTS will be ready when you come back.",
                parent=self,
            ):
                cable.request_reboot()

        self._run_async(lambda: cable.install_flow(progress=progress), done)

    def _cable_choice(self, candidates: list):
        """Keep the user's pick across a re-render; otherwise take the best match."""
        if not candidates:
            return None
        wanted = getattr(self, "cable_var", None)
        if wanted is not None:
            match = next((c for c in candidates if c.output_name == wanted.get()), None)
            if match is not None:
                return match
        return candidates[0]

    def _describe_cable(self, chosen) -> None:
        detail = (
            f"Voice2TTS will play into  {chosen.output_name}\n"
            f"In Discord, choose        {chosen.discord_input}  as your input device."
        )
        if chosen.caveat:
            detail += f"\n\n{chosen.caveat}"
        self.cable_detail.config(text=detail, wraplength=560, justify="left")

    def _on_cable_selected(self) -> None:
        chosen = self._cable_choice(self._cable_choices)
        self._describe_cable(chosen)
        self._apply_cable_output(chosen.output_name)

    def _apply_cable_output(self, name: str) -> None:
        """Point the cable row at `name`, adding a row only if there isn't one.

        Matched with cable.is_virtual_device rather than a "cable" substring: a
        VB-Audio Matrix channel is called "VBMatrix In 1", so a substring test would
        miss it and append a fresh row every time the channel was changed.
        """
        for target in self.cfg.audio.outputs:
            if target.match == name or cable.is_virtual_device(target.match):
                target.match, target.enabled = name, True
                return
        self.cfg.audio.outputs.insert(0, OutputTarget(match=name, gain=1.0, enabled=True))

    # -- step 3: GPU ----------------------------------------------------------

    def _build_gpu(self, parent: ttk.Frame) -> None:
        pack = gpupack.status()
        has_gpu = gpupack.gpu_present()

        self.gpu_status = ttk.Label(parent, text="", font=("", 10, "bold"))
        self.gpu_status.pack(anchor="w")
        self.gpu_detail = ttk.Label(parent, text="", justify="left", foreground="#444")
        self.gpu_detail.pack(anchor="w", pady=(4, 10))
        self.gpu_actions = ttk.Frame(parent)
        self.gpu_actions.pack(anchor="w", fill="x")
        self.gpu_log = ttk.Label(parent, text="", foreground="#666", justify="left")
        self.gpu_log.pack(anchor="w", pady=(10, 0))
        self.gpu_bar = ttk.Progressbar(parent, mode="indeterminate", length=320)

        if pack.usable:
            self.gpu_status.config(text="GPU acceleration is installed", foreground="#2a7")
            self.gpu_detail.config(
                text=f"{pack.dll_count} libraries, {pack.size_mb:.0f} MB.\n"
                     "Transcription runs on your graphics card."
            )
            ttk.Button(self.gpu_actions, text="Remove GPU pack",
                       command=self._remove_gpu).pack(side="left")
        elif has_gpu:
            self.gpu_status.config(text="NVIDIA GPU detected", foreground="#c80")
            self.gpu_detail.config(
                text=(
                    "Voice2TTS works right now on your CPU, which is usually fast "
                    "enough.\n\n"
                    "Adding GPU acceleration makes transcription roughly 20x faster and "
                    "lets it\nuse a more accurate model. It downloads about 1.3 GB of "
                    "NVIDIA libraries.\n\n"
                    "You can do this later from Settings if you'd rather not wait."
                )
            )
            ttk.Button(self.gpu_actions, text="Download GPU acceleration (1.3 GB)",
                       command=self._install_gpu).pack(side="left")
        else:
            self.gpu_status.config(text="Running on CPU")
            self.gpu_detail.config(
                text=(
                    "No NVIDIA GPU was found, so Voice2TTS will use your processor.\n"
                    "That works fine -- expect roughly a second of extra delay per "
                    "sentence."
                )
            )

    def _install_gpu(self) -> None:
        def progress(msg: str) -> None:
            self.after(0, lambda: self.gpu_log.config(text=msg))

        def done(result, error) -> None:
            self.gpu_bar.stop()
            self.gpu_bar.pack_forget()
            if error is not None:
                messagebox.showerror("Download failed", str(error), parent=self)
                return
            # A GPU machine can afford the larger model.
            self.cfg.stt.model = gpupack.GPU_WHISPER_MODEL
            self.cfg.stt.device = SttDevice.AUTO
            self._render()

        self.gpu_bar.pack(anchor="w", pady=(8, 0))
        self.gpu_bar.start(12)
        self._run_async(lambda: gpupack.install(progress=progress), done)

    def _remove_gpu(self) -> None:
        if not messagebox.askyesno(
            "Remove GPU pack",
            "Delete the downloaded CUDA libraries? Transcription will fall back to "
            "your CPU.",
            parent=self,
        ):
            return
        gpupack.uninstall()
        self.cfg.stt.model = "base.en"
        self._render()

    # -- step 4: devices and voice -------------------------------------------

    def _build_devices(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Microphone").grid(row=0, column=0, sticky="w", pady=3)
        self.w_input = tk.StringVar(value=self.cfg.audio.input_match)
        combo = ttk.Combobox(parent, textvariable=self.w_input, width=44)
        combo["values"] = ["", *devices.annotate(devices.list_inputs())]
        combo.grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(parent, text="Voice").grid(row=1, column=0, sticky="w", pady=3)
        self.w_voice = tk.StringVar(value=self.cfg.tts.voice)
        vcombo = ttk.Combobox(parent, textvariable=self.w_voice, width=44,
                              state="readonly")
        vcombo["values"] = voices.installed_keys()
        vcombo.grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(parent, text="Hear yourself on").grid(row=2, column=0, sticky="w", pady=3)
        self.w_monitor = tk.StringVar(value="")
        mcombo = ttk.Combobox(parent, textvariable=self.w_monitor, width=44)
        mcombo["values"] = ["(none)", *devices.annotate(devices.list_outputs())]
        existing = next(
            (t.match for t in self.cfg.audio.outputs
             if t.enabled and "cable" not in t.match.lower()),
            None,
        )
        self.w_monitor.set(existing or "(none)")
        mcombo.grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Label(
            parent,
            text="Use headphones. Through speakers, the app hears itself and loops.",
            foreground="#666",
        ).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(parent, text="Push-to-talk key").grid(row=4, column=0, sticky="w", pady=(12, 3))
        self.w_hotkey = tk.StringVar(value=self.cfg.trigger.hotkey)
        ttk.Entry(parent, textvariable=self.w_hotkey, width=24).grid(
            row=4, column=1, sticky="w", padx=6, pady=(12, 3)
        )
        self.hotkey_err = ttk.Label(parent, text="", foreground="#a33")
        self.hotkey_err.grid(row=5, column=1, sticky="w", padx=6)
        self.w_hotkey.trace_add("write", lambda *_: self.hotkey_err.config(
            text=describe(self.w_hotkey.get())
        ))

        ttk.Label(parent, text="Mode").grid(row=6, column=0, sticky="w", pady=(12, 3))
        self.w_mode = tk.StringVar(value=self.cfg.trigger.mode)
        modes = ttk.Frame(parent)
        modes.grid(row=6, column=1, sticky="w", padx=6, pady=(12, 3))
        for label, value in (("Hold the key", "ptt"),
                             ("Detect speech automatically", "vad"),
                             ("Both", "both")):
            ttk.Radiobutton(modes, text=label, value=value,
                            variable=self.w_mode).pack(anchor="w")
        parent.columnconfigure(1, weight=1)

    def _collect_devices(self) -> None:
        if not hasattr(self, "w_input"):
            return
        self.cfg.audio.input_match = devices.strip_display(self.w_input.get())
        self.cfg.tts.voice = self.w_voice.get().strip() or self.cfg.tts.voice
        self.cfg.trigger.mode = (TriggerMode.parse(self.w_mode.get())
                                 or TriggerMode.PTT)
        hotkey = self.w_hotkey.get().strip()
        if not describe(hotkey):
            self.cfg.trigger.hotkey = hotkey

        monitor = devices.strip_display(self.w_monitor.get())
        monitor = "" if monitor == "(none)" else monitor
        # Rewrite the non-cable output row to match the chosen monitor.
        others = [t for t in self.cfg.audio.outputs if "cable" not in t.match.lower()]
        if monitor:
            if others:
                others[0].match, others[0].enabled = monitor, True
            else:
                self.cfg.audio.outputs.append(
                    OutputTarget(match=monitor, gain=0.7, enabled=True)
                )
        else:
            for t in others:
                t.enabled = False
        self.cfg.ensure_usable_output()

    # -- step 5: test ---------------------------------------------------------

    def _build_test(self, parent: ttk.Frame) -> None:
        self._collect_devices()
        summary = [
            f"Microphone   {self.cfg.audio.input_match or '(system default)'}",
            f"Voice        {self.cfg.tts.voice}",
            f"Mode         {self.cfg.trigger.mode.value}   "
            f"Hotkey  {self.cfg.trigger.hotkey}",
            "Outputs      " + (", ".join(
                t.label for t in self.cfg.audio.outputs if t.enabled) or "none"),
        ]
        ttk.Label(parent, text="\n".join(summary), justify="left",
                  font=("Consolas", 9)).pack(anchor="w")

        ttk.Label(
            parent,
            text="\nPlay a test phrase to confirm audio reaches your outputs.",
            justify="left",
        ).pack(anchor="w", pady=(8, 6))

        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x")
        ttk.Button(row, text="Play test phrase", command=self._play_test).pack(side="left")
        ttk.Button(row, text="Check the Discord path",
                   command=self._check_path).pack(side="left", padx=6)
        self.test_log = ttk.Label(row, text="", foreground="#666", wraplength=560,
                                  justify="left")
        self.test_log.pack(side="left", padx=10)

        ttk.Label(
            parent,
            text=(
                "\nIf you routed to a virtual cable, also open Windows Sound settings "
                "and\nconfirm the matching recording device shows a moving level bar.\n\n"
                "Then in Discord: pick that device as your input, and turn OFF noise\n"
                "suppression (Krisp) -- it filters out synthesized speech."
            ),
            justify="left",
            foreground="#444",
        ).pack(anchor="w")

    def _check_path(self) -> None:
        """Prove audio reaches the device Discord will listen on."""
        from . import loopback

        info = self._cable_choice(getattr(self, "_cable_choices", []) or
                                  cable.list_devices())
        if info is None:
            self.test_log.config(text="No virtual cable configured.")
            return

        def work():
            return loopback.verify_cable(
                info, progress=lambda m: self.after(
                    0, partial(self.test_log.config, text=m))
            )

        def done(result, error) -> None:
            if error is not None:
                self.test_log.config(text=f"Check failed: {error}")
                return
            self.test_log.config(text=result.message)
            if not result.ok:
                messagebox.showwarning("Discord path", result.message, parent=self)

        self.test_log.config(text="Checking...")
        self._run_async(work, done)

    def _play_test(self) -> None:
        from .output import OutputSink
        from .tts import PiperEngine

        def work():
            engine = PiperEngine(self.cfg.tts)
            sink = OutputSink(self.cfg.audio)
            failures = sink.configure(self.cfg.audio.outputs, engine.rate)
            if not sink.targets:
                raise RuntimeError(
                    "No outputs could be opened"
                    + (f": {failures[0][1]}" if failures else "")
                )
            try:
                sink.begin_utterance()
                try:
                    for chunk in engine.stream(
                        "Voice2TTS is working. If you can hear this, setup is complete."
                    ):
                        sink.write(chunk)
                finally:
                    sink.end_utterance()
                sink.wait_drain()
            finally:
                sink.close()
            return [t.name for t in sink.targets]

        def done(result, error) -> None:
            if error is not None:
                self.test_log.config(text="")
                messagebox.showerror("Test failed", str(error), parent=self)
                return
            self.test_log.config(text=f"Played to: {', '.join(result)}")

        self.test_log.config(text="Loading voice...")
        self._run_async(work, done)

    # -- completion -----------------------------------------------------------

    def _finish(self) -> None:
        self._collect_devices()
        self.cfg.first_run_complete = True
        self.cfg.validate()
        self.cfg.save()
        log.info("wizard complete")
        self._close(finished=True)

    def _on_close(self) -> None:
        if self._busy and not messagebox.askyesno(
            "Cancel", "Something is still downloading. Cancel setup?", parent=self
        ):
            return
        # Mark it done anyway; nagging on every launch is worse than a skipped step.
        self.cfg.first_run_complete = True
        self.cfg.save()
        self._close(finished=False)

    def _close(self, finished: bool) -> None:
        self._cancelled = not finished
        if self._on_finish:
            try:
                self._on_finish(finished)
            except Exception:
                log.exception("wizard completion callback failed")
        self.destroy()
