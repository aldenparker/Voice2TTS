"""The Voice Studio's recording panel.

Kept out of gui.py, which is long enough already, and separable because it
shares nothing with the main window beyond a palette and a place to put itself.

The design intent from the roadmap: reading for thirty minutes is a chore,
reading the next sentence is not. So the panel always shows exactly one prompt
and one obvious button, keeps the running total visible, and gives a verdict on
each take immediately -- a clip that is too quiet is worth thirty seconds to
redo now and worthless to discover after the recording session is over.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import dataset, devices, prompts, recorder
from .theme import Palette

log = logging.getLogger(__name__)

METER_MAX = 100


class RecordingPanel(ttk.Frame):
    """Record prompts into a dataset, one take at a time."""

    def __init__(self, parent, palette: Palette, all_apis: bool = False):
        super().__init__(parent, padding=10)
        self.palette = palette
        self.all_apis = all_apis

        self.session: dataset.RecordingSession | None = None
        self.recorder: recorder.ClipRecorder | None = None
        self.corpus = prompts.load()
        self.queue: list[prompts.Prompt] = []
        self.current: prompts.Prompt | None = None
        self._last_key: str | None = None
        self._tick: str | None = None

        self._build()
        self._refresh_devices()
        self._update_state()

    # -- layout --------------------------------------------------------------

    def _build(self) -> None:
        ttk.Label(self, text="Record your voice", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            self,
            text="Read each sentence aloud in your normal speaking voice. "
                 "Stop when the bar is full.",
            foreground=self.palette.muted, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 8))

        # -- session ---------------------------------------------------------
        setup = ttk.Frame(self)
        setup.grid(row=2, column=0, columnspan=4, sticky="ew")

        ttk.Label(setup, text="Voice name").pack(side="left")
        self.name_var = tk.StringVar(value="My Voice")
        ttk.Entry(setup, textvariable=self.name_var, width=22).pack(
            side="left", padx=(6, 12))

        ttk.Label(setup, text="Target").pack(side="left")
        self.target_var = tk.StringVar(value="30")
        ttk.Spinbox(setup, from_=5, to=120, increment=5, width=5,
                    textvariable=self.target_var).pack(side="left", padx=(6, 2))
        ttk.Label(setup, text="minutes").pack(side="left", padx=(0, 12))

        self.start_btn = ttk.Button(setup, text="Start", command=self._open_session)
        self.start_btn.pack(side="left")

        mic = ttk.Frame(self)
        mic.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(mic, text="Microphone").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(mic, textvariable=self.device_var,
                                       state="readonly", width=44)
        self.device_box.pack(side="left", padx=6)
        ttk.Button(mic, text="Refresh", command=self._refresh_devices).pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=10)

        # -- progress --------------------------------------------------------
        self.progress_label = ttk.Label(self, text="No session yet.")
        self.progress_label.grid(row=5, column=0, columnspan=4, sticky="w")
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        self.remaining_label = ttk.Label(self, text="",
                                         foreground=self.palette.muted)
        self.remaining_label.grid(row=7, column=0, columnspan=4, sticky="w")

        # -- the prompt ------------------------------------------------------
        box = ttk.LabelFrame(self, text="Read this", padding=10)
        box.grid(row=8, column=0, columnspan=4, sticky="ew", pady=10)
        box.columnconfigure(0, weight=1)
        self.prompt_label = ttk.Label(
            box, text="Press Start to begin.", font=("", 13),
            wraplength=520, justify="left",
        )
        self.prompt_label.grid(row=0, column=0, sticky="w")

        # -- controls --------------------------------------------------------
        controls = ttk.Frame(self)
        controls.grid(row=9, column=0, columnspan=4, sticky="ew")
        self.record_btn = ttk.Button(controls, text="Record",
                                     command=self._toggle_record)
        self.record_btn.pack(side="left")
        self.skip_btn = ttk.Button(controls, text="Skip", command=self._skip)
        self.skip_btn.pack(side="left", padx=6)
        self.redo_btn = ttk.Button(controls, text="Redo last", command=self._redo)
        self.redo_btn.pack(side="left")
        ttk.Button(controls, text="Import files…",
                   command=self._import).pack(side="right")

        meter = ttk.Frame(self)
        meter.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(meter, text="Level").pack(side="left")
        self.meter = ttk.Progressbar(meter, mode="determinate", maximum=METER_MAX,
                                     length=220)
        self.meter.pack(side="left", padx=6)
        self.timer_label = ttk.Label(meter, text="", foreground=self.palette.muted)
        self.timer_label.pack(side="left")

        self.verdict = ttk.Label(self, text="", wraplength=560, justify="left")
        self.verdict.grid(row=11, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.columnconfigure(3, weight=1)

    # -- devices -------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._inputs = devices.list_inputs(self.all_apis)
        self.device_box["values"] = [d.display for d in self._inputs]
        if self._inputs and not self.device_var.get():
            default = next((d for d in self._inputs if d.default), self._inputs[0])
            self.device_var.set(default.display)

    def _selected_device(self):
        wanted = self.device_var.get()
        return next((d for d in self._inputs if d.display == wanted), None)

    # -- session -------------------------------------------------------------

    def _open_session(self) -> None:
        if self.session is not None:          # button doubles as "Finish"
            self._close_session()
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Voice Studio", "Give the voice a name first.",
                                   parent=self)
            return
        try:
            target = float(self.target_var.get())
        except ValueError:
            target = 30.0

        self.session = dataset.RecordingSession(name, target_minutes=target)
        # Resuming an existing session picks up its clips, so an interrupted
        # afternoon is not lost and the same sentences are not read twice.
        if self.session.clips:
            self._say(f"Resumed — {self.session.summary()}", self.palette.ok)
        self._rebuild_queue()
        self._advance()
        self._update_state()

    def _close_session(self) -> None:
        if self.recorder is not None:
            self._stop_recording(keep=False)
        self.session = None
        self.current = None
        self.prompt_label.config(text="Press Start to begin.")
        self._update_state()

    def _rebuild_queue(self) -> None:
        """Order the prompts still to read, longest-first by remaining need."""
        if self.session is None:
            self.queue = []
            return
        wpm = prompts.measured_wpm(self.session.words, self.session.seconds)
        needed = max(0.0, self.session.target_seconds - self.session.seconds)
        # Shuffled, so that stopping early still leaves broad phonetic coverage
        # rather than the first N sentences of one author's prose.
        order = prompts.shuffled(self.corpus, seed=0)
        self.queue = prompts.next_prompts(order, self.session.done_keys, needed, wpm)

    def _advance(self) -> None:
        if self.session is None:
            return
        if not self.queue:
            self._rebuild_queue()
        self.current = self.queue.pop(0) if self.queue else None
        self.prompt_label.config(
            text=self.current.text if self.current
            else "That is everything — the target is met.")

    # -- recording -----------------------------------------------------------

    def _toggle_record(self) -> None:
        if self.recorder is not None:
            self._stop_recording(keep=True)
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self.session is None or self.current is None:
            return
        device = self._selected_device()
        if device is None:
            messagebox.showwarning("Voice Studio", "No microphone selected.",
                                   parent=self)
            return
        try:
            self.recorder = recorder.ClipRecorder(device)
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            self.recorder = None
            self._say(f"Could not open {device.name}: {exc}", self.palette.error)
            return
        self.verdict.config(text="")
        self._update_state()
        self._poll_meter()

    def _stop_recording(self, keep: bool) -> None:
        rec, self.recorder = self.recorder, None
        if self._tick is not None:
            self.after_cancel(self._tick)
            self._tick = None
        self.meter["value"] = 0
        self.timer_label.config(text="")
        if rec is None:
            return
        audio, rate = rec.stop()
        self._update_state()
        if not keep or self.session is None or self.current is None:
            return
        if not len(audio):
            self._say("Nothing was recorded — is the right microphone selected?",
                      self.palette.error)
            return

        clip = self.session.add(self.current.key, self.current.text, audio, rate)
        self._last_key = clip.key
        if clip.ok:
            self._say(f"Good — {clip.seconds:.1f}s banked.", self.palette.ok)
            self._advance()
        else:
            # The prompt stays put, because the obvious next action is another
            # take of the same sentence rather than moving on.
            self._say("Needs another take: " + "; ".join(clip.issues),
                      self.palette.warn)
        self._update_state()

    def _poll_meter(self) -> None:
        rec = self.recorder
        if rec is None:
            return
        self.meter["value"] = min(METER_MAX, rec.peak * METER_MAX)
        self.timer_label.config(text=f"{rec.seconds:4.1f}s"
                                     + ("  CLIPPING" if rec.clipped else ""))
        if rec.overran:
            self._stop_recording(keep=True)
            return
        self._tick = self.after(80, self._poll_meter)

    def _skip(self) -> None:
        if self.recorder is not None:
            self._stop_recording(keep=False)
        self._advance()

    def _redo(self) -> None:
        """Drop the last take and put its prompt back in front."""
        if self.session is None or not self._last_key:
            return
        clip = next((c for c in self.session.clips if c.key == self._last_key), None)
        if clip is None:
            return
        self.session.remove(clip.key)
        self.current = prompts.Prompt(clip.key, clip.text)
        self.prompt_label.config(text=clip.text)
        self._last_key = None
        self._say("Dropped that take — read it again.", self.palette.muted)
        self._update_state()

    # -- importing -----------------------------------------------------------

    def _import(self) -> None:
        if self.session is None:
            messagebox.showinfo("Voice Studio", "Start a session first.",
                                parent=self)
            return
        paths = filedialog.askopenfilenames(
            parent=self, title="Import recordings",
            filetypes=[("Wave audio", "*.wav"), ("All files", "*.*")])
        if not paths:
            return
        # Training on someone else's voice without their agreement is the one
        # thing this feature makes easy and should not, so it is asked plainly.
        if not messagebox.askyesno(
                "Voice Studio",
                "Do you have the speaker's permission to train a voice on these "
                "recordings?\n\nIf the voice is not yours, you need their consent.",
                parent=self):
            return

        added = failed = 0
        for raw in paths:
            path = Path(raw)
            # Imported audio has no prompt text, and the trainer needs the
            # transcript, so it is asked for per file rather than guessed.
            text = _ask_text(self, path.name)
            if text is None:
                continue
            try:
                clip = self.session.import_file(path, text)
                added += 1
                if not clip.ok:
                    failed += 1
            except Exception as exc:  # noqa: BLE001 - one bad file is not fatal
                log.warning("import of %s failed: %s", path.name, exc)
                failed += 1
        note = f"Imported {added} file(s)."
        if failed:
            note += f" {failed} need attention — check the list."
        self._say(note, self.palette.warn if failed else self.palette.ok)
        self._rebuild_queue()
        self._update_state()

    # -- state ---------------------------------------------------------------

    def _say(self, text: str, colour: str) -> None:
        self.verdict.config(text=text, foreground=colour)

    def _update_state(self) -> None:
        live = self.session is not None
        busy = self.recorder is not None

        self.start_btn.config(text="Finish" if live else "Start")
        self.record_btn.config(text="Stop" if busy else "Record")
        for widget, enabled in (
            (self.record_btn, live and self.current is not None),
            (self.skip_btn, live and not busy),
            (self.redo_btn, live and not busy and bool(self._last_key)),
        ):
            widget.state(["!disabled"] if enabled else ["disabled"])

        if self.session is None:
            self.progress_label.config(text="No session yet.")
            self.progress["value"] = 0
            self.remaining_label.config(text="")
            return

        self.progress_label.config(text=self.session.summary())
        self.progress["value"] = self.session.progress * 100
        if self.session.ready:
            self.remaining_label.config(text="Target met — you can train now.",
                                        foreground=self.palette.ok)
            return
        wpm = prompts.measured_wpm(self.session.words, self.session.seconds)
        left = self.session.target_seconds - self.session.seconds
        count, _ = prompts.remaining_estimate(
            self.corpus, self.session.done_keys, left, wpm)
        self.remaining_label.config(
            text=f"About {count} more sentences ({left / 60:.0f} min) "
                 f"at {wpm:.0f} words per minute.",
            foreground=self.palette.muted)


def _ask_text(parent, filename: str) -> str | None:
    """Ask what is said in an imported file. None cancels that file."""
    dialog = tk.Toplevel(parent)
    dialog.title("Transcript")
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(False, False)
    ttk.Label(dialog, text=f"What is said in {filename}?", padding=10).pack()
    var = tk.StringVar()
    entry = ttk.Entry(dialog, textvariable=var, width=60)
    entry.pack(padx=10, fill="x")
    entry.focus_set()

    result: dict[str, str | None] = {"text": None}

    def accept() -> None:
        result["text"] = var.get().strip() or None
        dialog.destroy()

    row = ttk.Frame(dialog, padding=10)
    row.pack()
    ttk.Button(row, text="OK", command=accept).pack(side="left")
    ttk.Button(row, text="Skip file", command=dialog.destroy).pack(side="left", padx=6)
    dialog.bind("<Return>", lambda _e: accept())
    dialog.bind("<Escape>", lambda _e: dialog.destroy())
    dialog.grab_set()
    parent.wait_window(dialog)
    return result["text"]
