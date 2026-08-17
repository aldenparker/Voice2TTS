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
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import (
    checkpoints,
    dataset,
    devices,
    prompts,
    recorder,
    studiopack,
    training,
    voices,
)
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

    # -- teardown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Release the microphone. The window is destroyed when it closes, so
        without this an interrupted take holds the device until the app exits."""
        if self.recorder is not None:
            self._stop_recording(keep=False)

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


class TrainingPanel(ttk.Frame):
    """Turn a recorded dataset into an installed voice.

    Everything slow happens in a subprocess, so this panel only ever starts
    things, reports what they are doing, and stops them. It deliberately keeps
    working across a restart: a run is identified by its work directory, and the
    trainer writes last.ckpt every epoch, so closing the app costs one epoch.
    """

    def __init__(self, parent, palette: Palette):
        super().__init__(parent, padding=10)
        self.palette = palette
        self.run: training.TrainingRun | None = None
        self.checkpoint: checkpoints.Checkpoint | None = None
        self.base_path: Path | None = None
        self._tick: str | None = None

        self._build()
        self.refresh()

    # -- layout --------------------------------------------------------------

    def _build(self) -> None:
        ttk.Label(self, text="Train the voice", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            self,
            text="Hours of work for the graphics card. It can be stopped and "
                 "picked up later —\nprogress is saved every epoch.",
            foreground=self.palette.muted, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 10))

        row = ttk.Frame(self)
        row.grid(row=2, column=0, columnspan=4, sticky="ew")
        ttk.Label(row, text="Dataset").pack(side="left")
        self.dataset_var = tk.StringVar()
        self.dataset_box = ttk.Combobox(row, textvariable=self.dataset_var,
                                        state="readonly", width=34)
        self.dataset_box.pack(side="left", padx=6)
        self.dataset_box.bind("<<ComboboxSelected>>", lambda _e: self._show_dataset())
        ttk.Button(row, text="Refresh", command=self.refresh).pack(side="left")

        self.dataset_label = ttk.Label(self, text="", foreground=self.palette.muted)
        self.dataset_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 8))

        row = ttk.Frame(self)
        row.grid(row=4, column=0, columnspan=4, sticky="ew")
        ttk.Label(row, text="Start from").pack(side="left")
        self.base_var = tk.StringVar()
        self.base_box = ttk.Combobox(row, textvariable=self.base_var,
                                     state="readonly", width=34)
        self.base_box.pack(side="left", padx=6)
        self.base_box.bind("<<ComboboxSelected>>", lambda _e: self._clear_base())
        self.base_btn = ttk.Button(row, text="Get checkpoint",
                                   command=self._get_checkpoint)
        self.base_btn.pack(side="left")

        self.base_label = ttk.Label(self, text="", foreground=self.palette.muted,
                                    wraplength=560, justify="left")
        self.base_label.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 8))

        ttk.Separator(self, orient="horizontal").grid(
            row=6, column=0, columnspan=4, sticky="ew", pady=6)

        row = ttk.Frame(self)
        row.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(4, 8))
        ttk.Label(row, text="Batch size").pack(side="left")
        self.batch_var = tk.StringVar(value="12")
        ttk.Spinbox(row, from_=2, to=64, increment=2, width=5,
                    textvariable=self.batch_var).pack(side="left", padx=6)
        self.batch_hint = ttk.Label(row, text="", foreground=self.palette.muted)
        self.batch_hint.pack(side="left")

        row = ttk.Frame(self)
        row.grid(row=8, column=0, columnspan=4, sticky="ew")
        self.train_btn = ttk.Button(row, text="Start training",
                                    command=self._toggle_training)
        self.train_btn.pack(side="left")
        self.audition_btn = ttk.Button(row, text="Listen",
                                       command=self._audition)
        self.audition_btn.pack(side="left", padx=6)
        self.export_btn = ttk.Button(row, text="Export voice",
                                     command=self._export)
        self.export_btn.pack(side="left")

        self.train_progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.train_progress.grid(row=9, column=0, columnspan=4, sticky="ew",
                                 pady=(10, 4))
        self.train_status = ttk.Label(self, text="Not running.")
        self.train_status.grid(row=10, column=0, columnspan=4, sticky="w")
        self.train_log = ttk.Label(self, text="", foreground=self.palette.muted,
                                   wraplength=580, justify="left")
        self.train_log.grid(row=11, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.columnconfigure(3, weight=1)

    # -- data ----------------------------------------------------------------

    def refresh(self) -> None:
        self._sessions = dataset.RecordingSession.list_sessions()
        self.dataset_box["values"] = [p.name for p in self._sessions]
        if self._sessions and not self.dataset_var.get():
            self.dataset_var.set(self._sessions[0].name)

        self.base_box["values"] = voices.installed_keys()
        if not self.base_var.get() and self.base_box["values"]:
            self.base_var.set(self.base_box["values"][0])

        hardware = studiopack.probe()
        suggested = training.suggest_batch_size(hardware.vram_gb)
        self.batch_var.set(str(suggested))
        self.batch_hint.config(
            text=f"suggested for {hardware.vram_gb:.0f} GB of VRAM"
            if hardware.vram_gb else "no NVIDIA GPU detected")

        self._show_dataset()
        self._update_state()

    def _session(self) -> dataset.RecordingSession | None:
        wanted = self.dataset_var.get()
        path = next((p for p in self._sessions if p.name == wanted), None)
        if path is None:
            return None
        try:
            return dataset.RecordingSession.load(path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            log.warning("could not load session %s: %s", path, exc)
            return None

    def _show_dataset(self) -> None:
        session = self._session()
        if session is None:
            self.dataset_label.config(text="No recorded datasets yet — "
                                           "use the Record tab first.")
            return
        note = session.summary()
        if not session.ready:
            note += "  (below target — training will still run, but on less audio)"
        self.dataset_label.config(text=note)
        self._update_state()

    def _work_dir(self) -> Path | None:
        session = self._session()
        return session.root / "training" if session else None

    # -- base checkpoint -----------------------------------------------------

    def _clear_base(self) -> None:
        self.checkpoint = None
        self.base_path = None
        self.base_label.config(text="", foreground=self.palette.muted)
        self._update_state()

    def _get_checkpoint(self) -> None:
        """Find, describe and download the checkpoint for the chosen voice."""
        key = self.base_var.get()
        if not key:
            return
        self.base_label.config(text=f"Looking up {key}…",
                               foreground=self.palette.muted)
        self.update_idletasks()
        try:
            found = checkpoints.resolve(key)
        except LookupError as exc:
            # Bundled voices do not all have a published checkpoint; the message
            # from resolve() names the qualities that do.
            self.base_label.config(text=str(exc), foreground=self.palette.error)
            return
        except Exception as exc:  # noqa: BLE001 - network, shown to the user
            self.base_label.config(text=f"Could not reach the checkpoint "
                                        f"repository: {exc}",
                                   foreground=self.palette.error)
            return

        card = checkpoints.model_card(found)
        licence = checkpoints.licence_from_card(card)
        # A trained voice inherits these terms, so they are agreed to before the
        # download rather than mentioned afterwards.
        terms = licence or "not stated on the model card"
        if not messagebox.askyesno(
                "Base checkpoint",
                f"{key}\n\n{found.size_gb:.2f} GB download.\n\n"
                f"Your voice will be built on this one and inherits its terms.\n"
                f"Licence: {terms}\n\nDownload it now?",
                parent=self):
            return

        dest = studiopack.studio_dir() / "checkpoints" / key
        self.checkpoint = found

        def progress(done: int, total: int) -> None:
            pct = done * 100 // total if total else 0
            self.base_label.config(
                text=f"Downloading {found.filename}: {pct}% "
                     f"({done / 1e6:.0f} of {total / 1e6:.0f} MB)")
            self.update_idletasks()

        try:
            self.base_path = checkpoints.download(found, dest, progress)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.base_label.config(text=f"Download failed: {exc}",
                                   foreground=self.palette.error)
            self.base_path = None
            return
        self.base_label.config(
            text=f"Ready: {found.filename} — licence: {terms}",
            foreground=self.palette.ok)
        self._update_state()

    # -- training ------------------------------------------------------------

    def _toggle_training(self) -> None:
        if self.run is not None and self.run.running:
            self.run.stop()
            self.train_status.config(text="Stopping…")
            return
        self._start_training()

    def _start_training(self) -> None:
        session = self._session()
        work = self._work_dir()
        if session is None or work is None:
            return
        if not session.usable:
            messagebox.showwarning("Voice Studio",
                                   "That dataset has no usable clips yet.",
                                   parent=self)
            return
        if not studiopack.status().usable:
            messagebox.showwarning(
                "Voice Studio",
                "The training environment is not installed yet. "
                "Set it up on the Setup tab first.", parent=self)
            return

        resuming = training.resume_point(work) is not None
        if not resuming and self.base_path is None:
            messagebox.showwarning(
                "Voice Studio",
                "Choose a voice to start from and download its checkpoint "
                "first.\n\nTraining from nothing needs far more audio than a "
                "recording session provides.", parent=self)
            return

        try:
            csv_path = session.prepare()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Voice Studio", f"Could not prepare the "
                                                 f"dataset:\n{exc}", parent=self)
            return

        try:
            batch = int(self.batch_var.get())
        except ValueError:
            batch = 12

        cfg = training.TrainingConfig(
            voice_name=_slug(session.name),
            dataset_csv=csv_path,
            work_dir=work,
            base_checkpoint=self.base_path,
            batch_size=batch,
        )
        self.run = training.TrainingRun(cfg, on_line=self._on_line)
        try:
            self.run.start()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.run = None
            messagebox.showerror("Voice Studio",
                                 f"Training would not start:\n{exc}", parent=self)
            return
        self.train_status.config(text="Starting…", foreground=self.palette.text)
        self._update_state()
        self._poll_training()

    def _on_line(self, line: str) -> None:
        # Called from the reader thread; Tk is not thread-safe, so this only
        # stores the line and _poll_training does the widget work.
        self._last_line = line

    def _poll_training(self) -> None:
        run = self.run
        if run is None:
            return
        progress = run.progress
        self.train_progress["value"] = progress.fraction * 100
        loss = f", loss {progress.loss:.3f}" if progress.loss is not None else ""
        if run.running:
            self.train_status.config(
                text=f"Epoch {progress.epoch}, step {progress.step}"
                     f"/{progress.total_steps}{loss} — "
                     f"{run.elapsed / 60:.0f} min elapsed")
            self.train_log.config(text=getattr(self, "_last_line", "")[:160])
            self._tick = self.after(500, self._poll_training)
            return

        if run.error:
            self.train_status.config(text="Training stopped with an error.",
                                     foreground=self.palette.error)
            self.train_log.config(text=run.error[-400:])
        else:
            self.train_status.config(
                text=f"Stopped after {run.elapsed / 60:.0f} minutes. "
                     "Export the voice, or start again to continue.",
                foreground=self.palette.text)
        self._update_state()

    # -- teardown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop polling, but leave training running.

        Killing it would throw away hours of GPU work because a window closed.
        The pid file is what lets the next panel find it again.
        """
        if self._tick is not None:
            self.after_cancel(self._tick)
            self._tick = None

    # -- auditioning ---------------------------------------------------------

    def _audition(self) -> None:
        """Listen to the latest checkpoint, mid-run if need be.

        Allowed while training is going: it loads the checkpoint file, which is
        already written, and runs on CPU, so it does not disturb the run.
        """
        work = self._work_dir()
        if work is None:
            return
        best = training.best_checkpoint(work)
        if best is None:
            messagebox.showinfo("Voice Studio", "Nothing has been trained yet.",
                                parent=self)
            return
        self.train_log.config(text=f"Synthesising from {best.name} on the CPU — "
                                   "this takes a few seconds…")
        self.update_idletasks()
        try:
            wav = training.audition(best, work / "config.json", work / "audition")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.train_log.config(text=f"Could not synthesise: {exc}")
            return
        try:
            _play_wav(wav)
        except Exception as exc:  # noqa: BLE001 - the file is still on disk
            self.train_log.config(text=f"Rendered {wav.name} but could not "
                                       f"play it: {exc}")
            return
        self.train_log.config(text=f"Played {best.name}.")

    # -- export --------------------------------------------------------------

    def _export(self) -> None:
        session = self._session()
        work = self._work_dir()
        if session is None or work is None:
            return
        best = training.best_checkpoint(work)
        if best is None:
            messagebox.showinfo("Voice Studio",
                                "Nothing has been trained yet.", parent=self)
            return

        name = _slug(session.name)
        prov = training.Provenance(
            voice_name=name,
            base_checkpoint=self.checkpoint.filename if self.checkpoint else "",
            dataset_clips=len(session.usable),
            dataset_seconds=round(session.seconds, 1),
            epochs=self.run.progress.epoch if self.run else 0,
        )
        self.train_status.config(text=f"Exporting from {best.name}…")
        self.update_idletasks()
        try:
            final = training.export(best, work / "config.json",
                                    voices.user_voices_dir(), name, prov)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.train_status.config(text=f"Export failed: {exc}",
                                     foreground=self.palette.error)
            return
        self.train_status.config(
            text=f"Installed as {final.name}. Pick it on the Voice tab.",
            foreground=self.palette.ok)

    # -- state ---------------------------------------------------------------

    def _update_state(self) -> None:
        work = self._work_dir()
        has_data = self._session() is not None
        trained = work is not None and training.best_checkpoint(work) is not None

        ours = self.run is not None and self.run.running
        # A run started before this window was last closed is still going, and
        # this panel has no handle on it. It must not be offered a second start.
        orphan = (None if ours or work is None
                  else training.running_elsewhere(work))
        busy = ours or orphan is not None
        if orphan is not None:
            self.train_status.config(
                text=f"Training is already running for this dataset "
                     f"(process {orphan}). Progress is not shown here because it "
                     f"was started before this window was reopened; it will "
                     f"finish on its own.")

        self.train_btn.config(text="Stop" if ours else "Start training")
        for widget, enabled in (
            # An orphan cannot be stopped from here -- there is no handle on it.
            (self.train_btn, has_data and not orphan),
            (self.export_btn, trained and not busy),
            # Auditioning reads a written checkpoint on the CPU, so it stays
            # available mid-run -- that is when it is most worth doing.
            (self.audition_btn, trained),
            (self.base_btn, not busy),
            (self.dataset_box, not busy),
        ):
            widget.state(["!disabled"] if enabled else ["disabled"])


def _play_wav(path: Path) -> None:
    """Play a wav on the default output. Blocks, which is fine for one sentence."""
    import sounddevice as sd

    audio, rate = dataset.read_wav(path)
    sd.play(audio, rate)
    sd.wait()


def _slug(name: str) -> str:
    """A filename-safe voice name. Piper voices are dash-separated by habit."""
    kept = [c if c.isalnum() else "-" for c in name.strip().lower()]
    return re.sub(r"-+", "-", "".join(kept)).strip("-") or "my-voice"


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
