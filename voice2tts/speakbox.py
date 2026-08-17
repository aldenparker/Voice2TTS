"""A small always-available window for typing text to speak.

The tray's "Speak text..." prompt is modal and one-shot, which suits speaking one
thing and suits nothing else. Anyone using this as their voice types repeatedly, so
the box stays open, keeps a history, and never steals focus back from the game or
call it sits beside.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

log = logging.getLogger(__name__)

RECENT = 12


class SpeakBox(tk.Toplevel):
    def __init__(self, master: tk.Tk, pipeline, on_close=None):
        super().__init__(master)
        self.pipeline = pipeline
        self._on_close = on_close
        self._recent: list[str] = []
        self._recent_index = -1

        self.title("Speak")
        self.geometry("460x210")
        self.minsize(360, 170)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.entry = tk.Text(self, height=4, wrap="word")
        self.entry.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self.entry.focus_set()

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(bar, text="Speak", command=self.speak).pack(side="right")
        ttk.Button(bar, text="Stop", command=self._stop).pack(side="right", padx=6)
        self.keep_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Keep on top", variable=self.keep_var,
                        command=self._apply_topmost).pack(side="left")
        self.clear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Clear after", variable=self.clear_var).pack(
            side="left", padx=8)

        self.status = ttk.Label(self, text="Enter speaks.  Shift+Enter for a new line."
                                           "  Up/Down for recent.",
                                foreground="#666")
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        # Enter speaks; Shift+Enter inserts a newline. "break" stops Tk also
        # inserting the newline that triggered us.
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda _e: None)
        self.entry.bind("<Control-Return>", self._on_return)
        self.entry.bind("<Up>", self._history_back)
        self.entry.bind("<Down>", self._history_forward)
        self.bind("<Escape>", lambda _e: self.close())
        self._apply_topmost()

    # -- actions --------------------------------------------------------------

    def _apply_topmost(self) -> None:
        self.attributes("-topmost", bool(self.keep_var.get()))

    def _on_return(self, _event) -> str:
        self.speak()
        return "break"

    def speak(self) -> None:
        text = self.entry.get("1.0", "end-1c").strip()
        if not text:
            return
        if not self.pipeline.running:
            self.status.config(text="Start Voice2TTS first.", foreground="#a33")
            return

        self.pipeline.say_text(text, source="typed")
        if not self._recent or self._recent[-1] != text:
            self._recent.append(text)
            del self._recent[:-RECENT]
        self._recent_index = len(self._recent)
        self.status.config(text=f"Speaking: {text[:48]}", foreground="#2a7")
        if self.clear_var.get():
            self.entry.delete("1.0", "end")

    def _stop(self) -> None:
        if self.pipeline.running and self.pipeline.stop_speaking():
            self.status.config(text="Stopped.", foreground="#666")

    # -- recent entries -------------------------------------------------------

    def _history_back(self, _event) -> str | None:
        # Only take over Up when the caret is on the first line, so arrow keys
        # still navigate a multi-line message normally.
        if not self._recent or self.entry.index("insert").startswith("1."):
            if not self._recent:
                return None
            self._recent_index = max(0, self._recent_index - 1)
            self._show_recent()
            return "break"
        return None

    def _history_forward(self, _event) -> str | None:
        if not self._recent:
            return None
        if self._recent_index >= len(self._recent) - 1:
            self._recent_index = len(self._recent)
            self.entry.delete("1.0", "end")
            return "break"
        self._recent_index += 1
        self._show_recent()
        return "break"

    def _show_recent(self) -> None:
        if 0 <= self._recent_index < len(self._recent):
            self.entry.delete("1.0", "end")
            self.entry.insert("1.0", self._recent[self._recent_index])

    def close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()
