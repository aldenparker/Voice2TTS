"""Light and dark theming for the Tk interface.

Default ttk on Windows looks like 2005, and there is no dark mode at all -- an app
that sits open beside a game or a call at night is the wrong place for a white
window. This restyles the built-in "clam" theme rather than pulling in a widget
toolkit, so it costs no dependency and no packaging weight.

Colours are also used directly by the code that draws status text, so semantic
names live here rather than as hex literals scattered through gui.py.
"""

from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

log = logging.getLogger(__name__)

MODES = ("system", "light", "dark")


@dataclass(frozen=True)
class Palette:
    bg: str
    surface: str
    field: str
    text: str
    muted: str
    border: str
    accent: str
    ok: str
    warn: str
    error: str
    selection: str

    @property
    def is_dark(self) -> bool:
        return self.bg < "#808080"


LIGHT = Palette(
    bg="#f3f3f3", surface="#fafafa", field="#ffffff", text="#1c1c1c",
    muted="#5f5f5f", border="#c8c8c8", accent="#2f6fd0",
    ok="#1a7f4b", warn="#a86400", error="#b3261e", selection="#cfe0f7",
)

DARK = Palette(
    bg="#23252a", surface="#2b2e34", field="#1c1e22", text="#e8e8ea",
    muted="#9aa0a8", border="#3c4048", accent="#6ea8ff",
    ok="#5ad18b", warn="#e0a34a", error="#ff7b72", selection="#31435e",
)


def windows_prefers_dark() -> bool:
    """Whether Windows is set to a dark app theme."""
    import winreg

    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return not value
    except OSError:
        return False


def resolve(mode: str) -> Palette:
    if mode == "dark":
        return DARK
    if mode == "light":
        return LIGHT
    return DARK if windows_prefers_dark() else LIGHT


def apply(root: tk.Misc, mode: str = "system") -> Palette:
    """Restyle every ttk widget class. Returns the palette in use."""
    palette = resolve(mode)
    style = ttk.Style(root)
    try:
        # clam is the only built-in theme that honours colour options properly;
        # vista and xpnative draw from native bitmaps and ignore most of this.
        style.theme_use("clam")
    except tk.TclError:
        log.debug("clam theme unavailable; keeping the default")
        return palette

    p = palette
    try:
        root.configure(bg=p.bg)
    except tk.TclError:
        pass

    style.configure(".", background=p.bg, foreground=p.text,
                    fieldbackground=p.field, bordercolor=p.border,
                    lightcolor=p.surface, darkcolor=p.bg, focuscolor=p.accent)
    style.configure("TFrame", background=p.bg)
    style.configure("TLabel", background=p.bg, foreground=p.text)
    style.configure("TLabelframe", background=p.bg, foreground=p.text)
    style.configure("TLabelframe.Label", background=p.bg, foreground=p.muted)
    style.configure("TCheckbutton", background=p.bg, foreground=p.text)
    style.configure("TRadiobutton", background=p.bg, foreground=p.text)

    style.configure("TButton", background=p.surface, foreground=p.text,
                    bordercolor=p.border, padding=(10, 4))
    style.map("TButton",
              background=[("pressed", p.selection), ("active", p.selection),
                          ("disabled", p.bg)],
              foreground=[("disabled", p.muted)])

    style.configure("TEntry", fieldbackground=p.field, foreground=p.text,
                    insertcolor=p.text, bordercolor=p.border)
    style.configure("TSpinbox", fieldbackground=p.field, foreground=p.text,
                    insertcolor=p.text, arrowcolor=p.text)
    style.configure("TCombobox", fieldbackground=p.field, foreground=p.text,
                    arrowcolor=p.text, bordercolor=p.border)
    # The dropdown list is a plain Tk listbox, not a ttk widget, so it has to be
    # coloured through the option database or it stays white in dark mode.
    root.option_add("*TCombobox*Listbox.background", p.field)
    root.option_add("*TCombobox*Listbox.foreground", p.text)
    root.option_add("*TCombobox*Listbox.selectBackground", p.selection)
    root.option_add("*TCombobox*Listbox.selectForeground", p.text)

    style.configure("TNotebook", background=p.bg, bordercolor=p.border)
    style.configure("TNotebook.Tab", background=p.bg, foreground=p.muted,
                    padding=(12, 6), bordercolor=p.border)
    style.map("TNotebook.Tab",
              background=[("selected", p.surface)],
              foreground=[("selected", p.text)])

    style.configure("Treeview", background=p.field, fieldbackground=p.field,
                    foreground=p.text, bordercolor=p.border, rowheight=22)
    style.configure("Treeview.Heading", background=p.surface, foreground=p.muted)
    style.map("Treeview",
              background=[("selected", p.selection)],
              foreground=[("selected", p.text)])

    style.configure("TProgressbar", background=p.accent, troughcolor=p.field,
                    bordercolor=p.border, lightcolor=p.accent, darkcolor=p.accent)
    style.configure("TScale", background=p.bg, troughcolor=p.field)
    style.configure("TSeparator", background=p.border)
    style.configure("TScrollbar", background=p.surface, troughcolor=p.bg,
                    arrowcolor=p.text, bordercolor=p.border)

    return palette


def style_text_widget(widget: tk.Text, palette: Palette) -> None:
    """Colour a classic Tk Text, which ttk styling does not reach."""
    try:
        widget.configure(
            background=palette.field, foreground=palette.text,
            insertbackground=palette.text, selectbackground=palette.selection,
            selectforeground=palette.text, highlightthickness=1,
            highlightbackground=palette.border, highlightcolor=palette.accent,
            relief="flat", borderwidth=0,
        )
    except tk.TclError as exc:
        log.debug("could not style text widget: %s", exc)
