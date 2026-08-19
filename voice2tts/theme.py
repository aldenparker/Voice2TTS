"""Semantic colours, and optional light/dark theming.

The DEFAULT is "native": Windows' own ttk theme, untouched. This is a functional
utility, and native widgets are what a functional utility should look like -- they
match the rest of the system, respect accessibility settings, and never look
subtly wrong in the way a hand-rolled theme eventually does.

Light and dark remain available for anyone who wants a dark window beside a game
at night. Those repaint the built-in "clam" theme, which is the only bundled theme
that honours colour options; vista and xpnative draw from native bitmaps and
ignore most styling.

Either way, semantic colour names live here rather than as hex literals scattered
through gui.py, so status text stays legible in whichever mode is active.
"""

from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from .modes import Theme

log = logging.getLogger(__name__)

# One definition, shared with the config, so the picker cannot offer a value
# validate() rejects -- or accept one it does not offer, which is how "system"
# came to survive validation while not appearing in the combo box it was
# written to. native first: it is the default, and the order is what shows.
MODES = Theme.values()


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
    # When set, widgets keep their platform appearance and only the semantic
    # colours below are used. Nothing is restyled.
    native: bool = False

    @property
    def is_dark(self) -> bool:
        return not self.native and self.bg < "#808080"


# The status colours the app used before theming existed. Keeping the exact values
# means "native" looks identical to how it always did.
NATIVE = Palette(
    bg="#f0f0f0", surface="#f0f0f0", field="#ffffff", text="#000000",
    muted="#666666", border="#a0a0a0", accent="#0078d7",
    ok="#2a7745", warn="#cc8800", error="#aa3333", selection="#cce4f7",
    native=True,
)

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


def resolve(mode: Theme | str) -> Palette:
    # Tolerates a plain string because Tk hands one back from the picker, and
    # "system" because that was this setting's name before schema 3.
    if mode == "system":
        return DARK if windows_prefers_dark() else LIGHT
    match Theme.parse(mode):
        case Theme.DARK:
            return DARK
        case Theme.LIGHT:
            return LIGHT
        case _:
            return NATIVE


def apply(root: tk.Misc, mode: Theme | str = Theme.NATIVE) -> Palette:
    """Apply a theme. Returns the palette in use.

    In native mode nothing is restyled: Windows' own widget appearance is left
    exactly as it is, and only the semantic colours are used by callers.
    """
    palette = resolve(mode)
    style = ttk.Style(root)

    if palette.native:
        # Restore the platform default, in case a previous call switched to clam
        # during this session.
        for candidate in ("vista", "winnative", "xpnative", "default"):
            if candidate in style.theme_names():
                try:
                    style.theme_use(candidate)
                    break
                except tk.TclError:
                    continue
        _restore_listbox_colours(root)
        return palette

    try:
        style.theme_use("clam")
    except tk.TclError:
        log.debug("clam theme unavailable; keeping the default")
        return palette

    p = palette
    try:
        root.configure(bg=p.bg)  # type: ignore[call-arg]  # Misc in the stubs
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


# The combobox dropdown is a plain Tk listbox reached only through the option
# database. Windows' own colours are used by name so the dropdown follows a
# high-contrast or custom system theme; the hex values are a fallback for anywhere
# those names are not recognised.
_NATIVE_LISTBOX = (
    ("background", "SystemWindow", NATIVE.field),
    ("foreground", "SystemWindowText", NATIVE.text),
    ("selectBackground", "SystemHighlight", NATIVE.selection),
    ("selectForeground", "SystemHighlightText", NATIVE.text),
)


def _restore_listbox_colours(root: tk.Misc) -> None:
    """Put the combobox dropdown back to the platform colours.

    These options are sticky, so a previous light/dark apply() has to be undone
    explicitly. Setting them to "" does NOT unset them -- it stores an empty string
    that Tk then fails to parse, and every dropdown in the application silently
    stops opening with 'unknown color name ""'. Always write a real colour.
    """
    for option, system_name, fallback in _NATIVE_LISTBOX:
        colour = system_name
        try:
            root.winfo_rgb(system_name)
        except tk.TclError:
            colour = fallback
        try:
            root.option_add(f"*TCombobox*Listbox.{option}", colour)
        except tk.TclError as exc:
            log.debug("could not restore %s: %s", option, exc)


def style_text_widget(widget: tk.Text, palette: Palette) -> None:
    """Colour a classic Tk Text, which ttk styling does not reach.

    A no-op in native mode: the platform default appearance is the point.
    """
    if palette.native:
        return
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
