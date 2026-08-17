"""Global hotkey with press AND release events, for push-to-talk.

pynput's GlobalHotKeys only reports activation, which is useless for hold-to-talk,
so this tracks modifier state manually over a raw Listener.

Keys are deliberately not suppressed: swallowing them would break the hotkey inside
games and chat apps. Pick a combo you do not otherwise use.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from pynput import keyboard

log = logging.getLogger(__name__)

_MODIFIER_ALIASES = {
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt",
    "shift": "shift",
    "win": "cmd", "cmd": "cmd", "super": "cmd", "meta": "cmd",
}

_MODIFIER_KEYS: dict[str, set] = {
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}


class HotkeySpec:
    """A parsed combo such as 'ctrl+alt+v' or 'f8'."""

    def __init__(self, text: str):
        self.text = text
        self.modifiers: set[str] = set()
        self.key: object | None = None
        self._parse(text)

    def _parse(self, text: str) -> None:
        parts = [p.strip().lower() for p in text.split("+") if p.strip()]
        if not parts:
            raise ValueError("empty hotkey")
        for part in parts[:-1]:
            mod = _MODIFIER_ALIASES.get(part)
            if mod is None:
                raise ValueError(f"unknown modifier {part!r}")
            self.modifiers.add(mod)

        main = parts[-1]
        if main in _MODIFIER_ALIASES:
            # Combo is modifiers only, e.g. "ctrl+shift"; use the last as the trigger.
            self.modifiers.discard(_MODIFIER_ALIASES[main])
            self.key = ("modifier", _MODIFIER_ALIASES[main])
        elif len(main) == 1:
            self.key = keyboard.KeyCode.from_char(main)
        else:
            named = getattr(keyboard.Key, main, None)
            if named is None:
                raise ValueError(f"unknown key {main!r}")
            self.key = named

    def matches_main(self, key) -> bool:
        if isinstance(self.key, tuple):
            return key in _MODIFIER_KEYS[self.key[1]]
        if isinstance(self.key, keyboard.KeyCode) and isinstance(key, keyboard.KeyCode):
            a, b = self.key.char, key.char
            return a is not None and b is not None and a.lower() == b.lower()
        return self.key == key

    def __str__(self) -> str:
        return self.text


def describe(text: str) -> str:
    """Validate a hotkey string, returning '' if it parses or an error message."""
    try:
        HotkeySpec(text)
    except ValueError as exc:
        return str(exc)
    return ""


class HotkeyListener:
    """Fires on_press when the combo completes and on_release when it breaks."""

    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ):
        self._spec = HotkeySpec(hotkey)
        self._on_press = on_press
        self._on_release = on_release
        self._held_mods: set[str] = set()
        self._engaged = False
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.daemon = True
        self._listener.start()
        log.info("hotkey listening for %s", self._spec)

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
        with self._lock:
            was = self._engaged
            self._engaged = False
            self._held_mods.clear()
        if was:
            self._safe(self._on_release)

    def set_hotkey(self, hotkey: str) -> None:
        spec = HotkeySpec(hotkey)  # parse first so a bad string changes nothing
        running = self._listener is not None
        self.stop()
        self._spec = spec
        if running:
            self.start()

    # -- listener thread ------------------------------------------------------

    def _canonical(self, key):
        try:
            return self._listener.canonical(key) if self._listener else key
        except Exception:  # noqa: BLE001 - canonical() can throw on exotic keys
            return key

    @staticmethod
    def _modifier_name(key) -> str | None:
        for name, keys in _MODIFIER_KEYS.items():
            if key in keys:
                return name
        return None

    def _handle_press(self, key) -> None:
        raw = key
        key = self._canonical(key)
        # Modifier identity is only reliable on the raw key; canonical() maps
        # left/right variants inconsistently across layouts.
        mod = self._modifier_name(raw) or self._modifier_name(key)
        fire = False
        with self._lock:
            if mod:
                self._held_mods.add(mod)
            if (
                not self._engaged
                and (self._spec.matches_main(key) or self._spec.matches_main(raw))
                and self._spec.modifiers <= self._held_mods
            ):
                self._engaged = True
                fire = True
        if fire:
            self._safe(self._on_press)

    def _handle_release(self, key) -> None:
        raw = key
        key = self._canonical(key)
        mod = self._modifier_name(raw) or self._modifier_name(key)
        fire = False
        with self._lock:
            if mod:
                self._held_mods.discard(mod)
            if self._engaged:
                is_main = self._spec.matches_main(key) or self._spec.matches_main(raw)
                # Releasing either the main key or any required modifier ends it.
                if is_main or (mod and mod in self._spec.modifiers):
                    self._engaged = False
                    fire = True
        if fire:
            self._safe(self._on_release)

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback failed")
