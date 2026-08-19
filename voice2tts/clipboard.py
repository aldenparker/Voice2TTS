"""Read the Windows clipboard.

Done with ctypes rather than Tk's clipboard_get() because this is called from a
hotkey handler on a worker thread, and Tk may only be touched from the thread
running its mainloop. It also keeps --cli mode working, where no Tk root exists.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass
from enum import StrEnum

log = logging.getLogger(__name__)

CF_UNICODETEXT = 13
MAX_CHARS = 20_000  # a runaway paste should not become a ten-minute utterance

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.OpenClipboard.argtypes = [wintypes.HWND]
_user32.OpenClipboard.restype = wintypes.BOOL
_user32.GetClipboardData.argtypes = [wintypes.UINT]
_user32.GetClipboardData.restype = wintypes.HANDLE
_user32.CloseClipboard.argtypes = []
_user32.CloseClipboard.restype = wintypes.BOOL
_user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
_user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
_kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalLock.restype = wintypes.LPVOID
_kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalUnlock.restype = wintypes.BOOL


class Clipboard(StrEnum):
    """Why there is no text, when there is none."""

    TEXT = "text"
    EMPTY = "the clipboard holds no text"
    BUSY = "another program is holding the clipboard open"
    UNREADABLE = "the clipboard could not be read"


@dataclass(frozen=True)
class Contents:
    """What was on the clipboard, and if nothing, why not."""

    text: str = ""
    why: Clipboard = Clipboard.TEXT

    def __bool__(self) -> bool:
        return bool(self.text)


def get_text(retries: int = 5, delay: float = 0.05) -> str:
    """Current clipboard text, or "" if there is none."""
    return read(retries, delay).text


def read(retries: int = 5, delay: float = 0.05) -> Contents:
    """Current clipboard text, and why there is none when there is none.

    The clipboard is a single shared resource and another process may hold it
    open for a moment, so a failed open is retried rather than treated as empty.
    Reporting that as "empty" told the user to copy something they had already
    copied.
    """
    if not _user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        return Contents(why=Clipboard.EMPTY)

    for _attempt in range(retries):
        if _user32.OpenClipboard(None):
            break
        time.sleep(delay)
    else:
        log.warning("clipboard busy after %d attempts", retries)
        return Contents(why=Clipboard.BUSY)

    try:
        handle = _user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return Contents(why=Clipboard.EMPTY)
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            # The format is there and the open succeeded, so this is not empty:
            # something went wrong reaching the memory behind it.
            return Contents(why=Clipboard.UNREADABLE)
        try:
            text = ctypes.c_wchar_p(pointer).value or ""
        finally:
            _kernel32.GlobalUnlock(handle)
    except Exception as exc:  # noqa: BLE001 - never let a hotkey raise
        log.warning("could not read clipboard: %s", exc)
        return Contents(why=Clipboard.UNREADABLE)
    finally:
        _user32.CloseClipboard()

    if len(text) > MAX_CHARS:
        log.info("clipboard truncated from %d to %d characters", len(text), MAX_CHARS)
        text = text[:MAX_CHARS]
    if not text:
        return Contents(why=Clipboard.EMPTY)
    return Contents(text=text)


def get_speakable_text() -> Contents:
    """Clipboard text tidied for speech, or why there is none.

    Newlines become spaces so a copied paragraph is spoken as prose rather than
    with the long pauses Piper inserts at line breaks.
    """
    found = read()
    if not found:
        return found
    collapsed = " ".join(found.text.split())
    if not collapsed:
        return Contents(why=Clipboard.EMPTY)
    return Contents(text=collapsed)
