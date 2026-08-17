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


def get_text(retries: int = 5, delay: float = 0.05) -> str:
    """Current clipboard text, or "" if there is none.

    The clipboard is a single shared resource and another process may hold it open
    for a moment, so a failed open is retried rather than treated as empty.
    """
    if not _user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        return ""

    for _attempt in range(retries):
        if _user32.OpenClipboard(None):
            break
        time.sleep(delay)
    else:
        log.warning("clipboard busy after %d attempts", retries)
        return ""

    try:
        handle = _user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return ""
        try:
            text = ctypes.c_wchar_p(pointer).value or ""
        finally:
            _kernel32.GlobalUnlock(handle)
    except Exception as exc:  # noqa: BLE001 - never let a hotkey raise
        log.warning("could not read clipboard: %s", exc)
        return ""
    finally:
        _user32.CloseClipboard()

    if len(text) > MAX_CHARS:
        log.info("clipboard truncated from %d to %d characters", len(text), MAX_CHARS)
        text = text[:MAX_CHARS]
    return text


def get_speakable_text() -> str:
    """Clipboard text tidied for speech.

    Newlines become spaces so a copied paragraph is spoken as prose rather than
    with the long pauses Piper inserts at line breaks.
    """
    text = get_text()
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed
