"""Windows integration: DPI awareness, single-instance guard, run-at-login.

Small pieces of ctypes and registry work that have to happen at very specific
moments -- DPI before Tk exists, the instance guard before anything grabs the
microphone or the global hotkey.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from pathlib import Path

from .paths import is_frozen

log = logging.getLogger(__name__)

# A GUID-ish name keeps this from colliding with unrelated software. "Local\\"
# scopes it to the session, so two different users can each run their own copy.
MUTEX_NAME = r"Local\Voice2TTS-7B3C9F1E-4A2D-4E7B-9C51-2F8E6D4A1B93"
ACTIVATE_EVENT_NAME = r"Local\Voice2TTS-activate-7B3C9F1E"

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "Voice2TTS"

_ERROR_ALREADY_EXISTS = 183


# -- DPI --------------------------------------------------------------------


def enable_dpi_awareness() -> str:
    """Opt into per-monitor DPI so Tk is not bitmap-stretched on scaled displays.

    Must be called before the first Tk window exists; Windows latches the process
    DPI mode at first use. Returns a short description of what was applied.
    """
    try:
        # -4 == PER_MONITOR_AWARE_V2: correct scaling when a window moves between
        # monitors with different scaling, not just at startup.
        ctx = ctypes.c_void_p(-4)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctx):
            return "per-monitor-v2"
    except Exception:  # noqa: BLE001 - older Windows lacks this entry point
        pass
    try:
        # 2 == PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:  # noqa: BLE001
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except Exception as exc:  # noqa: BLE001 - not fatal, just blurry
        log.warning("could not set DPI awareness: %s", exc)
        return "none"


def apply_tk_scaling(root) -> float:
    """Match Tk's point-to-pixel ratio to the actual display DPI.

    DPI awareness alone stops Windows blurring the window, but Tk still assumes
    96 DPI internally, so text renders tiny on a scaled display without this.
    """
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if dpi and dpi != 96:
            scaling = dpi / 72.0
            root.tk.call("tk", "scaling", scaling)
            log.info("display %d DPI, Tk scaling %.2f", dpi, scaling)
            return scaling
    except Exception as exc:  # noqa: BLE001
        log.warning("could not apply Tk scaling: %s", exc)
    return 0.0


# -- single instance --------------------------------------------------------


class SingleInstance:
    """Named-mutex guard.

    Two copies running at once is not a cosmetic problem: both register the same
    global hotkey (so one keypress starts two captures), both open the microphone,
    and both show a tray icon. Detected here rather than left to confuse the user.
    """

    def __init__(self, name: str = MUTEX_NAME):
        self.name = name
        self._handle = None
        self.already_running = False

    def acquire(self) -> bool:
        """True if we are the first instance. Safe to call once per process."""
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.CreateMutexW.argtypes = [
                wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCWSTR
            ]
            self._handle = kernel32.CreateMutexW(None, False, self.name)
            last_error = ctypes.get_last_error()
        except Exception as exc:  # noqa: BLE001 - never block startup over this
            log.warning("single-instance check failed: %s", exc)
            return True

        if not self._handle:
            log.warning("could not create mutex; allowing this instance")
            return True

        self.already_running = last_error == _ERROR_ALREADY_EXISTS
        return not self.already_running

    def signal_existing(self) -> bool:
        """Ask the running instance to surface its settings window."""
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenEventW.restype = wintypes.HANDLE
            kernel32.OpenEventW.argtypes = [
                wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR
            ]
            handle = kernel32.OpenEventW(0x0002, False, ACTIVATE_EVENT_NAME)  # EVENT_MODIFY_STATE
            if not handle:
                return False
            kernel32.SetEvent(handle)
            kernel32.CloseHandle(handle)
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("could not signal existing instance: %s", exc)
            return False

    def release(self) -> None:
        if self._handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001
                pass
            self._handle = None


def listen_for_activation(callback) -> None:
    """Block on the activation event, calling `callback` each time it fires.

    Run this on a daemon thread in the first instance so a second launch surfaces
    the existing window instead of doing nothing visible.
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateEventW.restype = wintypes.HANDLE
        kernel32.CreateEventW.argtypes = [
            wintypes.LPCVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR
        ]
        handle = kernel32.CreateEventW(None, False, False, ACTIVATE_EVENT_NAME)
        if not handle:
            return
        while True:
            if kernel32.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
                return
            try:
                callback()
            except Exception:
                log.exception("activation callback failed")
    except Exception as exc:  # noqa: BLE001
        log.debug("activation listener stopped: %s", exc)


# -- run at login -----------------------------------------------------------


def _executable_command() -> str | None:
    if not is_frozen():
        return None
    return f'"{Path(sys.executable)}"'


def run_at_login() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, RUN_VALUE)
            return bool(value)
    except OSError:
        return False


def set_run_at_login(enabled: bool) -> bool:
    """Add or remove the HKCU Run entry. Returns True on success.

    HKCU rather than HKLM so this never needs administrator rights.
    """
    import winreg

    command = _executable_command()
    if enabled and command is None:
        log.warning("run-at-login is only meaningful for an installed build")
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                            winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, command)
                log.info("run at login enabled: %s", command)
            else:
                try:
                    winreg.DeleteValue(key, RUN_VALUE)
                    log.info("run at login disabled")
                except FileNotFoundError:
                    pass
        return True
    except OSError as exc:
        log.error("could not update run-at-login: %s", exc)
        return False
