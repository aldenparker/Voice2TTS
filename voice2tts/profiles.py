"""Named configuration snapshots.

Different situations want different settings: a meeting wants push-to-talk and a
neutral voice; a game wants automatic detection and a different output. Editing
Settings each time is friction enough that people simply do not bother.

A profile stores only the fields worth varying. Device selections, models and
update settings stay global, because those describe the machine rather than the
situation -- and a profile that silently changed your microphone would be a
surprise, not a convenience.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Only these are captured. Anything else is machine-level and stays put.
#
# Deliberately NOT here: anything that decides which models get loaded --
# stt.model, and how translation happens. Switching profile is meant to be
# instant, and those cost a reload of Whisper or a translation chain. A profile
# that took four seconds to apply, or half-applied, would be worse than editing
# Settings.
PROFILED_FIELDS = (
    "trigger.mode",
    "trigger.hotkey",
    "trigger.ptt_latch",
    "tts.voice",
    "tts.length_scale",
    "tts.volume",
    # Paired with trigger.mode on purpose: streaming needs automatic detection,
    # so a profile that set push-to-talk without this left an impossible pair
    # for validate() to pull apart afterwards.
    "stt.mode",
    "vad.threshold",
    "vad.min_silence_ms",
    "text.review_before_speaking",
    "text.substitutions_enabled",
    "audio.mute_mic_during_playback",
)


@dataclass
class Profile:
    name: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    # Executable names that select this profile automatically, e.g. "discord.exe".
    match_apps: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        mode = str(self.values.get("trigger.mode", "?"))
        voice = str(self.values.get("tts.voice", "?"))
        return f"{mode}, {voice}"


def _get(cfg, path: str):
    section, _, name = path.partition(".")
    return getattr(getattr(cfg, section), name)


def _set(cfg, path: str, value) -> None:
    section, _, name = path.partition(".")
    setattr(getattr(cfg, section), name, value)


def capture(cfg, name: str) -> Profile:
    """Snapshot the profiled fields of `cfg`."""
    values = {}
    for path in PROFILED_FIELDS:
        try:
            values[path] = copy.deepcopy(_get(cfg, path))
        except AttributeError:
            log.debug("profile field %s missing; skipped", path)
    return Profile(name=name, values=values)


def apply(cfg, profile: Profile) -> list[str]:
    """Write a profile into `cfg`. Returns the fields that actually changed."""
    changed = []
    for path, value in profile.values.items():
        if path not in PROFILED_FIELDS:
            log.warning("ignoring unknown profile field %s", path)
            continue
        try:
            if _get(cfg, path) != value:
                _set(cfg, path, copy.deepcopy(value))
                changed.append(path)
        except AttributeError:
            log.debug("profile field %s missing; skipped", path)
    # A profile is a partial config, so applying one can produce a combination
    # neither half asked for. Repairs are published rather than dropped, the
    # same way loading a file publishes them.
    cfg.repairs = cfg.validate()
    return changed


def to_dict(profile: Profile) -> dict:
    return dataclasses.asdict(profile)


def from_dict(data: dict) -> Profile:
    return Profile(
        name=str(data.get("name", "")),
        values={k: v for k, v in (data.get("values") or {}).items()
                if k in PROFILED_FIELDS},
        match_apps=[str(a).lower() for a in (data.get("match_apps") or [])],
    )


def find_for_app(profiles: list[Profile], executable: str) -> Profile | None:
    """The profile claiming `executable`, if any."""
    exe = (executable or "").lower()
    if not exe:
        return None
    for profile in profiles:
        if any(app and app in exe for app in profile.match_apps):
            return profile
    return None


def foreground_executable() -> str:
    """Name of the process owning the foreground window, or "".

    Used for automatic switching. Failure is not interesting -- there may be no
    foreground window at all -- so it returns "" rather than raising.
    """
    import ctypes
    from ctypes import wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""

        # PROCESS_QUERY_LIMITED_INFORMATION: enough for the image name, and
        # available without elevation for processes we do not own.
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return ""
        try:
            buffer = ctypes.create_unicode_buffer(512)
            size = wintypes.DWORD(len(buffer))
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                return ""
            return buffer.value.rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(handle)
    except Exception as exc:  # noqa: BLE001
        log.debug("foreground lookup failed: %s", exc)
        return ""
