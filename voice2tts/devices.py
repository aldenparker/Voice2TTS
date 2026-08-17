"""Audio device discovery.

Devices are addressed by name substring rather than PortAudio index: indices are
not stable across reboots or USB hotplug, and a config that silently starts routing
speech to the wrong device is a nasty failure mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sounddevice as sd

log = logging.getLogger(__name__)

CABLE_HINTS = ("cable input", "vb-audio", "voicemeeter input", "virtual cable")
WASAPI = "Windows WASAPI"


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    hostapi: str
    rate: int
    max_in: int
    max_out: int

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.hostapi}]"


def _all() -> list[Device]:
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            Device(
                index=i,
                name=d["name"],
                hostapi=apis[d["hostapi"]]["name"],
                rate=int(d["default_samplerate"]),
                max_in=int(d["max_input_channels"]),
                max_out=int(d["max_output_channels"]),
            )
        )
    return out


def refresh() -> None:
    """Re-scan hardware. PortAudio caches the device list at init."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:  # noqa: BLE001 - refresh is best-effort
        log.warning("device refresh failed: %s", exc)


def _sorted(devs: list[Device], prefer_wasapi: bool) -> list[Device]:
    if not prefer_wasapi:
        return devs
    return sorted(devs, key=lambda d: (d.hostapi != WASAPI, d.index))


def list_inputs(prefer_wasapi: bool = True) -> list[Device]:
    return _sorted([d for d in _all() if d.max_in > 0], prefer_wasapi)


def list_outputs(prefer_wasapi: bool = True) -> list[Device]:
    return _sorted([d for d in _all() if d.max_out > 0], prefer_wasapi)


def default_input() -> Device | None:
    idx = sd.default.device[0]
    return next((d for d in _all() if d.index == idx), None) if idx is not None else None


def default_output() -> Device | None:
    idx = sd.default.device[1]
    return next((d for d in _all() if d.index == idx), None) if idx is not None else None


def resolve_input(match: str, prefer_wasapi: bool = True) -> Device | None:
    if not match:
        return default_input() or next(iter(list_inputs(prefer_wasapi)), None)
    return _first_match(list_inputs(prefer_wasapi), match)


def resolve_output(match: str, prefer_wasapi: bool = True) -> Device | None:
    if not match:
        return default_output() or next(iter(list_outputs(prefer_wasapi)), None)
    return _first_match(list_outputs(prefer_wasapi), match)


def _first_match(devs: list[Device], match: str) -> Device | None:
    needle = match.strip().lower()
    if not needle:
        return None
    exact = [d for d in devs if d.name.lower() == needle]
    if exact:
        return exact[0]
    partial = [d for d in devs if needle in d.name.lower()]
    if partial:
        return partial[0]
    log.warning("no device matching %r", match)
    return None


def find_cable_output() -> Device | None:
    """Locate a VB-CABLE / VoiceMeeter style virtual input device, if installed."""
    for dev in list_outputs():
        low = dev.name.lower()
        if any(h in low for h in CABLE_HINTS):
            return dev
    return None


def cable_installed() -> bool:
    return find_cable_output() is not None
