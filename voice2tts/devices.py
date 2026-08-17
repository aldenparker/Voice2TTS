"""Audio device discovery.

Devices are addressed by name substring rather than PortAudio index: indices are
not stable across reboots or USB hotplug, and a config that silently starts routing
speech to the wrong device is a nasty failure mode.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

import sounddevice as sd

log = logging.getLogger(__name__)

WASAPI = "Windows WASAPI"

# PortAudio exposes every endpoint through four host APIs, so an unfiltered list
# shows each device up to four times -- 54 entries on a machine with 12 real ones.
# Windows and Discord show one row per endpoint, so matching that is what users
# expect. WASAPI is also the only API that reports untruncated names: MME cuts them
# to 31 characters, which loses the "(driver)" suffix cable pairing depends on.
#
# Routing aggregates rather than real endpoints. Selecting one "works" but silently
# follows the Windows default instead of the device the user picked.
_PSEUDO_HINTS = (
    "microsoft sound mapper",
    "primary sound driver",
    "primary sound capture driver",
)


# Dropdown annotations. Defined here so the picker that adds them and the code that
# strips them before saving cannot drift apart -- a stale suffix in the config would
# stop the device resolving at all.
DEFAULT_TAG = "   (system default)"
VIRTUAL_TAG = "   (virtual cable)"

# Two endpoints can share a name -- a dual-monitor setup shows two identical
# "ED340CUR X0 (NVIDIA High Definition Audio)" outputs. Since config stores a name
# and resolution takes the first match, the second would be unselectable. A "#2"
# suffix disambiguates, and unlike the other tags it is KEPT in the stored value
# because it carries meaning.
_ORDINAL = re.compile(r"\s+#(\d+)\s*$")


def strip_display(value: str) -> str:
    """Remove decorative annotations, keeping any meaningful #N ordinal."""
    value = (value or "").strip()
    for tag in (DEFAULT_TAG.strip(), VIRTUAL_TAG.strip()):
        if value.endswith(tag):
            value = value[: -len(tag)].strip()
            break
    return value


def split_ordinal(match: str) -> tuple[str, int]:
    """Split 'Name #2' into ('Name', 2). Returns index 1 when unnumbered."""
    found = _ORDINAL.search(match or "")
    if not found:
        return (match or "").strip(), 1
    return match[: found.start()].strip(), max(1, int(found.group(1)))


def annotate(devs: list[Device]) -> list[str]:
    """Unique, human-readable labels for a dropdown."""
    counts = Counter(d.name for d in devs)
    seen: dict[str, int] = {}
    labels = []
    for d in devs:
        label = d.name
        if counts[d.name] > 1:
            seen[d.name] = seen.get(d.name, 0) + 1
            label = f"{label}   #{seen[d.name]}"
        if d.default:
            label += DEFAULT_TAG
        labels.append(label)
    return labels


def is_pseudo_device(name: str) -> bool:
    low = name.lower()
    # WDM-KS surfaces raw driver strings like
    # "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0)" -- an
    # internal resource reference, not something to show anyone.
    return any(h in low for h in _PSEUDO_HINTS) or "@system32" in low


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    hostapi: str
    rate: int
    max_in: int
    max_out: int
    default: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.hostapi}]"

    @property
    def display(self) -> str:
        """Label for a dropdown: the name, marked if it is the system default."""
        return f"{self.name}{DEFAULT_TAG}" if self.default else self.name


def _all() -> list[Device]:
    apis = sd.query_hostapis()
    try:
        default_in, default_out = sd.default.device
    except Exception:  # noqa: BLE001 - no default device configured
        default_in = default_out = None
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
                default=i in (default_in, default_out),
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


def _present(devs: list[Device], all_apis: bool) -> list[Device]:
    """Trim to what a person should see, unless explicitly asked for everything."""
    if all_apis:
        return sorted(devs, key=lambda d: (d.hostapi != WASAPI, d.index))
    trimmed = [d for d in devs if d.hostapi == WASAPI and not is_pseudo_device(d.name)]
    # Some machines expose no WASAPI endpoints at all (remote sessions, unusual
    # drivers). Showing nothing would be worse than showing duplicates.
    if not trimmed:
        return sorted(devs, key=lambda d: (d.hostapi != WASAPI, d.index))
    return sorted(trimmed, key=lambda d: (not d.default, d.name.lower()))


def list_inputs(all_apis: bool = False) -> list[Device]:
    """Capture devices, deduplicated to one row per real endpoint by default."""
    return _present([d for d in _all() if d.max_in > 0], all_apis)


def list_outputs(all_apis: bool = False) -> list[Device]:
    """Playback devices, deduplicated to one row per real endpoint by default."""
    return _present([d for d in _all() if d.max_out > 0], all_apis)


def default_input() -> Device | None:
    idx = sd.default.device[0]
    return next((d for d in _all() if d.index == idx), None) if idx is not None else None


def default_output() -> Device | None:
    idx = sd.default.device[1]
    return next((d for d in _all() if d.index == idx), None) if idx is not None else None


def resolve_input(match: str, all_apis: bool = False) -> Device | None:
    if not match:
        return default_input() or next(iter(list_inputs(all_apis)), None)
    found = _first_match(list_inputs(all_apis), match)
    # A config written before the lists were trimmed -- or edited by hand -- may name
    # a device only reachable through another host API. Honour it rather than
    # silently falling back to the default microphone.
    if found is None and not all_apis:
        found = _first_match(list_inputs(all_apis=True), match)
        if found is not None:
            log.info("input %r found only via %s", match, found.hostapi)
    return found


def resolve_output(match: str, all_apis: bool = False) -> Device | None:
    if not match:
        return default_output() or next(iter(list_outputs(all_apis)), None)
    found = _first_match(list_outputs(all_apis), match)
    if found is None and not all_apis:
        found = _first_match(list_outputs(all_apis=True), match)
        if found is not None:
            log.info("output %r found only via %s", match, found.hostapi)
    return found


def _first_match(devs: list[Device], match: str) -> Device | None:
    name, ordinal = split_ordinal(match)
    needle = name.lower()
    if not needle:
        return None
    candidates = [d for d in devs if d.name.lower() == needle]
    if not candidates:
        candidates = [d for d in devs if needle in d.name.lower()]
    if not candidates:
        log.warning("no device matching %r", match)
        return None
    if ordinal > len(candidates):
        log.warning("%r asked for #%d but only %d match; using the first",
                    name, ordinal, len(candidates))
        return candidates[0]
    return candidates[ordinal - 1]


def find_cable_output() -> Device | None:
    """Deprecated: use cable.detect(), which also reports the Discord-side device.

    Kept as a thin delegate so there is exactly one detection implementation. Two
    rival ones disagreed -- this returned a VB-Audio Matrix channel by substring
    while cable.detect() reported nothing at all, so the wizard said "not
    installed" while the config silently wired up channel 8.
    """
    from . import cable

    info = cable.detect()
    if info is None:
        return None
    return _first_match(list_outputs(), info.output_name)


def cable_installed() -> bool:
    from . import cable

    return cable.installed()
