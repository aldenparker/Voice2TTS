"""A one-click support report.

Nearly every "it doesn't work" needs the same facts: version, devices, whether the
cable and GPU pack are present, and the tail of the log. Gathering them by hand
means talking someone through AppData, so this assembles them into text that can be
pasted into an issue.

Deliberately excludes anything sensitive: no transcripts, no audio, no tokens, and
the user name is stripped from paths.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from . import __version__

LOG_TAIL_LINES = 60


def _redact(text: str) -> str:
    """Replace the Windows user name with a placeholder.

    Paths are useful for debugging; the person's name is not our business to paste
    into a public issue tracker.
    """
    user = os.environ.get("USERNAME") or ""
    if user and len(user) > 2:
        text = text.replace(user, "<user>")
    home = str(Path.home())
    return text.replace(home, "%USERPROFILE%")


def diagnostics(cfg=None, pipeline=None) -> str:
    from . import cable, devices, gpupack, plan, voices
    from .cuda import cuda_available
    from .paths import cache_dir, config_path, is_frozen, log_path

    out: list[str] = []

    def section(title: str) -> None:
        out.append("")
        out.append(f"--- {title} ---")

    out.append(f"Voice2TTS {__version__}")
    out.append(f"Build: {'installed' if is_frozen() else 'source checkout'}")
    out.append(f"Windows: {platform.platform()}")
    out.append(f"Python: {sys.version.split()[0]}")

    section("Capabilities")
    pack = gpupack.status()
    out.append(f"NVIDIA GPU present : {gpupack.gpu_present()}")
    out.append(f"CUDA usable        : {cuda_available()}")
    out.append(f"GPU pack           : {pack.dll_count} libs, {pack.size_mb:.0f} MB")

    from . import studiopack

    studio = studiopack.status()
    if studio.installed:
        out.append(f"Studio pack        : {studio.size_gb:.1f} GB, "
                   f"python={'ok' if studio.python_ready else 'missing'}, "
                   f"torch={'ok' if studio.torch_ready else 'missing'}")
    else:
        out.append("Studio pack        : not installed")
    if cfg is not None and cfg.studio.ignore_hardware_check:
        # Surfaced deliberately: a training bug report reads very differently if
        # the machine was below the recommended specification.
        out.append("Studio hardware    : CHECK OVERRIDDEN by the user")
    found = cable.detect()
    out.append(f"Virtual cable      : {found.product if found else 'none'}")
    if found:
        out.append(f"  output           : {found.output_name}")
        out.append(f"  Discord input    : {found.input_name}")

    section("Configuration")
    if cfg is not None:
        # The plan, not a fresh guess at it. This section used to re-derive the
        # languages and got it wrong the same way the settings window once did:
        # it reported "voice/model language mismatch" without ever mentioning
        # that translation was on, so every bug report from a translate user
        # arrived with a red herring at the top.
        out.extend(plan.build(cfg).describe())
        out.append(f"hotkey        : {cfg.trigger.hotkey}")
        out.append(f"input         : {cfg.audio.input_match or '(system default)'}")
        for t in cfg.audio.outputs:
            state = "on " if t.enabled else "off"
            out.append(f"output [{state}] : {t.label}  gain={t.gain}")
        out.append(f"whisper model : {cfg.stt.model} "
                   f"({cfg.stt.device.value}/{cfg.stt.compute_type.value})")
        out.append(f"vad threshold : {cfg.vad.threshold}")
        out.append(f"vad silence   : {cfg.vad.min_silence_ms} ms")
        out.append(f"mute on play  : {cfg.audio.mute_mic_during_playback}")
        out.append(f"update repo   : {cfg.updates.repo or '(unset)'}")
        for repair in cfg.repairs:
            out.append(f"REPAIRED: {repair}")

    section("Runtime")
    if pipeline is not None:
        try:
            for key, value in pipeline.status().items():
                out.append(f"{key:12}: {value}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"(status unavailable: {exc})")

    section("CPU right now")
    # Sampled rather than described, because "the app feels heavy" is not
    # something anyone can act on, and Task Manager stops at the process
    # boundary -- it cannot say WHICH thread.
    from .perf import sample

    for line in sample(1.0).report():
        out.append(line)

    section("Installed voices")
    out.extend(f"  {k}" for k in voices.installed_keys())

    section("Audio devices")
    # all_apis: a support report should show everything, including the host APIs the
    # pickers hide, since "my device is missing" is a common reason to file one.
    try:
        for d in devices.list_inputs(all_apis=True):
            out.append(f"  IN   {d.rate:>6} Hz  {d.hostapi:<18} {d.name}")
        for d in devices.list_outputs(all_apis=True):
            out.append(f"  OUT  {d.rate:>6} Hz  {d.hostapi:<18} {d.name}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"  (enumeration failed: {exc})")

    section("Paths")
    out.append(f"config : {config_path()}")
    out.append(f"cache  : {cache_dir()}")
    out.append(f"log    : {log_path()}")

    section(f"Log (last {LOG_TAIL_LINES} lines)")
    try:
        lines = log_path().read_text(encoding="utf-8", errors="replace").splitlines()
        out.extend(lines[-LOG_TAIL_LINES:])
    except OSError as exc:
        out.append(f"(could not read log: {exc})")

    return _redact("\n".join(out))
