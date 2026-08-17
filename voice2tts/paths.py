"""Filesystem locations, resolved the same way whether running from source or frozen."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def resource_root() -> Path:
    """Directory holding bundled read-only assets (models, voices)."""
    if is_frozen():
        # PyInstaller onedir/onefile both expose _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Per-user writable directory for config, logs and downloaded voices."""
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    d = Path(base) / "Voice2TTS"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Large, regenerable data: CUDA libraries, Whisper weights.

    Deliberately LOCALAPPDATA rather than APPDATA -- the GPU pack alone is ~1.3 GB,
    and a roaming profile would try to sync all of it.
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    d = Path(base) / "Voice2TTS"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cuda_dir() -> Path:
    return cache_dir() / "cuda"


def whisper_cache() -> Path:
    d = cache_dir() / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bundled_whisper(name: str) -> Path | None:
    """A Whisper model shipped inside the build, if this one was bundled."""
    d = resource_root() / "models" / "whisper" / name
    return d if (d / "model.bin").exists() else None


def config_path() -> Path:
    return user_data_dir() / "config.toml"


def log_path() -> Path:
    return user_data_dir() / "voice2tts.log"


def voices_dirs() -> list[Path]:
    """Where to look for Piper voices: user dir first so it can shadow bundled ones."""
    return [user_data_dir() / "voices", resource_root() / "models" / "voices"]


def vad_model_path() -> Path:
    return resource_root() / "models" / "vad" / "silero_vad.onnx"


def find_voice(name: str) -> Path | None:
    """Resolve a voice by bare name ('en_US-lessac-medium'), stem, or absolute path."""
    if not name:
        return None
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for d in voices_dirs():
        for cand in (d / name, d / f"{name}.onnx"):
            if cand.exists():
                return cand
    return None


def list_voices() -> list[Path]:
    seen: dict[str, Path] = {}
    for d in voices_dirs():
        if not d.is_dir():
            continue
        for onnx in sorted(d.glob("*.onnx")):
            seen.setdefault(onnx.stem, onnx)
    return list(seen.values())
