"""Make the pip-installed NVIDIA CUDA DLLs loadable by CTranslate2 on Windows.

CTranslate2 resolves cublas64_12.dll / cudnn64_9.dll by bare name at first GPU use.
The nvidia-*-cu12 wheels install them under site-packages/nvidia/*/bin, which Windows
does not search. Measured on this machine: os.add_dll_directory() does NOT fix it,
prepending to PATH does NOT fix it, and ctypes.CDLL by bare name fails under both.

What works is preloading each DLL by absolute path with ctypes.WinDLL, in dependency
order. Once a module is resident, CTranslate2's LoadLibrary by base name matches it.

prepare_cuda() must be called BEFORE importing faster_whisper.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Dependencies first. cuDNN 9 is a thin dispatcher over the cudnn_* backends, so those
# are pulled in explicitly rather than left to lazy resolution mid-inference.
_PRELOAD_ORDER = (
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
)

# Without these two, CUDA inference cannot work at all.
_REQUIRED = ("cublas64_12.dll", "cudnn64_9.dll")

_handles: list[ctypes.CDLL] = []  # keep references alive for the process lifetime
_result: bool | None = None


def reset() -> None:
    """Forget the cached probe result so the next call re-scans.

    Needed after the GPU pack is downloaded at runtime: without this, the negative
    result cached at startup would persist and CUDA would stay "unavailable" until
    the whole app was restarted, even though the libraries are now on disk.
    """
    global _result
    _result = None


def _search_dirs() -> list[Path]:
    """Candidate DLL directories, covering venv, frozen and downloaded-pack layouts.

    Two different depths are in play, which is easy to get wrong:

        site-packages/nvidia/cublas/bin/...    <- root is "nvidia", so */bin
        %LOCALAPPDATA%/Voice2TTS/cuda/nvidia/cublas/bin/...  <- root is "cuda", so */*/bin

    gpupack.py preserves the nvidia/<pkg>/bin layout inside its own directory, so a
    single-level glob silently misses the downloaded pack -- and in a packaged
    install that is the ONLY place these libraries exist.
    """
    from .paths import cuda_dir

    # The downloaded GPU pack comes first: in a packaged install it is the only
    # place these DLLs exist, since the installer ships without CUDA.
    roots = [cuda_dir(), Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        roots += [base / "nvidia", base]

    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        candidates = []
        if any(root.glob("*.dll")):
            candidates.append(root)
        for pattern in ("*/bin", "*/*/bin"):
            candidates.extend(sorted(root.glob(pattern)))
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen and any(path.glob("*.dll")):
                seen.add(resolved)
                dirs.append(path)
    return dirs


def prepare_cuda() -> bool:
    """Preload CUDA libraries. Returns True if CUDA looks usable. Idempotent."""
    global _result
    if _result is not None:
        return _result

    dirs = _search_dirs()
    if not dirs:
        log.warning("no NVIDIA DLL directories found; CUDA unavailable")
        _result = False
        return _result

    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except OSError:  # non-existent or already-removed directory
            pass

    index: dict[str, Path] = {}
    for d in dirs:
        for dll in d.glob("*.dll"):
            index.setdefault(dll.name, dll)

    missing: list[str] = []
    for name in _PRELOAD_ORDER:
        path = index.get(name)
        if path is None:
            missing.append(name)
            continue
        try:
            _handles.append(ctypes.WinDLL(str(path)))
            log.debug("preloaded %s", path)
        except OSError as exc:
            log.warning("failed to preload %s: %s", name, exc)
            missing.append(name)

    _result = not any(r in missing for r in _REQUIRED)
    if not _result:
        log.warning("CUDA libraries missing: %s", ", ".join(missing))
    return _result


def cuda_available() -> bool:
    """True if a CUDA device is present AND its libraries can be loaded."""
    if not prepare_cuda():
        return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable CUDA"
        log.warning("CUDA probe failed: %s", exc)
        return False
