"""Make the pip-installed NVIDIA CUDA DLLs visible to CTranslate2 on Windows.

faster-whisper's backend (CTranslate2) resolves cublas64_12.dll / cudnn64_9.dll by
bare name at first GPU use. The pip wheels put them under site-packages/nvidia/*/bin,
which Windows will not search: since Python 3.8 neither PATH nor, in CTranslate2's
case, os.add_dll_directory() is enough -- both were verified to fail here.

What does work is preloading each DLL by absolute path. Once a module is in the
process, a later LoadLibrary("cublas64_12.dll") matches it by base name and
succeeds. Load order matters because these depend on each other.

Call prepare_cuda() BEFORE importing faster_whisper.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Dependencies first. cuDNN 9 is a thin dispatcher over the cudnn_* backends, so
# those get pulled in too rather than left to lazy resolution at inference time.
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

_handles: list[ctypes.CDLL] = []  # keep alive for the process lifetime
_prepared = False


def nvidia_bin_dirs() -> list[Path]:
    root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    return sorted(p for p in root.glob("*/bin") if any(p.glob("*.dll")))


def prepare_cuda(verbose: bool = False) -> bool:
    """Return True if the core CUDA libs are loadable. Safe to call repeatedly."""
    global _prepared
    if _prepared:
        return True

    dirs = nvidia_bin_dirs()
    if not dirs:
        if verbose:
            print("no site-packages/nvidia/*/bin found")
        return False

    for d in dirs:
        os.add_dll_directory(str(d))

    index = {p.name: p for d in dirs for p in d.glob("*.dll")}
    ok = True
    for name in _PRELOAD_ORDER:
        path = index.get(name)
        if path is None:
            if verbose:
                print(f"  {name}: not present")
            continue
        try:
            _handles.append(ctypes.WinDLL(str(path)))
            if verbose:
                print(f"  {name}: loaded")
        except OSError as exc:
            ok = ok and name not in ("cublas64_12.dll", "cudnn64_9.dll")
            if verbose:
                print(f"  {name}: FAILED {exc}")

    _prepared = ok
    return ok


if __name__ == "__main__":
    print("cuda prepared:", prepare_cuda(verbose=True))
