"""On-demand GPU acceleration download.

The installer ships without CUDA because the runtime is ~1.3 GB (cublasLt64_12.dll
alone is 637 MB) and most of that is dead weight on a machine with no NVIDIA GPU.
Instead the app installs with a CPU Whisper model that works immediately, and this
module fetches the GPU pieces afterwards if the user wants them.

The DLLs come from NVIDIA's own wheels on PyPI, downloaded at runtime and unpacked
into %LOCALAPPDATA%\\Voice2TTS\\cuda. Nothing is hosted or redistributed by us; a
wheel is just a zip, and cuda.py already searches that directory first.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .net import USER_AGENT
from .paths import cuda_dir, whisper_cache

log = logging.getLogger(__name__)

PYPI_JSON = "https://pypi.org/pypi/{package}/json"

# Version-pinned to what was verified working with ctranslate2 4.8.1 on Blackwell.
PACKAGES = (
    ("nvidia-cublas-cu12", "12.9.2.10"),
    ("nvidia-cudnn-cu12", "9.24.0.43"),
)

# Model pulled once the GPU is available -- bigger than the bundled CPU default.
GPU_WHISPER_MODEL = "small.en"

MAX_WHEEL_BYTES = 1_500_000_000


@dataclass
class PackStatus:
    installed: bool
    dll_count: int
    size_mb: float
    has_cublas: bool
    has_cudnn: bool

    @property
    def usable(self) -> bool:
        return self.has_cublas and self.has_cudnn


def status() -> PackStatus:
    root = cuda_dir()
    if not root.is_dir():
        return PackStatus(False, 0, 0.0, False, False)
    dlls = list(root.rglob("*.dll"))
    names = {p.name.lower() for p in dlls}
    total = sum(p.stat().st_size for p in dlls) / 1e6
    return PackStatus(
        installed=bool(dlls),
        dll_count=len(dlls),
        size_mb=round(total, 1),
        has_cublas="cublas64_12.dll" in names,
        has_cudnn="cudnn64_9.dll" in names,
    )


def gpu_present() -> bool:
    """True if an NVIDIA GPU is visible, regardless of whether CUDA libs exist yet.

    Checked without importing ctranslate2, because that is exactly what fails when
    the pack is missing.
    """
    import ctypes

    for lib in ("nvcuda.dll", "nvml.dll"):
        try:
            ctypes.WinDLL(lib)
            return True
        except OSError:
            continue
    return False


def _wheel_url(package: str, version: str, timeout: float = 20.0) -> tuple[str, int]:
    """Find the win_amd64 wheel for a pinned version on PyPI."""
    req = urllib.request.Request(
        PYPI_JSON.format(package=package), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    releases = data.get("releases", {}).get(version) or []
    for entry in releases:
        name = entry.get("filename", "")
        if name.endswith("win_amd64.whl"):
            return entry["url"], int(entry.get("size") or 0)

    # Pin missing (yanked, or PyPI reorganised) -- fall back to the newest win wheel.
    newest = data.get("urls", [])
    for entry in newest:
        if entry.get("filename", "").endswith("win_amd64.whl"):
            log.warning("%s %s not found; using %s", package, version, entry["filename"])
            return entry["url"], int(entry.get("size") or 0)
    raise RuntimeError(f"no Windows wheel found for {package}")


def _download(url: str, dest: Path, progress=None, timeout: float = 120.0) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total > MAX_WHEEL_BYTES:
            raise RuntimeError(f"wheel is implausibly large ({total} bytes)")
        got = 0
        with dest.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                got += len(chunk)
                fh.write(chunk)
                if progress and total:
                    progress(got, total)
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest.name} is not a valid wheel")
    return dest


def _extract_dlls(wheel: Path, dest_root: Path) -> int:
    """Pull nvidia/*/bin/*.dll out of a wheel, preserving the nvidia/<pkg>/bin layout."""
    count = 0
    with zipfile.ZipFile(wheel) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".dll"):
                continue
            parts = Path(member).parts
            if ".." in parts or member.startswith("/"):
                raise RuntimeError(f"unsafe path in wheel: {member}")
            if "nvidia" not in parts:
                continue
            # Keep everything from "nvidia/" onward so cuda.py's glob still matches.
            rel = Path(*parts[parts.index("nvidia"):])
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def install(progress=None, fetch_model: bool = True) -> PackStatus:
    """Download and unpack the CUDA runtime. Returns the resulting status."""

    def say(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    dest = cuda_dir()
    dest.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="voice2tts-gpu-"))
    try:
        for package, version in PACKAGES:
            say(f"Locating {package} {version}...")
            url, size = _wheel_url(package, version)
            mb = size / 1e6 if size else 0
            say(f"Downloading {package} ({mb:.0f} MB)..." if mb else f"Downloading {package}...")
            wheel = _download(url, workdir / f"{package}.whl")
            say(f"Extracting {package}...")
            n = _extract_dlls(wheel, dest)
            say(f"  {n} libraries installed")
            wheel.unlink(missing_ok=True)  # 600 MB each; do not keep them around
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    result = status()
    if not result.usable:
        raise RuntimeError(
            "GPU pack incomplete: "
            f"cublas={'ok' if result.has_cublas else 'missing'}, "
            f"cudnn={'ok' if result.has_cudnn else 'missing'}"
        )

    # Startup cached "no CUDA"; the libraries exist now, so let the next probe see
    # them. Otherwise restarting the pipeline still falls back to CPU.
    from .cuda import reset as reset_cuda

    reset_cuda()

    if fetch_model:
        say(f"Downloading the {GPU_WHISPER_MODEL} model...")
        download_whisper_model(GPU_WHISPER_MODEL)

    say(f"GPU acceleration ready ({result.size_mb:.0f} MB).")
    return result


def download_whisper_model(name: str) -> None:
    """Pre-fetch CTranslate2 weights into our own cache, not the default HF one."""
    from faster_whisper import WhisperModel

    # Loaded on CPU purely to force the download; the real run picks its own device.
    WhisperModel(name, device="cpu", compute_type="int8", download_root=str(whisper_cache()))


def uninstall() -> bool:
    """Remove the downloaded CUDA libraries, freeing ~1.3 GB.

    Already-loaded DLLs stay resident until the process exits, so a running app keeps
    working on GPU until restarted -- deliberate, since yanking them mid-utterance
    would be worse.
    """
    root = cuda_dir()
    if not root.is_dir():
        return False
    shutil.rmtree(root, ignore_errors=True)
    from .cuda import reset as reset_cuda

    reset_cuda()
    log.info("removed GPU pack at %s", root)
    return not root.exists()
