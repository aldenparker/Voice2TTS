"""On-demand training environment for the Voice Studio.

Fine-tuning needs PyTorch and friends -- around 3 GB that most people will never
want. Like the GPU pack, it is downloaded only if asked for.

Unlike the GPU pack, this is NOT loaded into the running application. It is a
separate, self-contained Python interpreter under %LOCALAPPDATA%, and training
runs as a subprocess against it. Three reasons:

  * A frozen PyInstaller build cannot pip-install into itself.
  * torch ships its own copies of cuBLAS and cuDNN. Our CUDA pack preloads those
    by absolute path (see cuda.py), and a second set resident in the same process
    is exactly the ambiguity that took two attempts to get right the first time.
  * `piper.train` is a command-line trainer, so a subprocess is its natural shape.
    Uninstalling is then deleting one directory.

The hardware gate is advisory. Under-spec hardware fails by running out of memory,
which costs the time since the last checkpoint and nothing else, and a smaller card
can plausibly train at a reduced batch size. Refusing outright would be wrong, so
there is an override -- recorded in diagnostics, so a bug report says it was used.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .paths import cache_dir

log = logging.getLogger(__name__)

USER_AGENT = "Voice2TTS/0.6"

# Matches the interpreter the application itself is built with, so the Piper
# training code sees the Python it expects.
PYTHON_VERSION = "3.12.10"
PYTHON_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}"
    f"/python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# cu128 is the first CUDA build line with Blackwell (sm_120) kernels, which the
# 50-series needs. Pinned rather than floating: a silent torch major bump would
# break training runs that were working yesterday.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_SPEC = "torch==2.9.1+cu128"
TORCH_CPU_SPEC = "torch==2.9.1"

# Pinned for the same reason as torch: training.py builds a command line against
# this exact version's LightningCLI arguments, and an upstream rename would turn
# into a failure hours into a run rather than at install time.
TRAINING_PACKAGES = (
    "piper-tts[train]==1.7.0",
    "torchaudio",
    "librosa",
)

# What the gate wants. Both are advisory; see the module docstring.
MIN_VRAM_GB = 8.0
MIN_DISK_GB = 25.0
APPROX_DOWNLOAD_GB = 3.0


@dataclass
class Hardware:
    gpu_name: str = ""
    vram_gb: float = 0.0
    free_disk_gb: float = 0.0
    cuda_pack_installed: bool = False

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_name)


@dataclass
class GateResult:
    ok: bool
    hardware: Hardware
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overridden: bool = False

    @property
    def summary(self) -> str:
        if self.ok and not self.blockers:
            return "This machine can train a voice."
        if self.overridden:
            return "Hardware check overridden: " + "; ".join(self.blockers)
        return "; ".join(self.blockers)


def _nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except Exception as exc:  # noqa: BLE001 - no NVIDIA tooling is a normal state
        log.debug("nvidia-smi %s failed: %s", query, exc)
        return ""


def probe() -> Hardware:
    """What this machine has. Never raises."""
    from .gpupack import status as gpu_status

    hw = Hardware()
    hw.gpu_name = _nvidia_smi("name")
    total = _nvidia_smi("memory.total")
    if total:
        try:
            hw.vram_gb = round(float(total) / 1024.0, 1)
        except ValueError:
            pass
    try:
        hw.free_disk_gb = round(shutil.disk_usage(cache_dir()).free / 1e9, 1)
    except OSError as exc:
        log.debug("disk check failed: %s", exc)
    hw.cuda_pack_installed = gpu_status().usable
    return hw


def gate(override: bool = False, hardware: Hardware | None = None) -> GateResult:
    """Decide whether training should be offered. Advisory unless `override`."""
    hw = hardware or probe()
    blockers: list[str] = []
    warnings: list[str] = []

    if not hw.has_gpu:
        blockers.append(
            "No NVIDIA GPU found. Training on a CPU is possible in principle but "
            "would take days rather than hours."
        )
    elif hw.vram_gb < MIN_VRAM_GB:
        blockers.append(
            f"{hw.gpu_name} has {hw.vram_gb:.0f} GB of memory; training is "
            f"comfortable from {MIN_VRAM_GB:.0f} GB. It may still work at a smaller "
            "batch size."
        )

    if hw.free_disk_gb < MIN_DISK_GB:
        blockers.append(
            f"{hw.free_disk_gb:.0f} GB free on the cache drive; training needs "
            f"about {MIN_DISK_GB:.0f} GB for the environment, dataset and "
            "checkpoints."
        )

    if not hw.cuda_pack_installed and hw.has_gpu:
        warnings.append(
            "GPU acceleration is not installed. Training brings its own CUDA "
            "libraries, so this is not required, but recognition will stay on the "
            "CPU until you add it."
        )

    return GateResult(
        ok=not blockers or override,
        hardware=hw,
        blockers=blockers,
        warnings=warnings,
        overridden=bool(blockers) and override,
    )


# -- the environment --------------------------------------------------------


def studio_dir() -> Path:
    return cache_dir() / "studio"


def python_exe() -> Path:
    return studio_dir() / "python" / "python.exe"


@dataclass
class PackStatus:
    installed: bool
    python_ready: bool
    torch_ready: bool
    size_gb: float
    torch_version: str = ""
    cuda_available: bool | None = None

    @property
    def usable(self) -> bool:
        return self.python_ready and self.torch_ready


def status(deep: bool = False) -> PackStatus:
    """What is installed. `deep` runs the interpreter, which costs a second."""
    root = studio_dir()
    if not root.is_dir():
        return PackStatus(False, False, False, 0.0)

    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e9
    python_ready = python_exe().exists()
    torch_ready = any(root.rglob("torch/__init__.py"))

    result = PackStatus(True, python_ready, torch_ready, round(size, 2))
    if deep and python_ready and torch_ready:
        result.torch_version, result.cuda_available = _query_torch()
    return result


def _query_torch() -> tuple[str, bool | None]:
    """Ask the studio interpreter what it has. Returns ("", None) on failure."""
    script = (
        "import torch,sys;"
        "sys.stdout.write(torch.__version__+'|'+str(torch.cuda.is_available()))"
    )
    try:
        out = subprocess.run(
            [str(python_exe()), "-c", script],
            capture_output=True, text=True, timeout=120,
            creationflags=0x08000000,
        )
        version, _, cuda = out.stdout.strip().partition("|")
        return version, cuda == "True"
    except Exception as exc:  # noqa: BLE001
        log.debug("torch probe failed: %s", exc)
        return "", None


def _download(url: str, dest: Path, progress=None, timeout: float = 120.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            got += len(chunk)
            fh.write(chunk)
            if progress and total:
                progress(got, total)
    return dest


def _prepare_interpreter(root: Path, progress=None) -> Path:
    """Unpack the embeddable Python and make it able to import site-packages."""
    python_dir = root / "python"
    if (python_dir / "python.exe").exists():
        return python_dir

    archive = root / "python-embed.zip"
    if progress:
        progress(f"Downloading Python {PYTHON_VERSION}...")
    _download(PYTHON_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(python_dir)
    archive.unlink(missing_ok=True)

    # The embeddable build ships isolated: site is disabled and Lib\site-packages
    # is absent from the path, so pip installs would be invisible. Enabling
    # "import site" in the ._pth is the documented way to un-isolate it.
    for pth in python_dir.glob("python*._pth"):
        text = pth.read_text(encoding="utf-8")
        if "#import site" in text:
            pth.write_text(text.replace("#import site", "import site"),
                           encoding="utf-8")
        if "Lib\\site-packages" not in text:
            pth.write_text(pth.read_text(encoding="utf-8").rstrip()
                           + "\nLib\\site-packages\n", encoding="utf-8")
    return python_dir


def _bootstrap_pip(python_dir: Path, progress=None) -> None:
    if (python_dir / "Scripts" / "pip.exe").exists():
        return
    if progress:
        progress("Installing pip...")
    get_pip = python_dir / "get-pip.py"
    _download(GET_PIP_URL, get_pip)
    _run([str(python_dir / "python.exe"), str(get_pip), "--no-warn-script-location"])
    get_pip.unlink(missing_ok=True)


def _run(args: list[str], progress=None) -> None:
    """Run a subprocess, streaming its output to `progress`."""
    log.info("studio: %s", " ".join(args[:4]))
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=0x08000000,
    )
    tail: list[str] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-40]
        if progress and (line.startswith(("Collecting", "Downloading", "Installing",
                                          "Successfully"))):
            progress(line[:110])
    if proc.wait() != 0:
        raise RuntimeError(
            "Command failed:\n" + "\n".join(tail[-12:] or ["(no output)"])
        )


def install(progress=None, use_gpu: bool = True) -> PackStatus:
    """Create the training environment. Several GB and several minutes."""

    def say(msg: str) -> None:
        log.info("studio: %s", msg)
        if progress:
            progress(msg)

    root = studio_dir()
    root.mkdir(parents=True, exist_ok=True)

    python_dir = _prepare_interpreter(root, say)
    _bootstrap_pip(python_dir, say)
    python = str(python_dir / "python.exe")

    say("Installing PyTorch (this is the large one)...")
    if use_gpu:
        _run([python, "-m", "pip", "install", "--no-warn-script-location",
              TORCH_SPEC, "--index-url", TORCH_INDEX], say)
    else:
        _run([python, "-m", "pip", "install", "--no-warn-script-location",
              TORCH_CPU_SPEC], say)

    say("Installing the Piper trainer...")
    _run([python, "-m", "pip", "install", "--no-warn-script-location",
          *TRAINING_PACKAGES], say)

    result = status(deep=True)
    if not result.usable:
        raise RuntimeError(
            "Studio environment incomplete: "
            f"python={'ok' if result.python_ready else 'missing'}, "
            f"torch={'ok' if result.torch_ready else 'missing'}"
        )
    say(f"Ready. torch {result.torch_version}, "
        f"CUDA {'available' if result.cuda_available else 'not available'}, "
        f"{result.size_gb:.1f} GB.")
    return result


def uninstall() -> bool:
    """Delete the training environment, freeing several GB."""
    root = studio_dir()
    if not root.is_dir():
        return False
    shutil.rmtree(root, ignore_errors=True)
    log.info("removed studio pack at %s", root)
    return not root.exists()


def environment_for_training() -> dict[str, str]:
    """Environment for a training subprocess.

    The studio interpreter must not inherit our PYTHONPATH or PYTHONHOME, or it
    would import the application's packages instead of its own.
    """
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    return env
