"""The Japanese phonemizer, downloaded on demand.

Japanese Piper voices are trained on OpenJTalk phonemes, and Piper imports
`pyopenjtalk` to produce them. It does not declare that as a dependency, so
without it every Japanese utterance raises ModuleNotFoundError from inside
synthesis -- which reaches the user as "Failed to process utterance" and no
speech at all.

WHY IT IS NOT SIMPLY BUNDLED
    ~100 MB of wheels, ~330 MB unpacked, for one language. That is a lot to add
    to every installer for something most people will never select. The GPU pack
    established the pattern: pay for it when you ask for it.

WHAT IS IN IT, AND WHY EACH PIECE
    pyopenjtalk-plus   the phonemizer itself. The original `pyopenjtalk` has no
                       Windows wheels at all -- it builds C++ from source -- so
                       this maintained fork, which publishes them and provides
                       the same module name, is what makes this possible.
    sudachipy          imported at pyopenjtalk's module level; without it the
                       import fails outright.
    sudachidict_core   NOT optional, though it is most of the download.
                       pyopenjtalk asks for it lazily, to settle kanji with more
                       than one reading. Measured: eight ordinary sentences, and
                       the one containing 人 (hitori / futari / nin) was the one
                       that needed it -- Piper caught the error and produced
                       0.0 s of audio, so the sentence was silently not spoken.
                       `dictionary.Dictionary()` defaults to `core` and
                       pyopenjtalk passes no argument, so the 42 MB `small`
                       dictionary is never even looked for.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import probe
from .net import USER_AGENT, ByteProgress, Json, StepProgress
from .paths import japanese_dir

log = logging.getLogger(__name__)

PYPI_JSON = "https://pypi.org/pypi/{package}/json"

# Pinned to what was verified working with the bundled Piper. An empty version
# means "newest", which is right for the dictionary: it is data, released on its
# own schedule, and any recent one settles the same readings.
PACKAGES = (
    ("pyopenjtalk-plus", "0.4.1.post9"),
    ("sudachipy", "0.6.11"),
    ("sudachidict_core", ""),
)

# The module Piper actually imports, which is what decides whether this worked.
PROBE_MODULE = "pyopenjtalk"

MAX_WHEEL_BYTES = 500_000_000
APPROX_DOWNLOAD_MB = 100
APPROX_INSTALLED_MB = 330


@dataclass
class PackStatus:
    installed: bool
    size_mb: float
    # Empty when the phonemizer loads. A pack can unpack perfectly and still be
    # unusable -- a wheel built for the wrong Python ABI is the case that
    # actually happened -- and `installed` cannot tell the difference.
    problem: str = ""

    @property
    def usable(self) -> bool:
        return self.installed and not self.problem


def status() -> PackStatus:
    """What is on disk, and whether it works.

    `usable` requires the phonemizer to IMPORT, not merely to be unpacked. The
    directory check alone reported a broken pack as installed, and the failure
    surfaced one utterance at a time as "Failed to process utterance".
    """
    root = japanese_dir()
    if not (root / PROBE_MODULE).is_dir():
        return PackStatus(False, 0.0, probe.MISSING)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    activate()
    trouble = probe.import_problem(PROBE_MODULE)
    return PackStatus(installed=True, size_mb=round(total / 1e6, 1),
                      problem="" if trouble is None else trouble)


def activate() -> bool:
    """Put the pack on the import path. Safe to call repeatedly.

    Called at startup rather than at first use: `importlib.util.find_spec`
    searches sys.path, so anything asking "can this build speak Japanese?"
    before this ran would be told no.
    """
    root = japanese_dir()
    if not (root / PROBE_MODULE).is_dir():
        return False
    entry = str(root)
    if entry not in sys.path:
        # Appended, not prepended: a pack should never be able to shadow a
        # module the application itself ships.
        sys.path.append(entry)
    return True


def _wheel_url(package: str, version: str,
               timeout: float = 20.0) -> tuple[str, int, str]:
    """The Windows wheel for a package, or the pure-python one if that is all
    there is -- the dictionary is data and ships as py3-none-any."""
    request = urllib.request.Request(PYPI_JSON.format(package=package),
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    # A compiled wheel is built against one Python ABI. PyPI lists cp310 first
    # for these packages, and installing that under 3.12 gives a pack that
    # unpacks perfectly and then cannot import -- so the running interpreter's
    # tag is what decides.
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    def pick(entries: list[Json]) -> Json | None:
        for entry in entries:
            name = entry.get("filename", "")
            if name.endswith("win_amd64.whl") and tag in name:
                return entry
        # Pure-python wheels carry no ABI at all: the dictionary is data.
        for entry in entries:
            if entry.get("filename", "").endswith("-none-any.whl"):
                return entry
        return None

    if version:
        found = pick(data.get("releases", {}).get(version) or [])
        if found:
            return found["url"], int(found.get("size") or 0), found["filename"]
        log.warning("%s %s not on PyPI; falling back to the newest",
                    package, version)
    found = pick(data.get("urls", []))
    if found:
        return found["url"], int(found.get("size") or 0), found["filename"]
    raise RuntimeError(
        f"no {tag} Windows wheel found for {package}. This build of Python "
        "is newer than anything the phonemizer publishes.")


def _download(url: str, dest: Path,
              progress: ByteProgress = None,
              timeout: float = 180.0) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        if total > MAX_WHEEL_BYTES:
            raise RuntimeError(f"wheel is implausibly large ({total} bytes)")
        got = 0
        with dest.open("wb") as fh:
            while chunk := response.read(1024 * 256):
                got += len(chunk)
                fh.write(chunk)
                if progress and total:
                    progress(got, total)
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest.name} is not a valid wheel")
    return dest


def _extract(wheel: Path, dest_root: Path) -> int:
    """Unpack a wheel into `dest_root`, which is what pip --target does.

    Everything is kept, not just the DLLs the GPU pack wants: these are Python
    packages, and the compiled extensions sit inside them at paths that have to
    survive.
    """
    count = 0
    with zipfile.ZipFile(wheel) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            if ".." in parts or member.startswith("/"):
                raise RuntimeError(f"unsafe path in wheel: {member}")
            target = dest_root / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def install(progress: StepProgress = None) -> PackStatus:
    """Download and unpack the phonemizer. Returns the resulting status."""

    def say(message: str) -> None:
        log.info(message)
        if progress:
            progress(message)

    dest = japanese_dir()
    dest.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="voice2tts-jp-"))
    try:
        for package, version in PACKAGES:
            say(f"Locating {package}...")
            url, size, filename = _wheel_url(package, version)
            megabytes = size / 1e6 if size else 0
            say(f"Downloading {package} ({megabytes:.0f} MB)..." if megabytes
                else f"Downloading {package}...")
            wheel = _download(url, workdir / filename)
            say(f"Unpacking {package}...")
            _extract(wheel, dest)
            wheel.unlink(missing_ok=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    activate()
    result = status()
    if not result.usable:
        raise RuntimeError("Japanese pack incomplete: the phonemizer is missing")

    # Prove it imports before claiming success. A pack that unpacks but cannot
    # load is worse than one that failed to download, because nothing says so
    # until someone tries to speak.
    try:
        import importlib

        importlib.invalidate_caches()
        importlib.import_module(PROBE_MODULE)
    except Exception as exc:
        raise RuntimeError(
            f"Japanese pack installed but {PROBE_MODULE} will not load: {exc}"
        ) from exc

    say(f"Japanese voices ready ({result.size_mb:.0f} MB).")
    return result


def uninstall() -> bool:
    root = japanese_dir()
    if not root.is_dir():
        return False
    shutil.rmtree(root, ignore_errors=True)
    entry = str(root)
    while entry in sys.path:
        sys.path.remove(entry)
    # The phonemizer was imported to prove the pack worked. Left in sys.modules
    # it would keep importing, so the pack would report itself usable for the
    # rest of the session -- after being deleted.
    probe.forget()
    log.info("removed the Japanese pack")
    return not root.is_dir()
