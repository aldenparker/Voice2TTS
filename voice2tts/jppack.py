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
    pydantic           and its own dependencies. pyopenjtalk-plus declares it
                       and nothing here noticed for months, because the machine
                       this was written on already had one. See WANTED: the
                       download list is now READ from each wheel's metadata
                       rather than typed out, because a typed list is only ever
                       correct on the machine it was typed on.
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
PYPI_VERSION_JSON = "https://pypi.org/pypi/{package}/{version}/json"

# What to ask for. Pinned to what was verified working with the bundled Piper;
# an empty version means "newest", which is right for the dictionary -- it is
# data, released on its own schedule, and any recent one settles the same
# readings.
#
# What these NEED is not written here. It used to be: this tuple was the whole
# download list, and pyopenjtalk-plus depends on pydantic, which nobody had
# noticed because the development machine already had one. The pack unpacked
# perfectly and then would not import, on every machine but the author's.
# Dependencies are read from each wheel's own metadata by _resolve().
WANTED = (
    ("pyopenjtalk-plus", "0.4.1.post9"),
    ("sudachipy", "0.6.11"),
    ("sudachidict_core", ""),
)

# Shipped inside the application, so the pack must not carry a second copy:
# activate() appends to sys.path, so ours wins anyway and the download would be
# pure weight.
#
# Declared, NOT probed with find_spec. Resolving against whatever happens to be
# importable is precisely how a pack that worked in a development venv shipped
# without pydantic -- the venv had it and the installer did not. This list is
# the same on every machine, which is the point.
PROVIDED = frozenset({"numpy"})

# A runaway dependency graph is a download nobody asked for. Nothing here needs
# more than about eight packages; well past that is a bug in the resolver.
MAX_PACKAGES = 24

# The module Piper actually imports, which is what decides whether this worked.
PROBE_MODULE = "pyopenjtalk"

MAX_WHEEL_BYTES = 500_000_000
# Measured on a real install, not estimated: 105 MB of wheels, 340 MB unpacked.
APPROX_DOWNLOAD_MB = 105
APPROX_INSTALLED_MB = 340


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


def _metadata(package: str, version: str = "", timeout: float = 20.0) -> Json:
    """PyPI's record for one package, at one version if given.

    Split out so the resolver and the wheel picker read the same thing, and so
    the tests can hand both of them a fixture instead of the network.
    """
    url = (PYPI_JSON.format(package=package) if not version
           else PYPI_VERSION_JSON.format(package=package, version=version))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        loaded: Json = json.loads(response.read().decode("utf-8"))
    return loaded


def _canonical(name: str) -> str:
    """PyPI treats these as the same name; so must we, or we download twice."""
    return name.strip().lower().replace("_", "-")


def _wanted_here(marker: str) -> bool:
    """Whether a requirement's environment marker applies to this build.

    Deliberately small. The only markers these packages use are `extra` (an
    optional feature nobody asked for) and platform or version guards, and
    anything this does not understand is INCLUDED -- a few megabytes of
    something unnecessary is a far better failure than a missing module
    discovered one utterance at a time.
    """
    if not marker:
        return True
    text = marker.lower()
    if "extra ==" in text:
        return False   # an optional feature; we asked for none of them
    for name in ("sys_platform", "platform_system", "os_name"):
        if name in text and not any(
                token in text for token in ("win32", "windows", "nt")):
            return False
    return True


def _split_requirement(requirement: str) -> tuple[str, str]:
    """One requirement as (name, exact version or "").

    The version matters, not only the name. pydantic pins pydantic-core to one
    exact release and raises SystemError on any other -- so resolving names and
    taking the newest of each produced a pack that downloaded cleanly, unpacked
    cleanly, and then refused to import, which is the same failure this
    resolver was written to fix.

    Only an unambiguous `==` pin is honoured. A range means the newest will do;
    where it will not, the package belongs in WANTED with a version.
    """
    text = requirement.strip()

    # "pydantic<3.0.0,>=2.0.0" and "pyopenjtalk-plus[onnxruntime]" both reduce
    # to the name in front of the first delimiter.
    name = text
    for delimiter in ("[", "(", "<", ">", "=", "!", "~", " ", ";"):
        name = name.split(delimiter)[0]
    name = _canonical(name)
    if not name:
        return "", ""

    pinned = ""
    for clause in _specifiers(text):
        if clause.startswith(("===", "==")) and "*" not in clause:
            pinned = clause.lstrip("=").strip() or pinned
    return name, pinned


def _specifiers(requirement: str) -> list[str]:
    """The comparison clauses of a requirement, e.g. ["<3.0.0", ">=2.0.0"]."""
    text = requirement.strip()
    start = next((i for i, character in enumerate(text) if character in "<>=!~"),
                 len(text))
    if start >= len(text):
        return []
    return [part.strip() for part in text[start:].split(",") if part.strip()]


def _requirements(data: Json) -> list[tuple[str, str]]:
    """The packages this one declares it needs, as (name, exact version or "").

    Reading them rather than maintaining a list is the whole point: a list is
    what shipped a phonemizer without pydantic.
    """
    found = []
    for entry in (data.get("info", {}).get("requires_dist") or []):
        requirement, _, marker = str(entry).partition(";")
        if not _wanted_here(marker):
            continue
        name, pinned = _split_requirement(requirement)
        if name:
            found.append((name, pinned))
    return found


def _resolve(wanted: tuple[tuple[str, str], ...] = WANTED,
             timeout: float = 20.0) -> list[tuple[str, str]]:
    """Every package that has to be downloaded, in the order to install them.

    A breadth-first walk of what each wheel says it needs. The alternative --
    running pip -- is not available: the frozen build has no interpreter to run
    it with, which is why this module exists at all.
    """
    queue = [(_canonical(name), version) for name, version in wanted]
    seen = {name for name, _ in queue} | {_canonical(p) for p in PROVIDED}
    resolved: list[tuple[str, str]] = []

    while queue:
        name, version = queue.pop(0)
        resolved.append((name, version))
        if len(resolved) > MAX_PACKAGES:
            raise RuntimeError(
                f"the phonemizer's dependencies came to more than "
                f"{MAX_PACKAGES} packages, which is not something this pack "
                "should ever need")
        try:
            data = _metadata(name, version, timeout)
        except Exception as exc:  # noqa: BLE001 - reported, not fatal per package
            log.warning("could not read what %s needs (%s); continuing without "
                        "its dependencies", name, exc)
            continue
        for dependency, pinned in _requirements(data):
            if dependency in seen:
                continue
            seen.add(dependency)
            # The pin is honoured where one package demands an exact release of
            # another. Ignoring it gave a pydantic and a pydantic-core that
            # refuse to run together.
            queue.append((dependency, pinned))

    log.info("phonemizer needs %d packages: %s", len(resolved),
             ", ".join(f"{name} {version}" if version else name
                       for name, version in resolved))
    return resolved


def _wheel_url(package: str, version: str,
               timeout: float = 20.0) -> tuple[str, int, str]:
    """The Windows wheel for a package, or the pure-python one if that is all
    there is -- the dictionary is data and ships as py3-none-any."""
    data = _metadata(package, timeout=timeout)

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
        say("Working out what the phonemizer needs...")
        packages = _resolve()
        for package, version in packages:
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
