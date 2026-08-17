"""Update checking and one-click install, backed by GitHub Releases.

Flow: ask the GitHub API for the newest release, compare against __version__, and
if it is newer offer to download the installer asset and run it.

Applying the update is the fiddly part. The app cannot replace its own files while
running, so:

  1. the installer is downloaded and verified,
  2. it is launched detached with /SILENT and a /relaunch=1 flag,
  3. the app quits immediately so its files are unlocked.

Inno's CloseApplications handles the residual race if step 3 has not finished, and
the installer relaunches the app when it is done. The install is per-user, so no UAC
prompt appears. Files fetched programmatically also carry no mark-of-the-web, so
SmartScreen does not challenge the downloaded installer either.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .paths import cache_dir, is_frozen

log = logging.getLogger(__name__)

USER_AGENT = f"Voice2TTS/{__version__}"
API_TEMPLATE = "https://api.github.com/repos/{repo}/releases/latest"
RELEASES_PAGE = "https://github.com/{repo}/releases/latest"

MAX_INSTALLER_BYTES = 3_000_000_000
_VERSION_PART = re.compile(r"\d+")

# Exactly "owner/name". A looser "must contain a slash" check passes things like
# "https://example.test", which then get pasted into the API URL and produce a
# confusing network error instead of a clear "that is not a repo".
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    notes: str
    asset_name: str
    asset_url: str
    asset_size: int
    sha256_url: str | None
    page_url: str

    @property
    def size_mb(self) -> float:
        return round(self.asset_size / 1e6, 1)


def parse_version(text: str) -> tuple[int, ...]:
    """Loose semver parse: 'v1.2.3', '1.2.3-beta-4' and '1.2.3+build' all work.

    Pre-releases sort BELOW the release they lead to, which is the semver rule
    and the reason this is not just a tuple of the numbers:

        1.2.2  <  1.2.3-beta-1  <  1.2.3-beta-2  <  1.2.3

    Betas did not exist when this was first written and the suffix was simply
    discarded, which made 1.2.3-beta-1 compare EQUAL to 1.2.3. The visible
    consequence would have been a beta tester never being offered the finished
    release, because the app could not tell it was behind.

    The numeric part is padded to a fixed width so the pre-release marker is
    always compared in the same position; without that, 1.2 and 1.2.0 would
    order by length instead of by value.
    """
    head, _, _build = text.partition("+")
    base, _, pre = head.partition("-")

    numbers = [int(n) for n in _VERSION_PART.findall(base)][:4]
    numbers += [0] * (4 - len(numbers))

    if not pre:
        # A release outranks every pre-release of the same numbers.
        return (*numbers, 1, 0)
    # "beta-2", "beta.2" and "rc2" all yield 2; an unnumbered "beta" yields 0.
    found = _VERSION_PART.findall(pre)
    return (*numbers, 0, int(found[0]) if found else 0)


def is_newer(candidate: str, current: str = __version__) -> bool:
    return parse_version(candidate) > parse_version(current)


def current_version() -> str:
    return __version__


# -- discovery --------------------------------------------------------------


def check(repo: str, timeout: float = 15.0) -> Release | None:
    """Ask GitHub for the latest release. Returns None if it is not newer.

    Raises on network or API failure so callers can distinguish "up to date" from
    "could not tell".
    """
    if not _REPO_RE.match(repo or ""):
        raise ValueError(
            f"update repo must look like 'owner/name', got {repo!r}. "
            "Set it in Settings -> Updates."
        )

    req = urllib.request.Request(
        API_TEMPLATE.format(repo=repo),
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # GitHub returns 404 rather than 403 for a private repo when the caller
            # is unauthenticated, so "not found" and "no access" are indistinguishable
            # here. Say both instead of guessing wrong.
            raise RuntimeError(
                f"No releases found for {repo}.\n\n"
                "Either no release has been published yet (use scripts/release.ps1), "
                "or the repository is private — update checks send no credentials, so "
                "a private repository always looks empty. Make it public for updates "
                "to work."
            ) from exc
        if exc.code in (403, 429):
            raise RuntimeError(
                "GitHub rate limit reached. Try again later; unauthenticated checks "
                "are limited to 60 per hour per address."
            ) from exc
        raise

    tag = str(data.get("tag_name") or "")
    version = tag.lstrip("vV") or "0"
    if not is_newer(version):
        log.info("up to date (latest %s, running %s)", version or "?", __version__)
        return None

    assets = data.get("assets") or []
    installer = next(
        (a for a in assets
         if a.get("name", "").lower().endswith(".exe")
         and "setup" in a.get("name", "").lower()),
        None,
    )
    if installer is None:
        raise RuntimeError(f"Release {tag} has no *Setup*.exe asset attached")

    checksum = next(
        (a for a in assets if a.get("name", "").lower().endswith(".sha256")), None
    )

    return Release(
        version=version,
        tag=tag,
        notes=str(data.get("body") or "").strip(),
        asset_name=installer["name"],
        asset_url=installer["browser_download_url"],
        asset_size=int(installer.get("size") or 0),
        sha256_url=checksum["browser_download_url"] if checksum else None,
        page_url=data.get("html_url") or RELEASES_PAGE.format(repo=repo),
    )


def should_check(last_check_epoch: float, interval_hours: int) -> bool:
    if interval_hours <= 0:
        return False
    return (time.time() - last_check_epoch) >= interval_hours * 3600


# -- download ---------------------------------------------------------------


def download(release: Release, progress=None, timeout: float = 120.0) -> Path:
    """Fetch the installer into the cache directory and verify it."""
    dest_dir = cache_dir() / "updates"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / release.asset_name

    req = urllib.request.Request(release.asset_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or release.asset_size or 0)
        if total > MAX_INSTALLER_BYTES:
            raise RuntimeError(f"installer is implausibly large ({total} bytes)")
        got = 0
        with dest.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                got += len(chunk)
                if got > MAX_INSTALLER_BYTES:
                    raise RuntimeError("download exceeded size limit")
                fh.write(chunk)
                if progress:
                    progress(got, total)

    _verify(dest, release)
    log.info("downloaded update to %s", dest)
    return dest


def _verify(path: Path, release: Release) -> None:
    size = path.stat().st_size
    if release.asset_size and size != release.asset_size:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch: expected {release.asset_size} bytes, got {size}"
        )
    # Refuse anything that is not a Windows executable before we run it.
    with path.open("rb") as fh:
        if fh.read(2) != b"MZ":
            path.unlink(missing_ok=True)
            raise RuntimeError("downloaded file is not a Windows executable")

    if release.sha256_url:
        expected = _fetch_expected_sha(release.sha256_url)
        if expected:
            actual = _sha256(path)
            if actual.lower() != expected.lower():
                path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"checksum mismatch\n  expected {expected}\n  actual   {actual}"
                )
            log.info("checksum verified")
    else:
        log.warning("release published no .sha256 asset; skipping checksum check")


def _fetch_expected_sha(url: str, timeout: float = 20.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read(4096).decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001 - a missing checksum is not fatal
        log.warning("could not fetch checksum: %s", exc)
        return None
    # Accept both "<hash>" and the "<hash>  <filename>" shasum format.
    token = text.split()[0] if text.split() else ""
    return token if re.fullmatch(r"[0-9a-fA-F]{64}", token) else None


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -- apply ------------------------------------------------------------------


def apply(installer: Path, relaunch: bool = True) -> None:
    """Launch the installer detached and return so the caller can quit the app.

    The caller MUST exit promptly afterwards: Inno cannot overwrite files this
    process still holds open.
    """
    if not is_frozen():
        raise RuntimeError(
            "Refusing to run an installer over a source checkout. "
            "Update with git instead."
        )
    if not installer.exists():
        raise FileNotFoundError(installer)

    args = [str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    if relaunch:
        args.append("/relaunch=1")

    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so the installer outlives us.
    creation = 0x00000008 | 0x00000200
    log.info("launching updater: %s", " ".join(args))
    subprocess.Popen(
        args,
        creationflags=creation,
        close_fds=True,
        cwd=str(installer.parent),
    )


def cleanup_old_downloads(keep: str | None = None) -> int:
    """Delete previously downloaded installers; they are hundreds of MB each."""
    folder = cache_dir() / "updates"
    if not folder.is_dir():
        return 0
    removed = 0
    for item in folder.glob("*.exe"):
        if keep and item.name == keep:
            continue
        try:
            item.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def installed_location() -> Path:
    """Where the running app lives, for diagnostics."""
    return Path(sys.executable).parent if is_frozen() else Path(os.getcwd())
