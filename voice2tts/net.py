"""Shared HTTP: one User-Agent, one resumable download, one checksum check.

Six modules had grown their own copy of the same twelve lines of urllib
boilerplate, and four of them announced themselves as "Voice2TTS/0.2" long
after 0.2 shipped -- a User-Agent that lies about its version is worse than
none, because it is what a host looks at when deciding whether to rate-limit us.

`download()` is the resumable one from checkpoints.py, generalised: the range
handling there was written for an 850 MB file over a domestic connection and is
worth having anywhere a download is big enough to be interrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import __version__

log = logging.getLogger(__name__)

CHUNK = 1024 * 512


def user_agent(note: str = "") -> str:
    """Identify this build honestly. `note` adds a contact URL for hosts that
    ask for one (VB-Audio does, on their download page)."""
    return f"Voice2TTS/{__version__}" + (f" (+{note})" if note else "")


USER_AGENT = user_agent()


def request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    combined = {"User-Agent": USER_AGENT}
    combined.update(headers or {})
    return urllib.request.Request(url, headers=combined)


def fetch(url: str, timeout: float = 30.0,
          headers: dict[str, str] | None = None) -> bytes:
    with urllib.request.urlopen(request(url, headers), timeout=timeout) as resp:
        return resp.read()


def fetch_json(url: str, timeout: float = 30.0,
               headers: dict[str, str] | None = None):
    return json.loads(fetch(url, timeout, headers).decode("utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, *, expected_size: int = 0, sha256: str = "",
             progress: Callable[[int, int], None] | None = None,
             timeout: float = 60.0) -> Path:
    """Fetch `url` to `dest`, resuming a partial file if there is one.

    Verifies the checksum when one is given. A file that arrives corrupt is
    deleted rather than left in place: keeping it means every later run finds
    something at the right path, decides the download is done, and fails
    somewhere much further away from the cause.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _is_complete(dest, expected_size, sha256):
        log.info("already downloaded: %s", dest.name)
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if expected_size and have >= expected_size:
        # A leftover at or past the full size cannot be resumed -- asking for
        # bytes beyond the end returns 416, and would do so on every later run.
        log.info("discarding an oversized partial download (%d bytes)", have)
        partial.unlink(missing_ok=True)
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        log.info("resuming %s at %.1f MB", dest.name, have / 1e6)

    with urllib.request.urlopen(request(url, headers), timeout=timeout) as resp:
        # A server that ignores Range answers 200 with the whole file. Opening
        # "wb" rather than "ab" is what keeps that from corrupting the file --
        # the counter reset below only keeps the progress bar honest, which is
        # why it is worth stating: it is not the safety net it looks like.
        resuming = resp.status == 206
        if have and not resuming:
            log.info("server ignored the range request; starting over")
            have = 0
        total = int(resp.headers.get("Content-Length") or 0) + have
        with partial.open("ab" if resuming else "wb") as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                have += len(chunk)
                if progress:
                    progress(have, total or expected_size)

    if expected_size and partial.stat().st_size != expected_size:
        size = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name} is {size} bytes, expected {expected_size}. "
            "The download was cut short; try again.")

    if sha256:
        got = sha256_of(partial)
        if got != sha256.lower():
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"{dest.name} does not match its published checksum. The file "
                "was corrupted in transit or changed at the source; it has "
                "been discarded.")

    partial.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _is_complete(path: Path, expected_size: int, sha256: str) -> bool:
    if expected_size and path.stat().st_size != expected_size:
        return False
    if sha256 and sha256_of(path) != sha256.lower():
        log.warning("%s is the right size but the wrong file; refetching", path.name)
        return False
    return bool(expected_size or sha256)
