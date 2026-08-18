"""Base checkpoints to fine-tune from, and their licences.

Fine-tuning starts from somebody else's finished voice, and the voices we ship
are .onnx files -- inference weights with the optimizer state stripped out. They
cannot be trained from. The training checkpoints live in a separate HuggingFace
*dataset* repo (rhasspy/piper-checkpoints), at about 850 MB each.

Two things here are discovered rather than assumed, because assuming either one
produces a broken download that only fails after several hundred megabytes:

  * The filename embeds the epoch and step it stopped at, e.g.
    "epoch=2164-step=1355540.ckpt". There is no predicting it; the directory has
    to be listed.
  * Not every voice has a checkpoint at every quality. en_US-ryan-high is
    bundled with this app, but upstream only publishes ryan at medium.

Each checkpoint sits beside a MODEL_CARD naming the dataset it was trained on
and that dataset's licence. A voice fine-tuned from it inherits those terms, so
the card is fetched and shown rather than buried.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

REPO = "rhasspy/piper-checkpoints"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
USER_AGENT = "Voice2TTS"

# For training a voice from scratch rather than fine-tuning one.
BASE_MODEL_DIR = "_base_model"

_TREE_CACHE: dict[str, list[dict]] = {}


@dataclass(frozen=True)
class Checkpoint:
    """One trainable checkpoint in the upstream repo."""

    path: str          # "en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
    size: int
    voice: str = ""    # "en_US-lessac-medium", empty for the from-scratch base

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def directory(self) -> str:
        return self.path.rsplit("/", 1)[0]

    @property
    def url(self) -> str:
        # "=" is legal in a path segment but quoting it avoids any argument
        # about it with intermediate proxies.
        return f"{RESOLVE}/{urllib.parse.quote(self.path)}"

    @property
    def size_gb(self) -> float:
        return self.size / 1e9


# -- locating ---------------------------------------------------------------


def voice_directory(voice_key: str) -> str:
    """"en_US-lessac-medium" -> "en/en_US/lessac/medium".

    Voice keys are <language>-<name>-<quality>. The name may contain
    underscores (en_GB-northern_english_male-medium) but never a dash, so
    splitting on dashes is unambiguous.
    """
    parts = voice_key.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"{voice_key!r} is not a <language>-<name>-<quality> voice key")
    language, name, quality = parts
    family = language.split("_")[0]
    return f"{family}/{language}/{name}/{quality}"


def _tree(path: str, timeout: float = 30.0) -> list[dict]:
    if path in _TREE_CACHE:
        return _TREE_CACHE[path]
    url = f"{API}/{urllib.parse.quote(path)}".rstrip("/")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        entries = json.load(resp)
    _TREE_CACHE[path] = entries
    return entries


def available_qualities(voice_key: str) -> list[str]:
    """Which qualities upstream actually publishes for this voice."""
    directory = voice_directory(voice_key).rsplit("/", 1)[0]
    try:
        return sorted(e["path"].rsplit("/", 1)[-1] for e in _tree(directory)
                      if e.get("type") == "directory")
    except Exception as exc:  # noqa: BLE001 - reported as "none found"
        log.debug("could not list %s: %s", directory, exc)
        return []


def pick_checkpoint(entries: list[dict]) -> Checkpoint | None:
    """The .ckpt in a directory listing. Pure, so it needs no network."""
    files = [e for e in entries
             if e.get("type") == "file" and e.get("path", "").endswith(".ckpt")]
    if not files:
        return None
    # More than one would mean upstream kept several epochs; the largest step
    # is the most trained, and the name sorts correctly by step within a voice.
    best = sorted(files, key=lambda e: e["path"])[-1]
    return Checkpoint(path=best["path"], size=int(best.get("size") or 0))


def resolve(voice_key: str) -> Checkpoint:
    """Find the checkpoint for a voice. Raises with something actionable."""
    directory = voice_directory(voice_key)
    try:
        entries = _tree(directory)
    except Exception as exc:
        qualities = available_qualities(voice_key)
        if qualities:
            raise LookupError(
                f"No training checkpoint for {voice_key}. Upstream publishes "
                f"this voice at: {', '.join(qualities)}."
            ) from exc
        raise LookupError(
            f"No training checkpoint found for {voice_key} ({exc})."
        ) from exc

    found = pick_checkpoint(entries)
    if found is None:
        raise LookupError(f"{directory} exists upstream but holds no .ckpt file.")
    return Checkpoint(path=found.path, size=found.size, voice=voice_key)


def base_model() -> Checkpoint:
    """The generic checkpoint for training a voice from scratch."""
    found = pick_checkpoint(_tree(BASE_MODEL_DIR))
    if found is None:
        raise LookupError("The from-scratch base model is not where it used to be.")
    return found


# -- licensing --------------------------------------------------------------

_LICENSE_LINE = re.compile(r"^\s*\*\s*License:\s*(.+?)\s*$", re.MULTILINE)
_DATASET_LINE = re.compile(r"^\s*\*\s*URL:\s*(.+?)\s*$", re.MULTILINE)


def licence_from_card(card: str) -> str:
    """Pull the licence out of a MODEL_CARD, or "" if it does not say.

    Returning "" rather than guessing matters: a voice whose terms are unknown
    should be shown as unknown, not quietly assumed to be permissive.
    """
    match = _LICENSE_LINE.search(card)
    return match.group(1).strip() if match else ""


def dataset_from_card(card: str) -> str:
    match = _DATASET_LINE.search(card)
    return match.group(1).strip() if match else ""


def model_card(checkpoint: Checkpoint, timeout: float = 30.0) -> str:
    url = f"{RESOLVE}/{urllib.parse.quote(checkpoint.directory)}/MODEL_CARD"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - absence is not fatal
        log.warning("no model card for %s: %s", checkpoint.directory, exc)
        return ""


# -- downloading ------------------------------------------------------------


def download(checkpoint: Checkpoint, dest_dir: Path, progress=None,
             timeout: float = 60.0) -> Path:
    """Fetch a checkpoint, resuming a partial file if there is one.

    850 MB over a domestic connection is long enough that an interruption is
    likely, and starting again from zero each time is what makes a feature like
    this feel broken. The CDN supports range requests, so a partial download is
    continued rather than discarded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / checkpoint.filename
    if final.exists() and checkpoint.size and final.stat().st_size == checkpoint.size:
        log.info("checkpoint already downloaded: %s", final.name)
        return final

    partial = final.with_suffix(final.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if checkpoint.size and have >= checkpoint.size:
        # A leftover at or past the full size cannot be resumed -- asking for
        # bytes beyond the end returns 416, and it would do so every time from
        # then on. Start again rather than wedge the download permanently.
        log.info("discarding an oversized partial download (%d bytes)", have)
        partial.unlink(missing_ok=True)
        have = 0

    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
        log.info("resuming %s at %.0f MB", checkpoint.filename, have / 1e6)

    req = urllib.request.Request(checkpoint.url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # A server that ignores Range answers 200 with the whole file, and
        # appending to what we already have would corrupt it.
        resuming = resp.status == 206
        if have and not resuming:
            log.info("server ignored the range request; starting over")
            have = 0
        total = int(resp.headers.get("Content-Length") or 0) + have

        with partial.open("ab" if resuming else "wb") as fh:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
                if progress:
                    progress(have, total or checkpoint.size)

    if checkpoint.size and partial.stat().st_size != checkpoint.size:
        raise RuntimeError(
            f"{checkpoint.filename} is {partial.stat().st_size} bytes, expected "
            f"{checkpoint.size}. The download was cut short; try again."
        )
    partial.replace(final)
    log.info("downloaded %s (%.0f MB)", final.name, final.stat().st_size / 1e6)
    return final
