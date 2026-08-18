"""Local translation, between recognition and speech.

Runs on CTranslate2, which faster-whisper already depends on, so translation
adds no new inference runtime -- only SentencePiece for tokenisation. Measured
in `spike/08_translate.py`: 38 ms for a short sentence, 98 ms for a 25-word one,
against a pipeline that is ~300 ms end to end.

A model is a directory holding a CTranslate2 model and the SentencePiece
tokenizer it was built with:

    <cache>/translate/en_de/
        model/model.bin, config.json, shared_vocabulary.json
        sentencepiece.model

That layout is deliberately not tied to where the files came from. Publishers
package these differently, and the acquisition question is still open, so
nothing here knows or cares -- `install_package()` normalises whatever arrives
into the shape above, and everything else works off the directory.

Models live in the cache directory rather than beside the config: they are
~150 MB each and re-downloadable, and a roaming profile should not try to sync
them.
"""

from __future__ import annotations

import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .paths import cache_dir

log = logging.getLogger(__name__)

MODEL_DIR_NAME = "model"
TOKENIZER_NAMES = ("sentencepiece.model", "source.spm", "sp.model")

# Beam 4 costs 96 ms against beam 1's 78 ms on a long sentence, for output that
# was identical on the sentences measured. Cheap enough to keep the better
# search, and exposed so a slow machine can drop it.
DEFAULT_BEAM = 4
MAX_DECODING_LENGTH = 256

# MT models are trained on sentences, so a multi-sentence utterance translates
# better one sentence at a time. Split on terminal punctuation followed by
# space, keeping the punctuation -- deliberately simple, because this runs on
# recognised speech, which has no abbreviations like "Dr." unless a substitution
# put them there.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class TranslationUnavailable(Exception):
    """No usable model for the requested pair."""


@dataclass(frozen=True)
class Pair:
    """One installed direction, e.g. en -> de."""

    source: str
    target: str
    path: Path

    @property
    def code(self) -> str:
        return f"{self.source}_{self.target}"

    @property
    def label(self) -> str:
        return f"{self.source} → {self.target}"

    @property
    def size_mb(self) -> float:
        return sum(p.stat().st_size for p in self.path.rglob("*")
                   if p.is_file()) / 1e6


# -- what is installed ------------------------------------------------------


def models_dir() -> Path:
    d = cache_dir() / "translate"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tokenizer_in(directory: Path) -> Path | None:
    for name in TOKENIZER_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return next(iter(sorted(directory.glob("*.spm"))), None)


def is_usable(directory: Path) -> bool:
    """Both halves present. A model without its tokenizer produces gibberish."""
    return ((directory / MODEL_DIR_NAME / "model.bin").is_file()
            and _tokenizer_in(directory) is not None)


def installed_pairs() -> list[Pair]:
    """Every usable pair on this machine, by directory name `<src>_<dst>`."""
    found: list[Pair] = []
    for entry in sorted(models_dir().iterdir()) if models_dir().is_dir() else []:
        if not entry.is_dir() or "_" not in entry.name:
            continue
        source, _, target = entry.name.partition("_")
        if source and target and is_usable(entry):
            found.append(Pair(source=source, target=target, path=entry))
    return found


def find_pair(source: str, target: str) -> Pair | None:
    return next((p for p in installed_pairs()
                 if p.source == source and p.target == target), None)


def route(source: str, target: str, pivot: str = "en") -> list[Pair]:
    """How to get from `source` to `target` with what is installed.

    Returns the pairs to apply in order: one for a direct model, two for a
    pivot, empty if it cannot be done. Pivoting costs a second hop and
    compounds its errors, so a direct model always wins.
    """
    if source == target:
        return []
    direct = find_pair(source, target)
    if direct is not None:
        return [direct]
    if pivot in (source, target):
        return []
    first, second = find_pair(source, pivot), find_pair(pivot, target)
    return [first, second] if first and second else []


# -- installing -------------------------------------------------------------


def install_package(archive: Path, source: str, target: str) -> Path:
    """Unpack a downloaded model package into the layout above.

    Takes whatever shape the publisher used and keeps only the two things that
    matter. Notably it drops any sentence-splitting model shipped alongside:
    those are torch checkpoints, and this pipeline already works one utterance
    at a time.
    """
    destination = models_dir() / f"{source}_{target}"
    staging = models_dir() / f".{source}_{target}.partial"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)

        root = staging
        # Packages usually wrap everything in one directory.
        if not is_usable(root):
            inner = [p for p in root.iterdir() if p.is_dir()]
            root = next((p for p in inner if is_usable(p)), root)
        if not is_usable(root):
            raise TranslationUnavailable(
                f"{archive.name} has no CTranslate2 model and tokenizer in it")

        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        keep = models_dir() / f".{source}_{target}.keep"
        if keep.exists():
            shutil.rmtree(keep, ignore_errors=True)
        keep.mkdir()
        shutil.move(str(root / MODEL_DIR_NAME), str(keep / MODEL_DIR_NAME))
        tokenizer = _tokenizer_in(root)
        shutil.move(str(tokenizer), str(keep / "sentencepiece.model"))
        keep.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info("installed %s->%s (%.0f MB)", source, target,
             sum(p.stat().st_size for p in destination.rglob("*")
                 if p.is_file()) / 1e6)
    return destination


def remove_pair(source: str, target: str) -> bool:
    directory = models_dir() / f"{source}_{target}"
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()


# -- translating ------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Break an utterance into sentences for translation."""
    return [part.strip() for part in _SENTENCE.split(text.strip()) if part.strip()]


class Translator:
    """One loaded direction. Loading costs ~180 ms, so keep it around."""

    def __init__(self, pair: Pair, beam_size: int = DEFAULT_BEAM,
                 threads: int = 2):
        import ctranslate2
        import sentencepiece

        tokenizer = _tokenizer_in(pair.path)
        if tokenizer is None or not is_usable(pair.path):
            raise TranslationUnavailable(f"{pair.path} is not a usable model")

        self.pair = pair
        self.beam_size = beam_size
        self._engine = ctranslate2.Translator(
            str(pair.path / MODEL_DIR_NAME), device="cpu",
            inter_threads=1, intra_threads=threads)
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(tokenizer))
        log.info("translation ready: %s to %s", pair.source, pair.target)

    def translate(self, text: str) -> str:
        """Translate one utterance, sentence by sentence."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        batch = [self._tokenizer.encode(s, out_type=str) for s in sentences]
        results = self._engine.translate_batch(
            batch, beam_size=self.beam_size,
            max_decoding_length=MAX_DECODING_LENGTH)
        return " ".join(self._tokenizer.decode(r.hypotheses[0]) for r in results)

    def close(self) -> None:
        self._engine = None
        self._tokenizer = None


class Chain:
    """One or more hops, so a pivot looks the same as a direct translation."""

    def __init__(self, pairs: list[Pair], beam_size: int = DEFAULT_BEAM):
        if not pairs:
            raise TranslationUnavailable("no route between those languages")
        self.pairs = pairs
        self._steps = [Translator(p, beam_size=beam_size) for p in pairs]

    @property
    def pivoted(self) -> bool:
        """Whether this goes through a third language, and so compounds errors."""
        return len(self._steps) > 1

    @property
    def label(self) -> str:
        if not self.pivoted:
            return self.pairs[0].label
        return " → ".join([self.pairs[0].source]
                          + [p.target for p in self.pairs])

    def translate(self, text: str) -> str:
        for step in self._steps:
            text = step.translate(text)
        return text

    def close(self) -> None:
        for step in self._steps:
            step.close()
