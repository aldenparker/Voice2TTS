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

import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import net
from .paths import cache_dir

log = logging.getLogger(__name__)

MODEL_DIR_NAME = "model"

# Marian keeps two tokenizers: one for the language going in, one for the
# language coming out. Decoding the output with the source tokenizer does not
# fail -- it leaves raw "▁" pieces in the text and produces something that
# looks almost right. Some publishers ship a single shared tokenizer instead,
# which is why both layouts are recognised.
SOURCE_TOKENIZER = "source.spm"
TARGET_TOKENIZER = "target.spm"
SHARED_TOKENIZER_NAMES = ("sentencepiece.model", "sp.model")

# Marian is trained with an explicit end-of-sentence marker on the input.
# Without it the model does not error -- it rambles, repeating fragments until
# it hits the decoding limit ("Hallo Hallo Hallo, koennen Sie hoeren Sie mich
# hoeren????"). Fluent-looking nonsense is the worst failure mode there is, so
# this is appended on the way in and stripped on the way out.
EOS = "</s>"

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


def tokenizers_in(directory: Path) -> tuple[Path, Path] | None:
    """The (source, target) tokenizers for a model, or None if incomplete.

    A shared tokenizer is returned as the same path twice, so callers do not
    have to care which layout a publisher used.
    """
    source = directory / SOURCE_TOKENIZER
    target = directory / TARGET_TOKENIZER
    if source.is_file() and target.is_file():
        return source, target
    for name in SHARED_TOKENIZER_NAMES:
        shared = directory / name
        if shared.is_file():
            return shared, shared
    # A single loose .spm is a shared tokenizer under an unfamiliar name -- but
    # only if it is not half of a pair. A lone source.spm is a broken download,
    # and using it to decode is the silent-gibberish case above.
    loose = [p for p in sorted(directory.glob("*.spm"))
             if p.name not in (SOURCE_TOKENIZER, TARGET_TOKENIZER)]
    if len(loose) == 1:
        return loose[0], loose[0]
    return None


def is_usable(directory: Path) -> bool:
    """Both halves present. A model without its tokenizer produces gibberish."""
    return ((directory / MODEL_DIR_NAME / "model.bin").is_file()
            and tokenizers_in(directory) is not None)


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
        source_spm, target_spm = tokenizers_in(root)
        if source_spm == target_spm:
            shutil.move(str(source_spm), str(keep / SHARED_TOKENIZER_NAMES[0]))
        else:
            shutil.move(str(source_spm), str(keep / SOURCE_TOKENIZER))
            shutil.move(str(target_spm), str(keep / TARGET_TOKENIZER))
        # The attribution CC-BY requires has to travel with the model.
        for extra in ("LICENSE", "metadata.json"):
            if (root / extra).is_file():
                shutil.move(str(root / extra), str(keep / extra))
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


# -- the published catalogue ------------------------------------------------

# Models are attached to a release of their own rather than to an app release:
# they change on a different schedule (a new pair does not need a new build),
# and an 800 MB app release would be absurd. The tag is pinned rather than
# "latest" so a shipped build keeps working when a newer set is published --
# an old build should not start downloading models converted against a
# CTranslate2 it does not have.
MODELS_TAG = "models-2"
ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"

# A copy of the last catalogue fetched, so the picker can list what is
# available without a round trip, and still show something useful offline.
CATALOGUE_CACHE = "catalogue.json"

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "ru": "Russian",
    "pl": "Polish", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "cs": "Czech", "uk": "Ukrainian",
}


def language_name(code: str) -> str:
    """A name for a language code, falling back to the code itself.

    Unknown codes are shown as-is rather than as "Unknown": the code is at
    least a true thing the user can look up.
    """
    return LANGUAGE_NAMES.get(code, code)


@dataclass(frozen=True)
class Available:
    """One pair published in the catalogue, installed or not."""

    source: str
    target: str
    asset: str
    size: int = 0
    sha256: str = ""
    origin: str = ""
    licence: str = ""
    licence_url: str = ""

    @property
    def code(self) -> str:
        return f"{self.source}_{self.target}"

    @property
    def label(self) -> str:
        return f"{language_name(self.source)} → {language_name(self.target)}"

    @property
    def installed(self) -> bool:
        return is_usable(models_dir() / self.code)

    def url(self, repo: str, tag: str = MODELS_TAG) -> str:
        return ASSET_URL.format(repo=repo, tag=tag, asset=self.asset)


def parse_catalogue(data: dict) -> list[Available]:
    """Turn manifest.json into entries. Pure, so it needs no network.

    Skips entries it cannot use rather than failing the whole list: a future
    manifest may carry pairs this build does not understand, and one of those
    should not make every other pair unavailable.
    """
    entries = []
    for item in data.get("pairs", []):
        try:
            entries.append(Available(
                source=str(item["source"]), target=str(item["target"]),
                asset=str(item["asset"]), size=int(item.get("bytes") or 0),
                sha256=str(item.get("sha256") or ""),
                origin=str(item.get("origin") or ""),
                licence=str(item.get("licence") or ""),
                licence_url=str(item.get("licence_url") or ""),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("skipping an unreadable catalogue entry: %s", exc)
    return sorted(entries, key=lambda e: (e.source, e.target))


def fetch_catalogue(repo: str, tag: str = MODELS_TAG,
                    timeout: float = 30.0) -> list[Available]:
    """What is published, from the network, falling back to the last copy.

    A failed fetch returns the cached list rather than raising: someone who
    already downloaded a catalogue should still be able to see which models
    they have, and telling them "no models exist" because the network blinked
    would be worse than slightly stale.
    """
    url = ASSET_URL.format(repo=repo, tag=tag, asset="manifest.json")
    cache = models_dir() / CATALOGUE_CACHE
    try:
        data = net.fetch_json(url, timeout)
    except Exception as exc:  # noqa: BLE001 - offline is normal, not an error
        log.warning("could not fetch the model catalogue: %s", exc)
        if cache.is_file():
            try:
                return parse_catalogue(json.loads(cache.read_text(encoding="utf-8")))
            except (OSError, ValueError) as bad:
                log.warning("the cached catalogue is unreadable: %s", bad)
        return []

    entries = parse_catalogue(data)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        log.debug("could not cache the catalogue: %s", exc)
    return entries


def download_pair(entry: Available, repo: str, tag: str = MODELS_TAG,
                  progress=None, timeout: float = 60.0) -> Path:
    """Fetch and install one pair. Returns the installed directory.

    The archive is kept in the models directory while it downloads so a
    resumed download survives a restart, and removed once installed -- 63 MB
    of zip is not worth keeping beside the 63 MB it unpacked to.
    """
    archive = models_dir() / entry.asset
    net.download(entry.url(repo, tag), archive, expected_size=entry.size,
                 sha256=entry.sha256, progress=progress, timeout=timeout)
    try:
        installed = install_package(archive, entry.source, entry.target)
    finally:
        archive.unlink(missing_ok=True)
    log.info("installed %s", entry.label)
    return installed


def split_sentences(text: str) -> list[str]:
    """Break an utterance into sentences for translation."""
    return [part.strip() for part in _SENTENCE.split(text.strip()) if part.strip()]


class Translator:
    """One loaded direction. Loading costs ~180 ms, so keep it around."""

    def __init__(self, pair: Pair, beam_size: int = DEFAULT_BEAM,
                 threads: int = 2):
        import ctranslate2
        import sentencepiece

        found = tokenizers_in(pair.path)
        if found is None or not is_usable(pair.path):
            raise TranslationUnavailable(f"{pair.path} is not a usable model")
        source_spm, target_spm = found

        self.pair = pair
        self.beam_size = beam_size
        self._engine = ctranslate2.Translator(
            str(pair.path / MODEL_DIR_NAME), device="cpu",
            inter_threads=1, intra_threads=threads)
        self._source = sentencepiece.SentencePieceProcessor(
            model_file=str(source_spm))
        self._target = (self._source if target_spm == source_spm
                        else sentencepiece.SentencePieceProcessor(
                            model_file=str(target_spm)))
        log.info("translation ready: %s to %s", pair.source, pair.target)

    def translate(self, text: str) -> str:
        """Translate one utterance, sentence by sentence."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        batch = [[*self._source.encode(s, out_type=str), EOS] for s in sentences]
        results = self._engine.translate_batch(
            batch, beam_size=self.beam_size,
            max_decoding_length=MAX_DECODING_LENGTH)
        return " ".join(
            self._target.decode([t for t in r.hypotheses[0] if t != EOS])
            for r in results)

    def close(self) -> None:
        self._engine = None
        self._source = None
        self._target = None


def chain_for(source: str, target: str, pivot: str = "en",
              beam_size: int = DEFAULT_BEAM) -> Chain:
    """A ready chain for a language pair, or a failure that names the pair.

    Chain itself takes a route, because that is what it needs; callers almost
    always have two language codes and would otherwise each repeat the routing
    and write their own version of this message.
    """
    pairs = route(source, target, pivot)
    if not pairs:
        raise TranslationUnavailable(
            f"no model installed for {language_name(source)} to "
            f"{language_name(target)}")
    return Chain(pairs, beam_size=beam_size)


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
