"""Piper voice catalogue: what is installed, what is available, and downloading more.

Three voices ship in the installer. The rest of the catalogue (100+ voices across
many languages) is fetched on demand into the user data directory, which takes
precedence over the bundled folder so a downloaded voice can shadow a bundled one.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .net import USER_AGENT
from .paths import list_voices, resource_root, user_data_dir

log = logging.getLogger(__name__)

CATALOGUE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json?download=true"
)
CATALOGUE_PAGE = "https://huggingface.co/rhasspy/piper-voices"

# Shipped in the installer; never offered for deletion.
BUNDLED = ("en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-high")

QUALITY_ORDER = {"x_low": 0, "low": 1, "medium": 2, "high": 3}

# Whisper models ending in .en are English-only. Pairing one with, say, a German
# voice produces confident nonsense rather than an error, so the UI has to say so.
ENGLISH_PREFIX = "en"


# A catalogue key looks like "en_US-lessac-medium" or "en_GB-alba-medium".
_LANGUAGE_KEY = re.compile(r"^([a-z]{2,3})(?:_[A-Za-z]{2,4})?-")


# Some voices need a phonemizer beyond the espeak data that ships with Piper.
# Japanese voices are trained on OpenJTalk phonemes and Piper imports
# `pyopenjtalk` to produce them; Korean voices do the same with their own.
# Neither is declared as a dependency of piper-tts -- the import simply fails at
# synthesis time, once per utterance, with a traceback and no speech.
#
# Not shipped because of what they cost: pyopenjtalk-plus plus the dictionaries
# it needs measured 341 MB installed, against an installer that is currently a
# fraction of that. Japanese is one language; the download is not.
_PHONEMIZERS = {
    "japanese": ("pyopenjtalk", "Japanese"),
    "korean": ("g2pk", "Korean"),
}


def required_phonemizer(voice_key: str) -> tuple[str, str] | None:
    """The (module, language) a voice needs beyond Piper's own, or None.

    Read from the voice's own config rather than guessed from its name: the
    `phoneme_type` field is what Piper dispatches on, so it is the same thing
    that decides whether the import happens.
    """
    path = installed_path(voice_key)
    if path is None:
        return None
    try:
        config = json.loads(
            path.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("no config for %s: %s", voice_key, exc)
        return None
    return _PHONEMIZERS.get(str(config.get("phoneme_type") or "").lower())


def missing_phonemizer(voice_key: str) -> str:
    """Why this voice cannot be spoken here, or "" if it can.

    Checked before a voice is used rather than discovered inside synthesis: the
    failure there is a ModuleNotFoundError raised once per utterance, which
    reaches the user as "Failed to process utterance" and nothing else.
    """
    needed = required_phonemizer(voice_key)
    if needed is None:
        return ""
    module, language = needed

    # Put the optional pack on the import path first. Doing this only at
    # application startup meant every other entry point -- the tests, a tool,
    # anything importing this module directly -- reported an installed pack as
    # missing, which is a difference nobody should have to know about.
    from . import jppack

    jppack.activate()

    problem = _import_problem(module)
    if problem is None:
        return ""
    if problem == "missing":
        return (f"{voice_key} is a {language} voice, and speaking {language} "
                f"needs the {module} phonemizer, which is not installed. Get it "
                "from Settings -> Add-ons, or choose a different voice.")
    return (f"{voice_key} is a {language} voice, and the {module} phonemizer is "
            f"installed but will not load: {problem}. Try removing and "
            "re-downloading it in Settings -> Add-ons.")


# Importing a package costs real time, so the answer is remembered. Keyed by
# module name; cleared when a pack is installed or removed.
_IMPORT_CHECKS: dict[str, str | None] = {}


def forget_import_checks() -> None:
    """Re-test importability, after a pack is installed or removed."""
    _IMPORT_CHECKS.clear()


def _import_problem(module: str) -> str | None:
    """None if the module imports, "missing" if absent, else why it failed.

    Actually imports it. `find_spec` only proves a module can be FOUND, and a
    phonemizer that is present but will not load reported as usable and then
    failed inside synthesis -- once per utterance, as "Failed to process
    utterance" with the real reason buried in a traceback. Finding out here
    costs one import and says what is actually wrong.
    """
    if module in _IMPORT_CHECKS:
        return _IMPORT_CHECKS[module]

    result: str | None = None
    if importlib.util.find_spec(module) is None:
        result = "missing"
    else:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - any failure is disqualifying
            log.warning("%s is present but will not import: %s", module, exc)
            result = f"{type(exc).__name__}: {exc}"
    _IMPORT_CHECKS[module] = result
    return result


def is_speakable(voice_key: str) -> bool:
    return not missing_phonemizer(voice_key)


def voice_language(voice_key: str) -> str:
    """The voice's language family ("en", "de"), or "" if it cannot be told.

    The config is asked first and the name only as a fallback. Voices made in
    the Studio are named by their author -- "narator", "my voice" -- and carry
    no language in the filename at all, so reading the name was telling people
    their own English voice was not English.

    Returning "" for genuinely unknown matters: callers must be able to say
    nothing rather than guess, and a wrong warning is worse than none.
    """
    if not voice_key:
        return ""

    path = installed_path(voice_key)
    if path is not None:
        try:
            config = json.loads(
                path.with_suffix(".onnx.json").read_text(encoding="utf-8"))
            language = config.get("language") or {}
            family = str(language.get("family") or "").strip().lower()
            if family:
                return family
            code = str(language.get("code") or "").strip().lower()
            if code:
                return code.split("_")[0]
            # Older configs carry only the espeak voice, e.g. "en-us".
            espeak = str((config.get("espeak") or {}).get("voice") or "").lower()
            if espeak:
                return espeak.split("-")[0]
        except (OSError, ValueError, AttributeError) as exc:
            log.debug("no language in the config for %s: %s", voice_key, exc)

    match = _LANGUAGE_KEY.match(voice_key.strip())
    return match.group(1).lower() if match else ""


def is_english(voice_key: str) -> bool:
    return voice_language(voice_key) == ENGLISH_PREFIX


def language_mismatch(voice_key: str, whisper_model: str,
                      output_language: str = "") -> str:
    """Return a warning if this voice cannot say what it is going to be given.

    `output_language` is the language the TEXT will be in when it reaches the
    voice. With translation on that is the target, not the language being
    recognised -- so a Japanese voice is exactly right for English-to-Japanese,
    and warning about it because the recogniser is English-only is nonsense.

    Empty string means the pairing is fine.
    """
    if not voice_key:
        return ""
    family = voice_language(voice_key)
    # Unknown language: say nothing. Warning on a guess is how a voice built in
    # the Studio ended up being called foreign.
    if not family:
        return ""

    if output_language:
        if family == output_language:
            return ""
        return (
            f"{voice_key} speaks {family}, but the text reaching it will be in "
            f"{output_language}. It will mispronounce every word.\n\n"
            f"Pick a {output_language} voice, or change the target language."
        )

    if not whisper_model:
        return ""
    english_model = whisper_model.endswith(".en")
    if english_model and family != ENGLISH_PREFIX:
        return (
            f"{voice_key} is not an English voice, but the {whisper_model} "
            "recognition model only understands English. Speech will be "
            "transcribed as English and the result will be wrong.\n\n"
            "Switch to a multilingual model (large-v3) in Recognition, or pick an "
            "English voice."
        )
    return ""


@dataclass(frozen=True)
class VoiceEntry:
    key: str            # e.g. en_US-amy-medium
    language: str       # e.g. en_US
    name: str           # e.g. amy
    quality: str        # x_low | low | medium | high
    language_label: str  # e.g. English (United States)
    size_mb: float
    num_speakers: int = 1

    @property
    def installed(self) -> bool:
        return installed_path(self.key) is not None

    @property
    def multi_speaker(self) -> bool:
        """Whether the Voice Designer can move through this voice's speakers.

        25 of the 174 catalogue voices qualify. Reading it from the catalogue
        means the designer can offer them before anything is downloaded --
        otherwise the only way to find out is to fetch 100 MB and look.
        """
        return self.num_speakers > 1

    @property
    def bundled(self) -> bool:
        return self.key in BUNDLED

    @property
    def label(self) -> str:
        return f"{self.name} ({self.quality}) - {self.language_label}"


def installed_keys() -> list[str]:
    return [p.stem for p in list_voices()]


def installed_path(key: str) -> Path | None:
    for p in list_voices():
        if p.stem == key:
            return p
    return None


def user_voices_dir() -> Path:
    d = user_data_dir() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_removable(key: str) -> bool:
    """Only voices in the user directory can be deleted; bundled ones are read-only."""
    path = installed_path(key)
    if path is None or key in BUNDLED:
        return False
    try:
        return path.parent.resolve() == user_voices_dir().resolve()
    except OSError:
        return False


# -- catalogue --------------------------------------------------------------

_cache: list[VoiceEntry] | None = None
_cache_lock = threading.Lock()


def fetch_catalogue(timeout: float = 20.0, force: bool = False) -> list[VoiceEntry]:
    """Download and parse voices.json. Cached for the process lifetime."""
    global _cache
    with _cache_lock:
        if _cache is not None and not force:
            return _cache

    req = urllib.request.Request(CATALOGUE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    entries: list[VoiceEntry] = []
    for key, blob in raw.items():
        try:
            lang = blob.get("language", {})
            files = blob.get("files", {})
            onnx_bytes = next(
                (v.get("size_bytes", 0) for k, v in files.items() if k.endswith(".onnx")),
                0,
            )
            entries.append(
                VoiceEntry(
                    key=key,
                    language=blob.get("language", {}).get("code", ""),
                    name=blob.get("name", key),
                    quality=blob.get("quality", ""),
                    language_label=(
                        f"{lang.get('name_english', lang.get('code', '?'))}"
                        f" ({lang.get('country_english')})"
                        if lang.get("country_english")
                        else lang.get("name_english", lang.get("code", "?"))
                    ),
                    size_mb=round(onnx_bytes / 1e6, 1),
                    num_speakers=int(blob.get("num_speakers") or 1),
                )
            )
        except Exception:  # noqa: BLE001 - skip malformed entries, keep the rest
            log.debug("skipping malformed catalogue entry %r", key)

    entries.sort(key=lambda e: (e.language, e.name, QUALITY_ORDER.get(e.quality, 9)))
    with _cache_lock:
        _cache = entries
    log.info("fetched catalogue: %d voices", len(entries))
    return entries


def filter_catalogue(
    entries: list[VoiceEntry],
    language_prefix: str = "",
    query: str = "",
    installed_only: bool = False,
) -> list[VoiceEntry]:
    out = entries
    if language_prefix:
        out = [e for e in out if e.language.lower().startswith(language_prefix.lower())]
    if query:
        q = query.lower()
        out = [e for e in out if q in e.key.lower() or q in e.language_label.lower()]
    if installed_only:
        out = [e for e in out if e.installed]
    return out


def languages(entries: list[VoiceEntry]) -> list[str]:
    return sorted({e.language for e in entries if e.language})


# -- install / remove -------------------------------------------------------


def download_voice(key: str, progress=None) -> Path:
    """Download a voice into the user directory. Returns the .onnx path."""
    from piper.download_voices import download_voice as _dl

    dest = user_voices_dir()
    if progress:
        progress(f"Downloading {key}...")
    log.info("downloading voice %s to %s", key, dest)
    _dl(key, dest)

    onnx = dest / f"{key}.onnx"
    if not onnx.exists():
        raise RuntimeError(f"download finished but {onnx.name} is missing")
    if not (dest / f"{key}.onnx.json").exists():
        raise RuntimeError(f"{key}.onnx.json is missing; the voice will not load")
    if progress:
        progress(f"Installed {key}")
    return onnx


SAMPLE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{name}/{quality}/samples/speaker_0.mp3?download=true"
)


def sample_url(entry: VoiceEntry) -> str:
    """Where to hear a voice before committing to a 60 MB download."""
    family = entry.language.split("_")[0] if entry.language else ""
    return SAMPLE_URL.format(
        lang_family=family, lang_code=entry.language,
        name=entry.name, quality=entry.quality,
    )


def download_sample(entry: VoiceEntry, timeout: float = 20.0) -> Path:
    """Fetch a voice's preview clip into the cache. Returns the file path."""
    from .paths import cache_dir

    dest_dir = cache_dir() / "samples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{entry.key}.mp3"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    req = urllib.request.Request(sample_url(entry), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(5_000_000)
    if len(data) < 1000:
        raise RuntimeError("sample is empty or unavailable for this voice")
    dest.write_bytes(data)
    return dest


def play_sample(entry: VoiceEntry, device_index: int | None = None) -> float:
    """Download and play a preview. Returns its length in seconds.

    Decoded with PyAV, which faster-whisper already pulls in, so previewing costs
    no extra dependency.
    """
    import av
    import numpy as np
    import sounddevice as sd

    path = download_sample(entry)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate or 22050
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("could not decode the sample")

    audio = np.concatenate(chunks).astype(np.float32)
    sd.play(audio, samplerate=rate, device=device_index)
    return len(audio) / rate


# Everything that can sit beside a voice. This list is the contract: a voice is
# its model, its config, and whatever the Studio wrote next to it.
#
# It lives here because voices.py owns the voice directory. When the Studio
# wrote sidecars that only it knew about, deleting a voice left them behind --
# and a later voice with the same name silently inherited the dead one's effects
# chain, which is not something anyone could have diagnosed from the interface.
SIDECAR_SUFFIXES = (
    ".onnx.json",              # Piper's own config; a voice is unusable without it
    ".onnx.design.json",       # Voice Designer effects chain
    ".onnx.provenance.json",   # what a trained voice was built from
    ".v2tvoice",               # the recipe a designed voice was built from
)


def voice_files(model_path: Path) -> list[Path]:
    """Every file belonging to the voice at `model_path`, existing or not."""
    stem = model_path.with_suffix("")
    return [model_path] + [Path(f"{stem}{suffix}") for suffix in SIDECAR_SUFFIXES]


def clear_sidecars(model_path: Path, keep: tuple[str, ...] = ()) -> list[Path]:
    """Remove anything left beside a voice path, and say what went.

    Called after writing a new voice to a path, not only when deleting one:
    installing over an existing name means the previous occupant's sidecars are
    stale, and a stale effects chain is inherited in silence.

    `keep` spares suffixes the caller has just written itself -- the config
    above all, since a voice without one loads and then speaks nonsense.
    """
    gone = []
    for path in voice_files(model_path)[1:]:
        if any(path.name.endswith(suffix) for suffix in keep):
            continue
        if path.exists():
            path.unlink()
            gone.append(path)
    if gone:
        log.info("cleared %d stale file(s) beside %s",
                 len(gone), model_path.name)
    return gone


def remove_voice(key: str) -> bool:
    """Delete a user-downloaded voice. Bundled voices are never removed."""
    if not is_removable(key):
        log.warning("refusing to remove %s (bundled or not user-installed)", key)
        return False
    model = user_voices_dir() / f"{key}.onnx"
    removed = False
    for path in voice_files(model):
        if path.exists():
            path.unlink()
            removed = True
    log.info("removed voice %s", key)
    return removed


def bundled_present() -> list[str]:
    """Which of the three shipped voices actually made it into this build."""
    root = resource_root() / "models" / "voices"
    return [k for k in BUNDLED if (root / f"{k}.onnx").exists()]
