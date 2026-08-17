"""Recorded clips, their quality, and turning them into a Piper training set.

Two ways in, one dataset out. Clips recorded in the app and audio files imported
from elsewhere go through the same checks and the same preparation, so imported
material cannot quietly bypass the quality bar.

Quality matters more than quantity here. Twenty minutes of clean, consistent
speech trains a better voice than an hour with clipping, room echo and a fan in
the background -- and the failure is invisible until hours of GPU time have been
spent, so problems are caught at recording time and shown immediately.

Piper's trainer reads a pipe-delimited CSV beside the audio:

    wav|text                (single speaker)
    wav|speaker|text        (multi speaker)
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import cache_dir

log = logging.getLogger(__name__)

# Piper's medium-quality voices are 22050 Hz mono. Recording must capture at the
# device rate and resample to this -- NOT reuse the 16 kHz recognition path, which
# throws away everything above 8 kHz and would cap the trained voice's quality.
TARGET_RATE = 22050

# Quality thresholds. Deliberately forgiving: a warning that fires constantly gets
# ignored, and the point is to catch clips that will actively harm training.
MIN_PEAK = 0.03          # below this the speaker is inaudible or muted
CLIP_CEILING = 0.995     # samples at or above full scale
MAX_CLIPPED_FRACTION = 0.001
MIN_SECONDS = 0.4

# Noise is judged as a RATIO, not an absolute level. An absolute floor punishes a
# quiet but clean recording and passes a loud but hissy one -- and it treats
# continuous speech as noise, because "the quiet part of the clip" is only a
# meaningful measurement when the speaker actually pauses.
MIN_SNR_DB = 18.0
MIN_SPEECH_FRACTION = 0.30   # of the clip carrying speech rather than room tone


@dataclass
class Clip:
    key: str
    text: str
    filename: str
    seconds: float = 0.0
    peak: float = 0.0
    noise_floor: float = 0.0
    issues: list[str] = field(default_factory=list)
    source: str = "recorded"     # recorded | imported

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def words(self) -> int:
        return len(self.text.split())


def analyse(audio: np.ndarray, rate: int, text: str = "") -> tuple[dict, list[str]]:
    """Measure a clip and list what is wrong with it.

    Returns (measurements, issues). An empty issue list means it is usable.
    """
    issues: list[str] = []
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    seconds = len(audio) / float(rate) if rate else 0.0
    if not len(audio):
        return {"seconds": 0.0, "peak": 0.0, "noise_floor": 0.0}, ["empty recording"]

    magnitude = np.abs(audio)
    peak = float(magnitude.max())

    # Per-frame RMS, ~10 ms. RMS rather than peak because a single click should not
    # define the level of the frame it lands in.
    frame = max(1, rate // 100)
    usable_len = len(audio) // frame * frame
    frames = (np.sqrt((audio[:usable_len].reshape(-1, frame) ** 2).mean(axis=1))
              if usable_len else np.zeros(1, dtype=np.float32))

    noise_floor = float(np.percentile(frames, 5))
    speech_level = float(np.percentile(frames, 90))
    snr_db = 20.0 * np.log10(max(speech_level, 1e-9) / max(noise_floor, 1e-9))

    # Halfway between the floor and the speech level, so the split does not depend
    # on absolute loudness.
    threshold = noise_floor + 0.3 * (speech_level - noise_floor)
    speech_fraction = float((frames > threshold).mean())

    if seconds < MIN_SECONDS:
        issues.append("too short")
    if peak < MIN_PEAK:
        issues.append("too quiet — check the microphone is selected and unmuted")
    clipped = float((magnitude >= CLIP_CEILING).mean())
    if clipped > MAX_CLIPPED_FRACTION:
        issues.append("clipping — lower the input gain")
    # Only meaningful once there is enough clip to have both speech and gaps.
    if seconds >= 1.0 and snr_db < MIN_SNR_DB:
        issues.append("noisy background")
    if seconds > 1.5 and speech_fraction < MIN_SPEECH_FRACTION:
        issues.append("mostly silence — start speaking sooner or stop earlier")

    # A clip far shorter than its prompt usually means the recording was cut off.
    if text:
        expected = len(text.split()) / 150.0 * 60.0
        if expected > 1.0 and seconds < expected * 0.45:
            issues.append("shorter than the prompt — was it cut off?")

    return {
        "seconds": round(seconds, 3),
        "peak": round(peak, 4),
        "noise_floor": round(noise_floor, 4),
        "speech_fraction": round(speech_fraction, 3),
    }, issues


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """16-bit mono PCM, which is what the trainer reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio.reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path.name}: expected 16-bit audio, got {width * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def to_target_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE:
        return audio
    import soxr

    return soxr.resample(audio, rate, TARGET_RATE, quality="VHQ").astype(np.float32)


class RecordingSession:
    """A voice being collected: its clips, its progress, and its problems."""

    def __init__(self, name: str, root: Path | None = None,
                 target_minutes: float = 30.0):
        self.name = name
        self.target_seconds = max(60.0, target_minutes * 60.0)
        self.root = root or (cache_dir() / "studio" / "datasets" / _safe(name))
        self.clips: list[Clip] = []
        self.created = time.time()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- state ---------------------------------------------------------------

    @property
    def audio_dir(self) -> Path:
        return self.root / "wav"

    @property
    def index_path(self) -> Path:
        return self.root / "session.json"

    @property
    def usable(self) -> list[Clip]:
        return [c for c in self.clips if c.ok]

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.usable)

    @property
    def words(self) -> int:
        return sum(c.words for c in self.usable)

    @property
    def progress(self) -> float:
        return min(1.0, self.seconds / self.target_seconds)

    @property
    def done_keys(self) -> set[str]:
        return {c.key for c in self.usable}

    @property
    def ready(self) -> bool:
        return self.seconds >= self.target_seconds

    def summary(self) -> str:
        rejected = len(self.clips) - len(self.usable)
        tail = f", {rejected} needing another take" if rejected else ""
        return (f"{self.seconds / 60:.1f} of {self.target_seconds / 60:.0f} minutes"
                f" — {len(self.usable)} clips{tail}")

    # -- collecting ----------------------------------------------------------

    def add(self, key: str, text: str, audio: np.ndarray, rate: int,
            source: str = "recorded") -> Clip:
        """Store a clip at the training sample rate and measure it."""
        audio = to_target_rate(np.asarray(audio, dtype=np.float32).reshape(-1), rate)
        stats, issues = analyse(audio, TARGET_RATE, text)

        filename = f"{_safe(key)}.wav"
        write_wav(self.audio_dir / filename, audio, TARGET_RATE)

        clip = Clip(
            key=key, text=text, filename=filename,
            seconds=stats["seconds"], peak=stats["peak"],
            noise_floor=stats["noise_floor"], issues=issues, source=source,
        )
        # Re-recording a prompt replaces the previous take rather than adding a
        # second copy of the same sentence, which would skew training.
        self.clips = [c for c in self.clips if c.key != key]
        self.clips.append(clip)
        self.save()
        return clip

    def remove(self, key: str) -> bool:
        clip = next((c for c in self.clips if c.key == key), None)
        if clip is None:
            return False
        (self.audio_dir / clip.filename).unlink(missing_ok=True)
        self.clips.remove(clip)
        self.save()
        return True

    def import_file(self, path: Path, text: str, key: str | None = None) -> Clip:
        """Take an existing wav. Same checks as anything recorded here."""
        audio, rate = read_wav(path)
        return self.add(key or f"import_{path.stem}", text, audio, rate,
                        source="imported")

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        payload = {
            "name": self.name,
            "created": self.created,
            "target_seconds": self.target_seconds,
            "clips": [vars(c) for c in self.clips],
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    @classmethod
    def load(cls, root: Path) -> RecordingSession:
        data = json.loads((root / "session.json").read_text(encoding="utf-8"))
        session = cls(data.get("name", root.name), root=root,
                      target_minutes=data.get("target_seconds", 1800) / 60.0)
        session.created = data.get("created", time.time())
        session.clips = [Clip(**c) for c in data.get("clips", [])]
        return session

    @staticmethod
    def list_sessions() -> list[Path]:
        base = cache_dir() / "studio" / "datasets"
        if not base.is_dir():
            return []
        return sorted(p for p in base.iterdir() if (p / "session.json").exists())

    # -- output --------------------------------------------------------------

    def prepare(self, out_dir: Path | None = None) -> Path:
        """Write the dataset the trainer reads. Returns the CSV path."""
        usable = self.usable
        if not usable:
            raise RuntimeError("No usable clips; nothing to train on.")

        out_dir = out_dir or (self.root / "dataset")
        wav_dir = out_dir / "wav"
        if wav_dir.exists():
            shutil.rmtree(wav_dir)
        wav_dir.mkdir(parents=True, exist_ok=True)

        for clip in usable:
            shutil.copy2(self.audio_dir / clip.filename, wav_dir / clip.filename)

        csv_path = out_dir / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            for clip in usable:
                # Single speaker: wav|text. The trainer resolves the wav relative
                # to the CSV's own directory.
                writer.writerow([f"wav/{clip.filename}", clip.text])

        log.info("prepared %d clips (%.1f min) at %s",
                 len(usable), self.seconds / 60, csv_path)
        return csv_path


def _safe(name: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "_" for c in name.strip()]
    return "".join(keep)[:64] or "voice"
