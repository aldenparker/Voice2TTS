"""Speech recognition via faster-whisper."""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .config import SttConfig
from .cuda import cuda_available, prepare_cuda
from .modes import ComputeType, SttDevice, WhisperTask
from .paths import bundled_whisper, whisper_cache
from .vad import SAMPLE_RATE

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")


class Heard(StrEnum):
    """Why a recognition pass produced no text.

    It used to produce "" for all four of these, so the pipeline dropped the
    utterance without being able to say -- or even know -- whether the clip was
    too short, the user had said nothing, or Whisper had invented "thanks for
    watching" over a fan. Three of these are routine; which one it was decides
    whether it is worth telling anyone.
    """

    SPEECH = "speech"
    TOO_SHORT = "the clip was shorter than a tenth of a second"
    SILENCE = "the recogniser heard nothing"
    NOISE = "the recogniser returned a phrase it invents over silence"


@dataclass(frozen=True)
class TimedText:
    """A piece of transcript and where it sits in the audio, in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Recognition:
    """One pass over some audio: what was heard, and if nothing, why not."""

    text: str = ""
    timed: list[TimedText] = dataclasses.field(default_factory=list)
    why: Heard = Heard.SPEECH

    def __bool__(self) -> bool:
        return bool(self.text)


class WhisperEngine:
    def __init__(self, cfg: SttConfig, task: WhisperTask = WhisperTask.TRANSCRIBE):
        self.cfg = cfg
        # Not read from the config: which task Whisper is given follows from how
        # the user chose to translate, and storing it separately is exactly what
        # let the two disagree. See plan.SpeechPlan.whisper_task.
        self.task = task
        self.device, self.compute_type = self._resolve_backend(cfg)

        # Must happen before importing faster_whisper; see cuda.py for why.
        if self.device is SttDevice.CUDA:
            prepare_cuda()
        from faster_whisper import WhisperModel

        source, bundled = self._resolve_model(cfg.model)
        log.info(
            "loading whisper %s (%s) on %s/%s",
            cfg.model, "bundled" if bundled else "cache", self.device, self.compute_type,
        )
        t0 = time.perf_counter()
        # download_root is ignored for a local directory, but keeps any fetch out of
        # the roaming profile and inside our own cache.
        kwargs = {} if bundled else {"download_root": str(whisper_cache())}
        try:
            self.model = WhisperModel(
                source, device=self.device, compute_type=self.compute_type, **kwargs
            )
        except Exception as exc:
            if self.device is SttDevice.CPU:
                raise
            log.error("CUDA model load failed (%s); falling back to CPU", exc)
            self.device, self.compute_type = SttDevice.CPU, ComputeType.INT8
            self.model = WhisperModel(source, device="cpu", compute_type="int8", **kwargs)
        log.info("whisper loaded in %.2f s", time.perf_counter() - t0)

    @staticmethod
    def _resolve_model(name: str) -> tuple[str, bool]:
        """Prefer weights shipped in the build so a fresh install works offline."""
        local = bundled_whisper(name)
        if local is not None:
            return str(local), True
        return name, False

    @staticmethod
    def _resolve_backend(cfg: SttConfig) -> tuple[SttDevice, ComputeType]:
        device = cfg.device
        if device is SttDevice.AUTO:
            device = SttDevice.CUDA if cuda_available() else SttDevice.CPU
        elif device is SttDevice.CUDA and not cuda_available():
            log.warning("cuda requested but unavailable; using cpu")
            device = SttDevice.CPU

        compute = cfg.compute_type
        if compute is ComputeType.AUTO:
            compute = (ComputeType.FLOAT16 if device is SttDevice.CUDA
                       else ComputeType.INT8)
        return device, compute

    def warmup(self) -> float:
        """Run one throwaway inference.

        The first CUDA pass pays cuDNN kernel autotune -- measured at 6.7 s on this
        machine. Without this the user's first utterance appears to hang.
        """
        t0 = time.perf_counter()
        dummy = (np.random.randn(SAMPLE_RATE) * 0.01).astype(np.float32)
        try:
            list(self.model.transcribe(dummy, language=self._language,
                                       task=self.task.value, beam_size=1)[0])
        except Exception as exc:  # noqa: BLE001 - warmup failure is not fatal
            log.warning("warmup failed: %s", exc)
        elapsed = time.perf_counter() - t0
        log.info("whisper warmup %.2f s", elapsed)
        return elapsed

    @property
    def _language(self) -> str | None:
        """None asks Whisper to detect it, which is what "auto" means here."""
        return None if self.cfg.language == "auto" else self.cfg.language

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio. Returns "" if nothing usable.

        For callers that genuinely do not care why -- the streaming recogniser
        runs several passes a second and expects most of them to be empty.
        Anything user-facing should call recognise() and say which of the four
        things happened.
        """
        return self.recognise(audio).text

    def transcribe_timed(self, audio: np.ndarray) -> tuple[str, list[TimedText]]:
        """Transcribe, and also say where each piece sits in the audio."""
        result = self.recognise(audio)
        return result.text, result.timed

    def recognise(self, audio: np.ndarray) -> Recognition:
        """Transcribe, saying where each piece sits and why there is no text.

        Streaming needs the timings: it drops audio it has already committed,
        and can only cut on a boundary the recogniser itself drew. Sentence mode
        ignores them and reads `why` instead.
        """
        if audio is None or len(audio) < SAMPLE_RATE // 10:
            # Not logged: in streaming this happens several times a second by
            # design. The caller decides whether it is worth mentioning.
            return Recognition(why=Heard.TOO_SHORT)
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=self._language,
            # "translate" makes Whisper emit English whatever was spoken. It
            # costs nothing extra -- the same forward pass, a different token.
            task=self.task.value,
            beam_size=self.cfg.beam_size,
            # We already segmented with Silero; Whisper's own VAD would double-trim.
            vad_filter=False,
            # Each utterance is independent -- carrying context across them makes
            # Whisper invent continuations of whatever was said before.
            condition_on_previous_text=False,
        )
        # The generator is consumed once, so collect timings on the same pass.
        timed = [TimedText(text=s.text.strip(), start=float(s.start),
                           end=float(s.end))
                 for s in segments if s.text.strip()]
        text = " ".join(t.text for t in timed).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            return Recognition(why=Heard.SILENCE)
        if self._is_noise(text):
            log.info("dropped likely hallucination: %r", text)
            return Recognition(why=Heard.NOISE)

        log.info(
            "transcribed %.2f s audio in %.0f ms: %r",
            len(audio) / SAMPLE_RATE, (time.perf_counter() - t0) * 1000, text,
        )
        return Recognition(text=text, timed=timed)

    def _is_noise(self, text: str) -> bool:
        if len(text) < self.cfg.min_chars:
            return True
        normalized = _PUNCT.sub("", text).strip().lower()
        if not normalized:
            return True
        return normalized in {
            _PUNCT.sub("", p).strip().lower() for p in self.cfg.drop_phrases
        }
