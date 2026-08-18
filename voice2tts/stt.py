"""Speech recognition via faster-whisper."""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from .config import SttConfig
from .cuda import cuda_available, prepare_cuda
from .paths import bundled_whisper, whisper_cache
from .vad import SAMPLE_RATE

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")


class WhisperEngine:
    def __init__(self, cfg: SttConfig):
        self.cfg = cfg
        self.device, self.compute_type = self._resolve_backend(cfg)

        # Must happen before importing faster_whisper; see cuda.py for why.
        if self.device == "cuda":
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
            if self.device == "cpu":
                raise
            log.error("CUDA model load failed (%s); falling back to CPU", exc)
            self.device, self.compute_type = "cpu", "int8"
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
    def _resolve_backend(cfg: SttConfig) -> tuple[str, str]:
        device = cfg.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        elif device == "cuda" and not cuda_available():
            log.warning("cuda requested but unavailable; using cpu")
            device = "cpu"

        compute = cfg.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
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
                                       task=self.cfg.task, beam_size=1)[0])
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
        """Transcribe 16 kHz mono float32 audio. Returns "" if nothing usable."""
        if audio is None or len(audio) < SAMPLE_RATE // 10:
            return ""
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=self._language,
            # "translate" makes Whisper emit English whatever was spoken. It
            # costs nothing extra -- the same forward pass, a different token.
            task=self.cfg.task,
            beam_size=self.cfg.beam_size,
            # We already segmented with Silero; Whisper's own VAD would double-trim.
            vad_filter=False,
            # Each utterance is independent -- carrying context across them makes
            # Whisper invent continuations of whatever was said before.
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        text = re.sub(r"\s+", " ", text)

        if self._is_noise(text):
            log.info("dropped likely hallucination: %r", text)
            return ""

        log.info(
            "transcribed %.2f s audio in %.0f ms: %r",
            len(audio) / SAMPLE_RATE, (time.perf_counter() - t0) * 1000, text,
        )
        return text

    def _is_noise(self, text: str) -> bool:
        if len(text) < self.cfg.min_chars:
            return True
        normalized = _PUNCT.sub("", text).strip().lower()
        if not normalized:
            return True
        return normalized in {
            _PUNCT.sub("", p).strip().lower() for p in self.cfg.drop_phrases
        }
