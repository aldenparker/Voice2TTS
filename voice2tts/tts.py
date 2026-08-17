"""Text to speech via Piper.

Piper runs on CPU at ~50x realtime with ~60 ms to first chunk, so the GPU is left
entirely to Whisper. `synthesize()` yields one chunk per sentence, which we pass
straight through to the output sink -- playback of sentence 1 starts while sentence
2 is still being generated.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import TtsConfig
from .paths import find_voice, list_voices

log = logging.getLogger(__name__)


class PiperEngine:
    def __init__(self, cfg: TtsConfig):
        from piper import PiperVoice, SynthesisConfig

        self._SynthesisConfig = SynthesisConfig
        path = self._resolve_voice(cfg.voice)
        log.info("loading piper voice %s", path)
        t0 = time.perf_counter()
        self.voice = PiperVoice.load(path, use_cuda=False)
        self.voice_path = path
        log.info("piper loaded in %.2f s", time.perf_counter() - t0)

        self.rate = int(self.voice.config.sample_rate)
        self.num_speakers = int(getattr(self.voice.config, "num_speakers", 1) or 1)

        # A voice built in the Voice Designer carries an effects chain beside it.
        # Ordinary voices have no sidecar and pay nothing for this.
        from .designer import read_design

        self.design = read_design(path)
        if self.design is not None and self.design.is_neutral:
            self.design = None
        if self.design is not None:
            log.info("voice has a design chain: %s", self.design.to_dict())
        self.apply(cfg)

    @staticmethod
    def _resolve_voice(name: str) -> Path:
        path = find_voice(name)
        if path is not None:
            return path
        available = list_voices()
        if available:
            log.warning("voice %r not found; using %s", name, available[0].stem)
            return available[0]
        raise FileNotFoundError(
            f"No Piper voice found for {name!r} and no voices installed. "
            "Run scripts/fetch_models.py to download one."
        )

    def apply(self, cfg: TtsConfig) -> None:
        """Update synthesis parameters without reloading the model."""
        self.cfg = cfg
        # The designer's macros never touch length_scale. An earlier version
        # compensated the `size` resampling by asking the model to speak faster,
        # which does not work: Piper's duration is not proportional to
        # length_scale. The chain restores its own timing instead.
        self.syn = self._SynthesisConfig(
            # Single-speaker models reject an explicit speaker id.
            speaker_id=cfg.speaker_id if self.num_speakers > 1 else None,
            length_scale=cfg.length_scale,
            noise_scale=cfg.noise_scale,
            noise_w_scale=cfg.noise_w_scale,
            volume=cfg.volume,
            normalize_audio=cfg.normalize_audio,
        )

    def warmup(self) -> float:
        t0 = time.perf_counter()
        try:
            for _ in self.voice.synthesize("Ready.", syn_config=self.syn):
                pass
        except Exception as exc:  # noqa: BLE001 - warmup failure is not fatal
            log.warning("piper warmup failed: %s", exc)
        elapsed = time.perf_counter() - t0
        log.info("piper warmup %.2f s", elapsed)
        return elapsed

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 mono chunks at self.rate, one per sentence."""
        text = (text or "").strip()
        if not text:
            return
        t0 = time.perf_counter()
        first = True
        for chunk in self.voice.synthesize(text, syn_config=self.syn):
            if first:
                log.debug("piper first chunk in %.0f ms", (time.perf_counter() - t0) * 1000)
                first = False
            audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
            if chunk.sample_rate != self.rate:
                # Should not happen for a single voice, but a mismatch would play
                # back at the wrong pitch, so trust the chunk and re-tag.
                log.warning("chunk rate %d != voice rate %d", chunk.sample_rate, self.rate)
                self.rate = int(chunk.sample_rate)
            if self.design is not None:
                # Per sentence rather than per utterance, so the first sentence
                # still starts playing while the second is being generated. The
                # cost is that the reverb tail does not cross a sentence
                # boundary, which is inaudible at these tail lengths.
                from . import dsp

                audio = dsp.apply(audio, self.rate, self.design)
            yield audio

    def synth(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
