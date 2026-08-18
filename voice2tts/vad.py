"""Silero VAD (ONNX) plus the endpointing state machine that turns probabilities
into discrete utterances.

Uses the raw ONNX model via onnxruntime rather than the `silero-vad` pip package,
which pulls in PyTorch (~2.5 GB) to run a 2 MB model.

Silero v5 is fixed at 512-sample windows for 16 kHz audio; feeding any other size
produces garbage rather than an error, so WINDOW is not configurable.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import onnxruntime as ort

from .config import VadConfig
from .paths import vad_model_path

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW = 512  # samples; mandated by the v5 model
WINDOW_MS = WINDOW * 1000 // SAMPLE_RATE  # 32 ms

# v5 expects the previous window's last 64 samples prepended, so each inference sees
# 576 samples. The ONNX input shape is dynamic, so passing a bare 512 does NOT error
# -- it silently returns near-zero probabilities for even loud speech. Verified here:
# with context, clear speech scores 0.93 mean; without, 0.003.
CONTEXT = 64


class SileroVad:
    def __init__(self, model_path=None):
        path = model_path or vad_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {path}. Run scripts/fetch_models.py."
            )
        opts = ort.SessionOptions()
        # One thread: this runs per 32 ms window and is trivially small. Letting ORT
        # spin up a pool per session just adds scheduling jitter on the audio path.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        """Speech probability for exactly WINDOW samples of float32 mono @ 16 kHz."""
        if len(window) != WINDOW:
            raise ValueError(f"expected {WINDOW} samples, got {len(window)}")
        window = window.astype(np.float32, copy=False)
        inp = np.concatenate([self._context, window]).reshape(1, -1)
        out, self._state = self._sess.run(
            None, {"input": inp, "state": self._state, "sr": self._sr}
        )
        self._context = window[-CONTEXT:].copy()
        return float(out[0][0])


class VadSegmenter:
    """Turns a stream of 512-sample windows into utterances.

    Emits an utterance once speech has been followed by `min_silence_ms` of quiet.
    A pre-roll buffer is retained so the first phoneme is not clipped -- VAD only
    fires once speech is already underway, so without pre-roll every utterance loses
    its onset.
    """

    def __init__(
        self,
        cfg: VadConfig,
        preroll_ms: int = 300,
        max_utterance_s: float = 30.0,
    ):
        self.cfg = cfg
        self.vad = SileroVad()
        self._preroll = deque(maxlen=max(1, preroll_ms // WINDOW_MS))
        self._max_windows = int(max_utterance_s * 1000 / WINDOW_MS)
        self._pad_windows = max(0, cfg.speech_pad_ms // WINDOW_MS)
        self._min_speech_windows = max(1, cfg.min_speech_ms // WINDOW_MS)
        self._min_silence_windows = max(1, cfg.min_silence_ms // WINDOW_MS)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self._buf: list[np.ndarray] = []
        self._triggered = False
        self._speech_windows = 0
        self._silence_windows = 0
        self.last_prob = 0.0

    @property
    def active(self) -> bool:
        return self._triggered

    def captured(self) -> list[np.ndarray]:
        """The windows held so far, including the pre-roll before the trigger.

        Detection needs a moment of sustained speech before it fires, so by the
        time `active` goes true the first syllables are already in here. A
        caller that starts collecting only from the next window loses the start
        of every utterance -- "I can reproduce it" arrives as "reproduce it".
        """
        return list(self._buf)

    def process(self, window: np.ndarray) -> np.ndarray | None:
        """Feed one window; returns a complete utterance when one ends."""
        prob = self.vad(window)
        self.last_prob = prob
        speech = prob >= self.cfg.threshold

        if not self._triggered:
            self._preroll.append(window)
            if speech:
                self._speech_windows += 1
                # Require sustained speech so a cough or key click cannot trigger.
                if self._speech_windows >= self._min_speech_windows:
                    self._triggered = True
                    self._silence_windows = 0
                    self._buf = list(self._preroll)
                    self._preroll.clear()
            else:
                self._speech_windows = 0
            return None

        self._buf.append(window)

        if speech:
            self._silence_windows = 0
        else:
            self._silence_windows += 1
            if self._silence_windows >= self._min_silence_windows:
                return self._finish()

        if len(self._buf) >= self._max_windows:
            log.info("utterance hit max length; cutting")
            return self._finish()
        return None

    def flush(self) -> np.ndarray | None:
        """End any in-progress utterance, e.g. when stopping or switching modes."""
        return self._finish() if self._triggered and self._buf else None

    def suspend(self) -> np.ndarray | None:
        """Stop listening without throwing away what was already captured.

        Used while the app is speaking: those windows would be its own voice
        coming back, so they must not be fed in -- but the speech captured
        BEFORE playback started is real and was going to be said. reset() used
        to be called here, which simply threw that away.

        Returns anything worth speaking, so it is not merely dropped.
        """
        pending = self._finish() if self._triggered and self._buf else None
        # The model's recurrent state is about the audio it has just heard, and
        # what follows will be a different moment. That much IS worth clearing.
        self.vad.reset()
        self._preroll.clear()
        self._triggered = False
        self._speech_windows = 0
        self._silence_windows = 0
        return pending

    def _finish(self) -> np.ndarray | None:
        # Trim the detected trailing silence but leave speech_pad_ms of it, so the
        # final consonant is not chopped.
        keep = len(self._buf) - max(0, self._silence_windows - self._pad_windows)
        chunks = self._buf[: max(1, keep)]
        audio = np.concatenate(chunks) if chunks else None

        self._buf = []
        self._triggered = False
        self._speech_windows = 0
        self._silence_windows = 0
        self._preroll.clear()

        if audio is None or len(audio) < self._min_speech_windows * WINDOW:
            return None
        return audio
