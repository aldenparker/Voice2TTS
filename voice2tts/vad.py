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

        # The relaxation window. Below _soft_windows nothing changes at all, and
        # soft_endpoint_s = 0 turns the whole thing off -- including the segment
        # ceiling, which is part of the same feature. Leaving the ceiling armed
        # when the relaxation is disabled would give a third behaviour that is
        # nobody's intent.
        enabled = cfg.soft_endpoint_s > 0 and cfg.max_segment_s > 0
        self._soft_windows = (int(cfg.soft_endpoint_s * 1000 / WINDOW_MS)
                              if enabled else 0)
        self._segment_windows = (max(self._soft_windows + 1,
                                     int(cfg.max_segment_s * 1000 / WINDOW_MS))
                                 if enabled else 0)
        self._floor_silence_windows = max(1, cfg.min_silence_floor_ms // WINDOW_MS)
        # How far back a forced cut looks for a quiet moment. One second is long
        # enough to contain a gap between words at any speaking rate.
        self._recent_windows = max(1, 1000 // WINDOW_MS)
        self.reset()

    def _required_silence(self) -> int:
        """How much quiet ends the utterance, given how long it has run.

        Constant until `soft_endpoint_s`, then eased down to the floor by
        `max_segment_s`. Easing rather than stepping so there is no length at
        which the behaviour changes abruptly.
        """
        if not self._soft_windows or len(self._buf) <= self._soft_windows:
            return self._min_silence_windows
        span = self._segment_windows - self._soft_windows
        through = min(1.0, (len(self._buf) - self._soft_windows) / max(1, span))
        eased = self._min_silence_windows - through * (
            self._min_silence_windows - self._floor_silence_windows)
        return max(self._floor_silence_windows, round(eased))

    def _quietest_recent(self, lookback: int) -> int:
        """Index into _buf of the quietest window in the recent past.

        Used when a segment has run to its ceiling without any pause. Cutting on
        a fixed count would land mid-vowel; the quietest moment is at worst
        between two words.
        """
        if not self._probs:
            return len(self._buf)
        recent = list(self._probs)[-lookback:]
        offset = len(self._buf) - len(recent)
        return offset + int(min(range(len(recent)), key=recent.__getitem__))

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self._buf: list[np.ndarray] = []
        # Probabilities alongside _buf, so a forced cut can be placed at the
        # quietest moment rather than wherever the counter happened to land.
        self._probs: deque[float] = deque(maxlen=self._segment_windows + 1)
        self._triggered = False
        self._speech_windows = 0
        self._silence_windows = 0
        self.last_prob = 0.0

    @property
    def active(self) -> bool:
        return self._triggered

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
                    # One entry per window in _buf, or _quietest_recent would
                    # index into the wrong place. The pre-roll predates speech,
                    # so it is not a candidate cut point: score it high.
                    self._probs = deque([1.0] * len(self._buf),
                                        maxlen=self._segment_windows + 1)
                    self._preroll.clear()
            else:
                self._speech_windows = 0
            return None

        self._buf.append(window)
        self._probs.append(prob)

        if speech:
            self._silence_windows = 0
        else:
            self._silence_windows += 1
            if self._silence_windows >= self._required_silence():
                return self._finish()

        # No pause is coming. Cut at the quietest recent moment and carry the
        # rest forward, so continuous speech is still delivered in pieces
        # instead of being held until the speaker stops.
        if self._segment_windows and len(self._buf) >= self._segment_windows:
            cut = self._quietest_recent(self._recent_windows)
            log.info("segment ceiling reached; cutting at the quietest of the "
                     "last %d ms", self._recent_windows * WINDOW_MS)
            return self._split_at(cut)

        if len(self._buf) >= self._max_windows:
            log.info("utterance hit max length; cutting")
            return self._finish()
        return None

    def _split_at(self, index: int) -> np.ndarray | None:
        """Emit _buf[:index] and keep the remainder as the utterance in progress.

        Not _finish(): the speaker has not stopped, so dropping the tail would
        lose whatever they said after the cut point.
        """
        index = max(1, min(index, len(self._buf)))
        head, tail = self._buf[:index], self._buf[index:]

        self._buf = tail
        self._probs = deque(list(self._probs)[index:],
                            maxlen=self._probs.maxlen)
        self._silence_windows = 0
        # _triggered stays True: this is the same stretch of speech continuing.

        audio = np.concatenate(head) if head else None
        if audio is None or len(audio) < self._min_speech_windows * WINDOW:
            return None
        return audio

    def flush(self) -> np.ndarray | None:
        """End any in-progress utterance, e.g. when stopping or switching modes."""
        return self._finish() if self._triggered and self._buf else None

    def _finish(self) -> np.ndarray | None:
        # Trim the detected trailing silence but leave speech_pad_ms of it, so the
        # final consonant is not chopped.
        keep = len(self._buf) - max(0, self._silence_windows - self._pad_windows)
        chunks = self._buf[: max(1, keep)]
        audio = np.concatenate(chunks) if chunks else None

        self._buf = []
        self._probs.clear()
        self._triggered = False
        self._speech_windows = 0
        self._silence_windows = 0
        self._preroll.clear()

        if audio is None or len(audio) < self._min_speech_windows * WINDOW:
            return None
        return audio
