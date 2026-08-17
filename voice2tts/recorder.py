"""Capture single clips for the Voice Studio, at the microphone's own rate.

This deliberately does NOT reuse MicCapture. That path resamples to 16 kHz
inside the PortAudio callback, at draft quality, because it feeds speech
recognition and nothing there benefits from more. Recording training material
through it would throw away everything above 8 kHz before the clip was ever
saved, and permanently cap the trained voice -- an invisible loss, since the
clip would still sound fine in a preview.

So blocks are stored exactly as the device delivers them and resampled once,
offline, at the highest quality, by dataset.to_target_rate().
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd

from .devices import Device

log = logging.getLogger(__name__)

# Nothing useful is said in half a minute of one prompt, and an unattended
# recorder left running should not eat memory indefinitely.
MAX_CLIP_SECONDS = 120.0


class ClipRecorder:
    """One clip at a time: start, watch the level, stop, get audio back."""

    def __init__(self, device: Device, max_seconds: float = MAX_CLIP_SECONDS):
        self.device = device
        self.rate = device.rate
        self.channels = min(2, max(1, device.max_in))
        self.max_seconds = max_seconds

        self.peak = 0.0          # most recent block, for the level meter
        self.clipped = False     # sticky: any block that touched full scale
        self.overran = False     # hit max_seconds and stopped itself

        self._blocks: list[np.ndarray] = []
        self._frames = 0
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._started = 0.0

    # -- state ---------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._frames / float(self.rate) if self.rate else 0.0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._blocks.clear()
            self._frames = 0
        self.peak = 0.0
        self.clipped = False
        self.overran = False
        self._started = time.monotonic()

        self._stream = sd.InputStream(
            device=self.device.index,
            samplerate=self.rate,
            channels=self.channels,
            dtype="float32",
            blocksize=0,
            callback=self._callback,
        )
        self._stream.start()
        log.info("recording from %s @ %d Hz %dch",
                 self.device.name, self.rate, self.channels)

    def stop(self) -> tuple[np.ndarray, int]:
        """Stop and return (mono float32 at the device rate, rate)."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.warning("error closing recorder stream: %s", exc)

        with self._lock:
            blocks, self._blocks = self._blocks, []
            self._frames = 0
        if not blocks:
            return np.zeros(0, dtype=np.float32), self.rate
        audio = np.concatenate(blocks)
        log.info("recorded %.2fs at %d Hz", len(audio) / self.rate, self.rate)
        return audio, self.rate

    def cancel(self) -> None:
        """Throw the take away. Used when a prompt is skipped mid-recording."""
        self.stop()

    # -- audio ---------------------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("recorder status: %s", status)

        # Copy: PortAudio reuses this buffer for the next block.
        block = indata.copy()
        if block.ndim > 1 and block.shape[1] > 1:
            block = block.mean(axis=1)
        block = block.reshape(-1).astype(np.float32)

        peak = float(np.abs(block).max()) if block.size else 0.0
        self.peak = peak
        if peak >= 0.999:
            # Worth surfacing live: clipping cannot be repaired afterwards, and
            # the analyser would only report it once the take was already spent.
            self.clipped = True

        with self._lock:
            if self._frames / float(self.rate) >= self.max_seconds:
                self.overran = True
                return
            self._blocks.append(block)
            self._frames += len(block)
