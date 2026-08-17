"""Microphone capture, normalized to 16 kHz mono float32 windows.

Whisper and Silero both want 16 kHz mono. Windows shared-mode devices are almost
always 44.1/48 kHz stereo, and asking WASAPI for 16 kHz directly tends to fail or
silently engage a poor resampler, so we open at the device's native rate and
downsample ourselves with soxr.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import soxr

from .devices import Device
from .vad import SAMPLE_RATE, WINDOW

log = logging.getLogger(__name__)


class MicCapture:
    """Streams `WINDOW`-sample chunks of 16 kHz mono audio onto a queue.

    The PortAudio callback only resamples and enqueues; all analysis happens on the
    consumer thread. Blocking inside the callback would cause input glitches.
    """

    def __init__(self, device: Device, queue_windows: int = 200):
        self.device = device
        self.rate = device.rate
        self.channels = min(2, max(1, device.max_in))
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_windows)
        self.dropped = 0
        self.peak = 0.0  # most recent block peak, for the level meter in the UI

        self._stream: sd.InputStream | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        # Set when the device goes away mid-session (USB microphone unplugged is the
        # common case). Without this the stream just stops delivering audio and the
        # app looks alive while hearing nothing.
        self.failed = False
        self.failure_reason = ""
        self._last_callback = time.monotonic()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        if self.rate != SAMPLE_RATE:
            self._resampler = soxr.ResampleStream(
                self.rate, SAMPLE_RATE, 1, dtype="float32", quality="QQ"
            )
        self.failed = False
        self.failure_reason = ""
        self._last_callback = time.monotonic()
        self._stream = sd.InputStream(
            device=self.device.index,
            samplerate=self.rate,
            channels=self.channels,
            dtype="float32",
            blocksize=0,  # let PortAudio pick its optimal block size
            callback=self._callback,
            finished_callback=self._on_finished,
        )
        self._stream.start()
        log.info(
            "capture started: %s @ %d Hz %dch -> %d Hz mono",
            self.device.name, self.rate, self.channels, SAMPLE_RATE,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.warning("error closing capture stream: %s", exc)
        self._resampler = None
        self._tail = np.zeros(0, dtype=np.float32)
        self.drain()
        log.info("capture stopped")

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    # -- health ---------------------------------------------------------------

    def check_alive(self, silence_timeout: float = 5.0) -> bool:
        """False if the device has stopped delivering audio.

        PortAudio does not always raise when a device disappears -- sometimes the
        callback simply stops being called. Poll this to notice.
        """
        if self.failed:
            return False
        if self._stream is None:
            return True
        if time.monotonic() - self._last_callback > silence_timeout:
            self._mark_failed("device stopped delivering audio")
            return False
        return True

    def _mark_failed(self, reason: str) -> None:
        if not self.failed:
            self.failed = True
            self.failure_reason = reason
            log.error("capture failed on %s: %s", self.device.name, reason)

    def _on_finished(self) -> None:
        """PortAudio calls this when the stream stops, including unexpectedly."""
        if self._stream is not None and not self.failed:
            self._mark_failed("audio stream closed unexpectedly")

    # -- audio thread ---------------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._last_callback = time.monotonic()
        if status:
            log.debug("input status: %s", status)
            if getattr(status, "input_overflow", False):
                pass  # transient; the ring buffer below already handles it

        mono = indata[:, 0] if indata.ndim > 1 and indata.shape[1] > 1 else indata.reshape(-1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if len(mono):
            self.peak = float(np.abs(mono).max())

        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
            if not len(mono):
                return

        with self._lock:
            buf = np.concatenate([self._tail, mono]) if len(self._tail) else mono
            n_windows, rest = divmod(len(buf), WINDOW)
            self._tail = buf[len(buf) - rest:].copy() if rest else np.zeros(0, dtype=np.float32)
            windows = [buf[i * WINDOW:(i + 1) * WINDOW].copy() for i in range(n_windows)]

        for w in windows:
            try:
                self.queue.put_nowait(w)
            except queue.Full:
                # Consumer has stalled. Drop the oldest window so we stay near
                # realtime instead of accumulating unbounded latency.
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(w)
                except queue.Empty:
                    pass
                self.dropped += 1
                if self.dropped % 100 == 1:
                    log.warning("capture queue full; dropped %d windows", self.dropped)
