"""Fan synthesized speech out to several audio devices at once.

Separate devices run on independent clocks and can differ in sample rate and channel
count, so they cannot share a stream. Each target owns a stream, a resampler and a
gain, and the same source chunks are pushed into all of them.

Streams stay open while running and emit silence when idle: opening a WASAPI stream
costs tens of milliseconds, which would land squarely on the user's speech latency.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soxr

from .config import AudioConfig, OutputTarget
from .devices import Device, resolve_output

log = logging.getLogger(__name__)


class TargetStream:
    """One output device, fed independently of the others."""

    def __init__(self, device: Device, gain: float, src_rate: int, blocksize: int):
        self.device = device
        self.gain = gain
        self.src_rate = src_rate
        self.rate = device.rate
        self.channels = min(2, max(1, device.max_out))
        self.blocksize = blocksize

        self._buf: deque[np.ndarray] = deque()
        self._head = 0
        self._pending = 0            # frames buffered, at this device's rate
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._producing = False  # True while the synthesizer is still feeding us
        self.underruns = 0
        # Most recent block peak, decayed so a meter falls smoothly rather than
        # snapping to zero between words.
        self.peak = 0.0

    @property
    def name(self) -> str:
        return self.device.name

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def open(self) -> None:
        if self._stream is not None:
            return
        if self.rate != self.src_rate:
            self._resampler = soxr.ResampleStream(
                self.src_rate, self.rate, 1, dtype="float32", quality="HQ"
            )
        self._stream = sd.OutputStream(
            device=self.device.index,
            samplerate=self.rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        log.info("output open: %s @ %d Hz %dch", self.name, self.rate, self.channels)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.warning("error closing %s: %s", self.name, exc)
        self._resampler = None
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._head = 0
            self._pending = 0

    def set_producing(self, producing: bool) -> None:
        self._producing = producing

    def push(self, mono: np.ndarray) -> None:
        """Resample on the producer side and enqueue for this device's callback."""
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        if not len(mono):
            return
        with self._lock:
            self._buf.append(np.ascontiguousarray(mono, dtype=np.float32))
            self._pending += len(mono)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("output status on %s: %s", self.name, status)

        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._buf:
                head = self._buf[0]
                take = min(frames - filled, len(head) - self._head)
                out[filled:filled + take] = head[self._head:self._head + take]
                filled += take
                self._head += take
                if self._head >= len(head):
                    self._buf.popleft()
                    self._head = 0
            self._pending = max(0, self._pending - filled)

        # A short block only means something went wrong if the synthesizer was still
        # feeding us. Every utterance ends on a partial block as the buffer drains,
        # and idle blocks are silence by design -- counting either would make this
        # number useless as a diagnostic.
        if self._producing and filled < frames:
            self.underruns += 1

        if filled:
            np.clip(out * self.gain, -1.0, 1.0, out=out)
            block_peak = float(np.abs(out[:filled]).max())
        else:
            block_peak = 0.0
        # Decay towards the new value so the meter reads smoothly at ~10 Hz refresh.
        self.peak = max(block_peak, self.peak * 0.75)

        outdata[:] = out.reshape(-1, 1) if self.channels == 1 else np.repeat(
            out.reshape(-1, 1), self.channels, axis=1
        )


class OutputSink:
    """Owns the set of TargetStreams and mirrors audio into all of them."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.targets: list[TargetStream] = []
        self.failures: list[tuple[str, str]] = []  # (device match, reason)
        self._src_rate = 22050
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        """True while any device still has audio queued."""
        return any(t.pending > 0 for t in self.targets)

    def configure(self, wanted: list[OutputTarget], src_rate: int) -> list[tuple[str, str]]:
        """(Re)open streams for the enabled targets. Returns any failures."""
        with self._lock:
            self._close_locked()
            self._src_rate = src_rate
            self.failures = []

            for spec in wanted:
                if not spec.enabled:
                    continue
                device = resolve_output(spec.match, all_apis=not self.cfg.prefer_wasapi)
                if device is None:
                    self.failures.append((spec.label, "device not found"))
                    log.warning("output %r not found; skipping", spec.label)
                    continue
                stream = TargetStream(device, spec.gain, src_rate, self.cfg.output_blocksize)
                try:
                    stream.open()
                except Exception as exc:  # noqa: BLE001 - one bad device must not
                    # take down the others; a missing VB-CABLE is the common case.
                    self.failures.append((spec.label, str(exc)))
                    log.error("could not open %s: %s", spec.label, exc)
                    continue
                self.targets.append(stream)

            if not self.targets:
                log.warning("no usable output devices")
            return list(self.failures)

    def levels(self) -> dict[str, float]:
        """Current peak per open target, for meters in the UI."""
        return {t.name: t.peak for t in self.targets}

    def set_gain(self, match: str, gain: float) -> None:
        for t in self.targets:
            if match.lower() in t.name.lower():
                t.gain = gain

    def write(self, mono: np.ndarray) -> None:
        for t in self.targets:
            t.push(mono)

    def begin_utterance(self) -> None:
        """Mark the start of synthesis, so buffer starvation counts as an underrun."""
        for t in self.targets:
            t.set_producing(True)

    def end_utterance(self) -> None:
        """Synthesis finished; remaining short blocks are just the drain tail."""
        for t in self.targets:
            t.set_producing(False)

    def clear(self) -> None:
        """Drop queued audio -- used to interrupt speech."""
        for t in self.targets:
            t.set_producing(False)
            t.clear()

    def wait_drain(self, timeout: float = 30.0) -> bool:
        """Block until every device has played out. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.active:
                # Let the final buffered block actually reach the DAC.
                time.sleep(self.cfg.output_blocksize / max(1, self._src_rate))
                return True
            time.sleep(0.005)
        return False

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        for t in self.targets:
            t.close()
        self.targets = []
