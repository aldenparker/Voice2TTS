"""Prove the virtual cable actually carries audio, without opening Discord.

A virtual cable is a loop: what is played into its playback endpoint comes back out
of its recording endpoint. So we can play a tone into one and listen on the other,
and know for certain whether the path Discord will use is working.

This is the difference between "Discord can't hear me" being a mystery and being a
specific, named failure. It also catches the case that is otherwise invisible: the
right cable but the wrong channel, where audio flows perfectly into a device nobody
is listening to.

Detection is done on a single frequency bin rather than raw loudness, so background
noise on a real microphone cannot be mistaken for the test signal.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import devices

log = logging.getLogger(__name__)

TONE_HZ = 997.0          # prime-ish, so it never lands on a harmonic of mains hum
TONE_SECONDS = 1.0
BASELINE_SECONDS = 0.25
RATE = 48000
AMPLITUDE = 0.25         # audible but not alarming if it reaches real speakers

# The tone must stand this far above the noise floor to count as detected.
DETECT_MARGIN_DB = 12.0
# ...and be audible in absolute terms, so a dead-silent loop is not "detected".
DETECT_FLOOR_DB = -60.0


@dataclass
class LoopbackResult:
    ok: bool
    message: str
    tone_db: float = -120.0
    noise_db: float = -120.0
    output_name: str = ""
    input_name: str = ""

    @property
    def margin_db(self) -> float:
        return self.tone_db - self.noise_db

    @property
    def detail(self) -> str:
        return (f"tone {self.tone_db:.0f} dB, noise floor {self.noise_db:.0f} dB, "
                f"margin {self.margin_db:.0f} dB")


def _tone_level_db(samples: np.ndarray, rate: int, freq: float) -> float:
    """Energy at `freq` only, in dBFS. Ignores everything else in the signal."""
    if len(samples) < 256:
        return -120.0
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / rate)
    bin_index = int(np.argmin(np.abs(freqs - freq)))
    # Sum a couple of neighbouring bins so a slight clock difference between the
    # two endpoints does not move the tone out of the bin we measure.
    lo, hi = max(0, bin_index - 2), min(len(spectrum), bin_index + 3)
    magnitude = float(np.sqrt(np.sum(spectrum[lo:hi] ** 2)))
    normalised = magnitude / (len(windowed) / 4.0)
    return 20.0 * np.log10(max(normalised, 1e-9))


class _Recorder:
    """Captures mono float32 from a device until stopped."""

    def __init__(self, device: devices.Device, rate: int):
        self.device = device
        self.rate = rate
        self._chunks: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> _Recorder:
        self._stream = sd.InputStream(
            device=self.device.index,
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.debug("recorder teardown: %s", exc)
            self._stream = None

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            log.debug("loopback input status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.reshape(-1)
        self._chunks.put(mono.copy())

    def drain(self) -> np.ndarray:
        parts = []
        while True:
            try:
                parts.append(self._chunks.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def verify(output_match: str, input_match: str, progress=None) -> LoopbackResult:
    """Play a tone into `output_match` and listen for it on `input_match`."""

    def say(msg: str) -> None:
        log.info("loopback: %s", msg)
        if progress:
            progress(msg)

    out_dev = devices.resolve_output(output_match)
    in_dev = devices.resolve_input(input_match)
    if out_dev is None:
        return LoopbackResult(False, f"Playback device not found: {output_match}")
    if in_dev is None:
        return LoopbackResult(False, f"Recording device not found: {input_match}")

    rate = RATE if max(out_dev.rate, in_dev.rate) >= RATE else in_dev.rate
    channels = min(2, max(1, out_dev.max_out))

    try:
        with _Recorder(in_dev, rate) as rec:
            say("Measuring the noise floor...")
            threading.Event().wait(BASELINE_SECONDS)
            baseline = rec.drain()

            say("Playing test tone...")
            t = np.arange(int(TONE_SECONDS * rate), dtype=np.float32) / rate
            tone = (AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
            # Fade the edges so the click of an abrupt start does not smear energy
            # across the spectrum and inflate the noise measurement.
            fade = int(0.01 * rate)
            envelope = np.ones_like(tone)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            tone *= envelope

            block = np.repeat(tone.reshape(-1, 1), channels, axis=1)
            with sd.OutputStream(device=out_dev.index, samplerate=rate,
                                 channels=channels, dtype="float32") as ostream:
                ostream.write(np.ascontiguousarray(block))

            threading.Event().wait(0.15)  # let the tail arrive
            captured = rec.drain()
    except Exception as exc:
        log.exception("loopback failed")
        return LoopbackResult(False, f"Could not run the test: {exc}",
                              output_name=out_dev.name, input_name=in_dev.name)

    noise_db = _tone_level_db(baseline, rate, TONE_HZ)
    tone_db = _tone_level_db(captured, rate, TONE_HZ)
    result = LoopbackResult(
        ok=False, message="", tone_db=tone_db, noise_db=noise_db,
        output_name=out_dev.name, input_name=in_dev.name,
    )

    if len(captured) < rate * 0.2:
        result.message = (
            f"{in_dev.name} produced almost no audio. It may be disabled in Windows "
            "Sound settings, or in exclusive use by another application."
        )
        return result

    if tone_db >= DETECT_FLOOR_DB and result.margin_db >= DETECT_MARGIN_DB:
        result.ok = True
        result.message = (
            f"Signal confirmed. Audio played into {out_dev.name} arrives at "
            f"{in_dev.name}, so this is the device to select in Discord."
        )
        return result

    result.message = (
        f"No test tone came back from {in_dev.name}.\n\n"
        "The cable channel is probably mismatched -- the recording device that "
        f"pairs with {out_dev.name} may be a different one. Check the channel "
        "number, and that neither endpoint is disabled in Windows Sound settings."
    )
    return result


@dataclass
class ScanHit:
    input_name: str
    tone_db: float
    noise_db: float

    @property
    def margin_db(self) -> float:
        return self.tone_db - self.noise_db


def scan(output_match: str, progress=None, limit: int = 16) -> list[ScanHit]:
    """Play one tone and report every recording device that hears it.

    Naming conventions cannot be trusted for router products, where which capture
    endpoint carries the audio depends on the user's routing rather than on the
    device name. Measuring is the only honest answer, and it costs one tone.

    All candidates are recorded simultaneously, so this takes about as long as a
    single test regardless of how many devices there are.
    """

    def say(msg: str) -> None:
        log.info("scan: %s", msg)
        if progress:
            progress(msg)

    out_dev = devices.resolve_output(output_match)
    if out_dev is None:
        raise RuntimeError(f"Playback device not found: {output_match}")

    candidates = devices.list_inputs()[:limit]
    if not candidates:
        return []

    rate = RATE
    channels = min(2, max(1, out_dev.max_out))
    recorders: list[tuple[devices.Device, _Recorder]] = []
    opened: list[_Recorder] = []

    say(f"Listening on {len(candidates)} recording devices...")
    try:
        for dev in candidates:
            rec = _Recorder(dev, rate)
            try:
                rec.__enter__()
            except Exception as exc:  # noqa: BLE001 - a busy device is not fatal
                log.debug("cannot open %s: %s", dev.name, exc)
                continue
            opened.append(rec)
            recorders.append((dev, rec))

        if not recorders:
            return []

        threading.Event().wait(BASELINE_SECONDS)
        baselines = {dev.name: rec.drain() for dev, rec in recorders}

        say("Playing test tone...")
        t = np.arange(int(TONE_SECONDS * rate), dtype=np.float32) / rate
        tone = (AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
        fade = int(0.01 * rate)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        block = np.repeat(tone.reshape(-1, 1), channels, axis=1)
        with sd.OutputStream(device=out_dev.index, samplerate=rate,
                             channels=channels, dtype="float32") as ostream:
            ostream.write(np.ascontiguousarray(block))
        threading.Event().wait(0.15)

        hits = []
        for dev, rec in recorders:
            captured = rec.drain()
            tone_db = _tone_level_db(captured, rate, TONE_HZ)
            noise_db = _tone_level_db(baselines.get(dev.name, np.zeros(0)), rate, TONE_HZ)
            hit = ScanHit(dev.name, tone_db, noise_db)
            if tone_db >= DETECT_FLOOR_DB and hit.margin_db >= DETECT_MARGIN_DB:
                hits.append(hit)
        hits.sort(key=lambda h: h.tone_db, reverse=True)
        say(f"{len(hits)} device(s) received the tone")
        return hits
    finally:
        for rec in opened:
            rec.__exit__(None, None, None)


def verify_cable(info, progress=None) -> LoopbackResult:
    """Verify a detected cable end to end, explaining router failures properly."""
    if not info.input_name:
        return LoopbackResult(
            False,
            "The matching recording device could not be determined, so the loop "
            "cannot be tested automatically.",
            output_name=info.output_name,
        )

    result = verify(info.output_name, info.input_name, progress=progress)
    if result.ok or not info.is_router:
        return result

    # For a router, silence is expected unless the application is running and
    # configured, so say that rather than blaming the channel number.
    if not info.app_running:
        result.message = (
            f"No signal, because {info.product} is not running.\n\n"
            f"{info.product} is an audio router: {info.output_name} is an input "
            f"port on it, not one end of a fixed cable. Start {info.product} and "
            f"route that input to {info.input_name}, then test again.\n\n"
            "If you want something that works without an application running, "
            "install VB-CABLE instead."
        )
    else:
        result.message = (
            f"{info.product} is running but no signal came back.\n\n"
            f"Open {info.product} and route {info.output_name} through to "
            f"{info.input_name}. Use 'Find the right device' to see which "
            "recording device, if any, currently receives this audio."
        )
    return result
