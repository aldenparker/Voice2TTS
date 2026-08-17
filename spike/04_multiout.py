"""Spike 4: fan one TTS stream out to N audio devices at once.

This is the prototype for the app's output layer. Key constraint: separate devices
run on independent clocks and may want different sample rates and channel counts,
so they cannot share a stream. Each target gets its own thread, queue, resampler
and gain, and we push the same source chunks into all of them.

    python -u 04_multiout.py                  # list selectable outputs
    python -u 04_multiout.py momentum         # play to one
    python -u 04_multiout.py momentum cable   # play to both at once

Match strings are case-insensitive substrings of the device name; WASAPI is
preferred because that is what the real app will use.
"""

import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soxr

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "spike" / "out" / "tts_sample.wav"

BLOCK = 1024          # frames pushed per write at the device's own rate
SENTINEL = object()   # queue marker: no more audio for this target


def wasapi_outputs() -> list[tuple[int, dict]]:
    """Output devices, WASAPI first since that is the app's target host API."""
    apis = sd.query_hostapis()
    devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
    devs.sort(key=lambda t: (apis[t[1]["hostapi"]]["name"] != "Windows WASAPI", t[0]))
    return devs


def resolve(pattern: str) -> tuple[int, dict]:
    for idx, dev in wasapi_outputs():
        if pattern.lower() in dev["name"].lower():
            return idx, dev
    raise SystemExit(f"no output device matching {pattern!r}")


class Target:
    """One output device, fed independently of all the others."""

    def __init__(self, idx: int, dev: dict, src_rate: int, gain: float = 1.0):
        self.idx = idx
        self.name = dev["name"]
        self.rate = int(dev["default_samplerate"])
        self.channels = min(2, dev["max_output_channels"])
        self.gain = gain
        self.src_rate = src_rate
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.underruns = 0
        self.first_write: float | None = None
        self._resampler = soxr.ResampleStream(src_rate, self.rate, 1, dtype="float32")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def push(self, mono: np.ndarray) -> None:
        """Resample on the producer side, then hand frames to this target's thread."""
        out = self._resampler.resample_chunk(mono)
        if len(out):
            self.q.put(out)

    def close(self) -> None:
        tail = self._resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        if len(tail):
            self.q.put(tail)
        self.q.put(SENTINEL)

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        stream = sd.OutputStream(
            device=self.idx,
            samplerate=self.rate,
            channels=self.channels,
            dtype="float32",
            blocksize=BLOCK,
        )
        with stream:
            buf = np.zeros(0, dtype=np.float32)
            done = False
            while not done:
                item = self.q.get()
                if item is SENTINEL:
                    done = True
                else:
                    buf = np.concatenate([buf, item])

                # Write in whole blocks; on the final pass, flush what is left.
                while len(buf) >= BLOCK or (done and len(buf)):
                    take = buf[:BLOCK]
                    buf = buf[BLOCK:]
                    if len(take) < BLOCK:
                        take = np.pad(take, (0, BLOCK - len(take)))
                    frames = np.clip(take * self.gain, -1.0, 1.0)
                    if self.channels > 1:
                        frames = np.repeat(frames[:, None], self.channels, axis=1)
                    if self.first_write is None:
                        self.first_write = time.perf_counter()
                    stream.write(np.ascontiguousarray(frames))
            self.underruns = stream.write_available  # informational only


def main() -> None:
    patterns = sys.argv[1:]
    if not patterns:
        apis = sd.query_hostapis()
        print("selectable outputs:")
        for idx, dev in wasapi_outputs():
            api = apis[dev["hostapi"]]["name"]
            print(f"  [{idx:>2}] {int(dev['default_samplerate']):>6} Hz "
                  f"{dev['max_output_channels']}ch  {api:<18} {dev['name']}")
        print("\npass one or more name fragments to play to them simultaneously")
        return

    if not WAV.exists():
        raise SystemExit(f"missing {WAV} -- run 02_tts.py first")

    with wave.open(str(WAV), "rb") as w:
        src_rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    print(f"source: {len(audio) / src_rate:.2f} s @ {src_rate} Hz\n")

    targets = []
    for p in patterns:
        idx, dev = resolve(p)
        t = Target(idx, dev, src_rate)
        targets.append(t)
        print(f"-> [{idx}] {t.name}  ({t.rate} Hz, {t.channels}ch)")

    for t in targets:
        t.start()

    print("\nplaying...")
    t0 = time.perf_counter()
    # Feed all targets from one source, in chunks, as a live pipeline would.
    for i in range(0, len(audio), BLOCK):
        chunk = audio[i : i + BLOCK]
        for t in targets:
            t.push(chunk)
    for t in targets:
        t.close()
    for t in targets:
        t.join()

    print(f"\ndone in {time.perf_counter() - t0:.2f} s")
    for t in targets:
        lat = (t.first_write - t0) * 1000 if t.first_write else float("nan")
        print(f"  {t.name}: first write at {lat:.1f} ms")


if __name__ == "__main__":
    main()
