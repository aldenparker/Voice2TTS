"""Spike 2: Piper TTS -> wav, with per-sentence streaming latency measured.

The number that matters for the app is time-to-first-chunk: how long after we hand
Piper the text before we have audio we can start pushing at the cable.
"""

import time
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
VOICE = ROOT / "models" / "voices" / "en_US-lessac-medium.onnx"
OUT = ROOT / "spike" / "out" / "tts_sample.wav"

TEXT = (
    "This is a test of the voice to text to speech pipeline. "
    "It should split into a few sentences. "
    "Each one gets synthesized separately so playback can start early."
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    voice = PiperVoice.load(VOICE, use_cuda=False)
    t_load = time.perf_counter() - t0
    print(f"model load        : {t_load * 1000:7.1f} ms")

    syn = SynthesisConfig(length_scale=1.0, volume=1.0, normalize_audio=True)

    # Warm the ONNX graph so the first real utterance isn't paying setup cost.
    t0 = time.perf_counter()
    list(voice.synthesize("warm up", syn_config=syn))
    print(f"warmup synth      : {(time.perf_counter() - t0) * 1000:7.1f} ms")

    chunks, marks = [], []
    t0 = time.perf_counter()
    for i, chunk in enumerate(voice.synthesize(TEXT, syn_config=syn)):
        marks.append((i, time.perf_counter() - t0, len(chunk.audio_float_array)))
        chunks.append(chunk.audio_float_array)
    total = time.perf_counter() - t0

    rate = chunk.sample_rate
    audio = np.concatenate(chunks)
    dur = len(audio) / rate

    print()
    for i, at, n in marks:
        print(f"  chunk {i}: ready at {at * 1000:7.1f} ms  ({n / rate:5.2f} s audio)")
    print()
    print(f"time to first chunk: {marks[0][1] * 1000:7.1f} ms   <-- startup latency")
    print(f"total synth time   : {total * 1000:7.1f} ms")
    print(f"audio duration     : {dur:7.2f} s")
    print(f"realtime factor    : {dur / total:7.2f}x")
    print(f"sample rate        : {rate} Hz")

    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
