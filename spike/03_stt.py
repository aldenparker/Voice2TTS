"""Spike 3: does faster-whisper actually run on this Blackwell (sm_120) GPU?

This is the highest-risk question in the whole design, so it runs ONE backend per
invocation -- a single script trying every fallback in sequence just looks hung.

    python -u 03_stt.py cuda float16
    python -u 03_stt.py cpu int8

Assumes the model is already in the HF cache (see 00_fetch_model.py); a cold
download is ~250 MB and will otherwise stall with no output.
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cuda import prepare_cuda  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
COMPUTE = sys.argv[2] if len(sys.argv) > 2 else "float16"
MODEL = "small.en"

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "spike" / "out" / "tts_sample.wav"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, rate


def main() -> None:
    print(f"=== {MODEL} on {DEVICE} / {COMPUTE} ===", flush=True)

    if DEVICE == "cuda":
        print("preparing CUDA libraries:", flush=True)
        prepare_cuda(verbose=True)

    from faster_whisper import WhisperModel

    if not WAV.exists():
        sys.exit(f"missing {WAV} -- run 02_tts.py first")

    audio, rate = load_wav(WAV)
    if rate != 16000:
        import soxr

        audio = soxr.resample(audio, rate, 16000)
    dur = len(audio) / 16000
    print(f"test audio: {dur:.2f} s", flush=True)

    print("loading model...", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE, local_files_only=True)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s", flush=True)

    # First call pays lazy CUDA context + kernel autotune cost; discard it.
    print("warmup pass...", flush=True)
    t0 = time.perf_counter()
    list(model.transcribe(audio, language="en", beam_size=1)[0])
    print(f"  warmup {time.perf_counter() - t0:.2f} s", flush=True)

    print("timed pass...", flush=True)
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    text = " ".join(s.text.strip() for s in segments)
    t_run = time.perf_counter() - t0

    print(f"\nRESULT  {DEVICE}/{COMPUTE}: {t_run * 1000:.1f} ms "
          f"for {dur:.2f} s audio  ({dur / t_run:.1f}x realtime)", flush=True)
    print(f"text: {text}", flush=True)


if __name__ == "__main__":
    main()
