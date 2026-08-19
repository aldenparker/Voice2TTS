"""Does streaming recognition actually buy latency, and what does it cost?

Two numbers decide whether the mode is worth building, and neither is the
transcription time:

1. **Stabilisation latency** -- how long after a word is spoken before we can be
   sure enough of it to speak it. That is the real latency, and it is set by the
   agreement rule, not by how fast Whisper runs.
2. **The compute bill** -- the same audio is decoded once per interval, over a
   growing buffer, so cost grows with the square of utterance length.

The rule under test is LocalAgreement-2, which whisper-streaming uses: transcribe
a growing buffer every `interval` seconds and commit the longest common prefix of
the last two passes. Whatever two consecutive passes agree on has stopped moving.

    python spike/09_streaming.py                # both devices, 1.0 s interval
    python spike/09_streaming.py --device cpu
    python spike/09_streaming.py --interval 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RATE = 16000


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    """The LocalAgreement rule itself: what two passes both say, from the start.

    Pure, and the only part of streaming that needs no model -- which is exactly
    why it is worth having as a function rather than inline in a loop.
    """
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def words(text: str) -> list[str]:
    return text.split()


def run(engine, audio: np.ndarray, interval: float, label: str) -> dict:
    """Feed `audio` in real time, re-transcribing every `interval` seconds."""
    total_s = len(audio) / RATE
    print(f"\n=== {label}: {total_s:.1f}s of speech, "
          f"re-transcribing every {interval:.1f}s ===", flush=True)

    committed: list[str] = []
    previous: list[str] = []
    passes = 0
    decode_time = 0.0
    # When each word was first committed, against when it finished being spoken.
    stabilised_at: list[tuple[str, float]] = []
    over_budget = 0

    at = interval
    while at <= total_s + interval:
        buffer = audio[:int(min(at, total_s) * RATE)]
        if len(buffer) < RATE // 4:
            at += interval
            continue

        started = time.perf_counter()
        text = engine.transcribe(buffer)
        elapsed = time.perf_counter() - started
        decode_time += elapsed
        passes += 1
        if elapsed > interval:
            over_budget += 1

        current = words(text)
        agreed = common_prefix(previous, current)
        if len(agreed) > len(committed):
            for word in agreed[len(committed):]:
                stabilised_at.append((word, at))
            committed = agreed
        previous = current
        print(f"  t={at:5.1f}s  {elapsed * 1000:5.0f} ms  "
              f"committed {len(committed):3d} words  {' '.join(committed[-6:])}",
              flush=True)
        at += interval

    # The tail never gets a second agreeing pass, so a real implementation
    # commits it when the segment closes. Counting it as free would flatter the
    # result, so it is reported separately.
    tail = previous[len(committed):]

    return {
        "label": label,
        "passes": passes,
        "decode_time": decode_time,
        "audio_s": total_s,
        "committed": committed,
        "tail": tail,
        "stabilised_at": stabilised_at,
        "over_budget": over_budget,
    }


def report(result: dict, interval: float) -> None:
    audio_s = result["audio_s"]
    decode = result["decode_time"]
    print(f"\n  passes            : {result['passes']}")
    print(f"  total decode time : {decode:.1f}s for {audio_s:.1f}s of audio "
          f"({decode / audio_s:.2f}x realtime)")
    print(f"  passes over budget: {result['over_budget']} "
          f"(a pass slower than {interval:.1f}s means the buffer runs away)")
    print(f"  committed         : {len(result['committed'])} words")
    print(f"  held to the end   : {len(result['tail'])} words "
          f"({' '.join(result['tail'])[:50]})")

    # Compare against the segmented mode we ship: nothing is spoken until the
    # segment closes, so every word waits for the end of the utterance.
    stabilised = result["stabilised_at"]
    if stabilised:
        _first_word, first_at = stabilised[0]
        print(f"  first word out at : {first_at:.1f}s "
              f"(segmented mode: {audio_s:.1f}s) -> "
              f"{audio_s - first_at:.1f}s earlier")
        # Words are committed in blocks; the useful figure is how far behind the
        # audio the commit point runs.
        lag = [at - (i + 1) / len(stabilised) * audio_s
               for i, (_, at) in enumerate(stabilised)]
        print(f"  mean commit lag   : {np.mean(lag):.1f}s behind the speech")
        print(f"  worst commit lag  : {max(lag):.1f}s")


def long_sample() -> np.ndarray | None:
    """Half a minute of varied English, synthesized.

    NOT the 8.5 s test clip repeated. Whisper handles duplicated audio badly --
    it falls into repetition loops that take many seconds to decode, which showed
    up as a 26 s pass and made the whole measurement meaningless. Distinct
    sentences are what a real utterance looks like anyway.
    """
    from voice2tts import voices
    from voice2tts.config import TtsConfig
    from voice2tts.tts import PiperEngine

    voice = next((k for k in voices.installed_keys()
                  if voices.voice_language(k) == "en"), None)
    if voice is None:
        from selftest import load_sample_16k
        return load_sample_16k()

    lines = [
        "The build finished about ten minutes ago, but two of the tests are "
        "still failing on Windows.",
        "It looks like the audio device enumeration is picking up something "
        "that is not really a microphone.",
        "I can reproduce it here if I unplug the interface and plug it back in "
        "while the app is running.",
        "Let me know whether you want me to fix that first, or finish the "
        "translation work and come back to it.",
        "Either way it should not hold up the release, because the installer "
        "does not touch that code path at all.",
    ]
    engine = PiperEngine(TtsConfig(voice=voice))
    import soxr

    clips = []
    for line in lines:
        clips.append(soxr.resample(engine.synth(line).astype(np.float32),
                                   engine.rate, RATE))
        clips.append(np.zeros(int(0.35 * RATE), dtype=np.float32))
    return np.concatenate(clips)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    ap.add_argument("--model", default="")
    args = ap.parse_args()


    from voice2tts.config import SttConfig
    from voice2tts.logging_setup import setup_logging
    from voice2tts.stt import WhisperEngine

    setup_logging("ERROR")
    audio = long_sample()
    if audio is None:
        print("no speech sample available")
        return 1

    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    results = []
    for device in devices:
        model = args.model or "base.en"
        try:
            engine = WhisperEngine(SttConfig(
                model=model, language="en", beam_size=1, device=device,
                compute_type="auto"))
        except Exception as exc:  # noqa: BLE001 - a missing GPU is not a failure
            print(f"\n=== {device}: unavailable ({exc}) ===")
            continue
        engine.warmup()
        results.append(run(engine, audio, args.interval,
                           f"{model} on {engine.device}/{engine.compute_type}"))
        report(results[-1], args.interval)

    if results:
        print("\n--- verdict ---")
        for result in results:
            realtime = result["decode_time"] / result["audio_s"]
            verdict = ("viable" if realtime < 0.5 and not result["over_budget"]
                       else "too expensive")
            print(f"  {result['label']:40s} {realtime:.2f}x realtime  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
