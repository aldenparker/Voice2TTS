"""Offline self-test: exercises config, VAD, STT and the output sink.

Opens real output streams but writes silence, so it is safe to run with headphones
on. Does not touch the microphone.

    python scripts/selftest.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice2tts.config import Config, OutputTarget, load_config  # noqa: E402
from voice2tts.logging_setup import setup_logging  # noqa: E402

SAMPLE = ROOT / "spike" / "out" / "tts_sample.wav"

# Kept in sync with what test_stt asserts on; "pipeline" must survive recognition.
SAMPLE_TEXT = (
    "This is a test of the voice to text to speech pipeline. "
    "It should split into a few sentences. "
    "Each one gets synthesized separately so playback can start early."
)

passed = failed = 0


def ensure_sample() -> Path | None:
    """Return a speech sample, synthesizing one if it is not already there.

    The sample used to be a leftover from spike/02_tts.py, which meant CI failed on
    a clean checkout -- spike/out is gitignored. Generating it with Piper keeps the
    suite self-contained and costs about a second.
    """
    if SAMPLE.exists():
        return SAMPLE
    try:
        from voice2tts.config import TtsConfig
        from voice2tts.tts import PiperEngine

        engine = PiperEngine(TtsConfig())
        audio = engine.synth(SAMPLE_TEXT)
        if not len(audio):
            return None
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(SAMPLE), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(engine.rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print(f"  (generated {SAMPLE.name}: {len(audio) / engine.rate:.2f}s)")
        return SAMPLE
    except Exception as exc:  # noqa: BLE001 - reported by the caller's check()
        print(f"  (could not generate sample: {exc})")
        return None


def load_sample_16k() -> np.ndarray | None:
    """The speech sample as 16 kHz mono float32, or None if unavailable."""
    path = ensure_sample()
    if path is None:
        return None
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if rate != 16000:
        import soxr

        audio = soxr.resample(audio, rate, 16000)
    return audio


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def test_config() -> None:
    print("\n[config]")
    cfg = Config()
    cfg.audio.outputs = [
        OutputTarget(match="CABLE Input", gain=0.8, enabled=True),
        OutputTarget(match="", gain=0.5, enabled=False),
    ]
    cfg.trigger.hotkey = "ctrl+alt+v"
    tmp = ROOT / "spike" / "out" / "roundtrip.toml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(tmp)

    import tomllib

    with tmp.open("rb") as fh:
        back = Config.from_dict(tomllib.load(fh))
    check("toml round-trip preserves outputs", len(back.audio.outputs) == 2)
    check("output fields survive",
          back.audio.outputs[0].match == "CABLE Input"
          and abs(back.audio.outputs[0].gain - 0.8) < 1e-6
          and back.audio.outputs[0].enabled is True)
    check("second output disabled", back.audio.outputs[1].enabled is False)

    partial = Config.from_dict({"trigger": {"mode": "nonsense"}, "junk": 1})
    check("unknown mode falls back to ptt", partial.trigger.mode == "ptt")
    check("unknown keys ignored", isinstance(partial, Config))

    # An all-disabled config makes the app silent with a confusing error; loading
    # one must repair it rather than leave the user with nowhere for audio to go.
    dead = Config()
    dead.audio.outputs = [
        OutputTarget(match="CABLE Input", gain=1.0, enabled=False),
        OutputTarget(match="", gain=0.7, enabled=False),
    ]
    check("all-disabled outputs detected", dead.ensure_usable_output())
    check("system default enabled by repair",
          any(t.enabled and not t.match for t in dead.audio.outputs))
    check("repair is idempotent", not dead.ensure_usable_output())

    healthy = Config()
    healthy.audio.outputs = [OutputTarget(match="CABLE Input", enabled=True)]
    check("healthy config untouched", not healthy.ensure_usable_output())

    empty = Config()
    empty.audio.outputs = []
    check("empty output list repaired", empty.ensure_usable_output()
          and len(empty.audio.outputs) == 1)
    tmp.unlink(missing_ok=True)


def test_hotkey() -> None:
    print("\n[hotkey]")
    from voice2tts.hotkey import HotkeySpec, describe

    check("parses ctrl+alt+v", describe("ctrl+alt+v") == "")
    check("parses f8", describe("f8") == "")
    check("rejects gibberish", describe("ctrl+nope") != "")
    spec = HotkeySpec("ctrl+shift+k")
    check("modifiers extracted", spec.modifiers == {"ctrl", "shift"})


def test_vad() -> None:
    print("\n[vad]")
    from voice2tts.config import VadConfig
    from voice2tts.vad import WINDOW, SileroVad, VadSegmenter

    vad = SileroVad()
    silence = np.zeros(WINDOW, dtype=np.float32)
    check("silence scores low", vad(silence) < 0.3, f"p={vad(silence):.3f}")

    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return

    seg = VadSegmenter(VadConfig(), preroll_ms=300, max_utterance_s=30)
    utterances = []
    # Pad with silence so the final utterance gets its end-of-speech trigger.
    padded = np.concatenate([np.zeros(16000, np.float32), audio, np.zeros(24000, np.float32)])
    for i in range(0, len(padded) - WINDOW, WINDOW):
        got = seg.process(padded[i:i + WINDOW])
        if got is not None:
            utterances.append(got)
    tail = seg.flush()
    if tail is not None:
        utterances.append(tail)

    total = sum(len(u) for u in utterances) / 16000
    check("segmented speech from silence", len(utterances) >= 1,
          f"{len(utterances)} utterance(s), {total:.2f}s of {len(audio)/16000:.2f}s")


def test_stt() -> None:
    print("\n[stt]")
    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return
    from voice2tts.config import SttConfig
    from voice2tts.stt import WhisperEngine

    engine = WhisperEngine(SttConfig())
    warm = engine.warmup()
    check("engine loaded", True, f"{engine.device}/{engine.compute_type}")
    check("warmup completed", warm > 0, f"{warm:.2f}s")

    t0 = time.perf_counter()
    text = engine.transcribe(audio)
    elapsed = time.perf_counter() - t0
    check("transcribes sample", "pipeline" in text.lower(), f"{elapsed*1000:.0f} ms")
    check("hallucination filter drops 'Thank you.'", engine._is_noise("Thank you."))
    check("real text survives filter", not engine._is_noise("hello there friend"))


def test_tts_and_sink() -> None:
    print("\n[tts + output]")
    from voice2tts.config import TtsConfig
    from voice2tts.output import OutputSink
    from voice2tts.tts import PiperEngine

    cfg = load_config()
    tts = PiperEngine(TtsConfig())
    check("piper loaded", tts.rate > 0, f"{tts.rate} Hz")

    audio = tts.synth("Self test.")
    check("synthesized audio", len(audio) > 1000, f"{len(audio)/tts.rate:.2f}s")

    # Open the real default device but push silence: verifies stream setup and the
    # resampler path without making noise.
    sink = OutputSink(cfg.audio)
    failures = sink.configure([OutputTarget(match="", gain=1.0, enabled=True)], tts.rate)
    check("default output opened", len(sink.targets) == 1,
          sink.targets[0].name if sink.targets else str(failures))
    if sink.targets:
        t = sink.targets[0]
        check("resampler engaged where needed",
              (t._resampler is not None) == (t.rate != tts.rate),
              f"{tts.rate} -> {t.rate}")
        sink.begin_utterance()
        sink.write(np.zeros(tts.rate, dtype=np.float32))
        check("audio queued", sink.active)
        sink.end_utterance()
        drained = sink.wait_drain(timeout=5)
        check("drains cleanly", drained)
        check("no underruns", t.underruns == 0, f"{t.underruns}")
    sink.close()


def test_packaging_bits() -> None:
    """Offline checks for the cable, voices and GPU-pack modules."""
    print("\n[packaging]")
    import zipfile

    from voice2tts import cable, gpupack, voices
    from voice2tts.paths import bundled_whisper, cache_dir, cuda_dir

    # --- paths
    check("cache dir is LOCALAPPDATA", "Local" in str(cache_dir()), str(cache_dir()))
    check("cuda dir under cache", cuda_dir().parent == cache_dir())
    check("base.en bundled for offline install", bundled_whisper("base.en") is not None,
          str(bundled_whisper("base.en")))
    check("absent model reports None", bundled_whisper("nonexistent-model") is None)

    # --- voices
    installed = voices.installed_keys()
    missing = [v for v in voices.BUNDLED if v not in installed]
    check("all three voices bundled", not missing, f"missing={missing}" if missing
          else ", ".join(voices.BUNDLED))
    check("bundled voices are not removable",
          not any(voices.is_removable(v) for v in voices.BUNDLED))
    check("unknown voice resolves to None", voices.installed_path("no-such-voice") is None)

    # --- cable detection
    found = cable.detect()
    check("cable detection runs", True, found.product if found else "none installed")
    check("installed() agrees with detect()", cable.installed() == (found is not None))
    check("known cable list is non-trivial", len(cable.KNOWN_CABLES) >= 8,
          f"{len(cable.KNOWN_CABLES)} products")

    # The scraper regex is the fragile part; pin its behaviour on sample markup.
    sample = ('<a href="https://download.vb-audio.com/Download_CABLE/'
              'VBCABLE_Driver_Pack45.zip">Download</a>')
    m = cable._ZIP_LINK.search(sample)
    check("scraper finds a driver pack link", m is not None and m.group(0).endswith(".zip"),
          m.group(0) if m else "no match")
    check("scraper ignores unrelated zips",
          cable._ZIP_LINK.search('<a href="https://x.test/other.zip">') is None)

    # --- archive safety
    bad = ROOT / "spike" / "out" / "evil.zip"
    bad.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("VBCABLE_Setup_x64.exe", b"stub")
        zf.writestr("../../escaped.txt", b"nope")
    try:
        cable.extract(bad, ROOT / "spike" / "out" / "unpack")
        check("traversal path rejected", False, "extract() accepted ../ member")
    except RuntimeError as exc:
        check("traversal path rejected", "unsafe path" in str(exc), str(exc))
    finally:
        bad.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "spike" / "out" / "unpack", ignore_errors=True)

    # --- gpu pack
    pack = gpupack.status()
    check("gpu pack status readable", isinstance(pack.installed, bool),
          f"installed={pack.installed} usable={pack.usable} {pack.size_mb:.0f} MB")
    check("nvidia gpu probe runs", isinstance(gpupack.gpu_present(), bool),
          f"gpu_present={gpupack.gpu_present()}")

    # gpupack writes nvidia/<pkg>/bin INSIDE its own directory, one level deeper
    # than the pip layout. A single-level glob in cuda.py missed it entirely, and
    # the bug was invisible in a venv because the pip-installed wheels sit at the
    # depth the glob expected. Build the layout in a temp dir so nothing else can
    # mask a regression.
    import tempfile
    from unittest import mock

    from voice2tts import cuda as cuda_mod

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "cuda"
        for pkg, dll in (("cublas", "cublas64_12.dll"), ("cudnn", "cudnn64_9.dll")):
            d = fake / "nvidia" / pkg / "bin"
            d.mkdir(parents=True)
            (d / dll).write_bytes(b"MZ stub")
        with mock.patch("voice2tts.paths.cuda_dir", return_value=fake):
            found = cuda_mod._search_dirs()
        names = {p.name for d in found for p in d.glob("*.dll")}
        check("cuda search finds the gpupack layout",
              "cublas64_12.dll" in names and "cudnn64_9.dll" in names,
              f"{len(found)} dirs")

    if pack.usable:
        # If the pack is installed for real, the loader must actually see it.
        dirs = cuda_mod._search_dirs()
        has_cublas = any((d / "cublas64_12.dll").exists() for d in dirs)
        check("installed gpu pack is on the search path", has_cublas,
              f"{len(dirs)} dirs searched")


def test_updates() -> None:
    """Version comparison, throttling, and the refusal to self-install from source."""
    print("\n[updates]")
    import voice2tts
    from voice2tts import updater
    from voice2tts.config import Config

    check("version is a valid triple",
          len(updater.parse_version(voice2tts.__version__)) >= 2,
          voice2tts.__version__)

    cases = [
        ("0.3.0", "0.2.0", True),
        ("0.2.1", "0.2.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.2.0", "0.2.0", False),
        ("0.1.0", "0.2.0", False),
        ("0.10.0", "0.9.0", True),      # numeric, not lexicographic
        ("v0.3.0", "0.2.0", True),      # tolerates a v prefix
        ("0.3.0-beta", "0.2.0", True),  # tolerates a pre-release suffix
    ]
    wrong = [(a, b, e) for a, b, e in cases if updater.is_newer(a, b) != e]
    check("version comparison", not wrong, f"failed: {wrong}" if wrong else f"{len(cases)} cases")

    check("throttle blocks a recent check",
          not updater.should_check(time.time(), 24))
    check("throttle allows an old check",
          updater.should_check(time.time() - 25 * 3600, 24))
    check("interval 0 disables checking",
          not updater.should_check(0, 0))

    # Bad repo strings must fail loudly rather than build a nonsense URL.
    for bad in ("", "not-a-repo", "https://example.test"):
        try:
            updater.check(bad)
            check(f"rejects repo {bad!r}", False, "no error raised")
        except ValueError:
            check(f"rejects repo {bad!r}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"rejects repo {bad!r}", False, f"wrong error: {type(exc).__name__}")

    # A source checkout must never run an installer over itself.
    try:
        updater.apply(ROOT / "does-not-exist.exe")
        check("refuses to self-install from source", False, "no error raised")
    except RuntimeError as exc:
        check("refuses to self-install from source", "source checkout" in str(exc))
    except Exception as exc:  # noqa: BLE001
        check("refuses to self-install from source", False, str(exc))

    # Config normalisation: a pasted URL should become owner/name.
    cfg = Config()
    cfg.updates.repo = "https://github.com/someone/Voice2TTS/"
    cfg.validate()
    check("normalises a pasted repo URL", cfg.updates.repo == "someone/Voice2TTS",
          cfg.updates.repo)
    cfg.updates.interval_hours = -5
    cfg.validate()
    check("clamps negative interval", cfg.updates.interval_hours == 0)

    # Settings must survive an upgrade: everything user-owned lives outside {app}.
    from voice2tts.paths import cache_dir, config_path, user_data_dir

    app_dir = Path(sys.prefix)
    for name, path in (("config", config_path()), ("voices", user_data_dir()),
                       ("cache", cache_dir())):
        check(f"{name} stored outside the install dir",
              app_dir not in path.parents and path != app_dir, str(path))


def test_network() -> None:
    """Live checks against VB-Audio and HuggingFace. Needs internet."""
    print("\n[network]")
    from voice2tts import cable, voices

    try:
        url, source = cable.resolve_download_url()
        check("cable download URL resolves",
              url.lower().endswith(".zip"), f"{source}: {url.rsplit('/', 1)[-1]}")
    except Exception as exc:  # noqa: BLE001
        check("cable download URL resolves", False, str(exc))

    try:
        catalogue = voices.fetch_catalogue()
        check("voice catalogue fetched", len(catalogue) > 50, f"{len(catalogue)} voices")
        english = voices.filter_catalogue(catalogue, language_prefix="en_US")
        check("catalogue filters by language", len(english) > 5, f"{len(english)} en_US")
        keys = {e.key for e in catalogue}
        missing = [v for v in voices.BUNDLED if v not in keys]
        check("bundled voices exist in catalogue", not missing, str(missing))
        sized = [e for e in catalogue if e.size_mb > 0]
        check("catalogue reports sizes", len(sized) > len(catalogue) // 2,
              f"{len(sized)}/{len(catalogue)}")
    except Exception as exc:  # noqa: BLE001
        check("voice catalogue fetched", False, str(exc))


def test_pipeline_end_to_end() -> None:
    """Drive the real Pipeline threads with a recorded utterance.

    Outputs are all disabled so this makes no sound; the microphone is opened (the
    pipeline needs an input device) but nothing captured from it is used, because
    push-to-talk mode leaves VAD idle and we inject the utterance directly.
    """
    print("\n[pipeline end-to-end]")
    audio = load_sample_16k()
    if audio is None:
        check("speech sample available", False, "could not load or generate")
        return

    import threading

    from voice2tts.pipeline import Pipeline, State

    cfg = load_config()
    cfg.trigger.mode = "ptt"
    for target in cfg.audio.outputs:
        target.enabled = False  # silent run

    seen: list[tuple[str, str]] = []
    spoke = threading.Event()

    def on_state(state: State) -> None:
        if state is State.SPEAKING:
            spoke.set()

    pipeline = Pipeline(cfg, on_state=on_state, on_event=lambda k, m: seen.append((k, m)))
    try:
        pipeline.start()
        check("pipeline started", pipeline.running)
        check("state is idle", pipeline.state is State.IDLE, pipeline.state.value)

        pipeline._submit(audio)

        deadline = time.time() + 30
        while time.time() < deadline and not pipeline.last_transcript:
            time.sleep(0.05)

        check("produced a transcript", bool(pipeline.last_transcript),
              pipeline.last_transcript[:60])
        check("reached SPEAKING state", spoke.wait(timeout=10))

        deadline = time.time() + 15
        while time.time() < deadline and pipeline.state is not State.IDLE:
            time.sleep(0.05)
        check("returned to idle", pipeline.state is State.IDLE, pipeline.state.value)

        kinds = {k for k, _ in seen}
        check("emitted transcript event", "transcript" in kinds, str(sorted(kinds)))
        check("emitted latency event", "latency" in kinds,
              next((m for k, m in seen if k == "latency"), "-"))

        status = pipeline.status()
        check("no dropped capture windows", status["dropped"] == 0, str(status["dropped"]))
    finally:
        pipeline.shutdown()
    check("shut down cleanly", not pipeline.running)


def test_platform() -> None:
    """Windows integration bits that do not need audio hardware."""
    print("\n[platform]")
    from voice2tts import platform_win
    from voice2tts.diagnostics import diagnostics

    guard = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("first instance acquires the mutex", guard.acquire())
    second = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("second instance is refused", not second.acquire())
    check("second instance reports why", second.already_running)
    second.release()
    guard.release()

    after = platform_win.SingleInstance(name=r"Local\Voice2TTS-selftest-probe")
    check("mutex is reusable once released", after.acquire())
    after.release()

    check("run_at_login readable", isinstance(platform_win.run_at_login(), bool))
    # Source checkouts have no exe to register, so this must decline rather than
    # write a broken registry entry pointing at python.exe.
    check("run_at_login refuses from source", not platform_win.set_run_at_login(True))

    report = diagnostics(load_config(), None)
    check("diagnostics report generated", len(report.splitlines()) > 20,
          f"{len(report.splitlines())} lines")
    user = os.environ.get("USERNAME") or ""
    if len(user) > 2:
        check("diagnostics redacts the user name", user not in report)
    check("diagnostics has no transcript text", "transcript" not in report.lower()
          or "last transcript" not in report.lower())


def test_language_guard() -> None:
    print("\n[language guard]")
    from voice2tts import voices

    check("english voice + english model is fine",
          not voices.language_mismatch("en_US-amy-medium", "base.en"))
    warning = voices.language_mismatch("de_DE-thorsten-medium", "small.en")
    check("german voice + english model warns", bool(warning))
    check("warning names both sides",
          "de_DE-thorsten-medium" in warning and "small.en" in warning)
    check("multilingual model accepts any voice",
          not voices.language_mismatch("de_DE-thorsten-medium", "large-v3"))
    check("blank inputs do not warn", not voices.language_mismatch("", "base.en"))


def main() -> int:
    setup_logging("WARNING")
    no_audio = "--no-audio" in sys.argv
    print("Voice2TTS self-test" + (" (no audio hardware)" if no_audio else ""))
    test_config()
    test_hotkey()
    test_vad()
    test_stt()
    test_platform()
    test_language_guard()
    if not no_audio:
        test_tts_and_sink()
    test_packaging_bits()
    test_updates()
    if "--no-network" not in sys.argv:
        test_network()
    if not no_audio and "--no-e2e" not in sys.argv:
        test_pipeline_end_to_end()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
