"""Entry point.

    python -m voice2tts                 tray app (default)
    python -m voice2tts --cli           headless, logs to console
    python -m voice2tts --devices       list audio devices and exit
    python -m voice2tts --say "hello"   synthesize one phrase and exit
    python -m voice2tts --check         report on models, CUDA and the virtual cable
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config
from .logging_setup import setup_logging
from .paths import config_path, list_voices


def _cmd_devices() -> int:
    from . import devices

    print(f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>6}  {'hostapi':<20} name")
    print("-" * 92)
    for d in sorted(devices._all(), key=lambda d: d.index):
        print(f"{d.index:>4}  {d.max_in:>3} {d.max_out:>3}  {d.rate:>6}  "
              f"{d.hostapi:<20} {d.name}")

    from . import cable as cable_mod

    din, dout = devices.default_input(), devices.default_output()
    print(f"\ndefault input : {din.name if din else '-'}")
    print(f"default output: {dout.name if dout else '-'}")

    found = cable_mod.list_devices()
    if not found:
        print("virtual cables: NOT INSTALLED")
        return 0
    print(f"\nvirtual cables: {len(found)} found (* = used by default)")
    for i, c in enumerate(found):
        mark = "*" if i == 0 else " "
        flag = "" if c.certain else "   [pairing inferred]"
        print(f" {mark} {c.label}")
        print(f"     play to  {c.output_name}")
        print(f"     Discord  {c.discord_input}{flag}")
    return 0


def _cmd_check() -> int:
    from . import cable as cable_mod
    from . import gpupack
    from .cuda import cuda_available
    from .paths import bundled_whisper, vad_model_path

    ok = True
    cfg = load_config()

    # Distinguish "no GPU" from "GPU present but the pack was never downloaded" --
    # a bare "not available" sends people looking for a driver problem they don't have.
    pack = gpupack.status()
    if cuda_available():
        print(f"GPU acceleration: available ({pack.size_mb:.0f} MB of libraries)"
              if pack.installed else "GPU acceleration: available")
    elif gpupack.gpu_present():
        print("GPU acceleration: NVIDIA GPU found, CUDA libraries not installed")
        print("                  add them from Settings -> Recognition, or the wizard")
    else:
        print("GPU acceleration: no NVIDIA GPU (CPU mode)")

    vad = vad_model_path()
    print(f"Silero VAD      : {'ok' if vad.exists() else f'MISSING at {vad}'}")
    ok &= vad.exists()

    found_voices = list_voices()
    print(f"Piper voices    : {len(found_voices)} found"
          + (f" ({', '.join(v.stem for v in found_voices[:4])})"
             if found_voices else " -- MISSING"))
    ok &= bool(found_voices)

    model = cfg.stt.model
    local = bundled_whisper(model)
    if local is not None:
        print(f"Speech model    : {model} (bundled)")
    else:
        print(f"Speech model    : {model} (downloaded on first use)")

    found_cable = cable_mod.detect()
    if found_cable is not None:
        print(f"Virtual mic     : {found_cable.product}")
        print(f"                  play to  {found_cable.output_name}")
        print(f"                  Discord  {found_cable.input_name}")
    else:
        print("Virtual mic     : NOT INSTALLED -- run the setup wizard")

    print(f"Config          : {config_path()}")
    if not ok:
        print("\nSomething is missing. Run: python scripts/fetch_models.py --bundle")
    return 0 if ok else 1


def _report_no_outputs(cfg) -> None:
    """Explain *why* there is nowhere to send audio, rather than just that there isn't."""
    print("\nerror: no usable output devices\n", file=sys.stderr)
    if not cfg.audio.outputs:
        print("  No outputs are configured at all.", file=sys.stderr)
    else:
        print("  Configured outputs:", file=sys.stderr)
        for t in cfg.audio.outputs:
            mark = "enabled " if t.enabled else "DISABLED"
            print(f"    [{mark}] {t.label}", file=sys.stderr)
    print(
        f"\n  Edit {config_path()}, or open Settings -> Audio in the tray app.\n"
        "  Deleting that file regenerates working defaults.",
        file=sys.stderr,
    )


def _cmd_say(text: str) -> int:
    from .output import OutputSink
    from .tts import PiperEngine

    cfg = load_config()
    tts = PiperEngine(cfg.tts)
    sink = OutputSink(cfg.audio)
    failures = sink.configure(cfg.audio.outputs, tts.rate)
    for label, reason in failures:
        print(f"warning: output {label} unavailable ({reason})", file=sys.stderr)
    if not sink.targets:
        _report_no_outputs(cfg)
        return 1
    try:
        sink.begin_utterance()
        try:
            for chunk in tts.stream(text):
                sink.write(chunk)
        finally:
            sink.end_utterance()
        sink.wait_drain()
    finally:
        sink.close()
    return 0


def _cmd_cli() -> int:
    from .pipeline import Pipeline

    cfg = load_config()

    def on_event(kind: str, message: str) -> None:
        print(f"[{kind}] {message}")

    pipeline = Pipeline(cfg, on_event=on_event)
    pipeline.start()
    print(f"\nRunning in {cfg.trigger.mode.value} mode. "
          f"Hotkey: {cfg.trigger.hotkey}")
    print("Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        pipeline.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice2tts", description=__doc__)
    parser.add_argument("--cli", action="store_true", help="run headless")
    parser.add_argument("--devices", action="store_true", help="list audio devices")
    parser.add_argument("--check", action="store_true", help="verify the installation")
    parser.add_argument("--say", metavar="TEXT", help="speak one phrase and exit")
    parser.add_argument("--no-autostart", action="store_true",
                        help="open the tray without starting the pipeline")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING...")
    args = parser.parse_args(argv)

    cfg_level = args.log_level or load_config().log_level
    setup_logging(cfg_level)

    if args.devices:
        return _cmd_devices()
    if args.check:
        return _cmd_check()
    if args.say:
        return _cmd_say(args.say)
    if args.cli:
        return _cmd_cli()

    from .platform_win import SingleInstance, enable_dpi_awareness

    # Two copies would both grab the global hotkey and the microphone, so one
    # keypress would start two captures. Bail out and surface the existing window.
    guard = SingleInstance()
    if not guard.acquire():
        print("Voice2TTS is already running.", file=sys.stderr)
        if guard.signal_existing():
            print("Brought the existing window to the front.", file=sys.stderr)
        return 0

    # Must happen before Tk creates its first window.
    mode = enable_dpi_awareness()
    logging.getLogger(__name__).info("DPI awareness: %s", mode)

    # Optional packs go on the import path before anything asks what this build
    # can do. Whether a Japanese voice is usable is answered by find_spec, so
    # asking before this ran would always answer no.
    from . import jppack

    if jppack.activate():
        logging.getLogger(__name__).info("Japanese phonemizer available")

    from .app import TrayApp

    try:
        TrayApp(autostart=not args.no_autostart, instance_guard=guard).run()
    finally:
        guard.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
