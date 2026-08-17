"""Run the test suites as a CI runner sees them: no GPU, no microphone.

Three separate CI failures have come from tests that quietly assert the
*development machine's* hardware rather than the behaviour of the code:

  * "input combo populated"   -- needs a microphone
  * "probe finds this GPU"    -- needs an NVIDIA card
  * "device list is trimmed"  -- needs more devices than WASAPI exposes

Each one passes locally and fails on every runner, and none of them is visible
until CI runs. This makes that environment reproducible in about a minute:

    python scripts/bare_machine.py                  # both suites, bare machine
    python scripts/bare_machine.py guitest          # just one
    python scripts/bare_machine.py --keep-outputs   # speakers but no microphone

It stubs the hardware probes rather than the test assertions, so what runs is
the real suite against an empty machine.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SUITES = {
    "selftest": (ROOT / "scripts" / "selftest.py", ["--no-audio"]),
    "guitest": (ROOT / "scripts" / "guitest.py", []),
}


def strip_hardware(keep_outputs: bool = False) -> dict[str, int]:
    """Make this machine look like a fresh runner. Returns what was hidden.

    `keep_outputs` leaves the speakers in place, which is not a runner but IS a
    very common real machine: a desktop with speakers and no microphone. That
    asymmetric case has its own bugs -- it is how "the trimmed list is smaller"
    was found to fail whenever one kind of device was absent.
    """
    from voice2tts import devices, studiopack

    hidden = {"inputs": len(devices.list_inputs(False)),
              "outputs": len(devices.list_outputs(False))}

    studiopack._nvidia_smi = lambda _query: ""
    studiopack._GPU_CACHE = None

    # Patch the enumeration, not the list_* helpers. default_input() and
    # resolve_input() do not go through those, so stubbing them produces a
    # machine that reports no devices from one function and real ones from
    # another -- a state no real computer is ever in, and one that manufactures
    # failures rather than finding them.
    real_all = devices._all
    if keep_outputs:
        devices._all = lambda: [d for d in real_all() if d.max_in == 0]
        hidden["outputs"] = 0
    else:
        devices._all = lambda: []
    devices.refresh()
    return hidden


def main() -> int:
    args = sys.argv[1:]
    keep_outputs = "--keep-outputs" in args
    wanted = [a for a in args if not a.startswith("-")] or list(SUITES)
    unknown = [name for name in wanted if name not in SUITES]
    if unknown:
        sys.exit(f"unknown suite(s): {', '.join(unknown)}. "
                 f"Choose from: {', '.join(SUITES)}")

    hidden = strip_hardware(keep_outputs)
    gone = "no microphone" if keep_outputs else "no microphone and no speakers"
    print(f"Pretending this machine has no NVIDIA GPU and {gone} "
          f"(really has {hidden['inputs']} inputs, "
          f"{hidden['outputs'] or 'kept'} outputs).\n")

    failures = []
    for name in wanted:
        path, args = SUITES[name]
        print(f"{'=' * 60}\n{name}\n{'=' * 60}")
        sys.argv = [str(path), *args]
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            if exc.code:
                failures.append(name)

    if failures:
        print(f"\nFAILED on a bare machine: {', '.join(failures)}")
        return 1
    print("\nBoth suites pass on a bare machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
