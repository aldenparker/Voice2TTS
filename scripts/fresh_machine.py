"""Run the interface tests as a machine with nothing installed sees them.

This developer machine has downloaded voices, translation models, the GPU pack
and the Japanese pack. A fresh install and a CI runner have none of them, and a
smaller screen. Every one of those differences has produced a test that passed
here and failed there:

    * "the window opens big enough for its content" read winfo_height() on a
      window that is withdrawn -- which reports the natural size here and the
      window manager's clamped size on a runner with a shorter screen.
    * "recognition points at the add-ons tab" assumed there was a GPU to point
      at.
    * Before those, tests that assumed a GPU, a microphone, and a populated
      device list.

`bare_machine.py` covers the no-audio case for both suites. This covers the
no-optional-anything case for the interface, which is where the differences
show up.

    python scripts/fresh_machine.py
"""

from __future__ import annotations

import sys
import tempfile
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Patched before anything builds a window. A runner's virtual display is small,
# and the settings window sizes itself to its content within the screen.
tk.Misc.winfo_screenwidth = lambda self: 1024
tk.Misc.winfo_screenheight = lambda self: 768

from voice2tts import devices, gpupack, jppack, paths, studiopack, translate  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="voice2tts-fresh-"))

# Only what the installer carries: the bundled voices live beside the code, the
# downloaded ones in the roaming profile.
_real_voice_dirs = paths.voices_dirs
paths.voices_dirs = lambda: [d for d in _real_voice_dirs()
                             if "Roaming" not in str(d)]

# Nothing optional installed.
translate.models_dir = lambda: _TMP / "translate"
jppack.japanese_dir = lambda: _TMP / "japanese"
paths.japanese_dir = lambda: _TMP / "japanese"
gpupack.cuda_dir = lambda: _TMP / "cuda"
studiopack.studio_dir = lambda: _TMP / "studio"

# No NVIDIA card, which changes what the Add-ons tab can offer.
gpupack.gpu_present = lambda: False
studiopack.probe = lambda force=False: studiopack.Hardware(
    gpu_name="", vram_gb=0.0, free_disk_gb=200.0)

# No audio hardware.
devices.list_inputs = lambda *a, **k: []
devices.list_outputs = lambda *a, **k: []
devices.default_input = lambda *a, **k: None
devices.default_output = lambda *a, **k: None
devices.refresh = lambda *a, **k: None


def main() -> int:
    print("Running the interface tests as a fresh machine sees them")
    print(f"  voices           : {[p.stem for p in paths.list_voices()]}")
    print(f"  translation models: {[p.code for p in translate.installed_pairs()]}")
    print(f"  japanese pack    : {jppack.status().installed}")
    print(f"  gpu pack         : {gpupack.status().usable}")
    print("  screen           : 1024x768")
    print("-" * 70, flush=True)

    import guitest

    code = guitest.main()
    print("\nThe interface tests pass on a fresh machine."
          if code == 0 else "\nFAILED on a fresh machine.")
    return code


if __name__ == "__main__":
    sys.exit(main())
