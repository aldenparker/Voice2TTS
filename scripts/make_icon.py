"""Generate installer/voice2tts.ico from the runtime tray artwork.

Keeps the .exe icon, installer icon and tray icon visually identical without
checking a binary asset into the tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from voice2tts.icon import make_icon  # noqa: E402
from voice2tts.pipeline import State  # noqa: E402

SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> int:
    dest = ROOT / "installer" / "voice2tts.ico"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # IDLE (blue) reads as the app's resting identity; STOPPED wears a slash.
    base = make_icon(State.IDLE).resize((256, 256), Image.LANCZOS)
    base.save(dest, format="ICO", sizes=SIZES)
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
