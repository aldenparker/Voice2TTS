"""Run the self-test as if nothing optional had ever been downloaded.

This is the machine CI actually runs on, and it is not the machine anyone
develops on. Four checks reached release verification green here and red there,
because they were quietly asking what the author happened to have downloaded
rather than what the rule under test says:

    the reported Japanese fault SKIPPED itself with no en->ja model, so the one
    case the whole plan module exists for was tested on one laptop

    three voice/language checks needed a model for the pair to exist before the
    text could reach the voice in the target language at all

`bare_machine.py` covers "no audio hardware" and `fresh_machine.py` covers "a
small screen and no optional packs in the interface". This covers "nothing has
been downloaded", which is the other half of a clean clone.

Run it before tagging. It takes about the same time as the plain self-test.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from voice2tts import gpupack, jppack, translate  # noqa: E402

# Empty rather than absent: every one of these creates its directory on demand,
# so pointing at somewhere that does not exist would prove nothing.
EMPTY = Path(tempfile.mkdtemp(prefix="voice2tts-nothing-"))
for name in ("translate", "japanese", "cuda"):
    (EMPTY / name).mkdir()

translate.models_dir = lambda: EMPTY / "translate"
jppack.japanese_dir = lambda: EMPTY / "japanese"
gpupack.cuda_dir = lambda: EMPTY / "cuda"

import selftest  # noqa: E402

if __name__ == "__main__":
    print("Running with no translation models, no Japanese pack and no GPU pack.")
    # The end-to-end tests need real audio hardware and real models; this is
    # about the checks that should hold without either.
    sys.argv = ["selftest", "--no-audio", "--no-network", "--no-e2e"]
    code = selftest.main()
    print("\nThe self-test passes with nothing installed."
          if code == 0 else
          "\nSomething here depends on a download. That is the bug.")
    sys.exit(code)
