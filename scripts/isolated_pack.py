"""Install the Japanese pack somewhere fresh and import it with no venv.

The frozen build carries numpy and not much else. A development venv carries
whatever anything has ever needed -- which for this project includes pydantic,
via an unrelated dependency. So the pack shipped missing pydantic outright: it
resolved, downloaded, unpacked and imported perfectly here, and failed on the
first machine that was not this one, as

    pyopenjtalk is present but will not import: No module named 'pydantic'

Nothing short of importing it without site-packages catches that.

Two things have to be right or this proves nothing, and the first version of
this script got the second one wrong:

    `python -S` leaves the standard library alone and does not add
    site-packages. Rewriting sys.path by hand instead loses `_socket`.

    what stands in for "the application" must be a directory containing ONLY
    what the application ships. Pointing at numpy's own parent is pointing at
    site-packages, which puts the entire venv back and quietly makes the whole
    exercise a no-op -- it reported a pack as self-contained while a deliberately
    broken one sailed through.

It earned its keep twice on the way in: it found that pydantic pins pydantic-core
to one exact release and raises SystemError on any other, so resolving names and
taking the newest of each produced a pack that unpacked cleanly and still would
not import.

Costs a ~105 MB download and about a minute. Worth running before tagging any
release that touches jppack.py, and whenever a dependency of the phonemizer moves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice2tts import jppack  # noqa: E402

SENTENCE = "こんにちは"


def stage_the_application() -> Path:
    """A directory holding exactly what the installed build provides.

    Copied, not pointed at: numpy's parent directory is site-packages, and
    putting that on the path hands the child everything the venv has ever
    installed -- including the pydantic whose absence this exists to catch.
    """
    import numpy

    staged = Path(tempfile.mkdtemp(prefix="voice2tts-app-"))
    source = Path(numpy.__file__).parent
    shutil.copytree(source, staged / "numpy")
    # numpy's compiled libraries sit beside it, not inside it.
    libs = source.parent / "numpy.libs"
    if libs.is_dir():
        shutil.copytree(libs, staged / "numpy.libs")
    return staged


def main() -> int:
    pack = Path(tempfile.mkdtemp(prefix="voice2tts-jp-isolated-"))
    jppack.japanese_dir = lambda: pack

    print("Staging what the application ships...")
    staged = stage_the_application()
    print(f"   {staged}: {', '.join(sorted(p.name for p in staged.iterdir()))}")

    print(f"\nInstalling the Japanese pack into {pack}")
    try:
        status = jppack.install(progress=lambda message: print("   ", message))
    except Exception as exc:  # noqa: BLE001 - this is the report
        print(f"\nThe install itself failed: {exc}")
        return 1
    print(f"\nInstalled {status.size_mb:.0f} MB:")
    print("   ", ", ".join(sorted(p.name for p in pack.iterdir()
                                  if not p.name.endswith("info"))))

    environment = dict(os.environ,
                       PYTHONPATH=os.pathsep.join([str(pack), str(staged)]))
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         f"import pyopenjtalk; print(pyopenjtalk.g2p({SENTENCE!r}))"],
        capture_output=True, text=True, errors="replace", env=environment)

    print("\nImporting with only the pack and the application's own numpy:")
    if result.returncode == 0:
        print("   ", result.stdout.strip())
        print("\nThe pack is self-contained.")
        return 0

    print(result.stderr.strip()[-900:])
    print("\nThe pack needs something it does not carry. That is the bug -- "
          "a development venv hides it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
