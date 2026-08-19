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

It then does it again with the pack deliberately broken and the phonemizer
loaded, which is the state every retry in the field failed from: Windows will
not let anything overwrite a loaded .pyd, so an install that extracted in place
failed on its first package and never reached the missing one.

Costs two ~105 MB downloads and a couple of minutes. Worth running before
tagging any release that touches jppack.py, and whenever a dependency of the
phonemizer moves.
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


def imports_cleanly(pack: Path, staged: Path) -> bool:
    """Import the pack in a child that can see nothing but it and numpy."""
    environment = dict(os.environ,
                       PYTHONPATH=os.pathsep.join([str(pack), str(staged)]))
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         f"import pyopenjtalk; print(pyopenjtalk.g2p({SENTENCE!r}))"],
        capture_output=True, text=True, errors="replace", env=environment)
    if result.returncode == 0:
        print("   ", result.stdout.strip().splitlines()[-1])
        return True
    print(result.stderr.strip()[-700:])
    return False


def main() -> int:
    pack = Path(tempfile.mkdtemp(prefix="voice2tts-jp-isolated-")) / "japanese"
    jppack.japanese_dir = lambda: pack

    print("Staging what the application ships...")
    staged = stage_the_application()
    print(f"   {staged}: {', '.join(sorted(p.name for p in staged.iterdir()))}")

    print(f"\n== 1. a clean install into {pack}")
    try:
        status = jppack.install(progress=lambda message: print("   ", message))
    except Exception as exc:  # noqa: BLE001 - this is the report
        print(f"\nThe install itself failed: {exc}")
        return 1
    print(f"   installed {status.size_mb:.0f} MB:",
          ", ".join(sorted(p.name for p in pack.iterdir()
                           if not p.name.endswith("info"))))

    print("\n   importing with only the pack and the application's own numpy:")
    if not imports_cleanly(pack, staged):
        print("\nThe pack needs something it does not carry. That is the bug "
              "-- a development venv hides it.")
        return 1
    print("   the pack is self-contained.")

    # -- 2. the reported situation, exactly ---------------------------------
    # A pack missing part of itself, in an app that has already loaded the
    # phonemizer. Windows will not let anything overwrite a loaded .pyd, so an
    # install that extracted in place failed on its FIRST package -- and the
    # missing one, several packages later, was never fetched. Every retry did
    # the same thing, which is what the bug report looked like.
    print("\n== 2. break it the way it was reported broken, then repair it")
    removed = [item for item in pack.iterdir()
               if item.name.startswith(("pydantic", "annotated", "typing_"))]
    for item in removed:
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    print("   removed:", ", ".join(sorted(item.name for item in removed)))

    jppack.activate()
    import pyopenjtalk  # noqa: F401 - loaded on purpose, to lock its extensions

    print("   phonemizer loaded, so its compiled extensions are now locked")

    try:
        jppack.install(progress=lambda message: print("   ", message))
    except Exception as exc:  # noqa: BLE001 - this is the report
        print(f"\nThe repair failed: {type(exc).__name__}: {exc}")
        print("An install must not need the app closed to replace a pack.")
        return 1

    back = sorted(p.name for p in pack.iterdir() if not p.name.endswith("info"))
    print("   contains:", ", ".join(back))
    if not any(name.startswith("pydantic") for name in back):
        print("\nThe repair did not restore what was missing.")
        return 1

    print("\n   importing the repaired pack in a fresh process:")
    if not imports_cleanly(pack, staged):
        print("\nThe repaired pack still will not import.")
        return 1

    print("\nA broken pack can be repaired while it is in use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
