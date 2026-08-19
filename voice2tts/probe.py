"""Proving that an optional pack actually works, rather than that it is there.

Five modules had grown the same shortcut. `jppack.status()` asked whether a
directory existed. `gpupack.usable` looked for two filenames. `translate` looked
for `model.bin`. `voices.download_voice` checked the `.onnx` arrived. None of
them loaded anything, so all five could say yes about something that would fail
at the moment it was used -- which for a phonemizer meant "Failed to process
utterance", once per sentence, with the real reason buried in a traceback and
the settings window still reporting the pack as installed.

The rule this module exists to enforce:

    prove it at the moment it is installed or enabled, cache the answer, and
    make the cheap check the one nobody is allowed to call "usable".

Every check here returns `None` for "this works" and a sentence for "it does
not, and here is why" -- a shape that is hard to accidentally get backwards, and
that carries the reason all the way out to the user instead of losing it.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# What a check said last time. Importing a package or loading a DLL costs real
# time, and these are asked on every settings refresh. Cleared by forget(),
# which every install and uninstall path calls.
_ANSWERS: dict[str, str | None] = {}

# Modules this module imported, so forget() can put them back. Proving a pack
# works means importing it, and an import is permanent: without this, a pack
# that had been uninstalled kept reporting itself usable until the next restart,
# because `import` found it still sitting in sys.modules.
_IMPORTED: set[str] = set()

# The reason string used when a thing is not there at all, as opposed to there
# and broken. Callers word those two cases very differently: one is "download
# it", the other is "remove it and download it again".
MISSING = "missing"


def forget() -> None:
    """Re-test everything, after a pack is installed or removed.

    Unloads what was imported to prove a pack worked, so the next check asks
    the filesystem again rather than the answer it cached in sys.modules. A
    compiled extension does not truly unload, but dropping the reference is
    enough for find_spec to report an uninstalled pack as gone.
    """
    _ANSWERS.clear()
    for module in sorted(_IMPORTED):
        for name in [n for n in sys.modules
                     if n == module or n.startswith(f"{module}.")]:
            del sys.modules[name]
    _IMPORTED.clear()


def import_problem(module: str) -> str | None:
    """None if the module imports, MISSING if absent, else why it failed.

    Actually imports it. `find_spec` only proves a module can be FOUND, and a
    phonemizer that is present but will not load reported as usable and then
    failed inside synthesis. Finding out here costs one import, once.
    """
    key = f"import:{module}"
    if key in _ANSWERS:
        return _ANSWERS[key]

    result: str | None = None
    if importlib.util.find_spec(module) is None:
        result = MISSING
    else:
        try:
            importlib.invalidate_caches()
            importlib.import_module(module)
            _IMPORTED.add(module)
        except Exception as exc:  # noqa: BLE001 - any failure is disqualifying
            log.warning("%s is present but will not import: %s", module, exc)
            result = f"{type(exc).__name__}: {exc}"
    _ANSWERS[key] = result
    return result


def library_problem(path: Path) -> str | None:
    """None if the DLL loads, MISSING if absent, else why it failed.

    A CUDA DLL of the right name built against the wrong runtime is exactly as
    unusable as no DLL at all, and only loading it can tell the difference. The
    handle is deliberately not freed: something is about to use it.
    """
    key = f"lib:{path}"
    if key in _ANSWERS:
        return _ANSWERS[key]

    result: str | None = None
    if not path.is_file():
        result = MISSING
    else:
        try:
            ctypes.WinDLL(str(path))
        except OSError as exc:
            log.warning("%s is present but will not load: %s", path.name, exc)
            result = f"{type(exc).__name__}: {exc}"
    _ANSWERS[key] = result
    return result


def files_problem(root: Path, required: dict[str, str]) -> str | None:
    """None if every required file is present and not empty, else what is wrong.

    `required` maps a relative path to what it is for, so the message names the
    consequence rather than the filename: a model without its tokenizer does not
    fail, it produces fluent gibberish, and "tokenizer.spm is missing" does not
    tell anyone that.

    Zero-length counts as absent. An interrupted download leaves the file there.
    """
    for relative, purpose in required.items():
        target = root / relative
        try:
            if not target.is_file():
                return f"{relative} is missing, so {purpose}"
            if target.stat().st_size == 0:
                return (f"{relative} is empty -- the download did not finish, "
                        f"so {purpose}")
        except OSError as exc:
            return f"{relative} could not be read ({exc}), so {purpose}"
    return None
