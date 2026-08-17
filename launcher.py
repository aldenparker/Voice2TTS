"""Frozen-build entry point.

PyInstaller cannot use voice2tts/__main__.py directly: freezing it makes that file
the top-level __main__ module, so its relative imports ("from .config import ...")
raise "attempted relative import with no known parent package". Importing the
package by name here keeps voice2tts a real package inside the bundle.

Running from source still works through `python -m voice2tts`.
"""

import multiprocessing
import sys

from voice2tts.__main__ import main

if __name__ == "__main__":
    # Without this, any child process in a frozen build re-runs the whole app.
    multiprocessing.freeze_support()
    sys.exit(main())
