"""Voice2TTS -- speak, get recognized, get re-spoken through a virtual microphone."""

# Single source of truth for the version. build.ps1 reads this and passes it to
# PyInstaller and Inno Setup, and updater.py compares against it. Do not hardcode
# a version anywhere else.
__version__ = "0.4.0"
