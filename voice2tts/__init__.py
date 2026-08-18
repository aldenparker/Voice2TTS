"""Voice2TTS -- speak, get recognized, get re-spoken through a virtual microphone."""

# Single source of truth for the version. build.ps1 reads this and passes it to
# PyInstaller and Inno Setup, and updater.py compares against it. Do not hardcode
# a version anywhere else.
__version__ = "0.7.0"

# Where this build looks for updates. Baked in so a normal install updates itself
# without the user having to find and type a repository name. Forks should change
# this; users can override it in Settings -> Updates, and clearing it disables
# update checking entirely.
DEFAULT_UPDATE_REPO = "aldenparker/Voice2TTS"
