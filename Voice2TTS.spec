# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Voice2TTS.

Built via build.ps1, which fetches the bundle assets first. To run directly:

    & "$env:USERPROFILE\.venvs\voice2tts\Scripts\pyinstaller.exe" Voice2TTS.spec

Produces dist/Voice2TTS/ (onedir). Onefile is deliberately avoided: it unpacks the
whole payload to a temp directory on every launch, which for ~400 MB of models means
a multi-second startup and duplicated disk use.

CUDA is NOT bundled. The nvidia-*-cu12 wheels are ~1.3 GB and useless on a machine
without an NVIDIA GPU, so voice2tts/gpupack.py downloads them on demand into
%LOCALAPPDATA%\\Voice2TTS\\cuda, which voice2tts/cuda.py searches first. Set
VOICE2TTS_BUNDLE_CUDA=1 to produce a fat offline build instead.

Things that break frozen builds if forgotten, all handled below:
  * piper's espeak-ng-data -- synthesis fails with a phonemizer error without it
  * onnxruntime / ctranslate2 native DLLs -- not discoverable by import analysis
  * pystray and pynput backends -- selected by runtime import, so invisible to Analysis
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

PROJECT = Path(SPECPATH)
BUNDLE_CUDA = os.environ.get("VOICE2TTS_BUNDLE_CUDA") == "1"

binaries = []
datas = []

# --- native dependencies ---------------------------------------------------
# Piper: espeak-ng phonemizer data plus its bundled native libraries.
datas += collect_data_files("piper")
binaries += collect_dynamic_libs("piper")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("ctranslate2")

# --- bundled models --------------------------------------------------------
vad_model = PROJECT / "models" / "vad" / "silero_vad.onnx"
if not vad_model.exists():
    raise SystemExit("Missing Silero VAD. Run: python scripts/fetch_models.py --bundle")
datas.append((str(vad_model), "models/vad"))

# The Voice Studio's recording script. ~75 KB of text; absence is survivable
# (prompts.py falls back to a built-in set) so this does not abort the build.
prompts = PROJECT / "models" / "prompts" / "arctic.txt"
if prompts.exists():
    datas.append((str(prompts), "models/prompts"))

voice_files = sorted((PROJECT / "models" / "voices").glob("*.onnx*"))
if not voice_files:
    raise SystemExit("No Piper voices. Run: python scripts/fetch_models.py --bundle")
for voice in voice_files:
    datas.append((str(voice), "models/voices"))

whisper_root = PROJECT / "models" / "whisper"
whisper_models = [d for d in whisper_root.glob("*") if (d / "model.bin").exists()]
if not whisper_models:
    raise SystemExit(
        "No bundled Whisper model. Run: python scripts/fetch_models.py --bundle"
    )
for model_dir in whisper_models:
    for item in model_dir.rglob("*"):
        if item.is_file():
            rel = item.parent.relative_to(PROJECT)
            datas.append((str(item), str(rel)))

# --- optional fat build ----------------------------------------------------
if BUNDLE_CUDA:
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in sorted(nvidia_root.glob("*/bin")):
        for dll in bin_dir.glob("*.dll"):
            # Preserve nvidia/<pkg>/bin so cuda.py's search still matches.
            binaries.append((str(dll), str(dll.parent.relative_to(nvidia_root.parent))))

a = Analysis(
    # launcher.py, NOT voice2tts/__main__.py: freezing the latter makes it the
    # top-level __main__, and its relative imports then fail with "attempted
    # relative import with no known parent package".
    ["launcher.py"],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "voice2tts.app",
        "voice2tts.gui",
        "voice2tts.wizard",
        "voice2tts.cable",
        "voice2tts.voices",
        "voice2tts.gpupack",
        # The Studio panels are imported inside _build_studio, so they are named
        # here rather than relying on Analysis to find a function-level import.
        # They pull in training, checkpoints and recorder with them. Note
        # piper.train is deliberately absent: training runs in the studio pack's
        # own interpreter, not this one.
        "voice2tts.studioui",
        # Backends chosen by runtime import; Analysis cannot see these.
        "pystray._win32",
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
        # Pulled in lazily by the voice library and GPU pack downloaders.
        "piper.download_voices",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "pandas", "pytest",
        "torch", "torchaudio",  # never used; silero runs on onnxruntime
        "tkinter.test", "test",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

icon_path = PROJECT / "installer" / "voice2tts.ico"
icon = str(icon_path) if icon_path.exists() else None

_common = dict(
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts some ONNX Runtime and CUDA DLLs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

# Windowed build: the tray app, no console flash on launch.
exe = EXE(pyz, a.scripts, [], name="Voice2TTS", console=False, **_common)

# Console build. A windowed bootloader has no stdout at all, so "Voice2TTS --cli"
# from the windowed exe would print into the void; --cli and --check need this one.
exe_console = EXE(
    pyz, a.scripts, [], name="Voice2TTS-console", console=True, **_common
)

coll = COLLECT(
    exe,
    exe_console,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Voice2TTS",
)
