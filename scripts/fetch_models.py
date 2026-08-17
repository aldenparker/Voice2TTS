"""Download everything the app needs to run offline.

    python scripts/fetch_models.py                       # defaults
    python scripts/fetch_models.py --voice en_GB-alba-medium --whisper base.en
    python scripts/fetch_models.py --list-voices
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
VAD_DEST = ROOT / "models" / "vad" / "silero_vad.onnx"
VOICES_DIR = ROOT / "models" / "voices"

# What the installer ships. Three distinct American timbres; everything else in the
# catalogue is downloadable from the app's Voice library tab.
BUNDLED_VOICES = ["en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-high"]
BUNDLED_WHISPER = "base.en"

POPULAR_VOICES = [
    "en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-high",
    "en_US-joe-medium", "en_US-kristin-medium", "en_GB-alba-medium",
    "en_GB-northern_english_male-medium", "en_GB-jenny_dioco-medium",
]


def _progress(name: str):
    def hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = done * 100 // total
        sys.stdout.write(f"\r  {name}: {pct:3d}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")

    return hook


def fetch_vad(force: bool = False) -> None:
    if VAD_DEST.exists() and not force:
        print(f"VAD model already present ({VAD_DEST.stat().st_size / 1e6:.1f} MB)")
        return
    VAD_DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Silero VAD...")
    urllib.request.urlretrieve(VAD_URL, VAD_DEST, _progress("silero_vad.onnx"))


def fetch_voice(name: str, force: bool = False) -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    if (VOICES_DIR / f"{name}.onnx").exists() and not force:
        print(f"Voice {name} already present")
        return
    print(f"Downloading Piper voice {name}...")
    from piper.download_voices import download_voice

    download_voice(name, VOICES_DIR)
    print(f"  saved to {VOICES_DIR / name}.onnx")


# festvox.org serves no HTTPS at all -- port 443 is actively refused -- so this
# can only be fetched over plain HTTP. The file is therefore COMMITTED rather than
# downloaded at build time: 75 KB is small enough to version, it makes builds
# reproducible and offline, and it removes an unauthenticated download from the
# path of something we ship inside the installer. The checksum below is what was
# reviewed; refreshing verifies against it.
PROMPTS_URL = "http://festvox.org/cmu_arctic/cmuarctic.data"
PROMPTS_SHA256 = "60e3d9a4dc33732c9100baadd747312bdc1a200fc891766507397289753a25c7"
PROMPTS_DEST = ROOT / "models" / "prompts" / "arctic.txt"


def fetch_prompts(force: bool = False) -> None:
    """Refresh the CMU ARCTIC prompt list. Normally unnecessary -- it is committed.

    1132 phonetically balanced sentences drawn from out-of-copyright Project
    Gutenberg texts, published by CMU for building voices.
    """
    if PROMPTS_DEST.exists() and not force:
        count = sum(1 for ln in PROMPTS_DEST.read_text(encoding="utf-8").splitlines()
                    if ln.strip())
        print(f"Prompt corpus present ({count} prompts, committed to the repo)")
        return

    import hashlib

    PROMPTS_DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the CMU ARCTIC prompt list from {PROMPTS_URL}...")
    tmp = PROMPTS_DEST.with_suffix(".tmp")
    urllib.request.urlretrieve(PROMPTS_URL, tmp, _progress("arctic.txt"))

    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != PROMPTS_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Prompt corpus checksum mismatch -- refusing it.\n"
            f"  expected {PROMPTS_SHA256}\n  got      {digest}\n"
            "This is fetched over plain HTTP, so a mismatch means either the "
            "upstream file changed or the download was tampered with. Review the "
            "content before updating PROMPTS_SHA256."
        )
    tmp.replace(PROMPTS_DEST)
    count = sum(1 for ln in PROMPTS_DEST.read_text(encoding="utf-8").splitlines()
                if ln.strip())
    print(f"  {count} prompts, checksum verified -> {PROMPTS_DEST}")


def fetch_whisper(model: str) -> None:
    """Pull the CTranslate2 weights into our cache ahead of first run."""
    print(f"Fetching Whisper model {model} (this can take a few minutes)...")
    from voice2tts.cuda import prepare_cuda
    from voice2tts.paths import whisper_cache

    prepare_cuda()
    from faster_whisper import WhisperModel

    # Load on CPU purely to force the download; the real run picks its own device.
    WhisperModel(model, device="cpu", compute_type="int8",
                 download_root=str(whisper_cache()))
    print(f"  {model} cached")


def bundle_whisper(model: str, force: bool = False) -> None:
    """Materialise a Whisper model into models/whisper/<name> for packaging.

    The installer ships this so a fresh install transcribes offline with no first-run
    download. Fetched as a plain snapshot rather than via the HF cache, because
    PyInstaller needs real files at a predictable path, not a tree of symlinks.
    """
    dest = ROOT / "models" / "whisper" / model
    if (dest / "model.bin").exists() and not force:
        size = sum(p.stat().st_size for p in dest.rglob("*")) / 1e6
        print(f"Bundled Whisper {model} already present ({size:.0f} MB)")
        return

    repo = f"Systran/faster-whisper-{model}"
    print(f"Downloading {repo} for bundling...")
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=["*.bin", "*.json", "*.txt"],
    )
    if not (dest / "model.bin").exists():
        raise RuntimeError(f"{repo} did not yield a model.bin")
    size = sum(p.stat().st_size for p in dest.rglob("*")) / 1e6
    print(f"  bundled to {dest} ({size:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", default="en_US-lessac-medium")
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--bundle", action="store_true",
                    help="fetch everything the installer ships (3 voices + base.en)")
    args = ap.parse_args()

    if args.bundle:
        fetch_vad(args.force)
        fetch_prompts(args.force)
        for voice in BUNDLED_VOICES:
            fetch_voice(voice, args.force)
        bundle_whisper(BUNDLED_WHISPER, args.force)
        print("\nBundle assets ready. Build with:  .\\build.ps1")
        return 0

    if args.list_voices:
        print("Popular Piper voices (full catalogue: "
              "https://huggingface.co/rhasspy/piper-voices):")
        for v in POPULAR_VOICES:
            print(f"  {v}")
        return 0

    fetch_vad(args.force)
    fetch_voice(args.voice, args.force)
    if not args.skip_whisper:
        fetch_whisper(args.whisper)

    print("\nDone. Verify with:  python -m voice2tts --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
