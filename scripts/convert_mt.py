"""Convert Helsinki-NLP OPUS-MT models to CTranslate2, for release assets.

Runs at build time, never on a user's machine: it needs `transformers` and
`torch` purely to read weights we then rewrite, and neither belongs in the
installer.

WHY WE CONVERT RATHER THAN USE SOMEBODY ELSE'S BUILD
----------------------------------------------------
Pre-converted OPUS-MT packages exist and would have saved this work. The
problem is licensing: the ones surveyed state terms for their *software* and
say nothing at all about the model weights -- not in the index, not in the
package, not in the README. "Probably inherits the upstream licence" is not a
licence, and a GPL application should not put a download button on a guess.

Helsinki-NLP publish the originals under CC-BY-4.0, stated plainly on each
model card. Converting them ourselves means the terms are known, the
attribution CC-BY requires travels with the model, and the bytes come from our
own release rather than a third party who might change them.

    python scripts/convert_mt.py en de
    python scripts/convert_mt.py --all          # every pair in PAIRS
    python scripts/convert_mt.py en de --out dist/mt

Each pair produces a directory holding what voice2tts/translate.py expects,
plus the attribution:

    en_de/
        model/                  CTranslate2
        source.spm, target.spm  the two tokenizers
        LICENSE                 CC-BY-4.0 notice naming the source model
        metadata.json           languages, source repo, checksum

and a zip of it ready to attach to a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dist" / "mt"

REPO_TEMPLATE = "Helsinki-NLP/opus-mt-{source}-{target}"
MODEL_CARD_URL = "https://huggingface.co/{repo}"

# The pairs a release ships. English both ways first -- pivoting through
# English is how everything else is reached, so these carry the most weight.
PAIRS = [
    ("en", "de"), ("de", "en"),
    ("en", "fr"), ("fr", "en"),
    ("en", "es"), ("es", "en"),
    ("en", "it"), ("it", "en"),
    ("en", "nl"), ("nl", "en"),
    ("en", "ru"), ("ru", "en"),
    ("en", "zh"), ("zh", "en"),
    ("en", "ja"), ("ja", "en"),
]

LICENCE_NOTICE = """\
{source} to {target} translation model
=======================================

Converted for Voice2TTS from:

    {repo}
    {card}

Licence: CC-BY-4.0 (Creative Commons Attribution 4.0 International)
    https://creativecommons.org/licenses/by/4.0/

Original work by the Helsinki-NLP group (Language Technology Research Group at
the University of Helsinki), trained on the OPUS corpus. If you use this model,
CC-BY-4.0 requires that you credit them.

    Tiedemann, J. and Thottingal, S. (2020) OPUS-MT -- Building open
    translation services for the World. Proceedings of the 22nd Annual
    Conference of the European Association for Machine Translation (EAMT).

WHAT WAS CHANGED
    The weights were converted from the PyTorch format to CTranslate2 with
    `ct2-transformers-converter`, quantised to {quantization}. Nothing was
    retrained or fine-tuned. The SentencePiece tokenizer is the original file,
    unmodified.
"""


def convert(source: str, target: str, out_dir: Path,
            quantization: str = "int8") -> Path:
    """Convert one pair. Returns the directory produced."""
    repo = REPO_TEMPLATE.format(source=source, target=target)
    destination = out_dir / f"{source}_{target}"
    staging = out_dir / f".{source}_{target}.partial"

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"  converting {repo} -> {destination.name} ({quantization})")
    # The converter is a console script from ctranslate2. Run it as a module so
    # it works the same whether or not Scripts/ is on PATH.
    result = subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", repo,
         "--output_dir", str(staging / "model"),
         "--quantization", quantization,
         "--force"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError(f"converting {repo} failed:\n" + "\n".join(tail))

    for name in _fetch_tokenizers(repo, staging):
        print(f"    tokenizer: {name.name} ({name.stat().st_size / 1e3:.0f} KB)")

    (staging / "LICENSE").write_text(
        LICENCE_NOTICE.format(source=source, target=target, repo=repo,
                              card=MODEL_CARD_URL.format(repo=repo),
                              quantization=quantization),
        encoding="utf-8")

    size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    (staging / "metadata.json").write_text(json.dumps({
        "schema": 1,
        "source": source,
        "target": target,
        "origin": repo,
        "licence": "CC-BY-4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "quantization": quantization,
        "bytes": size,
    }, indent=2), encoding="utf-8")

    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
    print(f"    {size / 1e6:.0f} MB")
    return destination


def _fetch_tokenizers(repo: str, staging: Path) -> list[Path]:
    """Download BOTH SentencePiece models the weights were built with.

    Marian keeps two: one for the language going in, one for the language
    coming out. Decoding the output with the source tokenizer leaves raw
    "▁" pieces in the text -- it does not fail, it just produces something
    that looks almost right.

    A model without its tokenizers produces confident nonsense rather than an
    error, so a miss has to be loud.
    """
    from huggingface_hub import hf_hub_download

    fetched: list[Path] = []
    for name in ("source.spm", "target.spm"):
        try:
            path = hf_hub_download(repo_id=repo, filename=name)
        except Exception as exc:
            raise RuntimeError(f"{repo} has no {name}: {exc}") from exc
        destination = staging / name
        shutil.copy2(path, destination)
        fetched.append(destination)
    return fetched


def package(directory: Path) -> Path:
    """Zip a converted pair for attaching to a release."""
    archive = directory.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(directory))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"    packaged {archive.name} "
          f"({archive.stat().st_size / 1e6:.0f} MB)")
    return archive


def write_manifest(out_dir: Path) -> Path:
    """List every packaged pair, for the app to read before downloading.

    Uploaded alongside the archives so the app can show what is available, how
    big it is, and under what terms -- without downloading 60 MB to find out,
    and without the list being hardcoded into a build that then goes stale.
    """
    entries = []
    for archive in sorted(out_dir.glob("*.zip")):
        source, _, target = archive.stem.partition("_")
        checksum = archive.with_suffix(".zip.sha256")
        metadata = out_dir / archive.stem / "metadata.json"
        info = (json.loads(metadata.read_text(encoding="utf-8"))
                if metadata.is_file() else {})
        entries.append({
            "source": source,
            "target": target,
            "asset": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": (checksum.read_text(encoding="utf-8").split()[0]
                       if checksum.is_file() else ""),
            "origin": info.get("origin", ""),
            "licence": info.get("licence", ""),
            "licence_url": info.get("licence_url", ""),
        })

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "pairs": entries}, indent=2),
                        encoding="utf-8")
    print(f"manifest: {len(entries)} pair(s) -> {manifest.name}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="source language, e.g. en")
    ap.add_argument("target", nargs="?", help="target language, e.g. de")
    ap.add_argument("--all", action="store_true", help="convert every pair in PAIRS")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quantization", default="int8",
                    choices=["int8", "int8_float16", "float16", "float32"])
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if args.all:
        wanted = PAIRS
    elif args.source and args.target:
        wanted = [(args.source, args.target)]
    else:
        ap.error("give a language pair, or --all")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    total = 0
    for source, target in wanted:
        print(f"{source} -> {target}")
        try:
            directory = convert(source, target, out_dir, args.quantization)
        except Exception as exc:  # noqa: BLE001 - one bad pair is not fatal
            print(f"    FAILED: {exc}")
            failures.append(f"{source}-{target}")
            continue
        if not args.no_zip:
            total += package(directory).stat().st_size

    if not args.no_zip:
        write_manifest(out_dir)

    print(f"\n{len(wanted) - len(failures)}/{len(wanted)} converted"
          + (f", {total / 1e6:.0f} MB of archives" if total else ""))
    if failures:
        print(f"failed: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
