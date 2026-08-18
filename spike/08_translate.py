"""Spike (0.7.0, step 1): is local translation fast enough to sit in the pipeline?

The pipeline currently runs about 300 ms from the end of an utterance to the
first audio out. Translation goes between recognition and speech, so whatever it
costs is added to that, and the whole point of this app is that the other side
hears you in something close to real time. If a small model costs 150 ms this is
easy; if it costs 800 ms the feature needs sentence-level pipelining before it is
worth building at all.

The engine is OPUS-MT on CTranslate2, which faster-whisper already depends on --
so translation adds no new inference runtime. The models come pre-converted from
Argos Translate, whose packages are a zip holding a CTranslate2 model directory
and the SentencePiece tokenizer that goes with it. That avoids needing torch and
transformers just to convert weights we would then never train.

    python spike/08_translate.py            # download if needed, then measure
    python spike/08_translate.py --keep     # leave the model on disk

Downloads ~150 MB on first run.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spike" / "out" / "mt"
INDEX = ("https://raw.githubusercontent.com/argosopentech/argospm-index/"
         "main/index.json")
PAIR = ("en", "de")

# Sentences of the shape this app actually produces: one utterance of speech,
# not a paragraph. The long one is the tail of what push-to-talk yields.
SENTENCES = [
    "Hello, can you hear me?",
    "I am going to share my screen in a moment.",
    "The build finished, but two of the tests are still failing on Windows.",
    "If you can hear this clearly then the virtual microphone is working, "
    "and we can carry on with the rest of the meeting as planned.",
]

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def fetch_package(src: str, dst: str) -> Path:
    """Download and unpack one Argos package. Returns its directory."""
    target = OUT / f"{src}_{dst}"
    if (target / "model").is_dir():
        print(f"  already unpacked at {target}")
        return target

    print(f"  looking up {src} -> {dst} in the Argos index...")
    req = urllib.request.Request(INDEX, headers={"User-Agent": "Voice2TTS"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        index = json.load(resp)
    entry = next((p for p in index
                  if p.get("from_code") == src and p.get("to_code") == dst), None)
    if entry is None:
        sys.exit(f"no {src}->{dst} package published")
    url = (entry.get("links") or [None])[0]
    if not url:
        sys.exit(f"{src}->{dst} has no download link")

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"{src}_{dst}.argosmodel"
    if not archive.exists():
        print(f"  downloading {url}")
        last = [0.0]

        # NOT urlretrieve: it sends Python's default User-Agent and
        # argos-net.com answers that with 403. Worth remembering for the real
        # downloader.
        request = urllib.request.Request(url, headers={"User-Agent": "Voice2TTS"})
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with partial.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total and (done == total or time.time() - last[0] > 1.0):
                        last[0] = time.time()
                        print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB",
                              end="", flush=True)
        print()
        partial.replace(archive)

    print(f"  unpacking {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")
    with zipfile.ZipFile(archive) as zf:
        # Packages contain a single top-level directory.
        zf.extractall(OUT / "_unpack")
    inner = next((p for p in (OUT / "_unpack").iterdir() if p.is_dir()), None)
    if inner is None:
        sys.exit("package did not contain a directory")
    if target.exists():
        import shutil

        shutil.rmtree(target)
    inner.rename(target)
    return target


def main() -> int:
    src, dst = PAIR
    print(f"=== {src} -> {dst} ===")
    package = fetch_package(src, dst)

    print("\n=== what is in the package? ===")
    for item in sorted(package.rglob("*")):
        if item.is_file():
            print(f"  {item.relative_to(package)}  "
                  f"{item.stat().st_size / 1e6:.1f} MB")

    model_dir = package / "model"
    # Argos names it sentencepiece.model. The stanza/ directory in the package
    # is Argos's own sentence splitter -- a torch model we do not need, because
    # this pipeline already works one utterance at a time.
    spm = next((p for p in (package / "sentencepiece.model",
                            *package.rglob("*.spm")) if p.exists()), None)
    check("the package holds a CTranslate2 model", (model_dir / "model.bin").exists(),
          str(model_dir))
    check("and a SentencePiece tokenizer", spm is not None, str(spm))
    if spm is None or not (model_dir / "model.bin").exists():
        return 1

    print("\n=== load ===")
    import ctranslate2
    import sentencepiece

    t0 = time.perf_counter()
    translator = ctranslate2.Translator(str(model_dir), device="cpu",
                                        inter_threads=1, intra_threads=2)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(spm))
    load = time.perf_counter() - t0
    print(f"  loaded in {load * 1000:.0f} ms")
    check("loading is quick enough to do at startup", load < 5.0,
          f"{load:.2f}s")

    def translate(text: str) -> str:
        tokens = tokenizer.encode(text, out_type=str)
        result = translator.translate_batch([tokens], beam_size=4,
                                            max_decoding_length=256)
        return tokenizer.decode(result[0].hypotheses[0])

    print("\n=== quality ===")
    translate("warm up")
    for sentence in SENTENCES:
        print(f"  EN  {sentence}")
        print(f"  DE  {translate(sentence)}")

    print("\n=== latency (what gets added to the pipeline) ===")
    worst = 0.0
    for sentence in SENTENCES:
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            translate(sentence)
            runs.append(time.perf_counter() - t0)
        median = sorted(runs)[len(runs) // 2]
        worst = max(worst, median)
        print(f"  {median * 1000:6.0f} ms  ({len(sentence.split()):2d} words)  "
              f"{sentence[:52]}")

    # The budget: the app is ~300 ms utterance-to-audio today. Adding more than
    # about half that again would be felt in a conversation.
    check("a typical sentence stays inside the latency budget", worst < 0.3,
          f"worst median {worst * 1000:.0f} ms")

    print("\n=== beam size against latency ===")
    long_one = SENTENCES[-1]
    for beam in (1, 2, 4, 8):
        tokens = tokenizer.encode(long_one, out_type=str)
        t0 = time.perf_counter()
        for _ in range(3):
            out = translator.translate_batch([tokens], beam_size=beam,
                                             max_decoding_length=256)
        each = (time.perf_counter() - t0) / 3
        print(f"  beam {beam}: {each * 1000:6.0f} ms   "
              f"{tokenizer.decode(out[0].hypotheses[0])[:60]}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
