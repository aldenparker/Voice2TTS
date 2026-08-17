"""Spike (Phase 3, step 1): can a blend be BAKED into an ordinary Piper voice?

Phase 0 proved a blended embedding can be fed into a patched graph. That leaves
the designer needing its own synthesis path, separate from PiperVoice -- and so
separate from the streaming, the sentence splitting, the phonemizer, previews,
profiles and everything else the app already does.

This asks a better question: instead of turning `sid` into an input, can the
blended vector be frozen into the graph as a constant, so the result is just a
single-speaker Piper voice?

If yes, a designed voice needs no engine changes at all. It appears in the voice
library, loads through PiperVoice, and works everywhere.

    python spike/07_bake_blend.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "spike" / "out" / "models" / "en_GB-vctk-medium.onnx"
OUT = ROOT / "spike" / "out" / "models"
BAKED = OUT / "en_GB-vctk-baked.onnx"

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def embedding_table(model) -> tuple[str, np.ndarray]:
    tensor = next(i for i in model.graph.initializer if "emb_g" in i.name)
    return tensor.name, numpy_helper.to_array(tensor)


def bake(model, vector: np.ndarray):
    """Freeze `vector` in place of the speaker lookup.

    The Gather that indexed the table is replaced by an initializer holding the
    chosen vector, so nothing downstream can tell the difference between this and
    a model that only ever had one speaker.
    """
    graph = model.graph
    table_name, _ = embedding_table(model)
    gather = next(n for n in graph.node
                  if n.op_type == "Gather" and table_name in n.input)
    produced = gather.output[0]
    graph.node.remove(gather)

    frozen = numpy_helper.from_array(
        vector.reshape(1, -1).astype(np.float32), name=produced)
    graph.initializer.append(frozen)

    # sid fed only the lookup. Leaving it would be a required input that a
    # single-speaker config never supplies.
    sid = next((i for i in graph.input if i.name == "sid"), None)
    if sid is not None:
        graph.input.remove(sid)

    # The table is now dead weight -- 109 x 512 floats we no longer index.
    table = next(i for i in graph.initializer if i.name == table_name)
    graph.initializer.remove(table)
    return model


def main() -> int:
    if not SRC.exists():
        sys.exit(f"missing {SRC}; run spike/04 first")

    print("=== bake a 50/50 blend ===")
    model = onnx.load(str(SRC))
    _name, table = embedding_table(model)
    print(f"  table: {table.shape}")

    blend = (table[0] + table[1]) / 2.0
    bake(model, blend)
    onnx.checker.check_model(model)
    onnx.save(model, str(BAKED))

    grew = BAKED.stat().st_size - SRC.stat().st_size
    print(f"  wrote {BAKED.name}: {BAKED.stat().st_size / 1e6:.1f} MB "
          f"({grew / 1e3:+.0f} KB vs the base)")
    check("baking does not inflate the model", abs(grew) < 1e6, f"{grew:+d} bytes")

    # A designed voice must look single-speaker, or PiperVoice will try to pass a
    # speaker id that the baked graph no longer accepts.
    config = json.loads((SRC.with_suffix(".onnx.json")).read_text(encoding="utf-8"))
    print(f"  base config says num_speakers={config.get('num_speakers')}")
    config["num_speakers"] = 1
    config["speaker_id_map"] = {}
    BAKED.with_suffix(".onnx.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== does PiperVoice load and speak it? ===")
    from piper import PiperVoice, SynthesisConfig

    try:
        voice = PiperVoice.load(BAKED, use_cuda=False)
    except Exception as exc:  # noqa: BLE001 - this is the result being measured
        check("PiperVoice loads the baked voice", False, f"{type(exc).__name__}: {exc}")
        print(f"\n{passed} passed, {failed} failed")
        return 1
    check("PiperVoice loads the baked voice", True,
          f"{voice.config.sample_rate} Hz, num_speakers={voice.config.num_speakers}")

    syn = SynthesisConfig(length_scale=1.0)
    t0 = time.perf_counter()
    chunks = [np.asarray(c.audio_float_array, dtype=np.float32)
              for c in voice.synthesize("The blended voice speaks.", syn_config=syn)]
    elapsed = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    check("it synthesises through the normal path", len(audio) > 1000,
          f"{len(audio)} samples in {elapsed * 1000:.0f} ms")
    check("the audio is not silence", float(np.abs(audio).max()) > 0.01,
          f"peak {float(np.abs(audio).max()):.3f}")

    print("\n=== is it actually the blend, not one of the parents? ===")
    # Compare against parents baked the same way, at zero noise so the comparison
    # is between voices rather than between random draws.
    def bake_and_speak(vector, tag):
        m = onnx.load(str(SRC))
        bake(m, vector)
        path = OUT / f"en_GB-vctk-{tag}.onnx"
        onnx.save(m, str(path))
        path.with_suffix(".onnx.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        v = PiperVoice.load(path, use_cuda=False)
        quiet = SynthesisConfig(length_scale=1.0, noise_scale=0.0, noise_w_scale=0.0)
        parts = [np.asarray(c.audio_float_array, dtype=np.float32)
                 for c in v.synthesize("The blended voice speaks.", syn_config=quiet)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    a = bake_and_speak(table[0], "spk0")
    b = bake_and_speak(table[1], "spk1")
    mid = bake_and_speak(blend, "mid")

    # Pinning the noise makes each voice reproducible, but NOT the same length as
    # another voice: the duration predictor is conditioned on the speaker
    # embedding, so a different speaker genuinely speaks the sentence in a
    # different time. Comparing waveforms sample-by-sample is therefore
    # meaningless between voices -- the signals are time-misaligned, and the
    # difference measures the misalignment rather than the timbre.
    check("different speakers give different durations, even with noise pinned",
          len({len(a), len(b), len(mid)}) > 1, f"{len(a)}, {len(b)}, {len(mid)}")

    def spectrum(x: np.ndarray) -> np.ndarray:
        """Long-term average spectrum: duration-independent, so it compares timbre."""
        frame = 1024
        usable = len(x) // frame * frame
        if not usable:
            return np.zeros(frame // 2 + 1)
        frames = x[:usable].reshape(-1, frame) * np.hanning(frame)
        mag = np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)
        return mag / (np.linalg.norm(mag) or 1.0)

    sa, sb, smid = spectrum(a), spectrum(b), spectrum(mid)
    da = float(np.linalg.norm(smid - sa))
    db = float(np.linalg.norm(smid - sb))
    dab = float(np.linalg.norm(sa - sb))
    check("the blend is a different voice from both parents",
          da > 1e-3 and db > 1e-3, f"|mid-a|={da:.4f} |mid-b|={db:.4f}")
    check("and sits between them rather than beyond",
          da < dab and db < dab, f"{da:.4f}, {db:.4f} vs |a-b|={dab:.4f}")

    print("\n=== round trip: does a baked parent match the original speaker? ===")
    original = PiperVoice.load(SRC, use_cuda=False)
    quiet = SynthesisConfig(speaker_id=0, length_scale=1.0,
                            noise_scale=0.0, noise_w_scale=0.0)
    parts = [np.asarray(c.audio_float_array, dtype=np.float32)
             for c in original.synthesize("The blended voice speaks.", syn_config=quiet)]
    want = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    n = min(len(want), len(a))
    peak = float(np.abs(want[:n] - a[:n]).max()) if n else 1.0
    check("baking speaker 0 reproduces speaker 0", len(want) == len(a) and peak < 1e-4,
          f"{len(want)} vs {len(a)} samples, peak diff {peak:.2e}")

    for tag in ("spk0", "spk1", "mid"):
        for suffix in (".onnx", ".onnx.json"):
            (OUT / f"en_GB-vctk-{tag}{suffix}").unlink(missing_ok=True)
    shutil.rmtree(OUT / "__pycache__", ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
