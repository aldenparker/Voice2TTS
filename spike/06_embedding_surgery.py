"""Spike (Phase 0, step 2): swap the speaker lookup for a direct vector input.

A multi-speaker Piper model takes a speaker index and gathers its embedding inside
the graph. Blending needs to supply the vector directly, so this replaces

    emb_g.weight, sid  ->  Gather  ->  /emb_g/Gather_output_0

with a new graph input feeding that same tensor. One node in, one node out.

The proof is not "it runs" but "it produces byte-identical audio when fed the
vector the original would have gathered". Anything less means the surgery moved
the model somewhere subtly different.

    python spike/06_embedding_surgery.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "spike" / "out" / "models" / "en_GB-vctk-medium.onnx"
DST = ROOT / "spike" / "out" / "models" / "en_GB-vctk-medium-embedding.onnx"

EMBEDDING_INPUT = "speaker_embedding"

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def find_lookup(graph) -> tuple[str, object]:
    """The speaker-embedding table and the Gather node that indexes it."""
    table = next((i for i in graph.initializer if "emb_g" in i.name), None)
    if table is None:
        sys.exit("no emb_g tensor; is this a multi-speaker model?")
    gather = next((n for n in graph.node
                   if n.op_type == "Gather" and table.name in n.input), None)
    if gather is None:
        sys.exit(f"nothing gathers {table.name}")
    return table.name, gather


def rewire(model, table_name: str, gather) -> tuple[object, int]:
    """Replace the Gather with a graph input. Returns (model, embedding width)."""
    graph = model.graph
    width = next(numpy_helper.to_array(i).shape[1]
                 for i in graph.initializer if i.name == table_name)
    produced = gather.output[0]

    graph.node.remove(gather)

    # The Gather emitted (1, width); the new input takes its place exactly, so
    # nothing downstream needs to change.
    graph.input.append(
        helper.make_tensor_value_info(
            EMBEDDING_INPUT, onnx.TensorProto.FLOAT, [1, width])
    )
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == produced:
                node.input[i] = EMBEDDING_INPUT

    # sid fed only the lookup, so it is now dead; leaving it would be a required
    # input nobody can meaningfully supply.
    sid = next((i for i in graph.input if i.name == "sid"), None)
    if sid is not None:
        graph.input.remove(sid)

    # The table itself is now unused, but keep it: reading a speaker's vector out
    # of the model is exactly how a blend is seeded.
    return model, width


# scales = (noise_scale, length_scale, noise_w). VITS is stochastic: both noise
# terms feed the duration predictor, so the SAME model produces different audio --
# and different lengths -- on every call. Zeroing them is the only way to compare
# two models for equivalence.
DETERMINISTIC = np.array([0.0, 1.0, 0.0], dtype=np.float32)
NORMAL = np.array([0.667, 1.0, 0.8], dtype=np.float32)


def synthesize(sess, phonemes, embedding=None, sid=None, scales=NORMAL) -> np.ndarray:
    feed = {
        "input": np.array([phonemes], dtype=np.int64),
        "input_lengths": np.array([len(phonemes)], dtype=np.int64),
        "scales": scales,
    }
    if embedding is not None:
        feed[EMBEDDING_INPUT] = embedding
    if sid is not None:
        feed["sid"] = np.array([sid], dtype=np.int64)
    return sess.run(None, feed)[0].reshape(-1)


def main() -> int:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")

    print("=== surgery ===")
    model = onnx.load(str(SRC))
    table_name, gather = find_lookup(model.graph)
    print(f"  table  : {table_name}")
    print(f"  gather : {gather.name}")

    table = next(numpy_helper.to_array(i) for i in model.graph.initializer
                 if i.name == table_name)
    model, width = rewire(model, table_name, gather)
    onnx.checker.check_model(model)
    onnx.save(model, str(DST))
    print(f"  wrote  : {DST.name} ({DST.stat().st_size / 1e6:.1f} MB), "
          f"embedding width {width}")

    print("\n=== signature after surgery ===")
    original = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
    patched = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
    names = [i.name for i in patched.get_inputs()]
    for i in patched.get_inputs():
        print(f"  IN   {i.name:20} {str(i.shape):16} {i.type}")
    check("embedding input added", EMBEDDING_INPUT in names, str(names))
    check("sid input removed", "sid" not in names, str(names))

    # A short phoneme sequence; the exact content does not matter, only that both
    # models get the same one.
    phonemes = [1, 0, 24, 0, 31, 0, 44, 0, 31, 0, 3, 0, 2]

    print("\n=== is the model stochastic? (sanity check on the comparison) ===")
    a1 = synthesize(original, phonemes, sid=0, scales=NORMAL)
    a2 = synthesize(original, phonemes, sid=0, scales=NORMAL)
    check("normal scales vary between runs", len(a1) != len(a2) or
          float(np.abs(a1[:min(len(a1), len(a2))]
                       - a2[:min(len(a1), len(a2))]).max()) > 1e-6,
          f"{len(a1)} vs {len(a2)} samples")
    d1 = synthesize(original, phonemes, sid=0, scales=DETERMINISTIC)
    d2 = synthesize(original, phonemes, sid=0, scales=DETERMINISTIC)
    check("zero noise is reproducible",
          len(d1) == len(d2) and float(np.abs(d1 - d2).max()) < 1e-6,
          f"{len(d1)} vs {len(d2)} samples")

    print("\n=== does it reproduce the original speaker exactly? ===")
    mismatches = []
    for speaker in (0, 1, 42, 108):
        want = synthesize(original, phonemes, sid=speaker, scales=DETERMINISTIC)
        got = synthesize(patched, phonemes, scales=DETERMINISTIC,
                         embedding=table[speaker].reshape(1, -1).astype(np.float32))
        if want.shape != got.shape:
            mismatches.append((speaker, "shape", want.shape, got.shape))
            continue
        peak = float(np.abs(want - got).max())
        if peak > 1e-5:
            mismatches.append((speaker, f"peak {peak:.2e}"))
    check("gathered vector reproduces the indexed speaker", not mismatches,
          str(mismatches) if mismatches else "speakers 0, 1, 42, 108 identical")

    print("\n=== does a blend produce something new? ===")
    a, b = table[0].astype(np.float32), table[1].astype(np.float32)
    blend = ((a + b) / 2.0).reshape(1, -1)
    blended = synthesize(patched, phonemes, embedding=blend)
    voice_a = synthesize(patched, phonemes, embedding=a.reshape(1, -1))
    voice_b = synthesize(patched, phonemes, embedding=b.reshape(1, -1))
    check("blend runs", len(blended) > 0, f"{len(blended)} samples")
    n = min(len(blended), len(voice_a), len(voice_b))
    if n:
        da = float(np.abs(blended[:n] - voice_a[:n]).mean())
        db = float(np.abs(blended[:n] - voice_b[:n]).mean())
        check("blend differs from both parents", da > 1e-4 and db > 1e-4,
              f"mean |diff| a={da:.4f} b={db:.4f}")

    print("\n=== speed cost ===")
    for label, sess, kwargs in (
        ("original", original, {"sid": 0}),
        ("patched ", patched, {"embedding": table[0].reshape(1, -1).astype(np.float32)}),
    ):
        synthesize(sess, phonemes, **kwargs)  # warm
        t0 = time.perf_counter()
        for _ in range(5):
            audio = synthesize(sess, phonemes, **kwargs)
        each = (time.perf_counter() - t0) / 5
        print(f"  {label}: {each * 1000:6.1f} ms  ({len(audio)} samples)")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
