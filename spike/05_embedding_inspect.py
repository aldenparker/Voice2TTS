"""Spike (Phase 0, step 1): can we reach the speaker embedding in a Piper model?

Multi-speaker Piper models take a speaker INDEX and look the embedding up inside
the ONNX graph. Blending voices needs to supply a VECTOR instead, which means
finding that lookup and understanding what feeds it.

Read-only: this only reports what is in the graph.

    python spike/05_embedding_inspect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "spike" / "out" / "models" / "en_GB-vctk-medium.onnx"


def main() -> int:
    if not MODEL.exists():
        sys.exit(f"missing {MODEL}\nDownload it first (see ROADMAP Phase 0).")

    print(f"model: {MODEL.name}  ({MODEL.stat().st_size / 1e6:.1f} MB)")

    print("\n=== runtime signature ===")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  IN   {i.name:16} {str(i.shape):24} {i.type}")
    for o in sess.get_outputs():
        print(f"  OUT  {o.name:16} {str(o.shape):24} {o.type}")

    model = onnx.load(str(MODEL))
    graph = model.graph
    print(f"\n=== graph ===\n  {len(graph.node)} nodes, "
          f"{len(graph.initializer)} initializers")

    print("\n=== candidate speaker-embedding tensors ===")
    # VITS names the speaker embedding emb_g; fall back to shape-matching in case
    # this export used different names.
    candidates = []
    for init in graph.initializer:
        dims = list(init.dims)
        named = "emb_g" in init.name
        # A speaker table is (num_speakers, channels) with a plausible width.
        shaped = len(dims) == 2 and 2 <= dims[0] <= 5000 and 64 <= dims[1] <= 1024
        if named or shaped:
            candidates.append((init.name, dims, named))
    for name, dims, named in candidates:
        tag = "  <-- named emb_g" if named else ""
        print(f"  {name:44} {dims}{tag}")
    if not candidates:
        print("  none found")
        return 1

    emb_name, emb_dims, _ = next(
        (c for c in candidates if c[2]), candidates[0])
    print(f"\n  using: {emb_name} {emb_dims}")

    print("\n=== who consumes it? ===")
    consumers = [n for n in graph.node if emb_name in n.input]
    for node in consumers:
        print(f"  {node.op_type:12} name={node.name!r}")
        print(f"     inputs : {list(node.input)}")
        print(f"     outputs: {list(node.output)}")

    print("\n=== where does 'sid' go? ===")
    sid_users = [n for n in graph.node if any("sid" in i for i in n.input)]
    for node in sid_users[:6]:
        print(f"  {node.op_type:12} inputs={list(node.input)} "
              f"outputs={list(node.output)}")

    print("\n=== embedding statistics (what a blend would operate on) ===")
    table = None
    for init in graph.initializer:
        if init.name == emb_name:
            table = numpy_helper.to_array(init)
            break
    if table is not None:
        print(f"  shape {table.shape}, dtype {table.dtype}")
        print(f"  per-vector norm: mean {np.linalg.norm(table, axis=1).mean():.3f} "
              f"min {np.linalg.norm(table, axis=1).min():.3f} "
              f"max {np.linalg.norm(table, axis=1).max():.3f}")
        # If speakers were all near-identical, blending would achieve nothing.
        a, b = table[0], table[1]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"  cosine similarity speaker 0 vs 1: {cos:.3f}")
        pairs = min(40, len(table))
        sims = []
        for i in range(pairs):
            for j in range(i + 1, pairs):
                x, y = table[i], table[j]
                sims.append(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        print(f"  mean pairwise cosine over {pairs} speakers: {np.mean(sims):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
