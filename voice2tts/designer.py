"""Build a voice by moving through a multi-speaker model's speaker space.

A multi-speaker VITS model holds a table of speaker embeddings and gathers one by
index. Any point in that space is a voice, not just the ones that happen to have
an index -- so a blend of several speakers is a new voice that nobody recorded.

The blended vector is **baked** into a copy of the model as a constant, which
makes the result an ordinary single-speaker Piper voice. That is the whole design
decision: no separate synthesis path, no engine changes, and a designed voice
gets sentence splitting, streaming, previews and profiles for free because it is
indistinguishable from any other voice in the library. See
`spike/07_bake_blend.py` for the proof, including that baking a speaker back
reproduces that speaker to within 1.3e-6.

Speaker ids are opaque -- `p239`, `TXHC` -- with no gender, age or accent
metadata anywhere in the catalogue. With 904 unlabelled speakers in
en_US-libritts-high, a list is useless and navigating by ear is the only option,
which is why the map is the primary interface.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# The tensor holding the speaker table, and the op that indexes it.
EMBEDDING_TENSOR_HINT = "emb_g"
SPEAKER_INPUT = "sid"


class NotMultiSpeaker(Exception):
    """Raised when a model has no speaker table to move through."""


# -- reading the speaker space ----------------------------------------------


def speaker_table(model_path: Path) -> np.ndarray:
    """The [speakers, width] embedding table from a Piper .onnx."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path), load_external_data=False)
    tensor = next((i for i in model.graph.initializer
                   if EMBEDDING_TENSOR_HINT in i.name), None)
    if tensor is None:
        raise NotMultiSpeaker(
            f"{model_path.name} has no speaker table, so there is no space to "
            "move through. Pick a multi-speaker voice."
        )
    table = numpy_helper.to_array(tensor)
    if table.ndim != 2 or table.shape[0] < 2:
        raise NotMultiSpeaker(
            f"{model_path.name} has a speaker table of shape {table.shape}; "
            "at least two speakers are needed to blend."
        )
    return np.asarray(table, dtype=np.float32)


def speaker_names(config_path: Path, count: int) -> list[str]:
    """Speaker labels from the voice config, or positional ids as a fallback."""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("no speaker map in %s: %s", config_path, exc)
        return [str(i) for i in range(count)]

    mapping = config.get("speaker_id_map") or {}
    names = [str(i) for i in range(count)]
    for name, index in mapping.items():
        if isinstance(index, int) and 0 <= index < count:
            names[index] = str(name)
    return names


def is_multi_speaker(config_path: Path) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(config.get("num_speakers") or 1) > 1


# -- blending ---------------------------------------------------------------


@dataclass
class Recipe:
    """Named speakers and their weights. The reproducible form of a blend."""

    base_voice: str = ""
    weights: dict[str, float] = field(default_factory=dict)

    def normalised(self) -> dict[str, float]:
        """Weights summing to 1. An all-zero recipe stays empty rather than
        dividing by zero."""
        total = sum(abs(w) for w in self.weights.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in self.weights.items()}

    @property
    def is_empty(self) -> bool:
        return not self.normalised()


def blend(table: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    """Weighted average of speaker vectors. Returns a [width] float32 vector.

    Weights are normalised, because the embedding space is not scale-invariant:
    doubling every weight would move the result away from the speaker manifold
    entirely and produce something that does not sound like a voice at all.
    """
    total = sum(abs(w) for w in weights.values())
    if total <= 0:
        raise ValueError("A blend needs at least one speaker with a non-zero weight.")

    out = np.zeros(table.shape[1], dtype=np.float64)
    for index, weight in weights.items():
        if not 0 <= index < table.shape[0]:
            raise IndexError(
                f"speaker {index} is outside this model's {table.shape[0]} speakers")
        out += table[index] * (weight / total)
    return out.astype(np.float32)


def nearest(table: np.ndarray, point: np.ndarray, count: int = 4
            ) -> list[tuple[int, float]]:
    """The `count` speakers closest to a point, as (index, distance)."""
    distances = np.linalg.norm(table - point.reshape(1, -1), axis=1)
    order = np.argsort(distances)[:max(1, count)]
    return [(int(i), float(distances[i])) for i in order]


def blend_by_distance(table: np.ndarray, point: np.ndarray, count: int = 4,
                      power: float = 2.0) -> tuple[np.ndarray, dict[int, float]]:
    """Blend the nearest speakers to a point, weighted by inverse distance.

    This is what clicking an empty area of the map does. Returns the vector and
    the weights used, so a click can be turned into an editable recipe.
    """
    neighbours = nearest(table, point, count)
    # A click that lands exactly on a speaker is that speaker, not a division
    # by zero.
    exact = [(i, d) for i, d in neighbours if d < 1e-9]
    if exact:
        index = exact[0][0]
        return table[index].astype(np.float32), {index: 1.0}

    weights = {i: 1.0 / (d ** power) for i, d in neighbours}
    return blend(table, weights), weights


# -- the map ----------------------------------------------------------------


def project(table: np.ndarray) -> np.ndarray:
    """Speaker table to 2D, by PCA. Returns [speakers, 2], roughly centred.

    PCA rather than UMAP deliberately: UMAP means numba and llvmlite, hundreds of
    megabytes added to an installer, to lay out a few hundred points. PCA is a
    single SVD, needs nothing beyond numpy, and is deterministic -- the same
    model always produces the same map, so a position stays meaningful between
    sessions and between machines.

    The projection is only a way to navigate. Distances in 2D are approximate;
    every blend is computed in the full space.
    """
    centred = table - table.mean(axis=0, keepdims=True)
    # economy SVD: for 904x512 this is milliseconds.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T

    # Scale to roughly [-1, 1] so the UI does not need to know about the model.
    extent = np.abs(coords).max() or 1.0
    return (coords / extent).astype(np.float32)


def from_map(table: np.ndarray, coords: np.ndarray, x: float, y: float,
             count: int = 4) -> tuple[np.ndarray, dict[int, float]]:
    """Turn a click on the 2D map into a blend of nearby speakers.

    Neighbours are chosen in the 2D projection, because that is what the user can
    see and point at, but the blend itself is computed from the full embeddings.
    Choosing them in 512 dimensions instead would pick speakers that are nowhere
    near the click on screen.
    """
    point = np.array([x, y], dtype=np.float32)
    distances = np.linalg.norm(coords - point.reshape(1, -1), axis=1)
    order = np.argsort(distances)[:max(1, count)]

    if float(distances[order[0]]) < 1e-6:
        index = int(order[0])
        return table[index].astype(np.float32), {index: 1.0}

    weights = {int(i): 1.0 / float(distances[i]) ** 2 for i in order}
    return blend(table, weights), weights


# -- baking -----------------------------------------------------------------


def bake(base_model: Path, base_config: Path, vector: np.ndarray,
         dest: Path, name: str = "") -> Path:
    """Freeze `vector` into a copy of the model. Returns the .onnx path.

    The result is a single-speaker voice in every respect that matters: the
    Gather is gone, `sid` is no longer an input, the config says one speaker, and
    the now-unused table is dropped (which makes the file smaller than the base).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(base_model))
    graph = model.graph

    tensor = next((i for i in graph.initializer
                   if EMBEDDING_TENSOR_HINT in i.name), None)
    if tensor is None:
        raise NotMultiSpeaker(f"{base_model.name} has no speaker table to replace")
    width = numpy_helper.to_array(tensor).shape[1]
    if vector.size != width:
        raise ValueError(
            f"embedding is {vector.size} wide, this model expects {width}")

    gather = next((n for n in graph.node
                   if n.op_type == "Gather" and tensor.name in n.input), None)
    if gather is None:
        raise NotMultiSpeaker(f"nothing indexes {tensor.name} in {base_model.name}")

    produced = gather.output[0]
    graph.node.remove(gather)
    graph.initializer.append(numpy_helper.from_array(
        vector.reshape(1, width).astype(np.float32), name=produced))

    # sid fed only the lookup. Leaving it in place would make it a required input
    # that a single-speaker config never supplies.
    sid = next((i for i in graph.input if i.name == SPEAKER_INPUT), None)
    if sid is not None:
        graph.input.remove(sid)
    graph.initializer.remove(tensor)

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix(".onnx.partial")
    onnx.save(model, str(staging))

    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["num_speakers"] = 1
    config["speaker_id_map"] = {}
    if name:
        config.setdefault("dataset", name)
    # Written before the model is moved into place, so a voice never exists
    # without the config that says how to read it.
    dest.with_suffix(".onnx.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.replace(dest)

    # Only now that the new voice is complete: whatever used to live at this
    # path is not this voice, and a previous occupant's effects sidecar would
    # be inherited with nothing in the interface to explain the difference.
    # Done last so there is never a moment where the model has no config.
    from .voices import clear_sidecars

    clear_sidecars(dest, keep=(".onnx.json",))

    log.info("baked %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def design_path(model_path: Path) -> Path:
    """Where the macro settings live for a voice, if it has any.

    Kept beside the model rather than inside its config.json: that file is
    Piper's, and adding keys to it risks a future Piper version rejecting or
    overwriting them.
    """
    return model_path.with_suffix(".onnx.design.json")


def write_design(model_path: Path, design, name: str = "") -> Path:
    """Record the effects chain for a baked voice. Returns the sidecar path."""
    path = design_path(model_path)
    payload = {"schema": 1, "name": name, "design": design.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_design(model_path: Path):
    """The macro settings for a voice, or None if it is an ordinary one."""
    from .dsp import Design

    path = design_path(model_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # A voice that speaks without its effects is much better than a voice
        # that refuses to speak.
        log.warning("ignoring unreadable design sidecar %s: %s", path.name, exc)
        return None
    return Design.from_dict(data.get("design") or {})


def remove_designed(model_path: Path) -> bool:
    """Delete a designed voice and everything written beside it."""
    removed = False
    for suffix in (".onnx", ".onnx.json", ".onnx.design.json", ".v2tvoice"):
        candidate = model_path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()
            removed = True
    shutil.rmtree(model_path.parent / "__pycache__", ignore_errors=True)
    return removed
