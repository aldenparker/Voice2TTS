"""The `.v2tvoice` file: a designed voice as a recipe, not as weights.

A few hundred bytes of TOML naming a base voice, the speakers blended to make
this one, and where the macro controls sit. It does not contain the model.

That is the point. A recipe can be pasted into a chat message, diffed in git,
and kept in a repository next to the code that uses it. It also sidesteps
licensing entirely: sharing one distributes a pointer and some numbers, not a
derivative of anybody's weights. The recipient needs the same base voice, which
they download from the same place everyone else does.

The baked `.onnx` is a build artifact. Delete it and it can be rebuilt from the
recipe; lose the recipe and all you have is a model you can no longer adjust.

Format:

    schema = 1
    name = "Narrator"
    base_voice = "en_GB-vctk-medium"

    [speakers]
    p225 = 0.6
    p243 = 0.4

    [design]
    size = 0.2
    warmth = 0.35
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from .designer import Recipe
from .dsp import MACROS, Design

log = logging.getLogger(__name__)

SCHEMA = 1
SUFFIX = ".v2tvoice"


class UnreadableVoice(Exception):
    """The file is not a voice recipe this version can use."""


@dataclass
class DesignedVoice:
    """Everything needed to rebuild a voice, and nothing that can be rebuilt."""

    name: str = "My Voice"
    base_voice: str = ""
    speakers: dict[str, float] = field(default_factory=dict)
    design: Design = field(default_factory=Design)
    notes: str = ""

    @property
    def recipe(self) -> Recipe:
        return Recipe(base_voice=self.base_voice, weights=dict(self.speakers))

    def to_dict(self) -> dict:
        data: dict = {
            "schema": SCHEMA,
            "name": self.name,
            "base_voice": self.base_voice,
        }
        if self.notes:
            data["notes"] = self.notes
        data["speakers"] = {k: round(float(v), 6) for k, v in self.speakers.items()}
        # Only non-neutral macros are written, so a plain blend produces a file
        # with no [design] section at all rather than six zeroes.
        macros = {k: round(float(v), 4) for k, v in self.design.to_dict().items()
                  if abs(v) > 1e-6}
        if macros:
            data["design"] = macros
        return data

    def save(self, path: Path) -> Path:
        path = path.with_suffix(SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tomli_w.dumps(self.to_dict()).encode("utf-8"))
        log.info("saved %s (%d bytes)", path.name, path.stat().st_size)
        return path


def from_dict(data: dict) -> DesignedVoice:
    schema = data.get("schema")
    if schema is None:
        raise UnreadableVoice("No schema version; this is not a voice recipe.")
    if not isinstance(schema, int) or schema > SCHEMA:
        # Refusing beats guessing: a newer file may mean something different by
        # the same key, and silently misreading it produces a voice that is
        # wrong in ways nobody can trace back to here.
        raise UnreadableVoice(
            f"This voice needs a newer version of Voice2TTS (file schema "
            f"{schema}, this build reads {SCHEMA})."
        )

    base = str(data.get("base_voice") or "").strip()
    if not base:
        raise UnreadableVoice("The recipe does not say which voice it is built on.")

    raw = data.get("speakers") or {}
    if not isinstance(raw, dict) or not raw:
        raise UnreadableVoice("The recipe names no speakers to blend.")
    speakers: dict[str, float] = {}
    for key, value in raw.items():
        try:
            speakers[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise UnreadableVoice(
                f"Speaker {key!r} has a weight that is not a number.") from exc
    if sum(abs(w) for w in speakers.values()) <= 0:
        raise UnreadableVoice("Every speaker weight is zero, so there is no voice.")

    macros = data.get("design") or {}
    if not isinstance(macros, dict):
        raise UnreadableVoice("The [design] section is not a table of macros.")
    unknown = sorted(set(macros) - set(MACROS))
    if unknown:
        # Written by a newer build, or by hand with a typo. Either way, ignoring
        # it silently would leave the voice sounding wrong for no visible reason.
        log.warning("ignoring unknown macros in the recipe: %s", ", ".join(unknown))

    return DesignedVoice(
        name=str(data.get("name") or "My Voice"),
        base_voice=base,
        speakers=speakers,
        design=Design.from_dict({k: v for k, v in macros.items() if k in MACROS}),
        notes=str(data.get("notes") or ""),
    )


def load(path: Path) -> DesignedVoice:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UnreadableVoice(f"Could not read {path.name}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise UnreadableVoice(f"{path.name} is not valid TOML: {exc}") from exc
    return from_dict(data)


def resolve_speakers(voice: DesignedVoice, names: list[str]) -> dict[int, float]:
    """Map the recipe's speaker labels onto this model's indices.

    Labels are matched first, then bare indices, because a recipe written
    against `en_GB-vctk-medium` names speakers like `p225` and those labels are
    what make it readable. A recipe that names a speaker the model does not have
    is refused rather than quietly dropped -- losing one speaker from a blend
    changes the voice, and doing that silently produces something that is not
    what the author heard.
    """
    lookup = {name: index for index, name in enumerate(names)}
    resolved: dict[int, float] = {}
    missing: list[str] = []

    for label, weight in voice.speakers.items():
        if label in lookup:
            resolved[lookup[label]] = resolved.get(lookup[label], 0.0) + weight
            continue
        try:
            index = int(label)
        except ValueError:
            missing.append(label)
            continue
        if 0 <= index < len(names):
            resolved[index] = resolved.get(index, 0.0) + weight
        else:
            missing.append(label)

    if missing:
        raise UnreadableVoice(
            f"{voice.base_voice} has no speaker called "
            + ", ".join(repr(m) for m in missing)
            + ". The recipe was probably written for a different base voice."
        )
    return resolved
