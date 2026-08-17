"""Sentences for the speaker to read while recording a voice.

Handing people the script is the difference between a chore and a task. "Read for
thirty minutes" means finding something to read and deciding when to stop; "read
this sentence" is over in five seconds and the progress bar moves. It also gives
far better phonetic coverage than whatever someone would improvise, which is what
the voice quality actually depends on.

The corpus is CMU ARCTIC's prompt list: 1132 sentences selected for phonetic
balance, drawn from out-of-copyright Project Gutenberg texts and published by CMU
for exactly this purpose. Fetched as a build asset like the voice models rather
than committed here.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from .paths import resource_root

log = logging.getLogger(__name__)

# Festival format: ( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
_ARCTIC_LINE = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')

# Speaking rates vary enormously, so the estimate starts here and is replaced by
# the speaker's own measured rate as soon as there is anything to measure.
DEFAULT_WPM = 150.0

# Enough to get started if the corpus file is missing, so the studio is never
# completely unusable. Phonetically varied, but no substitute for the real list.
FALLBACK = (
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on a sunny afternoon.",
    "Please call Stella and ask her to bring these things with her.",
    "The rainbow is a division of white light into many beautiful colours.",
    "Six thick slabs of blue cheese sat beside the warm bread.",
    "How much wood would a woodchuck chuck if it could chuck wood?",
    "They journeyed north through the quiet valley before dawn.",
    "A joyful mixture of laughter and music filled the evening air.",
)


@dataclass(frozen=True)
class Prompt:
    key: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())

    def estimated_seconds(self, wpm: float = DEFAULT_WPM) -> float:
        # Plus a beat at each end for the breath before and the pause after.
        return (self.words / max(60.0, wpm)) * 60.0 + 0.8


def corpus_path():
    return resource_root() / "models" / "prompts" / "arctic.txt"


def parse(text: str) -> list[Prompt]:
    """Read the Festival-style prompt format, ignoring anything unrecognised."""
    prompts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _ARCTIC_LINE.match(line)
        if match:
            key, body = match.group(1), match.group(2).strip()
            if body:
                prompts.append(Prompt(key, body))
        elif not line.startswith((";", "#")):
            # A plain-text file, one sentence per line, is also accepted so people
            # can drop in their own script.
            prompts.append(Prompt(f"line{len(prompts) + 1:05d}", line))
    return prompts


def load() -> list[Prompt]:
    """The prompt corpus, falling back to a small built-in set."""
    path = corpus_path()
    if path.exists():
        try:
            prompts = parse(path.read_text(encoding="utf-8"))
            if prompts:
                log.info("loaded %d prompts from %s", len(prompts), path.name)
                return prompts
        except OSError as exc:
            log.warning("could not read %s: %s", path, exc)
    log.warning("prompt corpus missing; using the built-in fallback")
    return [Prompt(f"fallback{i:03d}", t) for i, t in enumerate(FALLBACK, 1)]


def measured_wpm(total_words: int, total_seconds: float) -> float:
    """The speaker's own rate, or the default until there is enough to measure."""
    if total_seconds < 20.0 or total_words < 40:
        return DEFAULT_WPM
    return max(60.0, min(320.0, total_words / (total_seconds / 60.0)))


def next_prompts(
    prompts: list[Prompt],
    done_keys: set[str],
    seconds_needed: float,
    wpm: float = DEFAULT_WPM,
) -> list[Prompt]:
    """The prompts still to read to reach `seconds_needed`, in order.

    Anything already recorded is skipped: re-reading the same sentence adds no new
    coverage and duplicate text skews training toward it.
    """
    if seconds_needed <= 0:
        return []
    chosen: list[Prompt] = []
    accumulated = 0.0
    for prompt in prompts:
        if accumulated >= seconds_needed:
            break
        if prompt.key in done_keys:
            continue
        chosen.append(prompt)
        accumulated += prompt.estimated_seconds(wpm)
    return chosen


def remaining_estimate(
    prompts: list[Prompt],
    done_keys: set[str],
    seconds_needed: float,
    wpm: float = DEFAULT_WPM,
) -> tuple[int, float]:
    """How many more prompts, and how long, to reach `seconds_needed`.

    Counting prompts rather than time would mislead: a slow reader and a fast one
    need very different numbers of sentences for the same amount of audio.
    """
    chosen = next_prompts(prompts, done_keys, seconds_needed, wpm)
    return len(chosen), sum(p.estimated_seconds(wpm) for p in chosen)


def shuffled(prompts: list[Prompt], seed: int = 0) -> list[Prompt]:
    """Deterministic shuffle.

    Recording in file order means the first session is all one author's prose and
    the phonetic balance only arrives if somebody reads all 1132 sentences.
    Shuffling spreads the variety across however much they actually record.
    """
    order = list(prompts)
    random.Random(seed).shuffle(order)
    return order
