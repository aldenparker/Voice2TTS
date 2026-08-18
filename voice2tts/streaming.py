"""Recognising while someone is still speaking, rather than after they stop.

Sentence mode waits for a pause and recognises the whole utterance. That is
right when what matters is getting the words correct, and it is what the app has
always done. This is the other choice: re-transcribe a growing buffer every
second or so, and release whatever two consecutive passes agree on. Agreement is
the signal that Whisper has stopped changing its mind about the early part of
the buffer, so that text can be spoken without risk of contradicting it later --
you cannot un-speak a word.

That rule is LocalAgreement-2, and `agree()` below is all of it.

WHAT THE SPIKE FOUND (spike/09_streaming.py, on 28.5 s of varied speech)

- **The GPU does not help.** 0.51x realtime against the CPU's 0.49x. Decoding a
  long transcript is a sequence of small dependent steps, so it is latency-bound
  and there is nothing for a GPU to bite on. Streaming is not a GPU feature.
- **Cost grows with buffer length, and runs away.** ~35 ms per second of buffer
  on CUDA, ~17 ms + 269 ms fixed on CPU, so a pass exceeds a 1 s interval at
  roughly 29 s (GPU) or 43 s (CPU) of unbroken speech. Past that the buffer
  grows faster than it can be read and the lag comes back worse than before.
  **Trimming is therefore mandatory, not an optimisation**, and `TRIM_AFTER_S`
  keeps the buffer well below the crossover.
- **The win is real but narrower than it sounds.** Mean commit lag 2.2 s. Not a
  removal of the delay -- a smoothing of it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .vad import SAMPLE_RATE

log = logging.getLogger(__name__)

# How often to re-read the buffer. A word needs two agreeing passes before it
# can be spoken, so this interval is paid twice on every phrase -- it is the
# single biggest lever on how far behind the speaker we run.
#
# Measured end to end: at 1.0 s the text was ready 0.9 s after the sentence
# ended; at 0.5 s, 0.5 s after. The cost is 27% of one core rather than 13%,
# and the adaptive stretch below protects any machine that cannot afford it.
DEFAULT_INTERVAL_S = 0.5

# The longest buffer ever kept, whatever the timings say. A hard ceiling on top
# of the adaptive target below, so a machine that reads a wildly optimistic cost
# once cannot decide to keep a minute of audio.
TRIM_AFTER_S = 15.0

# Aim to finish each pass in this fraction of the interval. The slack absorbs
# the variation between passes -- decode time depends on how much was SAID, not
# only on how long the buffer is, so identical buffers differ by a third.
COST_TARGET = 0.6

# Cost per second of buffer is measured, not assumed, because it varies far more
# than any constant could cover: base.en on this machine runs ~35 ms per second
# of buffer, small.en ~100 ms. A fixed trim point that suits one starves the
# other. Seeded pessimistically so the first few passes are cautious.
INITIAL_COST_PER_S = 0.10

# Never trim away the audio a pass is still working on: the tail is where the
# unstable text lives, and cutting into it would throw away context Whisper
# needs to settle the words just before it.
KEEP_TAIL_S = 5.0

# When passes get slower the interval stretches rather than the passes piling
# up. Updates arrive less often, which is a graceful loss; overlapping passes
# are not a loss but a collapse.
MAX_INTERVAL_S = 2.5

# Past that ceiling, stretching further would make streaming slower than just
# waiting for the pause -- so it stops pretending. One slow pass is a hiccup (a
# page fault, another process); several in a row is a verdict about the machine.
SLOW_PASSES_BEFORE_GIVING_UP = 3

# Sentence-ending punctuation. Speaking word by word sounds robotic and loses
# the intonation contour, so stable text is held to a phrase boundary.
_ENDINGS = ".!?"
_PAUSES = ",;:"


def agree(previous: list[str], current: list[str]) -> list[str]:
    """The words two consecutive passes both start with. LocalAgreement-2.

    Pure, and the whole of the rule -- which is why it is a function rather than
    a few lines inside a loop. Everything else in this module is bookkeeping.
    """
    settled: list[str] = []
    for before, now in zip(previous, current, strict=False):
        if before != now:
            break
        settled.append(now)
    return settled


def phrase_boundary(words: list[str]) -> int:
    """How many words can be spoken now, ending on a natural break.

    Returns 0 when there is no boundary yet, so the caller holds the words. A
    sentence read out in fragments loses its intonation and sounds like a robot;
    waiting for a break costs a little latency and buys back the prosody.
    """
    for index in range(len(words) - 1, -1, -1):
        if words[index].rstrip('"\')]').endswith(tuple(_ENDINGS)):
            return index + 1
    return 0


def soft_boundary(words: list[str]) -> int:
    """A comma or clause break, used when a sentence runs long.

    Someone who talks for twenty seconds without a full stop should still hear
    something come out, so a clause break counts once enough words have piled up.
    """
    for index in range(len(words) - 1, -1, -1):
        if words[index].rstrip('"\')]').endswith(tuple(_PAUSES)):
            return index + 1
    return 0


@dataclass
class StreamResult:
    """What one pass produced."""

    speakable: str = ""          # stable text, ended on a boundary, ready to say
    committed: list[str] = field(default_factory=list)   # newly stable words
    unstable: str = ""           # still moving; shown, never spoken
    elapsed: float = 0.0
    trimmed: float = 0.0         # seconds of audio dropped this pass
    fell_behind: bool = False    # this machine cannot keep up


class StreamingRecognizer:
    """A growing audio buffer, re-read on a timer, releasing settled text.

    Not thread-safe: feed it and poll it from the same thread. The pipeline runs
    it on the segmenter thread, which is the only one that has the audio.
    """

    def __init__(self, engine, interval_s: float = DEFAULT_INTERVAL_S,
                 max_sentence_words: int = 25,
                 max_interval_s: float = MAX_INTERVAL_S,
                 sentences_only: bool = False):
        self.engine = engine
        self.interval_s = max(0.25, interval_s)
        self.max_interval_s = max(self.interval_s, max_interval_s)
        self.max_sentence_words = max_sentence_words
        # Translation is done a sentence at a time, and a translator handed half
        # a clause produces something fluent and wrong rather than an error. So
        # when translating, wait for a real full stop however long it takes --
        # the alternative is worse than the delay.
        self.sentences_only = sentences_only
        # Stretches towards max_interval_s on a slower model or machine.
        self._interval_now = self.interval_s

        self._audio: list[np.ndarray] = []
        self._samples = 0
        self._previous: list[str] = []      # last pass, for the agreement rule
        self._settled: list[str] = []       # agreed but not yet spoken
        self._spoken_words = 0              # how much of _settled has gone out
        self._last_pass = 0.0
        self._samples_at_last_pass = -1
        self._slow_passes = 0
        self._cost_per_s = INITIAL_COST_PER_S
        self.behind = False

    # -- feeding -------------------------------------------------------------

    def feed(self, window: np.ndarray) -> None:
        self._audio.append(window)
        self._samples += len(window)

    @property
    def buffered_s(self) -> float:
        return self._samples / SAMPLE_RATE

    @property
    def target_buffer_s(self) -> float:
        """How much audio can be re-read inside the interval, as measured.

        This is what keeps the mode working across a fourfold difference in
        model cost. Clamped below so there is always enough context for the
        recogniser to settle the recent words, and above by TRIM_AFTER_S.
        """
        affordable = (self._interval_now * COST_TARGET) / max(self._cost_per_s, 1e-3)
        return max(KEEP_TAIL_S + 2.0, min(TRIM_AFTER_S, affordable))

    @property
    def due(self) -> bool:
        """Whether it is time for another pass, and there is anything new to read.

        The second half matters more than it looks. Re-reading a buffer that has
        not changed cannot produce a different answer, and if audio stops
        arriving mid-utterance -- a microphone dropping out is the usual way --
        this otherwise re-transcribes the same seconds forever, at whatever the
        chosen model costs. On `medium.en` that is a GPU pinned at full load
        with nobody speaking.
        """
        if self._samples <= self._samples_at_last_pass:
            return False
        return (time.monotonic() - self._last_pass) >= self._interval_now

    def reset(self) -> None:
        self._audio.clear()
        self._samples = 0
        self._previous.clear()
        self._settled.clear()
        self._spoken_words = 0
        self._samples_at_last_pass = -1
        self._slow_passes = 0
        self.behind = False
        # The measured cost is a property of the model and the machine, not of
        # this utterance, so it deliberately survives a reset.

    # -- reading -------------------------------------------------------------

    def poll(self) -> StreamResult | None:
        """Re-read the buffer if it is time. None when there is nothing to do."""
        if not self._audio or not self.due:
            return None
        self._last_pass = time.monotonic()
        self._samples_at_last_pass = self._samples

        audio = np.concatenate(self._audio)
        started = time.perf_counter()
        text, timed = self.engine.transcribe_timed(audio)
        elapsed = time.perf_counter() - started

        result = StreamResult(elapsed=elapsed)

        # Smooth rather than jump: one slow pass (a page fault, another process)
        # should nudge the target, not halve it.
        measured = elapsed / max(self.buffered_s, 0.5)
        self._cost_per_s = 0.7 * self._cost_per_s + 0.3 * measured

        # Adapt before giving up: spread the passes out so they stop overlapping.
        # Updates then arrive less often, which is a worse experience but still
        # a working one.
        self._interval_now = min(self.max_interval_s,
                                 max(self.interval_s, elapsed / COST_TARGET))

        # Giving up is measured, not assumed. Past the ceiling there is no more
        # room to stretch, and streaming would be delivering text less often
        # than simply waiting for the speaker to pause.
        if elapsed > self.max_interval_s:
            self._slow_passes += 1
            if self._slow_passes >= SLOW_PASSES_BEFORE_GIVING_UP:
                log.warning("streaming cannot keep up (%d passes over %.1fs); "
                            "falling back", self._slow_passes,
                            self.max_interval_s)
                self.behind = True
                result.fell_behind = True
        else:
            self._slow_passes = 0

        words = text.split()
        settled = agree(self._previous, words)
        self._previous = words

        if len(settled) > len(self._settled):
            result.committed = settled[len(self._settled):]
            self._settled = settled

        result.unstable = " ".join(words[len(self._settled):])
        result.speakable = self._take_speakable()
        result.trimmed = self._trim(timed)
        return result

    def finish(self) -> str:
        """Everything still held, for when the speaker stops.

        Reads the buffer one final time first. Using the last scheduled pass
        instead loses whatever was said after it -- up to a full interval of
        speech, which is how "Let me know whether..." came out as "know
        whether...". This last read also has the whole utterance for context,
        so it is the most accurate one of the lot.

        The tail never gets a second agreeing pass, because there is no later
        pass to agree with. At the end of speech that is fine: nothing more is
        coming that could change it.
        """
        remaining = self._previous[self._spoken_words:]
        if self._audio:
            try:
                text, _ = self.engine.transcribe_timed(
                    np.concatenate(self._audio))
                words = text.split()
                # Only trust it if it still covers what has already gone out;
                # a shorter read would mean re-speaking or skipping words.
                if len(words) >= self._spoken_words:
                    remaining = words[self._spoken_words:]
            except Exception:
                log.exception("final streaming pass failed; using the last one")
        text = " ".join(remaining).strip()
        self.reset()
        return text

    # -- internals -----------------------------------------------------------

    def _take_speakable(self) -> str:
        """Stable text up to a natural break, or "" while there is none."""
        pending = self._settled[self._spoken_words:]
        if not pending:
            return ""

        take = phrase_boundary(pending)
        if not take and self.sentences_only:
            return ""
        if not take and len(pending) >= self.max_sentence_words:
            # A long sentence with no end in sight: a clause break will do, and
            # failing that just say it, because holding forever is worse.
            take = soft_boundary(pending) or len(pending)
        if not take:
            return ""

        self._spoken_words += take
        return " ".join(pending[:take])

    def _trim(self, timed) -> float:
        """Drop audio that has been committed, keeping recent context.

        Cuts only on a boundary the recogniser itself drew: an arbitrary cut
        lands mid-word and the next pass reads a fragment. Returns the seconds
        dropped, for the caller's bookkeeping.
        """
        if self.buffered_s <= self.target_buffer_s or not timed:
            return 0.0

        limit = self.buffered_s - KEEP_TAIL_S
        cut_at = 0.0
        words_before = 0
        for piece in timed:
            if piece.end > limit:
                break
            cut_at = piece.end
            words_before += len(piece.text.split())

        # Only drop what has actually been spoken. Trimming audio whose text is
        # still unstable would delete the evidence for words we have not said.
        if cut_at <= 0 or words_before > self._spoken_words:
            return 0.0

        keep_from = int(cut_at * SAMPLE_RATE)
        audio = np.concatenate(self._audio)[keep_from:]
        dropped = self._samples - len(audio)
        self._audio = [audio]
        self._samples = len(audio)
        # Trimming shrinks the buffer; without this the "is there new audio"
        # test above would never be true again.
        self._samples_at_last_pass -= dropped

        # The transcript restarts from the new beginning of the buffer, so the
        # counters that index into it have to move back by the same amount.
        self._previous = self._previous[words_before:]
        self._settled = self._settled[words_before:]
        self._spoken_words -= words_before
        log.debug("trimmed %.1fs (%d words) from the streaming buffer",
                  cut_at, words_before)
        return cut_at
