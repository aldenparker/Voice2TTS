"""Every mode the app can be in, as a type rather than a bare string.

Three bugs in a row came from modes being strings. The worst was a genuine
collision: `plan.TRANSLATE` and `stt.task == "translate"` were the same literal
with *opposite* meanings -- one said "a downloaded model will translate this",
the other said "Whisper will, so no model is involved". Anything comparing the
two got the answer exactly backwards.

The others came from the same value set being written out in four places
(`config.MODES`, and again in the settings window, the tray menu and the
wizard), so adding a mode meant finding all four. Nobody ever did.

So every closed set of values lives here, once, as a `StrEnum`. That buys three
things:

    - `TriggerMode.PTT == "ptt"` is still true and TOML still round-trips, so
      nothing about the config file format changes.
    - `match` over a mode with `assert_never` in the default arm makes adding a
      member a *type* error at every site that does not handle it. The checker
      finds the four places; you do not have to.
    - `parse()` gives one answer to "is this string one of ours", so a
      hand-edited config is repaired the same way everywhere instead of
      crashing somewhere far downstream.

The values are the strings that were already being written to config.toml. They
are load-bearing -- changing one silently invalidates every config in the wild.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class Mode(StrEnum):
    """A closed set of config values that knows how to check itself.

    No members, so it can be subclassed -- which is the whole point.
    """

    @classmethod
    def parse(cls, value: object) -> Self | None:
        """The matching member, or None if this is not one of our values.

        Case- and space-insensitive, because these come out of a file a human
        may have edited.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            return None
        wanted = value.strip().lower()
        for member in cls:
            if member.value == wanted:
                return member
        return None

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Every legal value, in declaration order -- for combo boxes and errors."""
        return tuple(m.value for m in cls)


class TriggerMode(Mode):
    """How the app decides you have started talking."""

    PTT = "ptt"          # hold (or latch) a key
    VAD = "vad"          # listen continuously and detect speech
    BOTH = "both"        # either one starts a capture


class RecognitionMode(Mode):
    """When recognition happens. A genuine choice, not a tuning knob.

    0.6.1 tried to have both at once by cutting continuous speech into segments.
    It cut mid-thought and the app talked over itself; it was reverted. The two
    behaviours want different things, so they are offered as different modes.
    """

    # Wait for the speaker to pause, then recognise the whole thing. Best words,
    # whole sentences, natural intonation. The delay is however long they talk.
    SENTENCE = "sentence"
    # Recognise while they are still speaking and say each phrase as it settles.
    # Much lower typical delay, at the cost of ~0.5x realtime CPU for as long as
    # anyone is talking, and speech delivered a phrase at a time.
    STREAMING = "streaming"


class TranslationMode(Mode):
    """Whether the far end hears your words or a translation of them.

    This used to be two fields -- `stt.task` and `translation.enabled` -- that
    were mutually exclusive but never checked against each other outside the
    settings window. A hand-edited config or a profile could set both, and then
    Whisper translated to English *and* the model chain translated that again.
    The plan reported English; the app spoke German.

    One field cannot contradict itself. That combination is now unwritable.
    """

    OFF = "off"                 # speak what was said
    MODELS = "models"           # a downloaded OPUS-MT chain translates it
    RECOGNISER = "recogniser"   # Whisper translates it, to English only

    @property
    def translating(self) -> bool:
        return self is not TranslationMode.OFF


class WhisperTask(StrEnum):
    """What we ask faster-whisper for. Derived from TranslationMode, never stored.

    Storing it was half the problem: two fields, one truth. It is computed by
    `SpeechPlan.whisper_task` and passed to the engine.
    """

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class AddonState(Mode):
    """What an optional download is, from the settings window's side.

    Three states, not two. It was a boolean, and a pack that was unpacked but
    would not load had to be one or the other: reported as MISSING it offered a
    Download button on something already downloaded, which was the one action
    that could not help; reported as present it offered only Remove.
    """

    MISSING = "missing"
    READY = "ready"
    BROKEN = "broken"   # here, but known not to work


class Theme(Mode):
    """Widget appearance. `native` leaves Windows' own widgets untouched."""

    NATIVE = "native"
    LIGHT = "light"
    DARK = "dark"


class SttDevice(Mode):
    """Where Whisper runs. `auto` picks cuda when it is genuinely usable."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class ComputeType(Mode):
    """CTranslate2 quantisation. `auto` is float16 on cuda, int8 on cpu.

    These names are CTranslate2's, not ours: an unrecognised one reaches
    `WhisperModel(...)` and fails there with a message about nothing the user
    typed. Checking here means the error names the setting.
    """

    AUTO = "auto"
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class LogLevel(Mode):
    """Verbosity, as the names logging itself uses."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    @classmethod
    def parse(cls, value: object) -> Self | None:
        # logging's names are upper case, so the base class's lower-casing
        # would reject every one of them.
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            return None
        wanted = value.strip().upper()
        return next((m for m in cls if m.value == wanted), None)
