"""What the app is actually going to do with your voice, worked out in one place.

There are three ways to run -- speak what you said, speak a translation made by
a downloaded model, or speak a translation made by the recogniser itself -- and
three questions that decide whether any of them works:

    what language will be HEARD      the recogniser has to understand it
    what language will be SPOKEN     the voice has to pronounce it
    can this build speak that at all the phonemizer has to be present

Every one of those used to be answered separately, in the place that happened to
need it, and the answers disagreed. The interface warned that a Japanese voice
"is not an English voice" while translating English INTO Japanese -- because
that check compared the voice against the recognition model and knew nothing
about translation. Meanwhile the pipeline decided the same question a third way
and the diagnostics a fourth.

So: one function works out the whole picture, and everything else reads it.
Nothing else is allowed to reason about languages on its own. The settings
window, the log the pipeline writes on start, and the diagnostics report all
render the same SpeechPlan, which is why they can no longer contradict one
another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

from .modes import RecognitionMode, TranslationMode, TriggerMode, WhisperTask

if TYPE_CHECKING:
    from .config import Config
    from .translate import Pair

log = logging.getLogger(__name__)

# Recognition models that can technically translate and should not be asked to.
SMALL_MODELS = ("tiny", "base")


@dataclass(frozen=True)
class Problem:
    """Something that will make the output wrong, and where to fix it."""

    text: str
    where: str = ""
    # A problem that changes what the far end HEARS, rather than merely making
    # it worse. Those are worth an error; the rest are worth a note.
    serious: bool = False

    def __str__(self) -> str:
        return f"{self.text} ({self.where})" if self.where else self.text


@dataclass(frozen=True)
class SpeechPlan:
    """The whole picture: what goes in, what comes out, and what is wrong."""

    mode: TranslationMode
    heard: str          # language the recogniser must understand
    spoken: str         # language the voice must actually pronounce
    voice: str
    # The target that was ASKED for, which is not always the one that will come
    # out: with translation on and no model for the pair, `spoken` falls back to
    # `heard`, and a summary built from that alone reads "English to English"
    # while the user is looking at a form that says German.
    requested: str = ""
    trigger: TriggerMode = TriggerMode.PTT
    recognition: RecognitionMode = RecognitionMode.SENTENCE
    hops: tuple[Pair, ...] = ()   # translation steps, empty when not translating
    problems: list[Problem] = field(default_factory=list)

    @property
    def translating(self) -> bool:
        return self.mode.translating

    @property
    def whisper_task(self) -> WhisperTask:
        """What faster-whisper is asked for.

        Derived, never stored. It used to live in the config next to
        `translation.enabled`, where the two could disagree and did: a config
        with both set had Whisper translate to English and then a model chain
        translate that again, while every report said the output was English.
        """
        match self.mode:
            case TranslationMode.RECOGNISER:
                return WhisperTask.TRANSLATE
            case TranslationMode.OFF | TranslationMode.MODELS:
                return WhisperTask.TRANSCRIBE
        assert_never(self.mode)

    @property
    def needs_chain(self) -> bool:
        """Whether a downloaded translation model has to be loaded."""
        return self.mode is TranslationMode.MODELS

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def serious(self) -> list[Problem]:
        return [p for p in self.problems if p.serious]

    @property
    def summary(self) -> str:
        """One line describing the plan, in the terms a user thinks in."""
        from .translate import language_name

        match self.mode:
            case TranslationMode.OFF:
                return f"Speaking {language_name(self.spoken)} as recognised"
            case TranslationMode.RECOGNISER:
                return (f"Translating {language_name(self.heard)} to English "
                        "by the recogniser")
            case TranslationMode.MODELS:
                if not self.hops:
                    return (f"No model for {language_name(self.heard)} to "
                            f"{language_name(self.requested)}; speaking as "
                            "recognised")
                route = " -> ".join([language_name(self.heard)]
                                    + [language_name(hop.target)
                                       for hop in self.hops])
                return f"Translating {route}"
        assert_never(self.mode)

    def describe(self) -> list[str]:
        """The plan as lines for a log or a bug report.

        Diagnostics used to re-derive all of this and got it wrong in exactly
        the way the settings window used to: it reported a voice/model language
        mismatch without ever mentioning that translation was on.
        """
        lines = [
            f"mode          : {self.mode.value}",
            f"triggering    : {self.trigger.value}",
            f"recognition   : {self.recognition.value} "
            f"({self.whisper_task.value})",
            f"heard         : {self.heard}",
            f"spoken        : {self.spoken}",
            f"voice         : {self.voice}",
            f"summary       : {self.summary}",
        ]
        if self.hops:
            lines.append("chain         : "
                         + " via ".join(hop.label for hop in self.hops))
        for problem in self.problems:
            lines.append(
                f"{'PROBLEM' if problem.serious else 'note':<14}: {problem}")
        return lines


def build(cfg: Config, voice: str = "") -> SpeechPlan:
    """Work out what will happen, and everything wrong with it.

    `voice` overrides the configured one, so the interface can ask "what would
    happen if I picked this instead?" without writing it to the config first.
    """
    from . import translate, voices

    voice = voice or cfg.tts.voice
    model = cfg.stt.model
    mode = cfg.translation.mode
    problems: list[Problem] = []

    # -- what is heard -------------------------------------------------------
    heard = (cfg.stt.language if mode is TranslationMode.OFF
             else cfg.translation.source)

    english_only = model.endswith(".en")
    if english_only and heard not in ("en", "auto"):
        problems.append(Problem(
            f"{model} only understands English, but you are speaking "
            f"{translate.language_name(heard)}. Speech will be transcribed as "
            "English and the result will be wrong.",
            where="Normal -> Recognition", serious=True))

    # -- what is spoken ------------------------------------------------------
    hops: tuple[Pair, ...] = ()
    match mode:
        case TranslationMode.RECOGNISER:
            spoken = "en"
            if model in SMALL_MODELS:
                # Measured on synthesized German: base returns "can you be nice
                # to me?" for "kannst du mich hoeren?". small gets it right, at
                # about 1.1 s per utterance on this CPU against base's 0.36 s.
                problems.append(Problem(
                    f"{model} is too small to translate well -- it produces "
                    "fluent sentences that do not mean what you said. small or "
                    "better is worth the extra second.",
                    where="Normal -> Recognition"))
            if cfg.translation.target not in ("", "en"):
                problems.append(Problem(
                    "The recogniser can only translate into English, so the "
                    f"target language "
                    f"({translate.language_name(cfg.translation.target)}) is "
                    "ignored. Switch to downloaded models to reach it.",
                    where="Translate"))
        case TranslationMode.MODELS:
            route = translate.route(cfg.translation.source,
                                    cfg.translation.target,
                                    cfg.translation.pivot)
            if route:
                hops = tuple(route)
                spoken = cfg.translation.target
                if len(hops) > 1:
                    problems.append(Problem(
                        "No direct model for this pair, so it goes through "
                        f"{translate.language_name(hops[0].target)}. Each hop "
                        "compounds the errors of the one before it.",
                        where="Translate -> Download"))
            else:
                # Nothing can translate it, so the text stays as it was
                # recognised. This is the case that produced "English read in a
                # German accent": the voice followed the target while the words
                # did not.
                spoken = heard
                problems.append(Problem(
                    "No model is installed for "
                    f"{translate.language_name(cfg.translation.source)} to "
                    f"{translate.language_name(cfg.translation.target)}, so "
                    "your own words will be spoken untranslated.",
                    where="Translate -> Download", serious=True))
        case TranslationMode.OFF:
            spoken = heard
        case _:
            assert_never(mode)

    if mode.translating and cfg.translation.source == cfg.translation.target:
        problems.append(Problem(
            "The language you speak and the language they hear are the same, "
            "so translation does nothing.", where="Translate"))

    # -- can the voice say it ------------------------------------------------
    unspeakable = voices.missing_phonemizer(voice)
    if unspeakable:
        problems.append(Problem(unspeakable, where="Add-ons", serious=True))
    else:
        voice_language = voices.voice_language(voice)
        # An unknown language says nothing: a voice built in the Studio carries
        # no language at all, and guessing produced a wrong warning.
        if voice_language and spoken not in ("auto", "") \
                and voice_language != spoken:
            fix = next((key for key in voices.installed_keys()
                        if voices.voice_language(key) == spoken), None)
            problems.append(Problem(
                f"{voice} speaks {translate.language_name(voice_language)}, but "
                f"the text reaching it will be in "
                f"{translate.language_name(spoken)}. It will mispronounce every "
                "word"
                + (f" (use {fix})." if fix else
                   "; the Voice library tab can fetch one."),
                where="Translate -> Voice" if mode.translating
                else "Normal -> Voice",
                serious=True))

    return SpeechPlan(mode=mode, heard=heard, spoken=spoken, voice=voice,
                      requested=cfg.translation.target,
                      trigger=cfg.trigger.mode, recognition=cfg.stt.mode,
                      hops=hops, problems=problems)
