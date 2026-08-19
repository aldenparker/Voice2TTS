"""What the app is actually going to do with your voice, worked out in one place.

There are two modes -- speak what you said, or speak a translation of it -- and
three questions that decide whether either works:

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
Nothing else is allowed to reason about languages on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# What the app is doing with what you said.
NORMAL = "normal"          # speak it back as recognised
TRANSLATE = "translate"    # speak a translation, from a downloaded model
RECOGNISER = "recogniser"  # speak a translation, made by Whisper itself


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

    mode: str
    heard: str          # language the recogniser must understand
    spoken: str         # language the voice must actually pronounce
    voice: str
    hops: tuple = ()    # translation steps, empty when not translating
    problems: list[Problem] = field(default_factory=list)

    @property
    def translating(self) -> bool:
        return self.mode in (TRANSLATE, RECOGNISER)

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

        if not self.translating:
            return f"Speaking {language_name(self.spoken)} as recognised"
        route = " → ".join([language_name(self.heard)]
                           + [language_name(hop.target) for hop in self.hops]) \
            if self.hops else \
            f"{language_name(self.heard)} → {language_name(self.spoken)}"
        by = "by the recogniser" if self.mode is RECOGNISER else ""
        return f"Translating {route} {by}".strip()


def build(cfg, voice: str = "") -> SpeechPlan:
    """Work out what will happen, and everything wrong with it.

    `voice` overrides the configured one, so the interface can ask "what would
    happen if I picked this instead?" without writing it to the config first.
    """
    from . import translate, voices

    voice = voice or cfg.tts.voice
    model = cfg.stt.model

    # -- which mode ----------------------------------------------------------
    if cfg.stt.task == "translate":
        mode = RECOGNISER
    elif cfg.translation.enabled:
        mode = TRANSLATE
    else:
        mode = NORMAL

    problems: list[Problem] = []

    # -- what is heard -------------------------------------------------------
    if mode is NORMAL:
        heard = cfg.stt.language if cfg.stt.language != "auto" else "auto"
    else:
        heard = cfg.translation.source

    english_only = model.endswith(".en")
    if english_only and heard not in ("en", "auto"):
        problems.append(Problem(
            f"{model} only understands English, but you are speaking "
            f"{translate.language_name(heard)}. Speech will be transcribed as "
            "English and the result will be wrong.",
            where="Normal → Recognition", serious=True))

    # -- what is spoken ------------------------------------------------------
    hops: tuple = ()
    if mode is RECOGNISER:
        spoken = "en"
        if cfg.translation.target not in ("", "en"):
            problems.append(Problem(
                "The recogniser can only translate into English, so the target "
                f"language ({translate.language_name(cfg.translation.target)}) "
                "is ignored.", where="Translate"))
    elif mode is TRANSLATE:
        route = translate.route(cfg.translation.source, cfg.translation.target,
                                cfg.translation.pivot)
        if route:
            hops = tuple(route)
            spoken = cfg.translation.target
        else:
            # Nothing can translate it, so the text stays as it was recognised.
            # This is the case that produced "English read in a German accent":
            # the voice followed the target while the words did not.
            spoken = heard
            problems.append(Problem(
                "Translation is on, but no model is installed for "
                f"{translate.language_name(cfg.translation.source)} to "
                f"{translate.language_name(cfg.translation.target)}. Your own "
                "words will be spoken untranslated.",
                where="Translate → Download", serious=True))
    else:
        spoken = heard

    if mode is not NORMAL and cfg.translation.source == cfg.translation.target:
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
            problems.append(Problem(
                f"{voice} speaks {translate.language_name(voice_language)}, but "
                f"the text reaching it will be in "
                f"{translate.language_name(spoken)}. It will mispronounce every "
                "word.", where="Normal → Voice", serious=True))

    return SpeechPlan(mode=mode, heard=heard, spoken=spoken, voice=voice,
                      hops=hops, problems=problems)
