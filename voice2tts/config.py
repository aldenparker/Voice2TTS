"""Typed configuration, persisted as TOML in %APPDATA%\\Voice2TTS\\config.toml."""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import tomli_w

from . import DEFAULT_UPDATE_REPO
from .modes import (
    ComputeType,
    LogLevel,
    Mode,
    RecognitionMode,
    SttDevice,
    Theme,
    TranslationMode,
    TriggerMode,
)
from .paths import config_path
from .profiles import PROFILED_FIELDS

log = logging.getLogger(__name__)

# Raise when a change needs a migration in Config.migrate(). Adding a field does
# not: the loader fills unknown-but-missing fields from the dataclass defaults.
#   1 -> 2  update repository became prefilled rather than blank
#   2 -> 3  theme default returned to the native Windows appearance
#   3 -> 4  stt.task and translation.enabled folded into translation.mode
CURRENT_SCHEMA = 4
# The .en models hear English and nothing else, and are noticeably better at it
# than the multilingual ones of the same size. The plain names are multilingual:
# needed to recognise anything but English, and so a prerequisite for
# translating *from* another language.
WHISPER_MODELS = (
    "tiny.en", "base.en", "small.en", "medium.en", "distil-small.en",
    "tiny", "base", "small", "medium", "large-v3",
)


@dataclass(frozen=True)
class Repair:
    """Something the config asked for that the app could not do, and what it did.

    validate() used to write these to the log and nothing else. The log is not
    the app: a user whose translation had been quietly switched off saw a
    settings window that looked exactly like a working one. So repairs are
    returned, and whoever loaded the config is expected to show them.
    """

    what: str            # a whole sentence, in the words the settings window uses
    where: str = ""      # which tab to go and fix it on, when there is one

    def __str__(self) -> str:
        return f"{self.what} ({self.where})" if self.where else self.what


@dataclass(frozen=True)
class Bound:
    """A numeric field and the range outside which it stops working.

    Four fields used to be clamped and fifteen were not, which is how
    `review_timeout_s = 0` came to mean "wait forever" (threading.Event.wait
    treats 0 as no timeout) in a feature whose whole purpose is that nothing
    unreviewed gets spoken. Declaring the range next to the reason means adding
    a field is one line here rather than a paragraph in validate().
    """

    path: str            # dotted, e.g. "vad.threshold"
    low: float
    high: float
    why: str = ""        # what goes wrong outside the range, for the repair note
    integer: bool = False

    def get(self, cfg: Config) -> float:
        obj: Any = cfg
        for part in self.path.split("."):
            obj = getattr(obj, part)
        return float(obj)

    def set(self, cfg: Config, value: float) -> None:
        obj: Any = cfg
        parts = self.path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], int(value) if self.integer else float(value))


# Every numeric field with a range, in one place. The `why` is shown to the user
# when a value had to be pulled back, so it says what would have happened.
BOUNDS: tuple[Bound, ...] = (
    Bound("audio.output_blocksize", 32, 8192, integer=True,
          why="an extreme block size stutters or adds delay"),
    Bound("trigger.preroll_ms", 0, 3000, integer=True,
          why="pre-roll longer than this just adds latency"),
    Bound("trigger.max_utterance_s", 1.0, 300.0,
          why="a cap of zero discards every push-to-talk press as too long"),
    Bound("vad.threshold", 0.05, 0.95,
          why="0 hears the room and 1 hears nothing"),
    Bound("vad.min_speech_ms", 20, 5000, integer=True,
          why="too high and short words never start a capture"),
    Bound("vad.min_silence_ms", 100, 5000, integer=True,
          why="too low cuts you off mid-sentence"),
    Bound("vad.speech_pad_ms", 0, 2000, integer=True,
          why="padding past this overlaps the next utterance"),
    Bound("stt.beam_size", 1, 10, integer=True,
          why="a beam below 1 is not a search"),
    Bound("stt.min_chars", 0, 40, integer=True,
          why="set high, every utterance is classified as noise and nothing is "
              "ever spoken"),
    Bound("tts.length_scale", 0.4, 3.0,
          why="outside this the voice is unintelligible"),
    Bound("tts.noise_scale", 0.0, 2.0, why="outside this the voice distorts"),
    Bound("tts.noise_w_scale", 0.0, 2.0, why="outside this the voice distorts"),
    Bound("tts.volume", 0.0, 2.0, why="above 2 the output clips"),
    Bound("text.review_timeout_s", 1.0, 600.0,
          why="a timeout of zero waits forever, so a review dialog you never "
              "answer holds the app open"),
    Bound("text.history_size", 0, 1000, integer=True,
          why="an unbounded history grows until the app runs out of memory"),
    Bound("translation.beam_size", 1, 10, integer=True,
          why="a beam below 1 is not a search"),
    Bound("updates.interval_hours", 0, 8760, integer=True,
          why="an interval beyond a year never fires"),
)


@dataclass
class OutputTarget:
    """One destination for synthesized speech.

    `match` is a case-insensitive substring of the device name rather than an index,
    because indices shuffle whenever a USB device is plugged in or removed.
    """

    match: str = ""
    gain: float = 1.0
    enabled: bool = True

    @property
    def label(self) -> str:
        return self.match or "(system default)"


@dataclass
class AudioConfig:
    input_match: str = ""            # "" = system default input
    outputs: list[OutputTarget] = field(default_factory=list)
    prefer_wasapi: bool = True
    output_blocksize: int = 480      # frames per callback at device rate (10 ms @ 48k)
    mute_mic_during_playback: bool = True


@dataclass
class TriggerConfig:
    mode: TriggerMode = TriggerMode.PTT
    hotkey: str = "ctrl+alt+v"
    ptt_latch: bool = False          # tap to toggle instead of hold
    preroll_ms: int = 300            # audio kept from before speech onset
    max_utterance_s: float = 30.0    # hard cap so a stuck key cannot run forever
    # Speak whatever is on the clipboard. Empty disables the binding.
    clipboard_hotkey: str = "ctrl+alt+c"
    # Cut off speech in progress. Without this, interrupting means starting a whole
    # new capture just to stop the current one.
    stop_hotkey: str = "ctrl+alt+x"


@dataclass
class VadConfig:
    threshold: float = 0.5           # Silero speech probability
    min_speech_ms: int = 250         # ignore blips shorter than this
    min_silence_ms: int = 600        # trailing silence that ends an utterance
    speech_pad_ms: int = 150         # keep a little audio past the endpoint



@dataclass
class SttConfig:
    # base.en ships inside the installer so a fresh install works offline; the GPU
    # pack upgrades this to small.en when the user opts in.
    model: str = "base.en"
    device: SttDevice = SttDevice.AUTO
    compute_type: ComputeType = ComputeType.AUTO
    # "auto" lets Whisper detect it per utterance, which costs a little accuracy
    # and is only meaningful on a multilingual model.
    language: str = "en"
    # See RecognitionMode. "sentence" is the default because it is the one that
    # costs nothing and never sounds chopped.
    mode: RecognitionMode = RecognitionMode.SENTENCE
    # Whisper's own "translate" task is NOT stored here. It used to be, next to
    # translation.enabled, and the two could contradict each other; both are now
    # translation.mode, and the task faster-whisper is given is derived from it
    # by SpeechPlan.whisper_task.
    beam_size: int = 1
    # Whisper hallucinates stock phrases on near-silent input; drop exact matches.
    drop_phrases: list[str] = field(
        default_factory=lambda: [
            "you", "thank you", "thanks for watching", "thank you for watching",
            "bye", "bye.", ".", "...", "[blank_audio]", "subtitles by the amara.org community",
        ]
    )
    min_chars: int = 2


@dataclass
class TtsConfig:
    voice: str = "en_US-lessac-medium"
    speaker_id: int = 0
    length_scale: float = 1.0        # >1 slower
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    volume: float = 1.0
    normalize_audio: bool = True


@dataclass
class SubstitutionRule:
    """One text rewrite applied between recognition and speech."""

    pattern: str = ""
    replacement: str = ""
    enabled: bool = True
    whole_word: bool = True
    regex: bool = False
    case_sensitive: bool = False


@dataclass
class TextConfig:
    # Two lists, because translation sits between them and they are corrections
    # to different things.
    #
    # SOURCE rules fix what the recogniser misheard -- names, jargon, homophones.
    # They correct the words you actually said, so they must run BEFORE
    # translation, or the translator faithfully carries the mistake across.
    #
    # TARGET rules fix what the voice says badly. Pronunciation is a property of
    # the output language and voice, so applying an English list to German
    # output would be nonsense.
    #
    # With translation off the two run back to back over the same text, which is
    # exactly what one list used to do -- so existing configurations behave
    # identically, and their rules stay in `substitutions`.
    substitutions: list[SubstitutionRule] = field(default_factory=list)
    target_substitutions: list[SubstitutionRule] = field(default_factory=list)
    substitutions_enabled: bool = True

    # Show the transcript for approval before speaking it. Costs latency, but stops
    # a misrecognition reaching a call.
    review_before_speaking: bool = False
    # Seconds to wait for that approval. On timeout the utterance is DISCARDED --
    # the point of review is that nothing unreviewed gets spoken, and a dialog
    # hidden behind a game must not blurt something out minutes later.
    review_timeout_s: float = 30.0
    # Recent utterances kept in memory for the History tab. Never written to disk.
    history_size: int = 50


@dataclass
class TranslationConfig:
    # Off unless asked for. Translation is a second model in the hot path and a
    # download the user has to choose, so it should never appear by surprise.
    #
    # One field, three states, because the two ways of translating are mutually
    # exclusive and used to be stored separately -- see TranslationMode.
    mode: TranslationMode = TranslationMode.OFF

    # What you speak, and what the far end should hear. `source` has to agree
    # with the recognition language: Whisper's bundled base.en hears English and
    # nothing else, so anything other than "en" here needs a multilingual model
    # selected in [stt] first. validate() says so rather than failing silently.
    source: str = "en"
    target: str = "de"

    # Pairs OPUS-MT does not publish directly go through this language, at the
    # cost of a second hop and its compounding errors. English because that is
    # what almost every pair is published against.
    pivot: str = "en"

    # Beam 4 costs ~20 ms more than beam 1 on a long sentence for output that
    # was identical on the sentences measured. Exposed so a slow machine can
    # trade it away.
    beam_size: int = 4


@dataclass
class StudioConfig:
    # "I know what I am doing": proceed with training even though the hardware
    # check objected. Under-spec hardware fails by running out of memory, which
    # costs the time since the last checkpoint and nothing else -- so this is a
    # legitimate choice rather than a footgun, and refusing outright would be
    # wrong. Recorded in diagnostics so a bug report shows it was used.
    ignore_hardware_check: bool = False
    # Minutes of clean speech to aim for before training is offered.
    target_minutes: float = 30.0
    # Speaker consented to their voice being used. Recorded when audio is imported
    # rather than recorded in the app.
    import_attested: bool = False


@dataclass
class ProfileEntry:
    name: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    match_apps: list[str] = field(default_factory=list)


@dataclass
class ProfilesConfig:
    entries: list[ProfileEntry] = field(default_factory=list)
    active: str = ""
    # There was an `auto_switch` here, persisted since 0.4 and read by nothing:
    # no poll loop, no checkbox, no way to set `match_apps` either. A setting
    # that claims a capability the build does not have is worse than a missing
    # one. profiles.find_for_app() and foreground_executable() are the tested
    # pieces it needs, and they stay; the promise does not. Removing the field
    # is safe -- the loader ignores keys it does not recognise.


@dataclass
class UpdateConfig:
    # "owner/name" on GitHub, prefilled with this build's own repository so updates
    # work out of the box. Clearing it disables update checking entirely.
    repo: str = DEFAULT_UPDATE_REPO
    check_on_start: bool = True
    interval_hours: int = 24        # 0 also disables automatic checks
    last_check: float = 0.0         # epoch seconds; 0 means never
    skipped_version: str = ""       # "remind me never about this one"

    # Opt in to pre-releases. Off by default and deliberately not remembered
    # across a reinstall of the stable build: a beta is something you choose,
    # not something you drift into. See updater.check().
    include_prereleases: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    text: TextConfig = field(default_factory=TextConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    profiles: ProfilesConfig = field(default_factory=ProfilesConfig)
    studio: StudioConfig = field(default_factory=StudioConfig)
    updates: UpdateConfig = field(default_factory=UpdateConfig)
    # New configs are written at the current schema; files predating the field are
    # read as 1 in from_dict() so migrate() can bring them forward.
    schema_version: int = CURRENT_SCHEMA
    start_minimized: bool = True
    run_at_login: bool = False
    # native = Windows' own widget appearance, untouched. light/dark repaint the
    # interface for anyone who wants a dark window; native is the default because
    # this is a utility and should look like one.
    theme: Theme = Theme.NATIVE
    log_level: LogLevel = LogLevel.INFO
    first_run_complete: bool = False

    def __post_init__(self) -> None:
        # Deliberately not a field: repairs describe THIS load rather than a
        # setting, so they must never round-trip into the file. Set here so
        # every Config has the attribute, however it was built.
        self.repairs: list[Repair] = []

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build from a possibly-partial dict, ignoring unknown keys.

        Tolerant by design: a config written by an older or newer build should load
        with sane defaults rather than crash the app on startup.
        """

        def section(kind: type[Any], blob: Any) -> Any:
            if not isinstance(blob, dict):
                return kind()
            names = {f.name for f in dataclasses.fields(kind)}
            return kind(**{k: v for k, v in blob.items() if k in names})

        audio = section(AudioConfig, data.get("audio"))
        raw_outputs = (data.get("audio") or {}).get("outputs") or []
        audio.outputs = [
            OutputTarget(
                match=str(o.get("match", "")),
                gain=float(o.get("gain", 1.0)),
                enabled=bool(o.get("enabled", True)),
            )
            for o in raw_outputs
            if isinstance(o, dict)
        ]

        def rules(blob: Any) -> list[SubstitutionRule]:
            return [
                SubstitutionRule(
                    pattern=str(r.get("pattern", "")),
                    replacement=str(r.get("replacement", "")),
                    enabled=bool(r.get("enabled", True)),
                    whole_word=bool(r.get("whole_word", True)),
                    regex=bool(r.get("regex", False)),
                    case_sensitive=bool(r.get("case_sensitive", False)),
                )
                for r in (blob or [])
                if isinstance(r, dict)
            ]

        text_cfg = section(TextConfig, data.get("text"))
        # Both lists need rebuilding, not just the source one: `section` copies
        # whatever was in the file, which for these is a list of plain dicts.
        raw_text = data.get("text") or {}
        text_cfg.substitutions = rules(raw_text.get("substitutions"))
        text_cfg.target_substitutions = rules(raw_text.get("target_substitutions"))

        profiles_cfg = section(ProfilesConfig, data.get("profiles"))
        raw_profiles = (data.get("profiles") or {}).get("entries") or []
        profiles_cfg.entries = [
            ProfileEntry(
                name=str(p.get("name", "")),
                # Filtered here as well as in profiles.from_dict: a profile is
                # written straight into the live config, so an unknown key from
                # a hand-edited file would set an arbitrary attribute on it.
                values={k: v for k, v in (p.get("values") or {}).items()
                        if k in PROFILED_FIELDS},
                match_apps=[str(a).lower() for a in (p.get("match_apps") or [])],
            )
            for p in raw_profiles
            if isinstance(p, dict) and p.get("name")
        ]

        cfg = cls(
            audio=audio,
            trigger=section(TriggerConfig, data.get("trigger")),
            vad=section(VadConfig, data.get("vad")),
            stt=section(SttConfig, data.get("stt")),
            tts=section(TtsConfig, data.get("tts")),
            text=text_cfg,
            translation=section(TranslationConfig, data.get("translation")),
            profiles=profiles_cfg,
            studio=section(StudioConfig, data.get("studio")),
            updates=section(UpdateConfig, data.get("updates")),
            schema_version=_as_int(data.get("schema_version"), 1),
            start_minimized=bool(data.get("start_minimized", True)),
            run_at_login=bool(data.get("run_at_login", False)),
            # The enum fields go in as whatever the file said and come out of
            # validate() as members. This is the one place untrusted values
            # enter, and validate() runs on the next line -- casting here keeps
            # every other module honestly typed.
            theme=cast(Theme, data.get("theme", Theme.NATIVE)),
            log_level=cast(LogLevel, data.get("log_level", LogLevel.INFO)),
            first_run_complete=bool(data.get("first_run_complete", False)),
        )
        # A file written before schema 4 has no translation.mode at all; folding
        # the old pair has to happen before anything reads either one.
        cfg._migrate_translation(data)
        cfg.repairs = cfg.validate()
        return cfg

    def _migrate_translation(self, data: dict[str, Any]) -> None:
        """Fold the pre-schema-4 `stt.task` + `translation.enabled` into one mode.

        They were mutually exclusive and only the settings window knew it, so a
        file can legitimately contain both. The recogniser wins, because that is
        what the old pipeline did: `stt.task` reached Whisper regardless, and a
        model chain then translated its English output a second time.
        """
        if self.schema_version >= CURRENT_SCHEMA:
            return
        raw_stt = data.get("stt")
        stt: dict[str, Any] = raw_stt if isinstance(raw_stt, dict) else {}
        raw_trans = data.get("translation")
        trans: dict[str, Any] = raw_trans if isinstance(raw_trans, dict) else {}
        if str(stt.get("task", "")).strip().lower() == "translate":
            self.translation.mode = TranslationMode.RECOGNISER
        elif bool(trans.get("enabled", False)):
            self.translation.mode = TranslationMode.MODELS
        else:
            self.translation.mode = TranslationMode.OFF

    def migrate(self) -> list[str]:
        """Bring an older config forward. Returns a description of what changed.

        Only for changes the tolerant loader cannot handle on its own. Adding a
        field needs nothing here; changing what an existing value *means* does.
        """
        notes: list[str] = []
        # Before schema 2 the repository was blank by default, so an empty value
        # there meant "never set" rather than "deliberately disabled". Adopt the
        # built-in default; from schema 2 on, empty means the user cleared it and
        # we leave it alone.
        if self.schema_version < 2 and not self.updates.repo:
            self.updates.repo = DEFAULT_UPDATE_REPO
            notes.append(f"update repository set to {DEFAULT_UPDATE_REPO}")

        # 0.5.0 briefly defaulted to a repainted interface. Native widgets suit a
        # utility better, so anything still on the old default moves across; an
        # explicit light or dark choice is left alone.
        if self.schema_version < 3 and self.theme == "system":
            self.theme = Theme.NATIVE
            notes.append("theme reset to the native Windows appearance")

        # 3 -> 4: `stt.task` and `translation.enabled` said the same thing two
        # ways and could disagree. _migrate_translation() has already folded
        # them into `translation.mode` from the raw file; this only reports it,
        # because migrate() runs after the dataclass exists and the old fields
        # are gone by then.
        if self.schema_version < 4:
            notes.append(
                "how translation happens is now one setting "
                f"(translation.mode = {self.translation.mode.value})")
        self.schema_version = CURRENT_SCHEMA
        return notes

    def ensure_usable_output(self) -> bool:
        """Guarantee at least one enabled output; returns True if it had to repair.

        A config with everything disabled produces an app that loads models, listens,
        transcribes, synthesizes -- and makes no sound, reporting only "no usable
        output devices". Falling back to the system default is far friendlier than
        silence, so long as we say so loudly.
        """
        if any(t.enabled for t in self.audio.outputs):
            return False
        fallback = next((t for t in self.audio.outputs if not t.match), None)
        if fallback is None:
            fallback = OutputTarget(match="", gain=1.0)
            self.audio.outputs.append(fallback)
        fallback.enabled = True
        return True

    def validate(self) -> list[Repair]:
        """Repair anything that cannot work, and say what was repaired.

        Returns the repairs rather than logging them. A config that has been
        quietly corrected looks exactly like one that was right all along, and
        the user is left wondering why translation is off. Whoever loads the
        config shows these; see app.py.

        Structural only -- "is this value one this build understands, and is it
        in a range that does something". Whether the *combination* makes sense
        for the languages involved is plan.build()'s job, and is a warning to
        the user rather than a change to their settings.
        """
        repairs: list[Repair] = []

        def as_mode(owner: Any, name: str, kind: type[Mode], fallback: Mode,
                    where: str, what: str) -> None:
            """Coerce one field to its enum, reporting an unrecognised value."""
            raw = getattr(owner, name)
            parsed = kind.parse(raw)
            if parsed is None:
                repairs.append(Repair(
                    f"{what} was set to {raw!r}, which this build does not "
                    f"understand. Using {fallback.value!r} instead "
                    f"(the choices are: {', '.join(kind.values())}).", where))
                parsed = fallback
            setattr(owner, name, parsed)

        as_mode(self.trigger, "mode", TriggerMode, TriggerMode.PTT,
                "Misc -> Triggers", "The way speech is started")
        as_mode(self.stt, "mode", RecognitionMode, RecognitionMode.SENTENCE,
                "Normal -> Recognition", "The recognition mode")
        as_mode(self.stt, "device", SttDevice, SttDevice.AUTO,
                "Normal -> Recognition", "The device Whisper runs on")
        as_mode(self.stt, "compute_type", ComputeType, ComputeType.AUTO,
                "Normal -> Recognition", "The Whisper compute type")
        as_mode(self.translation, "mode", TranslationMode, TranslationMode.OFF,
                "Translate", "Translation")
        as_mode(self, "theme", Theme, Theme.NATIVE, "Misc -> Appearance",
                "The theme")
        as_mode(self, "log_level", LogLevel, LogLevel.INFO, "Misc -> Advanced",
                "The log level")

        # -- numbers that have to be in range ---------------------------------
        for bound in BOUNDS:
            try:
                value = float(bound.get(self))
            except (TypeError, ValueError):
                value = float("nan")
            if value != value:  # NaN: no comparison is true, so clamping cannot fix it
                bound.set(self, bound.low)
                repairs.append(Repair(
                    f"{bound.path} was not a number; using {bound.low:g}.", ""))
                continue
            clamped = min(max(value, bound.low), bound.high)
            if clamped != value:
                bound.set(self, clamped)
                repairs.append(Repair(
                    f"{bound.path} was {value:g}, outside {bound.low:g}-"
                    f"{bound.high:g}: {bound.why}. Using {clamped:g}.", ""))
            else:
                bound.set(self, value)  # normalise ints written as floats
        for t in self.audio.outputs:
            t.gain = min(max(float(t.gain), 0.0), 4.0)

        # -- combinations this build cannot honour ----------------------------
        if not self.stt.model.strip():
            # Not a closed set: any faster-whisper name works, including a
            # HuggingFace id the user pasted, so only emptiness is an error.
            repairs.append(Repair(
                "No recognition model was named, so the bundled base.en is "
                "being used.", "Normal -> Recognition"))
            self.stt.model = "base.en"

        if (self.stt.mode is RecognitionMode.STREAMING
                and self.trigger.mode is TriggerMode.PTT):
            # Streaming needs to know when speech STARTS, which is what the VAD
            # is for. Push-to-talk hands over one finished recording, so there
            # is nothing left to stream. The pair used to be accepted and merely
            # logged, which meant picking streaming under push-to-talk changed
            # nothing at all and said nothing about it.
            repairs.append(Repair(
                "Streaming recognition needs automatic speech detection, and "
                "push-to-talk hands over a finished recording. Waiting for "
                "whole sentences instead -- switch triggering to Automatic or "
                "Both to stream.", "Normal -> Recognition"))
            self.stt.mode = RecognitionMode.SENTENCE

        if (self.translation.mode is TranslationMode.RECOGNISER
                and self.stt.model.endswith(".en")):
            repairs.append(Repair(
                f"{self.stt.model} only hears English, so the recogniser has "
                "nothing to translate from. Translation is off -- choose a "
                "multilingual model to translate.", "Translate"))
            self.translation.mode = TranslationMode.OFF

        self.translation.source = str(self.translation.source).strip().lower()
        self.translation.target = str(self.translation.target).strip().lower()
        self.translation.pivot = str(self.translation.pivot).strip().lower()
        if (self.translation.mode is TranslationMode.MODELS
                and self.translation.source == self.translation.target):
            repairs.append(Repair(
                f"Translation was set to turn {self.translation.source} into "
                f"{self.translation.target}, which does nothing. It is off.",
                "Translate"))
            self.translation.mode = TranslationMode.OFF

        if self.stt.language == "auto" and self.stt.model.endswith(".en"):
            # Detection on an English-only model can only ever answer "English",
            # so this is free to fix -- but silence here once left the settings
            # window showing "Detect" over a model that cannot.
            repairs.append(Repair(
                f"{self.stt.model} recognises English only, so language "
                "detection has nothing to choose between. Set to English.",
                "Normal -> Recognition"))
            self.stt.language = "en"

        self.updates.repo = str(self.updates.repo).strip().strip("/")
        # Tolerate a pasted URL where "owner/name" was expected.
        if "github.com/" in self.updates.repo:
            self.updates.repo = self.updates.repo.split("github.com/", 1)[1]

        for repair in repairs:
            log.warning("config repaired: %s", repair)
        return repairs

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".toml.tmp")
        with tmp.open("wb") as fh:
            tomli_w.dump(self.to_dict(), fh)
        tmp.replace(path)  # atomic, so a crash mid-write cannot corrupt the config
        log.info("saved config to %s", path)
        return path


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        cfg = default_config()
        cfg.save(path)
        return cfg
    try:
        with path.open("rb") as fh:
            cfg = Config.from_dict(tomllib.load(fh))
    except Exception as exc:  # noqa: BLE001 - never let a bad file block startup
        # Running on defaults is right -- an unreadable config must not stop the
        # app -- but the next Save used to overwrite the original without ever
        # having shown the user that it was unreadable. Keep a copy and say so.
        kept = _quarantine(path)
        log.error("could not read %s (%s); using defaults", path, exc)
        cfg = default_config()
        cfg.repairs = [Repair(
            f"Your settings file could not be read ({exc}). The app is running "
            f"on defaults; the original was kept as {kept.name}."
            if kept else
            f"Your settings file could not be read ({exc}). The app is running "
            "on defaults.", "Misc -> Advanced")]
        return cfg

    if cfg.schema_version < CURRENT_SCHEMA:
        previous = cfg.schema_version
        for note in cfg.migrate():
            log.info("config migrated: %s", note)
        log.info("config schema %d -> %d", previous, cfg.schema_version)
        cfg.save(path)

    if cfg.ensure_usable_output():
        cfg.repairs.append(Repair(
            "Every output device was switched off, so nothing could be heard. "
            "The system default has been enabled -- pick your real outputs.",
            "Misc -> Audio"))
        log.warning("every output in %s was disabled; enabled the system default",
                    path)
        cfg.save(path)
    return cfg


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _quarantine(path: Path) -> Path | None:
    """Move an unreadable config aside so the next Save cannot destroy it.

    Whatever is in there is the user's work -- hand-written substitution rules,
    a profile they spent an afternoon on. Overwriting it with defaults because
    of one bad line is not a recovery.
    """
    spare = path.with_suffix(".toml.unreadable")
    try:
        if spare.exists():
            spare.unlink()
        path.replace(spare)
    except OSError as exc:  # best effort; defaults still load either way
        log.warning("could not set %s aside: %s", path, exc)
        return None
    return spare


def default_config() -> Config:
    """Defaults with a best-effort guess at the virtual cable and a local monitor."""
    from . import cable as cable_mod

    cfg = Config()
    found = cable_mod.detect()
    have_cable = found is not None
    if found is not None:
        cfg.audio.outputs.append(
            OutputTarget(match=found.output_name, gain=1.0, enabled=True)
        )
    else:
        # Placeholder so the settings UI has a row to edit once VB-CABLE is installed.
        cfg.audio.outputs.append(OutputTarget(match="CABLE Input", gain=1.0, enabled=False))

    # Local monitor. Normally off, because hearing yourself through speakers feeds
    # the mic -- but with no cable installed it is the only way to hear anything at
    # all, and an app that is silent on first run just looks broken.
    cfg.audio.outputs.append(OutputTarget(match="", gain=0.7, enabled=not have_cable))
    return cfg
