"""Typed configuration, persisted as TOML in %APPDATA%\\Voice2TTS\\config.toml."""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from . import DEFAULT_UPDATE_REPO
from .paths import config_path

log = logging.getLogger(__name__)

MODES = ("ptt", "vad", "both")

# Raise when a change needs a migration in Config.migrate(). Adding a field does
# not: the loader fills unknown-but-missing fields from the dataclass defaults.
#   1 -> 2  update repository became prefilled rather than blank
#   2 -> 3  theme default returned to the native Windows appearance
CURRENT_SCHEMA = 3
WHISPER_MODELS = ("tiny.en", "base.en", "small.en", "medium.en", "distil-small.en", "large-v3")


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
    mode: str = "ptt"                # ptt | vad | both
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

    # Someone speaking quickly never leaves a 600 ms gap, so the rule above never
    # fires and nothing is spoken until they finally stop. Measured on continuous
    # speech: the probability sits at 1.0, only 7% of windows fall below the
    # threshold at all, and NO gap reaches 600 ms -- the utterance ends only when
    # the speech does. Lowering min_silence_ms cannot fix it; at 300 ms that same
    # audio still offers zero cut points.
    #
    # So past `soft_endpoint_s` the silence requirement is relaxed towards
    # `min_silence_floor_ms`, and by `max_segment_s` the utterance is cut at the
    # quietest moment available rather than waiting for a pause that is not coming.
    # Measured on 34 s of continuous speech: first audio at 30 s before (the old
    # hard cap, cutting mid-word), 5 s after. The ceiling is what bounds the
    # wait, so it is set for conversation rather than for transcript quality --
    # 6 s still leaves ~5 s of context per chunk, well above where recognition
    # starts to suffer. Raise both for fewer, longer, better-punctuated chunks.
    soft_endpoint_s: float = 3.0     # 0 disables the whole thing
    max_segment_s: float = 6.0       # hard ceiling for one segment of speech
    min_silence_floor_ms: int = 120  # shortest gap ever accepted as an endpoint


@dataclass
class SttConfig:
    # base.en ships inside the installer so a fresh install works offline; the GPU
    # pack upgrades this to small.en when the user opts in.
    model: str = "base.en"
    device: str = "auto"             # auto | cuda | cpu
    compute_type: str = "auto"       # auto -> float16 on cuda, int8 on cpu
    language: str = "en"
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
    # Switch profile when the foreground application changes. Off by default: a
    # setting that changes itself is surprising unless asked for.
    auto_switch: bool = False


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
    theme: str = "native"            # native | light | dark
    log_level: str = "INFO"
    first_run_complete: bool = False

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build from a possibly-partial dict, ignoring unknown keys.

        Tolerant by design: a config written by an older or newer build should load
        with sane defaults rather than crash the app on startup.
        """

        def section(kind, blob: Any):
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

        text_cfg = section(TextConfig, data.get("text"))
        raw_rules = (data.get("text") or {}).get("substitutions") or []
        text_cfg.substitutions = [
            SubstitutionRule(
                pattern=str(r.get("pattern", "")),
                replacement=str(r.get("replacement", "")),
                enabled=bool(r.get("enabled", True)),
                whole_word=bool(r.get("whole_word", True)),
                regex=bool(r.get("regex", False)),
                case_sensitive=bool(r.get("case_sensitive", False)),
            )
            for r in raw_rules
            if isinstance(r, dict)
        ]

        profiles_cfg = section(ProfilesConfig, data.get("profiles"))
        raw_profiles = (data.get("profiles") or {}).get("entries") or []
        profiles_cfg.entries = [
            ProfileEntry(
                name=str(p.get("name", "")),
                values=dict(p.get("values") or {}),
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
            profiles=profiles_cfg,
            studio=section(StudioConfig, data.get("studio")),
            updates=section(UpdateConfig, data.get("updates")),
            schema_version=int(data.get("schema_version", 1)),
            start_minimized=bool(data.get("start_minimized", True)),
            run_at_login=bool(data.get("run_at_login", False)),
            theme=str(data.get("theme", "native")),
            log_level=str(data.get("log_level", "INFO")),
            first_run_complete=bool(data.get("first_run_complete", False)),
        )
        cfg.validate()
        return cfg

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
            self.theme = "native"
            notes.append("theme reset to the native Windows appearance")
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

    def validate(self) -> None:
        if self.trigger.mode not in MODES:
            log.warning("unknown mode %r, falling back to ptt", self.trigger.mode)
            self.trigger.mode = "ptt"
        self.vad.threshold = min(max(self.vad.threshold, 0.05), 0.95)
        self.tts.length_scale = min(max(self.tts.length_scale, 0.4), 3.0)
        self.tts.volume = min(max(self.tts.volume, 0.0), 2.0)
        self.stt.beam_size = max(1, int(self.stt.beam_size))
        for t in self.audio.outputs:
            t.gain = min(max(t.gain, 0.0), 4.0)

        self.updates.repo = self.updates.repo.strip().strip("/")
        # Tolerate a pasted URL where "owner/name" was expected.
        if "github.com/" in self.updates.repo:
            self.updates.repo = self.updates.repo.split("github.com/", 1)[1]
        self.updates.interval_hours = max(0, int(self.updates.interval_hours))
        if self.theme not in ("native", "light", "dark", "system"):
            log.warning("unknown theme %r, using native", self.theme)
            self.theme = "native"

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
        log.error("could not read %s (%s); using defaults", path, exc)
        return default_config()

    if cfg.schema_version < CURRENT_SCHEMA:
        previous = cfg.schema_version
        for note in cfg.migrate():
            log.info("config migrated: %s", note)
        log.info("config schema %d -> %d", previous, cfg.schema_version)
        cfg.save(path)

    if cfg.ensure_usable_output():
        log.warning(
            "every output in %s was disabled; enabled the system default device so "
            "the app can be heard. Pick your real outputs in Settings -> Audio.",
            path,
        )
        cfg.save(path)
    return cfg


def default_config() -> Config:
    """Defaults with a best-effort guess at the virtual cable and a local monitor."""
    from . import cable as cable_mod

    cfg = Config()
    found = cable_mod.detect()
    have_cable = found is not None
    if have_cable:
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
