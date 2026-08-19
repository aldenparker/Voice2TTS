"""The orchestrator: mic -> VAD/PTT -> Whisper -> Piper -> output devices.

Threading model, one job each so a slow stage never stalls the one before it:

  PortAudio callback  capture -> 16 kHz windows onto a queue
  segmenter thread    windows -> utterances (Silero endpointing or PTT gating)
  worker thread       utterance -> transcript -> synthesis -> output sink
  PortAudio callbacks playback, one per output device

Only the worker thread touches the STT and TTS engines, so neither needs a lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from . import devices
from . import plan as plan_mod
from .capture import MicCapture
from .config import Config
from .devices import resolve_input
from .hotkey import HotkeyManager
from .modes import RecognitionMode, TriggerMode
from .output import OutputSink
from .streaming import StreamingRecognizer
from .stt import Heard, WhisperEngine
from .substitutions import Rule, Substituter
from .translate import Chain
from .tts import PiperEngine, UnspeakableVoice, VoiceSubstituted
from .vad import SAMPLE_RATE, WINDOW, VadSegmenter

log = logging.getLogger(__name__)

_STOP = object()  # queue sentinel

# Phrases waiting to be spoken before streaming admits it is behind.
# Two is normal jitter; three means the gap is not closing.
_BACKLOG_WARN_DEPTH = 3

# How long the pipeline may sit in a working state before it is treated as
# stalled. Transcribing and speaking one utterance is seconds; a minute is not
# slow, it is stuck.
STALL_SECONDS = 60.0

# Consecutive VAD errors before we stop trying and fall back to push-to-talk.
_MAX_VAD_FAILURES = 5

# How often to retry a microphone that has gone away. Long enough not to thrash
# PortAudio, short enough that replugging feels like it just works.
_RECOVERY_INTERVAL_S = 3.0

# How long a reopened microphone has to keep working before it counts as
# recovered. PortAudio reports failure from its callback, not from start().
_RECONNECT_SETTLE_S = 0.75

# Retries back off to here. A microphone that is gone for good must not keep
# the machine busy: every attempt re-enumerates the audio devices, which is not
# cheap, and at a flat three seconds that ran for as long as the app was open.
_RECOVERY_MAX_INTERVAL_S = 60.0


@dataclass
class HistoryEntry:
    """One utterance, as heard and as spoken."""

    heard: str
    spoken: str
    at: float
    source: str = "speech"   # speech | clipboard | typed

    @property
    def edited(self) -> bool:
        return self.heard != self.spoken

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.at))


class State(Enum):
    STOPPED = "stopped"
    LOADING = "loading"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    REVIEWING = "reviewing"
    SPEAKING = "speaking"


# What may follow what. The state is set from eighteen places across three
# threads, and nothing said which orders were meant to be possible -- so a race
# that took the app from SPEAKING straight back to LISTENING, skipping the
# drain, looked exactly like normal operation in the log.
#
# A transition outside this table is a bug in the pipeline, not in the user's
# settings, so it is logged loudly and then allowed: refusing it would wedge
# whichever thread got there, which is a worse outcome than a wrong tray icon.
_ALLOWED: dict[State, frozenset[State]] = {
    State.STOPPED: frozenset({State.LOADING}),
    # Loading can fail back to STOPPED, or finish into IDLE.
    State.LOADING: frozenset({State.IDLE, State.STOPPED}),
    State.IDLE: frozenset({State.LISTENING, State.THINKING, State.REVIEWING,
                           State.SPEAKING, State.STOPPED}),
    # Speech was detected: it either produces something to work on, or turns
    # out to have been noise and drops back.
    State.LISTENING: frozenset({State.THINKING, State.IDLE, State.SPEAKING,
                                State.STOPPED}),
    # Recognised text either goes for review, goes straight out, or was noise.
    State.THINKING: frozenset({State.REVIEWING, State.SPEAKING, State.IDLE,
                               State.LISTENING, State.STOPPED}),
    State.REVIEWING: frozenset({State.SPEAKING, State.IDLE, State.STOPPED}),
    # Streaming feeds the next phrase in while the last is still going out, so
    # SPEAKING may follow itself through THINKING or LISTENING.
    State.SPEAKING: frozenset({State.IDLE, State.THINKING, State.LISTENING,
                               State.STOPPED}),
}


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        on_state: Callable[[State], None] | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ):
        self.cfg = cfg
        self._on_state = on_state
        self._on_event = on_event

        self.state = State.STOPPED
        self.stt: WhisperEngine | None = None
        self.tts: PiperEngine | None = None
        self.sink: OutputSink | None = None
        self.capture: MicCapture | None = None
        self.hotkey: HotkeyManager | None = None
        self.segmenter: VadSegmenter | None = None
        self.substituter = Substituter()          # what was misheard
        self.target_substituter = Substituter()   # what is said badly
        self.translator: Chain | None = None      # a translate.Chain, when on
        self.streamer: StreamingRecognizer | None = None
        # What this run is doing: which languages, which way of translating,
        # what is wrong with it. Rebuilt on start() and whenever a setting that
        # could change it is applied, and read by everything that used to work
        # the answer out for itself. Available before start() so the interface
        # can ask about a config that is not running yet.
        self.plan = plan_mod.build(cfg)
        # Windows on their way to the streaming thread. Bounded: if that
        # thread cannot keep up, dropping the oldest window is far better
        # than growing without limit, and falling behind is already
        # detected and reported by the recogniser itself.
        self._stream_windows: queue.Queue[Any] = queue.Queue(maxsize=400)
        self._stream_active = threading.Event()
        self._streaming_degraded = False
        self._backlog_reported = False
        # Snapshotted at start(), never read live: the segmenter thread
        # decides per window whether to forward audio to the streaming
        # thread, and that thread only exists if it was spawned. Reading
        # the config live would let a settings change point audio at a
        # thread that is not running.
        self._streaming_started = False
        # Set alongside _vad_dirty when the in-progress utterance must be
        # thrown away rather than spoken -- see _degrade_streaming.
        self._vad_drop_pending = False
        self.apply_text_changes()
        self.last_transcript = ""
        self.output_failures: list[tuple[str, str]] = []
        # Outputs already reported as gone, so the watchdog says it once rather
        # than once a second.
        self._reported_dead_outputs: set[str] = set()

        self.history: deque[HistoryEntry] = deque(maxlen=max(1, cfg.text.history_size))
        # Set by the UI to approve or edit text before it is spoken. Called on the
        # worker thread and expected to block; returns the text to speak, or None
        # to discard.
        self.review_hook: Callable[[str], str | None] | None = None

        # Unbounded. Speaking N seconds of speech takes N seconds, so a talker
        # who does not pause will always be ahead of playback -- dropping the
        # overflow silently loses words that were actually said. Falling behind
        # is visible and recoverable; a dropped sentence is neither.
        self._utterances: queue.Queue[Any] = queue.Queue()
        # Observer notices ARE droppable: they only drive the tray and the log.
        self._notices: queue.Queue[Any] = queue.Queue(maxsize=256)
        # Set when the VAD settings change; the segmenter thread rebuilds.
        self._vad_dirty = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        # Set to abandon speech already in progress. Checked between synthesized
        # chunks, so a long utterance stops at the next sentence boundary rather
        # than only after it has finished.
        self._cancel_speech = threading.Event()
        self._ptt_engaged = threading.Event()
        self._ptt_buf: list[np.ndarray] = []
        self._ptt_lock = threading.Lock()
        self._lifecycle = threading.Lock()  # serializes start/stop

    # -- observers ------------------------------------------------------------

    def _set_state(self, state: State) -> None:
        if state is self.state:
            return
        previous = self.state
        # Only police a pipeline that is actually running. A stopped one has no
        # state machine to violate -- the settings window drives it directly to
        # audition a voice, and so do the tests.
        policed = self._running.is_set() or previous is not State.STOPPED
        if policed and state not in _ALLOWED.get(previous, frozenset()):
            # Loud, and then allowed. See _ALLOWED for why refusing would be
            # worse. This is the line that turns "the tray said listening while
            # it was speaking" from a mystery into a stack to look at.
            log.error("illegal state change %s -> %s; the pipeline reached a "
                      "combination that is not meant to exist",
                      previous.value, state.value)
        self.state = state
        log.debug("state -> %s", state.value)
        self._notify(("state", state))

    def _notify(self, notice: tuple[str, Any]) -> None:
        """Hand an observer call to the notifier thread.

        NEVER call an observer from the audio path. The tray and settings
        window marshal onto the Tk thread with `root.after()`, and a
        cross-thread Tkinter call serialises on the Tcl interpreter lock -- so
        a busy interface can block whichever thread announced the change. When
        that thread is the worker, it blocks inside _set_state(THINKING),
        before it ever reaches speak(): the app falls silent, the tray sits on
        "thinking", and nothing recovers it short of a restart.

        Queueing means the worst a stalled interface can do is fall behind.
        """
        try:
            self._notices.put_nowait(notice)
        except queue.Full:
            # Observers are cosmetic; audio is not. Drop the notice rather than
            # block or grow without limit.
            log.debug("observer queue full; dropped %r", notice[0])

    def _notifier_loop(self) -> None:
        while True:
            notice = self._notices.get()
            if notice is _STOP:
                return
            kind, payload = notice
            try:
                if kind == "state" and self._on_state:
                    self._on_state(payload)
                elif kind == "event" and self._on_event:
                    self._on_event(*payload)
            except Exception:
                log.exception("observer failed")

    def _emit(self, kind: str, message: str) -> None:
        # "partial" fires several times a second in streaming mode; at info it
        # would bury everything else in the log.
        log.log(logging.DEBUG if kind == "partial" else logging.INFO,
                "[%s] %s", kind, message)
        self._notify(("event", (kind, message)))

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle:
            if self._running.is_set():
                return
            self._set_state(State.LOADING)
            try:
                # Worked out BEFORE the engines, because it decides which task
                # Whisper is loaded with and whether a translation chain is
                # needed at all. Everything downstream reads this one answer.
                self.plan = plan_mod.build(self.cfg)
                self._load_engines()
                self._open_audio()
            except Exception as exc:
                log.exception("startup failed")
                self._emit("error", f"Startup failed: {exc}")
                self._teardown()
                self._set_state(State.STOPPED)
                raise

            self._streaming_started = (
                self.cfg.stt.mode is RecognitionMode.STREAMING)
            self._streaming_degraded = False
            self._backlog_reported = False
            self._report_plan()
            self._running.set()
            self._spawn(self._notifier_loop, "notifier")
            self._spawn(self._segmenter_loop, "segmenter")
            self._spawn(self._worker_loop, "worker")
            self._spawn(self._watchdog_loop, "watchdog")
            if self._streaming_started:
                self._spawn(self._streaming_loop, "streaming")
            self._start_hotkey()
            self._set_state(State.IDLE)
            self._emit("info", "Running")

    @staticmethod
    def _first_speakable_voice() -> str | None:
        """Any installed voice this build can actually pronounce."""
        from . import voices

        return next((key for key in voices.installed_keys()
                     if voices.is_speakable(key)), None)

    def _load_engines(self) -> None:
        if self.stt is None:
            self.stt = WhisperEngine(self.cfg.stt, task=self.plan.whisper_task)
            self._emit("info", f"Whisper on {self.stt.device}/{self.stt.compute_type}")
        if self.tts is None:
            try:
                self.tts = PiperEngine(self.cfg.tts)
            except VoiceSubstituted as exc:
                # Take the substitute, but WRITE IT DOWN. Leaving cfg.tts.voice
                # pointing at the missing one meant every language check after
                # this reasoned about a voice that was never loaded.
                self._emit("warning", str(exc))
                self.cfg.tts.voice = exc.using.stem
                self.tts = PiperEngine(self.cfg.tts)
                self.plan = plan_mod.build(self.cfg)
            except UnspeakableVoice as exc:
                # Starting is better than not starting. The saved voice needs a
                # phonemizer this build does not carry, so fall back to one that
                # works and say so -- refusing to open at all would leave no way
                # to change the setting.
                fallback = self._first_speakable_voice()
                if fallback is None:
                    raise
                self._emit("error", f"{exc} Using {fallback} instead.")
                self.cfg.tts.voice = fallback
                self.tts = PiperEngine(self.cfg.tts)
            self._emit("info", f"Voice {self.tts.voice_path.stem}")

        self._load_translator()

        # Pay cuDNN autotune and ONNX graph setup now, not on the first utterance.
        self._emit("info", "Warming up models...")
        self.stt.warmup()
        self.tts.warmup()

    def _report_plan(self) -> None:
        """Say what this run is going to do, and everything wrong with it.

        The same reasoning the settings window shows, from the same function.
        They used to decide separately and disagree -- the window said a
        Japanese voice was wrong for translating INTO Japanese while the
        pipeline happily did it.
        """
        for line in self.plan.describe():
            log.info("plan: %s", line)
        self._emit("info", self.plan.summary)
        for problem in self.plan.problems:
            self._emit("error" if problem.serious else "warning", str(problem))

    @property
    def streaming_mode(self) -> bool:
        """Whether text should come out while someone is still speaking.

        Push-to-talk hands over one finished recording, so there is nothing to
        stream even when the mode is selected -- and `_streaming_degraded` turns
        it off for the rest of the session once the machine has proved it cannot
        keep up.
        """
        return (self._streaming_started
                and self.cfg.trigger.mode is not TriggerMode.PTT
                and not self._streaming_degraded)

    def _load_translator(self) -> None:
        """Build the translation chain, or leave it off and say why.

        A missing model must not stop the app starting. Speaking your own words
        untranslated is a degraded service; refusing to run at all is not a
        service, and the user cannot download a model from a window that will
        not open.
        """
        self.translator = None
        cfg = self.cfg.translation
        # needs_chain, not "is translation on": the recogniser mode translates
        # without a chain, and asking the wrong question here is how a config
        # with both settings came to translate twice.
        if not self.plan.needs_chain:
            return
        from . import translate

        try:
            self.translator = translate.chain_for(
                cfg.source, cfg.target, pivot=cfg.pivot, beam_size=cfg.beam_size)
        except translate.TranslationUnavailable as exc:
            # Not announced here: _report_plan already says this, in the same
            # words the settings window uses. Saying it twice, differently, is
            # how the two came to disagree in the first place.
            log.warning("no translation chain: %s", exc)
            return
        # Not announced here -- _report_plan says this, in the words the
        # settings window uses.
        log.info("translation chain ready: %s",
                 " via ".join(pair.label for pair in self.translator.pairs))

    def _open_audio(self) -> None:
        assert self.tts is not None
        self.sink = OutputSink(self.cfg.audio)
        self._reported_dead_outputs = set()
        self.output_failures = self.sink.configure(self.cfg.audio.outputs, self.tts.rate)
        for label, reason in self.output_failures:
            self._emit("warning", f"Output unavailable: {label} ({reason})")
        if not self.sink.targets:
            enabled = [t.label for t in self.cfg.audio.outputs if t.enabled]
            detail = f"none of {enabled} could be opened" if enabled else "none are enabled"
            self._emit(
                "warning",
                f"No output devices ({detail}) -- speech will go nowhere. "
                "Fix this in Settings -> Audio.",
            )

        device = resolve_input(
            self.cfg.audio.input_match, all_apis=not self.cfg.audio.prefer_wasapi
        )
        if device is None:
            raise RuntimeError("no usable input device")
        self.capture = MicCapture(device)
        self.capture.start()

        self.segmenter = VadSegmenter(
            self.cfg.vad,
            preroll_ms=self.cfg.trigger.preroll_ms,
            max_utterance_s=self.cfg.trigger.max_utterance_s,
        )

    def _start_hotkey(self) -> None:
        """(Re)bind every hotkey. All share one keyboard hook."""
        if self.hotkey is None:
            self.hotkey = HotkeyManager()
        self.hotkey.clear()

        trig = self.cfg.trigger
        # `object`, not None: speak_clipboard and stop_speaking both return
        # something useful to their other callers, and a hotkey ignores it.
        Binding = tuple[str, str, Callable[[], object], Callable[[], object] | None]
        wanted: list[Binding] = []
        if trig.mode in ("ptt", "both"):
            wanted.append(("ptt", trig.hotkey, self._on_ptt_press, self._on_ptt_release))
        if trig.clipboard_hotkey:
            wanted.append(("clipboard", trig.clipboard_hotkey,
                           self.speak_clipboard, None))
        if trig.stop_hotkey:
            wanted.append(("stop", trig.stop_hotkey, self.stop_speaking, None))

        for name, combo, press, release in wanted:
            try:
                self.hotkey.bind(name, combo, press, release)
            except ValueError as exc:
                self._emit("error", f"Bad {name} hotkey {combo!r}: {exc}")

        for a, b in self.hotkey.conflicts():
            self._emit("warning", f"Hotkeys {a} and {b} use the same combination")

        if self.hotkey.bound:
            self.hotkey.start()
        else:
            self.hotkey.stop()

    def _spawn(self, fn: Callable[[], None], name: str) -> None:
        t = threading.Thread(target=fn, name=f"v2t-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        with self._lifecycle:
            if not self._running.is_set():
                return
            self._running.clear()
            self._utterances.put(_STOP)
            try:
                self._notices.put_nowait(_STOP)
            except queue.Full:
                # Full of stale notices; clear one out so the sentinel lands.
                try:
                    self._notices.get_nowait()
                    self._notices.put_nowait(_STOP)
                except (queue.Empty, queue.Full):
                    pass
            self._teardown()
            for t in self._threads:
                t.join(timeout=2.0)
            self._threads = []
            self._set_state(State.STOPPED)
            self._emit("info", "Stopped")

    def _teardown(self) -> None:
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        if self.sink is not None:
            self.sink.close()
            self.sink = None
        with self._ptt_lock:
            self._ptt_buf = []
        self._ptt_engaged.clear()

    def shutdown(self) -> None:
        """Stop and release the models too."""
        self.stop()
        self.stt = None
        self.tts = None
        if self.translator is not None:
            self.translator.close()
            self.translator = None

    # -- runtime reconfiguration ---------------------------------------------

    def apply_audio_changes(self) -> None:
        """Reopen output devices after the user edits them in settings."""
        if not self._running.is_set() or self.sink is None or self.tts is None:
            return
        self.output_failures = self.sink.configure(self.cfg.audio.outputs, self.tts.rate)
        for label, reason in self.output_failures:
            self._emit("warning", f"Output unavailable: {label} ({reason})")

    def apply_tts_changes(self) -> None:
        """Apply speed/volume changes; reload the model only if the voice changed."""
        if self.tts is None:
            return
        if self.tts.voice_path.stem != self.cfg.tts.voice and self.cfg.tts.voice:
            try:
                self.tts = PiperEngine(self.cfg.tts)
                self.tts.warmup()
                self.apply_audio_changes()  # sample rate may differ between voices
            except Exception as exc:  # noqa: BLE001
                self._emit("error", f"Could not load voice: {exc}")
            return
        self.tts.apply(self.cfg.tts)

    def apply_text_changes(self) -> int:
        """Recompile both rule sets. Returns how many are active in total."""
        cfg = self.cfg.text
        if not cfg.substitutions_enabled:
            self.target_substituter.load([])
            return self.substituter.load([])

        def compile_rules(entries: Iterable[Any]) -> list[Rule]:
            return [
                Rule(
                    pattern=r.pattern,
                    replacement=r.replacement,
                    enabled=r.enabled,
                    whole_word=r.whole_word,
                    regex=r.regex,
                    case_sensitive=r.case_sensitive,
                )
                for r in entries
            ]

        active = self.substituter.load(compile_rules(cfg.substitutions))
        active += self.target_substituter.load(
            compile_rules(cfg.target_substitutions))
        # A rule that will not compile used to vanish into the log, so the count
        # said "6 rule(s) active" over seven rules and the missing one was the
        # one somebody had just typed.
        for reason in self.dropped_rules:
            self._emit("warning", f"Rule not used -- {reason}")
        return active

    @property
    def dropped_rules(self) -> list[str]:
        """Rules that could not be compiled, across both lists."""
        return self.substituter.dropped + self.target_substituter.dropped

    def apply_translation_changes(self) -> None:
        """Rebuild the chain after a settings change.

        Called on the interface thread, and loading a model takes ~200 ms, so
        this is only worth doing when something actually changed -- which the
        caller decides, because it is the one that knows what the user edited.
        """
        previous = self.translator
        self.plan = plan_mod.build(self.cfg)
        self._load_translator()
        if previous is not None and previous is not self.translator:
            previous.close()
        if self.stt is not None and self.stt.task is not self.plan.whisper_task:
            # Changing WHO translates changes what Whisper is asked for, and the
            # engine holds that from load. Say so rather than looking applied.
            self._emit("warning",
                       "Restart the app for the change of translation method to "
                       "take effect.")
        for line in self.plan.describe():
            log.info("plan: %s", line)

    def apply_vad_changes(self) -> None:
        """Ask the segmenter thread to rebuild, rather than swapping it here.

        Replacing self.segmenter from another thread drops whatever is
        mid-capture and races with the reader, which holds its own reference
        across process()/active. The thread picks this up between windows.
        """
        if self._running.is_set():
            self._vad_dirty.set()

    def set_mode(self, mode: TriggerMode) -> None:
        """Change how speech is started, repairing anything the change breaks.

        It used to assign the field and skip validate(), so switching to
        push-to-talk while streaming left an impossible pair that app.py then
        saved to disk.
        """
        self.cfg.trigger.mode = mode
        for repair in self.cfg.validate():
            self._emit("warning", str(repair))
        if not self._running.is_set():
            return
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        self._start_hotkey()
        if self.segmenter is not None:
            self.segmenter.reset()
        self._emit("info", f"Mode: {mode.value}")

    # -- push to talk ---------------------------------------------------------

    def _on_ptt_press(self) -> None:
        if not self._running.is_set():
            return
        if self.cfg.trigger.ptt_latch and self._ptt_engaged.is_set():
            self._finish_ptt()
            return
        # Holding the key while speech plays means "interrupt and talk over it".
        if self.sink is not None and self.sink.active:
            self.sink.clear()
        with self._ptt_lock:
            self._ptt_buf = []
        self._ptt_engaged.set()
        self._set_state(State.LISTENING)

    def _on_ptt_release(self) -> None:
        if self.cfg.trigger.ptt_latch or not self._ptt_engaged.is_set():
            return
        self._finish_ptt()

    def _finish_ptt(self) -> None:
        self._ptt_engaged.clear()
        with self._ptt_lock:
            chunks, self._ptt_buf = self._ptt_buf, []
        if not chunks:
            self._set_state(State.IDLE)
            return
        audio = np.concatenate(chunks)
        if len(audio) < SAMPLE_RATE // 5:  # under 200 ms is a mis-tap
            self._set_state(State.IDLE)
            return
        self._submit(audio)

    def _submit(self, item: Any) -> None:
        try:
            self._utterances.put_nowait(item)
            self._set_state(State.THINKING)
        except queue.Full:
            self._emit("warning", "Busy -- utterance dropped")
        self._check_backlog()

    def _check_backlog(self) -> None:
        """Notice when the voice cannot keep up with the speaker.

        Speaking takes about as long as saying it did, so in streaming mode the
        pipeline runs at roughly its own capacity: any delay it picks up it
        keeps, and if the voice is slower than the speaker the gap grows without
        limit. Nothing is dropped -- falling behind is the right trade -- but
        silently drifting further behind looks like a fault, so it is said once
        with the setting that actually fixes it.
        """
        if not self.streaming_mode or self._backlog_reported:
            return
        if self._utterances.qsize() < _BACKLOG_WARN_DEPTH:
            return
        self._backlog_reported = True
        self._emit(
            "warning",
            f"Speech is running behind ({self._utterances.qsize()} phrases "
            "waiting). The voice cannot speak faster than you are talking -- "
            "raise the speed on the Voice tab, or pause a little more.",
        )

    # -- threads --------------------------------------------------------------

    def _segmenter_loop(self) -> None:
        assert self.capture is not None
        max_ptt = int(self.cfg.trigger.max_utterance_s * SAMPLE_RATE / WINDOW)
        vad_failures = 0
        capture_reported = False
        next_recovery = 0.0
        recovery_wait = _RECOVERY_INTERVAL_S

        while self._running.is_set():
            # The capture stream can die under us -- unplugging a USB microphone is
            # the usual cause. Report it once and keep the thread alive so the user
            # can pick a different device without restarting.
            if self.capture is not None and not self.capture.check_alive():
                if not capture_reported:
                    capture_reported = True
                    self._emit(
                        "error",
                        f"Microphone stopped: {self.capture.failure_reason}. "
                        "Waiting for it to come back...",
                    )
                    self._set_state(State.IDLE)
                # Unplugging a USB microphone should not mean restarting the app.
                # Retry on a timer rather than only when something else pokes us,
                # and back off: a device that is not coming back must not keep
                # re-enumerating every audio interface for as long as the app is
                # open. Reported from the field as constant CPU load.
                if time.monotonic() >= next_recovery:
                    next_recovery = time.monotonic() + recovery_wait
                    if self._try_recover_capture():
                        capture_reported = False
                        recovery_wait = _RECOVERY_INTERVAL_S
                    else:
                        recovery_wait = min(recovery_wait * 2,
                                            _RECOVERY_MAX_INTERVAL_S)
                        if recovery_wait >= _RECOVERY_MAX_INTERVAL_S:
                            log.info("microphone still unavailable; retrying "
                                     "every %.0fs", _RECOVERY_MAX_INTERVAL_S)
                continue
            try:
                window = self.capture.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except AttributeError:
                # capture is momentarily absent; _running is the authority on
                # whether we should stop, so wait rather than killing the thread.
                time.sleep(0.05)
                continue

            if self._ptt_engaged.is_set():
                with self._ptt_lock:
                    self._ptt_buf.append(window)
                    overrun = len(self._ptt_buf) >= max_ptt
                if overrun:
                    self._emit("warning", "Max utterance length reached")
                    self._finish_ptt()
                continue

            if self.cfg.trigger.mode == "ptt" or self.segmenter is None:
                continue

            # Settings changed. Rebuild here, where nothing else is mid-window.
            if self._vad_dirty.is_set():
                self._vad_dirty.clear()
                drop_pending = self._vad_drop_pending
                self._vad_drop_pending = False
                pending = self.segmenter.flush()
                self.segmenter = VadSegmenter(
                    self.cfg.vad,
                    preroll_ms=self.cfg.trigger.preroll_ms,
                    max_utterance_s=self.cfg.trigger.max_utterance_s,
                )
                if pending is not None and not drop_pending:
                    self._submit(pending)

            # Suppress VAD while speaking, or the synthesized voice leaking through
            # speakers gets transcribed and spoken again in a loop.
            #
            # NOT in streaming mode. There the app is speaking for much of the
            # time someone is talking, and both ways of suppressing are worse
            # than the feedback they prevent -- measured, not assumed:
            #
            #   Ending the stream flushed whatever the last pass happened to
            #   say, text no second pass had agreed with, which is how "The
            #   build finished" was spoken as "It still finished".
            #
            #   Merely skipping the windows leaves a splice in the buffer, and
            #   Whisper reads straight across it: "the audio device enumeration
            #   is p-unplugged the interface".
            #
            # So streaming keeps listening, and the interface says plainly that
            # it needs headphones or a virtual-cable-only output. See
            # _refresh_stt_mode.
            if (
                self.cfg.audio.mute_mic_during_playback
                and not self.streaming_mode
                and self.sink is not None
                and self.sink.active
            ):
                if self.segmenter.active:
                    # Keep what was captured before playback began; only stop
                    # listening. reset() here used to bin it.
                    pending = self.segmenter.suspend()
                    if pending is not None:
                        self._submit(pending)
                continue

            # An exception here used to kill this thread outright: automatic mode
            # would stop working with the tray icon still showing "ready" and
            # nothing in the log. Recover in place instead, and only give up if it
            # keeps failing.
            try:
                utterance = self.segmenter.process(window)
                vad_failures = 0
            except Exception:
                log.exception("VAD failed on a window")
                vad_failures += 1
                if vad_failures == 1:
                    self._emit("warning", "Speech detection hiccup; recovering")
                if vad_failures >= _MAX_VAD_FAILURES:
                    # set_mode, not an assignment. Assigning the field announced
                    # push-to-talk and left the key unbound: in `vad` mode it had
                    # never been bound in the first place, so the app went deaf
                    # while still reporting that it was ready.
                    self._emit(
                        "error",
                        "Speech detection keeps failing; switching to "
                        f"push-to-talk ({self.cfg.trigger.hotkey}). "
                        "See the log for details.",
                    )
                    self.set_mode(TriggerMode.PTT)
                    vad_failures = 0
                else:
                    self.segmenter.reset()
                continue

            if self.segmenter.active and self.state is State.IDLE:
                self._set_state(State.LISTENING)
            elif not self.segmenter.active and self.state is State.LISTENING:
                self._set_state(State.IDLE)

            if self.streaming_mode:
                # The VAD is only here to say when speech starts and stops. The
                # audio itself goes to the streaming thread, and the utterance
                # the segmenter builds is deliberately NOT submitted -- doing
                # both would say every word twice.
                if self.segmenter.active:
                    if not self._stream_active.is_set():
                        self._stream_active.set()
                        # Hand over the onset too, or every utterance loses its
                        # first word or so to the detection delay.
                        held_windows = self.segmenter.captured()
                        for i, held in enumerate(held_windows):
                            try:
                                self._stream_windows.put_nowait(held)
                            except queue.Full:
                                # The first words of the utterance, gone. It used
                                # to break out with no log at all, so the app
                                # spoke a sentence with its opening missing and
                                # nothing anywhere said why.
                                log.warning(
                                    "streaming queue full at handover; lost "
                                    "%d of %d onset windows",
                                    len(held_windows) - i, len(held_windows))
                                self._emit("warning",
                                           "Machine is behind; the start of that "
                                           "sentence was lost")
                                break
                    else:
                        try:
                            self._stream_windows.put_nowait(window)
                        except queue.Full:
                            log.debug("streaming queue full; dropping a window")
                elif self._stream_active.is_set():
                    # Cleared HERE, by the thread that sends it. Leaving that to
                    # the streaming thread meant this branch fired again on
                    # every 32 ms window until it caught up -- thirteen
                    # teardowns for five utterances.
                    self._stream_active.clear()
                    self._stream_windows.put(_STOP)
                continue

            if utterance is not None:
                self._submit(utterance)

    def _streaming_loop(self) -> None:
        """Recognise while the speaker is still going, on a thread of its own.

        A pass takes around half a second, so this cannot live on the segmenter
        thread -- that thread has to keep draining the capture queue every 32 ms
        or windows are lost. The segmenter forwards audio here and this thread
        does nothing but read it.
        """
        assert self.stt is not None
        # A translator handed a clause fragment produces fluent nonsense, so
        # while translating this waits for a real sentence end.
        #
        # Re-read every pass rather than snapshotted here: translation can be
        # switched on from the settings window while this thread is running, and
        # a snapshot meant the clause fragments this exists to prevent were fed
        # to the translator for the rest of the session.
        translating = self.translator is not None
        self.streamer = StreamingRecognizer(self.stt, sentences_only=translating)
        log.info("streaming recognition started (sentences only: %s)", translating)

        while self._running.is_set():
            try:
                window = self._stream_windows.get(timeout=0.2)
            except queue.Empty:
                window = None

            if window is _STOP:
                # The speaker paused. Whatever is still held has no later pass
                # to agree with, so it goes out on the strength of the last one
                # rather than being silently dropped.
                tail = self.streamer.finish()
                if tail:
                    self._submit(tail)
                continue

            if window is not None:
                self.streamer.feed(window)

            if not self.streamer.buffered_s:
                continue

            try:
                result = self.streamer.poll()
            except Exception:
                log.exception("streaming pass failed")
                self._emit("warning", "Streaming hiccup; recovering")
                self.streamer.reset()
                continue
            if result is None:
                continue

            if result.speakable:
                self._submit(result.speakable)
            if result.unstable:
                # Shown, never spoken: this is the text still moving about.
                self._emit("partial", result.unstable)
            if result.fell_behind:
                self._degrade_streaming()
                return

    def _degrade_streaming(self) -> None:
        """Give up on streaming for the rest of the session, and say so.

        Not a silent downgrade: the user chose this mode, and a mode that
        quietly stops being the mode is worse than one that admits it. Anything
        still buffered is released first so nothing is lost in the handover.
        """
        if self._streaming_degraded:
            return
        self._streaming_degraded = True
        if self.streamer is not None:
            tail = self.streamer.finish()
            if tail:
                self._submit(tail)
        self._stream_active.clear()

        # The segmenter has been accumulating this whole utterance from the
        # moment speech started, and most of it has already been spoken. Handing
        # that buffer to sentence mode would say the lot a second time -- which
        # is exactly the talking-over-itself the soft endpoint was reverted for.
        self._vad_drop_pending = True
        self._vad_dirty.set()
        self._emit(
            "warning",
            "This machine cannot recognise fast enough to stream, so speech "
            "will now wait for a pause instead. Recognition -> Mode explains "
            "the trade-off; a smaller Whisper model may let streaming work.",
        )

    def _watchdog_loop(self) -> None:
        """Notice when the pipeline stops making progress, and say where.

        An utterance is transcribed and spoken in seconds. Sitting in THINKING
        or SPEAKING for far longer means something is blocked, and "it got
        stuck" is not a report anybody can act on. This logs the stack of every
        thread once per stall, which turns it into "it is blocked HERE".
        """
        stalled_since: float | None = None
        reported = False
        while self._running.is_set():
            time.sleep(1.0)
            # REVIEWING counts: it is bounded by review_timeout_s, so sitting
            # here past the stall window means the wait itself is wedged -- and
            # it was the one state the watchdog did not look at.
            self._check_outputs()
            busy = self.state in (State.THINKING, State.SPEAKING, State.REVIEWING)
            if not busy:
                stalled_since, reported = None, False
                continue
            now = time.monotonic()
            if stalled_since is None:
                stalled_since, reported = now, False
                continue
            if reported or now - stalled_since < STALL_SECONDS:
                continue

            reported = True
            from .perf import stacks

            log.error(
                "pipeline stalled in %s for %.0fs "
                "(utterances queued: %d, notices queued: %d, segmenter active: %s)",
                self.state.value, now - stalled_since,
                self._utterances.qsize(), self._notices.qsize(),
                self.segmenter.active if self.segmenter else "n/a",
            )
            for line in stacks():
                log.error("%s", line)
            self._emit(
                "error",
                f"Stopped responding while {self.state.value}. The log now holds "
                "a stack dump -- please include it in a bug report.",
            )

    def _check_outputs(self) -> None:
        """Notice a speaker that has gone away, and say so once.

        The microphone has been checked since 0.4; the speaker never was. An
        output stream is not reliably told when its device is unplugged -- the
        callback simply stops -- so the app went on listening, transcribing and
        synthesising into nothing, reporting "speaking" throughout.
        """
        if self.sink is None:
            return
        dead = self.sink.dead_targets()
        if not dead:
            return
        for name, reason in dead:
            if name in self._reported_dead_outputs:
                continue
            self._reported_dead_outputs.add(name)
            self._emit("error",
                       f"{name} is no longer accepting audio ({reason}). "
                       "Nothing said is being heard there -- reopen Settings "
                       "-> Audio once the device is back.")

    def _try_recover_capture(self) -> bool:
        """Reopen the microphone if it is available again. Returns True on success."""
        wanted = self.cfg.audio.input_match
        # Re-enumerate: a device that was unplugged and replugged gets a new
        # PortAudio index, so the cached list would point at nothing.
        devices.refresh()
        device = resolve_input(wanted, all_apis=not self.cfg.audio.prefer_wasapi)
        if device is None:
            return False

        try:
            fresh = MicCapture(device)
            fresh.start()
        except Exception as exc:  # noqa: BLE001 - the device may still be settling
            log.debug("microphone not ready yet: %s", exc)
            return False

        # start() only means PortAudio accepted the request. A device that is
        # present but unusable fails a moment later, from its own callback --
        # and reporting success on the strength of start() alone produced an
        # endless three-second cycle of "reconnected" and "stopped again", each
        # round re-enumerating every audio device on the machine. That loop is
        # what users saw as the fans spinning up.
        settle = time.monotonic() + _RECONNECT_SETTLE_S
        while time.monotonic() < settle:
            if fresh.failed:
                log.debug("microphone failed immediately after opening: %s",
                          fresh.failure_reason)
                fresh.stop()
                return False
            time.sleep(0.05)

        # Swap in one assignment and only then close the old one. Nulling
        # self.capture first would make the segmenter loop see None and treat it as
        # teardown, killing the thread mid-recovery.
        old, self.capture = self.capture, fresh
        if old is not None:
            old.stop()
        if self.segmenter is not None:
            self.segmenter.reset()
        self._emit("info", f"Microphone reconnected: {device.name}")
        return True

    def _worker_loop(self) -> None:
        while self._running.is_set():
            try:
                item = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            try:
                self._handle_utterance(item)
            except Exception:
                log.exception("utterance failed")
                self._emit("error", "Failed to process utterance")
            finally:
                if self._running.is_set():
                    self._set_state(State.IDLE)

    def _translate(self, text: str) -> str | None:
        """Translate, or keep going untranslated. None means drop the utterance.

        A translation failure mid-call is not worth going silent over, so the
        original text is spoken instead -- but it is said out loud in the event
        log, because otherwise the far end quietly starts hearing English again
        and nobody knows why.
        """
        assert self.translator is not None
        started = time.perf_counter()
        try:
            translated = self.translator.translate(text)
        except Exception as exc:  # falling back beats going silent
            log.exception("translation failed")
            self._emit("warning", f"Translation failed ({exc}); speaking as heard")
            return text
        if not translated.strip():
            # Same outcome as a thrown exception -- speak the original -- so it
            # gets the same announcement. Silence here meant the far end started
            # hearing untranslated English with nothing to explain it.
            log.warning("translation of %r came back empty", text[:60])
            self._emit("warning",
                       "Translation came back empty; speaking as heard")
            return text
        self._emit("translated",
                   f"{translated}  [{(time.perf_counter() - started) * 1000:.0f} ms]")
        return translated

    def _handle_utterance(self, item: Any) -> None:
        """One unit of work: audio to recognise, or text already recognised.

        Streaming mode does its own recognition on its own thread, so what
        arrives here is text. Everything after recognition is identical, which
        is why the two share this path rather than growing a second copy of the
        substitution, translation, review and history logic.
        """
        if isinstance(item, str):
            self._deliver(item, time.perf_counter())
            return

        assert self.stt is not None
        self._set_state(State.THINKING)
        t0 = time.perf_counter()
        result = self.stt.recognise(item)
        if not result:
            # "" used to mean all three of these, so a user whose min_chars was
            # set high, or whose mic was recording silence, saw a pipeline that
            # listened, thought, and then did nothing at all.
            log.info("nothing spoken: %s", result.why.value)
            if result.why is Heard.NOISE:
                self._emit("info", "Ignored: " + result.why.value)
            return
        self._deliver(result.text, t0)

    def _deliver(self, text: str, t0: float) -> None:
        """Rewrite, translate, review and speak an already-recognised utterance."""
        self.last_transcript = text
        self._emit("transcript", text)

        # Rewrite before speaking, not before displaying: the transcript shown is
        # what was heard, so a wrong substitution is visible rather than mysterious.
        #
        # Source rules first: they fix what the recogniser misheard, and must be
        # applied while the text is still in the language that was spoken.
        spoken = self.substituter.apply(text)

        if self.translator is not None:
            translated = self._translate(spoken)
            if translated is None:
                return
            spoken = translated

        # Target rules last: they fix what the VOICE says badly, which is a
        # property of the output language.
        spoken = self.target_substituter.apply(spoken)
        if spoken != text:
            self._emit("substituted", spoken)

        if self.cfg.text.review_before_speaking and self.review_hook is not None:
            self._set_state(State.REVIEWING)
            approved = self._review(spoken)
            if approved is None:
                self._emit("info", "Discarded before speaking")
                return
            spoken = approved

        self.record(text, spoken, "speech")
        self.speak(spoken, _t0=t0)

    def _review(self, text: str) -> str | None:
        """Ask the UI to approve `text`. Returns the approved text, or None.

        A hook that fails discards, exactly like a hook that times out. Those
        two paths used to disagree -- a timeout dropped the utterance, a crash
        spoke it unreviewed -- which made the one feature whose entire purpose
        is "nothing unreviewed gets spoken" do the opposite under a fault
        nobody would ever see coming.
        """
        assert self.review_hook is not None
        try:
            return self.review_hook(text)
        except Exception:
            log.exception("review hook failed; discarding unreviewed text")
            self._emit("error",
                       "The review window failed, so nothing was spoken. Turn "
                       "review off in Misc if this keeps happening.")
            return None

    def record(self, heard: str, spoken: str, source: str = "speech") -> None:
        """Add to the in-memory history shown in the History tab."""
        self.history.append(HistoryEntry(heard=heard, spoken=spoken, at=time.time(),
                                         source=source))

    def clear_history(self) -> None:
        self.history.clear()
        self._emit("info", "History cleared")

    # -- speech output --------------------------------------------------------

    def speak(self, text: str, _t0: float | None = None) -> None:
        """Synthesize text and play it to every configured output."""
        if self.tts is None or self.sink is None:
            return
        text = (text or "").strip()
        if not text:
            return

        self._cancel_speech.clear()
        self._set_state(State.SPEAKING)
        first = True
        self.sink.begin_utterance()
        cancelled = False
        try:
            for chunk in self.tts.stream(text):
                if not self._running.is_set() or self._cancel_speech.is_set():
                    cancelled = True
                    return
                if first and _t0 is not None:
                    self._emit(
                        "latency",
                        f"{(time.perf_counter() - _t0) * 1000:.0f} ms to first audio",
                    )
                    first = False
                self.sink.write(chunk)
        finally:
            self.sink.end_utterance()
            if cancelled:
                # Drop whatever is still queued, or the cancelled sentence keeps
                # playing for as long as the buffer holds.
                self.sink.clear()
        if not cancelled:
            self.sink.wait_drain()

    def stop_speaking(self) -> bool:
        """Cut off speech in progress. Returns False if nothing was playing."""
        if self.sink is None:
            return False
        speaking = self.state is State.SPEAKING or self.sink.active
        self._cancel_speech.set()
        self.sink.clear()
        if speaking:
            self._emit("info", "Speech stopped")
        return speaking

    def speak_clipboard(self) -> str:
        """Speak the clipboard. Returns what was said, or "" if there was nothing."""
        from . import clipboard

        found = clipboard.get_speakable_text()
        if not found:
            # Which of the three it was, rather than "empty or holds no text",
            # which was the answer even when another program had it locked and
            # pressing the key again would have worked.
            self._emit("warning", f"Nothing spoken: {found.why.value}.")
            return ""
        text = found.text
        preview = text if len(text) <= 60 else text[:57] + "..."
        self._emit("clipboard", preview)
        self.say_text(text, source="clipboard")
        return text

    def say_text(self, text: str, source: str = "typed") -> None:
        """Speak text supplied directly by the UI, off the worker thread."""
        threading.Thread(
            target=self._say_text_worker, args=(text, source), name="v2t-say",
            daemon=True,
        ).start()

    def _say_text_worker(self, text: str, source: str = "typed") -> None:
        try:
            self.record(text, text, source)
            self.speak(text)
        except Exception:
            log.exception("say_text failed")
        finally:
            if self._running.is_set():
                self._set_state(State.IDLE)

    # -- introspection --------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "mode": self.cfg.trigger.mode,
            "hotkey": self.cfg.trigger.hotkey,
            "stt": f"{self.stt.device}/{self.stt.compute_type}" if self.stt else "-",
            "voice": self.tts.voice_path.stem if self.tts else "-",
            "outputs": [t.name for t in self.sink.targets] if self.sink else [],
            "input": self.capture.device.name if self.capture else "-",
            "dropped": self.capture.dropped if self.capture else 0,
            "last": self.last_transcript,
        }
