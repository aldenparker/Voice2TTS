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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import devices
from .capture import MicCapture
from .config import Config
from .devices import resolve_input
from .hotkey import HotkeyManager
from .output import OutputSink
from .stt import WhisperEngine
from .substitutions import Rule, Substituter
from .tts import PiperEngine
from .vad import SAMPLE_RATE, WINDOW, VadSegmenter

log = logging.getLogger(__name__)

_STOP = object()  # queue sentinel

# How long the pipeline may sit in a working state before it is treated as
# stalled. Transcribing and speaking one utterance is seconds; a minute is not
# slow, it is stuck.
STALL_SECONDS = 60.0

# Consecutive VAD errors before we stop trying and fall back to push-to-talk.
_MAX_VAD_FAILURES = 5

# How often to retry a microphone that has gone away. Long enough not to thrash
# PortAudio, short enough that replugging feels like it just works.
_RECOVERY_INTERVAL_S = 3.0


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
        self.translator = None                    # a translate.Chain, when on
        self.apply_text_changes()
        self.last_transcript = ""
        self.output_failures: list[tuple[str, str]] = []

        self.history: deque[HistoryEntry] = deque(maxlen=max(1, cfg.text.history_size))
        # Set by the UI to approve or edit text before it is spoken. Called on the
        # worker thread and expected to block; returns the text to speak, or None
        # to discard.
        self.review_hook: Callable[[str], str | None] | None = None

        # Unbounded. Speaking N seconds of speech takes N seconds, so a talker
        # who does not pause will always be ahead of playback -- dropping the
        # overflow silently loses words that were actually said. Falling behind
        # is visible and recoverable; a dropped sentence is neither.
        self._utterances: queue.Queue = queue.Queue()
        # Observer notices ARE droppable: they only drive the tray and the log.
        self._notices: queue.Queue = queue.Queue(maxsize=256)
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
        self.state = state
        log.debug("state -> %s", state.value)
        self._notify(("state", state))

    def _notify(self, notice) -> None:
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
        log.info("[%s] %s", kind, message)
        self._notify(("event", (kind, message)))

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle:
            if self._running.is_set():
                return
            self._set_state(State.LOADING)
            try:
                self._load_engines()
                self._open_audio()
            except Exception as exc:
                log.exception("startup failed")
                self._emit("error", f"Startup failed: {exc}")
                self._teardown()
                self._set_state(State.STOPPED)
                raise

            self._running.set()
            self._spawn(self._notifier_loop, "notifier")
            self._spawn(self._segmenter_loop, "segmenter")
            self._spawn(self._worker_loop, "worker")
            self._spawn(self._watchdog_loop, "watchdog")
            self._start_hotkey()
            self._set_state(State.IDLE)
            self._emit("info", "Running")

    def _load_engines(self) -> None:
        if self.stt is None:
            self.stt = WhisperEngine(self.cfg.stt)
            self._emit("info", f"Whisper on {self.stt.device}/{self.stt.compute_type}")
        if self.tts is None:
            self.tts = PiperEngine(self.cfg.tts)
            self._emit("info", f"Voice {self.tts.voice_path.stem}")

        self._load_translator()

        # Pay cuDNN autotune and ONNX graph setup now, not on the first utterance.
        self._emit("info", "Warming up models...")
        self.stt.warmup()
        self.tts.warmup()

    def _load_translator(self) -> None:
        """Build the translation chain, or leave it off and say why.

        A missing model must not stop the app starting. Speaking your own words
        untranslated is a degraded service; refusing to run at all is not a
        service, and the user cannot download a model from a window that will
        not open.
        """
        self.translator = None
        cfg = self.cfg.translation
        if not cfg.enabled:
            return
        from . import translate

        try:
            self.translator = translate.chain_for(
                cfg.source, cfg.target, pivot=cfg.pivot, beam_size=cfg.beam_size)
        except translate.TranslationUnavailable as exc:
            self._emit("warning",
                       f"Translation off: {exc}. Speech will not be translated.")
            return
        hops = " via ".join(p.label for p in self.translator.pairs)
        self._emit("info", f"Translating {hops}")

    def _open_audio(self) -> None:
        assert self.tts is not None
        self.sink = OutputSink(self.cfg.audio)
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
        wanted: list[tuple[str, str, object, object]] = []
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

    def _spawn(self, fn, name: str) -> None:
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

        def compile_rules(entries):
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
        return active

    def apply_translation_changes(self) -> None:
        """Rebuild the chain after a settings change.

        Called on the interface thread, and loading a model takes ~200 ms, so
        this is only worth doing when something actually changed -- which the
        caller decides, because it is the one that knows what the user edited.
        """
        previous = self.translator
        self._load_translator()
        if previous is not None and previous is not self.translator:
            previous.close()

    def apply_vad_changes(self) -> None:
        """Ask the segmenter thread to rebuild, rather than swapping it here.

        Replacing self.segmenter from another thread drops whatever is
        mid-capture and races with the reader, which holds its own reference
        across process()/active. The thread picks this up between windows.
        """
        if self._running.is_set():
            self._vad_dirty.set()

    def set_mode(self, mode: str) -> None:
        self.cfg.trigger.mode = mode
        if not self._running.is_set():
            return
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None
        self._start_hotkey()
        if self.segmenter is not None:
            self.segmenter.reset()
        self._emit("info", f"Mode: {mode}")

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

    def _submit(self, audio: np.ndarray) -> None:
        try:
            self._utterances.put_nowait(audio)
            self._set_state(State.THINKING)
        except queue.Full:
            self._emit("warning", "Busy -- utterance dropped")

    # -- threads --------------------------------------------------------------

    def _segmenter_loop(self) -> None:
        assert self.capture is not None
        max_ptt = int(self.cfg.trigger.max_utterance_s * SAMPLE_RATE / WINDOW)
        vad_failures = 0
        capture_reported = False
        next_recovery = 0.0

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
                # Retry on a timer rather than only when something else pokes us.
                if time.monotonic() >= next_recovery:
                    next_recovery = time.monotonic() + _RECOVERY_INTERVAL_S
                    if self._try_recover_capture():
                        capture_reported = False
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
                pending = self.segmenter.flush()
                self.segmenter = VadSegmenter(
                    self.cfg.vad,
                    preroll_ms=self.cfg.trigger.preroll_ms,
                    max_utterance_s=self.cfg.trigger.max_utterance_s,
                )
                if pending is not None:
                    self._submit(pending)

            # Suppress VAD while speaking, or the synthesized voice leaking through
            # speakers gets transcribed and spoken again in a loop.
            if (
                self.cfg.audio.mute_mic_during_playback
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
                    self._emit(
                        "error",
                        "Speech detection keeps failing; switching to push-to-talk. "
                        "See the log for details.",
                    )
                    self.cfg.trigger.mode = "ptt"
                    vad_failures = 0
                else:
                    self.segmenter.reset()
                continue

            if self.segmenter.active and self.state is State.IDLE:
                self._set_state(State.LISTENING)
            elif not self.segmenter.active and self.state is State.LISTENING:
                self._set_state(State.IDLE)
            if utterance is not None:
                self._submit(utterance)

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
            busy = self.state in (State.THINKING, State.SPEAKING)
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
            log.warning("translation of %r came back empty", text[:60])
            return text
        self._emit("translated",
                   f"{translated}  [{(time.perf_counter() - started) * 1000:.0f} ms]")
        return translated

    def _handle_utterance(self, audio: np.ndarray) -> None:
        assert self.stt is not None
        self._set_state(State.THINKING)
        t0 = time.perf_counter()
        text = self.stt.transcribe(audio)
        if not text:
            return
        self.last_transcript = text
        self._emit("transcript", text)

        # Rewrite before speaking, not before displaying: the transcript shown is
        # what was heard, so a wrong substitution is visible rather than mysterious.
        #
        # Source rules first: they fix what the recogniser misheard, and must be
        # applied while the text is still in the language that was spoken.
        spoken = self.substituter.apply(text)

        if self.translator is not None:
            spoken = self._translate(spoken)
            if spoken is None:
                return

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
        """Ask the UI to approve `text`. Returns the approved text, or None."""
        try:
            return self.review_hook(text)
        except Exception:
            # A broken hook must not swallow the utterance; speaking unreviewed is
            # the lesser failure, and the traceback lands in the log.
            log.exception("review hook failed; speaking unreviewed text")
            return text

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

        text = clipboard.get_speakable_text()
        if not text:
            self._emit("warning", "Clipboard is empty or holds no text")
            return ""
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

    def status(self) -> dict:
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
