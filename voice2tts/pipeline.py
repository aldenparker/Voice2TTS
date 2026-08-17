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
from collections.abc import Callable
from enum import Enum

import numpy as np

from .capture import MicCapture
from .config import Config
from .devices import resolve_input
from .hotkey import HotkeyListener
from .output import OutputSink
from .stt import WhisperEngine
from .tts import PiperEngine
from .vad import SAMPLE_RATE, WINDOW, VadSegmenter

log = logging.getLogger(__name__)

_STOP = object()  # queue sentinel

# Consecutive VAD errors before we stop trying and fall back to push-to-talk.
_MAX_VAD_FAILURES = 5


class State(Enum):
    STOPPED = "stopped"
    LOADING = "loading"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
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
        self.hotkey: HotkeyListener | None = None
        self.segmenter: VadSegmenter | None = None
        self.last_transcript = ""
        self.output_failures: list[tuple[str, str]] = []

        self._utterances: queue.Queue = queue.Queue(maxsize=8)
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
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
        if self._on_state:
            try:
                self._on_state(state)
            except Exception:
                log.exception("state observer failed")

    def _emit(self, kind: str, message: str) -> None:
        log.info("[%s] %s", kind, message)
        if self._on_event:
            try:
                self._on_event(kind, message)
            except Exception:
                log.exception("event observer failed")

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
            self._spawn(self._segmenter_loop, "segmenter")
            self._spawn(self._worker_loop, "worker")
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

        # Pay cuDNN autotune and ONNX graph setup now, not on the first utterance.
        self._emit("info", "Warming up models...")
        self.stt.warmup()
        self.tts.warmup()

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

        device = resolve_input(self.cfg.audio.input_match, self.cfg.audio.prefer_wasapi)
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
        if self.cfg.trigger.mode not in ("ptt", "both"):
            return
        try:
            self.hotkey = HotkeyListener(
                self.cfg.trigger.hotkey, self._on_ptt_press, self._on_ptt_release
            )
            self.hotkey.start()
        except ValueError as exc:
            self._emit("error", f"Bad hotkey {self.cfg.trigger.hotkey!r}: {exc}")
            self.hotkey = None

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=f"v2t-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        with self._lifecycle:
            if not self._running.is_set():
                return
            self._running.clear()
            try:
                self._utterances.put_nowait(_STOP)
            except queue.Full:
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

    def apply_vad_changes(self) -> None:
        if self._running.is_set():
            self.segmenter = VadSegmenter(
                self.cfg.vad,
                preroll_ms=self.cfg.trigger.preroll_ms,
                max_utterance_s=self.cfg.trigger.max_utterance_s,
            )

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

        while self._running.is_set():
            # The capture stream can die under us -- unplugging a USB microphone is
            # the usual cause. Report it once and keep the thread alive so the user
            # can pick a different device without restarting.
            if (
                self.capture is not None
                and not capture_reported
                and not self.capture.check_alive()
            ):
                capture_reported = True
                self._emit(
                    "error",
                    f"Microphone stopped: {self.capture.failure_reason}. "
                    "Pick another device in Settings -> Audio.",
                )
            try:
                window = self.capture.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except AttributeError:  # capture torn down mid-loop
                break

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

            # Suppress VAD while speaking, or the synthesized voice leaking through
            # speakers gets transcribed and spoken again in a loop.
            if (
                self.cfg.audio.mute_mic_during_playback
                and self.sink is not None
                and self.sink.active
            ):
                if self.segmenter.active:
                    self.segmenter.reset()
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

    def _handle_utterance(self, audio: np.ndarray) -> None:
        assert self.stt is not None
        self._set_state(State.THINKING)
        t0 = time.perf_counter()
        text = self.stt.transcribe(audio)
        if not text:
            return
        self.last_transcript = text
        self._emit("transcript", text)
        self.speak(text, _t0=t0)

    # -- speech output --------------------------------------------------------

    def speak(self, text: str, _t0: float | None = None) -> None:
        """Synthesize text and play it to every configured output."""
        if self.tts is None or self.sink is None:
            return
        text = (text or "").strip()
        if not text:
            return

        self._set_state(State.SPEAKING)
        first = True
        self.sink.begin_utterance()
        try:
            for chunk in self.tts.stream(text):
                if not self._running.is_set():
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
        self.sink.wait_drain()

    def say_text(self, text: str) -> None:
        """Speak text supplied directly by the UI, off the worker thread."""
        threading.Thread(
            target=self._say_text_worker, args=(text,), name="v2t-say", daemon=True
        ).start()

    def _say_text_worker(self, text: str) -> None:
        try:
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
