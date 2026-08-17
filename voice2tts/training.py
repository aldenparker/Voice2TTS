"""Fine-tune a Piper voice from a recorded dataset, and export the result.

Training runs in the studio pack's interpreter as a subprocess, never in ours:
torch is several GB and would otherwise have to be installed for everyone.

Three commands make a voice, and the arguments are taken from piper-tts 1.7.0's
own signatures rather than from its documentation:

    python -m piper.train fit ...                  # trains, and writes config.json
    python -m piper.train.export_onnx --checkpoint X --output-file Y
    (config.json is then copied beside the .onnx as <voice>.onnx.json)

The training step writes the voice config as a side effect, which is why the
export needs no separate config-building step.

FINE-TUNING VERSUS RESUMING is the subtle part, and getting it backwards wastes
hours. `--ckpt_path` restores the *whole trainer state* -- optimizer, learning
rate schedule and epoch counter -- which is what resuming our own interrupted run
needs, and exactly wrong for starting from somebody else's finished voice, where
it would resume at their final epoch with their decayed learning rate. Starting
from a base voice uses `--model.warmstart_ckpt`, which copies matching weights
and begins with a fresh optimizer.
"""

from __future__ import annotations

import json
import logging
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, studiopack

log = logging.getLogger(__name__)

SAMPLE_RATE = 22050          # matches dataset.TARGET_RATE and Piper's medium voices
DEFAULT_ESPEAK_VOICE = "en-us"

# Lightning saves last.ckpt every epoch, so an abrupt kill costs one epoch at
# most. This is what makes stopping cheap enough to offer as a button.
LAST_CHECKPOINT = "last.ckpt"

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


# -- planning ---------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Everything a run needs. Separated from the run so it can be inspected."""

    voice_name: str
    dataset_csv: Path
    work_dir: Path
    base_checkpoint: Path | None = None
    espeak_voice: str = DEFAULT_ESPEAK_VOICE
    sample_rate: int = SAMPLE_RATE
    batch_size: int = 12
    max_epochs: int = 1000
    use_gpu: bool = True

    @property
    def audio_dir(self) -> Path:
        # dataset.prepare() writes wav paths relative to the CSV's own directory.
        return self.dataset_csv.parent

    @property
    def config_path(self) -> Path:
        """Written by the trainer; becomes the exported voice's .onnx.json."""
        return self.work_dir / "config.json"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"


def suggest_batch_size(vram_gb: float) -> int:
    """Batch size that should fit. Conservative: OOM wastes the whole run.

    VITS at 22 kHz costs roughly 0.7 GB per item once the discriminators are
    resident, and the gate lets under-spec cards through deliberately, so this
    has to return something usable for 6 GB rather than assuming a big card.
    """
    if vram_gb >= 24:
        return 32
    if vram_gb >= 16:
        return 24
    if vram_gb >= 12:
        return 16
    if vram_gb >= 8:
        return 12
    return 8


def build_command(cfg: TrainingConfig, resume_from: Path | None = None) -> list[str]:
    """The exact argv for a training run. Pure, so it can be tested."""
    args = [
        str(studiopack.python_exe()), "-m", "piper.train", "fit",
        "--data.voice_name", cfg.voice_name,
        "--data.csv_path", str(cfg.dataset_csv),
        "--data.audio_dir", str(cfg.audio_dir),
        "--data.config_path", str(cfg.config_path),
        "--data.cache_dir", str(cfg.cache_dir),
        "--data.espeak_voice", cfg.espeak_voice,
        "--data.batch_size", str(cfg.batch_size),
        # sample_rate is linked model -> data by VitsLightningCLI, so setting it
        # on the model is enough and setting it on both would conflict.
        "--model.sample_rate", str(cfg.sample_rate),
        "--trainer.max_epochs", str(cfg.max_epochs),
        "--trainer.default_root_dir", str(cfg.work_dir),
        "--trainer.accelerator", "gpu" if cfg.use_gpu else "cpu",
        "--trainer.devices", "1",
    ]
    if resume_from is not None:
        args += ["--ckpt_path", str(resume_from)]
    elif cfg.base_checkpoint is not None:
        args += ["--model.warmstart_ckpt", str(cfg.base_checkpoint)]
    return args


# -- progress ---------------------------------------------------------------


@dataclass
class Progress:
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float | None = None

    @property
    def fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.step / self.total_steps)


# Lightning's progress bar, e.g.
#   Epoch 3:  45%|####5     | 45/100 [00:12<00:15, 3.55it/s, v_num=0, loss=42.1]
_EPOCH = re.compile(r"Epoch\s+(\d+)")
_STEPS = re.compile(r"(\d+)/(\d+)\s*\[")
_LOSS = re.compile(r"\bloss[=\s]+([0-9]*\.?[0-9]+)")


def parse_progress(line: str, previous: Progress | None = None) -> Progress | None:
    """Read one line of trainer output. None when it carries no progress.

    Deliberately tolerant. If a future Lightning changes its bar this returns
    None and the UI simply shows no percentage -- training itself is unaffected,
    and checkpoints on disk remain the authoritative record of what happened.
    """
    epoch = _EPOCH.search(line)
    steps = _STEPS.search(line)
    if not epoch and not steps:
        return None

    result = Progress(**vars(previous)) if previous else Progress()
    if epoch:
        result.epoch = int(epoch.group(1))
    if steps:
        result.step, result.total_steps = int(steps.group(1)), int(steps.group(2))
    loss = _LOSS.search(line)
    if loss:
        try:
            result.loss = float(loss.group(1))
        except ValueError:
            pass
    return result


# -- checkpoints ------------------------------------------------------------

# ModelCheckpoint's filename= template, e.g. "epoch=12-val_mel=0.3141.ckpt".
_CKPT_SCORE = re.compile(r"epoch=(\d+)-val_(mel|mos)=([0-9]*\.?[0-9]+)\.ckpt$")


def checkpoints(work_dir: Path) -> list[Path]:
    """Every checkpoint under a work directory, newest first.

    Searched recursively because Lightning nests them under
    lightning_logs/version_N/checkpoints/, and the version number increments
    each time a run is resumed.
    """
    if not work_dir.is_dir():
        return []
    return sorted(work_dir.rglob("*.ckpt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def resume_point(work_dir: Path) -> Path | None:
    """The checkpoint to hand back to --ckpt_path, or None for a fresh start."""
    found = [p for p in checkpoints(work_dir) if p.name == LAST_CHECKPOINT]
    return found[0] if found else None


def best_checkpoint(work_dir: Path) -> Path | None:
    """The checkpoint most worth exporting.

    Prefers the best val_mel, since that is what the trainer ranks on. Falls
    back to the most recent file so an interrupted run is still exportable --
    last.ckpt is a real checkpoint, just not a chosen one.
    """
    scored: list[tuple[float, Path]] = []
    for path in checkpoints(work_dir):
        match = _CKPT_SCORE.search(path.name)
        if match and match.group(2) == "mel":
            scored.append((float(match.group(3)), path))
    if scored:
        return min(scored)[1]
    everything = checkpoints(work_dir)
    return everything[0] if everything else None


# -- running ----------------------------------------------------------------


class TrainingRun:
    """A training subprocess, its output, and the ability to stop it."""

    def __init__(self, cfg: TrainingConfig, on_progress=None, on_line=None,
                 on_finish=None):
        self.cfg = cfg
        self.progress = Progress()
        self.returncode: int | None = None
        self.error: str = ""
        self.started = 0.0
        self._on_progress = on_progress
        self._on_line = on_line
        self._on_finish = on_finish
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._tail: list[str] = []
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def elapsed(self) -> float:
        return time.time() - self.started if self.started else 0.0

    def start(self, resume: bool = True) -> None:
        if self.running:
            raise RuntimeError("This run is already going.")

        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        resume_from = resume_point(self.cfg.work_dir) if resume else None
        argv = build_command(self.cfg, resume_from)
        log.info("training %s (%s)", self.cfg.voice_name,
                 f"resuming from {resume_from.name}" if resume_from else "from scratch")

        self.started = time.time()
        self._stopping = False
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=studiopack.environment_for_training(),
            cwd=str(self.cfg.work_dir),
            # A new process group is what makes a graceful stop possible at all
            # on Windows: CTRL_BREAK_EVENT can only be sent to a group.
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None
        for raw in self._proc.stdout or []:
            # The progress bar redraws with \r, so one read can hold many frames.
            for line in raw.replace("\r", "\n").splitlines():
                line = line.strip()
                if not line:
                    continue
                self._tail.append(line)
                del self._tail[:-60]
                if self._on_line:
                    self._on_line(line)
                update = parse_progress(line, self.progress)
                if update is not None:
                    self.progress = update
                    if self._on_progress:
                        self._on_progress(update)

        self.returncode = self._proc.wait()
        if self.returncode != 0 and not self._stopping:
            self.error = "\n".join(self._tail[-15:]) or "training exited unexpectedly"
            log.error("training failed (%s):\n%s", self.returncode, self.error)
        if self._on_finish:
            self._on_finish(self)

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the trainer to stop, then insist.

        Losing this process is not a disaster -- last.ckpt is written every
        epoch -- so this escalates rather than waiting indefinitely.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._stopping = True
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=timeout)
            return
        except Exception as exc:  # noqa: BLE001 - any failure escalates below
            log.debug("graceful stop did not take (%s); terminating", exc)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    def wait(self, timeout: float | None = None) -> int | None:
        if self._thread:
            self._thread.join(timeout)
        return self.returncode


# -- export -----------------------------------------------------------------


@dataclass
class Provenance:
    """Where a voice came from. Written beside it, and never guessed at later."""

    voice_name: str
    created: float = field(default_factory=time.time)
    app_version: str = __version__
    base_checkpoint: str = ""
    base_licence: str = ""
    dataset_clips: int = 0
    dataset_seconds: float = 0.0
    epochs: int = 0
    sample_rate: int = SAMPLE_RATE
    checkpoint: str = ""
    trained_locally: bool = True

    def to_dict(self) -> dict:
        return dict(vars(self))


def export(checkpoint: Path, config_json: Path, dest_dir: Path, voice_name: str,
           provenance: Provenance | None = None, timeout: float = 900.0) -> Path:
    """Turn a checkpoint into an installed voice. Returns the .onnx path.

    A Piper voice is two files that must agree: the weights, and the config
    naming the phoneme ids they were trained against. Writing the .onnx without
    its config produces a voice that loads and then speaks nonsense, so the
    config is copied in the same step and the .onnx is only moved into place
    once both exist.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")
    if not config_json.is_file():
        raise FileNotFoundError(
            f"No voice config at {config_json}. It is written by training, so a "
            "missing one means the run never got past preparing the dataset."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{voice_name}.onnx"
    staging = dest_dir / f"{voice_name}.onnx.partial"

    argv = [str(studiopack.python_exe()), "-m", "piper.train.export_onnx",
            "--checkpoint", str(checkpoint), "--output-file", str(staging)]
    log.info("exporting %s -> %s", checkpoint.name, final.name)
    result = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=studiopack.environment_for_training(), timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )
    if result.returncode != 0 or not staging.exists():
        staging.unlink(missing_ok=True)
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError("Export failed:\n" + "\n".join(tail))

    final.with_suffix(".onnx.json").write_text(
        config_json.read_text(encoding="utf-8"), encoding="utf-8")
    staging.replace(final)

    if provenance is not None:
        provenance.checkpoint = checkpoint.name
        final.with_suffix(".onnx.provenance.json").write_text(
            json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")

    log.info("exported %s (%.1f MB)", final.name, final.stat().st_size / 1e6)
    return final
