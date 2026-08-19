"""Logging to both console and a rotating file in the user data directory."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading

from .paths import log_path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def setup_logging(level: str = "INFO", console: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")

    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # read-only profile: console logging alone is fine

    # A windowed (pythonw / --noconsole) build has no usable stderr.
    if console and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # These are chatty at DEBUG and say nothing useful about our own behaviour.
    for noisy in ("numba", "onnxruntime", "faster_whisper", "huggingface_hub", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    install_crash_handlers()


def install_crash_handlers() -> None:
    """Route unhandled exceptions to the log instead of a stderr that may not exist.

    A windowed (--noconsole) build has no stderr, so without these an unhandled
    exception in the main thread or, worse, in a worker thread vanishes completely:
    the thread dies, the app keeps running, and the only symptom is that something
    silently stopped working.
    """
    log = logging.getLogger("voice2tts.crash")

    def handle_exception(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical("unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    def handle_thread_exception(args) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        log.critical(
            "unhandled exception in thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception

    # Exceptions escaping a Tk callback otherwise print to a console nobody sees.
    try:
        import tkinter

        def report_callback_exception(self, exc, val, tb) -> None:
            log.critical("unhandled exception in Tk callback", exc_info=(exc, val, tb))

        # Tk calls this with (exc, val, tb) plus self; the stub declares the
        # three-argument sys.excepthook shape.
        # Tk calls this with self plus the (exc, val, tb) triple; the stub
        # declares the three-argument sys.excepthook shape.
        tkinter.Tk.report_callback_exception = (
            report_callback_exception)  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 - tkinter may be unavailable in a CLI-only run
        pass
