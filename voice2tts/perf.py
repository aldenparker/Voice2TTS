"""Where is this process spending CPU, right now?

"Something is using the CPU" is not a bug report anyone can act on, and the
usual tools stop at the process boundary: Task Manager will say Voice2TTS is at
300%, but not whether that is the recogniser, an audio callback, ONNX Runtime's
thread pool, or a timer in the interface.

This samples the process's own threads over a short window and attributes the
time. Python threads are named; native ones (PortAudio, ONNX Runtime, CUDA)
report their start address instead, which is still enough to tell "one runaway
loop" apart from "sixteen worker threads doing their job".

Windows only for now, and it degrades to a total rather than raising -- the
diagnostics report has to work everywhere.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

TH32CS_SNAPTHREAD = 0x00000004
THREAD_QUERY_LIMITED_INFORMATION = 0x0800


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


def _as_seconds(value: _FILETIME) -> float:
    return ((value.high << 32) | value.low) / 1e7


@dataclass
class Snapshot:
    seconds: float = 0.0
    process_cpu: float = 0.0          # seconds of CPU across the window
    threads: list[tuple[str, float]] = field(default_factory=list)
    cores: int = field(default_factory=lambda: os.cpu_count() or 1)
    supported: bool = True

    @property
    def busy(self) -> float:
        """Fraction of ONE core the process used. 1.0 means a core is pegged."""
        return self.process_cpu / self.seconds if self.seconds else 0.0

    def report(self) -> list[str]:
        if not self.supported:
            return ["CPU sampling is only implemented on Windows."]
        lines = [
            f"Process CPU        : {self.busy * 100:.0f}% of one core "
            f"({self.busy / self.cores * 100:.1f}% of {self.cores}), "
            f"sampled over {self.seconds:.1f}s",
            f"Threads            : {len(self.threads)}",
        ]
        hot = [(name, share) for name, share in self.threads
               if share / max(self.seconds, 1e-9) > 0.02]
        if not hot:
            lines.append("No thread used more than 2% of a core "
                         "-- the process is idle.")
            return lines
        lines.append("Busiest threads:")
        for name, share in hot[:12]:
            lines.append(f"  {share / self.seconds * 100:5.0f}% of a core  {name}")
        return lines


def stacks(skip_main: bool = False) -> list[str]:
    """Where every thread is, right now.

    The counterpart to sample(): that says which thread is busy, this says what
    it is doing. Together they cover the two ways this app goes wrong -- burning
    CPU, and stopping dead. A hang reported as "it got stuck" is not actionable;
    a hang reported with the frame it stopped in is.
    """
    import traceback

    names = {t.ident: t.name for t in threading.enumerate()}
    lines: list[str] = []
    for ident, frame in sys._current_frames().items():
        name = names.get(ident, f"thread-{ident}")
        if skip_main and name == "MainThread":
            continue
        lines.append(f"  {name} (id {ident})")
        for entry in traceback.extract_stack(frame)[-8:]:
            where = Path(entry.filename).name
            lines.append(f"      {where}:{entry.lineno} {entry.name}()")
            if entry.line:
                lines.append(f"          {entry.line.strip()[:100]}")
    return lines


def _python_thread_names() -> dict[int, str]:
    return {t.native_id: t.name for t in threading.enumerate()
            if t.native_id is not None}


def _kernel32():
    """kernel32 with real prototypes.

    Without these, ctypes assumes every function returns a C int, which
    truncates 64-bit HANDLEs to 32 bits. The calls then operate on a handle
    that is not the one they were given, which fails in ways that look like
    anything except a missing declaration.
    """
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    lib.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    lib.Thread32First.restype = wintypes.BOOL
    lib.Thread32First.argtypes = [wintypes.HANDLE,
                                  ctypes.POINTER(_THREADENTRY32)]
    lib.Thread32Next.restype = wintypes.BOOL
    lib.Thread32Next.argtypes = [wintypes.HANDLE,
                                 ctypes.POINTER(_THREADENTRY32)]
    lib.OpenThread.restype = wintypes.HANDLE
    lib.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    lib.GetThreadTimes.restype = wintypes.BOOL
    lib.GetThreadTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(_FILETIME)] * 4
    lib.CloseHandle.restype = wintypes.BOOL
    lib.CloseHandle.argtypes = [wintypes.HANDLE]
    return lib


def _thread_ids(kernel32) -> list[int]:
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    try:
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(_THREADENTRY32)
        pid = os.getpid()
        found: list[int] = []
        if not kernel32.Thread32First(snap, ctypes.byref(entry)):
            return []
        while True:
            if entry.th32OwnerProcessID == pid:
                found.append(int(entry.th32ThreadID))
            if not kernel32.Thread32Next(snap, ctypes.byref(entry)):
                break
        return found
    finally:
        kernel32.CloseHandle(snap)


def _thread_cpu(kernel32, tid: int) -> float | None:
    handle = kernel32.OpenThread(THREAD_QUERY_LIMITED_INFORMATION, False, tid)
    if not handle:
        return None
    try:
        creation, exit_, kernel, user = (_FILETIME() for _ in range(4))
        if not kernel32.GetThreadTimes(handle, ctypes.byref(creation),
                                       ctypes.byref(exit_), ctypes.byref(kernel),
                                       ctypes.byref(user)):
            return None
        return _as_seconds(kernel) + _as_seconds(user)
    finally:
        kernel32.CloseHandle(handle)


def sample(seconds: float = 1.0) -> Snapshot:
    """Measure per-thread CPU over `seconds`. Blocks for that long."""
    if os.name != "nt":
        return Snapshot(supported=False)
    try:
        kernel32 = _kernel32()
        before = {tid: _thread_cpu(kernel32, tid) for tid in _thread_ids(kernel32)}
        start = time.perf_counter()
        time.sleep(seconds)
        elapsed = time.perf_counter() - start
        after = {tid: _thread_cpu(kernel32, tid) for tid in _thread_ids(kernel32)}
    except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
        log.debug("cpu sampling failed: %s", exc)
        return Snapshot(supported=False)

    names = _python_thread_names()
    used: list[tuple[str, float]] = []
    total = 0.0
    for tid, end in after.items():
        begin = before.get(tid)
        if begin is None or end is None:
            continue
        delta = max(0.0, end - begin)
        total += delta
        used.append((names.get(tid, f"native thread {tid}"), delta))

    used.sort(key=lambda item: -item[1])
    return Snapshot(seconds=elapsed, process_cpu=total, threads=used)
