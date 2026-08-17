"""Virtual audio cable detection and assisted installation.

Only a kernel-mode driver can create an audio endpoint that Discord will see --
Windows enumerates endpoints through MMDevice, which user-mode code cannot add to.
Shipping our own driver would need an EV certificate issued to a registered business
plus Microsoft attestation signing, so this module instead:

  1. detects any compatible cable the user already has (many people do), then
  2. fetches VB-CABLE from VB-Audio's own servers and runs THEIR installer elevated.

We never redistribute VB-CABLE: VB-Audio route bundling through a negotiated
agreement. Downloading from the vendor at the user's request is a bootstrapper, not
redistribution. The donation notice is shown rather than buried -- they do the hard
part here.

Nothing in here installs silently. An unsigned program that downloads an executable
and elevates it is indistinguishable from malware if it does so quietly, so every
step is user-initiated and visible.
"""

from __future__ import annotations

import ctypes
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import devices

log = logging.getLogger(__name__)

CABLE_PAGE = "https://vb-audio.com/Cable/"
DONATE_PAGE = "https://vb-audio.com/Services/licensing.htm"

# Used only if scraping the page fails. Pack numbers move -- this was 43 when first
# written and 45 by the time the live test ran, which is why scraping comes first.
PINNED_ZIP = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"

_ZIP_LINK = re.compile(
    r"""https?://[^\s"'<>]*?VBCABLE_Driver_Pack\d*\.zip""", re.IGNORECASE
)

USER_AGENT = "Voice2TTS/0.2 (+https://vb-audio.com/Cable/)"

# Devices that can act as a loopback sink. Checked as case-insensitive substrings of
# the OUTPUT device name; the matching input side is what Discord then selects.
KNOWN_CABLES: tuple[tuple[str, str], ...] = (
    ("cable input", "VB-CABLE"),
    ("cable-a input", "VB-CABLE A+B"),
    ("cable-b input", "VB-CABLE A+B"),
    ("cable-c input", "VB-CABLE C+D"),
    ("cable-d input", "VB-CABLE C+D"),
    ("voicemeeter input", "VoiceMeeter"),
    ("voicemeeter aux input", "VoiceMeeter Banana"),
    ("voicemeeter vaio3 input", "VoiceMeeter Potato"),
    ("hi-fi cable input", "VB-Audio Hi-Fi Cable"),
    ("virtual audio cable", "Virtual Audio Cable (VAC)"),
    ("line 1 (virtual audio cable)", "Virtual Audio Cable (VAC)"),
    ("synchronous audio router", "Synchronous Audio Router"),
)

MIN_ZIP_BYTES = 200_000       # a valid driver pack is ~1-3 MB
MAX_ZIP_BYTES = 60_000_000    # sanity ceiling on what we will download


@dataclass(frozen=True)
class CableInfo:
    product: str
    output_name: str    # what we play into
    input_name: str     # what Discord listens to

    @property
    def summary(self) -> str:
        return f"{self.product}: play to {self.output_name!r}, Discord selects {self.input_name!r}"


def detect() -> CableInfo | None:
    """Find any already-installed virtual cable, not just VB-CABLE."""
    outputs = devices.list_outputs()
    inputs = devices.list_inputs()
    for needle, product in KNOWN_CABLES:
        out = next((d for d in outputs if needle in d.name.lower()), None)
        if out is None:
            continue
        # The matching capture endpoint is usually the same name with Input->Output.
        partner = _partner_input(out.name, inputs)
        return CableInfo(product=product, output_name=out.name,
                         input_name=partner or "(check Sound settings)")
    return None


def _partner_input(output_name: str, inputs: list[devices.Device]) -> str | None:
    low = output_name.lower()
    guess = low.replace("input", "output")
    exact = next((d for d in inputs if d.name.lower() == guess), None)
    if exact:
        return exact.name
    # Fall back to sharing the leading token, e.g. "CABLE Output (VB-Audio ...)".
    head = low.split("(")[0].strip().split()[0]
    partial = next((d for d in inputs if d.name.lower().startswith(head)), None)
    return partial.name if partial else None


def installed() -> bool:
    return detect() is not None


# -- download ---------------------------------------------------------------


def resolve_download_url(timeout: float = 15.0) -> tuple[str, str]:
    """Return (url, source) for the driver zip.

    Scraping first because the pinned URL carries a pack number that moves.
    """
    req = urllib.request.Request(CABLE_PAGE, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(400_000).decode("utf-8", errors="replace")
        match = _ZIP_LINK.search(html)
        if match:
            log.info("resolved cable download from page: %s", match.group(0))
            return match.group(0), "vb-audio.com"
    except Exception as exc:  # noqa: BLE001 - fall through to the pin
        log.warning("could not scrape %s: %s", CABLE_PAGE, exc)
    log.info("using pinned cable download URL")
    return PINNED_ZIP, "pinned fallback"


def download(url: str, dest_dir: Path, progress=None, timeout: float = 60.0) -> Path:
    """Fetch the driver zip. Raises on anything that does not look like one."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "VBCABLE_Driver_Pack.zip"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total and total > MAX_ZIP_BYTES:
            raise RuntimeError(f"download is implausibly large ({total} bytes)")
        got = 0
        with dest.open("wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                got += len(chunk)
                if got > MAX_ZIP_BYTES:
                    raise RuntimeError("download exceeded size limit")
                fh.write(chunk)
                if progress:
                    progress(got, total)

    if dest.stat().st_size < MIN_ZIP_BYTES:
        raise RuntimeError(f"downloaded file is too small ({dest.stat().st_size} bytes)")
    if not zipfile.is_zipfile(dest):
        raise RuntimeError("downloaded file is not a zip archive")
    return dest


def extract(zip_path: Path, dest_dir: Path) -> Path:
    """Unpack and return the x64 setup executable."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not any(n.lower().endswith("vbcable_setup_x64.exe") for n in names):
            raise RuntimeError("archive does not contain VBCABLE_Setup_x64.exe")
        for member in names:
            # Refuse absolute paths and traversal before writing anything.
            if member.startswith("/") or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(dest_dir)

    setup = next(
        (p for p in dest_dir.rglob("*") if p.name.lower() == "vbcable_setup_x64.exe"),
        None,
    )
    if setup is None:
        raise RuntimeError("VBCABLE_Setup_x64.exe not found after extraction")
    return setup


# -- install ----------------------------------------------------------------


def run_installer_elevated(setup: Path, uninstall: bool = False) -> bool:
    """Launch VB-Audio's installer with a UAC prompt. Returns False if declined.

    ShellExecuteW with the 'runas' verb is what raises the consent dialog; there is
    no way to install a driver without it, and no attempt is made to hide it.
    """
    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SW_SHOWNORMAL = 1

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("fMask", ctypes.c_ulong),
            ("hwnd", ctypes.c_void_p),
            ("lpVerb", ctypes.c_wchar_p),
            ("lpFile", ctypes.c_wchar_p),
            ("lpParameters", ctypes.c_wchar_p),
            ("lpDirectory", ctypes.c_wchar_p),
            ("nShow", ctypes.c_int),
            ("hInstApp", ctypes.c_void_p),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", ctypes.c_wchar_p),
            ("hkeyClass", ctypes.c_void_p),
            ("dwHotKey", ctypes.c_ulong),
            ("hIcon", ctypes.c_void_p),
            ("hProcess", ctypes.c_void_p),
        ]

    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(setup)
    # VB-Audio's setup takes -u to uninstall; with no argument it installs.
    info.lpParameters = "-u" if uninstall else None
    info.lpDirectory = str(setup.parent)
    info.nShow = SW_SHOWNORMAL

    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        err = ctypes.get_last_error()
        log.warning("elevation declined or failed (error %s)", err)
        return False

    if info.hProcess:
        # VB-Audio's installer is interactive; wait for the user to finish with it.
        ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
        ctypes.windll.kernel32.CloseHandle(info.hProcess)
    return True


def wait_for_device(timeout: float = 30.0, interval: float = 2.0) -> CableInfo | None:
    """Poll for the endpoint to appear. Often it will not until after a reboot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        devices.refresh()
        found = detect()
        if found is not None:
            return found
        time.sleep(interval)
    return None


def reboot_pending() -> bool:
    """True if Windows is holding a pending file-rename or component servicing op."""
    import winreg

    checks = [
        (r"SYSTEM\CurrentControlSet\Control\Session Manager", "PendingFileRenameOperations"),
        (r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired", None),
        (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending", None),
    ]
    for subkey, value in checks:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as key:
                if value is None:
                    return True
                winreg.QueryValueEx(key, value)
                return True
        except OSError:
            continue
    return False


def request_reboot(delay_seconds: int = 10) -> bool:
    """Ask Windows to restart. Always confirm with the user before calling this."""
    try:
        subprocess.run(
            ["shutdown", "/r", "/t", str(delay_seconds), "/c",
             "Restarting to finish installing the virtual audio cable (Voice2TTS)."],
            check=True, capture_output=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("could not schedule reboot: %s", exc)
        return False


def uninstall_flow(progress=None) -> tuple[bool, str]:
    """Run VB-Audio's uninstaller. Returns (needs_reboot, message).

    Same download as install: VB-Audio ship one executable that both installs and
    removes, and we do not keep a copy around after installing.
    """
    def say(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    if detect() is None:
        return False, "No virtual cable is installed."

    workdir = Path(tempfile.mkdtemp(prefix="voice2tts-cable-"))
    try:
        url, source = resolve_download_url()
        say(f"Fetching the VB-CABLE uninstaller from {source}...")
        zip_path = download(url, workdir)
        setup = extract(zip_path, workdir / "unpacked")

        say("Launching VB-Audio's uninstaller (approve the Windows prompt)...")
        if not run_installer_elevated(setup, uninstall=True):
            raise RuntimeError("Administrator approval was declined")

        devices.refresh()
        if detect() is None:
            return False, "Removed."
        return True, "Uninstalled. A restart is needed to remove the device."
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def install_flow(progress=None) -> tuple[bool, str]:
    """Full download-extract-install sequence. Returns (needs_reboot, message).

    Raises on failure so the caller can offer the manual path.
    """
    def say(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    workdir = Path(tempfile.mkdtemp(prefix="voice2tts-cable-"))
    try:
        url, source = resolve_download_url()
        say(f"Downloading VB-CABLE from {source}...")
        zip_path = download(url, workdir, progress=None)

        say("Extracting...")
        setup = extract(zip_path, workdir / "unpacked")

        say("Launching VB-Audio's installer (approve the Windows prompt)...")
        if not run_installer_elevated(setup):
            raise RuntimeError("Administrator approval was declined")

        say("Checking for the new device...")
        found = wait_for_device(timeout=20)
        if found is not None:
            return False, f"Installed. {found.summary}"
        return True, "Installed. A restart is needed before the device appears."
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
