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
from .net import user_agent

log = logging.getLogger(__name__)

CABLE_PAGE = "https://vb-audio.com/Cable/"
DONATE_PAGE = "https://vb-audio.com/Services/licensing.htm"

# Used only if scraping the page fails. Pack numbers move -- this was 43 when first
# written and 45 by the time the live test ran, which is why scraping comes first.
PINNED_ZIP = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"

_ZIP_LINK = re.compile(
    r"""https?://[^\s"'<>]*?VBCABLE_Driver_Pack\d*\.zip""", re.IGNORECASE
)

USER_AGENT = user_agent("https://vb-audio.com/Cable/")

# Windows names virtual audio endpoints "<FriendlyName> (<DriverName>)", and the
# driver name is IDENTICAL on the playback and recording side of the same virtual
# cable. That parenthetical is therefore the reliable way to pair the device we play
# into with the device the user selects in Discord -- far better than guessing at
# friendly names, which differ per product and change between driver versions.
#
#   VBMatrix In 8 (VB-Audio Matrix VAIO)  <-> VBMatrix Out 8 (VB-Audio Matrix VAIO)
#   CABLE Input (VB-Audio Virtual Cable)  <-> CABLE Output (VB-Audio Virtual Cable)
#   VoiceMeeter Input (VB-Audio VoiceMeeter VAIO) <-> VoiceMeeter Out B1 (same tag)
#
# Matched against the driver tag, longest pattern first so "voicemeeter aux vaio"
# beats "voicemeeter vaio". rank orders which product we pick when several exist.
# kind distinguishes two very different things that look alike in Windows:
#
#   CABLE  - a hardwired loop in the driver. Whatever is played into the playback
#            endpoint comes straight out of the recording endpoint. Works with no
#            application running, and the pairing is guaranteed.
#   ROUTER - a mixer whose endpoints are ports, not a loop. Audio played into
#            "VBMatrix In 1" goes INTO the Matrix application and only reaches
#            "VBMatrix Out 1" if the user has routed it there. Nothing passes at
#            all while the application is closed.
#
# Treating a router as a cable produces confident, wrong advice: it names a
# recording device for Discord that may carry nothing.
CABLE, ROUTER = "cable", "router"

_PRODUCTS: tuple[tuple[str, str, int, str], ...] = (
    ("vb-audio virtual cable", "VB-CABLE", 0, CABLE),
    ("vb-audio cable a", "VB-CABLE A+B", 1, CABLE),
    ("vb-audio cable b", "VB-CABLE A+B", 1, CABLE),
    ("vb-audio cable c", "VB-CABLE C+D", 1, CABLE),
    ("vb-audio cable d", "VB-CABLE C+D", 1, CABLE),
    ("vb-audio hi-fi cable", "VB-Audio Hi-Fi Cable", 2, CABLE),
    ("virtual audio cable", "Virtual Audio Cable (VAC)", 3, CABLE),
    ("vb-audio matrix vaio", "VB-Audio Matrix", 6, ROUTER),
    ("vb-audio voicemeeter vaio3", "VoiceMeeter Potato", 7, ROUTER),
    ("vb-audio voicemeeter aux vaio", "VoiceMeeter Banana", 7, ROUTER),
    ("vb-audio voicemeeter vaio", "VoiceMeeter", 7, ROUTER),
    ("synchronous audio router", "Synchronous Audio Router", 8, ROUTER),
)

# Process names for the router applications, so we can say when one is installed
# but not running -- in which case its endpoints carry nothing at all.
_ROUTER_PROCESSES: dict[str, tuple[str, ...]] = {
    "VB-Audio Matrix": ("vbaudiomatrix", "vbaudiomatrix_x64"),
    "VoiceMeeter": ("voicemeeter",),
    "VoiceMeeter Banana": ("voicemeeterpro", "voicemeeter"),
    "VoiceMeeter Potato": ("voicemeeter8", "voicemeeter8x64", "voicemeeter"),
    "Synchronous Audio Router": ("sar",),
}

# Anything from these families counts as a virtual cable even when the exact product
# is unrecognised -- VB-Audio ship new ones, and a device we cannot name is still
# perfectly usable. Better to detect it generically than to report nothing.
# Deliberately specific. A bare "virtual audio" would match NVIDIA's "NVIDIA Virtual
# Audio Device", which renders audio but has no capture side, so it can never carry
# speech to Discord.
_FAMILY_HINTS = (
    "vb-audio", "voicemeeter", "vaio", "vbmatrix",
    "virtual cable", "virtual audio cable",
)

_GENERIC_RANK = 9

_DRIVER_TAG = re.compile(r"\(([^()]+)\)\s*$")
_TRAILING_NUMBER = re.compile(r"(\d+)\s*$")

MIN_ZIP_BYTES = 200_000       # a valid driver pack is ~1-3 MB
MAX_ZIP_BYTES = 60_000_000    # sanity ceiling on what we will download


@dataclass(frozen=True)
class CableInfo:
    product: str
    output_name: str      # what we play into
    input_name: str       # what Discord listens to
    channel: int | None = None   # e.g. 8 for "VBMatrix In 8"
    certain: bool = True  # False when the Discord-side pairing had to be guessed
    kind: str = CABLE

    @property
    def label(self) -> str:
        return f"{self.product} {self.channel}" if self.channel else self.product

    @property
    def is_router(self) -> bool:
        return self.kind == ROUTER

    @property
    def discord_input(self) -> str:
        return self.input_name or "(check Windows Sound settings)"

    @property
    def app_running(self) -> bool:
        """For a router, whether its application is running. True for a cable."""
        return True if not self.is_router else router_running(self.product)

    @property
    def caveat(self) -> str:
        """What the user needs to know beyond the device names, or ''."""
        if not self.is_router:
            return "" if self.certain else (
                "The matching recording device was inferred; verify it in Windows "
                "Sound settings."
            )
        note = (
            f"{self.product} is an audio router, not a fixed cable. Speech sent to "
            f"{self.output_name} reaches Discord only if {self.product} is running "
            f"and routing it to {self.discord_input}."
        )
        if not self.app_running:
            note = f"{self.product} is not running, so nothing will pass. " + note
        return note

    @property
    def summary(self) -> str:
        tail = f"  [{self.caveat.splitlines()[0]}]" if self.caveat else ""
        return (f"{self.label}: play to {self.output_name!r}, "
                f"Discord selects {self.discord_input!r}{tail}")


def router_running(product: str) -> bool:
    """True if the given router application currently has a process."""
    names = _ROUTER_PROCESSES.get(product)
    if not names:
        return False
    try:
        import subprocess

        out = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=10,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        ).stdout.lower()
    except Exception as exc:  # noqa: BLE001 - absence of proof is not proof
        log.debug("could not list processes: %s", exc)
        return False
    return any(f'"{n}.exe"' in out for n in names)


def _driver_tag(name: str) -> str:
    """The trailing '(...)' driver name, which both sides of a cable share."""
    match = _DRIVER_TAG.search(name)
    return match.group(1).strip().lower() if match else ""


def _friendly(name: str) -> str:
    """The part before the driver tag, e.g. 'VBMatrix In 8'."""
    return _DRIVER_TAG.sub("", name).strip()


def _channel(name: str) -> int | None:
    match = _TRAILING_NUMBER.search(_friendly(name))
    return int(match.group(1)) if match else None


def _identify(name: str) -> tuple[str, int, str] | None:
    """Map a device to (product, rank, kind), or None if it is not virtual."""
    tag = _driver_tag(name)
    haystack = f"{tag} {name}".lower()
    for pattern, product, rank, kind in _PRODUCTS:
        if pattern in haystack:
            return product, rank, kind
    if any(hint in haystack for hint in _FAMILY_HINTS):
        # Unrecognised but clearly from a virtual-audio family; name it after its
        # own driver so the user can still tell which device this is. Assume a
        # plain cable, since that is the case where our advice is safe.
        return (tag.title() if tag else "Virtual audio device"), _GENERIC_RANK, CABLE
    return None


def _pair_input(out: devices.Device, inputs: list[devices.Device]) -> tuple[str, bool]:
    """Find the recording endpoint matching a playback endpoint.

    Returns (name, certain). Certainty matters: telling someone to select the wrong
    device in Discord wastes more time than admitting we are unsure.
    """
    tag = _driver_tag(out.name)
    same_driver = [d for d in inputs if tag and _driver_tag(d.name) == tag]

    if same_driver:
        # Multi-channel devices (Matrix has 8) must match on the channel number:
        # "VBMatrix In 8" pairs with "VBMatrix Out 8", not whichever comes first.
        channel = _channel(out.name)
        if channel is not None:
            numbered = [d for d in same_driver if _channel(d.name) == channel]
            if len(numbered) == 1:
                return numbered[0].name, True
            if numbered:
                return numbered[0].name, False
        if len(same_driver) == 1:
            return same_driver[0].name, True
        # Same driver, several candidates, no usable numbering: pick the first but
        # say it is a guess.
        return same_driver[0].name, False

    # No driver tag to match on (some products omit it). Fall back to the historical
    # Input->Output rename, then to a shared leading word.
    low = out.name.lower()
    for guess in (low.replace("input", "output"), low.replace(" in ", " out ")):
        exact = next((d for d in inputs if d.name.lower() == guess), None)
        if exact:
            return exact.name, True
    head = _friendly(out.name).lower().split()
    if head:
        partial = next((d for d in inputs if d.name.lower().startswith(head[0])), None)
        if partial:
            return partial.name, False
    return "", False


def list_devices() -> list[CableInfo]:
    """Every usable virtual cable, best first.

    Ordered by product rank then channel number, so a machine with eight Matrix
    channels defaults to channel 1 rather than whichever Windows enumerated first.
    """
    # WASAPI only, on BOTH sides. The same endpoint is also exposed through MME,
    # DirectSound and WDM-KS, so an unfiltered input list contains four copies of
    # every device -- enough to defeat the "exactly one match" pairing test and make
    # every correct pairing look uncertain. MME also truncates names to 31
    # characters, which destroys the driver tag we pair on.
    inputs = [d for d in devices.list_inputs() if d.hostapi == devices.WASAPI]
    found: list[tuple[int, int, CableInfo]] = []
    for out in devices.list_outputs():
        if out.hostapi != devices.WASAPI:
            continue
        identified = _identify(out.name)
        if identified is None:
            continue
        product, rank, kind = identified

        if rank == _GENERIC_RANK:
            # Only a recognised product is taken on trust. Anything matched by a
            # family hint alone must prove it is a cable by having a recording
            # endpoint on the same driver -- a loopback device has both halves,
            # while a render-only device (NVIDIA's virtual audio, game streaming
            # sinks) would silently swallow speech Discord could never hear.
            tag = _driver_tag(out.name)
            if not tag or not any(_driver_tag(d.name) == tag for d in inputs):
                log.debug("ignoring render-only virtual device: %s", out.name)
                continue

        partner, certain = _pair_input(out, inputs)
        channel = _channel(out.name)
        found.append((
            rank,
            channel if channel is not None else 0,
            CableInfo(
                product=product, output_name=out.name, input_name=partner,
                channel=channel,
                # A router's pairing is never certain: the endpoints are ports on a
                # mixer, so which recording device carries this audio depends on the
                # user's routing, not on the naming.
                certain=certain and kind == CABLE,
                kind=kind,
            ),
        ))
    found.sort(key=lambda t: (t[0], t[1], t[2].output_name))
    return [info for _, _, info in found]


def detect() -> CableInfo | None:
    """The virtual cable to use by default, or None if none is installed."""
    candidates = list_devices()
    return candidates[0] if candidates else None


def installed() -> bool:
    return detect() is not None


# Recognises a virtual device from a *fragment* too, since config stores substrings
# like "CABLE Input" rather than full device names, and the placeholder written when
# no cable is present has no driver tag to identify.
_MATCH_HINTS = (
    "vb-audio", "voicemeeter", "vaio", "vbmatrix", "hi-fi cable",
    "cable input", "cable output", "cable-a", "cable-b", "cable-c", "cable-d",
    "virtual cable", "virtual audio", "synchronous audio router",
)


def is_virtual_device(name: str) -> bool:
    """True if this device name -- or config match fragment -- is a virtual cable.

    Used to find the row in the output list that represents the cable, so changing
    which one is selected updates that row instead of appending another.
    """
    if not name:
        return False
    if _identify(name) is not None:
        return True
    return any(hint in name.lower() for hint in _MATCH_HINTS)


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
