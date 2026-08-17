# Security

## Reporting

Report vulnerabilities through the repository's private vulnerability reporting, or
by opening an issue if the problem is not sensitive. Please include the version from
Settings → Updates and the output of **Copy diagnostics**, which redacts your
Windows user name.

## What this application does

Voice2TTS processes audio entirely on your machine. Recordings, transcripts and
synthesized speech are never uploaded anywhere.

### Network connections

| Host | When | Why |
|---|---|---|
| `api.github.com` | On start, throttled; or manual check | Read the latest release version |
| `github.com` | When you accept an update | Download the installer |
| `huggingface.co` | When you download a voice, or first run without a bundled model | Fetch Piper voices and Whisper weights |
| `vb-audio.com` | Only when you choose to install the virtual cable | Fetch VB-CABLE |
| `pypi.org` | Only when you opt in to GPU acceleration | Fetch NVIDIA's CUDA libraries |

Everything except the update check is user-initiated. Update checks can be disabled
in Settings → Updates, either with the checkbox or by clearing the repository field.
No credentials, tokens or identifiers are sent with any request.

### Local data

| Path | Contents |
|---|---|
| `%APPDATA%\Voice2TTS` | Settings and log; downloaded voices |
| `%LOCALAPPDATA%\Voice2TTS` | Whisper models, CUDA libraries, downloaded updates |

The log records transcribed text at INFO level so you can debug recognition
problems. If that matters to you, set `log_level = "WARNING"` in `config.toml`.
**Copy diagnostics** includes the last 60 log lines, which may therefore contain
transcripts — review before pasting it anywhere public.

## Things worth knowing

**The installer is not code-signed.** Windows SmartScreen will warn on first run,
and antivirus software may flag it: an unsigned executable that downloads another
executable and runs it elevated matches a common malware pattern. Verify the
download against the published `.sha256` before installing.

**Updates are verified but not signed.** The updater checks the downloaded
installer's size and SHA-256 against the release's `.sha256` asset, and refuses
anything that is not a Windows executable. That protects against corruption and
tampering in transit, but the trust root is GitHub's TLS and your repository's
access controls — not a code-signing certificate.

**Installing the virtual cable requires administrator rights.** Voice2TTS never
elevates itself. It downloads VB-CABLE from VB-Audio and launches *their* installer,
which raises the UAC prompt. Nothing is installed silently, and you can always do
this step yourself from <https://vb-audio.com/Cable/>.

**The global hotkey is a low-level keyboard hook** and is not suppressed, so
keypresses still reach the focused application. Voice2TTS does not log keystrokes;
only the configured combination is acted on.
