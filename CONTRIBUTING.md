# Contributing

## Setup

```powershell
.\setup.ps1
```

Creates a virtualenv at `%USERPROFILE%\.venvs\voice2tts` — deliberately outside the
project, because it usually lives in OneDrive and syncing a multi-gigabyte venv
causes file locks and long stalls.

## Before you push

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\selftest.py
```

```powershell
& "$env:USERPROFILE\.venvs\voice2tts\Scripts\python.exe" scripts\guitest.py
```

```powershell
ruff check .
```

CI runs all three on every push. `selftest.py` takes flags: `--no-audio` (skip
device-dependent checks), `--no-network` (skip live GitHub/HuggingFace calls),
`--no-e2e` (skip the full pipeline run).

## Licence

Voice2TTS is **GPL-3.0-or-later**, because it bundles Piper, which is GPL because it
links eSpeak NG. Contributions are accepted under the same licence.

Before adding a dependency, check its licence. Anything strong-copyleft is already
accounted for; anything incompatible with GPL-3 (for example a proprietary or
GPL-incompatible licence) cannot go in. Record new dependencies in `LICENSE.txt`.

## Where things live

| Concern | File |
|---|---|
| Orchestration and threading | `voice2tts/pipeline.py` |
| Audio in | `voice2tts/capture.py` |
| Audio out (multi-device) | `voice2tts/output.py` |
| Endpointing | `voice2tts/vad.py` |
| Windows integration | `voice2tts/platform_win.py` |
| Packaging | `Voice2TTS.spec`, `installer/`, `build.ps1` |

`spike/FINDINGS.md` records measured results and the reasoning behind non-obvious
choices. Read it before changing the VAD, the CUDA loading, or the virtual cable
approach — several of those look wrong until you know why they are that way.

## Conventions

- Exceptions in audio callbacks and worker threads are caught and logged rather than
  raised. A thread that dies takes a feature with it and leaves the UI looking fine.
- Devices are addressed by name substring, never by index: PortAudio indices shift
  when USB hardware is plugged in.
- Anything the user owns (config, downloaded voices, GPU pack) lives outside the
  install directory so updates cannot destroy it.
- Comments explain *why*, particularly where code looks wrong but is not.

## Releasing

```powershell
.\scripts\release.ps1 -Bump 0.5.0 -Notes "What changed."
```

`voice2tts/__init__.py` is the single source of truth for the version. Update
`CHANGELOG.md` in the same commit.
