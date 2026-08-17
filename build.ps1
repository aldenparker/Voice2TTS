<#
.SYNOPSIS
    Build Voice2TTS into a distributable Windows installer.

.DESCRIPTION
    Runs, in order: bundle asset fetch, icon generation, PyInstaller, then Inno Setup.
    Produces dist\installer\Voice2TTS-Setup-<version>.exe

    CUDA is not bundled -- the app downloads it on demand. Use -BundleCuda for a fat
    offline build (~1.9 GB installer).

.PARAMETER SkipInstaller
    Stop after PyInstaller. Useful when Inno Setup is not installed.

.PARAMETER BundleCuda
    Include the ~1.3 GB NVIDIA CUDA runtime in the build.

.PARAMETER Clean
    Delete build\ and dist\ first.

.PARAMETER Version
    Override the version. By default it is read from voice2tts/__init__.py, which is
    the single source of truth -- the updater compares against that same value, so a
    mismatch would make the app fail to recognise its own release.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean -SkipInstaller
#>
[CmdletBinding()]
param(
    [string]$VenvPath = "$env:USERPROFILE\.venvs\voice2tts",
    # Explicit interpreter, for environments with no venv layout (CI runners).
    [string]$Python,
    [string]$Version,
    [switch]$SkipInstaller,
    [switch]$BundleCuda,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  + $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }

if ($Python) {
    $vpy = $Python
    if (-not (Test-Path $vpy)) { throw "No interpreter at $vpy" }
} else {
    $vpy = "$VenvPath\Scripts\python.exe"
    if (-not (Test-Path $vpy)) { throw "No virtualenv at $VenvPath. Run .\setup.ps1 first." }
}
$env:PYTHONPATH = $ProjectRoot

if (-not $Version) {
    $Version = & $vpy -c "import voice2tts, sys; sys.stdout.write(voice2tts.__version__)"
    if (-not $Version) { throw "Could not read __version__ from voice2tts/__init__.py" }
}
Write-Host "Building Voice2TTS $Version" -ForegroundColor Cyan

# A previously built exe still running holds its own files open and breaks the clean.
# Easy to hit: a windowed PyInstaller build that fails to start pops a modal
# traceback dialog and sits there forever.
# Scoped to processes running FROM dist\ -- an installed copy of Voice2TTS is the
# user's running app, and killing that out from under them would be unacceptable.
$distPath = Join-Path $ProjectRoot "dist"
$stale = Get-Process -Name "Voice2TTS*" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -and $_.Path.StartsWith($distPath, [StringComparison]::OrdinalIgnoreCase)
}
if ($stale) {
    Write-Warn "Stopping $($stale.Count) process(es) running from dist\"
    $stale | Stop-Process -Force
    Start-Sleep -Milliseconds 500
}

if ($Clean) {
    Write-Step "Cleaning"
    foreach ($d in @("$ProjectRoot\build", "$ProjectRoot\dist")) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d; Write-Ok "removed $d" }
    }
}

# --- PyInstaller present? --------------------------------------------------
Write-Step "Checking build tools"
# Probed via find_spec rather than a bare import: an ImportError would write to
# stderr, and PowerShell 5.1 turns native stderr into a terminating error while
# $ErrorActionPreference is Stop. This always exits 0 and prints one character.
$hasPyInstaller = & $vpy -c "import importlib.util,sys; sys.stdout.write('1' if importlib.util.find_spec('PyInstaller') else '0')"
if ($hasPyInstaller -ne "1") {
    Write-Warn "PyInstaller not installed; installing..."
    & $vpy -m pip install pyinstaller --quiet
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller" }
}
Write-Ok "PyInstaller ready"

# --- bundle assets ---------------------------------------------------------
Write-Step "Fetching bundle assets (3 voices, VAD, base.en)"
& $vpy "$ProjectRoot\scripts\fetch_models.py" --bundle
if ($LASTEXITCODE -ne 0) { throw "Asset fetch failed" }

Write-Step "Generating icon"
& $vpy "$ProjectRoot\scripts\make_icon.py"
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed" }

# --- freeze ----------------------------------------------------------------
Write-Step "Running PyInstaller"
if ($BundleCuda) {
    $env:VOICE2TTS_BUNDLE_CUDA = "1"
    Write-Warn "Bundling CUDA -- expect a ~1.9 GB installer"
} else {
    Remove-Item Env:\VOICE2TTS_BUNDLE_CUDA -ErrorAction SilentlyContinue
}

# Invoked as a module rather than via Scripts\pyinstaller.exe so this works with a
# bare interpreter too, not just a venv layout.
& $vpy -m PyInstaller --noconfirm --distpath "$ProjectRoot\dist" `
    --workpath "$ProjectRoot\build" "$ProjectRoot\Voice2TTS.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$appDir = "$ProjectRoot\dist\Voice2TTS"
$exe = "$appDir\Voice2TTS.exe"
$consoleExe = "$appDir\Voice2TTS-console.exe"
if (-not (Test-Path $exe)) { throw "Expected $exe but it was not produced" }
if (-not (Test-Path $consoleExe)) { throw "Expected $consoleExe but it was not produced" }
$sizeMb = [math]::Round(((Get-ChildItem $appDir -Recurse -File |
    Measure-Object Length -Sum).Sum / 1MB), 0)
Write-Ok "built $appDir ($sizeMb MB)"

# --- smoke test the frozen build -------------------------------------------
Write-Step "Smoke-testing the frozen build"
# Deliberately the CONSOLE exe. The windowed one has no stdout, and if it fails to
# start it shows a modal traceback dialog that would hang this script indefinitely.
$outFile = Join-Path $env:TEMP "v2t-check-out.txt"
$errFile = Join-Path $env:TEMP "v2t-check-err.txt"
$proc = Start-Process -FilePath $consoleExe -ArgumentList "--check" -NoNewWindow `
    -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
if (-not $proc.WaitForExit(180000)) {
    $proc | Stop-Process -Force
    throw "Frozen build did not exit within 180s during --check"
}
$checkExit = $proc.ExitCode
$check = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue) `
       + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
Write-Host $check

# Distinguish "app never started" from "app started but assets are missing".
# Reporting a missing-asset error for an ImportError sends you to the wrong file.
if ([string]::IsNullOrWhiteSpace($check)) {
    throw "Frozen build produced no output (exit $checkExit). Run '$exe --check' directly."
}
if ($check -match "Traceback|ImportError|ModuleNotFoundError") {
    throw "Frozen build failed to start -- see the traceback above."
}
if ($check -notmatch "Piper voices\s*:\s*[1-9]") {
    throw "Frozen build started but cannot see its bundled voices -- check the spec's datas"
}
if ($check -notmatch "Silero VAD\s*:\s*ok") {
    throw "Frozen build started but cannot see the VAD model -- check the spec's datas"
}
Write-Ok "frozen build resolves its bundled assets"

if ($SkipInstaller) {
    Write-Step "Done (installer skipped)"
    Write-Host "App: $appDir"
    exit 0
}

# --- installer -------------------------------------------------------------
Write-Step "Building installer"
$iscc = $null
# winget installs Inno per-user by default, which is neither Program Files path.
foreach ($c in @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)) { if (Test-Path $c) { $iscc = $c; break } }
if (-not $iscc) {
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}

if (-not $iscc) {
    Write-Warn "Inno Setup not found. Install it, then re-run:"
    Write-Host "    winget install JRSoftware.InnoSetup"
    Write-Host "  The frozen app is still usable at: $appDir"
    exit 0
}

& $iscc "/DAppVersion=$Version" "$ProjectRoot\installer\Voice2TTS.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

$setup = "$ProjectRoot\dist\installer\Voice2TTS-Setup-$Version.exe"
if (-not (Test-Path $setup)) { throw "Inno reported success but $setup is missing" }

# The updater verifies this before running anything it downloaded. Written in the
# standard "<hash>  <filename>" shasum format so it is checkable by hand too.
$hash = (Get-FileHash $setup -Algorithm SHA256).Hash.ToLower()
$shaFile = "$setup.sha256"
"$hash  $(Split-Path -Leaf $setup)" | Out-File -FilePath $shaFile -Encoding ascii -NoNewline
Write-Ok "checksum $($hash.Substring(0,16))... -> $(Split-Path -Leaf $shaFile)"

$mb = [math]::Round((Get-Item $setup).Length / 1MB, 0)
Write-Step "Done"
Write-Host "Installer: $setup ($mb MB)" -ForegroundColor Green
Write-Warn "Unsigned -- SmartScreen will warn on first run (More info -> Run anyway)"
Write-Host "`nPublish with:  .\scripts\release.ps1"
