<#
.SYNOPSIS
    One-time setup for Voice2TTS: virtualenv, dependencies, models.

.DESCRIPTION
    Creates the venv OUTSIDE this folder by default. This project usually lives in
    OneDrive, and a CUDA venv is gigabytes across thousands of files -- letting
    OneDrive sync it causes file locks and long stalls.

.PARAMETER VenvPath
    Where to create the virtualenv. Default: %USERPROFILE%\.venvs\voice2tts

.PARAMETER Cpu
    Skip the CUDA packages (~1.4 GB) and run Whisper on CPU.

.PARAMETER Voice
    Piper voice to download. Default: en_US-lessac-medium

.PARAMETER WhisperModel
    Whisper model to pre-cache. Default: small.en

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Cpu -WhisperModel base.en
#>
[CmdletBinding()]
param(
    [string]$VenvPath = "$env:USERPROFILE\.venvs\voice2tts",
    [switch]$Cpu,
    [string]$Voice = "en_US-lessac-medium",
    [string]$WhisperModel = "small.en"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Ok($msg)   { Write-Host "  + $msg" -ForegroundColor Green }

# --- locate a usable Python ------------------------------------------------
Write-Step "Locating Python 3.11 or 3.12"

$python = $null
foreach ($candidate in @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:PROGRAMFILES\Python312\python.exe",
    "$env:PROGRAMFILES\Python311\python.exe"
)) {
    if (Test-Path $candidate) { $python = $candidate; break }
}

if (-not $python) {
    # The WindowsApps stub is a redirector, not an interpreter -- reject it.
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($ver -in @("3.11", "3.12")) { $python = $cmd.Source }
    }
}

if (-not $python) {
    Write-Warn "No suitable Python found. Installing 3.12 via winget..."
    winget install --id Python.Python.3.12 --scope user --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity
    $python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    if (-not (Test-Path $python)) {
        throw "Python install failed. Install 3.12 from python.org and re-run."
    }
}
Write-Ok "Using $python ($(& $python --version))"

# --- virtualenv ------------------------------------------------------------
Write-Step "Creating virtualenv at $VenvPath"
if ($VenvPath -like "*OneDrive*") {
    Write-Warn "Venv path is inside OneDrive; sync churn will cause file locks."
}
if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    & $python -m venv $VenvPath
    Write-Ok "Created"
} else {
    Write-Ok "Already exists"
}
$vpy = "$VenvPath\Scripts\python.exe"
& $vpy -m pip install --upgrade pip --quiet

# --- dependencies ----------------------------------------------------------
Write-Step "Installing dependencies"
if ($Cpu) {
    Write-Warn "CPU mode: skipping CUDA packages"
    $reqs = Get-Content "$ProjectRoot\requirements.txt" | Where-Object { $_ -notmatch "^nvidia-" }
    $tmp = Join-Path $env:TEMP "v2t-requirements-cpu.txt"
    $reqs | Out-File -FilePath $tmp -Encoding utf8
    & $vpy -m pip install -r $tmp
    Remove-Item $tmp -ErrorAction SilentlyContinue
} else {
    & $vpy -m pip install -r "$ProjectRoot\requirements.txt"
}
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed" }
Write-Ok "Dependencies installed"

# --- models ----------------------------------------------------------------
Write-Step "Downloading models"
$env:PYTHONPATH = $ProjectRoot
& $vpy "$ProjectRoot\scripts\fetch_models.py" --voice $Voice --whisper $WhisperModel
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

# --- virtual cable ---------------------------------------------------------
Write-Step "Checking for a virtual audio cable"
& $vpy -m voice2tts --check

if (-not (& $vpy -c "from voice2tts.devices import cable_installed; print(cable_installed())" `
          | Select-String -Quiet "True")) {
    Write-Host ""
    Write-Warn "No virtual audio cable found. Discord cannot hear the app without one."
    Write-Host "    1. Download VB-CABLE:  https://vb-audio.com/Cable/"
    Write-Host "    2. Unzip, right-click VBCABLE_Setup_x64.exe -> Run as administrator"
    Write-Host "    3. Reboot"
    Write-Host "    4. Re-run:  .\run.ps1"
    Write-Host "  (This step needs admin rights and cannot be automated from here.)"
}

Write-Step "Setup complete"
Write-Host "Start the app with:  .\run.ps1"
