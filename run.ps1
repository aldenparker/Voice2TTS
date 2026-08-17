<#
.SYNOPSIS
    Launch Voice2TTS.

.PARAMETER Cli
    Run headless with console logging instead of the tray icon.

.PARAMETER Devices
    List audio devices and exit.

.PARAMETER Check
    Verify models, CUDA and the virtual cable, then exit.

.PARAMETER Say
    Speak one phrase through the configured outputs and exit.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Check
    .\run.ps1 -Say "hello from voice2tts"
#>
[CmdletBinding()]
param(
    [string]$VenvPath = "$env:USERPROFILE\.venvs\voice2tts",
    [switch]$Cli,
    [switch]$Devices,
    [switch]$Check,
    [string]$Say,
    [string]$LogLevel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

$vpy = "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    throw "No virtualenv at $VenvPath. Run .\setup.ps1 first."
}

$env:PYTHONPATH = $ProjectRoot

$argsList = @("-u", "-m", "voice2tts")
if ($Cli)      { $argsList += "--cli" }
if ($Devices)  { $argsList += "--devices" }
if ($Check)    { $argsList += "--check" }
if ($Say)      { $argsList += @("--say", $Say) }
if ($LogLevel) { $argsList += @("--log-level", $LogLevel) }

& $vpy @argsList
exit $LASTEXITCODE
