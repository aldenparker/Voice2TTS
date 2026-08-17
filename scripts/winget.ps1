<#
.SYNOPSIS
    Stamp the winget manifests for a released version.

.DESCRIPTION
    Fills in the version, release date, download URL and SHA-256 from a published
    GitHub release, writing a ready-to-submit manifest folder. The hash must match
    the asset winget will actually download, so it is read from the release's
    .sha256 file rather than recomputed locally.

    Submit the output with:
        winget-create submit <folder>
    or by opening a PR against microsoft/winget-pkgs.

.EXAMPLE
    .\scripts\winget.ps1 -Version 0.5.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Repo = "aldenparker/Voice2TTS",
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$templateDir = Join-Path $ProjectRoot "installer\winget"
if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "dist\winget\$Version" }

$asset = "Voice2TTS-Setup-$Version.exe"
$base = "https://github.com/$Repo/releases/download/v$Version"

Write-Host "Fetching checksum for $asset..." -ForegroundColor Cyan
try {
    $shaText = (Invoke-WebRequest -Uri "$base/$asset.sha256" -UseBasicParsing).Content
} catch {
    throw "Could not download $base/$asset.sha256 -- is the release published?"
}
$sha = ($shaText -split '\s+')[0].ToUpper()
if ($sha -notmatch '^[0-9A-F]{64}$') { throw "Unexpected checksum content: $shaText" }
Write-Host "  $sha" -ForegroundColor Green

New-Item -ItemType Directory -Force $OutDir | Out-Null
$today = (Get-Date).ToString("yyyy-MM-dd")

Get-ChildItem $templateDir -Filter *.yaml | ForEach-Object {
    # -Raw + -Encoding utf8 both ways: these files contain no non-ASCII today, but
    # a mangled manifest would be rejected by winget's validation.
    $text = Get-Content $_.FullName -Raw -Encoding utf8
    $text = $text -replace 'PackageVersion: .*', "PackageVersion: $Version"
    $text = $text -replace 'ReleaseDate: .*', "ReleaseDate: $today"
    $text = $text -replace 'InstallerUrl: .*', "InstallerUrl: $base/$asset"
    $text = $text -replace 'InstallerSha256: .*', "InstallerSha256: $sha"
    $dest = Join-Path $OutDir $_.Name
    [System.IO.File]::WriteAllText($dest, $text, [System.Text.UTF8Encoding]::new($false))
    Write-Host "  wrote $($_.Name)"
}

Write-Host "`nManifests in $OutDir" -ForegroundColor Green
Write-Host "Validate with:  winget validate --manifest `"$OutDir`""
Write-Host "Submit with:    wingetcreate submit `"$OutDir`""
