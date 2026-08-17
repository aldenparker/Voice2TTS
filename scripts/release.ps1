<#
.SYNOPSIS
    Build and publish a Voice2TTS release to GitHub.

.DESCRIPTION
    Bumps the version if asked, builds the installer, tags the commit, and creates a
    GitHub release with the installer and its checksum attached. The running app's
    updater reads exactly this release, so the tag and voice2tts/__init__.py must
    agree -- this script enforces that rather than trusting you to remember.

    Requires the GitHub CLI (winget install GitHub.cli) and `gh auth login`.

.PARAMETER Bump
    Version to release, e.g. 0.3.0. Rewrites voice2tts/__init__.py and commits it.
    Omit to release whatever __init__.py already says.

.PARAMETER Notes
    Release notes. Omit to have GitHub generate them from commits.

.PARAMETER Draft
    Create the release as a draft so you can review before it goes live.

.PARAMETER SkipBuild
    Reuse the existing dist\installer output instead of rebuilding.

.EXAMPLE
    .\scripts\release.ps1 -Bump 0.3.0 -Notes "Fixes the VAD endpoint delay."
    .\scripts\release.ps1 -Draft
#>
[CmdletBinding()]
param(
    [string]$Bump,
    [string]$Notes,
    [switch]$Draft,
    [switch]$SkipBuild,
    [string]$VenvPath = "$env:USERPROFILE\.venvs\voice2tts"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $ProjectRoot

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  + $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }

$vpy = "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $vpy)) { throw "No virtualenv at $VenvPath. Run .\setup.ps1 first." }
$initFile = "$ProjectRoot\voice2tts\__init__.py"

# --- preflight -------------------------------------------------------------
Write-Step "Preflight"

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    $ghPath = "$env:ProgramFiles\GitHub CLI\gh.exe"
    if (Test-Path $ghPath) { $gh = $ghPath } else {
        throw "GitHub CLI not found. Install it:  winget install GitHub.cli"
    }
} else { $gh = $gh.Source }

& $gh auth status
if ($LASTEXITCODE -ne 0) { throw "Not signed in to GitHub. Run:  gh auth login" }
Write-Ok "gh authenticated"

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    throw @"
No 'origin' remote. Create the repository first, for example:
    gh repo create Voice2TTS --private --source . --remote origin --push
"@
}
Write-Ok "origin: $remote"

# --- version ---------------------------------------------------------------
Write-Step "Version"

if ($Bump) {
    if ($Bump -notmatch '^\d+\.\d+\.\d+$') { throw "Version must look like 1.2.3, got '$Bump'" }
    $content = Get-Content $initFile -Raw
    $updated = [regex]::Replace($content, '__version__\s*=\s*"[^"]*"', "__version__ = `"$Bump`"")
    if ($updated -eq $content) { throw "Could not find __version__ in $initFile" }
    Set-Content -Path $initFile -Value $updated -Encoding utf8 -NoNewline
    Write-Ok "set __version__ = $Bump"
}

$version = & $vpy -c "import importlib, voice2tts, sys; importlib.reload(voice2tts); sys.stdout.write(voice2tts.__version__)"
if (-not $version) { throw "Could not read __version__" }
$tag = "v$version"
Write-Ok "releasing $tag"

$existing = & $gh release view $tag --json tagName 2>$null
if ($LASTEXITCODE -eq 0) {
    throw "Release $tag already exists. Bump the version with -Bump first."
}

# --- commit any version bump ----------------------------------------------
$dirty = git status --porcelain
if ($dirty) {
    Write-Warn "Working tree has uncommitted changes:"
    git status --short
    $answer = Read-Host "`nCommit them as 'Release $tag'? [y/N]"
    if ($answer -notmatch '^[Yy]') { throw "Aborted -- commit or stash first." }
    git add -A
    git commit -m "Release $tag"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    Write-Ok "committed"
}

# --- build -----------------------------------------------------------------
$setup = "$ProjectRoot\dist\installer\Voice2TTS-Setup-$version.exe"
if ($SkipBuild) {
    if (-not (Test-Path $setup)) { throw "-SkipBuild given but $setup does not exist" }
    Write-Warn "Reusing existing build"
} else {
    Write-Step "Building"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\build.ps1" -Clean
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
}
if (-not (Test-Path $setup)) { throw "Expected installer at $setup" }
if (-not (Test-Path "$setup.sha256")) { throw "Missing checksum at $setup.sha256" }

# --- tag and publish -------------------------------------------------------
Write-Step "Tagging"
git tag -a $tag -m "Voice2TTS $version" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Warn "Tag $tag already exists locally; reusing" }
git push origin HEAD
if ($LASTEXITCODE -ne 0) { throw "git push failed" }
git push origin $tag
if ($LASTEXITCODE -ne 0) { throw "git push --tags failed" }
Write-Ok "pushed $tag"

Write-Step "Publishing release"
$args = @("release", "create", $tag, $setup, "$setup.sha256",
          "--title", "Voice2TTS $version")
if ($Notes) { $args += @("--notes", $Notes) } else { $args += "--generate-notes" }
if ($Draft) { $args += "--draft" }

& $gh @args
if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }

Write-Step "Done"
Write-Host "Published $tag" -ForegroundColor Green
if ($Draft) { Write-Warn "Created as a DRAFT -- the updater ignores drafts until you publish it" }
$repo = ($remote -replace '^.*github\.com[:/]', '') -replace '\.git$', ''
Write-Host "`nMake sure the app's update repo is set to:  $repo"
Write-Host "(Settings -> Updates, or updates.repo in config.toml)"
