#Requires -Version 5.1
<#
.SYNOPSIS
    Reclaim Rust/Tauri build space without destroying what is expensive to rebuild.

.DESCRIPTION
    `cargo clean` is the wrong tool in this repo: it wipes all of target/, and target/
    also holds two things that are NOT cheap to regenerate --

      1. target/release/bundle/  - the current release installers (~3.1 GB for 0.5.1).
         Only a full `tauri build` puts them back, and the MSI has no copy anywhere
         (only the NSIS .exe is attached to the GitHub release).
      2. target/release/doc-assistant-api.exe - a ~1.5 GB build-time copy of the frozen
         sidecar. Deleting the copy is free, but `cargo clean` gives no way to keep (1)
         while dropping (2).

    So this drops only recompilable intermediates: target/debug in full, and everything
    under target/release except bundle/.

    Never touched, by design:
      * apps/desktop/src-tauri/binaries/ - the frozen PyInstaller sidecar. It lives
        outside target/, so no run of this script can force a re-freeze.
      * target/release/bundle/ - docs/RELEASE.md section 8 owns installer pruning, and
        deliberately does it by hand: the PREVIOUS release's files, after the tag is
        pushed, never before.

.PARAMETER Registry
    Also drop the shared cargo registry sources (~/.cargo/registry/src). Safe: cargo
    re-extracts them offline from the .crate tarballs kept in ~/.cargo/registry/cache,
    so this costs extraction time, not a re-download. Opt-in because that cache is
    shared with every other Rust project on the machine.

.PARAMETER DryRun
    Report what would be freed and delete nothing. Never blocked by the build guard --
    a dry run touches nothing, so an active build is irrelevant to it.

.PARAMETER Force
    Skip the active-build guard. The guard is already scoped so that another project's
    build does not block a target-only clean, so reaching for this means overriding a
    real signal: either a build in THIS repo, or (under -Registry) a build anywhere on
    the machine. Check nothing is actually compiling first.

.EXAMPLE
    just clean

.EXAMPLE
    just clean -DryRun

.EXAMPLE
    just clean -Registry
#>
[CmdletBinding()]
param(
    [switch]$Registry,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Get-PathSize {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return [int64]0 }
    $item = Get-Item $Path -Force
    if (-not $item.PSIsContainer) { return [int64]$item.Length }
    $sum = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return [int64]0 }
    return [int64]$sum
}

function Format-GB {
    param([int64]$Bytes)
    return ('{0,7:N2} GB' -f ($Bytes / 1GB))
}

$root   = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root 'apps\desktop\src-tauri\target'
$bundle = Join-Path $target 'release\bundle'
$frozen = Join-Path $root 'apps\desktop\src-tauri\binaries'

if (-not (Test-Path $target)) {
    Write-Host "Nothing to do: no target/ dir at $target"
    exit 0
}

# Half-deleting sources out from under a running rustc breaks that build, so refuse rather
# than race it -- but WHAT is at risk depends on scope, so the guard is scoped to match:
#
#   * target/ belongs to this repo alone. Only a build HERE can be hurt by clearing it.
#     This machine runs other Rust projects more or less constantly (harper, lever, ...);
#     aborting on those would mean `just clean` never runs without -Force, and a guard that
#     always cries wolf is one people bypass reflexively. So foreign builds do not block it.
#   * ~/.cargo/registry/src is shared by every Rust project on the box, so under -Registry
#     ANY live build is a real hazard and blocks.
#
# A dry run deletes nothing and is exempt from both.
if (-not $DryRun -and -not $Force) {
    $rust = @(Get-CimInstance Win32_Process `
                -Filter "Name='cargo.exe' OR Name='rustc.exe' OR Name='link.exe'" `
                -ErrorAction SilentlyContinue)

    if ($Registry -and $rust.Count -gt 0) {
        $names = ($rust | Select-Object -ExpandProperty Name -Unique) -join ', '
        Write-Host "ABORT: -Registry touches the cache shared with every Rust project, and a"
        Write-Host "       build is active ($names). Re-run once it finishes, or drop -Registry"
        Write-Host "       to clean this repo's target only."
        exit 1
    }

    # A build in THIS repo: rustc and link carry absolute --out-dir paths, which catches the
    # compile phase even when `cargo` itself was launched from the cwd with no path in argv.
    # A fresh mtime on target/ is the cwd-independent backstop for the same thing.
    $here = @($rust | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($root) })
    $hot  = ((Get-Item $target).LastWriteTime -gt (Get-Date).AddMinutes(-2))
    if ($here.Count -gt 0 -or ($hot -and $rust.Count -gt 0)) {
        Write-Host "ABORT: a build looks active in this repo. Re-run once it finishes,"
        Write-Host "       or pass -Force if you know it is not."
        exit 1
    }
}

# target/debug wholesale, then target/release minus bundle/.
$doomed = @()
$doomed += (Join-Path $target 'debug')
$release = Join-Path $target 'release'
if (Test-Path $release) {
    Get-ChildItem $release -Force |
        Where-Object { $_.Name -ne 'bundle' } |
        ForEach-Object { $doomed += $_.FullName }
}
if ($Registry) {
    $doomed += (Join-Path $env:USERPROFILE '.cargo\registry\src')
}

if ($DryRun) { Write-Host "DRY RUN - nothing will be deleted." }
Write-Host ''

[int64]$freed = 0
[int64]$stuck = 0
foreach ($path in $doomed) {
    $label = $path.Replace($root + '\', '').Replace($env:USERPROFILE, '~')
    if (-not (Test-Path $path)) { continue }

    $before = Get-PathSize $path
    if ($DryRun) {
        Write-Host ('  would free {0}  {1}' -f (Format-GB $before), $label)
        $freed += $before
        continue
    }

    # Transient locks (an editor indexer, a virus scanner) are common on Windows; a
    # couple of retries clears them without failing the whole run.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if (-not (Test-Path $path)) { break }
        try { Remove-Item $path -Recurse -Force -ErrorAction Stop; break }
        catch { Start-Sleep -Milliseconds 400 }
    }

    $after   = Get-PathSize $path
    $freed  += ($before - $after)
    $stuck  += $after
    if ($after -gt 0) {
        Write-Host ('  partial   {0}  {1}  ({2} still locked)' -f (Format-GB ($before - $after)), $label, (Format-GB $after))
    } else {
        Write-Host ('  freed     {0}  {1}' -f (Format-GB $before), $label)
    }
}

Write-Host ''
if ($DryRun) {
    Write-Host ('Would free {0}. Re-run without -DryRun to do it.' -f (Format-GB $freed))
} else {
    Write-Host ('Freed {0}.' -f (Format-GB $freed))
}
if ($stuck -gt 0) {
    Write-Host ('{0} could not be deleted (files in use). Close the holder and re-run.' -f (Format-GB $stuck))
}

Write-Host ''
Write-Host 'Kept:'
Write-Host ('  {0}  release installers (docs/RELEASE.md section 8 prunes these by hand)' -f (Format-GB (Get-PathSize $bundle)))
Write-Host ('  {0}  frozen sidecar in src-tauri/binaries (no PyInstaller re-freeze needed)' -f (Format-GB (Get-PathSize $frozen)))
if (-not $Registry) {
    $regSrc = Join-Path $env:USERPROFILE '.cargo\registry\src'
    $regSz  = Get-PathSize $regSrc
    if ($regSz -gt 0) {
        Write-Host ('  {0}  cargo registry sources - add -Registry to drop them too' -f (Format-GB $regSz))
    }
}
