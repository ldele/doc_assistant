# launch_app.ps1 -- one-shot dev launch for the doc_assistant desktop app.
#
# Starts the FastAPI backend (127.0.0.1:8001) and the Svelte/Vite dev UI (localhost:1420)
# in two separate console windows, waits for the backend's model load, then opens the app
# in the default browser. One command instead of README's two-shell flow.
#
# Usage:
#   just app
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_app.ps1
#   (or double-click scripts/launch_app.cmd)
#
# Notes:
# * ASCII-only on purpose: Windows PowerShell 5.1 misreads BOM-less UTF-8 .ps1 files.
# * The backend runs `uv run --no-sync`: it uses the venv exactly as `just sync` left it and
#   never mutates it -- a plain `uv run` would re-resolve WITHOUT the per-machine torch extra
#   (docs/specs/torch-backend-per-machine.md). Run `just sync` first on a fresh clone.
# * Already-running servers on 8001/1420 are reused, so re-running this is idempotent.
# * For the native Tauri window instead of a browser tab: `cd apps/desktop && npx tauri dev`
#   (needs the Rust toolchain); the API window this script opens serves it unchanged.
# * Stop everything by closing the two "doc_assistant ..." console windows.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot  # this file lives in scripts/

# --- preflight --------------------------------------------------------------
if (-not (Test-Path (Join-Path $root '.venv'))) {
    Write-Host 'No .venv found - run `just sync` first (see README Setup).' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $root 'apps\desktop\node_modules'))) {
    Write-Host 'Installing frontend dependencies (first run only)...'
    npm --prefix (Join-Path $root 'apps\desktop') install
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'npm install failed - see output above.' -ForegroundColor Red
        exit 1
    }
}

# Any listener counts, IPv4 or IPv6 -- Vite binds ::1 only, uvicorn 127.0.0.1 only, so a
# plain TcpClient('127.0.0.1', ...) probe would miss Vite and spawn a duplicate window.
function Test-Port([int]$port) {
    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
}

# --- backend (FastAPI + SSE, port 8001) -------------------------------------
if (Test-Port 8001) {
    Write-Host 'API already running on 127.0.0.1:8001 - reusing it.'
} else {
    Start-Process powershell -WorkingDirectory $root -ArgumentList @(
        '-NoProfile', '-NoExit', '-Command',
        "`$host.UI.RawUI.WindowTitle = 'doc_assistant API (8001)'; uv run --no-sync uvicorn apps.api.main:app --host 127.0.0.1 --port 8001"
    )
    Write-Host 'API starting in its own window (model load takes ~30-60 s cold)...'
}

# --- frontend (Vite dev server, port 1420) ----------------------------------
if (Test-Port 1420) {
    Write-Host 'Desktop dev server already running on localhost:1420 - reusing it.'
} else {
    Start-Process powershell -WorkingDirectory (Join-Path $root 'apps\desktop') -ArgumentList @(
        '-NoProfile', '-NoExit', '-Command',
        "`$host.UI.RawUI.WindowTitle = 'doc_assistant UI (1420)'; npm run dev"
    )
    Write-Host 'Desktop dev server starting in its own window...'
}

# --- wait for the API, then open the app ------------------------------------
$deadline = (Get-Date).AddSeconds(180)
$up = $false
$r = $null
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8001/api/health' -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if ($up) {
    $health = $r.Content | ConvertFrom-Json
    Write-Host ("Ready - {0} chunks | {1} | {2}" -f $health.chunk_count, $health.model, $health.embedding_model) -ForegroundColor Green
} else {
    Write-Host 'API did not answer within 3 min - check the "doc_assistant API (8001)" window; opening the UI anyway.' -ForegroundColor Yellow
}

Start-Process 'http://localhost:1420'
Write-Host 'App opened at http://localhost:1420 - close the two doc_assistant windows to stop.'
