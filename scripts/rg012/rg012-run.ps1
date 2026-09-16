# RG-012 Tier-2 - install Provenote on a clean, Python-free Windows box and drive three turns.
# ASCII ONLY on purpose: Windows PowerShell 5.1 reads a UTF-8-no-BOM file as ANSI, so any non-ASCII
# character (em-dash, arrow, curly quote) corrupts string literals and the whole file fails to parse.
# tests/unit/test_rg012_harness.py fails the suite if a non-ASCII byte gets in.
#
# Tracked in the repo since 2026-09-16 (it lived only in C:\rg012-host\script before, where no diff
# review or test could see a change to the ship gate). scripts/rg012/rg012-tier2.wsb maps THIS folder
# into the sandbox, so the run uses exactly the reviewed copy.
#
# The verdict that counts is computed on the HOST by `python -m scripts.release_preflight`, which
# re-reads the turn-N-result.json files below with the app's own citation parser
# (synthesis.audit_citations). The lines this script logs are a readable estimate for whoever watches
# the sandbox; they re-implement the contract, and a re-implemented contract drifts (KI-35).
$ErrorActionPreference = 'Continue'

# One directory per run. The log used to be appended to C:\rg012\out\rg012.log, so an out\ folder
# that was not cleared between runs held two runs in one log, and the host read the FIRST installer
# line and ANY pass line in it (the 2026-08-15 archive holds runs 3 and 4). A run directory cannot
# mix runs, and cannot leave a stale turn file for the next run to be judged on.
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path 'C:\rg012\out' ('run-' + $stamp)
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'rg012.log'
function L($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m
  Write-Host $line
  Add-Content -Path $log -Value $line -Encoding UTF8
}

L "=== RG-012 Tier-2 start ==="
L ("run: " + $stamp)
L ("python on PATH? " + [bool](Get-Command python -ErrorAction SilentlyContinue) + "   (must be False)")
L ("OS: " + (Get-CimInstance Win32_OperatingSystem).Caption)

# --- 1. silent install -------------------------------------------------------
# The bundle folder still holds the June 0.1.0 artifacts alongside the new ones, so pick the
# installer EXPLICITLY by product name and newest build - never "the first one found".
$all = Get-ChildItem 'C:\rg012\installer' -Filter '*setup.exe' | Sort-Object LastWriteTime -Descending
L ("installers visible: " + (($all | ForEach-Object { $_.Name }) -join ' | '))
$setup = $all | Where-Object { $_.Name -like 'Provenote*' } | Select-Object -First 1
if (-not $setup) { L "FAIL: no Provenote*setup.exe under C:\rg012\installer"; return }
L ("installer chosen: {0} ({1} MB, built {2})" -f $setup.Name, [math]::Round($setup.Length/1MB,1), $setup.LastWriteTime)
$t0 = Get-Date
Start-Process -FilePath $setup.FullName -ArgumentList '/S' -Wait
L ("silent install finished in {0:N0}s" -f ((Get-Date)-$t0).TotalSeconds)

# The install FOLDER takes tauri.conf's productName ("Provenote"); the EXECUTABLE takes the Cargo
# package name ("doc-assistant-desktop"). ADR-012's product/code identity split, working as designed.
# Locate the folder, then enumerate - never filter on a guessed exe name.
$roots = @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:APPDATA) | Where-Object { $_ -and (Test-Path $_) }
$appDir = (Get-ChildItem $roots -Directory -Filter 'Provenote' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $appDir) { L "FAIL: no Provenote install folder found"; return }
$exes = Get-ChildItem $appDir -File -Filter '*.exe' -ErrorAction SilentlyContinue | Sort-Object Length
$exes | ForEach-Object { L ("  {0,10:N1} MB  {1}" -f ($_.Length/1MB), $_.Name) }
$exe = $exes | Where-Object { $_.Name -notmatch 'uninstall' -and $_.Length -lt 200MB } | Select-Object -Last 1
if (-not $exe) { L "FAIL: could not identify the app shell executable"; return }
L ("app shell: " + $exe.FullName)

# Stale instances from earlier attempts would answer on 8001 with the WRONG provider (they
# inherited no env), so the turn would be measured against a process this run did not configure.
$stale = Get-Process -Name 'doc-assistant-api','doc-assistant-desktop' -ErrorAction SilentlyContinue
if ($stale) {
  L ("stopping {0} stale process(es) so this run owns the backend" -f $stale.Count)
  $stale | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 5
}

# --- 2. launch with a provider the sandbox can reach ------------------------
# The host is the sandbox's DEFAULT GATEWAY on the Hyper-V Default Switch. Discover it rather
# than hardcode it: that address is reassigned by Hyper-V (it was 172.29.224.1 through the
# 2026-08 runs and is 172.25.128.1 as of 2026-08-14). A stale constant does not fail loudly --
# the gate reaches the answer step, gets no LLM, and reports what looks like an app defect.
# The old values stay as the last fallbacks so a gateway-less run still tries something.
$env:LLM_PROVIDER = 'ollama'
$env:LLM_MODEL    = 'llama3.1:8b'
$gw = (Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
       Sort-Object RouteMetric | Select-Object -First 1).NextHop
$candidates = @()
if ($gw) { $candidates += $gw }
$candidates += '172.25.128.1'
$candidates += '172.29.224.1'
$env:OLLAMA_HOST = $null
foreach ($ip in ($candidates | Select-Object -Unique)) {
  $url = "http://${ip}:11434"
  try {
    $tags = Invoke-RestMethod "$url/api/tags" -TimeoutSec 5
    $env:OLLAMA_HOST = $url
    L ("host ollama reachable at {0}, {1} models" -f $url, $tags.models.Count)
    break
  } catch { L ("ollama not reachable at " + $url) }
}
if (-not $env:OLLAMA_HOST) {
  $fallbackIp = $candidates[0]
  $env:OLLAMA_HOST = "http://${fallbackIp}:11434"
  L ("WARN: no ollama endpoint answered; falling back to " + $env:OLLAMA_HOST)
}

Start-Process -FilePath $exe.FullName
L "app launched; waiting for the sidecar (bundled weights load on first run)..."

$health = $null
$waited = 0
for ($i = 0; $i -lt 120; $i++) {
  Start-Sleep -Seconds 5
  $waited += 5
  try { $health = Invoke-RestMethod 'http://127.0.0.1:8001/api/health' -TimeoutSec 5; break } catch {}
  if ($waited % 60 -eq 0) { L ("still waiting for /api/health ... {0}s" -f $waited) }
}
if (-not $health) { L "FAIL: no /api/health after ~600s"; return }
L ("health OK after ~{0}s: {1}" -f $waited, ($health | ConvertTo-Json -Compress))

try { L ("setup readiness: " + ((Invoke-RestMethod 'http://127.0.0.1:8001/api/setup' -TimeoutSec 15) | ConvertTo-Json -Compress)) } catch { L ("setup probe failed: " + $_.Exception.Message) }

# --- 3. find where this install keeps documents, then seed it ---------------
$settings = Invoke-RestMethod 'http://127.0.0.1:8001/api/settings' -TimeoutSec 20
$settings | ConvertTo-Json -Depth 6 | Out-File (Join-Path $out 'settings.json') -Encoding utf8
L ("settings: " + ($settings | ConvertTo-Json -Compress -Depth 4))

$srcDir = $null
foreach ($k in 'source_dir','sources_dir','documents_dir','data_dir','doc_dir') {
  if (($settings.PSObject.Properties.Name -contains $k) -and $settings.$k) { $srcDir = $settings.$k; break }
}
if (-not $srcDir) { $srcDir = Join-Path $env:LOCALAPPDATA 'doc_assistant\data\sources' }
L ("source dir: " + $srcDir)
New-Item -ItemType Directory -Force -Path $srcDir | Out-Null
Copy-Item 'C:\rg012\corpus\*.pdf' -Destination $srcDir -Force
L ("seeded {0} PDFs" -f (Get-ChildItem $srcDir -Filter *.pdf).Count)

# --- 4. ingest ---------------------------------------------------------------
try {
  Invoke-RestMethod 'http://127.0.0.1:8001/api/ingest' -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 30 | Out-Null
  L "ingest accepted (202)"
} catch { L ("ingest POST failed: " + $_.Exception.Message) }

for ($i = 0; $i -lt 180; $i++) {
  Start-Sleep -Seconds 5
  try {
    $st = Invoke-RestMethod 'http://127.0.0.1:8001/api/ingest/status' -TimeoutSec 10
    if ($i % 12 -eq 0) { L ("ingest: " + ($st | ConvertTo-Json -Compress)) }
    if ($st.state -and $st.state -notmatch 'running|pending|in_progress|started') {
      L ("ingest final: " + ($st | ConvertTo-Json -Compress)); break
    }
  } catch {}
}
$h2 = Invoke-RestMethod 'http://127.0.0.1:8001/api/health' -TimeoutSec 10
L ("chunk_count after ingest: " + $h2.chunk_count)
if ([int]$h2.chunk_count -le 0) { L "FAIL: 0 chunks, cannot produce a cited turn"; return }

# --- 5. THE GATE: three real turns, one question per corpus document ---------
# One turn was a coin flip. llama3.1:8b cites all-or-nothing per answer (KI-36), and the
# byte-identical 0.5.1 installer failed one run in four on the same single question
# (RIGOR_TODO RG-012, 2026-08-14). So: three questions, each about a different document in
# corpus\, each in its OWN session so no answer is shaped by another's history. The citation
# half passes when at least one turn cites and no turn tried to cite in a form nothing resolves.
# The packaging half needs every turn to come back with an answer.
$questions = @(
  'What is BERT re-ranking and how does it relate to retrieval?',
  'How can dropout regularize a recurrent neural network without disrupting its memory?',
  'Why does the order of inputs and outputs matter for sequence-to-sequence models?'
)
L ("turns planned: " + $questions.Count)

# The app's contract, restated for the in-sandbox estimate only (see the header): a bracket
# holding an optional source/ref label and a list of integers.
$CITE_BODY = '\[\s*(?:sources?|refs?)?\s*\d+(?:\s*(?:,|;|&|and)\s*\d+)*\s*\]'
$answered = 0
$cited = 0
$unresolved = 0
for ($k = 1; $k -le $questions.Count; $k++) {
  $q = $questions[$k - 1]
  L ("turn {0}/{1} asking: {2}" -f $k, $questions.Count, $q)
  $body = @{ text = $q; session_id = ('rg012-{0}-turn{1}' -f $stamp, $k) } | ConvertTo-Json -Compress
  $t1 = Get-Date
  try {
    $resp = Invoke-WebRequest 'http://127.0.0.1:8001/api/chat' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900 -UseBasicParsing
    $raw = $resp.Content
  } catch { L ("turn {0}: chat failed: {1}" -f $k, $_.Exception.Message); continue }
  L ("turn {0}: completed in {1:N0}s" -f $k, ((Get-Date) - $t1).TotalSeconds)
  $raw | Out-File (Join-Path $out ('turn-{0}-stream.txt' -f $k)) -Encoding utf8

  $resultLine = ($raw -split "`n" | Where-Object { $_ -match '^data:' -and $_ -match '"sources"' } | Select-Object -Last 1)
  if (-not $resultLine) { L ("turn {0}: no result event in the stream" -f $k); continue }
  $json = $resultLine -replace '^data:\s*', ''
  # Out-File -Encoding utf8 writes a BOM on PowerShell 5.1; the host reader opens it as utf-8-sig.
  $json | Out-File (Join-Path $out ('turn-{0}-result.json' -f $k)) -Encoding utf8
  $answered++

  try {
    $r = $json | ConvertFrom-Json
    L ("turn {0}: ANSWER: {1}" -f $k, ($r.answer -replace '\s+', ' '))
    $resolved = ([regex]::Matches($r.answer, $CITE_BODY, 'IgnoreCase')).Count
    $attempts = @([regex]::Matches($r.answer, '\[[^\]]*[A-Za-z][^\]]*\]') |
      Where-Object { $_.Value -notmatch ('^' + $CITE_BODY + '$') } | ForEach-Object { $_.Value })
    L ("turn {0}: {1} sources; {2} resolved citation(s); {3} unresolved attempt(s)" -f $k, $r.sources.Count, $resolved, $attempts.Count)
    if ($attempts.Count -gt 0) { L ("turn {0}:   unresolved: {1}" -f $k, (($attempts | Select-Object -First 5) -join ' ')) }
    if ($resolved -gt 0 -and $r.sources.Count -gt 0) { $cited++ }
    elseif ($attempts.Count -gt 0) { $unresolved++ }
  } catch { L ("turn {0}: could not parse result payload: {1}" -f $k, $_.Exception.Message) }
}

# --- 6. two verdicts, never one ----------------------------------------------
# A citation failure is not a broken build, and must not read as one (RIGOR_TODO RG-012).
if ($answered -eq $questions.Count) {
  L ("*** RG-012 PACKAGING: PASS - installed on a clean box, {0} chunks, {1} of {1} turns answered ***" -f $h2.chunk_count, $questions.Count)
} else {
  L ("*** RG-012 PACKAGING: FAIL - {0} of {1} turns returned an answer ***" -f $answered, $questions.Count)
}
if ($answered -lt $questions.Count) {
  L ("*** RG-012 CITATION (estimate): NOT JUDGED - {0} turn(s) got no answer to judge ***" -f ($questions.Count - $answered))
} elseif ($unresolved -gt 0) {
  L ("*** RG-012 CITATION (estimate): FAIL - {0} turn(s) tried to cite in a form nothing resolves (prompt/parser problem) ***" -f $unresolved)
} elseif ($cited -gt 0) {
  L ("*** RG-012 CITATION (estimate): PASS - {0} of {1} turns cited ***" -f $cited, $questions.Count)
} else {
  L ("*** RG-012 CITATION (estimate): FAIL - no turn cited (grounding problem, not packaging) ***")
}
L "The verdict that counts: run  uv run --no-sync python -m scripts.release_preflight  on the host."
L "=== done ==="
