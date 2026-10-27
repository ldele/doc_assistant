<!-- status: active · updated: 2026-10-27 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps at most 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Past 20, one batch cuts it to the newest 10 (`devlog_rotate_to`, cpc 1.12.0, since 2026-09-29).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive, or starts the next one when that is
> past `archive_max_tokens`, and verifies the bytes; then update the range below by hand (cpc ticket
> T-003), and check a new archive's title (cpc 1.12.0 titles every new one "archive 001"). A day may
> be split across two files at the cut. Every archived heading is listed in
> [`docs/archive/DEVLOG-INDEX.md`](archive/DEVLOG-INDEX.md). Older entries, newest-first, unedited:
> **2026-09-30 (3) → 2026-10-22** in [`docs/archive/DEVLOG-archive-008.md`](archive/DEVLOG-archive-008.md)
> (rotated 2026-10-17: 11 entries, 2026-10-21: 11 more, and 2026-10-27: 11 more) ·
> **2026-09-07 → 2026-09-30 (2)** in [`docs/archive/DEVLOG-archive-007.md`](archive/DEVLOG-archive-007.md)
> (rotated 2026-09-30, the first batch: 11 entries, and 2026-10-09: 11 more) ·
> **2026-08-12 (1) → 2026-09-04 (2)** (the first 2026-09-02 and 2026-09-04 entries are unnumbered) in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16, on 2026-09-17, 2026-09-18, 2026-09-20, twice on 2026-09-21, twice on 2026-09-22 and on 2026-09-29) ·
> **2026-08-08 (1) → 2026-08-11 (4)** in [`docs/archive/DEVLOG-archive-005.md`](archive/DEVLOG-archive-005.md)
> (rotated 2026-08-30) ·
> **2026-08-05 → 2026-08-07** in [`docs/archive/DEVLOG-archive-004.md`](archive/DEVLOG-archive-004.md)
> (rotated 2026-08-28) · **2026-08-02 → 2026-08-03** in
> [`docs/archive/DEVLOG-archive-003.md`](archive/DEVLOG-archive-003.md) (rotated 2026-08-26 — this
> row was missing until 2026-08-28) · **2026-07-15 → 2026-08-01** in
> [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **The 4,000-line cap in `tests/unit/test_doc_sizes.py` stays as the backstop** (an entry cap cannot
> see an entry that is itself an ADR in disguise). When either trips, rotate — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---

## 2026-10-27 (3) — The backend ships as a folder: 9–10 s to a working window instead of 72 (KI-73, part 2)

**What changed.**
- **Build.** `scripts/doc_assistant_api.spec` builds one-dir (`EXE(exclude_binaries=True)` +
  `COLLECT`); `scripts/build_sidecar.py` copies `dist/doc-assistant-api/` to
  `apps/desktop/src-tauri/backend/` (gitignored), which `tauri.conf.json` ships as
  `bundle.resources` instead of `bundle.externalBin`.
- **Shell.** `lib.rs` `bundled_backend` finds `backend/doc-assistant-api.exe` under the resource
  folder and starts it with the shell plugin's `command(path)`: the same launch token, parent PID
  and KI-71 tree-kill as the sidecar had, and no console window. Without the folder (dev) it says
  so and starts nothing, as before.
- **Capability.** The window's `shell:allow-execute` for the sidecar is gone — Rust starts the
  backend, and capabilities do not govern Rust. The window holds `core:default` and
  `dialog:allow-open` only (`test_desktop_security_config.py`, `docs/security.md`).
- **Installer.** `src-tauri/nsis/installer.nsi` is Tauri CLI 2.11.4's own NSIS template with one
  line changed, `SetCompressor` without `/SOLID`; `bundle.windows.nsis` sets it and `zlib`.
- **Preflight.** `backend_contents` replaces the one-file size floor: the folder must hold
  `_internal/pymupdf/layout/resources/onnx` with its `.yaml` files (what KI-34 lost) and
  `_internal/hf_cache/hub`, and weigh at least 2,490 MiB (built: 2,515.9 MiB, 17,855 files).
  KI-34's 17 MB is under the 1% margin, so the paths, not the floor, catch it. 3 tests replace 1.
- **Local gates.** The built folder puts third-party `.py` files under `apps/desktop/src-tauri/`
  (`backend/`, and its copy in `target/`), so CI's own `mypy src/ apps/` and `bandit -r src/ apps/`
  failed on this PC after a build; `pyproject.toml` excludes `src-tauri` from both (CI never has
  the folder). `.secrets.baseline` records the template's one detect-secrets hit, a registry path
  (`SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate`, line 621) read as high-entropy text.
- Paths in `clean_build.ps1`, the justfile and `docs/desktop-packaging.md`.

**Why.** KI-73: the one-file backend unpacked 1.6 GB to a temp folder at every launch — 29.4 s
of its 38.5 s to health alone, 52 s of 72 with the window. Part 1 (entry (2)) filled the wait;
this removes most of it.

**Measured** (dev PC, scratch data folder, `/api/health` polled every 200 ms). The backend alone:
38.3 and 38.8 s one-file, 6.1 and 7.3 s one-dir. The built app, window start to health: 72.2 s
one-file; one-dir 28.9 s on the first launch after the build (Windows scanning the new files),
then 9.3 and 9.9 s. One backend process each time, and nothing left after the window closed.
Two runs per arm: indicative, not a benchmark. Installer: 1,642,519,979 bytes in ~7 min (one-file:
1,574 MiB); installed, 2.5 GB instead of 1.6.

**The installer detour.** The first build failed in makensis: "Internal compiler error #12345:
error mmapping datablock". Tauri's template sets `SetCompressor /SOLID`, which gathers every
file uncompressed in one block, and the 32-bit makensis cannot map one past ~2 GB — with LZMA
and with zlib. An `installerHooks` file cannot change it ("can't change compressor after data
already got compressed or header already changed!"), and Tauri has no non-solid setting. So the
template: extracted from `cli.win32-x64-msvc.node` (978 lines, CRLF, 86 handlebars tags) and
checked against the script Tauri generated for this app — a sorted-line diff differs only in the
compressor line, its configured value, comments and the include of the hook file tried first.

**Rejected.** fp16 models to shrink the folder: changes the retrieval numbers, so it needs an eval
run first. An MSI (WiX): a per-machine, admin install, and RG-012 installs the NSIS build
silently per user. One-file kept, with an archive unpacked on first launch: the same files on
disk plus a first-launch path to build and test. A hook overriding the compressor: makensis
refuses it.

**What it opens.** The template is a copy: on a Tauri CLI upgrade, re-take it and re-apply the
one line (its header says so; `test_the_installer_compresses_file_by_file` fails if `/SOLID`
returns). 2.5 GB installed. The release build and RG-012 rerun come from this commit.

## 2026-10-27 (2) — The window shows the last session at once: a start screen, then the library, chats and graph until the engine answers (KI-73, part 1)

**What changed.** Committed with the Settings fixes in `a5d3231`; this entry and its CHANGELOG
line follow. `index.html` draws a start screen inside `#app` ("Starting Provenote…", the icon)
styled by `public/boot.css` — a file, since the window's CSP admits only its own; it colours
`html` in the system's light or dark `--bg`, which the app's `:root` rule outranks at mount, and
`main.ts` clears `#app` before `mount()`. `lib/core/lastSession.ts` (pure, 4 tests) keeps a
versioned copy of a list in the window's `localStorage` and reads it back; a copy over 2,000,000
characters is not kept and an older one is removed. The library documents, folders, chat list and
concept graph save a copy whenever they load for real; at launch `App.svelte` shows them
(`restoreConversations`, `restoreGraph`) with the notice "Showing your library as it was last time
while the engine starts. Changes work once it is ready.", and the readiness gate refreshes them
all once `/api/health` answers. A never-built graph is not kept. Before the engine answers, the
Graph tab no longer fetches (it would show an error) and reads as loading rather than "No
concept graph yet".

**Why.** The installed app takes 38–72 s before its backend answers (KI-73: 29–52 s of one-file
unpack, ~7.5 s of imports), and the window drew an empty shell meanwhile. The user, in the
walkthrough: load the library, past chats and graphs before the rest.

**Checked.** Live in the dev app over the 104-document scratch copy: the copies came to 75 KB
(documents), 20 KB (chats), 19 KB (graph). With the API stopped and the page reloaded, the Library
grid, the chat list and the Graph ("Covers 29 of your 104 documents") came from the copies under
the notice, with "starting the engine…" in the status bar; started again, the window went ready by
itself 16 s later, the notice went, and every failed request was from the down period. The start
screen is served, and nothing of it is left after mount.

**Rejected.** IndexedDB: a larger store for lists that fit in a few hundred KB, and a second async
path at startup. Disabling every write while the copy is on screen: many controls for a wait the
packaging change cuts to seconds; the notice says changes work once ready. A window colour in
`tauri.conf.json`: one fixed colour cannot follow the system theme, and in light mode it is the
same white.

**What it opens.** At 75 KB per 104 documents the cap is reached near 2,800 documents; a library
past it starts as before. Part 2 of KI-73: package the backend as a folder.

## 2026-10-27 — Setting up an answer engine from Settings: an instant panel, verdicts in place, no impossible model (KI-72)

**What changed.**
- **Server.** `llm.model_mismatch(provider, model, installed)`: the one rule for a pair that can
  never answer — a Claude name on Ollama, any other name on Anthropic, or a model an answering,
  probed Ollama lacks (`llama3` and `llama3:latest` are the same). `app_settings.set_llm_selection`
  refuses the first two (names only, no network) and no longer tells an installed-app user to
  "add it to .env". `readiness._with_selection` marks the active engine not ready when its
  selection is impossible, so *Getting started* names it and says what fixes it.
- **Getting started** (`ProviderSetup.svelte`) draws from `GET /api/setup?probe=false` first and
  replaces it with the probed state; while the probe is out, the Ollama card reads *checking…* and,
  when Ollama is the active engine, its step reads "Checking whether Ollama is answering…"
  (`setup.ts` `displaySteps`, `ollamaPending`). Each card has its own result line under its own
  buttons; a refused key reads "Anthropic refused this key, so nothing was saved. <reason>"
  (`keyRefusal`). The Ollama model box is seeded with a model Ollama can serve, and *Use Claude
  for answers* also shows when the active selection is the broken one.
- **Provider & model** (`Settings.svelte`): a keyless provider gets a line and a button to
  *Getting started*; changing the provider fills in a model it serves (`modelForProvider`); errors
  read without "Error:"; the hint no longer says *Getting started* is "above". After a switch from
  either section the status bar's health is re-read: it kept naming the old model.

**Found by:** the walkthrough's §0 in a clean sandbox (the user). *Getting started* said
"Checking…" until the Ollama probe gave up (4.1 s there, against 0.5 s without the probe); a key
seemed not to register — the verdict was drawn at the panel's foot, under the Ollama card; *Provider
& model* could not reach Anthropic, and the sandbox's own `settings.json` held `ollama` with
`claude-haiku-4-5-20251001`. Checked then: the installed app reads its key from the environment or
the data folder's `credentials.json` and its selection from `settings.json`, both hand-editable, and
the `.env` the dev setup uses is looked for beside the unpacked program, so it never applies —
Settings is in practice the only door.

**Checked.** Server tests (rule, refusal, readiness; two that matched the old wording updated);
frontend helpers (3 tests). Live in the dev app over a scratch folder seeded with the stuck
selection: the panel drew in 69 ms with Ollama *checking…*, the probed verdict named the Claude
model on Ollama at 2.4 s, *Use this model* offered `llama3.1:8b` and fixed it; switching provider
filled `claude-haiku-4-5-20251001` and back `llama3.1:8b`; the Claude-on-Ollama pair was refused with
its sentence; the status bar followed both switches. A refused key's line appeared under the key
field — the request intercepted in the page, so no key, fake or real, was sent.

**Rejected.** Checking with Ollama at switch time whether the model is installed: it puts the
network in the save and makes the tests depend on the box; *Getting started* reports it instead.
A key field in *Provider & model* too: two places to enter a secret; one button to the existing
one. Clearing a refused key from its box: it is masked, and keeping it lets a typo be fixed.

**What it opens.** The real key flow on the installed build (the user's own key, in the sandbox).

## 2026-10-26 (4) — Closing the app ends its backend; the backend stops when the app is gone (KI-71)

**What changed.** `apps/desktop/src-tauri/src/lib.rs`: the shell keeps the sidecar's
`CommandChild` (`Sidecar`, managed state) and on `RunEvent::Exit` ends its process tree:
`taskkill /PID <pid> /T /F`, by full path under `%SystemRoot%\System32`, with `CREATE_NO_WINDOW`;
`kill()` when that fails or off Windows. `run()` now builds the app and passes an event handler.
It also sets `DOC_PARENT_PID` to its own process id. `apps/api/parent_watch.py` (new): a daemon
thread waits for that process (on Windows on its handle, `WaitForSingleObject`; elsewhere by
polling) and `__main__` then sets uvicorn's `should_exit`, with a forced exit 15 s later if a
pool or a stream holds the shutdown open. `__main__` now runs `uvicorn.Server` directly to reach
`should_exit`. `rustfmt` also wrapped one existing line in `launch_token`.
`tests/unit/test_api_parent_watch.py`: the pid parsing, a real process watched before and after
it ends, `main` watching only when a pid is named, and the Rust contract pinned in its source.

**Found by:** the 0.7.0 walkthrough in the RG-012 sandbox. After the window closed, both
`doc-assistant-api` processes — the one-file launcher and the server it starts — still ran and
held 8001. A reopened window's own backend could not bind, so the window read through the old one
with its new launch token, and *new folder* was refused ("needs the token the app was started
with"). A plain launch from the desktop icon did the same: on any machine, the second launch could
change nothing until a reboot. The cause: the shell plugin ends on exit only the children the
window spawns (`tauri-plugin-shell` 2.3.5 `lib.rs`), and `lib.rs` dropped the handle at spawn.

**Checked.** The new tests (10). `cargo check` and `cargo fmt --check`. The server half end to
end: `python -m apps.api` with `DOC_PARENT_PID` naming a stand-in process — healthy after 16 s,
then `desktop_shell_gone` and a clean shutdown 1.0 s after the stand-in was killed. The shell half
needs the installed build: the sandbox after the rebuild.

**Rejected.** Killing only the launcher (`CommandChild::kill`): a one-file launcher's server may
survive it, which is the state we were in. A single-instance lock instead: it would stop a second
window but not free a backend whose window had closed. Accepting any token from a backend that
outlived its window: it would re-open the hole S-6 closed.

**What it opens.** A second Provenote window opened while one runs still gets a backend it cannot
write to (its own cannot bind 8001); a single-instance guard is the follow-up.

## 2026-10-26 (3) — A failed turn says why; quotes drop the extractor's emphasis markers; two claims corrected

**What changed.** From a pre-launch check of the 0.7.0 rows of `docs/release-ux-checklist.md`, run
in the dev app over a scratch copy of the working library (`DOC_DATA_DIR`, nothing written to
`data/`):
- **A failed turn ends with a sentence.** `chat_controller.events.Failure` is a terminal event:
  `handle_message` wraps the answer path and yields one when it raises, its text from
  `helpers._turn_failure_sentence` (engine unreachable, by exception name through the cause chain:
  "No answer: Ollama is not answering at <host>…"; another failure: the provider, model and
  exception type, never its message). The router maps it to an SSE `error` event, and its worker
  thread catches anything that escapes the controller (`_WORKER_FAILED`); the window shows the
  sentence on the turn, and a stream that ends with no answer, result or error says so
  (`EMPTY_TURN`). The CLI prints it; `measure_latency` exits with it. Tests: the controller turn
  with `httpx.ConnectError`, the sentence rules, and two SSE tests that fail without the router
  change.
- **Quotes read as text.** `graph/emphasis.ts` marks the characters that are emphasis delimiters
  (`**` runs, an `_` outside a word); `ConceptDefinition.svelte` leaves them out of every quote and
  passes them to `markedSegments`, which now takes a drop set so the server's highlight offsets
  still land on their words. Display only (ADR-043). Five frontend tests.
- **Settings' Ollama card** says its 36% / 14% citation figures predate KI-66's fix.
- **`CHANGELOG`**: "known junk is gone" becomes "most"; the keyword limit lists what remains.

**Found by the check.** With Ollama stopped, a question got nothing under it: the worker thread
died on `httpx.ConnectError` and the stream ended with only `done` (release checklist §6, the
user's row). The Definition card and term panels showed `_**Jakobs et al., 2019**_` verbatim.
On the library's 1,560 keywords, about 50 are not terms in classes the filters do not cover:
own author names from title pages and headers (23 in 14 documents, a few false hits), "Cell Press"
split by letter-spacing (`ce ress`, `ll primer`), accents (`g ttingen`, `ram n`), possessives
(`cajal s`), notation (`t cat`) — which the CHANGELOG's "letter-spaced text … no longer" denied.
The Ollama card stated the pre-KI-66 rates unqualified. Also filed, not fixed: KI-68 (the
staleness notice names the concept, not the form), KI-69 (no sentence for zero `is_a`
proposals), KI-70 (rail names cut at 800 px). Passed: re-ingest hint; Manage keywords at size
(95 ms), forms, exact/broad, the delete preview; staleness → Rebuild (7 s); the Definition card's
every action; term signals (5 cited names, 31 offers, 26 places over 333 rows); proposals accept
and reject, an accepted placement surviving a rebuild; KL1 labels; S-5; Settings with the
engine down.

**Rejected.** Rendering emphasis as `<em>`/`<strong>`: italics would survive, but a quote is
plain text everywhere else and the markers sit mostly on citations. A fallback to another
provider when the engine is unreachable: it would spend money the user did not choose to.

**What it opens.** Row 115 still wants Settings to take another Ollama address (ROADMAP 100).
Keyword classes above: ADR-056 (letter-spacing) and ADR-057 (accents) are parked work.

## 2026-10-26 (2) — The API answers the Windows window's origin, and RG-012 asks as the window does (KI-67)

**What changed.** `apps/api/main.py` `_ALLOWED_ORIGINS` gains `http://tauri.localhost`, the origin
Tauri 2 serves the window from on Windows; `tauri://localhost` (macOS, Linux) and the two dev-server
origins stay. `tests/integration/test_api_window_origins.py` pins a read and a token-carrying write
from both packaged origins, and that any other origin gets no `Access-Control-Allow-Origin`; its two
Windows cases fail on the old list. The RG-012 harness (`scripts/rg012/rg012-run.ps1`) asks
`/api/health` with `Origin: http://tauri.localhost` and logs `window origin …: allowed|REFUSED`;
REFUSED fails its packaging verdict. `release_preflight.rg012_packaging` fails a run whose line says
REFUSED and a run with no line at all (every run before this one); `test_rg012_harness` pins the
line against the reader and the harness's origin against the API's list. `CHANGELOG` [0.7.0] Fixed
leads with it.

**Found by:** the walkthrough's first look at the installed 0.7.0 build, in the RG-012 sandbox after
a passing run (`run-20261026-104551`: 3 of 3 answered, 3 of 3 cited). The window "did not respond,
Settings included". A read-only diagnostic run inside the sandbox (`wsb exec`, at the user's go)
showed otherwise: Windows called the window responsive, the webview was idle, and the screenshot
showed "Loading…" and the status bar's *still starting — retrying*, while `/api/health` answered
200 in 0.5 s from the same box. `/api/health` with `Origin: http://tauri.localhost` came back without
`Access-Control-Allow-Origin` (with `tauri://localhost`, with it), and the window's storage was
written under `http://tauri.localhost`. So the readiness loop in `App.svelte` never saw a health
answer it could read. 0.6.0 had the same list on the same Tauri (2.11.3): its installed window
cannot have worked on Windows either, and KI-39's report was probably this. RG-012 missed it on
every release because PowerShell ignores CORS.

**Also seen, not fixed here.** The window process running at the diagnostic had started at 10:56,
while the backend still running had started with the run at 10:49, so a backend outlived its first
window (`lib.rs` never stops the sidecar). A window opened next would talk to that old backend with
a different launch token, so its writes would get 401. The walkthrough's §6 row "closing the window
ends the sidecar process" checks it on the fixed build.

**Rejected.** `allow_origin_regex` for any `*.localhost`: wider than the two origins Tauri uses. Routing
the window's calls through a Rust-side HTTP client to skip CORS: a new plugin and permission surface
for what one list entry fixes. Checking every packaged origin in RG-012: the sandbox is Windows, so it
can only see the Windows window's.

**What it opens.** The release's walkthrough can now be driven at all. Nothing yet checks the
macOS/Linux origin on a real window; no installer is built for either.

## 2026-10-26 — Every Ollama call asks for an 8,192-token window: answers no longer lose the instructions to cite

**What changed.** `config.OLLAMA_CONTEXT_TOKENS` (8,192) is passed as `num_ctx` to the streamed
answer (`pipeline.build_chat_model`'s `OllamaLLM`) and to every one-shot client
(`llm.make_client`: reviewer, eval judge, taxonomy, gap suggestions). The wiki narrows it through
`fit_window` as before. Two tests pin it, one per path. `CHANGELOG` [0.7.0] gains the fix and
moves to 2026-10-26, the date the release commit will carry; README's local-model figures are
marked as predating it.

**Found by:** RG-012 Tier-2 for 0.7.0 (run `run-20261026-090139`). Packaging passed on a clean
sandbox, and 0 of 3 answers cited anything; preflight read all three as *uncited*. Ollama's own log
gave the cause: `vram-based default context … default_num_ctx=4096`, then `truncating input prompt
limit=2050` on the answer prompts of 4,032–5,176 tokens (10 sources). On overflow llama.cpp keeps
only the end, so the system prompt, which carries the instructions to cite, never reached the
model, and the call still succeeded. It is KI-65's cause, on the answer path: `828fddc` gave only
the wiki a window.

**Checked.** The same three questions on the host, over a scratch library of the same three PDFs
(322 chunks), with the dev API on Ollama `llama3.1:8b`. Before: 0 of 20 sentences cited, and four
truncation warnings. After: all three answers cite, 11 of 13 sentences ([1]–[5]; [5]; [3], [4],
[6], [9]); Ollama starts its runner with `-c 8192`, and every prompt runs whole (`truncated = 0`);
0.6 GB more VRAM (5,675 against 5,035 MiB).

**Rejected.** Sizing the window per prompt from an estimate (the wiki's `material_budget`): the
answer prompt is bounded by the locked `TOP_K`, and one fixed window keeps Ollama from reloading the
model between the answer and the reviewer. Leaving it to `OLLAMA_CONTEXT_LENGTH` on the server: an
installed app's user would have to know to set it.

**What it opens.** KI-36's local citation rates (`llama3.1:8b` 37.6%, `qwen2.5:7b` 18.0%,
2026-08-07) were measured with prompts that were probably being cut, so they likely overstate the
gap to Haiku: re-measure. Nothing yet warns when an answer prompt exceeds 8,192 tokens.

## 2026-10-25 — 0.7.0 release commit: the version in nine places, a dated CHANGELOG, and no developer command in the re-ingest dialog

**What changed.** `docs/RELEASE.md` §1–§2. The version reads 0.7.0 in all nine places
(`pyproject.toml`, `src/doc_assistant/__init__.py`, `package.json`, `tauri.conf.json`,
`Cargo.toml`, and the three locks re-locked by their own tools, `uv lock`, `npm install
--package-lock-only`, `cargo update -p doc-assistant-desktop --offline`). `CHANGELOG.md`: the
Unreleased section becomes `[0.7.0] — 2026-10-25`, with a line for the libraries upgraded over
advisories during the release. `README.md` Status names 0.7.0 with its date and today's test count
(2,776 Python, 299 desktop). The 0.7.0 notes are drafted locally; their install size, hash and
verification are filled in the release's second session.

**Found on the way, and fixed here.** The re-ingest dialog, changed in session 5c, told the user to
run `python -m scripts.extract_keywords --apply --force` — a command an installed app has no Python
or checkout to run. It now says the app does not run those passes yet and that they are redone from
a source checkout. Behind it: nothing in `apps/` or `src/` calls `extract_keywords`, so keyword
extraction exists only on the command line — a library the installed app builds has none, and one
carried over from 0.6.0 keeps its old keywords, listed alphabetically. The CHANGELOG line and the
README limitation for keywords now say so. `release_preflight`'s `dev_commands` check reads `just`
recipes only, which is why it passed this.

**What it opens.** Keyword extraction inside the app (an action, or a step of ingest) — after the
release, with the LLM scan design (ADR-058) that would replace its candidates.

## 2026-10-24 (3) — Keyword cleanup (ROADMAP 99): junk filtered, keywords listed by rank, and the first precision baseline shows no gain

**What changed.** The code is `344b3d6`. `knowledge/keywords.py` has five filters, each with a
unit test of what it drops and keeps. They drop a `br` token (table residue), a phrase with two or
more one-letter tokens (letter-spaced text, initials), a token with no letter (pages, volumes,
counts, years), and a surname the document itself cites with "et al." on at least half of its uses
(`cited_authors`, the term signals' reading; `majority` moved here and is re-imported by
`term_signals.py`). Web addresses and DOIs are removed before tokenising (`strip_urls`).
`document_keywords` gained `score` and `rank` (additive, through `init_db`; old links read NULL).
`Document.keywords` is ordered by rank, so the Library lists a document's keywords best first.
`scripts/extract_keywords.py` runs `init_db()` before `--apply`. The re-ingest dialog no longer
sends keywords to Settings: Rebuild on the Graph tab redoes gaps, and keywords are redone from the
command line. The working library was re-extracted 2026-10-22 (backup
`data/library.db.bak-20261022-pre5c`). This entry adds the baseline,
`tests/eval/baselines/keyword_cleanup_2026-10-23.md`. **Found by:** the 2026-10-20 assessment:
keyword scores computed and thrown away, about a quarter of keywords noise in a 12-document probe,
precision never measured.

**Measured.** The five patterns went from 123 links to 0 across 104 documents. Every document has 15
keywords and none has fewer than before. Nothing the graph reads moved (row hashes, graph version,
the 22 marked forms). On 15 seeded documents labelled by the user, precision was 62% [46, 78]
before and 64% [50, 77] after, +1 point [−4, +6]; with the user's own labels only, +0 [−4, +4]. The
filters took out 44 keywords, 37 of them noise, but 34 of the 44 that replaced them were noise too.

**Why.** The probe found mechanical junk worth removing, and an unordered list hides which
keyword is the document's main subject. The baseline is what any later keyword change is
measured against.

**Rejected.** A list of names (`cajal`, `dbs`, `cre` are real vocabulary here); dropping
abbreviations outright (ADR-055 shows the long form instead); keeping numbers inside a phrase
(`type 2 diabetes`): none was seen, so the rule stays simple, and the cost is recorded (`sam 2`,
`mrr 10`); replacing `<br>` before tokenising, which would join words across table cells; a
percentile bootstrap at 15 documents.

**What it opens.** Filtering cannot raise precision while the candidates stay TF-IDF windows, so
better candidates come next (ADR-058, an LLM scan measured on these 15 documents). The label rule
gains a *Too vague* class: the user's point is that `connexion` is a term but not a precise one.
Also open: ADR-055, ADR-056 point 8, and two gaps in the baseline, alternating recto/verso running
heads under the furniture floor and accent-split names (the tokeniser is eval-gated).

## 2026-10-24 (2) — `filelock` 4 and OpenTelemetry 1.45.1 upgraded; a `setuptools` advisory reviewed

**What changed.** `uv.lock`: `filelock` 3.29.0 → 4.2.4 (a new major; 4.1.0 for the Python 3.10 fork)
and the OpenTelemetry stack 1.42.0 → 1.45.1, which `chromadb` pulls in (`uv lock
--upgrade-package filelock --upgrade-package opentelemetry-api`; one exporter package added).
`pip-audit-ignore.toml`: GHSA-xmj3-9gf7-h52w on `setuptools`, reviewed. **Found by:** the local
audit while staging the entry below — `filelock` CVE-2026-112907, -112913 and -112940;
`opentelemetry-api` CVE-2026-114603; the `setuptools` advisory (low).

**Why.** `filelock` guards the Hugging Face model cache and torch's caches, so its major was tested
where it is used: the embedder and the reranker load offline from the local cache (768
dimensions; 0.999 against 0.0 on a matching and an unrelated pair); both Chroma stores open on the
new OpenTelemetry with their full counts (18,012 and 45,581); the whole suite passes, 2,216 unit and
560 integration tests. The `setuptools` fix (84.1.0) is blocked by torch 2.12's `setuptools<82`,
and the advisory concerns the legacy wheel-to-egg installer, which nothing here calls — the
reason and what would reverse it are in the ignore entry.

**What it opens.** The sixth advisory round since 2026-10-20 and the second `setuptools` entry held
by torch's cap; both ignore entries reverse when torch lifts it.

## 2026-10-24 — `dompurify` 3.4.17: a critical advisory in the app's answer sanitiser

**What changed.** `apps/desktop/package-lock.json` only: `dompurify` 3.4.16 → 3.4.17
(`npm audit fix`, inside `package.json`'s `^3.4.15`). **Found by:** CI's frontend audit on
`344b3d6` (run 40041172523) — three bypasses rated critical together (a template-expression
bypass in shadow-root output, armed handlers left on a detached subtree, an SMIL `href` guard
skipped under an XHTML parser type).

**Why.** Unlike the `vite` advisory, this one ships: DOMPurify cleans every chat answer before the
one `{@html}` in `Markdown.svelte` (security S-3). Checked: `npm audit` 0; `svelte-check` 0 errors
over 229 files; 299 unit tests; a production build; `tests/unit/test_desktop_security_config.py`
(6). **Not checked here:** the sanitiser against hostile input in a DOM — no DOM library is
installed and the browser pane does not run local scripts; the release walkthrough's S-3 row
drives exactly that in the installed build.

**What it opens.** The same run failed one worker test on Linux
(`test_the_frozen_guard_can_be_lifted_deliberately`, `BrokenProcessPool`), which `344b3d6` cannot
reach (no module it changed is on the worker's import path) and which passes 3 of 3 here; the run
on this commit shows whether it is intermittent.
