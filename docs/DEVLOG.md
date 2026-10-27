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
> **2026-09-30 (3) → 2026-10-17 (3)** in [`docs/archive/DEVLOG-archive-008.md`](archive/DEVLOG-archive-008.md)
> (rotated 2026-10-17: 11 entries, and 2026-10-21: 11 more) ·
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

## 2026-10-22 — `virtualenv` 21.14.12: five advisories in a dev-only tool

**What changed.** `uv.lock` only: `virtualenv` 21.7.13 → 21.14.12 with `packaging` 26.2 → 26.3 and
`distlib` 0.4.0 → 0.4.4 (`uv lock --upgrade-package virtualenv`). As with `pillow` (2026-10-20
(5)), the lock keeps older versions for the Python 3.10 that `requires-python` still declares;
3.12 gets the fixed ones. **Found by:** the audit gate on `4655aae`, reported by session 5c —
CVE-2026-113722, -113725, -113726, -113732 and -113776, all fixed by 21.14.9.

**Why.** `virtualenv` reaches the tree only through `pre-commit` (the `dev` extra), so the app never
shipped it; CI's gate audits the whole environment. Checked: the gate 0 unreviewed; the commit
hooks run on the new version; 29 tooling tests (version, secret-scan gate, action pins) pass.

**What it opens.** The fifth advisory round in three days. The 3.10 fork now holds a `virtualenv`
below the fixes; raising `requires-python` to the 3.12 the project runs would end these forks.

## 2026-10-21 — `vite` 6.4.5: a high advisory in the dev server's editor launcher

**What changed.** `apps/desktop/package-lock.json` only: `vite` 6.4.3 → 6.4.5 (`npm audit fix`,
inside `package.json`'s `^6.0.0`). **Found by:** `npm audit` the morning after `96685c4`, before
any push — three `launch-editor` advisories (GHSA-698v-4r3x-3vcv, GHSA-r825-w8mm-5pxw,
GHSA-587h-7wmj-h5pm): Windows argument escaping, editor flags read as commands, an existence check
bypassed on Windows.

**Why.** The editor launcher runs only in the dev server, so the installed app never carried it;
but the dev loop runs on Windows here, and CI's frontend job fails on a high advisory. Checked:
`npm audit` 0; `svelte-check` 0 errors over 229 files; 299 unit tests; a production build.

**What it opens.** Nothing new — the fourth advisory round in two days, all fixed inside their
version ranges except `pillow`'s.

## 2026-10-20 (5) — `pillow` 13.0.0: eight advisories, fixed only in a new major

**What changed.** `uv.lock` only: `pillow` 12.3.0 → 13.0.0 (`uv lock --upgrade-package pillow`).
The lock keeps 12.3.0 for Python 3.10 alone, which `requires-python = ">=3.10"` still declares and
pillow 13 (`>=3.11`) no longer serves; the project runs 3.12 (KI-2), where 13.0.0 installs.
**Found by:** the local audit gate while staging entry (4), the same evening as entries (2) and (3)
— eight advisories, seven of them high: out-of-bounds reads in the TIFF decoder and encoder, a GIF
decompression-bomb DoS, unbounded recursion in pillow's PDF parser.

**Why.** Hostile documents are in the threat model (`docs/security.md` §1), so reviewing these into
`pip-audit-ignore.toml` was rejected. The app reaches pillow only through `pdfplumber` (its sole
dependent, `Pillow>=9.1`); its own code never imports it outside `scripts/debug_tables.py`, and
figure crops go through PyMuPDF. Checked: the gate 0 unreviewed; 406 PDF, table, figure, extractor
and image tests pass; a library page renders through `pdfplumber.to_image` on 13.0.0.

**What it opens.** Raising `requires-python` to the version the project actually runs would remove
the 3.10 fork from the lock — a decision for the KI-2 pin, not this fix.

## 2026-10-20 (4) — Release refresh for 0.7.0: the walkthrough, the security record and the CHANGELOG brought level with what shipped

**What changed.** `docs/RELEASE.md` §0 and §2, done ahead of the release session while session 5c
waits on the user's labels; no code. `docs/release-ux-checklist.md`: a row per surface shipped
since `v0.6.0` (20 product commits) — the extraction pool, Manage keywords at library size,
proposed placements, keywords in score order (5c), the KL1 labels, names and written case, the
Definition card, term signals — plus the "no console window" row and S-4, S-5 and S-6 checks.
`docs/security.md` §2 lists S-4, S-5, S-6, S-8 and S-12 as holding; the launch token is a rule by
method, so the write routes added since are covered without a line each. `CHANGELOG.md`
`[Unreleased]`: an *Added* block, the security, change and fix lines that were missing, and
*Known limits*, each of 0.6.0's re-read against the code (row 24 still planned; since the tag, the
code behind the others changed only for the ligature fix and the size caps).

**Why.** `preflight`'s `checklists` check fails a release whose walkthrough and security record
have no commit since the previous tag, and the Unreleased section stopped before the definitions
work: release criterion 5 needs it to say what the README says.

**What it opens.** Session 5c adds its own line (keywords in score order, junk filtered) and the
measured keyword numbers to the CHANGELOG. The version bump, the 0.7.0 notes, the build and the
gates stay with the release session.

## 2026-10-20 (3) — `certifi` 2026.10.19: one more advisory, published the same day

**What changed.** `uv.lock` only: `certifi` 2026.5.20 → 2026.10.19 (`uv lock --upgrade-package
certifi`). **Found by:** CI's blocking audit on `2ab7895` (run 39487577705), GHSA-6q48-6c7w-9gw2 —
entry (2)'s three `aiohttp` advisories were gone; the tests passed there (2,754).

**Why.** A newer certificate bundle from the same maintainer; nothing to review into
`pip-audit-ignore.toml`. Checked: the gate 0 unreviewed; two HTTPS connections verified with the new
bundle.

**What it opens.** Two red runs in one evening from advisories published between pushes; the
alerts toggle (entry (2)) still decides whether the next one is seen before a push.

## 2026-10-20 (2) — `aiohttp` 3.14.5: three advisories published overnight turned `main` red

**What changed.** `uv.lock` only: `aiohttp` 3.14.3 → 3.14.5 and its dependency `yarl` 1.24.2 →
1.25.1 (`uv lock --upgrade-package aiohttp`); nothing else moved. **Found by:** CI's blocking audit
on `c27ff9f` (run 39479200471) — CVE-2026-111063, CVE-2026-111064 and GHSA-2g87-pp6c-29x5, all fixed
in 3.14.4; the four Dependabot merges after it fail at the same step.

**Why.** An upgrade inside the same minor series, so no review into `pip-audit-ignore.toml` is
needed. Checked: `scripts.pip_audit_gate` 0 unreviewed; the API integration tests (184) pass on
the `cu130` venv.

**What it opens.** The sixth time a push that touched no dependency met a red audit (KI-58's
shape). Dependabot alerts, still off in the repository settings, would have flagged these between
pushes.

## 2026-10-20 — The README opens on its reader and stops over-stating how settings were chosen and how often local models cite; a project overview gives every figure its source

**What changed.** `README.md`: the opening says who the app is for and what sets it apart (the
printed page behind a citation, the sources kept apart from the model's interpretation, a local
model that keeps documents on the machine), then one result with its size (the public benchmark's
`contains_all` 0.932 ± 0.014 over 5 trials, and why `citation_overlap` at 1.000 cannot rank), then
Quick start, moved up. "Why it's built this way" moves below Architecture, and its first bullet no
longer says `TOP_K`, parent-child, chunk sizes and the BM25 mix "were each picked by measuring
alternatives": it names what the harness measured, says the chunk-size and weight sweeps kept the
defaults, that `TOP_K` and parent-child predate the harness, and that the reranker had no
experiment. Limitations quotes the shipped prompt's citation coverage, 83.5 / 37.6 / 18.0% (one run
each, `tests/eval/baselines/citation_coverage_2026-08-07.md`), in place of the earlier prompt's
81 / 36 / 14%, and replaces "Answers stay grounded either way" with the count behind it: 12 of 27
`llama3.1:8b` answers cited nothing. New `docs/OVERVIEW.md`: what the project is, what sets it
apart, how it works, each measured result with its size, conditions and source, and its state;
linked from the Documentation table. The old opening's second paragraph is gone; its map of the
library is already the concept bullet under What it does.

**Why.** Read against the record on 2026-10-19 and 2026-10-20, two README statements claimed more
than it holds: settings chosen by experiment, and citation rates from a prompt that no longer
ships. The overview keeps those facts in one place, each with its source, so a reader who reuses
them does not re-derive them from the README.

**Rejected.** Rewriting the 0.4.1 and 0.5.0 CHANGELOG entries that quote 81/36/14%: released
entries stay as they shipped. Leading with the citation-coverage figure: it comes from the private
library, one run each, where the public benchmark is one anyone can re-run. A new alt text for the
demo GIF: the GIF dates from 2026-08-11, before the page pane, and its alt text still describes
what it shows.

**What it opens.** The provider card (`apps/desktop/src/lib/settings/ProviderSetup.svelte`
l.254-256) still shows 36/14/81% and "Answers stay grounded in your documents"; the 2026-08-07
baseline planned that refresh for the next release that rebuilds, which 0.7.0 is, with a line under
the CHANGELOG's Unreleased beside it. `tests/eval/TESTING.md` says five scorers where `scorers.py`
has six (`figure_retrieval`). The demo GIF predates the page pane (0.6.0). ROADMAP row 98 (the docs
revamp after 0.7.0) starts from the README's new order. Docs only; no code or test changed.

## 2026-10-18 (3) — The README leads with what a reader gets and states 0.7.0's limits; CI's actions are pinned to commits (security S-12)

**What changed.** `README.md`: the opening says what you get (answers you can check, the page one
click away, what leaves the machine with each engine) before how it is built; the concept bullet
covers names, forms and definitions; Status names 0.7.0 as in preparation, with v0.6.0's date
corrected to 2026-09-04 and six things the release makes true; Limitations is grouped under four
headings and gains the reference-list cut (KI-59), pages holding the previous page's text (KI-57),
concepts versus extracted keywords, the definition grade, the wiki sample and the CLI provider
default (KI-4). Every count is from 2026-10-18: 104 documents, 3,341 pages, 44 reference links,
1,436 of 1,483 keywords in one document, 2,757 Python and 299 desktop tests (the full suite passed,
2,754, before the 3 guard tests were added). `ci.yml`: the 7 `uses:` lines pinned to commit SHAs,
each the commit its major tag pointed at that day; `.github/dependabot.yml` for `github-actions`
only, monthly, minor and patch grouped; `tests/unit/test_ci_actions_pinned.py` fails on a tag (seen
failing on `@v4`). Committed by the user with the checkpoint as `71b3903`.

**Why.** Plan session 5b: the README is the release's front page, and its Status still described
0.6.0 with the wrong date and 2,389 tests; two Limitations links pointed at gitignored files a
public reader cannot open. S-12 was decided at the checkpoint the same day: a tag can be moved to
other code, a SHA cannot.

**Rejected.** Writing Status as already released: the README is pushed before the release, so it
says "in preparation" and one line changes on release day. Bumping the actions' majors while
pinning: that is a behaviour change, and Dependabot proposes each as its own PR.

**What it opens.** Five Dependabot PRs, all new majors (`checkout` 4 → 7, `setup-node` 4 → 7,
`setup-buildx-action` 3 → 4, `setup-uv` 3 → 10, `build-push-action` 6 → 7): the user's to review.
Dependabot alerts read off (404). The CHANGELOG's Unreleased section stops before the definitions
work and the KI-62 to KI-65 fixes; release criterion 5 needs it to match the README.

## 2026-10-18 (2) — Release checkpoint: ADR-053 and ADR-041 accepted, 93a's suggestions stored, a docs revamp planned after 0.7.0

**What changed.** The user took the seven calls left before 0.7.0, one at a time. ADR-053 is
accepted, with expert vocabularies (93c) now built before model candidates (93b), which ride with
ROADMAP 25. ADR-041 is accepted as amended: option 6's buildable part is the gap list, so KL1b
closes and no separate per-concept status is built. `extract_definitions --apply` stored 20
suggestions for 11 of the 13 concepts (12 strong, 8 some), nothing chosen, after a `library.db`
backup. `docs/security.md` S-12 is decided: SHA-pinned actions and Dependabot for actions only,
before the release; alerts are the user's toggle. ROADMAP row 98 adds a docs revamp after the
release (row 84 folded in), and row 95 follows it.

**Why.** Both ADRs had been `proposed` for one and two months while what they decide was built
and used; the release describes them. Row 51's `--apply` had nothing to write: since ADR-054 the
proposer reads the 13 concepts, and none shares a head with another (the 27 candidates are between
terms). Row 53 (3)'s hand score leaves the release path: the merge default pairs 0 concepts, and
the review page's store was empty.

**Rejected.** Refreshing epistemics: a dry run marks 1,188 chunks (445 today), most `contested` —
the stance ADR-041 does not trust; markers stay off by default and the table is rebuilt with KL3.
Update PRs for pip, npm and cargo: weekly noise for one maintainer; upgrades stay in sessions, the
audit gate catches advisories.

**What it opens.** The README (the next session) must not imply `is_a` proposals exist on this
library. The docs revamp's evidence and order are in the local plan.

## 2026-10-18 — A wiki topic's prompt fits the model's context: a large topic is summarised from a fixed sample, and its note says so (KI-65)

**What changed.** `wiki.summarize_cluster` sends a bounded prompt. `material_budget` reserves the
answer (`max_tokens`), a chat frame and the prompt's fixed text out of `WIKI_CONTEXT_TOKENS` (new,
8,192); `select_material` fills the rest with members in `sample_order`, a hash of each
`doc_hash` (the same members give the same sample in any order, and no end of the library is
favoured), up to the first block that does not fit. Size is estimated without a tokenizer, as
UTF-8 bytes / 3 (`BYTES_PER_TOKEN`), above the true count for every script. Excerpts are read in
rounds of 16, so a topic reads the store for its sample, not for every member (48 reads for the
87-paper topic, not 87). A topic summarised from a sample says so: in the prompt's heading ("a
sample of 36 of 87"), in the note ("_Summary written from 36 of 87 papers_", `summary_from: 36` in
the frontmatter, *in summary* beside each sampled source) and in the CLI report. An empty topic,
or a window too small for the fixed text, makes no call. Tests: `tests/unit/test_wiki_budget.py`
at 0, 1, 20 and 10,000 documents and five windows (fake client, fake store, no network);
`tests/unit/test_llm.py` for `num_ctx` and `ollama_context_length`. `docs/knowledge-layer.md`
§6's wiki row no longer says the note is written from a cut prompt.

**`num_ctx`: passed, from the model.** `OllamaClient` takes `num_ctx`; the default `None` keeps
every other caller (reviewer, judge, taxonomy) on the server's default, unchanged.
`wiki.fit_window`, called by `build_wiki`, gives an Ollama client `num_ctx = min(WIKI_CONTEXT_TOKENS,
the model's trained length)`, read from `/api/show` by `llm.ollama_context_length`, and the budget
uses the same number. Keeping the budget below the default was rejected: the default depends on
the GPU (`vram-based default context`, 4,096 on this 12 GB card), and on overflow llama.cpp keeps
only half the window (the log's `limit=2050`), so "below the default" is a number the app cannot
read. Asking for the window makes the budget and the server agree on any card.

**Checked live, read-only** (Ollama forced; the library read through `sqlite3` `mode=ro`; notes
rendered to the session scratchpad, nothing under `data/`). The 87-paper topic, rebuilt from
`library.db` (cosine 0.90). Before: 15,232 real tokens on `llama3.1:8b` (15,245 in the
baseline), 15,775 on `qwen2.5:7b`, 15,766 on `qwen3.5:9b`. After, at a 8,192 window: 36 papers,
6,239 / 6,517 / 6,486 real tokens plus a 400-token answer, no `truncating input prompt` in the
server log, 5-11 s a topic. The estimate ran 14-17% above the real count (3.50-3.63 bytes per
token on this library). Two of the three summaries name both the machine-learning and the
neuroscience papers; `qwen3.5:9b`'s names machine-learning ones only. Their quality is not judged
here: that is RG-018's re-measure.

**Rejected.** Summarising in stages (map-reduce): a call per ~36 papers plus a merge, so a
10,000-paper topic costs hundreds of calls, and the merge summarises summaries, not papers. The
model's whole window (32k-262k tokens here): VRAM and time for one note, and the sample would
change with whichever model ran. A real tokenizer: an arbitrary Ollama model exposes none, and
the byte estimate held on all three. The "most central" papers as the sample: that needs a
centrality measure and its threshold; a hash sample is unbiased and needs neither.

**What it opens.** RG-018's re-measure with KI-65 fixed (baseline §4, condition (c)). KI-65 can
close in the local `KNOWN_ISSUES.md` once this lands. A large topic still leaves papers out (51 of
87); if a sample proves too thin, a second pass is the map-reduce above. `WIKI_CONTEXT_TOKENS`
is set by environment variable only; `build_wiki` has no flag for it.

## 2026-10-17 (4) — A write to the local API needs the token the app was started with (security S-6)

**What changed.** The Tauri shell mints 32 random bytes per launch (`getrandom` 0.3, already in
`Cargo.lock` through Tauri: one line in the lock, no new crate), passes them to the API sidecar as
`DOC_API_TOKEN`, and hands them to its window through a new `api_token` command. The API
(`apps/api/launch_token.py`, mounted inside CORS) answers 401 to any POST, PUT, PATCH or DELETE
without a matching `X-Doc-Api-Token`, compared in constant time, and logs `security_token_refused`
with the method and path, never a value. Reads stay open. The domain clients call `apiFetch`
(`core/api/_base.ts`), which adds the header to writes only (`core/launchToken.ts`, tested), so
reads cost no extra preflight. With `DOC_API_TOKEN` unset nothing is checked: `just api`, the test
suite (`conftest` removes a shell's token) and the headless image run as before.

**Why the shell accepts a preset token.** The RG-012 release check launches the installed app and
drives its API from PowerShell, outside the window; a token only the window knew would fail the
run as if the app were broken. The harness now sets a random `DOC_API_TOKEN` before launching and
sends it on its two writes (a test holds that). Whoever can set the shell's environment already
runs as the user.

**Checked.** 8 API tests (401 on `/add`, `/ingest`, `DELETE`; a wrong, empty or case-changed token
refused; the token reaches the route; reads and preflights pass; the 401 carries CORS headers;
unset means unchecked; the refusal log holds no token), 4 frontend tests, the harness test, `cargo
check`. Live in `tauri dev`: with no token set the app loads and a write succeeds; with one preset
on both sides, a POST without the header gets 401 from the window and from outside (`/add`,
`/ingest`, `DELETE`, `PATCH`), the app's own write succeeds, and `/api/health` stays open. **Not
checked:** the installed build (a walkthrough row) and the RG-012 sandbox run (the next release).

**Rejected.** Guarding reads too: CORS already stops a page reading answers and S-5 stops it
becoming same-origin, while every figure, health poll and page image would pay a preflight. A
`BaseHTTPMiddleware`: pure ASGI leaves the chat's event stream untouched. Wrapping `window.fetch`
globally: an explicit `apiFetch` can be read and tested. Failing closed when unset: the dev loop
and the suite have no shell to mint one.

**What it opens.** S-7 is the current step. S3's "arbitrary path" half: `/add` and `inspect` still
take any path, now only from the app's own window.
