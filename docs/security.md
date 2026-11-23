<!-- status: active · updated: 2026-11-23 · class: living -->

# Security

Provenote is a local-only desktop app: a Tauri window talking to its engine (FastAPI) on
`127.0.0.1:8001`, reading the person's own documents and calling either the Anthropic API or a
local Ollama. No server, no account, no second user. This file is the project's whole security
record: who can attack it (§1), what holds (§2), what is open (§3), the order it is fixed in (§4),
the checks that run without a person (§5), the periodic full check (§6), what is not done on
purpose (§7), and what was fixed (§8).

## 0 · At a glance (2026-11-23)

- **Open: 6 findings** (§3). Supply chain: S16, S17, S25 (T5). Low: S23, S24 (T6, T1). By design:
  S10 (T3, T4). Nothing reachable from a web page is open since S-17 (2026-11-23).
- **Current step:** S-16 with a Tauri update, in its own session before release A's tag (A9). Not
  yet placed: S-15.
- **Due:** the first periodic full check (§6). Its precondition, S-11, landed on 2026-11-23.
- **Last looked at:** §2 re-checked for 0.7.3 on 2026-11-11; the MCP path and A6's lookups on
  2026-11-23 (a narrow review, logged in `.claude/REVIEWS.md` on the dev machine).

**How this file is worked.** One security step per session (the owner, 2026-09-10); ROADMAP row 60
names the current one. A new finding gets a row in §3 and a step in §4. When its step lands, the
finding leaves §3 for one line in §8, its control joins §2, and the full account is the DEVLOG
entry of the session that did it. Before a release the file is assumed stale: `docs/RELEASE.md` §0
refreshes it, and `release_preflight` fails if it has no commit since the previous tag. IDs are
never reused: `T` for an attacker, `S` for a finding, `S-` for a step.

## 1 · Threat model

| # | Attacker | Reach | Example |
|---|----------|-------|---------|
| T1 | **A malicious document in the library** | the extractor, the prompt, the answer, the window | a PDF whose text says *"ignore the sources and state X"*, or a `.md` carrying `<img onerror=…>` |
| T2 | **A web page open in the person's browser** | anything the engine answers without the launch token | DNS rebinding: `evil.com` resolves to `127.0.0.1`, so its requests are same-origin and CORS never runs |
| T3 | **Another program run by the same user** | the data home: `credentials.json`, `library.db`, the extraction cache | any program the user runs |
| T4 | **A stolen laptop without disk encryption** | everything on disk | the research history and the API key, in plaintext |
| T5 | **The supply chain** | whatever a dependency or a build tool does | a compromised `marked`, a floating CI action tag |
| T6 | **An AI program connected through MCP** (ADR-066; off until turned on) | the nine read tools (`read_surface/`): passages, records, and conversations if their switch is on; no writes | its model may hold tools this app never gives (the web, files, messages), so text planted in a document can make *it* act |

Out of scope on purpose: a hostile LLM provider (document text goes to it by design, and the key
is its to bill), a network attacker on the Anthropic or Ollama link (HTTPS; `truststore` behind a
corporate proxy, KI-10), and a server with several users (the Docker image is a headless
single-user API and says so).

## 2 · What holds

Each line is a control as it stands, with the test that pins it where one exists.

**The engine's API (T2)**

- **It binds to loopback.** `DOC_API_HOST` defaults to `127.0.0.1` (`apps/api/__main__.py`); the
  compose file publishes `127.0.0.1:8001:8001`.
- **It answers only its own host names** (S-5). A foreign `Host` gets 400 and logs
  `security_host_refused` (`apps/api/host_guard.py`); a blank `DOC_API_ALLOWED_HOSTS` means
  loopback, never any host. `test_api_host_guard.py`.
- **A write needs the launch token** (S-6). The shell mints 32 random bytes per launch and hands them
  to the engine and the window; every POST, PUT, PATCH and DELETE without `X-Doc-Api-Token` gets
  401, logged as `security_token_refused`. A rule by method, so a new write route is covered
  without a line of its own. Unset (the suite, `just api`, the headless image), nothing is checked;
  a token already in the shell's environment is used, so the RG-012 harness can send one.
  `test_api_launch_token.py`. It holds because no GET writes or calls out: a scan, the update
  check and the model-server probe are POSTs (S-17), and
  `test_get_routes_have_no_side_effects.py` holds every GET route to it.
- **Routes that reach outside data stay narrow.** `POST /api/libraries/inspect` opens a folder's
  database read-only (SQLite URI mode) and answers counts, never contents. A re-run takes its files
  from the engine's own record, never from the request. The model-list call is a POST because it
  sends a stored key, and a key goes only to the address it was saved with (ADR-060).
- **An unhandled error is a fixed sentence.** The 500 is answered inside CORS
  (`apps/api/server_errors.py`) with Starlette's fixed text, never a traceback.

**The window (T1)**

- **The CSP lets in only the app and its engine.** `default-src 'self'`, network to the engine
  only, and `form-action`, `base-uri` and `object-src` set to `'none'` (S-3, S-4). The dev server
  sends the same policy (`devCsp.ts`), since `tauri dev` loads Vite directly and Tauri adds the
  policy only to what it serves. `test_desktop_security_config.py`, `devCsp.test.ts`.
- **Model output is sanitised before it becomes HTML.** DOMPurify on the one `{@html}`
  (`Markdown.svelte`), with `style` and form controls forbidden; the shared tooltip sets
  `textContent`.
- **A link in an answer is clickable only when a passage vouches for it.** A web link stays a link
  only when an unmarked passage the answer was given carries its address; any other shows as text
  (A2, `links.test.ts`). A record answer (A6) carries no such passage, so its links show as text.
- **The window can do little natively.** Its capabilities are `core:default`, `dialog:allow-open`
  and four title-bar commands (drag, minimise, maximise, close; ADR-065): no file system, network,
  shell, remote IPC or updater (ADR-044). The engine is started from Rust by path, so the window
  runs nothing (KI-73). The app's own commands are `api_token` (the launch token, which the window
  needs), `library_state` (reads its setting) and `open_library` (only a folder holding
  `library.db`, ADR-061). Pinned by `test_desktop_security_config.py`.

**Documents and stored paths (T1, T3)**

- **Ingest has caps** (S-1). A file over 1 GB, an archive that would expand past 1 GB, or an entry
  compressed 1000:1 or more is refused by name before it is opened, and logs `security_cap_hit`
  (`extractors.ingest_refusal`). `test_ingest_size_caps.py`.
- **One add names at most 50,000 files** (S-2, `MAX_ADD_FILES`), counted while walking a folder;
  `GET /api/documents/accepts` serves the limits so the add sheet can state them.
- **A stored path is used only where its kind of file belongs** (S-7, S21). A cache file under the
  cache folder, a figure under the figures folder, a source under the sources folder, its source
  folder or a registered root. Anything else is refused, logged as `security_path_refused`, and
  left alone (`stored_paths.py`). `test_stored_paths_confined.py`, `test_stored_paths.py`,
  `test_chunk_context.py`.
- **The selective-ingest route validates its keys.** `ingest/registry.py` rejects absolute paths,
  `..`, unknown suffixes and unknown keys by name, after normalising backslashes.

**What reaches a model (T1, T6)**

- **Each passage is fenced.** It sits inside `<source n="…">`; a fence tag inside a passage is
  defused and logged as `security_markup_sanitised` (S-9). `test_pipeline_helpers.py`,
  `test_security_events.py`. The fence does not stop a small model obeying planted text (§7).
- **Text written to an AI is marked.** In each passage an answer is given, and once per document
  after extraction, so the Library shows a marked document before any question (A2, A2b).
  `test_source_checks.py`.
- **The LLM proposes, never writes.** Every site that parses LLM JSON coerces it into a validated
  structure; no LLM output becomes a path, a URL, a shell argument or SQL.
- **MCP** (A3, A4). Off until the person turns it on; its own bearer token; a request carrying an
  `Origin` is refused; the conversation tools are hidden while their switch is off. Arguments are
  checked against each tool's schema, and a failing tool answers without its error text (S22).
  Passages are fenced and checked, and so are captions, reference lines and cited excerpts (S23).
  A passage id carries its text's version, and an older one finds nothing (S19). The call log keeps
  the tool, a count and a time, never the arguments. `test_api_mcp.py`,
  `test_read_surface_evidence.py`.

**Secrets**

- `.env` is ignored and was never committed. `credentials.py` is the one disk path for an in-app
  key, masked to four characters in logs and in `/api/setup`; no route echoes a key.
- `detect-secrets` runs in pre-commit on staged files and in CI over every tracked file, and fails
  on a secret `.secrets.baseline` does not record (S-13). The CI gate makes no network call and
  prints a finding's type and place, never its value. Both run in Python's UTF-8 mode (S-14).

**Code and the supply chain (T5)**

- **No code-execution or deserialisation primitive.** No `pickle`, `torch.load`,
  `trust_remote_code`, `yaml.load`, `eval`, `exec`, `shell=True` or `os.system` in `src/`, `apps/`,
  `scripts/` or `tools/`; every `subprocess` call is list-form, and Explorer is run by its full path.
- **No SQL from input.** f-string SQL is built from constants and placeholders only; FTS5 quotes
  every token and parameterises the scope; Chroma filters are dicts.
- **Everything is pinned.** `uv.lock` with `uv sync --locked`, `package-lock.json` with `npm ci`,
  `Cargo.lock`, Docker base tags. The embedders and the reranker load a pinned commit, and the
  frozen build runs offline (`HF_HUB_OFFLINE=1`) on those same snapshots, bundled (S-10).
- **An unreviewed advisory fails CI.** `scripts/pip_audit_gate.py` fails on any Python advisory not
  reviewed in `pip-audit-ignore.toml`, and fails closed when it cannot read either (S-8);
  `npm audit --audit-level=high` runs in CI. The Rust tree and the packaging extra are audited by
  nothing (S16).
- **CI's actions are pinned to commits** (S-12). Dependabot proposes updates for actions only;
  Python, npm and Rust upgrades stay in sessions, where the audits catch an advisory (the owner,
  2026-10-18). `test_ci_actions_pinned.py` fails on a tag. Dependabot *alerts* are the owner's GitHub setting:
  off (404) when read on 2026-10-18 and on 2026-11-23.
- **The gates cover the boundary but `scripts/`.** In CI, ruff runs on `src/`, `tests/`, `apps/` and
  `scripts/`, mypy and bandit on `src/` and `apps/` only (S25).

**Logs and the installer**

- **Logs go to stderr, never to a file.** No query, answer or document text is logged (read
  2026-09-10); absolute paths are (§7). Each control above logs a `security_*` event with its name
  and never the content (S-11).
- **The installer reaches outside its folder twice.** It ends `doc-assistant-api.exe`, and in
  `%TEMP%` it deletes a `_MEI*` folder only if it holds our reranker model (KI-80).
  `test_the_installer_clears_what_an_older_engine_left`.

## 3 · Open findings

| # | Finding | Threat | Step | Placed |
|---|---------|--------|------|--------|
| S16 | `Cargo.lock` and the packaging extra are audited by nothing | T5 | S-16 | after A8″, before A9 |
| S17 | CI installs the latest uv, and the workflow has no `permissions:` | T5 | S-15 | not placed |
| S25 | `scripts/` is not type-checked or scanned by bandit | T5 | none yet | not placed |
| S23 | Titles and authors reach a client's model without the text check | T6 | none yet | release B, with B2 |
| S24 | A record answer renders document metadata as Markdown, unescaped | T1 | none yet | release B, with B2 |
| S10 | The key, the conversations and the extraction cache are plaintext on disk | T3, T4 | ROADMAP 64 | by design (§7) |

- **S16** (2026-11-02). 445 crates, and PyInstaller with its hooks, `pefile`, `altgraph` and
  `pywin32-ctypes`, outside CI's `--extra cpu --extra dev`. The bootloader and hooks go into the
  installer. Audited once by hand on 2026-11-23 (`cargo audit`, RustSec copy of 2026-11-15): two
  advisories in `quick-xml` 0.39.4 (RUSTSEC-2026-0194, -0195: slow or unbounded parsing of hostile
  XML), which reaches the app through `plist` and `tauri-utils` and parses Tauri's own
  configuration, never a document; and nine warnings (six crates unmaintained, three unsound:
  `anyhow`, `glib` on Linux only, `tokio`'s Unix `AsyncFd`). S-16's CI gate needs a reviewed list
  for these, as `pip-audit-ignore.toml` is for Python.
- **S17** (2026-11-02). `setup-uv` is pinned to a commit but installs `version: "latest"`; the
  Docker image pins its uv.
- **S25** (2026-11-02, given a number 2026-11-23). 297 mypy errors and one Medium bandit finding in
  `scripts/` that day. The scripts run on the dev machine and in CI, not in the installer.
- **S23** (2026-11-23). Captions, reference lines and excerpts carry the check since A8′. Titles and
  authors are extracted text too, and the author scan can store a sentence of body text (KI-89).
- **S24** (2026-11-23). Links already show as text and the CSP blocks remote images, so what is left
  is formatting that could imitate the app.
- **S10**. `credentials.json` gets a best-effort `chmod`, a no-op on Windows. The README asks for
  disk encryption; an OS keychain for the key is ROADMAP 64, parked on the keyring decision.

## 4 · The plan

Each step is one session's security slot; its `Done when` is the test that lands with it. In order:

| Step | What | Done when | Rides in |
|------|------|-----------|----------|
| S-16 | Audit the two trees (S16): `cargo audit` in CI; `--extra packaging` in CI's sync; with it, Tauri updated | CI fails on an unreviewed advisory in either | its own session after A8″, before A9 (the owner, 2026-11-23) |
| Full | The periodic full check (§6), the first since S-1 to S-11 | a dated entry in `.claude/REVIEWS.md` row 7 | due, not placed |
| S-15 | Pin CI's uv and its permissions (S17): a `version:` under `setup-uv`; `permissions: contents: read` | `test_ci_actions_pinned.py` fails on `latest` or a missing `version:` | not placed |

## 5 · The floor: checks that run without a person

| # | Check | Command | Since |
|---|-------|---------|-------|
| 1 | Gates cover the boundary (CI) | `ruff check src/ tests/ apps/ scripts/` · `mypy src/ apps/` · `bandit -r src/ apps/ -c pyproject.toml`; pre-commit runs mypy and bandit on `src/` only | 2026-09-10 |
| 2 | The window's configuration | `pytest tests/unit/test_desktop_security_config.py` | 2026-09-10 |
| 3 | JavaScript tree audited | `npm audit --audit-level=high` (CI, after `npm ci`) | 2026-09-10 |
| 4 | Secrets, blocking | `detect-secrets-hook` (pre-commit) · `python -m scripts.secret_scan_gate` (CI, `just secret-scan`) | 2026-10-01 |
| 5 | Python tree audited, blocking | `python -m scripts.pip_audit_gate` (CI, `just audit`, the sprint-close keypoint) | 2026-09-30 |
| 6 | Host guard | `pytest tests/integration/test_api_host_guard.py` | 2026-09-30 |
| 7 | Stored paths confined | `pytest tests/integration/test_stored_paths_confined.py tests/unit/test_stored_paths.py tests/unit/test_chunk_context.py` | 2026-11-04 |
| 8 | Fence present, planted text marked, unsourced links shown as text | `pytest tests/unit/test_pipeline_helpers.py tests/unit/test_source_checks.py` · `npm test` (`links.test.ts`) | 2026-11-16 |
| 9 | Release: this file and the walkthrough touched since the last tag | `release_preflight` (`checklists`) | 2026-09-10 |
| 10 | MCP: off until on, its own token, no web page, arguments checked | `pytest tests/integration/test_api_mcp.py` | 2026-11-17 |
| 11 | Passages served are current, separate, honestly placed, gone once removed, stable | `pytest tests/integration/test_read_surface_evidence.py` | 2026-11-23 |
| 12 | Each control logs its event | `pytest tests/unit/test_security_events.py` | 2026-11-23 |
| 13 | No GET route writes or calls out | `pytest tests/integration/test_get_routes_have_no_side_effects.py` | 2026-11-23 |

The sprint-close keypoint (`scripts/conventions.toml`) names rows 1 to 5 as its checklist. cpc
ticket T-012 proposes one gate that runs the whole table.

## 6 · The periodic full check

**When.** Due now (S-11 landed 2026-11-23), then every six months (`.claude/REVIEWS.md` row 7), and
after any change that adds a route, a `{@html}`, a capability, a parser or a network call.

**How.** Read-only, one session. Each lens is a question and the command that answers it; write the
answer down even when it is "clean":

- **A · Where does the API bind, and who can call it?**
  `rg -n "DOC_API_HOST|allow_origins|TrustedHost|Depends\(" apps/api`; compose `ports:`.
- **B · Which routes take a path or serve a file, and is each confined?**
  `rg -n "FileResponse|Path\(|rglob|open\(" apps/api src/doc_assistant/library`, one line per route.
- **C · Where is untrusted text rendered as HTML?**
  `rg -n "@html|innerHTML" apps/desktop/src`: exactly the sanitised path.
- **D · Tauri: CSP, capabilities, remote IPC, updater.**
  `pytest tests/unit/test_desktop_security_config.py -q`; read `capabilities/*.json`.
- **E · Secrets: loaded, logged and echoed where?**
  `rg -n "ANTHROPIC_API_KEY|api_key|key_hint" src apps`; `git log --all -- .env` is empty.
- **F · Any exec or deserialisation primitive?** Expect zero from
  `rg -n "pickle|torch.load|trust_remote_code|yaml.load\(|eval\(|exec\(|shell=True|os.system" src apps scripts tools`.
- **G · Any SQL built from input?**
  `rg -n "execute\(f|\.format\(|% " src/doc_assistant --glob "*.py"`, reading near each `execute`.
- **H · Parsers and caps.** The cap constants exist and have tests;
  `bandit -r src apps -c pyproject.toml -q`.
- **I · Prompt injection and every LLM-output sink.** `rg -n "json.loads" src/doc_assistant`: each
  result is validated, and none becomes a path, a URL or an argument.
- **J · Supply chain.** `pip-audit`; `npm audit`; `uses:` pinned; Docker base tags; model commits
  pinned; `HF_HUB_OFFLINE` in the frozen build.
- **K · Logging and personal data.** `rg -n "log\.(info|warning|error)\(" src`, looking for
  `path=`, `query=` and `text=`; the `security_*` events are present.
- **L · What an outside program gets (T6).** Read `mcp_server.TOOL_SPECS` and each tool's result
  shape; `pytest tests/integration/test_api_mcp.py tests/integration/test_read_surface_evidence.py`.

**The log.** One dated entry in `.claude/REVIEWS.md` under row 7: **Covered** (the lenses, with the
commands as run), **Found** (new §3 rows, or "none"), **NOT covered** (what was skipped, and why).
The ledger row's `Last full review` and `Next due` move. A check that does not say what it skipped
is not a check.

## 7 · Not done on purpose, and what closed findings left

**Not done, on purpose:**

- **No encryption at rest** (S10). Local-first and single-user: the operating system's disk
  encryption is the control.
- **No sandboxing of extractors.** PyMuPDF, lxml and python-docx run in the engine's process; a
  crash costs a restart of the engine at the next launch, not the person's work.
- **No authentication between the app and Ollama.** Same machine, same user, by Ollama's design.
- **No rate limiting.** One user, one window.
- **No CodeQL.** Left out of S-12 (2026-10-18), and not planned.
- **`withGlobalTauri` stays on** (S11; the owner, 2026-11-23). Turning it off removes
  `window.__TAURI__`, but Tauri 2 injects its IPC object, `window.__TAURI_INTERNALS__`, in every
  window either way (read in Tauri 2.12.4's `manager/webview.rs` and `tauri-codegen`'s
  `context.rs`), so a script in the window keeps the same commands. What stops such a script is the
  CSP, DOMPurify and the capability list (§2). Turning it off would add two runtime npm packages
  and reverse W0's choice (`docs/specs/feature-add-documents.md`). If the window ever loads remote
  content, Tauri's isolation pattern is the control to weigh.

**What closed findings left, accepted:**

- **S2.** DOMPurify keeps links in model output; only a link a passage vouches for stays clickable,
  and clicking it is the person's act.
- **S4.** No page cap; an archive's central directory of millions of entries is bounded only by the
  1 GB file cap.
- **S5.** A planted instruction a model obeys can still change what the answer *says* (a language
  switch, a false claim repeated from its source): with and without the fence, 18 and 19 of 24
  hostile runs held on small local models (A2). Another program's model, maybe with tools, is free
  to ignore the fence and the marks (T6).
- **S9.** A folder registered by a reference add becomes a root, so its files pass the stored-path
  rule; the launch token is what stops a page from adding one.
- **S11.** Absolute paths are logged at INFO; `docs/setup.md` § Reporting a problem asks a person to
  replace them before posting a log.
- **S19.** An id without a version (the chat's own keys, a definition's `passage_id`) still resolves
  by position.

## 8 · Closed

One line per finding; the account is the DEVLOG entry of that date.

| # | What it was | Closed by | DEVLOG |
|---|-------------|-----------|--------|
| S1 | The engine had no request authentication, and DNS rebinding bypasses CORS | S-5, S-6 | 2026-09-30, 2026-10-17 |
| S2 | Model output was rendered as raw HTML, and the dev loop had no CSP | S-3, S-4 | 2026-09-17, 2026-09-20 |
| S3 | `inspect` and `/add` took any path, and `inspect` walked with no cap | S-2, S-6 | 2026-09-16, 2026-10-17 |
| S4 | Ingest had no size or archive cap | S-1 | 2026-09-16 |
| S5 | Document text entered the prompt with no boundary | S-9 (A2) | 2026-11-16 |
| S6 | `pip-audit` could not fail CI | S-8 | 2026-09-30 |
| S7 | `apps/` was outside every gate | floor row 1 | 2026-09-10 |
| S8 | No Dependabot, no `npm audit`, CI actions on floating tags | S-12, `npm audit` | 2026-10-18, 2026-09-10 |
| S9 | Stored paths were opened, deleted and served unchecked | S-7 | 2026-11-04 |
| S11 | Five small ones: export file, Explorer by PATH, unpinned models, paths in logs, the Tauri global | S-10 (A8′) | 2026-11-23 |
| S12 | No log line said a security control fired | S-11 (A8′) | 2026-11-23 |
| S13 | CI's secret scan could not fail | S-13 | 2026-10-01 |
| S14 | On Windows the secret scanner skipped 47 files it could not decode | S-14 | 2026-10-01 |
| S15 | Four GET routes wrote or called out, out of the launch token's reach | S-17 (A8″) | 2026-11-23 |
| S18 | A text re-run left the old passages in the running engine's keyword index | A8′ | 2026-11-23 |
| S19 | A passage id named a position, not a text | A8′ | 2026-11-23 |
| S20 | Two documents with the same file name merged in an answer's passages | A8′ | 2026-11-23 |
| S21 | A chunk's stored cache path was read outside the library | A8′ | 2026-11-23 |
| S22 | MCP answered a wrongly typed argument with a server error | A8′ | 2026-11-23 |
