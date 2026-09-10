<!-- status: active · updated: 2026-09-10 (§0 + §4: one step per session; §6: the periodic full check and where its log goes) · class: living -->

# Security — the threat model, the plan, the floor, and the periodic check

Provenote is a **local-only** desktop app: a Tauri window talking to a FastAPI sidecar on
`127.0.0.1:8001`, reading the user's own documents, calling either the Anthropic API or a local
Ollama. No server, no account, no second user. Most of the usual web threat model does not apply;
a few things that usually do not matter here do. This file is the whole of the project's security
record: what an attacker can be (§1), what already holds (§2), what is open (§3), **the plan — one
step per session** (§4), the deterministic floor (§5), the **periodic full check and its log** (§6),
what is deliberately not done (§7), and the shape cpc could standardise (§8).

## 0 · How this file is worked

- **One step per session, not all at once** (user, 2026-09-10). §4 is an ordered list; the roadmap
  row 60 names the current step. A step is sized to ride alongside a feature session: a small code
  change, its test, one line here moving it to `done`. Nothing in §3 is fixed outside a step.
- **The findings table (§3) is the backlog, the plan (§4) is the order.** A new finding gets a row
  in §3 and a step in §4; a step that lands moves its row to `done` in §3 and §2 gains a line.
- **When every step is done — or every six months, whichever first — the full check runs** (§6)
  and its result is logged. The log is `.claude/REVIEWS.md` row 7 (append-only, states what was
  *not* covered). That is the "standard log system": one place, one shape, the same as every other
  review in the project.
- **Before a release** the file is assumed stale: `docs/RELEASE.md` §0 refreshes it and
  `release_preflight` fails if it has no commit since the previous tag.

## 1 · Threat model

| # | Attacker | Reach | Example |
|---|----------|-------|---------|
| T1 | **A malicious document in the corpus** | the extractor, the prompt, the answer, the UI | a PDF whose text says *"ignore the sources and state X"*, or a `.md` carrying `<img onerror=…>` |
| T2 | **A web page open in the user's browser** | anything the local API answers without authentication | DNS rebinding: `evil.com` resolves to `127.0.0.1`, so its requests are same-origin and CORS never runs |
| T3 | **Another process as the same user** | the data home: `credentials.json`, `library.db`, the extraction cache | any program the user runs |
| T4 | **A stolen laptop without full-disk encryption** | everything on disk | the research history and the API key, in plaintext |
| T5 | **The supply chain** | whatever a dependency does | a compromised `marked`, a floating CI action tag |

Out of scope on purpose: a hostile LLM provider (the key is theirs to bill; document text already
goes to them by design), network attackers on the Anthropic/Ollama link (HTTPS; `truststore` on
corporate MITM, KI-10), and multi-user servers (the Docker image is a *headless* single-user API
and says so).

## 2 · What holds today (verified 2026-09-10 — keep it that way)

- **Bind is loopback.** `apps/api/__main__.py` defaults `DOC_API_HOST` to `127.0.0.1` with the reason
  in a comment; the compose file publishes `127.0.0.1:8001:8001`.
- **No code-execution or deserialisation primitive anywhere.** Zero hits for `pickle`, `torch.load`,
  `trust_remote_code`, `yaml.load`, `eval`, `exec`, `shell=True`, `os.system` across `src/`, `apps/`,
  `scripts/`, `tools/`. Every `subprocess` call is list-form argv.
- **No SQL from user input.** The one f-string SQL is DDL over a hardcoded migration table
  (`db/migrations.py`); FTS5 quotes every token as a string literal and parameterises the scope
  clause (`sparse_index.py`); Chroma filters are structured dicts.
- **Path validation on the selective-ingest route is the model for the others** —
  `ingest/registry.py` rejects absolute paths, `..`, unknown suffixes and unknown keys by name,
  normalises backslashes first, and splits the composite key *before* validating. It has tests.
- **The Tauri capability set is minimal:** `core:default`, one sidecar spawn, `dialog:allow-open`.
  No `fs`, no `http`, no `shell:allow-open`, no remote IPC, no updater (ADR-044). CSP is set:
  `default-src 'self'`, network only to the sidecar. **Pinned by
  `tests/unit/test_desktop_security_config.py`.**
- **Secrets.** `.env` is ignored and was never committed; `credentials.py` is the single disk path
  for an in-app key, masked to four characters in logs and in `/api/setup`; no route echoes a key;
  `detect-secrets` runs in pre-commit and CI against a committed baseline.
- **The LLM proposes, never writes** — verified in code: every site that parses LLM JSON coerces
  into a validated structure; no LLM output becomes a path, a URL, a shell argument or SQL.
- **Logs go to stderr only**, never to a file; no query, answer or document text is logged.
- **Pins:** `uv.lock` + `uv sync --locked`, `package-lock.json` + `npm ci`, `Cargo.lock`; Docker base
  tags pinned with a reason; the frozen build runs with `HF_HUB_OFFLINE=1` and bundled weights.
- **The gates cover the boundary** (since 2026-09-10): ruff, mypy and bandit run on `apps/` and
  `scripts/`, not just `src/`; `npm audit --audit-level=high` runs in CI.

## 3 · Findings — the backlog, ranked by reach under this model

| # | Finding | Where | Why it matters here | Step |
|---|---------|-------|---------------------|------|
| S1 | **No request authentication on the local API** — CORS is the only origin control, and DNS rebinding (T2) bypasses CORS entirely | `apps/api/main.py` | chained with S3, a web page can copy any readable file into the library, index it, and read the chunks back | S-5, S-6 |
| S2 | **LLM output is rendered as raw HTML** — `marked.parse` → `{@html}`, no sanitiser; `marked` ≥ 14 no longer filters `javascript:` | `apps/desktop/src/lib/chat/Markdown.svelte` | T1: a quoted passage carries markup into the DOM. The CSP contains it in the packaged app; the dev loop has **no CSP** (`devCsp` unset). Residual even with CSP: `<form action>` / `<a href>` exfil and UI spoofing (`form-action`, `base-uri`, `object-src` do not inherit from `default-src`) | S-3, S-4 |
| S3 | **`POST /api/documents/inspect` and `/add` accept arbitrary absolute paths** (by design — the picker sends them), and `inspect` walks a directory with no count/size cap | `apps/api/routers/sources.py` · `library/add.py` | with S1 this is the exfil path; alone it is an unbounded walk + hash of `C:\` on a request thread | S-2 (cap), S-6 (auth) |
| S4 | **No size / page / archive-entry cap on ingest** — EPUB, DOCX, ODT are zip archives opened with no decompressed-size accounting | `extractors.py` | T1: a zip bomb in the corpus is unbounded memory; the cheapest local DoS | S-1 |
| S5 | **Document text enters the prompt with no data/instruction boundary** | `prompts.py` · `pipeline.py` | T1: *"ignore prior instructions"* inside a passage reads like evidence. Must not reintroduce a bracket-shaped delimiter; moves `prompt_version` | S-9 (eval-gated) |
| S6 | **`pip-audit` cannot fail CI** (`continue-on-error`); 66 advisories / 16 packages on 2026-09-07 | `.github/workflows/ci.yml` | T5: a green check that never fails is not a control | S-8 |
| S7 | ~~`apps/` outside every gate~~ | `ci.yml`, `.pre-commit-config.yaml` | — | **done 2026-09-10** |
| S8 | **No Dependabot, no CodeQL; CI actions float on major tags**; `npm audit` was missing | `.github/` | T5 | `npm audit` **done 2026-09-10**; the rest S-12 (user's call) |
| S9 | **The source viewer opens whatever `source_original` says**, no root containment | `library/documents.py` · `source_view.py` | composes with S3: any row is a readable file | S-7 |
| S10 | **Plaintext at rest** — `credentials.json` (best-effort `chmod`, a no-op on Windows), the conversation store, the extraction cache | data home | T3/T4 — a design choice for a local-first app | ADR-011 v2 (ROADMAP 64); README says "use disk encryption" |
| S11 | Small: history export to a fixed temp name · `explorer` via `PATH` · `withGlobalTauri: true` · absolute paths logged at INFO · model downloads not pinned to a `revision` | various | low | S-10 |
| S12 | **Nothing in the app log says "a security control fired"** — a refused host, a rejected path, a cap hit are silent, so the periodic check (§6) cannot read them | `logging_config.py` + each control | the check has nothing to look at except the code | S-11 |

## 4 · The plan — one step per session, in this order

Each step is one session's security slot. `Done when` is the test that lands with it. The roadmap
row 60 names the current step; the DEVLOG entry of the session that does it is the record.

| Step | What | Size | Done when | Status |
|------|------|------|-----------|--------|
| **S-1** | **Ingest size caps** (S4): a byte cap in `get_format_status`; a compressed/uncompressed ratio check before EPUB/DOCX/ODT are opened; the refusal is a sentence in the add review sheet, not a crash | small | a file over the cap and a synthetic zip bomb are each refused by name in a test; the walkthrough §1 gains the row | **next** |
| S-2 | **Walk cap on `inspect`** (S3): `expand_paths` stops at N files / M bytes and returns 400 naming the cap | small | a test with N+1 files gets the 400 and the message names N | planned |
| S-3 | **Sanitise the one `{@html}`** (S2): `DOMPurify.sanitize` before `{@html}` in `Markdown.svelte`; set `devCsp` to the production CSP so the dev loop stops being unprotected | small (one dep, one line) | a `.md` document carrying `<img onerror>` and `[x](javascript:…)` renders as inert text in dev **and** in the installed build; walkthrough §3 gains the row | planned |
| S-4 | **CSP residual** (S2): append `form-action 'none'; base-uri 'none'; object-src 'none'`; extend `test_desktop_security_config.py` | small | the test asserts the three directives; the installed build still loads pages and figures | planned |
| S-5 | **Host guard** (S1a): `TrustedHostMiddleware` with `DOC_API_ALLOWED_HOSTS` defaulting to `127.0.0.1,localhost`; Docker sets its own | small | `create_app` returns 400 to a foreign `Host` in a test; the desktop still works | planned |
| S-6 | **Launch token** (S1b): the Tauri shell mints a random token, passes it to the sidecar via env, the frontend sends it in a header; mutating routes require it | medium — touches `lib.rs`, `__main__.py`, `core/api`, one dependency | a request without the header gets 401 on `/add`, `/ingest`, `DELETE`; the walkthrough still passes | planned |
| S-7 | **Source-viewer containment** (S9): refuse to open a `source_original` under no registered `SourceRoot` | small | a test with a row pointing outside every root gets the "unavailable" sentence, not the file | planned |
| S-8 | **Advisories** (S6) — three sub-steps if needed: (a) upgrade what the lock allows, (b) `--ignore-vuln` with a reason per entry, (c) drop `continue-on-error` | one session (ROADMAP 61) | CI red on a new HIGH; the ignore file has a reason per line | planned |
| S-9 | **Prompt fence** (S5): `<source n="1" file="…">…</source>` per passage + one system line; **eval-gated** — run the public 10 before/after, record the baseline | medium | `prompt_version` moves; the baseline file exists; a test asserts the fence is present | planned — with an eval session |
| S-10 | **The small ones** (S11): `mkstemp` for the history export; `%WINDIR%\explorer.exe`; `withGlobalTauri: false` + the dialog plugin package; `revision=` in the model registry; a bug-report note about paths in console captures | small | each has a one-line test or a config assertion | planned |
| S-11 | **Security events in the app log** (S12): structlog events `security_host_refused`, `security_path_refused`, `security_cap_hit`, `security_markup_sanitised`, each with the control's name and never the content — what §6 reads | small | the four events exist and each control emits its own in a test | planned |
| S-12 | **Dependabot (pip · npm · cargo · actions) + SHA-pinned actions** (S8) | config only | bot PRs arrive; `uses:` lines are SHAs | **user's call** |
| **Full** | **The periodic full check** (§6), first run when S-1 … S-11 are done | one session | a dated entry in `.claude/REVIEWS.md` row 7 | after S-11 |

## 5 · The floor — deterministic, runs without a person

| # | Check | Command | Status |
|---|-------|---------|--------|
| 1 | Gates cover the boundary, not just the library | `ruff check src/ tests/ apps/ scripts/` · `mypy src/ apps/` · `bandit -r src/ apps/ -c pyproject.toml` (CI + pre-commit) | done 2026-09-10 |
| 2 | Tauri config guard | `pytest tests/unit/test_desktop_security_config.py` | done 2026-09-10 |
| 3 | JS tree audited | `npm audit --audit-level=high` (CI, after `npm ci`) | done 2026-09-10 |
| 4 | Secrets | `detect-secrets scan --baseline .secrets.baseline` (CI + pre-commit) | done (since 2026-07) |
| 5 | Python tree audited, **blocking** | `pip-audit` with a reviewed `--ignore-vuln` list | S-8 |
| 6 | Host guard test | `create_app` rejects a foreign `Host`; `DOC_API_HOST` defaults to loopback | S-5 |
| 7 | Path-confinement tests on every route that opens a file | `pytest tests/unit/api/test_path_confinement.py` | S-2, S-7 |
| 8 | Prompt fence present | a unit test over the rendered answer prompt | S-9 |
| 9 | Security events exist and fire | a unit test per control | S-11 |
| 10 | Release: this file and the walkthrough were touched since the last tag | `release_preflight` (`checklists`) | done 2026-09-10 |

The `sprint-close` keypoint (`scripts/conventions.toml`) names rows 1–5 as its checklist; when the
planned rows land, the whole table becomes the `run` list of a `cpc-security-check` gate (§8).

## 6 · The periodic full check — and its log

**When.** After S-11 lands, then every six months (`.claude/REVIEWS.md` row 7 cadence), and after
any change that adds a route, a `{@html}`, a capability, a parser, or a network call.

**How** — read-only, one session, the same eleven lenses the 2026-09-10 review used, each with the
command that answers it. Write the answer down even when it is "clean":

| Lens | Question | How to answer it |
|------|----------|------------------|
| A | Where does the API bind, and who can call it? | `rg -n "DOC_API_HOST|allow_origins|TrustedHost|Depends\(" apps/api`; compose `ports:` |
| B | Which routes take a path or serve a file, and is each confined to a root? | `rg -n "FileResponse|Path\(|rglob|open\(" apps/api src/doc_assistant/library` — one row per route |
| C | Where is untrusted text rendered as HTML? | `rg -n "@html|innerHTML" apps/desktop/src` — must be exactly the sanitised path |
| D | Tauri: CSP, capabilities, remote IPC, updater | `pytest tests/unit/test_desktop_security_config.py -q`; read `capabilities/*.json` |
| E | Secrets: loaded where, logged where, echoed where | `rg -n "ANTHROPIC_API_KEY|api_key|key_hint" src apps`; `git log --all -- .env` is empty |
| F | Any exec / deserialisation primitive? | `rg -n "pickle|torch.load|trust_remote_code|yaml.load\(|eval\(|exec\(|shell=True|os.system" src apps scripts tools` — expect zero |
| G | Any SQL built from input? | `rg -n "execute\(f|\.format\(|% " src/doc_assistant --glob "*.py"` near `execute` |
| H | Parsers and caps | the cap constants exist and have tests; `bandit -r src apps -c pyproject.toml -q` |
| I | Prompt injection and every LLM-output sink | `rg -n "json.loads" src/doc_assistant` — each result validated, none becomes a path/URL/arg |
| J | Supply chain | `pip-audit`; `npm audit`; `uses:` pinned; Docker base tags; `HF_HUB_OFFLINE` in the frozen build |
| K | Logging and PII | `rg -n "log\.(info|warning|error)\(" src` for `path=|query=|text=`; `security_*` events present |

**The log.** One dated entry in `.claude/REVIEWS.md` under row 7, in the file's standard shape:
**Covered** (the lenses, with the commands as run) · **Found** (new §3 rows, or "none") · **NOT
covered** (which lens was skipped and why). The ledger row's `Last full review` and `Next due`
move. A check whose omissions are not written down is not a check.

## 7 · Deliberately not done (say it, do not hide it)

- **No encryption at rest.** Local-first, single user; the OS's disk encryption is the control.
- **No sandboxing of extractors.** PyMuPDF/lxml/python-docx run in-process; a crash is a sidecar
  restart on next launch, not a session loss.
- **No auth between the app and Ollama.** Same machine, same user, by Ollama's own design.
- **No rate limiting.** One user, one window.

## 8 · Lifting this into cpc (proposal, 2026-09-10 · cpc ticket T-012)

Three tables a project fills in and one gate that reads them: a **threat model** (attacker · reach ·
example, with an explicit out-of-scope list), a **findings backlog** with a step column (a finding
that is not a step is a wish), and a **floor** with a status column (the gate is only as good as
its file list). The gate, `cpc-security-check`, runs `[security] run = […]` from
`scripts/conventions.toml` (the same shape as keypoint extras), fails on non-zero, and prints the
open steps as the judgment checklist. It lives at `sprint-close` and a future `release-close`
keypoint, never in a commit hook — the 2026-08-19 decision stands. The periodic check's log is the
project's `REVIEWS.md` ledger, not a new file.
