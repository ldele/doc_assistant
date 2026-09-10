<!-- status: active · updated: 2026-09-10 (§0: the checklists are assumed stale and preflight checks they were touched since the last tag) · class: runbook -->

# Release runbook

How to cut a release of `doc_assistant` / **Provenote**, and the traps that have actually caught us.
Packaging mechanics (freeze, bundle, RG-012 harness) live in
[`docs/desktop-packaging.md`](desktop-packaging.md); this file is the **order of operations** and
the judgment steps.

> **The one rule this whole file exists to enforce**
>
> **Never ship an artifact you have not rebuilt from the commit you are tagging, and never treat a
> green source tree as evidence about a frozen binary.**
>
> KI-34 shipped a build that could not read a single PDF. Every test passed, every lint passed, and
> a source-install clean-room run passed — because the missing PyMuPDF data file is read at
> *extraction* time, not import time. Only installing on a clean machine could find it.

---

## The short version

```bash
# 0. refresh the two checklists FIRST — they are assumed stale (§0); preflight fails otherwise
git log $(git describe --tags --abbrev=0)..HEAD --oneline -- apps/desktop/src apps/api src   # what shipped
#    -> add a row per new/changed surface to docs/release-ux-checklist.md, move docs/security.md steps
#       that landed to "done", bump both `updated:` headers

# 1. mechanical checks (fast, read-only, run it early and often)
uv run --no-sync python -m scripts.release_preflight

# 2. gates
uv run --no-sync pytest -q
uv run --no-sync mypy src
npm --prefix apps/desktop test && npm --prefix apps/desktop run check
uv run --no-sync python tools/conventions/rungate.py docs_check --root . --strict
uv run --no-sync pip-audit                                  # security floor (docs/security.md)
npm --prefix apps/desktop audit --audit-level=high
uv run --no-sync bandit -r src apps -c pyproject.toml -q

# 3. rebuild the artifact FROM the release commit  (see desktop-packaging.md §1-4)
uv sync --extra cpu --extra dev --extra packaging   # KI-3: CPU torch only
uv run --no-sync python -m scripts.build_sidecar     # ~11-14 min
npm --prefix apps/desktop exec tauri build           # ~10 min
uv sync --extra cu130 --extra dev                    # restore the dev venv

# 4. clean-machine gate — RG-012 Tier-2 (desktop-packaging.md §5)
# 4b. drive every surface in the installed build — docs/release-ux-checklist.md (§5b below)
# 5. preflight again — it now ties the RG-012 PASS to THIS artifact
uv run --no-sync python -m scripts.release_preflight

# 6. tag, then push
git tag -a vX.Y.Z -m "..."
git push origin main && git push origin vX.Y.Z

# 7. delete the PREVIOUS installer + msi — after the push, never before (§8)
ls apps/desktop/src-tauri/target/release/bundle/{nsis,msi}
```

---

## 0 · Refresh the checklists — they are stale until you prove otherwise

**Assume `docs/release-ux-checklist.md` and `docs/security.md` are out of date.** They are edited
when someone remembers, and nobody remembers while building. A release driven against last
release's walkthrough tests last release. So the first step of every release is to bring them level
with what shipped, and `preflight`'s `checklists` check **fails** if either file has no commit (or
uncommitted edit) since the previous tag — it verifies that the refresh happened, not that it was
good.

The refresh, concretely (10–20 minutes):

1. List what shipped: `git log <previous-tag>..HEAD --oneline -- apps/desktop/src apps/api src`
   and the CHANGELOG's `[Unreleased]` block.
2. **UX walkthrough:** for every new or changed surface, add or amend a row under the right §
   (steps → expect, including the empty and failed states); delete rows for removed surfaces; add
   the new release's record table at the foot.
3. **Security:** move the steps that landed to `done` in §4 and §5, re-rank §3 if a finding closed
   or a new one was filed, and re-check §2's "holds" list against the diff (a new route, a new
   `{@html}`, a new capability, a new dependency each need a line).
4. Bump both `updated:` headers with one line saying what changed.

Then the rest of this file.

## 1 · Version bump

Nine places. `uv.lock` is the one a person forgets; the two Cargo files were missed by *everything*,
including this table — and so was `package-lock.json`, found at **0.4.2** on 2026-09-10 after three
releases had shipped past it.

| File | Field |
|---|---|
| `pyproject.toml` | `version` |
| `uv.lock` | the `doc-assistant` package entry — re-lock, do not hand-edit |
| `src/doc_assistant/__init__.py` | `__version__` — what the running app reports (ADR-044) |
| `apps/desktop/package.json` | `version` |
| `apps/desktop/package-lock.json` | `version` **twice** — top level and `packages[""]`; npm rewrites both only when npm *runs* (`npm install` / `npm audit fix`), so a hand-bumped `package.json` leaves the lock behind. Re-lock, do not hand-edit |
| `apps/desktop/src-tauri/tauri.conf.json` | `version` |
| `apps/desktop/src-tauri/Cargo.toml` | `[package] version` |
| `apps/desktop/src-tauri/Cargo.lock` | the `doc-assistant-desktop` entry — re-lock, see below |
| `CHANGELOG.md` | a dated `## [X.Y.Z]` section |

`preflight` checks all nine version strings (the CHANGELOG has a check of its own), and
`tests/unit/test_version.py` catches `__init__.py` drifting from
`pyproject.toml` at commit time rather than release time. **Why `__version__` matters:** it is what
the update check compares against the newest published release. Stale, and the app compares itself
to a lie — offering an "update" it already is, or staying quiet about one it is not.

**Why the rest matters:** v0.4.0 bumped the others and missed `uv.lock`,
which records the project's own version. CI and the Docker build install with `--locked`, which
*fails* rather than silently re-resolving — so every gate after dependency-install was skipped on
`main` for days, and nobody saw a red build because the job died before the gates ran.

**Why the Cargo files are here.** They sat at `0.4.1` through **v0.4.2, v0.5.0 and v0.5.1** — three
tagged releases — while `preflight` reported the versions green, because it never opened them and
neither did this table. That is the opposite failure from `uv.lock`'s: not a file someone forgot to
edit, but a file nothing was looking at. An agreement check is worth exactly as much as its file
list, so the list is now pinned by `test_every_file_carrying_a_version_is_actually_read` — adding a
version-carrying file to the repo means adding it *there* and *here*, or the next one drifts the
same way.

**The lock has an ordering trap.** Cargo rewrites `Cargo.lock` only when cargo *runs*, which on a
release is during the build — i.e. **after** the release commit. So bump `Cargo.toml` by hand, then
re-lock before committing:

```powershell
cargo update --manifest-path apps/desktop/src-tauri/Cargo.toml -p doc-assistant-desktop --offline
```

`--offline` is deliberate and it works: the lock already pins every dependency, so nothing needs
downloading, and the run touches exactly that one line. **`cargo metadata` is not a substitute** —
it wants metadata for every locked package, Android-only ones included, which this machine has
never had a reason to download; it exits 101 under `--offline` and needs the network without it.
Adding `--no-deps` makes it exit 0 and update nothing, which is worse.

Skip the re-lock and the build writes it for you *after* the commit, `tree_clean` fails with a
one-line `Cargo.lock` diff, and the tag no longer matches what was built — exactly what happened
at 0.6.0.

## 2 · The CHANGELOG is a judgment step, not a formatting step

A script can check that the section exists. It cannot check that it is **true**. Read the previous
release's "Known limits" line by line and ask of each one: *is this still true?*

The 0.4.1 draft claimed "a clean-machine install has not been verified yet" for three days after it
had been verified. That is the failure mode — a limit that silently becomes a lie.

Write for someone deciding whether to install it, not for the commit log:
- What can they now do that they could not before?
- What will still be wrong when they try it? Name it plainly.
- Numbers, with the conditions attached ("27 questions, 97-document library, same prompt and
  retrieval"), never adjectives.

## 3 · Gates

| Gate | Command | Notes |
|---|---|---|
| Python tests | `uv run --no-sync pytest -q` | |
| Types | `uv run --no-sync mypy src` | **never** `mypy --strict src` — different option set, wipes the cache both ways (`.claude/CONTEXT.md` §8) |
| Frontend tests | `npm --prefix apps/desktop test` | `node:test`, pure `lib/**/*.ts` only |
| Frontend types | `npm --prefix apps/desktop run check` | `svelte-check` |
| Docs | `rungate.py docs_check --root . --strict` | |
| Hooks | `uv run --no-sync pre-commit run` | ruff/format/mypy/bandit/secrets |
| Checklists refreshed | `preflight` (`checklists`) | both files touched since the previous tag — §0 |
| Security floor | `uv run --no-sync pip-audit` · `npm --prefix apps/desktop audit --audit-level=high` · `uv run --no-sync bandit -r src apps -c pyproject.toml -q` | the deterministic half of `docs/security.md`; `pip-audit` is still advisory in CI (ROADMAP 61) so read its output here |
| UX/UI walkthrough | `docs/release-ux-checklist.md` | §5b — a person drives every surface in the **installed** build |

> **`pre-commit` can eat your commit.** `ruff-format` **modifies files**, and a hook that modifies a
> file *aborts the commit* while leaving everything staged — which looks exactly like success. If
> you believe you committed and `git log` disagrees, this is why. Re-stage and re-run.

## 4 · Rebuild — and understand what needs rebuilding

| What you changed | Re-freeze sidecar (~11-14 min) | Re-bundle installer (~10 min) |
|---|---|---|
| `src/`, `apps/api/` (Python) | **yes** | yes |
| `apps/desktop/src/` (Svelte/CSS/TS) | no | **yes** |
| docs, tests only | no | no — but then you have nothing to release |

Two traps:
- **CPU torch only** (KI-3). `build_sidecar` refuses to freeze a `+cu*` torch — the `cu130` wheel
  segfaults on a GPU-less box and the installer must run anywhere. **Restore `--extra cu130`
  afterwards**, or your dev box quietly loses its GPU.
- **The dev sidecar does not hot-reload.** Vite HMR updates the frontend instantly, so a
  backend-rendered string keeps showing the OLD text after a Python edit and the change looks
  broken. Restart the sidecar before believing a backend-rendered string (KI-38).

## 5 · The clean-machine gate — RG-012 Tier-2

**This is the gate.** Procedure and its four traps: `docs/desktop-packaging.md` §5.

- Stage the installer as a **copy**; never map `target/release/bundle/nsis` into the sandbox (a
  running sandbox holds a handle and the next `tauri build` fails with os error 32).
- **Windows Sandbox runs one instance, and a stale VM silently eats the run.** Killing
  `WindowsSandboxServer` / `WindowsSandboxRemoteSession` does **not** take the VM down. Wait for
  every `vmmemWindowsSandbox` process to disappear before relaunching, or the `LogonCommand` never
  fires and the gate reports nothing while looking busy. A flat VM working-set is the tell.
- Tier 2 needs an answer engine the sandbox can reach: `OLLAMA_HOST=0.0.0.0:11434` on the host,
  restart Ollama — **and revert it afterwards.**
- **Archive the run before clearing `out\`: copy, *verify the copy*, then delete — as three
  separate commands.** A batched archive-then-delete was rejected as a whole by a path guard on this
  machine, so the copy never ran and the delete did. That destroyed the evidence for a headline
  finding (see KI-35).

`preflight`'s `rg012` check then ties the PASS to **this artifact** by matching the installer build
timestamp the harness logged. A PASS from a previous build is worse than no PASS — it reads as
evidence for something that was never tested.

## 5b · The UX/UI walkthrough — every surface, in the installed build

`preflight` and the suites prove the code; nothing above proves that a person can use the product.
Drive **`docs/release-ux-checklist.md`** end to end on the artifact from §3 — first run on an empty
data home, then the real library, Ollama, $0 — and fill its record table into
`.claude/release-notes-X.Y.Z.md`. Three rules that matter more than the list: drive the failure
branch of every surface (empty · error · unavailable), name the build you drove, and write down what
you did **not** drive. A blocker is a data-loss or a lie; everything else becomes a KI and a line
under "Known limits" (§2). Log the pass in `.claude/REVIEWS.md` row 6.

## 6 · Tag

Tag **after** the gate is green, and verify the tag's source is the source that was tested:

```bash
git diff --stat <tested-commit>..vX.Y.Z -- src apps scripts tests   # must be empty
```

Doc-only commits between the tested build and the tag are fine — that diff being empty is what
makes it fine. If it is not empty, you are tagging something you did not test.

## 7 · Push

`git push origin main && git push origin vX.Y.Z`. Publishing is deliberate and separate: everything
above is reversible, this is not.

**Both halves.** `git push origin main` does **not** push tags. v0.4.1's tag sat only on the build
machine for a day because the second command was skipped — the commits were public and the thing
they were tagged as was not.

## 7b · Cut the GitHub release — this is a step now, not a courtesy

**Since ADR-044 the app checks `releases/latest` and shows the user what it finds.** A pushed tag
with no *release object* behind it is invisible to that check: every install reports "no published
release to compare against" forever, which is honest and useless. Pushing the tag is not publishing.

```bash
gh release create vX.Y.Z apps/desktop/src-tauri/target/release/bundle/nsis/Provenote_X.Y.Z_x64-setup.exe \
  --verify-tag --title "Provenote X.Y.Z" --notes-file .claude/release-notes-X.Y.Z.md
```

Attach the NSIS `.exe` as a release asset, or the link the app hands the user leads to a page with
nothing to install on it. Keep the notes at `.claude/release-notes-X.Y.Z.md` (gitignored; the
release page is the body's home and the CHANGELOG is the record) with a `<!-- status: … -->`
comment on line 1 — `docs_check --strict` scans `.claude/*.md` and fails without it, and GitHub
does not render the comment. The command creates a **draft**, uploads the asset (1.6 GB, ten
minutes or more), then publishes; a draft with zero assets and an `untagged-…` URL mid-way is the
upload in progress, not a failure, and `releases/latest` keeps pointing at the previous release
until the flip. Write the body in the previous release's shape (Install · What's new · Known
limits · How this was verified · SHA-256), and state the RG-012 verdict in two halves — the
packaging half is strong evidence, the citation half is one sample (`RIGOR_TODO` 2026-08-14). And **repeat any "not yet verified" caveat from the tag annotation in the
release body** — the release page is the surface people actually read; a caveat that lives only in
`git show` is a caveat nobody sees.

## 8 · Delete the previous installer

**After the push, never before.** Once the tag is on the remote the artifact is reproducible from
it, so deleting the old one costs nothing but a rebuild; before that, it is the only copy.

```bash
cd apps/desktop/src-tauri/target/release/bundle
rm nsis/Provenote_<PREVIOUS>_x64-setup.exe msi/Provenote_<PREVIOUS>_x64_en-US.msi
```

Both formats — `tauri build` emits an NSIS `.exe` **and** an MSI, and only the `.exe` is what
anyone installs, so the MSI is the one that quietly accumulates.

**Why this is a step and not housekeeping.** A stale installer sitting beside a fresh one is the
condition that once had the RG-012 harness install a **two-month-old build** and report its results
as the new release's. The harness now filters by product name and sorts by build time, and
`preflight`'s `artifacts` check warns when more than one is present — but both of those are
mitigations for a mess that does not need to exist. One version in the directory, and neither
mitigation is ever load-bearing.

**The disk cost is the lesser reason, and still real:** each release leaves **~3.1 GiB** behind
(1,572 MiB exe + 1,563 MiB msi) and nothing prunes it. Three releases had accumulated 9.2 GiB by
0.4.2.

Keep only the version you just tagged. Every earlier one is rebuildable from its tag — that is what
tags are for — and the current one must stay, because `preflight`'s `rg012` check matches the
harness's recorded build timestamp against the installer on disk. Deleting the *current* artifact
does not just lose a file, it loses the evidence that the release passed its gate.

---

## What preflight cannot check

It is a floor, not a verdict. These need a person:

- **Is the CHANGELOG true?** (§2)
- **Are the known limits still limits?** Re-read them each release.
- **Are the answers any good?** Retrieval and citation quality are measured with the eval harness,
  not asserted — and quality claims need a baseline and a control (`evals/README.md`,
  `.claude/RIGOR_TODO.md`).
- **Did anything get worse for a user?** The gates are all pass/fail; none of them notices that an
  answer got less useful.
