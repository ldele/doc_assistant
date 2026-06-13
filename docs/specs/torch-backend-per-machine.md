# Spec — Per-machine torch backend (cu130 vs cpu) from one shared `uv.lock`

**Status:** ✅ **Implemented 2026-06-13** (this session) — `pyproject.toml` (extras + `conflicts` + two explicit indexes + `[tool.uv.sources]`), regenerated `uv.lock` (carries `+cpu` and `+cu130`), `.github/workflows/ci.yml`, README / `.env.example` / `docs/decisions.md` / `docs/DEVLOG.md`, and a `justfile`. Verified on the RTX box (see Definition of done). Originally designed + verified against `pyproject.toml`, `docs/decisions.md` §"Cross-machine toolchain", and empirical results on both machines.
**Owner of execution:** Claude Code (pyproject + lock + CI + docs), one PR, after explicit user review of the diff (lock changes are high-blast-radius).
**Pattern reference:** uv "Using uv with PyTorch" guide (conflicting-extras multi-backend pattern); supersedes the `torch-backend = "auto"` decision in `docs/decisions.md` §"Cross-machine toolchain — torch backend auto-detect" (added 2026-06-10).

> **One-line summary.** Replace the no-op `[tool.uv] torch-backend = "auto"` with two conflicting optional-dependency extras (`cpu`, `cu130`), each bound to an explicit PyTorch index. One committed `uv.lock` then holds both wheels; the GPU box runs `uv sync --extra cu130`, the CPU box and CI run `uv sync --extra cpu`. Selection is per-machine and survives `git pull`.

---

## Problem

The project must run unchanged on two **same-OS (Windows)** machines:

- **GPU box** (RTX 4070, CUDA driver 610.47) wants `torch 2.12.0+cu130` so the embedder and cross-encoder reranker use the GPU.
- **CPU-only box** **segfaults (exit 139)** on the `+cu130` wheel. It **must** get `+cpu`. **Hard invariant.**

Both share `sys_platform == 'win32'`, the same Python, and the same architecture, so **no PEP 508 marker can distinguish them** (this is why the prior `sys_platform == 'win32'` cu130 index-pin was doomed — it forced cu130 onto the CPU box → segfault, the failure recorded in `docs/decisions.md` and `docs/DEVLOG.md` 2026-06-02).

The current mitigation — `[tool.uv] torch-backend = "auto"` (pyproject.toml:164-165) — **does not work for this project's workflow**, verified on the user's machines and confirmed by uv docs:

- The uv settings reference states verbatim that `torch-backend` "is only respected by `uv pip` commands"; the PyTorch guide states "At present, `--torch-backend` is only available in the `uv pip` interface." So `torch-backend`/`UV_TORCH_BACKEND` is a **no-op for `uv lock`/`uv sync`/`uv run`**. Accelerator auto-detection runs at `uv pip install` time, never at lock time, and the lock has no "detect at sync" concept.
- Consequently `uv lock` records whatever single wheel the universal resolution picks — here the cross-platform-safe **`+cpu`** wheel — and `uv sync`/`uv run` enforce that one pinned wheel on **every** machine. The GPU box runs CPU torch (`cuda.is_available() == False`) despite the GPU.
- `UV_TORCH_BACKEND=cu130 uv sync --reinstall-package torch` still installs `+cpu` (backend ignored by sync). `UV_TORCH_BACKEND=cu130 uv pip install torch --reinstall` **does** install `+cu130`, but the next plain `uv run` auto-syncs to the lock and reverts torch to `+cpu`. Only `uv run --no-sync` / `UV_NO_SYNC=1` preserves the manual wheel — and that silently drifts the env from the lock.

A single universal lock can only route per-machine through a discriminator recorded *in* the lock. Markers can't discriminate two Windows boxes, and auto-detect isn't recorded in the lock. The only remaining discriminator uv supports is a **user-selected extra**.

The misleading pyproject comment ("One `uv sync` works everywhere — no win32 pin, no per-machine torch swap", pyproject.toml:161) is now known to be false on the GPU box and must be corrected as part of this work.

---

## Options considered

### (a) Conflicting `cpu`/`cu130` extras + explicit per-index `[tool.uv.sources]` + `[tool.uv] conflicts` — **CHOSEN**

Define `[project.optional-dependencies]` `cpu` and `cu130`, each pinning `torch` (and `torchvision` if ever added) to its own **explicit** index (`download.pytorch.org/whl/cpu`, `download.pytorch.org/whl/cu130`); declare the two extras conflicting. uv generates **one universal lock holding both variants**; the chosen extra selects which installs.

- **CPU invariant:** ✅ Held structurally. `cpu` is the default/safe choice; `conflicts` makes "both at once" an install-time error, not a silent mis-resolve. CPU box + CI **always** `--extra cpu`, never `--extra cu130`. (The cu130 wheel is *resolvable* on the CPU box; the invariant is that it is never *selected* there — enforced by discipline + CI defaulting to cpu.)
- **One shared lock:** ✅ Both wheels live in the single committed `uv.lock`.
- **Same-OS:** ✅ Selection is by explicit extra, not marker — works for two identical-marker Windows boxes.
- **Footgun:** **Low.** Durable across `git pull` (selection is a command, not lock state). Residual: each machine must remember its extra (uv has **no** default-extra setting yet — issue #10360 is open). Mitigated by a per-machine alias/justfile (below). `uv sync` is exact, so switching extras cleanly swaps the wheel with no stale variant left behind.

### (b) `torch-backend = "auto"` + `UV_NO_SYNC` + manual per-machine `uv pip install` cu130 — fallback only

Keep `torch-backend = "auto"`; on the GPU box run `UV_TORCH_BACKEND=cu130 uv pip install torch --reinstall` once, then set standing `UV_NO_SYNC=1` so plain `uv run` won't revert it.

- **CPU invariant:** ✅ (CPU box simply never runs the manual cu130 install).
- **One shared lock:** ✅ (lock stays `+cpu`; GPU box overrides only in its venv).
- **Same-OS:** ✅.
- **Footgun:** **High.** `--no-sync` implies `--frozen` and means "project dependencies will be ignored" — the env is **never reconciled to the lock**. A `git pull` that changes the lock (new dep, version bump) silently won't reach the GPU venv; the manual cu130 wheel can mask a lock the code no longer matches. `UV_NO_SYNC` has **no** project-file equivalent (only the CLI flag / env var), so it can't be scoped to one machine via committed config. Any forgotten plain `uv run`/`uv sync` re-reverts. This is a stopgap, not the goal.

### (c) Documented per-machine bootstrap script (no pyproject change)

A `scripts/bootstrap_gpu.ps1` that wraps option (b)'s steps, plus a CPU bootstrap that just runs `uv sync`.

- **CPU invariant:** ✅. **One lock / same-OS:** ✅.
- **Footgun:** **Medium–high.** Encapsulates the ceremony but inherits all of (b)'s drift risk (it *is* (b) behind a script). Better than raw (b) only because it documents the exact commands and is harder to half-apply. Worth shipping as a **thin wrapper around option (a)** (just the right `--extra`), not around (b).

### (d) Split `torch` out of locked deps entirely (unmanaged torch)

Remove `torch>=2.12` as a managed dep; let each machine `uv pip install` its own torch outside the resolver, standing `UV_NO_SYNC` to stop reverts.

- **CPU invariant:** ✅ (CPU box installs cpu). **Same-OS:** ✅.
- **One shared lock:** ⚠️ Torch is no longer *in* the shared lock at all — reproducibility for the most fragile dependency is lost; `sentence-transformers` still needs torch present, so resolution is partly out-of-band. **Footgun: high** (same `--no-sync`/drift family as (b), plus loss of lock coverage for torch). Rejected.

### (e) `[tool.uv] environments` / `required-environments` markers — not applicable

`environments` reduces the solved platform set; `required-environments` expands wheel coverage for build-only packages. Both key off **platform markers**, which cannot distinguish two win32 boxes by GPU presence. **Rejected** — same dead end as the old win32 pin.

### (f) Wait for project-level `--torch-backend`/auto at sync time — not viable

Extending backend selection to `uv add`/`lock`/`sync` is requested-but-unshipped (issue #12994 open; #18157 reports exactly this symptom; no torch-backend entry across the 0.11.x changelog). Don't wait for it.

---

## Decision

**Adopt option (a).** It is the single uv-supported path that satisfies the CPU invariant, the one-shared-lock constraint, and the same-OS constraint with the lowest footgun, and it survives a normal `git pull` workflow. Option (c) ships alongside as a **thin per-machine wrapper around (a)** to remove the "remember the flag" friction. Option (b) is retained **only** as a documented emergency stopgap in `.claude/KNOWN_ISSUES.md`, explicitly flagged as drift-prone.

### `pyproject.toml` changes

Remove the `[tool.uv] torch-backend = "auto"` block (pyproject.toml:156-165) and the misleading comment, and the bare `torch>=2.12` line from `[project.dependencies]` (pyproject.toml:39-42). Replace with:

```toml
[project.optional-dependencies]
# ... existing dev = [...] stays ...

# PyTorch backend variants — mutually exclusive (see [tool.uv] conflicts).
# Pick exactly ONE per machine at sync time:
#   GPU box (RTX, CUDA):   uv sync --extra cu130
#   CPU-only box  +  CI:   uv sync --extra cpu      <-- the safe default; the
#                                                       +cu130 wheel SEGFAULTS on
#                                                       a machine with no usable GPU.
cpu   = ["torch>=2.12"]
cu130 = ["torch>=2.12"]

[tool.uv]
# torch is a transitive dep of sentence-transformers; declared in the extras above
# so [tool.uv.sources] binds (uv only applies sources to direct deps). One shared
# uv.lock carries BOTH the +cpu and +cu130 wheels; the --extra flag selects which
# installs. There is NO marker that can tell our two Windows boxes apart, and
# torch-backend=/UV_TORCH_BACKEND is uv-pip-only (ignored by lock/sync/run) — the
# extra is the only durable per-machine discriminator.
conflicts = [
    [{ extra = "cpu" }, { extra = "cu130" }],
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true        # only torch* resolves here; jinja2 etc. stay on PyPI

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu",    extra = "cpu" },
    { index = "pytorch-cu130",  extra = "cu130" },
]
```

Notes:
- `explicit = true` is mandatory hygiene: it confines those indexes to torch (and torchvision/torchaudio if ever added) so the rest of the tree — `sentence-transformers`, `chromadb`, langchain — keeps resolving from PyPI.
- Windows CUDA torch wheels are self-contained (CUDA runtime bundled; only a compatible NVIDIA driver needed, no system CUDA toolkit, no separate `nvidia-*` PyPI deps as on Linux). So the cu130 extra adds essentially one wheel to the lock — the dual-variant lock stays clean.
- **macOS guard (forward-looking):** PyTorch publishes **no** CUDA wheel for macOS. If a Mac ever joins, the `cu130` extra must be marker-guarded (e.g. `marker = "sys_platform != 'darwin'"` on the cu130 source) or it will fail to install there. Out of scope today (both boxes are Windows) — noted so the next editor doesn't get surprised.

### Regenerating the lock (execution step, GPU box or any box)

```bash
# From a box that can reach both indexes. The lock records BOTH variants.
uv lock
git add pyproject.toml uv.lock
# review diff, then commit per project rule (never auto-commit)
```

---

## Per-machine setup (step by step)

**GPU box (RTX 4070):**
```bash
git pull
uv sync --extra cu130
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   -> 2.12.0+cu130 True
```

**CPU-only box (the segfault box):**
```bash
git pull
uv sync --extra cpu          # NEVER --extra cu130 here
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   -> 2.12.0+cpu False
```

**Removing the "remember the flag" footgun (pick one, per machine — committed where possible):**
- **justfile / Makefile** (committed, repo-shared): `just sync` → on the GPU box the recipe is `uv sync --extra cu130`; everywhere else `uv sync --extra cpu`. Simplest is two named recipes (`sync-gpu`, `sync-cpu`) so the file is identical on every box and the human picks once.
- **Per-machine shell alias** (not committed): GPU box `alias dsync='uv sync --extra cu130'`; CPU box `alias dsync='uv sync --extra cpu'`.
- **Per-machine `%APPDATA%\uv\uv.toml`:** can hold per-machine uv config, but **note** there is no `default-extra` setting (issue #10360 open), and `default-groups` + per-group index for torch is design-incomplete — do **not** rely on a uv.toml to auto-pick the extra. Use it only for unrelated per-machine prefs.

Whichever is chosen, the discipline is fixed: **CPU box and CI always `--extra cpu`; only the GPU box ever passes `--extra cu130`.**

---

## CI implications

CI stays CPU and must fail loudly if the lock drifts or the wrong variant sneaks in:

**As implemented** (`.github/workflows/ci.yml`): one explicit locked sync + a job-level `UV_NO_SYNC`:

```yaml
env:
  UV_NO_SYNC: "1"          # every `uv run` step stays on the env synced below
# ...
run: uv sync --locked --extra cpu --extra dev
```

- `--extra cpu` forces the CPU resolution (CI never selects cu130); `--extra dev` brings the pytest/ruff/mypy/bandit toolchain.
- `--locked` asserts the committed lock is unchanged — a drifted lock fails the job rather than silently re-resolving, so cu130 can never enter CI through a stale lock.
- **`UV_NO_SYNC: "1"` at job level is REQUIRED here (refines the original "never UV_NO_SYNC" note):** without it, a plain `uv run` step on Linux re-resolves torch *without* the cpu extra and pulls the heavy PyPI CUDA wheel. `uv sync` itself ignores `UV_NO_SYNC` (it always syncs), so the one explicit `--locked` sync still runs and every subsequent `uv run` reuses that exact, lock-verified env — no drift, no torch-variant thrash. Still **never** set `UV_TORCH_BACKEND` in CI (it's a sync no-op).
- The 40% coverage floor and `bandit` HIGH gate are unaffected; `pip-audit` resolves fine with the explicit pytorch indexes present.

---

## Files owned

| File | Change |
|---|---|
| `pyproject.toml` | Remove `torch-backend = "auto"` block + the bare `torch>=2.12` dep + the misleading "one `uv sync` works everywhere" comment; add `cpu`/`cu130` extras, `[tool.uv] conflicts`, two `[[tool.uv.index]]` (explicit), `[tool.uv.sources] torch`. |
| `uv.lock` | Regenerate via `uv lock`; now carries both `+cpu` and `+cu130` torch entries. Committed. |
| `.github/workflows/ci.yml` | Add `--extra cpu` to all `uv sync`/`uv run`; add `--locked` to the sync step. |
| `README.md` | Fix the Setup block (lines ~41-51) and the GPU/CPU hardware note (lines ~53-69): the install command is now `uv sync --extra cu130` (GPU) **or** `uv sync --extra cpu` (CPU/CI), not a bare `uv sync`. Add the CPU-box-segfault warning + the "never --extra cu130 on a non-GPU box" rule. Correct the implication that auto-detection picks the GPU wheel. |
| `.env.example` | No torch env vars belong here (backend is selected by the sync flag, not env). Add a short comment block pointing at the README torch-extra instructions so users don't look for a `TORCH_*` knob. |
| `docs/decisions.md` | Append a new dated decision **superseding** §"Cross-machine toolchain — torch backend auto-detect" (2026-06-10): record that `torch-backend = "auto"` is uv-pip-only and was a no-op for the lock/sync/run workflow (lock pinned `+cpu`, GPU box silently ran CPU torch), and that the conflicting-extras pattern replaces it. Keep the old entry (append-only ADR home); cite uv docs + issues #12994/#18157/#10360. |
| `docs/DEVLOG.md` | One entry: what changed (extras + conflicts + indexes), why (auto is pip-only, markers can't split two win32 boxes), rejected (no-sync stopgap, env markers, splitting torch out), opens (default-extra #10360, macOS guard). |
| `.claude/KNOWN_ISSUES.md` | Replace/append the torch entry: document the verified `torch-backend`-ignored-by-sync behaviour and the `UV_NO_SYNC` revert as the **known stopgap** (flagged drift-prone), and record the new extras workflow + the hard "CPU box always `--extra cpu`" rule. |
| `justfile` *(new, optional)* | `sync-gpu` (`uv sync --extra cu130`) and `sync-cpu` (`uv sync --extra cpu`) recipes to remove the remember-the-flag footgun. |

`apps/` is untouched (thin-shell rule). No `src/` code changes — torch is consumed transitively by `sentence-transformers`; this is a packaging/toolchain change only.

---

## Definition of done

- `uv.lock` contains **both** `torch …+cpu` and `torch …+cu130` entries from the two explicit indexes; committed.
- **GPU box:** `uv sync --extra cu130` → `torch.cuda.is_available() == True`, version `2.12.0+cu130`; embedder + reranker run on GPU.
- **CPU box:** `uv sync --extra cpu` → version `2.12.0+cpu`, `cuda.is_available() == False`; full `uv run pytest` green; **no exit-139 segfault**.
- **Conflict guard:** `uv sync --extra cpu --extra cu130` **fails** with a uv conflict error (proves the invariant can't be violated by enabling both).
- **CI:** `uv sync --extra cpu --locked` succeeds on the CPU runner; full suite green; coverage ≥ 40%; `bandit` HIGH clean; `pip-audit` resolves.
- All docs in **Files owned** updated in the same PR (docs and code land together, per project rule). The misleading pyproject comment is gone.
- Diff staged and summarized for explicit user review before any commit/push (no auto-commit; lock changes especially).

---

## Invariant (non-negotiable)

> **The CPU-only box and CI MUST resolve the `+cpu` torch wheel and MUST NEVER materialize `+cu130`.** `+cu130` segfaults (exit 139) on a machine with no usable GPU. This is enforced by: (1) `cpu`/`cu130` declared mutually exclusive in `[tool.uv] conflicts`, so the two variants can never co-install; (2) CI pinned to `uv sync --extra cpu --locked`; (3) the standing discipline that **only the GPU box ever passes `--extra cu130`** — every other machine and CI pass `--extra cpu`. The lock containing a resolvable cu130 entry does **not** violate this: the invariant is about what is *selected*, not what is *resolvable*. Any future change that could let a non-GPU machine select cu130 (e.g. making cu130 a default extra/group) is forbidden until uv ships safe per-machine defaults (#10360) and is re-reviewed against this invariant.