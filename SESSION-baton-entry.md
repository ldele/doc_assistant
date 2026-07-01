## Baton — 2026-06-29 (active tool: Cowork) — concept-graph rebuild + KI-10 fix procedures folded into docs

Planning/docs session (no code, no RTX box, nothing committed). Two design-locked items were turned into
in-repo build procedures so Claude Code can execute them without re-deriving the plan.

**Done this session (Cowork, docs only — staged where the file tool could write):**
- **`docs/specs/concept-graph-redesign.md`** — added a "Build sequence (PR order)" subsection to the Build
  node section: dependency-chain diagram, the two Feature-6 prerequisites, the PR-A → RG-001 gate → PR-B →
  PR-C order, and 5 risk/watch-points. Key framing now in the spec: **RG-001/008/009 are threshold-setting
  gates, not build blockers** — Node A is buildable + fully testable on fakes now; only the real `--apply`
  run + threshold-locking need the RTX/Ollama box.
- **`docs/desktop-packaging.md` §5** — added a "KI-10 — the truststore fix" subsection: Step 0 (read the
  already-staged stderr-WARN) → branch A (PyInstaller runtime hook / bundling) → branch B (explicit OS-trust
  `http_client` in `llm.py` — the robust fix) → Step C (verify + record). Header date bumped 06-22 → 06-29.
- **NOT done (Cowork is write-protected on `.claude/`):** the KI-10 entry in `.claude/KNOWN_ISSUES.md` still
  carries only the root-cause lead — it does **not** yet point at the new `docs/desktop-packaging.md` §5
  procedure. One-line pointer edit left for Code.

**Pick up: code (Windows). Two independent tasks, neither blocking, both dev-reproducible without the RTX box:**

1. **KI-10 truststore fix (work/proxy box — cheapest open blocker; ≈ a re-freeze + ≤2 paid calls).**
   Full procedure: `docs/desktop-packaging.md` §5 "KI-10 — the truststore fix".
   - **Step 0 (no code):** `just sidecar` (re-freeze; the stderr-WARN is already staged in
     `apps/api/__main__.py:37-48`) → run ONE on-proxy Anthropic turn → read stderr. WARN present ⇒ branch A;
     no WARN but still `CERTIFICATE_VERIFY_FAILED` ⇒ branch B. **Don't write the fix before this read.**
   - **Recommended regardless = branch B:** `src/doc_assistant/llm.py` `AnthropicClient.__init__` (~96-100,
     currently `Anthropic(api_key=...)` with no custom `http_client`) → hand it
     `httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))`, guarded with a clean fallback when
     truststore is absent. Keeps TLS inside `AnthropicClient` (encapsulation); reviewer/judge inherit it free.
   - **Step C:** verify on-proxy (first token, cents billed) → records the last open **RG-011** piece
     (frozen-build paid first-token on this box) → flip KI-10 → RESOLVED. Construction-only regression test
     (no live paid call, cpc §13).
   - **Also do the deferred one-liner:** add a pointer in `.claude/KNOWN_ISSUES.md` KI-10 →
     `docs/desktop-packaging.md` §5 (Cowork couldn't write `.claude/`).

2. **Concept-graph rebuild — PR-A (Node A, the deterministic skeleton).** Full sequence:
   `docs/specs/concept-graph-redesign.md` → "Build sequence (PR order)".
   - **Prereqs first (both bit Feature 6):** `scripts/compute_doc_vectors.py --apply` (populate
     `DocSimilarity`, else similarity-provenance is silently absent) + seed a starter `Concept`/`ConceptAlias`
     vocabulary (`scripts/seed_concepts.py` + `--promote`; **no vocabulary → empty graph**).
   - PR-A is buildable + testable on fakes **now** (guard tests use toy inputs, no DB/LLM) → can land staged
     behind the dry-run + provisional defaults. **The RG-001/008/009 validation run needs the RTX/Ollama box**
     (free, on the host, KI-5) — it sets `min_cooccurrence` + presence mode from the run, not guessed, and
     gates marking the graph *usable* + the whole gap layer (ADR-004). Do NOT let "no RTX today" stall PR-A.
   - **KI-7 stays parked:** don't touch `concept_graph.py`/`epistemics.py`/`wiki.py` (connected change).

**Smaller open items (complete tail — none blocking; do alongside the above or standalone):**
- **`bibtex` cross-boundary import smell** (DEVLOG 2026-06-27, the one un-homed item): `bibtex` imports the
  **private** `_first_author_surname` across the `ingest` package boundary. Pre-existing coupling — promote it
  to a public function or re-export from `ingest/__init__`. Code-only, no box, ~10 min.
- **Two `.claude/` one-liners Cowork couldn't write (write-protected):** (a) `KNOWN_ISSUES.md` KI-10 → add a
  pointer to the new `docs/desktop-packaging.md` §5 fix procedure; (b) `CONTEXT.md` open-questions → add the
  matching gap-layer bullet (flagged write-protected in DEVLOG 2026-06-26). Both trivial Code edits on Windows.
- **`chat_controller.py:323` TODO(PR-M1)** = **KI-8** (coarse PC→baseline marker mapping) in code-comment form.
  Already tracked, advisory + fail-safe; the precise re-projection is the documented upgrade *if* containment
  proves too coarse on real data. **Leave it** unless real data shows over-attribution — not a standalone task.

**Open by design — do NOT action (tracked + deferred for a reason; listed so they're not re-surfaced as new):**
KI-2 (3.12 pin — reopen only when native deps ship cp314 wheels) · KI-4 (credit leak — standing rule:
`--provider ollama` on enrichment) · KI-5 (sandbox can't write — environmental) · KI-6 (uv-Python SSL —
per-machine, deliberately not pinned in-repo) · RG-013 (structlog in freeze — rides the next M4 re-freeze) ·
RG-002..007 (the *old* concept graph's tuning — moot under the redesign; close as "redesign re-founds this,"
do NOT run against the superseded graph) · RG-012 Tier-2 (parked release gate, not a dev/merge gate).

**Merge note (asked this session):** branch `docs/desktop-shell-specs` is **27 ahead / 0 behind main** —
clean fast-forward whenever the user wants it; nothing waits on the RTX box (the box is only for the *future*
concept-graph build, not started). RG-012 Tier-2 is a release gate, not a merge gate. User chose "not yet."

**Cowork-side caveat:** the doc edits above won't show in `git diff` from the Cowork sandbox (mount quirk,
KI-5) but are in the Windows checkout — review there before committing. **Stage + summarize + ask before any
commit (cpc §13).**

---
