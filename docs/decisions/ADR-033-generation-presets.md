<!-- status: active · updated: 2026-07-27 · class: append-only -->

# ADR-033 — Generation presets: frozen citation contract, swappable brief

- **Status:** proposed (stub — needs `grill-me` before the Decision section is filled)
- **Date:** 2026-07-27
- **Deciders:** user (product), Claude (Cowork planning session 2026-07-27)
- **Plan:** `docs/PLAN_2026-07-27_maps-trust-reports.md` Track 3

## Context

The user wants named, reusable generation presets — a system prompt plus a retrieval scope —
producing structured written outputs (briefing, study guide, FAQ, comparison), in the spirit of
NotebookLM's Reports but over Provenote's corpus.

**The hazard is already measured and recorded** (`ui-checklist.md` §3, *"User-customizable chat
modes"*, 2026-07-17): `ANSWER_PROMPT` is not a prompt, it is **the wire format the integrity layer
parses**. `synthesis.py` comments that `_CITATION_RE = \[(\d+)\]` matches markers *"produced by
ANSWER_PROMPT"*, and `prompts.py`'s bullet list of prohibitions (never `[Source 3]`, never
`[2, 4]`, never words in brackets) mirrors that parser's grammar. There is a logged 2026-07-14
incident where **model drift alone** — haiku emitting `[Source 2]` — made cited claims read as
uncited. The failure chain is: `cited_source_numbers` → `[]` → `claim_marker` → every claim
`MARKER_UNSUPPORTED` → `record_claims` persists false rows → the gap layer's `unsourced_claim`
detector and the `MIN_FAILURE_TAG_COUNT` self-improvement gates learn from noise. A user-editable
prompt is that incident with the safety off.

Current structure: `prompts.py` holds three module-level `ChatPromptTemplate` constants and **no
functions**; they are consumed in `pipeline.py`. There is no prompt registry and no mode dispatch —
`SYNTHESIS_MODE=ai|human` is a branch in `chat_controller` selecting between two result builders,
not a prompt swap. `app_settings.py` is a flat JSON blob at `data/settings.json`, suited to five
scalars. `wiki.py` is the closest existing generated artifact: deterministic clustering, one
confined LLM call for title/summary/tags only, `apply=False` means zero LLM calls, graceful
fallback on any failure. ADR-025 S1 deliberately keeps retrieval **scope** out of `RagOverrides`,
because scope is a content filter and `RagOverrides` is governed as quality knobs.

## Options

1. **Fully user-editable system prompt.** — *Pros:* maximum flexibility; matches NotebookLM's
   "create your own" ergonomics. *Cons:* the integrity layer's parser contract becomes
   user-editable; the failure is silent and it corrupts persisted `answer_claims`, the gap layer,
   and the self-improvement gates. Non-starter as stated.
2. **Frozen contract segment + swappable brief segment (proposed).** The citation grammar, the
   only-from-sources rule, and the no-invention rules live in a non-editable segment composed
   **last** so no earlier instruction can override it; the preset supplies audience, purpose,
   section structure, tone, length — nothing about citations. Backed by a **citation-audit
   regression gate**: `audit_citations()` already exists and detects malformed and out-of-range
   markers; a generated section that degrades past the locked baseline fails loudly instead of
   persisting bad claims. — *Pros:* real customisation with the load-bearing contract out of reach;
   the safety belt is existing code. *Cons:* users will eventually want to change something the
   frozen segment owns; `prompts.py` grows its first function, a structural change to a
   constants-only module.
3. **Fixed built-in report types only, no custom presets.** — *Pros:* every prompt stays
   hand-verified. *Cons:* refuses the user's actual request; the built-in set is guesswork about
   what corpora people hold.

## Decision

*(open — fill after `grill-me`)*

Candidate: **option 2**, with these as part of the decision:

- **Storage: a `report_presets` table**, not `settings.json` — a growing library of user-authored
  objects is not a settings blob. Precedent for user-authored sidecar tables: `DocumentMeta`
  (ADR-013), `GapTriage` (E5).
- **Built-ins ship read-only and are duplicated-to-edit, never edited in place** — otherwise a
  release either overwrites the user's edits or can never update its own defaults. Starting set:
  Briefing · Study guide · FAQ · Method comparison · Literature snapshot.
- **Scope stays a separate channel.** A preset carries a scope *reference*, resolved through the
  existing `scope_folder_id` path. It does not enter `RagOverrides` (ADR-025 S1 is not reopened by
  a preset feature).
- **A report is a job, not a turn.** N sections = N retrievals + N generations, so: a **$0 dry
  run** that resolves scope and shows the sources each section would draw on without calling a
  model (the `wiki.py --apply` pattern plus `llm.assert_provider_intent`), a cost preview before
  running, progress and cancellation over the existing SSE boundary, and per-section provenance.
- **Trust-annotated output** (depends on ADR-031): each section states its evidence footing, and
  the report carries an evidence appendix. This is the part NotebookLM structurally cannot ship,
  and the natural on-ramp to Phase 9 / PRISMA-trAIce.

## Consequences

*(open — fill with the Decision)*

Provisional:

- **Easy:** rendering extends `export.py`'s existing markdown substrate rather than forking it;
  the citation-audit gate is a call to shipped code.
- **Hard / committed:** `prompts.py` becomes a composer, and the frozen/brief boundary must be
  enforced structurally (composition order + tests), not by documentation. Every new preset field
  is a new opportunity to leak an instruction into contract territory — the brief schema should be
  **fields, not free text**, wherever a field will do. Report cost is user-visible and must be
  previewed before it is spent.
- **Boundary:** presets shape *style and structure*. They never touch citation grammar, retrieval
  quality knobs (ADR-010), or the evidence/interpretation split. A preset that wants to change any
  of those reopens the relevant ADR.
