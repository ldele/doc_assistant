<!-- status: active · updated: 2026-07-19 · class: living -->

# GLOSSARY — doc_assistant

The pinned vocabulary for this project: one canonical term per concept, the synonyms that are
forbidden, and where each concept is owned. Use the canonical form in code, docs, prompts, and UI.
Enforced (opt-in, on-call) by the `cpc-glossary` gate and `cpc-dod-lint` rule N001 — cpc
CONVENTIONS §14. Laid + filled 2026-07-19 (ADR-021): the concept/keyword/family/skeleton
vocabulary has produced real confusion (see the 2026-07-17/18 "junk labels" trap) — pin it.

**Owner:** Lucas

---

## C-001 — Concept (curated vocabulary row)

**Canonical:** `Concept`
**Definition:** A curated vocabulary entry (row in `concepts`) that can appear in document text via
presence matching; the unit both the concept skeleton and keyword families are built from.
**Forbidden:** `topic`, `entity`
**Authoritative in:** `src/doc_assistant/db/models.py::Concept`

## C-002 — Keyword (mined candidate)

**Canonical:** `Keyword`
**Definition:** A per-document mined candidate term (`source="extracted"`); a candidate **only** —
never auto-promoted; the user promotes one into a `Concept` (`promote_keyword`). The 2026-07-05
`--promote-all` incident (ADR-018) is why this boundary is load-bearing.
**Forbidden:** —
**Authoritative in:** `src/doc_assistant/keywords.py` + `db/models.py::Keyword`

## C-003 — keyword family

**Canonical:** `keyword family`
**Definition:** A user-facing grouping of `Concept` rows for library filtering (ADR-015); shares
the `Concept` table with the graph and is **deliberately unfiltered** by `graph_include`.
("tag family" is the historical spec name — do not reintroduce it in new code or UI.)
**Forbidden:** `tag family`, `tag`
**Authoritative in:** `src/doc_assistant/keyword_families.py` + `library.list_keyword_families`

## C-004 — concept skeleton (the graph artifact)

**Canonical:** `concept skeleton`
**Definition:** The deterministic Node-A graph artifact (nodes = `graph_include` Concepts; edges =
co-occurrence with citation/similarity provenance), persisted as `concept_edges` + `skeleton.json`;
rebuilt, never hand-edited. "Concept graph" names the *feature/UI view* over it, not the artifact.
**Forbidden:** `knowledge graph`
**Authoritative in:** `src/doc_assistant/concept_skeleton.py`

## C-005 — Node A / Node B

**Canonical:** `Node A` (deterministic skeleton build) · `Node B` (confined LLM enrichment)
**Definition:** Node A builds the skeleton with zero LLM calls; Node B annotates **existing** edges
with relation/stance and never creates a node or edge. Host rebuild is `--apply --enrich` together
(`--apply` alone wipes Node-B annotations).
**Forbidden:** —
**Authoritative in:** `concept_skeleton.py` / `concept_skeleton_enrich.py`

## C-006 — gap

**Canonical:** `gap` (a `GapRow`)
**Definition:** A finding over the skeleton. Deterministic kinds are delete-and-replaced on every
rebuild; stochastic kinds (`suggested_*`) are status-preserving upserts — two different write
disciplines (the KI-17 orphan class lives in that difference).
**Forbidden:** —
**Authoritative in:** `src/doc_assistant/gaps.py`

## C-007 — epistemics markers

**Canonical:** `epistemics markers` (`contested` / `superseded_trend` / `stable` / `unique`)
**Definition:** Advisory chunk-level chips derived from skeleton node weights; inform-don't-block —
they never change synthesis, ranking, or the answer (byte-identical when absent).
**Forbidden:** `confidence score`
**Authoritative in:** `src/doc_assistant/epistemics.py`

## C-008 — graph_include

**Canonical:** `graph_include`
**Definition:** The additive **opt-in** flag (ADR-018) scoping which Concept rows the graph loads;
families ignore it by design. The curation verb is **demote** (`set_graph_include(cid, False)`),
never delete — unfamiliar short labels are usually real specialist vocabulary.
**Forbidden:** —
**Authoritative in:** `db/models.py::Concept.graph_include` + `concept_skeleton.load_concepts`

## C-009 — Enrichment-Layer Pattern

**Canonical:** `Enrichment-Layer Pattern`
**Definition:** Derived data ships as a separate module + idempotent CLI runner, sidecar by
default, and never mutates the primary chunk store.
**Forbidden:** —
**Authoritative in:** `.claude/CONTEXT.md` non-negotiable #4

## C-010 — locked setting

**Canonical:** `locked setting`
**Definition:** A `config.py` value changeable only via an eval-harness experiment (`--repeat`,
beat the control beyond variance, record a baseline in `tests/eval/baselines/`).
**Forbidden:** —
**Authoritative in:** `.claude/CONTEXT.md` → Locked settings table

## D-001 — Provenote vs doc_assistant

**Canonical:** `Provenote` (product) · `doc_assistant` (code)
**Definition:** Provenote is the product/installer identity (wordmark, window title, bundle id);
`doc_assistant` is the code identity (Python package, commands, npm name, sidecar binary). The
split is intentional (ADR-012) — never "finish" the rename into code.
**Forbidden:** `provenote` (as a code identifier)
**Authoritative in:** `docs/decisions/ADR-012-provenote-installer-identity.md`
