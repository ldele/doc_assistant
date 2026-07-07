# Spec — Gap detection: two-tier deterministic/stochastic layer

**Status:** ✅ **Tier 1 + the Tier-2a deterministic floor BUILT (2026-07-07, SPRINT-002, ROADMAP row
G2)** — `src/doc_assistant/gaps.py` + `GapRow` + `scripts/build_gaps.py`. RG-001's edge-precision
validation ran and passed first (R5, ADR-008); `min_degree=3` is set from this corpus's degree
distribution, not guessed (`tests/eval/baselines/gap_min_degree_2026-07.md`). **Still not built:**
the Tier-2a stochastic ceiling (`gap_suggest.py`) and Tier 2b (external reach) — both deferred, out
of the G2 sprint's scope; see `docs/sprints/SPRINT-002-gap-layer-deterministic.md`.
**Owner of execution:** Claude Code (code + tests), when Phase 7 is active.
**Pattern reference:** Enrichment-Layer Pattern (`docs/decisions.md`); Research Integrity Layer
(Chunk 2a markers, Chunk 2b reviewer); Decision C skeleton (`docs/decisions.md`); 7d epistemics
(`docs/specs/feature-7d-knowledge-currency.md`).

**Goal (the why).** Phase 7's headline capability is surfacing what the user — and the LLM — cannot
see: concepts the corpus under-supports, claims it cannot source, and exploration directions the user
did not think to look. A *curated* concept graph (Decision C) gives control and cuts cost but can only
find gaps **inside the vocabulary the user chose** — precise on the known, blind on the unknown. This
layer resolves that by splitting gaps on the project's existing **deterministic / stochastic** line:
deterministic gaps are trustworthy facts about the graph; stochastic gaps are *rated suggestions* that
can reach past the curated set but can never write it. The wall is what makes reaching for the unknown
safe — a wrong suggestion is only ever a candidate the user (or a deterministic check) must accept.

---

## ADR recap — what this layer is (full rationale: ADR-004)

**Context.** Two pre-existing gap signals already exist but were never collected into a layer: the
wiki's `citation_thin` / `single_source` flags (Feature 6b) and the concept graph's isolated-node /
thin-bridge signals (7c). Separately, the integrity layer already marks each answer-claim `unsupported`
when it cites no real source, and the reviewer already emits an answer-level `failure_tag` — both are,
in effect, corpus-gap signals sitting unused.

**Decision.** One typed gap object, three sub-tiers, the determinism label first-class:

- **Tier 1 — deterministic, within the curated vocabulary.** Isolated / single-source / thin-bridge /
  under-connected nodes over the deterministic skeleton. Subsumes 6b + 7c.
- **Tier 2a — within the corpus, gated in-app.** Deterministic floor (aggregate the per-claim
  `unsupported` markers + citation-layer gaps) + a stochastic ceiling (a quarantined LLM pass over
  under-connected nodes and reviewer `failure_tag`s that *suggests* missing links/concepts).
- **Tier 2b — true anti-blind-spot pass.** Deferred; needs an external reach (concept bank /
  citation-chasing / web). **Not** the idea-generator (it closes inward — ADR-004 option 3).

**Cross-cutting rules.** (1) The determinism label is first-class — read it, do not re-derive it;
stochastic gaps carry their LLM inputs (observability). (2) Stochastic gaps feed the curated vocabulary
— every suggestion is a candidate promotion (the compounding arrow). (3) Three gap *types* stay
distinct: `unsupported` (no source) ≠ `contested` (sources disagree) ≠ `superseded_trend` (field moved
on).

---

## Decisions (locked 2026-06-26, ADR-004)

| # | Decision |
|---|---|
| 1 | **Gaps split on the deterministic/stochastic line.** Deterministic = facts about the graph (trustworthy floor). Stochastic = rated suggestions, **never written to the skeleton as fact**. The label is a first-class field on the gap object. |
| 2 | **One gap object, three sub-tiers** (1 / 2a / 2b). Tier 1 subsumes the wiki-6b and concept-7c signals; do not leave them as separate ad-hoc outputs once Tier 1 ships. |
| 3 | **The deterministic 2a floor is a query, not new ML.** It reads already-persisted `answer_claims.marker == "unsupported"` and the `Citation` graph. No new model on this path. |
| 4 | **The stochastic ceiling is quarantined.** The LLM only *suggests* (missing link / missing concept / thin-area verdict); suggestions are rated and surfaced for promotion; they never auto-write the graph. Consistent with "don't auto-remediate reviewer issues" (ROADMAP). |
| 5 | **The compounding arrow.** Accepted suggestions promote into the curated `Concept`/`ConceptAlias` set → next deterministic rebuild is richer. The gate is the user's judgment. |
| 6 | **Three gap types stay distinct.** `unsupported` / `contested` / `superseded_trend` are never flattened into one "gap" — each drives a different user action. |
| 7 | **Tier 2b is deferred and external.** Its "outside the corpus" source is an open choice (concept bank vs. citation-chasing vs. web). The idea-generator is explicitly rejected for it. |
| 8 | **Shared observability spine.** Tier-2a's stochastic pass and Phase 9 review-generation consume the same "expose LLM inputs + rate output" substrate. Build it once; the gap object's `rating` field is its first output. |

---

## Contracts (build-time; Tier 1 + the Tier-2a floor are the first increment)

### `src/doc_assistant/gaps.py` (new) — the gap object + deterministic detectors

```
GapTier      = Literal["t1", "t2a", "t2b"]
Determinism  = Literal["deterministic", "stochastic"]
GapKind      = Literal[
    # Tier 1 (deterministic, over the skeleton)
    "isolated", "single_source", "thin_bridge", "under_connected",
    # Tier 2a floor (deterministic, over persisted answer/citation data)
    "unsourced_claim", "citation_missing",
    # Tier 2a ceiling + Tier 2b (stochastic, suggestions)
    "suggested_link", "suggested_concept", "thin_area",
]

@dataclass(frozen=True)
class Gap:
    concept_id: str            # curated Concept id; for a suggestion, the candidate label
    tier: GapTier
    determinism: Determinism   # first-class — a consumer reads this, never re-derives it
    kind: GapKind
    evidence: GapEvidence      # deterministic: graph-fact ids; stochastic: the LLM inputs (observability)
    rating: float | None       # provenance/evidence strength (shared spine); None for raw deterministic facts
    status: Literal["surfaced", "promoted", "dismissed"]  # curation lifecycle → the compounding arrow
```

`GapEvidence` carries, for deterministic gaps, the supporting fact ids
(`citation` / `doc_similarity` / `cooccurrence` row keys, or the `answer_claims` ids an
`unsourced_claim` aggregates); for stochastic gaps, the exact inputs the LLM was handed (the present
concepts, the doc context, the prompt) so the suggestion is auditable.

**Tier 1 — deterministic detectors (pure; over the Decision-C skeleton graph):**
- `detect_isolated(graph) -> list[Gap]` — degree-0 curated concepts.
- `detect_single_source(graph) -> list[Gap]` — concepts asserted by exactly one document. **Coverage
  rule (carried from 7d Decision 4):** single-source = *flagged for attention*, never a defect — a sole
  authoritative source is among the library's most valuable. Distinguished from contested by absence of
  contradicting edges.
- `detect_thin_bridges(graph) -> list[Gap]` — `networkx.bridges` over each connected component (the 7c
  mechanism, re-homed here).
- `detect_under_connected(graph, *, min_degree: int) -> list[Gap]` — curated concepts with degree below
  `min_degree`. **`min_degree` is provisional — set on the validation run** (RG-001); it is the routing
  signal into the Tier-2a ceiling.

All four are pure and deterministic given the graph; each returns `Gap`s with
`determinism="deterministic"`, `tier="t1"`, `rating=None`.

### `src/doc_assistant/gaps.py` — Tier-2a deterministic floor (pure over persisted data)

- `detect_unsourced_claims(claims: Iterable[AnswerClaimRow]) -> list[Gap]` — aggregate
  `answer_claims.marker == "unsupported"` across answers, grouped to the curated concept(s) the claim
  text matches (presence match, Decision C). Output `kind="unsourced_claim"`, `tier="t2a"`,
  `determinism="deterministic"`, `evidence` = the contributing `answer_claims` ids. **Consumes existing
  persisted data** (`synthesis.claim_marker()` already writes the marker — `docs/specs/chunk-2a-*`).
- `detect_citation_gaps(citation_graph, ingested_doc_ids) -> list[Gap]` — references the corpus cites
  (the `Citation` graph) but does not contain. `kind="citation_missing"`.

### `src/doc_assistant/gap_suggest.py` (new) — Tier-2a stochastic ceiling (quarantined)

- `suggest_for_thin(gaps: list[Gap], graph, client: LLMClient) -> list[Gap]` — for under-connected
  nodes (+ optionally answer-level reviewer `failure_tag`s), one quarantined LLM call per concept,
  handed only the concept and its present neighbours, returning a missing-link / missing-concept /
  `thin_area` **suggestion**. Output carries `determinism="stochastic"`, a `rating`, `status="surfaced"`,
  and the LLM inputs in `evidence`. **Provider isolation applies** (`--provider ollama`; never inherit
  the all-Anthropic default — `docs/specs/llm-provider-isolation.md`, credit-leak hazard). The LLM never
  writes the graph; it only emits `Gap`s for promotion.

### Persistence — `src/doc_assistant/db/models.py` + `db/migrations.py`

- New `GapRow` (`gaps`): `id · concept_id (indexed) · tier · determinism · kind · evidence_json ·
  rating · status · graph_version · computed_at`. **Sidecar; regenerable**; the deterministic rows are
  dropped + rebuilt with the graph (Enrichment-Layer Pattern). Stochastic rows persist their
  `status` (`surfaced`/`promoted`/`dismissed`) so the compounding arrow survives a rebuild.

### CLI runner — `scripts/build_gaps.py` (new)

- Idempotent, sidecar-only, re-runnable after every skeleton rebuild. Deterministic tiers run free;
  the stochastic ceiling is opt-in (`--suggest`, provider-gated). Enrichment-Layer Pattern.

### Surfacing (read-only; no retrieval change)

- The deterministic tiers surface as an inspectable gap list (a curated-concept view, or the wiki
  layer). The stochastic suggestions surface as **promotable candidates** into the curated vocabulary
  (the compounding arrow) — never auto-applied. No change to the retrieval path; gaps are an additive
  read over the graph + persisted answer/citation data.

---

## Build node

**Depends on:** the **Decision-C curated-vocabulary skeleton** (the `Concept`/`ConceptAlias` schema +
deterministic edges) — **not yet built; build spec `docs/specs/concept-graph-redesign.md`** (design-locked
2026-06-27, its deterministic Node A is the dependency this layer needs); the `Citation` graph (shipped,
`db/models.py`); the per-claim
`unsupported` marker (shipped, `synthesis.claim_marker()` → `answer_claims.marker`); the reviewer
`failure_tag` enum (shipped, `reviewer.ReviewResult`). The shared observability/rating spine is
introduced here (its first consumer) and reused by Phase 9 review-generation.
**Files owned:** `src/doc_assistant/gaps.py` (new), `src/doc_assistant/gap_suggest.py` (new),
`scripts/build_gaps.py` (new), `src/doc_assistant/db/models.py` + `db/migrations.py` (`gaps` table),
the curated-concept/wiki surfacing seam, tests as below.
**Status:** blocked (design-locked) — on the Decision-C skeleton **and** the RG-001 edge-precision run.

### Guard tests (written with the build)
- `tests/unit/test_gaps.py` — fixed toy skeleton: degree-0 concept → `isolated`; sole-source concept →
  `single_source` with **flagged-not-penalized** coverage (the regression that matters — mirrors the 7d
  unique-source rule); cut edge → `thin_bridge`; degree-below-`min_degree` → `under_connected`. Pure, no
  DB/LLM.
- `tests/unit/test_gaps_floor.py` — fixed `answer_claims` rows: claims with `marker=="unsupported"`
  aggregate to the right `unsourced_claim` gaps with the contributing ids in `evidence`; cited claims
  produce none. `Citation`-vs-ingested fixture → `citation_missing`. Pure.
- `tests/unit/test_gap_suggest.py` — `ScriptedBackend` (no live LLM, no cost): a stochastic suggestion
  carries `determinism=="stochastic"`, a `rating`, and the handed inputs in `evidence`; the detector
  **never** mutates the input graph (quarantine guard).
- `tests/integration/test_build_gaps.py` — mocked skeleton + persisted rows → `scripts/build_gaps.py`
  writes deterministic `gaps` rows; idempotent re-run is a no-op; `--suggest` off makes zero LLM calls;
  promoted stochastic rows survive a deterministic rebuild.

### Definition of done
- Skeleton rebuild → `build_gaps` produces deterministic Tier-1 + Tier-2a-floor rows; idempotent re-run
  is a no-op; the deterministic path makes **zero** LLM calls.
- Single-source / sole-authority concepts are **flagged, never penalized**; the three gap types
  (`unsourced_claim` / `contested` / `superseded_trend`) remain distinct in output.
- Every `Gap` carries a correct first-class `determinism` label; stochastic gaps carry their LLM inputs
  in `evidence`; deterministic gaps carry graph-fact ids.
- Stochastic suggestions are surfaced as promotable candidates and **never** auto-write the graph;
  promotion updates the curated vocabulary (the compounding arrow) and `status`.
- No change to retrieval output (public eval byte-identical with the gap layer present); the stochastic
  path honours provider isolation (no paid calls in tests — cpc §13); ruff / mypy --strict / bandit
  clean.
- **Gate:** not marked done until the RG-001 edge-precision run confirms Tier-1 signals are meaningful
  on the real corpus and `min_degree` / presence-recall thresholds are set from it (not guessed).

## Out of scope
- **Tier 2b** (the external anti-blind-spot reach) — deferred; its outside-source choice is open. The
  idea-generator is rejected for it (ADR-004 option 3).
- Retrieval-rank integration of gaps (the gap layer is read-only; any rank use is a separate eval-gated
  experiment, same rule as 7d).
- Truth adjudication of any kind — the layer surfaces gaps + suggestions; the user disposes.
- The full shape of the shared observability/rating contract beyond the gap object's `rating` field —
  settled with Phase 9 review-generation, its co-consumer.
- Auto-promotion of suggestions — promotion is always a user act.
