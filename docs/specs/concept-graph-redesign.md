# Spec — Concept-graph redesign: curated vocabulary + deterministic skeleton + confined LLM enrichment

**Status:** ✅ **NODE A BUILT** (2026-06-30, Claude Code) — the deterministic, zero-LLM skeleton
(`src/doc_assistant/concept_skeleton.py` + `scripts/{seed_concepts,build_concept_skeleton}.py` + the four
new tables + `CONCEPT_SKELETON_*` config; 23 guard tests, full gate green). **Node B (LLM relation/stance)
and the RG-001/008/009 threshold-setting `--apply` run on the real corpus remain.** Was 🔒 DESIGNED, NOT
BUILT (2026-06-27, Cowork). Turns "Feature 7 — concept-graph
REDESIGN" ("Decision C", `docs/decisions.md`, 2026-06-18) into a code-level build contract.
**Supersedes** the open-vocabulary LLM-extraction core of the shipped PR-16 graph (`.claude/KNOWN_ISSUES.md`
KI-7) — do **not** build on `concept_graph.py` or `data/graph/graph.json`. The two carried-over PR-16
decisions (ADR-1 Louvain, ADR-4 composite chunk key) are preserved here unchanged.
**Owner of execution:** Claude Code (code + tests), when Phase 7 is active.
**Pattern reference:** Enrichment-Layer Pattern (`docs/decisions.md`); LLM provider isolation
(`docs/specs/llm-provider-isolation.md`); Decision C (`docs/decisions.md` → "Feature 7 — concept-graph
REDESIGN"); 7d epistemics (`docs/specs/feature-7d-knowledge-currency.md`); the gap layer this unblocks
(`docs/decisions/ADR-004-gap-detection-layer.md` + `docs/specs/feature-gap-detection.md`).

**Goal (the why).** The shipped graph derives its nodes from a per-document open-vocabulary LLM
extraction (36–40 calls/doc, `budget_exhausted` over the corpus) and fragments concepts on a same-domain
library. The library already maintains two zero-LLM doc-level graphs — `Citation` (who-cites-whom) and
`DocSimilarity` (BGE cosine) — that the old design ignored. This redesign makes the **node vocabulary
user-curated**, computes **presence and the edge skeleton with no LLM** from those existing graphs plus
chunk-level co-occurrence, and **confines the LLM** to the one thing only it can do: annotate a
relation verb + stance over concepts a document already contains. The result is deterministic, free, and
byte-stable to rebuild; every edge is auditable to a citation / similarity / co-occurrence fact; and it
is the skeleton the gap-detection layer (ADR-004) is defined against.

---

## ADR recap — what this layer is (full rationale: `docs/decisions.md` → Decision C)

**Context.** Decision C chose **option C** (compose: deterministic skeleton + LLM enrichment over a
user-curated vocabulary) over keeping open-vocabulary extraction (option A — the cost + fragmentation
source) and over a full zero-LLM replacement that drops 7d (option B — loses the typed/polarized edges
that are the project's integrity differentiator).

**Decision (recap).** Nodes become a user-curated `Concept`/`ConceptAlias` set (this spec: seeded from
existing `Keyword` rows; the user promotes — candidate mining + the LLM dedupe/label pass are deferred,
see Out of scope). On top: **presence is deterministic** (string/alias match of curated terms against
chunks — the LLM never decides presence or defines the vocabulary); **edges = deterministic skeleton +
LLM annotation** (skeleton from chunk co-occurrence / `Citation` / `DocSimilarity`; the LLM, handed only
the concepts already present in a document, adds a relation verb + stance — by construction it only ever
annotates a co-occurrence edge, never extends the graph); **trust becomes a provenance set**
`{cooccurrence, citation, similarity, llm_relation}` + stance-agreement, replacing the single
`EXTRACTED|INFERRED|AMBIGUOUS` tag. Unconfirmed deterministic edges are kept and ranked by provenance,
never dropped — the skeleton is never gated by the LLM.

**Carried over from PR-16 unchanged (do not re-decide):**

- **ADR-1 — community detection is `networkx.louvain_communities(weight="weight", resolution=…,
  seed=42)`**, not Leiden/graspologic (`leiden_communities` is an unbacked dispatch stub; graspologic
  hard-pins `numpy<2`, which corrupts the locked torch/chroma stack). Keep a
  `detect_communities(algorithm="louvain")` seam for a future numpy-2 Leiden. *Reopens if a numpy-2
  Leiden backend lands.*
- **ADR-4 — back-pointers use the deterministic composite chunk key `{document_id}:p{parent_index}`**
  (Chroma generates throwaway UUIDs; there is no stable persisted chunk id). No chunk-store schema
  change. *Reopens if ingest later stamps a stable chunk id.*

---

## Decisions (locked 2026-06-27, this spec; under Decision C 2026-06-18)

| # | Decision |
|---|---|
| 1 | **The node vocabulary is curated `Concept`/`ConceptAlias` rows, never auto-generated.** The LLM never defines or extends the vocabulary. The first increment seeds candidates from existing `Keyword` rows; the user promotes a `Keyword` to a `Concept` (and attaches aliases). Mining + the LLM dedupe/label pass are deferred (Out of scope). |
| 2 | **Presence is deterministic: exact + alias string match.** A `Concept` is *present* in a chunk iff a case-folded surface form (the concept's canonical label or one of its `ConceptAlias` forms) occurs in the chunk text. Zero model, byte-stable. Recall is bounded by alias coverage — that is the curation burden, and the presence-recall RG gate measures it (see Gate). Embedding-fallback presence is explicitly deferred (Out of scope). |
| 3 | **The deterministic skeleton is the first build node and makes ZERO LLM calls.** Presence + edges + provenance + communities + gaps all compute from `Concept`/`ConceptAlias`, the chunk store (for presence + co-occurrence), `Citation`, and `DocSimilarity`. Free, byte-stable, auditable. This is the PR that unblocks gap-detection Tier 1. |
| 4 | **Co-occurrence is chunk-level, not document-level.** Two concepts co-occur when both are present in the **same chunk** (composite key `{document_id}:p{parent_index}`, ADR-4); an edge needs ≥ `min_cooccurrence` such chunks. Document-level co-occurrence on a same-domain corpus is near-complete (most concepts appear in most papers) and produces a meaningless dense graph — the same saturation Feature 6 hit at cosine 0.88–0.96. Chunk-level is the primary precision lever. `min_cooccurrence` is **provisional**, set on RG-001 (see Gate). |
| 5 | **Edges are typed by a provenance set, kept, and ranked — never dropped.** Each edge carries a `provenance: frozenset[str] ⊆ {cooccurrence, citation, similarity, llm_relation}` and an edge `weight` derived from it. A co-occurrence-only edge and a citation+similarity+co-occurrence+llm-confirmed edge are distinguishable; the gap layer and RG-001 threshold on provenance/weight, not raw adjacency. This is how density is controlled without gating the skeleton by the LLM (Decision C: "kept and ranked by provenance, never dropped"). |
| 6 | **The LLM relation/stance pass is a SEPARATE, deferred build node (Node B), confined and provider-isolated.** Handed only the concepts already present in a document, it adds a relation verb + a stance ∈ `POLARITIES` (`supports｜refines｜contradicts｜supersedes`) to an *existing* co-occurrence edge (`provenance ∪= {llm_relation}`). It never creates a node or an edge. It defaults to **local Ollama explicitly** (`CONCEPT_SKELETON_LLM_PROVIDER`, NOT `LLM_PROVIDER`) and routes through `assert_provider_intent` (credit-leak hazard, KI-4 / `llm-provider-isolation.md`). Node B is **specified, not built** in the first increment. |
| 7 | **Trust replaces the single integrity tag with provenance + stance-agreement.** The old `EXTRACTED｜INFERRED｜AMBIGUOUS` scalar is gone. Cross-source **stance disagreement** on an edge (≥2 documents assert it with opposing polarities) is the 7d `contested` signal; the `POLARITIES` vocabulary + `SUPPORTING/OPPOSING` partition carry over verbatim from `concept_graph.py`. Deterministic edges with no LLM stance are `unannotated`, not `inferred`. |
| 8 | **New module alongside the old; the old graph is retired only when 7d re-founds on the new skeleton.** Build the redesign as new modules (`concept_skeleton.py`, `scripts/build_concept_skeleton.py`); leave `concept_graph.py` + `scripts/build_concept_graph.py` in place. Retiring them is a connected change across `epistemics.py` → `compute_epistemics.py` / `chat_controller.py` / `wiki.py` + tests (KI-7: not safe as standalone cleanup) and is **out of scope** for this spec. |
| 9 | **Vocabulary scope carries an optional `folder_id`, present-but-unused.** Decision C ties concepts to projects-as-folders (`folder_id`). The schema includes a nullable `folder_id` so the projects feature lands without a migration (same convention as the `Figure.vlm_*` columns shipped null); the first increment builds **global** (folder-agnostic) presence. |
| 10 | **The graph is a regenerable sidecar artifact + sidecar tables — never the chunk store, never a graph DB.** A `skeleton.json` artifact under `CONCEPT_SKELETON_DIR` + the persistence tables below; dropped + rebuilt on each run (Enrichment-Layer Pattern). NetworkX in memory + a JSON file, per the roadmap's "don't make the concept graph a graph database" rule. |

---

## Contracts (build-time; the deterministic skeleton — Node A — is the first increment)

### `src/doc_assistant/concept_skeleton.py` (new) — pure core + impure boundary

Mirror the existing enrichment-module split (pure core behind a thin impure layer), as in
`concept_graph.py` / `wiki.py` / `figures.py`.

**Carried-over vocabulary (import or re-home from `concept_graph.py`, do not re-decide — Decision 7):**

```
POLARITIES: tuple[str, ...] = ("supports", "refines", "contradicts", "supersedes")
SUPPORTING_POLARITIES: frozenset[str] = frozenset({"supports", "refines"})
OPPOSING_POLARITIES:   frozenset[str] = frozenset({"contradicts", "supersedes"})

PROVENANCE_SOURCES: tuple[str, ...] = ("cooccurrence", "citation", "similarity", "llm_relation")
```

**Data classes (frozen where pure):**

```
@dataclass(frozen=True)
class ConceptPresence:
    concept_id: str
    document_id: str
    chunk_keys: tuple[str, ...]   # "{document_id}:p{parent_index}" (ADR-4); the chunks the term hit
    n_mentions: int

@dataclass(frozen=True)
class SkeletonEdge:
    source_concept_id: str
    target_concept_id: str
    provenance: frozenset[str]    # ⊆ PROVENANCE_SOURCES; Node A fills {cooccurrence, citation, similarity}
    weight: float                 # derived from provenance (Decision 5); deterministic
    n_cooccurrence_chunks: int    # chunk-level count (Decision 4)
    # LLM annotation (Node B; None/empty after Node A):
    stance_by_doc: tuple[tuple[str, str], ...] = ()   # (document_id, polarity) per asserting doc
    relation: str | None = None

@dataclass(frozen=True)
class ConceptNode:
    id: str            # the Concept.id
    label: str         # Concept.label (display surface form)
    doc_ids: tuple[str, ...]
    degree: int
    community: int

@dataclass(frozen=True)
class ConceptSkeleton:
    nodes: tuple[ConceptNode, ...]
    edges: tuple[SkeletonEdge, ...]
    communities: tuple[Community, ...]      # Louvain (ADR-1), seeded
    meta: dict[str, Any]                    # n_documents, n_concepts, embedding_model, graph_version
```

**Pure core (deterministic given its inputs; no DB, no LLM, no network):**

- `match_presence(concepts, aliases, chunk_texts, *, mode="boundary") -> list[ConceptPresence]` — case-folded
  surface-form match of each curated form against each chunk (Decision 2). The single presence primitive; the
  LLM never touches it. **`mode` (R2, 2026-07-02):** `"boundary"` (default) counts only alnum-bounded
  whole-word occurrences — the substring primitive over-matched short forms (`bert` inside `sbert`/`colbert`/
  `roberta`), fabricating co-occurrence edges and confounding RG-008/009; `"substring"` is retained as the A/B
  lever for the RG-008 comparison (`CONCEPT_SKELETON_PRESENCE_MODE` / `--presence-mode`). Word-boundary was
  the spec's own named upgrade lever (see RG-009).
- `cooccurrence_edges(presences, *, min_cooccurrence) -> list[SkeletonEdge]` — concept pairs co-present
  in ≥ `min_cooccurrence` **chunks** (Decision 4); each gets `provenance={"cooccurrence"}`,
  `n_cooccurrence_chunks` set.
- `add_citation_provenance(edges, citation_pairs, concept_doc_index) -> list[SkeletonEdge]` — for a
  `Citation(source_doc → target_doc)`, any concept present in `source_doc` and any present in
  `target_doc` that are *already* a co-occurrence edge gain `provenance ∪= {"citation"}`. **Never creates
  an edge** (Decision 5: citation/similarity annotate the co-occurrence skeleton; they do not link every
  cross-doc interest pair, which is the density blow-up Decision C's "links every interest in X to every
  interest in Y" warns about and RG-001 measures).
- `add_similarity_provenance(edges, doc_sim_pairs, concept_doc_index) -> list[SkeletonEdge]` — same shape
  for `DocSimilarity` edges; `provenance ∪= {"similarity"}`. Never creates an edge.
- `edge_weight(provenance, n_cooccurrence_chunks) -> float` — pure, deterministic weight from the
  provenance set + co-occurrence count (Decision 5). The Louvain `weight=` input.
- `contested_edges(edges) -> list[SkeletonEdge]` — edges whose `stance_by_doc` holds ≥2 docs with
  opposing polarities (the 7d `contested` signal, Decision 7). Empty after Node A (no stances yet).
- `analyze_skeleton(nodes, edges, *, seed=42, resolution=…) -> ConceptSkeleton` — degree, Louvain
  communities (ADR-1, seeded → bit-deterministic), `meta` + a `graph_version` fingerprint.
- `skeleton_to_dict` / `skeleton_from_dict` — JSON (de)serialisers (round-trip the structural payload, as
  `concept_graph.graph_to_dict`/`graph_from_dict` do). **Provide both** (`graph_from_dict` was a missing
  inverse that bit Feature 6 — do not repeat).
- `node_weights_for_epistemics(skeleton) -> dict[str, NodeWeight]` — expose the **same `NodeWeight`
  contract shape** `epistemics.compute_node_weights` consumes today (coverage ∈
  `{corroborated, unique, contested}`, direction ∈ `{stable, contested, superseded_trend}`, the
  unique-source = neutral rule). This is the seam 7d re-founds on; the actual re-point of `epistemics.py`
  is a separate change (Decision 8, Out of scope).

**Impure boundary:**

- `load_concepts(db) -> tuple[list[Concept], list[ConceptAlias]]` — read the curated vocabulary.
- `load_presence_inputs(...)` — pull chunk text + the `{document_id}:p{parent_index}` keys (the existing
  retrieval/store accessors; KI-5 — runs on the host, not the sandbox).
- `load_doc_graphs(db) -> tuple[citation_pairs, doc_sim_pairs]` — read `Citation` + `DocSimilarity`.
- `build_concept_skeleton(*, apply, min_cooccurrence, ...) -> SkeletonResult` — orchestrator: load →
  presence → co-occurrence edges → citation/similarity provenance → weight → Louvain → write
  `skeleton.json` + sidecar rows. **Zero LLM calls** (Node A).

### `src/doc_assistant/concept_skeleton_enrich.py` (new module, Node B — DEFERRED, specified not built)

- `annotate_relations(skeleton, present_by_doc, client: LLMClient) -> ConceptSkeleton` — for each
  document, one LLM call handed **only the concepts already present in that document**, returning a
  relation verb + stance ∈ `POLARITIES` for co-present pairs. Sets `relation` + appends to `stance_by_doc`
  on the *existing* edge; `provenance ∪= {"llm_relation}`. **Never** creates a node/edge (the
  by-construction guarantee, Decision 6). Provider-isolated (`assert_provider_intent`, default Ollama).
- Built in a later PR, gated on RG-001 landing (Decision 6 / Gate).

### Persistence — `src/doc_assistant/db/models.py` + `db/migrations.py`

New **tables** (all created by `create_all` — additive + idempotent, the `figures`/`chunk_epistemics`
precedent; **no `_ADDITIVE_COLUMNS` entry** since these are new tables, not new columns on existing ones):

- `Concept` (`concepts`): `id · label (indexed) · folder_id (nullable, FK folders, present-but-unused —
  Decision 9) · source ("keyword"｜"manual") · created_at`.
- `ConceptAlias` (`concept_aliases`): `id · concept_id (FK concepts, CASCADE, indexed) · alias (indexed)
  · created_at`. Unique `(concept_id, alias)`.
- `ConceptEdge` (`concept_edges`): `id · source_concept_id · target_concept_id (both FK concepts) ·
  provenance_json (JSON list ⊆ PROVENANCE_SOURCES) · weight · n_cooccurrence_chunks · relation
  (nullable) · stance_json (JSON [[doc_id, polarity], …]; nullable) · graph_version · computed_at`.
  **Sidecar; regenerable** — dropped + rebuilt with the skeleton (Enrichment-Layer Pattern). Index on
  `source_concept_id` and `target_concept_id`.
- `ConceptPresenceRow` (`concept_presence`): `id · concept_id (FK, indexed) · document_id (FK, indexed) ·
  chunk_keys_json · n_mentions · graph_version · computed_at`. Sidecar; regenerable.

`Concept`/`ConceptAlias` are **curated** (user data — survive a rebuild); `concept_edges` /
`concept_presence` are **derived** (dropped + rebuilt). Keep the two lifecycles distinct, exactly as
ADR-004 keeps curated vocabulary vs. regenerable gap rows distinct.

### Vocabulary seeding — `scripts/seed_concepts.py` (new) or a `build_concept_skeleton` sub-command

- Read `Keyword` rows → present them as **candidate** `Concept`s for the user to promote (the candidate
  is not auto-written as a `Concept` — Decision 1). The first increment's seeding is deterministic and
  free (no mining, no LLM). A `--promote <keyword>` / list flow attaches the keyword surface form as the
  concept label and seeds an alias. (Curation UI is a later surface; this is the CLI seam.)

### CLI runner — `scripts/build_concept_skeleton.py` (new)

- Mirror `scripts/build_concept_graph.py` exactly: `--apply` / `--force` / `--doc` + dry-run default
  (assemble from existing curated vocabulary + sidecar, no writes). **Node A is free** (deterministic);
  the dry-run prints node/edge/community/gap counts + the provenance breakdown.
- `--enrich` (Node B, deferred) gates the LLM relation/stance pass behind `--provider` (default `ollama`)
  + `assert_provider_intent`. Off by default → zero LLM calls.
- Windows stdout UTF-8 reconfigure + the `_format_report` shape as in the existing runner.

### `src/doc_assistant/config.py` — new block (mirror the `CONCEPT_GRAPH_*` block)

```
CONCEPT_SKELETON_DIR            = DATA_PATH / "skeleton"          # sidecar artifact root
CONCEPT_SKELETON_MIN_COOCCURRENCE = int(os.getenv(..., "2"))      # PROVISIONAL — set on RG-001
CONCEPT_SKELETON_SEED           = int(os.getenv(..., "42"))       # Louvain determinism (ADR-1)
CONCEPT_SKELETON_LLM_PROVIDER   = os.getenv(..., "ollama")        # Node B — local default, NOT LLM_PROVIDER
CONCEPT_SKELETON_LLM_MODEL      = os.getenv(..., "llama3.1:8b")   # Node B
```

Add `data/skeleton/` to `.gitignore` (sidecar artifact, like `data/graph/`).

---

## Build node

**Node A — deterministic skeleton (the first increment).**
**Depends on:** the curated `Concept`/`ConceptAlias` vocabulary (introduced here); `Citation` (shipped,
`db/models.py`); `DocSimilarity` (shipped — note it must be populated: `scripts/compute_doc_vectors.py
--apply`, the same prerequisite Feature 6 hit); the chunk store + the `{document_id}:p{parent_index}` key
(shipped). **No LLM, no network.** Files owned: `concept_skeleton.py`, `scripts/build_concept_skeleton.py`,
`scripts/seed_concepts.py`, the four new tables in `db/models.py`, `config.py` block, `.gitignore`, tests
below. **Status:** blocked (design-locked) on the RG-001 edge-precision run for *threshold-setting*
(`min_cooccurrence`, presence recall) — the model is buildable now; the thresholds are set from the run,
not guessed.

**Node B — confined LLM relation/stance enrichment (deferred).** Files: `concept_skeleton_enrich.py`,
the `--enrich` path. Re-founds 7d's stance layer on the skeleton. Gated on Node A + RG-001 landing.

**Out of this spec entirely (named so they are not silently pulled in):** candidate mining + the LLM
dedupe/label vocabulary pass; embedding-fallback presence; retiring `concept_graph.py` + re-pointing
`epistemics.py`/`wiki.py` (KI-7 connected change); the gap layer itself (`docs/specs/feature-gap-detection.md`);
the curation UI.

### Build sequence (PR order)

The load-bearing insight: **RG-001/008/009 are threshold-setting gates, not build blockers.** Node A is
buildable and fully testable on fakes (the guard tests below use toy inputs, no DB/LLM); only the real
`--apply` run and the threshold-locking need the RTX/Ollama box (KI-5: enrichment runs on the host). So
Node A can land **staged** behind the dry-run + provisional defaults now; the validation run closes the
gate later.

**Dependency chain (what must be true before each step):**

```
Curated vocabulary (Concept/ConceptAlias)  ─┐
DocSimilarity populated (compute_doc_vectors --apply)  ─┤
Citation graph (shipped)                    ─┼─> Node A skeleton ─> RG-001/008/009 run ─> thresholds locked
Chunk store + {doc_id}:p{parent_index} key  ─┘                              │
                                                                            └─> Node B (stance) ─> gap layer (ADR-004)
```

**Two prerequisites that bit Feature 6 — check them first:** (1) `DocSimilarity` must be **populated**
(`scripts/compute_doc_vectors.py --apply`) or similarity-provenance is silently absent; (2) the curated
`Concept`/`ConceptAlias` rows must exist — the first increment seeds candidates from `Keyword` + manual
`--promote` (Decision 1). **No vocabulary → empty graph.** Curation is a real product step, not a formality.

- **PR-A — Node A (this spec's first increment).** Schema + config → pure core (`match_presence` →
  `cooccurrence_edges` → `add_*_provenance` → `edge_weight` → `analyze_skeleton` → dict round-trip →
  `node_weights_for_epistemics`) → impure boundary + orchestrator → `seed_concepts.py` → CLI runner → guard
  tests → gate + docs. TDD, additive (new tables only). The no-edge-creation guard (citation/similarity
  annotate-never-create) is the most important test — it is the density-control invariant.
- **Gate — RG-001/008/009 (the keystone; needs the RTX/Ollama box or a paid OK).** Not a code step — a
  human-in-the-loop spot-check `--apply` run on the real ~61-doc corpus. **RG-008** (blocks-ship): edge
  precision per provenance tier; set `min_cooccurrence` + the trust threshold *from the run*. **RG-009**
  (degrades): presence recall vs a few hand-labelled docs. Record baselines in `tests/eval/baselines/`.
  RG-002/004 are moot under the redesign (nodes curated; co-occurrence is in the skeleton) — they close as
  "redesign re-founds this," don't re-run them against the old graph.
- **PR-B — Node B (deferred, gated on PR-A + RG-001).** `concept_skeleton_enrich.py`; re-founds 7d's stance
  layer. Provider-isolated, Ollama-default.
- **PR-C — gap layer (separate spec, gated on the skeleton + RG-001).** `docs/specs/feature-gap-detection.md`
  Tier-1 + Tier-2a floor; subsumes the wiki-6b + concept-7c signals.

**Risks / watch points (flagged, not re-decided):** (1) **empty-vocabulary trap** — PR-A is useless without
promoted concepts; budget time for a starter set. (2) **DocSimilarity must be populated** first. (3)
**density blow-up is the headline risk (RG-008)** — the chunk-level co-occurrence + annotate-never-create
design exists to prevent "every concept in doc X linked to every in doc Y"; if the run shows a dense
low-precision graph the lever is `min_cooccurrence` ↑, **not** a design change. (4) the run is
**environment-gated, not design-gated** — don't let "can't run RG-001 today" stall PR-A. (5) the **KI-7
connected change stays parked** (`epistemics.py` imports `concept_graph.py`); name the coupling, don't
silently wire it.

### Guard tests (written with the build)

- `tests/unit/test_concept_skeleton.py` — fixed toy inputs (curated concepts + aliases + synthetic chunk
  texts), pure, no DB/LLM:
  - alias + canonical surface forms match presence; a non-present term yields no `ConceptPresence`
    (presence determinism, Decision 2).
  - two concepts in the same chunk ≥ `min_cooccurrence` → one `SkeletonEdge` with
    `provenance={"cooccurrence"}`; below threshold → no edge (Decision 4).
  - a `Citation`/`DocSimilarity` over a pair that is **already** a co-occurrence edge adds the provenance
    token; a citation/similarity over a pair that is **not** a co-occurrence edge creates **nothing**
    (the no-edge-creation guard — the density-control invariant, Decision 5).
  - `edge_weight` is deterministic and ranks a multi-provenance edge above a co-occurrence-only edge.
  - `skeleton_to_dict` → `skeleton_from_dict` round-trips the structural payload exactly (the
    missing-inverse regression, Decision-7 carry-over).
  - Louvain communities are identical across two runs with the same `seed` (ADR-1 determinism).
- `tests/unit/test_concept_skeleton_weights.py` — `node_weights_for_epistemics` honours the
  **unique-source = neutral** rule (the 7d regression that matters): a sole-source concept is `unique`,
  never `contested`; an edge with ≥2 opposing-stance docs is `contested`. (Stances are injected directly
  in the fixture — no LLM.)
- `tests/integration/test_build_concept_skeleton.py` — seeded curated vocabulary + a small fixture
  corpus → `scripts/build_concept_skeleton.py` (Node A) writes `concept_edges` / `concept_presence` rows
  + `skeleton.json`; **idempotent re-run is a no-op and makes zero LLM calls**; `Concept`/`ConceptAlias`
  curated rows survive a `--force` rebuild while derived rows are dropped + rebuilt (the two-lifecycle
  guard, Decision 8 persistence split).
- `tests/integration/test_seed_concepts.py` — `Keyword` rows surface as candidates; a promote writes a
  `Concept` + seed alias; an un-promoted keyword writes no `Concept` (Decision 1).

### Definition of done

- A skeleton build (Node A) over the curated vocabulary produces `concept_edges` + `concept_presence`
  rows + `skeleton.json`; **the deterministic path makes ZERO LLM calls**; idempotent re-run is a no-op.
- Presence is exact/alias string match only; co-occurrence is chunk-level; citation/similarity **annotate
  but never create** edges (the no-edge-creation guard is tested).
- Every edge carries a `provenance` set + a deterministic `weight`; edges are **kept and ranked, never
  dropped** by the absence of an LLM stance.
- `node_weights_for_epistemics` exposes the existing `NodeWeight` shape with unique-source = neutral
  preserved (7d can re-found on it without a contract change).
- `Concept`/`ConceptAlias` (curated) survive a rebuild; `concept_edges`/`concept_presence` (derived) are
  dropped + rebuilt. Sidecar only — the chunk store is never mutated; `skeleton.json` + `data/skeleton/`
  are gitignored.
- No change to the retrieval path (public eval byte-identical — the skeleton is an additive read).
- ruff / `mypy --strict src` / bandit clean; no paid calls in tests (cpc §13); provider isolation holds
  on the Node-B `--enrich` path (when built).
- **Gate (RG-001, blocks marking this done as a *usable* graph):** the edge-precision validation run
  confirms the chunk-level skeleton is neither near-complete nor near-empty on the real corpus, and
  `min_cooccurrence` + the presence-recall threshold are **set from the run, not guessed**. Until then
  the model ships behind the dry-run + the provisional defaults; the gap layer (ADR-004) stays blocked on
  this same gate. Run free on local Ollama / host (KI-5); record a baseline under `tests/eval/baselines/`
  via the `rigor-gate` discipline.

## Out of scope

- **Candidate mining + the LLM dedupe/label vocabulary pass** — deferred; the first increment seeds from
  `Keyword` rows + manual promotion only (Decision 1).
- **Embedding-fallback presence** — exact/alias match only for now (Decision 2); the embedding path is a
  later recall lever and reintroduces a tunable threshold into the deterministic path.
- **Node B (LLM relation/stance enrichment)** — specified above, built in a later PR gated on RG-001.
- **Retiring `concept_graph.py` + re-pointing `epistemics.py` / `compute_epistemics.py` / `wiki.py`** —
  the KI-7 connected change; not safe as standalone cleanup until 7d re-founds on this skeleton
  (Decision 8).
- **The gap-detection layer itself** — `docs/decisions/ADR-004-gap-detection-layer.md` +
  `docs/specs/feature-gap-detection.md`; this spec only delivers the skeleton it is defined against.
- **The curation UI** — the first increment exposes the CLI seam (`seed_concepts.py`); the desktop
  surface is later.
- **Retrieval-rank use of the graph** — read-only; any rank use is a separate eval-gated experiment
  (same rule as 7d).
