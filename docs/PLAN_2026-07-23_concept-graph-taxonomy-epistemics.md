<!-- status: active · updated: 2026-07-23 · class: disposable -->

# Concept graph → taxonomy → epistemic health — vision & pre-grill design note

**What this is.** A working note that clarifies *what we want from the concept-graph feature and where
we're going*, before grilling [ADR-019](decisions/ADR-019-concept-taxonomy-classification-layer.md)
into an amendment + a spec. **The taxonomy-shape spine was grilled and resolved 2026-07-23 — see §6
for the decision ledger.** It will feed an ADR-019 amendment (architecture-decision); the
epistemic-health cluster is parked to its own ADR (RG-023). Supersedes nothing yet.

Origin: a design conversation (Claude Code session 2026-07-23) that started from "work towards the
graph feature" and widened into three strands — the taxonomy shape, using the graph to boost the RAG,
and tying the graph to an epistemic-health system. This note keeps all three together because they
share one substrate (the concept graph) and one governing culture (rigor + surface-don't-block).

---

## 0 · Why the graph exists — two value props, both partly built

The concept graph is **not** navigation decoration. It earns its keep two ways, and the taxonomy sits
under both:

1. **Boost the RAG** — use concept structure to improve retrieval (recall expansion, re-ranking,
   BM25 query expansion). Strand B. *Eval-gated.*
2. **Drive curation & acquisition via epistemic health** — surface where the corpus is thin, stale,
   contested, or untrustworthy, and turn that into "what to read / acquire next." Strand C. This is
   ADR-004's gap layer + the acquire loop (gap → find → download → ingest → rebuild → gap closes),
   with external discovery (ui-checklist B13) as the action slot.

Neither is finished; both are *mostly wiring existing, disconnected machinery together* rather than
green-field building (the 2026-07-21 review headline: "richly built but almost entirely disconnected
from the answer/exploration flow").

---

## 1 · LOCKED this session (2026-07-23)

- **Polyhierarchy**, not a tree — a concept may have multiple parents. Structurally this is a **DAG**
  of the `broader`/`narrower` relation. (Reverses ADR-019 **D1**, which chose a tree; ADR-019 named
  polyhierarchy as the documented reopener.)
- **SKOS-shaped store.** Model the hierarchy as [SKOS](https://www.w3.org/TR/skos-primer/): concept
  nodes + `broader`/`narrower` edges (the hierarchy DAG) + `related` edges (the associative graph we
  already have as Node A/B). SKOS is the *vocabulary/shape*; it does **not** enforce acyclicity — that
  is **our invariant** (a cycle-check on edge insert; `nx.is_directed_acyclic_graph` on the build-time
  NetworkX artifact). This makes the hierarchy **data, not schema** → user-editable, scheme-agnostic.
- **Seed from a standard, but stay editable.** A strong foundation from a research classification, then
  the user reshapes freely. Standards are *imported seed rows*, never a vendored immutable asset
  (ADR-019 already decided this).
- **Every mechanic is cost-evaluated and eval-gated.** Prove offline first; no paid LLM at scale on an
  unproven pass (cost-discipline). Retrieval changes must beat the hybrid baseline on eval-10 with
  `--repeat`.
- **Two governing constraints hold throughout** (see §4).

---

## 2 · Strand A — the taxonomy (SKOS / DAG over the concept graph)

**Schema consequence of polyhierarchy.** ADR-019 **D5** (a single nullable `parent_id` FK = a tree)
becomes a **`concept_broader` edge table** (many-to-many). ADR-019 **C1** (domain nodes as a separate
entity) still holds — a domain is an abstract, zero-presence node kind distinct from a text-bearing
`Concept`. The associative `related` graph (co-occurrence / Node B stance) stays as it is; the two
layers coexist over the same nodes (a DAG + an undirected graph).

**Seed strategy — a graft, because no single scheme spans this corpus deeply** (ADR-019's measured
finding; the schemes agree at the trunk and diverge at the leaves):

| Layer | Source | Why | License |
|---|---|---|---|
| Trunk (domain → topic) | **[ANZSRC 2020 FoR](https://www.abs.gov.au/statistics/classifications/australian-and-new-zealand-standard-research-classification-anzsrc/latest-release)** — 23 divisions · 213 groups · ~1,700 fields; 2/4/6-digit codes | Only scheme covering both CS/ML *and* bio/neuro; cleanest license | CC-BY (visible attribution required) |
| Bio/neuro depth | **[MeSH](https://hhs.github.io/meshrdf/tree-numbers)** — descriptors with multiple tree numbers (native polyhierarchy) | Deep where ANZSRC stops | Public domain (US NLM) |
| CS/ML depth | **[ACM CCS 2012](https://dl.acm.org/ccs)** — poly-hierarchical, **ships as SKOS/XML** | Drops straight into a SKOS store | Free for education/research |

All land as **editable rows**; attribution obligations tracked per source. (Wikidata `P279 subclass-of`
is the one scheme with a *typed* is-a — relevant to the is-a open question below.)

**Open questions (grill):** (a) **is-a typing** — model "specific types of neurons at the bottom" as a
plain `broader` edge (simple; SKOS doesn't distinguish is-a from part-of from broader-topic) or a
*typed* edge (richer; the line between a classification and an ontology)? (b) which subtrees to graft vs
author by hand; (c) **coverage math under polyhierarchy** — if a paper sits under ML *and* neuroscience,
does it count once or twice? (ADR-019 flagged this; polyhierarchy forces the answer.)

---

## 3 · Strand B — graph → RAG (retrieval augmentation), eval-gated

**Locked principle (existing):** the graph is an **auxiliary signal on the existing hybrid pipeline**,
never the primary index — "don't make the concept graph a graph database" (ROADMAP). This is also the
answer to the *explosion* worry: don't traverse the graph as retrieval; layer bounded signals onto
`[BM25 + vector] → rerank → top_k`.

Three insertion points, by risk:

| Point | What | Explosion control | Verdict |
|---|---|---|---|
| **A · query expansion** (pre-retrieval) | query concepts → top-N graph neighbors/aliases → add terms to the query | bounded neighbor fan-out | **safest, highest value** (improves *recall*) |
| **B · candidate injection** (pre-rerank) | pull graph-linked docs that weren't retrieved into the candidate pool | 1 hop, top-N by edge weight, threshold; **`RERANK_CANDIDATE_CAP` (shipped 2026-07-23) is the ceiling** | **good, if bounded** |
| **C · rerank blend** (rerank stage) | blend a graph-proximity score into the cross-encoder score | none (fixed set) | **speculative** — cross-encoder already matches semantics |

**BM25 leverage:** BM25 is term-based, and the graph is a source of terms — **concept-driven sparse
query expansion** (add neighbor/alias terms to the BM25 arm) is the most natural fit, deterministic and
$0 (a cheaper cousin of the opt-in multi-query). Honest limit: because the reranker re-scores the full
union, BM25's leverage is on **recall (which docs become candidates), not final rank**.

**Gate:** any of A/B/C must beat the hybrid baseline on eval-10 with `--repeat`. Quality is **downstream
of graph quality** — on this box the graph is association-only (Node B stance NULL, KI-4), so
co-occurrence expansion drifts. Build a real graph (Node B / hierarchy) *before* measuring augmentation.

---

## 4 · Strand C — the epistemic-health system

Provenote already has ~70% of this, **scattered** across `epistemics.py` (markers), `gaps.py`
(structural), `reviewer.py` (hedging/citation/faithfulness), `provenance.py`, E2 (source strip), E4
(citations), E5 (triage). The valuable work is the **synthesis** into one diagnosis surface + a few
genuinely-new detectors.

### 4.1 Audit — built / partial / missing

| Mechanic | Status | Reality |
|---|---|---|
| **Temporal staleness** | 🟡 | **Semantic** supersession built (`superseded_trend`, G3/G6 — contradicting docs newer than supporting, gated to ≥2 dated docs/side). **Document-age** version **not** built (`Document.year` exists, 66/76). *We want BOTH* — see §4.3. |
| **Source provenance** | 🟢 | `provenance.py` + card + per-chunk attribution + `AnswerRecord`. |
| **Source trustworthiness** | 🔴 | No source-quality/reputation scoring anywhere. The "quality list" (B13). *Now first-class* — see §4.3. |
| **Scope & status (SKOS notes)** | 🟡 | No SKOS notes yet (the taxonomy). Status scaffolding exists: gap `status` + `GapTriage` (E5). `skos:editorialNote "needs refinement"` ≈ a gap flag. |
| **Confidence score** | 🟢* | Retrieval-derived (rerank score, coverage, `single_source`) + per-claim `ok/weak/unsupported`. *Hard rule:* never LLM self-report (§4.2). |
| **Contradiction detection** | 🟡 | `contested` = concept has opposing stances across docs (Node B `stance_by_doc`) — graph-level, **stance NULL on this box**, and RG-019 flags it *saturates* (1 disputing doc → contested). **Text-level claim-pair NLI not built.** |
| **Hedging & uncertainty** | 🟡 | Reviewer scores `hedging_adequacy` (0–5) + `no_hedge` flag **per answer** (persisted). Per-**source** syntactic hedge detection ("we think that") **not** built (cheap deterministic). |
| **Citation density ratio** | 🟡 | Reviewer `citation_density` (0–5) **per answer** (persisted). Per-concept density off the E4 citation graph **not** built. |

**Structural graph health** (the `calculate_concept_health` sketch) is ~2/3 the existing ADR-004 gap
layer: `is_orphan` (degree 0) = **`isolated`**; `is_underdocumented` (evidence < 2) = **`single_source`**
(RG-014's strongest signal); low degree = **`under_connected`** (+ `thin_area`, `thin_bridge`,
`unsourced_claim`). Only `is_outdated` (age) is new.

### 4.2 The dashboard matrix (graph-metric × text-metric → diagnosis → **user-triggered** action)

A good organizing frame; mostly *crossing signals we already have*:

| Graph metric | Text metric | Diagnosis | Action (user-triggered) | Built? |
|---|---|---|---|---|
| High links | High contradiction | Active dispute | Present both views (dual-interpretation synthesis) | pieces exist; cross not computed |
| Low links (<2) | Low citation | Underdocumented / speculative | Flag "needs refinement"; **prioritize for ingestion** | pieces exist (`single_source`/`thin_area` + gap-triage + acquire loop) |
| Any links | Stale (age) | Outdated | Suggest re-index / newer source | age-staleness not built |
| **Zero parents** | High evidence | **Island / misclassified** | **Suggest a `skos:broader` parent** | **taxonomy-dependent** — the new cell ADR-019 unlocks (= its auto-propose-parent-on-NULL) |

### 4.3 Refinements decided this session

- **Staleness must be DUAL and content-type-robust.** Keep the smart semantic `superseded_trend`
  *and* add document-age. Reason: age/recency matters most exactly where supersession-by-stance is
  weakest — **non-research content (SOPs, personal notes)** that isn't peer-reviewed and may carry no
  citations or clear polarity. The two answer different questions ("is this claim contradicted by newer
  work?" vs "is this document simply old?").
- **Source trust is what makes a heterogeneous corpus safe.** Peer-reviewed papers are a good default
  baseline; the moment we add Wikipedia, web pages, or personal notes, trust varies wildly (Wikipedia
  is uneven; personal notes contain mistakes). A **source-trust signal** (likely tiered by source
  *type* first — peer-reviewed > reputable-web > wiki > personal-notes — before anything fancier) is
  the precondition for widening ingestion. This is the "quality list" from B13, promoted to first-class.
- **Contradiction / hedging / citation-density are interesting at TWO levels** — the RAG-**answer**
  level (built, per-answer, reviewer) *and* the per-**concept** level (new: "once a concept is
  validated, how contested / hedged / well-cited is it across the corpus?"). The per-concept level is
  the open **cost** question — it implies a pass over a concept's evidence chunks; measure offline
  before scaling (see §5).
- **Content-type heterogeneity breaks paper-shaped assumptions.** The abstract/citation extraction
  strategy assumes paper structure (abstract, references section); a **book / SOP / note has neither**.
  So citation-density and citation-graph signals must **degrade honestly** on non-paper content, not
  silently report zero as if it were a finding (the 0-doc / robustness contract, generalized to
  content type). This is a cross-cutting constraint on every citation-dependent mechanic.

### 4.4 Two governing constraints (locked, project-wide)

1. **No LLM self-reported confidence.** Confidence/uncertainty is retrieval-derived or reviewer-derived
   only (coded rule in `synthesis.py` / `figures.py`). The matrix is clean on this (contradiction comes
   from graph stance, not from asking the model how sure it is).
2. **Surface, don't auto-remediate.** The matrix's "action" column must be **user-triggered suggestions**,
   never automatic re-scraping / re-indexing / acquisition ("don't auto-retry or auto-remediate — surface;
   the user decides" + inform-don't-block). E5's gap-triage (promote/dismiss/reset) is the pattern.

---

## 5 · Cost model (a first-class gate, per mechanic)

| Tier | Mechanics | Note |
|---|---|---|
| **$0 / deterministic** | age-staleness, structural health (gaps), per-source hedge lexicon, citation-density *count*, BM25/graph query expansion, SKOS structure | build/measure freely |
| **Local LLM ($0, RTX box)** | Node B stance (lights up `contested`/dispute), auto-propose parent, LLM source-classification | KI-4: force `--provider ollama`; RTX box only |
| **Paid LLM (measure first)** | per-concept contradiction (NLI), any per-concept LLM pass over evidence chunks | cost-discipline: prove offline on messy output + a local model before any at-scale run |

The per-concept passes (contradiction, hedging, density *at the concept level*) are the expensive ones
because cost scales with concept × evidence-chunks. **Measure the per-concept cost offline before
committing** — this is the recurring "evaluate computation cost" gate.

---

## 6 · Grill resolutions — decision ledger (2026-07-23)

The taxonomy-shape spine was grilled to closure. Each row routes to the **ADR-019 amendment**
(architecture-decision). "Reverses if" bounds each lock.

| Branch | Resolution | Deciding reason | Reverses if |
|---|---|---|---|
| **R · hierarchy scope** | **B — one unified typed SKOS DAG** (fields + concepts are all nodes; the domain→topic→concept→subtype spine is `broader`/`narrower` edges) | SKOS `broader` already expresses the whole spine; typing is the only addition needed; node-kind buys ADR-019's detector-safety without a second table. **Reverses ADR-019 D9** (which forbade concept is-a) | typed-edge curation proves too fiddly → fall back to two structures (C) or fields-only (A) |
| **Max depth** | **No hard cap; acyclicity is the structural invariant; deep-chain flagging is optional-advisory** | robustness contract bans magic-number caps; "depth" is multi-valued under polyhierarchy; acyclicity bounds structure and guarantees traversal termination | a 10k-concept traversal needs a hard bound for cost (a RIGOR_TODO measurement, not a design cap) |
| **Q1 · edge types** | **`is-a` + `in-field` + `related`; park `part-of`** | two hierarchical types deliver the is-a/field-of distinction; `part-of` re-invites Node B's measured meronymy mess (189/221) | coverage math needs `field→field` split from `concept→field` (→ split `in-field`), or a feature needs curated `part-of` |
| **Q2 · schema** | **(a) domains = `Concept` + `kind` column (supersedes ADR-019 C1); (b) new curated `concept_hierarchy` table (survives rebuild); `related` stays derived in `concept_edges`** | cross-kind edges need one node-id space; the curated hierarchy must survive the skeleton rebuild that *drops* `concept_edges` (KI-17/KI-20 class). Detector-safety via one canonical `presence_nodes()` accessor | domains grow domain-only columns that pollute `Concept` → separate table |
| **Q3 · coverage math** | **Set-semantics counting (distinct members, rollup dedup, no fractional, no forced primary); explicit `document_field` many-to-many (supersedes ADR-019 D6's single FK), auto-proposed on NULL** | distinct-set counts are explainable and don't leak sideways between sibling fields; explicit attachment covers the 25/47 concept-less docs derived-only would miss | rollup cost at 10k → materialize counts (RIGOR_TODO); auto-propose accuracy poor → coverage gaps stay behind RG-015 |
| **Q4 · seed graft** | **ANZSRC 2-level trunk only (23 div + 213 groups), editable rows; on-demand MeSH/ACM grafts; attribution required (About/Settings + seed header)** | bulk-importing 30k MeSH / 2k ACM nodes is PR-2.7's "facet that partitions nothing" at scale; seed the trunk that maps the corpus | concepts pile under one coarse group (trunk too shallow) → pull 6-digit fields or graft earlier |
| **Q8 · auto-propose parent** | **ADR-019 E1 pattern**: propose an `in-field` parent where the edge is NULL, from existing taxonomy nodes first (a new MeSH/ACM graft is the heavier secondary action), user-gated accept/edit, **$0/Ollama** (KI-4) | same mechanism as `document_field` auto-propose; already decided by E1 | — (settled by E1) |

**Parked to a dedicated epistemic-health ADR (ADR-EH), sequenced *after* the concept graph is
validated** — routed to **RG-023**. Blocked on measurement (per-concept contradiction/hedging/density
cost) and a heterogeneous corpus (source-trust). Design already captured in §4–6 of this note:
contradiction/hedging/density placement (answer vs per-concept), source-trust scoring, dual +
non-paper staleness, content-type degradation, and degree-based-detector retirement (RG-015).

---

## 7 · Where we want to go (sequence)

1. **Grill the ADR-019 forks** (§2/§6) → **amend ADR-019** (polyhierarchy + SKOS shape + is-a decision).
2. **Spec + build the taxonomy**: seed (ANZSRC trunk + grafts) → curation view (edit the DAG) →
   auto-propose assignments where the FK is NULL.
3. **Node B stance regen** (RTX box, KI-4) → lights up `contested` / the dispute cell / stance-dependent
   retrieval quality.
4. **Epistemic-health synthesis** — the dashboard matrix as a surface, crossing existing signals + the
   new detectors (age-staleness, source-trust, per-source hedging), all surface-don't-remediate.
5. **Graph → RAG augmentation** as a measured experiment (Strand B; A+B first, eval-gated).
6. **Acquisition loop** — external discovery / quality list (B13), closing the gap → acquire cycle.

> This note is `class: disposable` — once ADR-019 is amended and specs exist, they are canonical and
> this note is superseded.
