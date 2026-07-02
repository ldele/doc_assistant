<!-- status: active · updated: 2026-07-02 · class: append-only -->

# ADR-008 — Concept skeleton R5 decision run: lock K=2 + boundary, PASS, proceed to Node B

- **Status:** accepted
- **Date:** 2026-07-02
- **Deciders:** user (data-home = main corpus; vocabulary sign-off), Claude Code (measurement + verdict)

## Context

The concept-graph redesign (KI-7) built a deterministic, zero-LLM skeleton (Node A, `concept_skeleton.py`)
whose two headline knobs — `CONCEPT_SKELETON_MIN_COOCCURRENCE` and `CONCEPT_SKELETON_PRESENCE_MODE` — were
shipped **PROVISIONAL** (config comments: "set from the RG-001/008 edge-precision run"). The 2026-07-01
runs were confounded (KI-14 placeholder noise, substring presence inflation, DF-only keyword scoring). The
2026-07 remediation plan R1–R4 removed those confounds deterministically; **R5 is the clean re-measurement
and the go/no-go** on the edge model + the ADR-004 gap layer. ADR-004 explicitly blocked Tier-1 gap
detection on an *unvalidated* edge precision (line 165–173): an over-connected graph reports near-zero
gaps, an under-connected one reports nearly everything.

The dedicated multi-domain data home (`data_multidomain/`) and its manifest are absent on this box, so R5
ran on the **main corpus** (76 docs) per the user's decision. That corpus turned out to be a natural
two-cluster mix (IR/representation-learning + neuroscience/connectome/animal-pose), giving a **partial**
doc-similarity graph — the regime where R4's graded provenance strength can discriminate.

## Options

1. **Lock K=2 + boundary (chosen).** The pre-registered sweep (bands chosen a priori) put K=2 at 21.5%
   density with a clean 3-topic community split and a spread provenance-strength distribution. K=2 +
   boundary were already the provisional defaults, so this **validates** them without a value change. —
   *Pros:* matches the measured sweet spot; no locked-setting edit; closes RG-008/009; unblocks ADR-004.
   *Cons:* validated on the main corpus, not the (absent) dedicated multi-domain home.
2. **Lock K=3.** 18.8% density, splits a contrastive-representation sub-community out — a defensible finer
   granularity. Rejected as the default: K=2's 3-way split already maps to the corpus's real topics and is
   denser (more relational signal); K=3 is available per-run via the env var / CLI.
3. **FAIL / descope.** Keep the glossary + presence layer, park the edge model + gap layer. Rejected — the
   measurement passed every band.

## Decision

**PASS.** Lock (promote provisional → validated) `CONCEPT_SKELETON_MIN_COOCCURRENCE = 2` and
`CONCEPT_SKELETON_PRESENCE_MODE = boundary`. Values unchanged; only the config comments move from
"provisional" to "validated by R5". Full evidence: `tests/eval/baselines/rg001_concept_skeleton_r5_2026-07-02.md`.

Measured (boundary K=2, 26 curated concepts / 17 aliases, user-signed-off):
- **Density 21.5%** (band 15–35% ✅), 3 communities mapping to retrieval / pose-vision / connectome (✅).
- **Provenance strength** median 0.52, range [0.09, 1.0] (✅ spreads — R4 discriminates on the partial
  graph; the substring A/B halves the median to 0.23 and inflates density to 36%, confirming R2).
- **Presence recall** 26/26 concepts present (✅).
- **Gap layer** (ADR-004 Tier-1): 0 isolated, 3 single-source (PHATE, Res2Net, SBERT), 1 thin bridge
  (MedSAM—Embeddings), 1 under-connected. ≥3 meaningful, actionable signals; degree spread 1→20 =
  healthy edge precision.

## Consequences

- **Closes RG-008 / RG-009** (edge-precision + presence-recall rigor items).
- **Unblocks ADR-004 Tier-1** gap detection — edge precision is now validated (the healthy, discriminating
  gap set is the evidence ADR-004 required).
- **Next:** Node B / PR-B (the confined LLM relation/stance pass, `concept_skeleton_enrich.py`,
  Ollama-default) is unblocked; R6 (BM25 core retrieval) remains independent.
- **Carries a scope caveat:** validated on the two-cluster *main* corpus. Re-running on the dedicated
  multi-domain home (6 domains) would be a stronger partial-graph stress test but is not required for the
  PASS — the main corpus already exhibits the partial-graph regime R4 targets.
- The curated 26-concept vocabulary now lives in `Concept`/`ConceptAlias` on the main data home (host,
  gitignored); the skeleton is regenerable from it via `build_concept_skeleton --apply`.
