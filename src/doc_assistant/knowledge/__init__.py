"""Knowledge layer (Phase 7) — the enrichment sidecars derived *from* the corpus.

Concept vocabulary + skeleton graph (``concept_skeleton`` Node A, ``concept_skeleton_enrich``
Node B, ``concept_curation``, ``concept_semantics``, ``concept_graph_view``), mined keywords and
families (``keywords``, ``keyword_families``), the wiki/synthesis notes (``wiki``), gap detection
(``gaps``, ``gap_suggest``) and chunk-level epistemics markers (``epistemics``).

Every module here follows the Enrichment-Layer Pattern (`.claude/CONTEXT.md` non-negotiable #4):
additive sidecar tables/artifacts, an idempotent CLI runner in ``scripts/``, and no writes to the
primary chunk store. The RAG answer path (``pipeline``, ``chat_controller``, ``synthesis``, …)
stays at the package top level and *reads* this layer; it never depends on it to answer.
Restructured out of the flat package by ADR-023 (2026-07-19); import as
``doc_assistant.knowledge.<module>``.
"""
