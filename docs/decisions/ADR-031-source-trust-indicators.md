<!-- status: active · updated: 2026-07-27 · class: append-only -->

# ADR-031 — Source-trust indicators: named signals, no composite score

- **Status:** proposed (stub — needs `grill-me` before the Decision section is filled)
- **Date:** 2026-07-27
- **Deciders:** user (product), Claude (Cowork planning session 2026-07-27)
- **Plan:** `docs/PLAN_2026-07-27_maps-trust-reports.md` Track 2

## Context

The user's stated goal: *not all sources are trustworthy; where they are not, help the user find
more reliable information in a guided way* — with the explicit acknowledgement that this is not
fully programmatic and that **indicators**, not verdicts, are what is wanted.

**The gap is real and specific: the app measures corroboration, not trustworthiness.** Everything
shipped under Feature 7d / ADR-027 answers "do this corpus's documents agree with each other" —
`coverage` (`corroborated` / `unique` / `contested`), `superseded_trend` from relative year
ordering of contradicting vs supporting documents, gated at ≥2 dated docs per side. Ten agreeing
weak sources score `corroborated`. A correct paper contradicted by three older wrong ones trends
`superseded`. The E2 strip is a good instrument aimed at a different question than the one now
being asked, and users will read it as a trust score unless the surface says otherwise.

Audited state: **there is no source-quality or reputation scoring anywhere in the codebase.**
`Document` carries `title`, `authors`, `year`, `doi`, `format`, `extractor_used`,
`extraction_health`, `page_count`, `chunk_count`. There is **no** venue, publisher, document-type,
or peer-review field. `DocumentMeta` (ADR-013) is the established user-override sidecar for
title/authors/year.

Two prior refusals govern the shape of any answer here: **no self-reported LLM confidence**
(project-wide, and measured — small local models' confidence was anti-correlated with correctness
on one model), and **surface, never auto-remediate** (ADR-004). ADR-027 further splits *assessment*
(always on) from *influence* (opt-in).

For comparison, NotebookLM ships nothing on this axis at all — Google's documented position is
that it "can't discern truth from fiction" and weighs no source differently from another. This is
the axis on which Provenote is not competing with them.

## Options

1. **A composite trust score (0–100, or a letter grade).** — *Pros:* one number, instantly legible,
   sortable, filterable. *Cons:* launders incommensurable signals — provenance completeness,
   corroboration, and external standing are three different questions — into false precision. It
   is the same error class already refused twice (LLM self-confidence; auto-remediation). It is
   also unfalsifiable: there is no ground truth to calibrate the weights against, and the project's
   own rule is that unmeasured numbers do not ship.
2. **Named, individually-inspectable indicators, grouped into bands (proposed).** — *Pros:* each
   signal keeps its own meaning, its own derivation, and its own "why am I seeing this" trace;
   honest about `unknown`; extends without renegotiating a weighting. *Cons:* more UI surface than
   one badge; requires the user to read three things instead of one; needs deliberate labelling so
   the bands are not mentally averaged anyway.
3. **LLM-judged source credibility.** — *Pros:* handles sources with no metadata. *Cons:* directly
   violates the no-self-reported-confidence rule; unfalsifiable; and the one thing measured about
   local models here is that their confidence carries almost no signal.

## Decision

*(open — fill after `grill-me`)*

Candidate: **option 2**, structured as three bands that are displayed together and never averaged:

1. **What kind of source is this** — a new `Document.source_type`
   (`peer_reviewed` · `preprint` · `book` · `thesis` · `standard` · `report` · `web_page` · `note` ·
   `unknown`), deterministically and *partially* derived (arXiv pattern → `preprint`; DOI present →
   *candidate* peer-reviewed, never auto-promoted, since DOIs are minted for preprints and reports
   too; EPUB with chapters → `book`), with a `DocumentMeta`-style user override. **`unknown` is a
   first-class displayed value, not a hole to be guessed.** Plus provenance completeness over
   fields that already exist — explicitly labelled a *hygiene* signal: a complete record is not a
   true one.
2. **How the corpus treats it** — the shipped E2 coverage/direction signals, unchanged.
3. **Retrieval confidence** — the shipped rerank score, `weak_retrieval`, `single_source_risk`.

Plus the **guided escalation**: when an indicator is weak the app offers a *lead*, never a verdict —
other documents touching the concept (`concept_presence`), related papers (`doc_similarities`),
the contested concept's ego view, the gap list, and (behind ADR-032) an outbound lookup. Escalation
logic lives in one read model (`knowledge/leads.py`), not re-derived per panel.

## Consequences

*(open — fill with the Decision)*

Provisional:

- **Easy:** bands 2 and 3 are already computed and already rendered; T1/T2 are an additive column,
  an override row, and deterministic derivation over existing fields. All $0, no LLM, no network.
- **Hard / committed:** the surface must actively prevent the three bands from reading as one
  score — labelling is load-bearing, not decoration. `unknown` must be shown, not hidden. Every
  indicator owes a "why am I seeing this" trace, which is a UI obligation on every future indicator
  added to the set.
- **Boundary:** indicators inform; they never gate retrieval, never reorder sources, and never
  suppress a document. Inherited directly from ADR-004 (surface, never auto-remediate) and ADR-027
  (assessment always on, influence opt-in). Any future proposal to *weight retrieval* by a trust
  indicator reopens this ADR and needs the eval harness.
