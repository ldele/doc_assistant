<!-- status: draft · updated: 2026-07-27 · class: append-only -->

# ADR-032 — Outbound source verification: the first network feature

- **Status:** proposed (stub — needs `grill-me` before the Decision section is filled)
- **Date:** 2026-07-27
- **Deciders:** user (product), Claude (Cowork planning session 2026-07-27)
- **Plan:** `docs/PLAN_2026-07-27_maps-trust-reports.md` Track 2, T5

## Context

ADR-031's local tier can say what the corpus knows about a document. It cannot say whether the
document was peer-reviewed, in what venue, whether it has been **retracted or corrected**, or
whether a newer version exists. Those are the signals that turn "this source looks thin" into
"here is a better one" — the user's actual ask.

They require leaving the machine, on an app whose identity is **local-first**. `ROADMAP.md`
already anticipates this under *External literature discovery* and already says it *"needs its own
ADR (first outbound-network feature on a local-first app)"*, naming OpenAlex, Semantic Scholar,
Crossref, arXiv, Unpaywall and CORE, and excluding Sci-Hub as unauthorized distribution.

Prior work to reuse rather than re-plan: the 2026-07-17 local plan file (Claude Code scratch,
uncommitted, referenced from `ui-checklist.md` §3) already carves **PR-2 DOI** and **PR-8 Crossref
runner**. `Document.doi` exists. `metadata_enrich.py` + `scripts/enrich_metadata.py` establish the
idempotent-backfill-runner pattern this would follow.

The precedent for freshness honesty is `graph_version` / the E2 strip's "assessed as of" footer:
derived state displays when it was computed and warns when stale, rather than pretending currency.

Scope note: this ADR covers *verification of documents already in the corpus*. Discovery and
acquisition of new documents (the parked B13 gap→acquisition action, PR 17 Zotero ingest) build on
it but are not decided here.

## Options

1. **Stay local-only forever.** — *Pros:* zero architectural risk; the local-first promise is
   unqualified; no rate limits, no API etiquette, no offline-degradation paths. *Cons:* the
   strongest trust signals — retraction, peer-review status, newer versions — are structurally
   unavailable, and the "guide me to something better" half of the user's goal stays a corpus-
   internal shuffle.
2. **Opt-in, user-initiated, cached outbound lookups (proposed).** Off by default; a per-document
   or per-selection action; results land in a sidecar with a fetch timestamp. — *Pros:* the
   capability exists without the identity changing — nothing leaves the machine unless the user
   asks for it, per document. *Cons:* a genuinely new failure surface (network, rate limits,
   partial matches, stale caches); DOI-less documents match poorly on title+author; the UI must
   distinguish "not checked" from "checked, nothing found" without ever implying "unverified".
3. **Background enrichment on ingest.** — *Pros:* the data is simply there. *Cons:* silently
   phones home on every ingest, which is the local-first promise broken by default; also
   unbounded cost against public APIs' polite-use expectations.

## Decision

*(open — fill after `grill-me`)*

Candidate: **option 2**, with these constraints treated as part of the decision, not
implementation detail:

- **Off by default**, user-initiated per document or per selection. **Never on the answer path** —
  no turn ever blocks on a network call.
- **Sources:** Crossref (work type, venue, publisher, `is-referenced-by-count`, and retraction /
  correction notices via `update-to`) and OpenAlex (open-access status, citation velocity,
  newer-version detection). Both free, both expecting a polite-pool mailto identifier.
- **Key:** `Document.doi`, with a title+author fallback whose match confidence is stored and
  displayed — a fuzzy match must never be presented as a confirmed identity.
- **Storage:** a `document_external` sidecar table with `fetched_at`, never columns on `Document`.
  Enrichment-Layer Pattern: additive, idempotent runner, never mutates the primary store.
- **Degradation:** a failed or empty lookup renders as **"not checked"** or "no record found",
  never as "unverified". Absence of evidence must not display as evidence of absence.
- **The one loud signal:** a retraction or major-correction notice warrants a persistent badge and
  a report-level warning. Everything else is advisory, per ADR-004 / ADR-031.

## Consequences

*(open — fill with the Decision)*

Provisional:

- **Easy:** the runner shape is established (`metadata_enrich.py`); the freshness-display discipline
  is established (`graph_version`); the DOI carve already exists in the 2026-07-17 plan.
- **Hard / committed:** the first network dependency in a codebase with none — offline behaviour,
  timeouts, rate limiting, caching, and a polite-pool identity all become permanent obligations.
  Privacy posture must be documented in-app, not just in an ADR: the user should be able to see
  exactly what was sent and when. Frozen-build TLS trust (KI-10) becomes load-bearing for a second
  code path.
- **Boundary:** verification of existing documents only. Discovery, acquisition, and any automatic
  ingestion of found documents are out of scope and need their own decision.
