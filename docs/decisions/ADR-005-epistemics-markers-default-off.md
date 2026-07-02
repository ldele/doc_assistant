<!-- status: active · updated: 2026-07-02 · class: append-only -->

# ADR-005 — Live 7d epistemics markers default OFF until Node B

- **Status:** accepted
- **Date:** 2026-07-02
- **Deciders:** user (option chosen), Claude Code (execution)

## Context

The app's differentiator is the research-integrity layer. PR-M1 surfaces live 7d "epistemics
markers" — `⚠ contested in corpus` / `⚠ trend superseded` chips — on source cards during a
chat turn (`ChatController._attach_markers`). But the marker **data** is computed from the
**superseded open-vocabulary concept graph** (KI-7), and for parent-child retrieval it reaches
sources through a coarse text-containment join (KI-8). So the one place the product visibly
asserts epistemic authority is showing noise from a component we have already decided not to
build on. Of the 2026-07-02 review findings, this is the only one that works *against* the core
promise rather than merely leaving value on the table (remediation plan R7).

Full retirement of the open-vocabulary graph is a connected change across `epistemics.py` →
`chat_controller.py` / `compute_epistemics.py` / `wiki.py` + tests (KI-7 "Cleanup when built"),
and is only worth doing bundled with **Node B** — the confined-LLM relation/stance layer that
will produce trustworthy marker data. We need a cheap containment move now, ahead of that.

## Options

1. **Config kill-switch, default off (chosen).** New `EPISTEMICS_MARKERS_ENABLED` (default
   `false`) gates `_attach_markers`. — *Pros:* the chip is already quiet-on-clean, so the UI
   needs no change; the M0/M1 parity guarantee (byte-identical turn when markers absent) becomes
   the shipped default path; trivially reversible — Node B flips one default. *Cons:* the
   plumbing stays in place, dormant.
2. **Keep on, label the chip "experimental" in the renderers.** — *Pros:* still shows the
   signal. *Cons:* it still surfaces KI-7 noise under the integrity layer's banner; a hedge label
   does not undo the trust cost, and it touches every renderer.
3. **Full retirement now.** — *Pros:* removes the dead path entirely. *Cons:* the known connected
   change across four modules + tests, only sensibly done with Node B; premature and risky here.

## Decision

Option 1. Add `EPISTEMICS_MARKERS_ENABLED` (default `false`); `_attach_markers` returns before any
epistemics read when it is off. Node B re-enables the default once markers rest on trustworthy
data. The marker code, tests, and API/renderer `markers` field are all retained.

## Consequences

- **Easy:** the default turn is the byte-identical no-marker path (the existing parity test now
  guards the shipped default); re-enabling is a one-line default flip; no renderer or API change.
- **Hard / committed:** the marker plumbing lives on unused until Node B — a small carrying cost,
  and a reminder that KI-7 retirement is still owed. Anyone testing marker behaviour must set the
  flag explicitly (the unit tests do).
- **Reversible:** purely a default; `EPISTEMICS_MARKERS_ENABLED=true` restores the PR-M1 behaviour
  for anyone who wants it today.
