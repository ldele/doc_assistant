<!-- status: active · updated: YYYY-MM-DD · class: living -->

# FEATURE-NNN — <feature name>

> Per-feature rationale — *why this works and why we built it this way*.
> **Not** code structure (that's `docs/architecture.md`), **not** status (that's `docs/ROADMAP.md`),
> **not** a decision record (that's `docs/decisions/`). This is the *why-it-works* layer.
> Update when understanding changes — not every session. One file per feature; no coding glue.

**Roadmap row:** <docs/ROADMAP.md PR/phase this maps to>
**Tier:** <prior-art | frontier>  <!-- prior-art = someone did this before; frontier = no reference, hypothesis→test stands in -->

## Hypothesis
<What we believe will work, and the outcome we expect. State it so reality can contradict it.>

## Grounding
<Prior-art tier: who/what org did this before, with a link — the source we're standing on.
Frontier tier: no prior art, so name the hypothesis→test that stands in for a source, and link
the rigor-check (RG-NNN) that runs it. One or the other is always present.>

## Design choices & why
<Per component: what it does · which algorithm/approach · why that one over the alternatives.
For a pipeline feature (e.g. RAG), one row per stage:>

| Component | What it does | Choice | Why this over alternatives |
|-----------|--------------|--------|----------------------------|
| <stage>   | <role>       | <algo> | <the reason, not "best practice"> |

## Real-life outcome
<Filled at validation — what actually happened when built vs. the hypothesis. Link the
rigor-check / eval that proved it, and any known issue it surfaced. Empty until validated.>

## Links
- Roadmap: <docs/ROADMAP.md row>
- ADRs: <docs/decisions/ADR-NNN-*.md>
- Rigor: <RG-NNN / test path>
- Known issues: <.claude/KNOWN_ISSUES.md KI-NNN>
