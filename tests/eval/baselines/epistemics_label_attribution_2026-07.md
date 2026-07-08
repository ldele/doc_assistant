<!-- status: active · updated: 2026-07-08 · class: baseline -->

# Epistemics chunk-attribution fix (SPRINT-007 / KI-15)

Records the real-corpus before/after for the `epistemics.concepts_in_text` fix — matching
concept **labels** instead of node **ids** — and a manual spot-check that the fix attributes
correctly, not just non-zero.

## The bug (KI-15)

`epistemics.py`'s structural attribution regex-searched chunk text for each skeleton node's
**id**. That was correct for the retired open-vocabulary `concept_graph.py` (node id =
`canonical_key(label)`, a real lowercase string). The curated `concept_skeleton.py` (Node A) that
replaced it uses the `Concept.id` **UUID** primary key as the node id — a UUID never occurs in
document text, so attribution silently returned nothing on the real corpus, on every chunk, for
every concept, regardless of how correct the underlying node weights were. Existing unit tests
didn't catch it because their fixture ids doubled as valid labels (`"bm25"`, `"rag"`) — see
`docs/sprints/SPRINT-007-fix-epistemics-label-attribution.md` for the full writeup and
`.claude/KNOWN_ISSUES.md` KI-15 for the discovery context (found during G6's real-corpus run).

## The fix

`concepts_in_text(text, labels_by_id)` now matches on `label`, casefolded, via
`concept_skeleton.compile_boundary_pattern` — the exact alnum-boundary regex Node A's own
presence matcher uses (R2: alnum lookarounds, not `\b`, so non-word edge characters like
`gpt-4`/`c++` are handled correctly). One shared definition, not two. `project_chunk_weights`
passes `{n.id: n.label for n in skeleton.nodes}` instead of a bare id list.

## Real-corpus before/after (this box, no skeleton rebuild — same snapshot as G6)

`compute_epistemics --apply` against the skeleton G6 already built this session (357 nodes, 226
contested / 9 superseded_trend, 46 Node-B LLM calls, no LLM cost for this step — projection is
free/read-only):

| Metric | Before (id-matching, the bug) | After (label-matching, the fix) |
|---|---|---|
| Chunks with a claim | 0 | **4008** / 6215 (64.5%) |
| Chunks marked (>=1 marker) | 0 | **3334** / 6215 (53.6%) |
| Rows written | 0 | 4008 |
| Runtime | n/a | ~34s (357 labels x 6215 chunks, regex cached) |

## Spot-check (not just non-zero — actually correct)

Pulled one marked chunk (`040bfe8e-03c7-4732-9f62-27fe1ddffb21:53`, a Res2Net paper) and manually
verified `concepts_in_text` against its real text:

> "...segmentation [46], salient object detection [21], interactive image segmentation [41]...
> Semi-supervised knowledge distillation solution [50] can also be applied to Res2Net... VOC07
> ResNet-50 / Res2Net-50..."

Attributed: `res2net-50`, `knowledge distillation`, `salient object`, `salient object detection`,
`res2net`, `image segmentation` — all six genuinely present in the text, no false positives from
the boundary regex (e.g. `res2net` doesn't spuriously fire on `res2net-50` being present too —
both are distinct curated concepts and both are legitimately in the text).

## Not investigated further (aside, not a defect)

53.6% of all real chunks carry at least one marker. That's a large fraction, driven upstream by
226/357 (63%) of concepts being `contested` in this corpus's skeleton — plausible for a
broad multi-domain corpus with many actively-debated/co-occurring concepts, not obviously a
false-positive artifact (the spot-check above found none), but worth a wider spot-check or a
tighter look at the corpus's `contested` rate before leaning on marker density as a UI signal.
Out of scope for this sprint, which fixes the attribution mechanism, not the marking rate.

## Follow-up (documented, not this sprint)

- Live UI smoke test: confirm the desktop chat's evidence chips now render `contested`/
  `superseded_trend` on a real answer (PR-M1's read side — `markers_for_chunk_keys`/
  `markers_for_parent` — was never the broken part, but hasn't been exercised end-to-end since
  before this fix).
- Parent-child (PC) chunk store re-projection remains a separate, already-documented follow-up
  (`docs/specs/pr-m1-epistemics-markers.md` ADR-1 option 2) — this sprint only fixes the existing
  baseline flat-chunk projection.
