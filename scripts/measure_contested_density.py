"""Measure `contested` marker density and what a threshold change would do to it (RG-019).

Read-only and free: no LLM, no writes, no `--apply`. Loads the live concept skeleton, computes
node weights exactly as the shipped `node_weights_for_epistemics` does, projects them onto the
real chunk segmentations, then re-derives coverage under candidate floors and re-projects — so
every "what if we raised the threshold" number is measured on real data rather than argued.

Three levers are reported, because RG-019 named only the first and the first turns out to be
inert (see `tests/eval/baselines/contested_density_2026-08-02.md`):

  * ``nc`` floor            — minimum disputing documents before a node is `contested`
  * ``agreement_ratio``     — the value computed beside `coverage` and consulted by nothing
  * the chunk-level rule    — a chunk is marked if **any** of its claims sits on a contested node

Then the finding that reframes all three: the **parent-field join**. Every graph concept carries
exactly one ANZSRC parent (ADR-028), and contestedness turns out to be near-perfectly confounded
with that parent — the marker tracks corpus density per field, not dispute. Reported last because
it is the reason a global cut point is the wrong primitive (ADR-040).

Usage::

    uv run --no-sync python -m scripts.measure_contested_density
    uv run --no-sync python -m scripts.measure_contested_density --skeleton-dir data/skeleton
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from doc_assistant.config import CONCEPT_SKELETON_DIR
from doc_assistant.knowledge.concept_skeleton import (
    SKELETON_NAME,
    ConceptSkeleton,
    NodeWeight,
    node_weights_for_epistemics,
    skeleton_from_dict,
)
from doc_assistant.knowledge.epistemics import (
    load_doc_chunks,
    load_pc_parent_chunks,
    project_chunk_weights,
)

NC_FLOORS = (1, 2, 3, 4)
AGREEMENT_FLOORS = (1.01, 0.75, 0.70, 0.60, 0.50)


def _recover(w: NodeWeight, *, min_nc: int = 1, max_agreement: float = 1.01) -> NodeWeight:
    """Re-decide one node's coverage under candidate floors.

    Keeps the shipped precedence exactly — contested first, then the unique-source neutrality
    rule (Decision 4: a sole source is never contested), else corroborated — so the only thing
    varying between runs is the threshold under test.
    """
    ns, nc = w.n_supporting_sources, w.n_contradicting_sources
    if nc >= min_nc and w.agreement_ratio < max_agreement:
        coverage, direction = "contested", w.direction
    elif ns <= 1:
        coverage, direction = "unique", "stable"
    else:
        coverage, direction = "corroborated", "stable"
    return dataclasses.replace(w, coverage=coverage, direction=direction)


def _load(skeleton_dir: Path | None) -> ConceptSkeleton:
    root = skeleton_dir or CONCEPT_SKELETON_DIR
    path = root / SKELETON_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No concept skeleton at {path} — run "
            "`python -m scripts.build_concept_skeleton --apply --enrich` first."
        )
    print(f"skeleton: {path}")
    return skeleton_from_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--skeleton-dir", type=Path, default=None, help="override CONCEPT_SKELETON_DIR"
    )
    args = ap.parse_args()

    skeleton = _load(args.skeleton_dir)
    weights = node_weights_for_epistemics(skeleton)
    doc_chunks = load_doc_chunks() + load_pc_parent_chunks()
    rows = project_chunk_weights(skeleton, weights, doc_chunks)
    assessed, segments = len(rows), len(doc_chunks)
    marked = sum(1 for r in rows if r.markers)

    # --- denominators. The headline "53% contested" is conditional on being assessed at all,
    # and quoting it without the second row overstates the reach of the marker by ~25x.
    print("\n=== denominators ===")
    print(f"chunk segments in the store : {segments}")
    print(f"carrying any claim          : {assessed} ({assessed / segments * 100:.1f}% of store)")
    print(f"of those, marked contested  : {marked} ({marked / assessed * 100:.1f}% of assessed)")
    print(f"marked, share of the store  : {marked / segments * 100:.1f}%")

    print("\n=== nodes ===")
    print(f"nodes: {len(weights)}  edges: {len(skeleton.edges)}")
    print("coverage today :", dict(Counter(w.coverage for w in weights.values())))
    labels = {n.id: n.label for n in skeleton.nodes}
    print(f"\n{'contested node':<34} {'ns':>4} {'nc':>4} {'agree':>7} {'direction':>18}")
    for w in sorted(
        (w for w in weights.values() if w.coverage == "contested"), key=lambda w: w.agreement_ratio
    ):
        print(
            f"{labels.get(w.node_id, w.node_id)[:33]:<34} {w.n_supporting_sources:>4} "
            f"{w.n_contradicting_sources:>4} {w.agreement_ratio:>7.3f} {w.direction:>18}"
        )

    def sweep(label: str, variants: dict[str, dict[str, float | int]]) -> None:
        print(f"\n=== {label} ===")
        print(f"{'rule':<30} {'contested nodes':>16} {'marked':>7} {'% assessed':>11}")
        for name, kwargs in variants.items():
            w2 = {k: _recover(v, **kwargs) for k, v in weights.items()}  # type: ignore[arg-type]
            r2 = project_chunk_weights(skeleton, w2, doc_chunks)
            m2 = sum(1 for r in r2 if r.markers)
            n2 = sum(1 for v in w2.values() if v.coverage == "contested")
            print(f"{name:<30} {n2:>16} {m2:>7} {m2 / len(r2) * 100:>10.1f}%")

    sweep(
        "lever A — minimum disputing documents",
        {f"nc >= {k}{'  (today)' if k == 1 else ''}": {"min_nc": k} for k in NC_FLOORS},
    )
    sweep(
        "lever B — agreement_ratio band",
        {
            f"agreement < {t:.2f}{'  (today)' if t > 1 else ''}": {"max_agreement": t}
            for t in AGREEMENT_FLOORS
        },
    )

    # --- lever C needs no re-projection: the rule reads counts already on each row.
    print("\n=== lever C — the chunk-level rule ===")
    print(
        "n_claims per assessed chunk   :", dict(sorted(Counter(r.n_claims for r in rows).items()))
    )
    print(
        "n_contested per assessed chunk:",
        dict(sorted(Counter(r.n_contested for r in rows).items())),
    )
    print(f"\n{'rule':<34} {'marked':>7} {'% assessed':>11}")
    chunk_rules = {
        "n_contested >= 1  (today)": lambda r: r.n_contested >= 1,
        "n_contested >= 2": lambda r: r.n_contested >= 2,
        "n_contested >= 3": lambda r: r.n_contested >= 3,
        "contested majority of claims": lambda r: (
            bool(r.n_claims) and r.n_contested * 2 > r.n_claims
        ),
        "ALL claims contested": lambda r: bool(r.n_claims) and r.n_contested == r.n_claims,
    }
    for name, rule in chunk_rules.items():
        n = sum(1 for r in rows if rule(r))
        print(f"{name:<34} {n:>7} {n / assessed * 100:>10.1f}%")

    _report_parent_fields(weights, labels)


def _report_parent_fields(weights: dict[str, NodeWeight], labels: dict[str, str]) -> None:
    """Join each node to its ANZSRC parent field (ADR-028) and cross-tabulate contestedness.

    This is the measurement that reframes the other three: if contested/not is predicted by the
    parent field, the quantity being surfaced is corpus density per field, not dispute, and no
    global threshold on a per-concept rate can separate them.
    """
    from doc_assistant.db.models import Concept, ConceptHierarchy
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        parents = {
            src: name
            for src, name in session.query(ConceptHierarchy.source_id, Concept.label).join(
                Concept, Concept.id == ConceptHierarchy.target_id
            )
        }

    print("\n=== parent field (ADR-028) x contestedness ===")
    per_field: dict[str, list[int]] = {}
    for node_id, w in weights.items():
        field = parents.get(node_id, "(no parent)")
        seen = per_field.setdefault(field, [0, 0])
        seen[0] += 1
        seen[1] += 1 if w.coverage == "contested" else 0

    print(f"{'parent field':<40} {'concepts':>9} {'contested':>10}")
    for field, (n, n_contested) in sorted(per_field.items(), key=lambda kv: -kv[1][1]):
        print(f"{field[:39]:<40} {n:>9} {n_contested:>10}")

    placed = sum(1 for n in weights if n in parents)
    print(f"\n{placed}/{len(weights)} concepts carry a parent field")
    print(f"{'concept':<34} {'contested':>10}  parent field")
    for node_id, w in sorted(weights.items(), key=lambda kv: labels.get(kv[0], "")):
        mark = "yes" if w.coverage == "contested" else "-"
        print(f"{labels.get(node_id, node_id)[:33]:<34} {mark:>10}  {parents.get(node_id, '-')}")


if __name__ == "__main__":
    main()
