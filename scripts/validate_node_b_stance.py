"""Validate the Node-B stance extractor that `contested` is built on (ADR-040 option 5, KI-33).

`contested` is the product thesis, and every one of its inputs is a Node-B stance annotation.
This characterises those annotations three ways, in increasing cost:

  (default)     read the shipped skeleton artifact — polarity mix, per-edge stance instability,
                and relation-verb x stance coherence. Free, no LLM, no writes.
  --replay      rebuild each document's ACTUAL Node-B prompt from `concept_presence` and re-run
                it, to check the recorded stances reproduce (instrument fidelity).
  --positions   hold document, present-concepts and pair list fixed and vary ONLY where the
                target pair sits in the numbered list. If the stance moves, it encodes generation
                position rather than anything about the document.

**Deliberately Ollama-only and $0.** The point is to characterise the *configured local Node-B
model* (`CONCEPT_SKELETON_LLM_MODEL`), so there is no `--provider` flag to leak paid credits
through (KI-4). Read-only throughout: nothing here writes the skeleton or the DB.

Usage::

    uv run --no-sync python -m scripts.validate_node_b_stance
    uv run --no-sync python -m scripts.validate_node_b_stance --replay --positions
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from doc_assistant.config import (
    CONCEPT_SKELETON_DIR,
    CONCEPT_SKELETON_LLM_MODEL,
    SQLITE_URL,
)
from doc_assistant.knowledge.concept_skeleton import (
    OPPOSING_POLARITIES,
    SKELETON_NAME,
    SUPPORTING_POLARITIES,
    ConceptSkeleton,
    skeleton_from_dict,
)
from doc_assistant.knowledge.concept_skeleton_enrich import build_messages, parse_annotations

#: Positions probed by --positions. Spread across a long real list rather than exhaustive:
#: the question is whether the verdict moves at all, not the shape of the curve.
PROBE_POSITIONS = (0, 2, 4, 8, 12)


def _load_skeleton() -> ConceptSkeleton:
    path = CONCEPT_SKELETON_DIR / SKELETON_NAME
    if not path.exists():
        raise FileNotFoundError(f"No concept skeleton at {path} — build it first.")
    return skeleton_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _present_by_doc() -> dict[str, list[str]]:
    """Per-document concept presence — the other half of the real Node-B input."""
    db = SQLITE_URL.replace("sqlite:///", "")
    con = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True)
    out: dict[str, list[str]] = defaultdict(list)
    try:
        for doc_id, concept_id in con.execute(
            "SELECT document_id, concept_id FROM concept_presence"
        ):
            out[doc_id].append(concept_id)
    finally:
        con.close()
    return out


def report_artifact(sk: ConceptSkeleton, labels: dict[str, str]) -> None:
    stances = [pol for e in sk.edges for _, pol in e.stance_by_doc]
    if not stances:
        print("No Node-B stance in this skeleton — run `build_concept_skeleton --apply --enrich`.")
        return
    total = len(stances)
    print("=== polarity mix over every stance assignment in the layer ===")
    for pol, n in Counter(stances).most_common():
        side = "OPPOSING" if pol in OPPOSING_POLARITIES else "supporting"
        print(f"  {pol:<14} {n:>5}  {n / total * 100:>5.1f}%   ({side})")
    opp = sum(1 for p in stances if p in OPPOSING_POLARITIES)
    print(f"  {'TOTAL':<14} {total:>5}")
    print(f"\n  opposing share {opp}/{total} = {opp / total * 100:.1f}%.")
    print("  There is NO neutral stance in the vocabulary; every co-present pair must take one")
    print("  of four labels, two of which count as opposing. Citation-polarity corpora put")
    print("  neutral above 60% and contrasting/negative as the rarest class.")

    print("\n=== per-edge stance instability ===")
    annotated = [e for e in sk.edges if e.stance_by_doc]
    multi = 0
    for e in sorted(annotated, key=lambda e: -len(e.stance_by_doc)):
        pols = [p for _, p in e.stance_by_doc]
        multi += len(set(pols)) > 1
        name = f"{labels.get(e.source_concept_id, '?')} <-> {labels.get(e.target_concept_id, '?')}"
        print(f"  {name[:46]:<48} {len(pols):>3} docs  {dict(Counter(pols))}  rel={e.relation!r}")
    print(
        f"\n  {multi}/{len(annotated)} annotated edges carry MORE THAN ONE stance across documents"
    )

    print("\n=== relation verb x stance ===")
    by_rel: dict[str, Counter[str]] = defaultdict(Counter)
    for e in sk.edges:
        for _, pol in e.stance_by_doc:
            by_rel[e.relation or "(none)"][pol] += 1
    for rel, c in sorted(by_rel.items(), key=lambda kv: -sum(kv[1].values())):
        sup = sum(v for k, v in c.items() if k in SUPPORTING_POLARITIES)
        opp_n = sum(v for k, v in c.items() if k in OPPOSING_POLARITIES)
        flag = "  <-- crosses the boundary" if sup and opp_n else ""
        print(f"  {rel[:32]:<34} sup={sup:>3} opp={opp_n:>3}  {dict(c)}{flag}")


def _most_contested_edge(sk: ConceptSkeleton) -> object:
    """The edge whose stance is least stable — the sharpest probe target."""
    return max(
        (e for e in sk.edges if e.stance_by_doc),
        key=lambda e: (len({p for _, p in e.stance_by_doc}), len(e.stance_by_doc)),
    )


def replay(sk: ConceptSkeleton, labels: dict[str, str]) -> None:
    from doc_assistant.llm import OllamaClient

    edge = _most_contested_edge(sk)
    pair = (edge.source_concept_id, edge.target_concept_id)  # type: ignore[attr-defined]
    recorded = dict(edge.stance_by_doc)  # type: ignore[attr-defined]
    edge_pairs = {(e.source_concept_id, e.target_concept_id) for e in sk.edges}
    present_by_doc = _present_by_doc()
    client = OllamaClient(CONCEPT_SKELETON_LLM_MODEL)

    print(
        f"\n=== replay: {labels[pair[0]]} <-> {labels[pair[1]]} ({CONCEPT_SKELETON_LLM_MODEL}) ==="
    )
    print(f"recorded: {dict(Counter(recorded.values()))}")
    print(
        f"\n{'document':<14} {'#present':>8} {'#pairs':>7} {'idx':>4} "
        f"{'recorded':<12} {'replayed':<12}"
    )
    agree = 0
    for doc_id in sorted(recorded):
        present = sorted({c for c in present_by_doc.get(doc_id, []) if c in labels})
        candidates = [(a, b) for a, b in combinations(present, 2) if (a, b) in edge_pairs]
        if pair not in candidates:
            continue
        idx = candidates.index(pair)
        messages = build_messages(
            [labels[c] for c in present], [(labels[a], labels[b]) for a, b in candidates]
        )
        ann = parse_annotations(
            client.complete(messages, temperature=0.0, max_tokens=2048), len(candidates)
        )
        got = next((s for i, _, s in ann if i == idx), "(none)")
        agree += got == recorded[doc_id]
        print(
            f"{doc_id[:13]:<14} {len(present):>8} {len(candidates):>7} {idx:>4} "
            f"{recorded[doc_id]:<12} {got:<12}"
        )
    print(f"\n  reproduced {agree}/{len(recorded)} recorded stances (instrument fidelity)")


def positions(sk: ConceptSkeleton, labels: dict[str, str]) -> None:
    from doc_assistant.llm import OllamaClient

    edge = _most_contested_edge(sk)
    pair = (edge.source_concept_id, edge.target_concept_id)  # type: ignore[attr-defined]
    edge_pairs = {(e.source_concept_id, e.target_concept_id) for e in sk.edges}
    present_by_doc = _present_by_doc()

    # the document with the longest pair list carrying this edge — the most room to vary position
    best: tuple[int, list[str], list[tuple[str, str]]] | None = None
    for doc_id, _ in edge.stance_by_doc:  # type: ignore[attr-defined]
        present = sorted({c for c in present_by_doc.get(doc_id, []) if c in labels})
        candidates = [(a, b) for a, b in combinations(present, 2) if (a, b) in edge_pairs]
        if pair in candidates and (best is None or len(candidates) > best[0]):
            best = (len(candidates), present, candidates)
    if best is None:
        print(
            "\n(no document carries this pair with a reconstructable list — skipping --positions)"
        )
        return
    _, present, candidates = best

    labelled = [(labels[a], labels[b]) for a, b in candidates]
    home = candidates.index(pair)
    target, rest = labelled[home], [p for i, p in enumerate(labelled) if i != home]
    client = OllamaClient(CONCEPT_SKELETON_LLM_MODEL)

    print(f"\n=== position probe: {labels[pair[0]]} <-> {labels[pair[1]]} ===")
    print(
        f"one document: {len(present)} concepts, {len(candidates)} pairs; "
        "only the pair's index moves"
    )
    print(f"\n{'index':>6}  {'relation':<26} stance")
    seen = []
    for pos in [*[p for p in PROBE_POSITIONS if p <= len(rest)], len(rest)]:
        ordered = [*rest[:pos], target, *rest[pos:]]
        ann = parse_annotations(
            client.complete(
                build_messages([labels[c] for c in present], ordered),
                temperature=0.0,
                max_tokens=2048,
            ),
            len(ordered),
        )
        got = next((s for i, _, s in ann if i == pos), "(none)")
        rel = next((r for i, r, _ in ann if i == pos), "-")
        seen.append(got)
        print(f"{pos:>6}  {rel[:25]:<26} {got}")
    crossed = {p in OPPOSING_POLARITIES for p in seen}
    print(f"\n  {len(set(seen))} distinct verdicts from position alone: {dict(Counter(seen))}")
    if len(crossed) > 1:
        print("  ** and they CROSS the supporting/opposing boundary — the input to `contested`")
        print("     partly encodes where a pair landed in a generated list. **")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--replay", action="store_true", help="re-run the real prompts (local LLM, $0)"
    )
    ap.add_argument(
        "--positions", action="store_true", help="vary only list position (local LLM, $0)"
    )
    args = ap.parse_args()

    sk = _load_skeleton()
    labels = {n.id: n.label for n in sk.nodes}
    report_artifact(sk, labels)
    if args.replay:
        replay(sk, labels)
    if args.positions:
        positions(sk, labels)


if __name__ == "__main__":
    main()
