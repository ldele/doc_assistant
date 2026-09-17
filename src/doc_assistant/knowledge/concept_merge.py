"""Apply and undo concept merges without losing curation (ROADMAP 53).

A merge folds a near-duplicate concept (``drop``) into the one that stays (``keep``) and deletes
the folded row. Deleting a ``Concept`` cascades into every table keyed on it, and until 2026-09-17
the merge did nothing else first: the dropped concept's taxonomy placements (``concept_hierarchy``,
``ON DELETE CASCADE``) vanished, its gap triage (no foreign key, by design) was orphaned, and no
record said a merge had happened. That contradicts ADR-028 (curated survives) and ADR-018 (demote,
don't delete) in the one curation path that deletes.

Each merge now, in order:

1. **Refuses** a plan whose ids are not both text-bearing concepts (the taxonomy kind guard), and a
   plan whose placements could not move without closing a cycle — checked before anything is
   written, so a skipped merge leaves both concepts exactly as they were.
2. **Moves** what the user curated to the survivor: the surface forms (the keyword family), the
   definition and the graph membership when the survivor has none, every placement — re-pointed
   through ``taxonomy.add_hierarchy_edge``, the one sanctioned writer — the gap triage (the
   survivor's own verdict wins a clash) and the stochastic gap suggestions, whose status persists.
3. **Deletes** the dropped row; derived rows (presence, edges, deterministic gaps) go with it and
   come back from the next skeleton and gap rebuild.
4. **Records** a ``ConceptMerge`` row with everything :func:`undo_merge` needs to split them again.

Zero LLM, zero network. The pure planning (which pairs, which survivor) stays in
``concept_curation``; this module is the write seam.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import networkx as nx
import structlog
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import (
    Concept,
    ConceptAlias,
    ConceptHierarchy,
    ConceptMerge,
    GapRow,
    GapTriage,
)
from doc_assistant.knowledge.concept_curation import MergePlan
from doc_assistant.knowledge.taxonomy import (
    add_hierarchy_edge,
    presence_node,
    remove_hierarchy_edge,
)

log = structlog.get_logger(__name__)


class MergeUndoError(ValueError):
    """An undo that cannot run as recorded; the message is a sentence for the user."""


@dataclass(frozen=True)
class MergeOutcome:
    """What :func:`apply_merges` did: the records written and the plans it refused, with why."""

    merged: tuple[str, ...] = ()  # ConceptMerge ids, in plan order
    skipped: tuple[tuple[MergePlan, str], ...] = ()

    @property
    def n_merged(self) -> int:
        return len(self.merged)


@dataclass(frozen=True)
class MergeSummary:
    """One merge record, for a listing."""

    id: str
    keep_label: str
    drop_label: str
    merged_at: datetime
    undone_at: datetime | None
    n_placements: int
    n_aliases_added: int


def _repoint(
    session: Session, keep_id: str, drop_id: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Plan how each placement touching ``drop`` moves to ``keep`` — without writing anything.

    Returns the per-edge record and, when the moved placements would make the hierarchy cyclic,
    the reason to skip the merge. An edge *between* the two concepts has nothing left to say once
    they are one, so it is dropped rather than turned into a self-edge.
    """
    rows = session.execute(
        select(ConceptHierarchy).where(
            or_(ConceptHierarchy.source_id == drop_id, ConceptHierarchy.target_id == drop_id)
        )
    ).scalars()
    placements: list[dict[str, Any]] = []
    for row in rows:
        source = keep_id if row.source_id == drop_id else row.source_id
        target = keep_id if row.target_id == drop_id else row.target_id
        existing = None
        if source != target:
            existing = session.execute(
                select(ConceptHierarchy.origin).where(
                    ConceptHierarchy.source_id == source,
                    ConceptHierarchy.target_id == target,
                    ConceptHierarchy.type == row.type,
                )
            ).scalar_one_or_none()
        placements.append(
            {
                "source_id": row.source_id,
                "target_id": row.target_id,
                "type": row.type,
                "origin": row.origin,
                "repointed": None if source == target else [source, target],
                "survivor_origin_before": existing,
            }
        )

    graph: nx.DiGraph = nx.DiGraph()
    for source, target in session.execute(
        select(ConceptHierarchy.source_id, ConceptHierarchy.target_id)
    ).all():
        if drop_id not in (source, target):
            graph.add_edge(source, target)
    for p in placements:
        if p["repointed"]:
            graph.add_edge(*p["repointed"])
    if not nx.is_directed_acyclic_graph(graph):
        return placements, "moving its taxonomy placements would close a cycle"
    return placements, None


def _merge_one(session: Session, plan: MergePlan) -> tuple[ConceptMerge | None, str | None]:
    """Fold ``plan.drop_id`` into ``plan.keep_id``; returns the record, or why it was skipped."""
    if plan.keep_id == plan.drop_id:
        return None, "a concept cannot merge into itself"
    keep = presence_node(session, plan.keep_id)
    drop = presence_node(session, plan.drop_id)
    if keep is None or drop is None:
        return None, "not both concepts (one is missing, or is a taxonomy field)"

    placements, refused = _repoint(session, keep.id, drop.id)
    if refused is not None:
        return None, refused

    drop_aliases = sorted({a.alias for a in drop.aliases})
    dropped = {
        "label": drop.label,
        "definition": drop.definition,
        "source": drop.source,
        "graph_include": drop.graph_include,
        "folder_id": drop.folder_id,
        "aliases": drop_aliases,
    }

    # Surface forms: the survivor gains the dropped label and aliases it lacks (the family).
    have = {a.alias for a in keep.aliases} | {keep.label}
    aliases_added = sorted((set(drop_aliases) | {drop.label}) - have)
    for alias in aliases_added:
        keep.aliases.append(ConceptAlias(alias=alias))
    definition_taken = keep.definition is None and drop.definition is not None
    if definition_taken:
        keep.definition = drop.definition
    graph_include_taken = bool(drop.graph_include) and not keep.graph_include
    if graph_include_taken:
        keep.graph_include = True

    # Triage: a verdict follows the concept; the survivor's own verdict on the same kind wins.
    triage: list[dict[str, Any]] = []
    for row in session.execute(select(GapTriage).where(GapTriage.concept_id == drop.id)).scalars():
        clash = session.get(GapTriage, (keep.id, row.kind))
        triage.append({"kind": row.kind, "status": row.status, "moved": clash is None})
        if clash is None:
            session.add(GapTriage(concept_id=keep.id, kind=row.kind, status=row.status))
        session.delete(row)

    # Stochastic suggestions keep their status across a rebuild, so they move too; deterministic
    # gap rows are derived and come back from the next `build_gaps`.
    suggestion_ids = [
        str(row.id)
        for row in session.execute(
            select(GapRow).where(GapRow.concept_id == drop.id, GapRow.determinism == "stochastic")
        ).scalars()
    ]
    for gap_id in suggestion_ids:
        gap = session.get(GapRow, gap_id)
        if gap is not None:
            gap.concept_id = keep.id

    for edge in session.execute(
        select(ConceptHierarchy).where(
            or_(ConceptHierarchy.source_id == drop.id, ConceptHierarchy.target_id == drop.id)
        )
    ).scalars():
        session.delete(edge)
    session.flush()
    for p in placements:
        if p["repointed"]:
            source, target = p["repointed"]
            add_hierarchy_edge(session, source, target, p["type"], origin=p["origin"])

    record = ConceptMerge(
        keep_id=keep.id,
        keep_label=keep.label,
        drop_id=drop.id,
        drop_label=drop.label,
        record_json=json.dumps(
            {
                "dropped": dropped,
                "aliases_added": aliases_added,
                "definition_taken": definition_taken,
                "graph_include_taken": graph_include_taken,
                "placements": placements,
                "triage": triage,
                "suggestion_gap_ids": suggestion_ids,
            }
        ),
    )
    session.add(record)
    session.delete(drop)
    session.flush()
    log.info(
        "concept_merged",
        merge_id=record.id,
        keep_id=keep.id,
        drop_id=record.drop_id,
        n_aliases_added=len(aliases_added),
        n_placements=len(placements),
        n_triage=len(triage),
    )
    return record, None


def apply_merges(plans: list[MergePlan]) -> MergeOutcome:
    """Apply each merge in order, keeping curation and recording it; refused plans are reported.

    One transaction for the batch. A refused plan (not two concepts, or a placement cycle) is
    decided before its first write, so it leaves no partial merge behind.
    """
    if not plans:
        return MergeOutcome()
    from doc_assistant.db.session import session_scope

    merged: list[str] = []
    skipped: list[tuple[MergePlan, str]] = []
    with session_scope() as session:
        for plan in plans:
            record, reason = _merge_one(session, plan)
            if record is None:
                skipped.append((plan, reason or "refused"))
                log.info(
                    "concept_merge_skipped",
                    keep_id=plan.keep_id,
                    drop_id=plan.drop_id,
                    reason=reason,
                )
            else:
                merged.append(record.id)
    return MergeOutcome(merged=tuple(merged), skipped=tuple(skipped))


def list_merges() -> list[MergeSummary]:
    """Every recorded merge, newest first."""
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(select(ConceptMerge)).scalars().all()
        out = []
        for row in rows:
            data = json.loads(row.record_json)
            out.append(
                MergeSummary(
                    id=row.id,
                    keep_label=row.keep_label,
                    drop_label=row.drop_label,
                    merged_at=row.merged_at,
                    undone_at=row.undone_at,
                    n_placements=len(data.get("placements", [])),
                    n_aliases_added=len(data.get("aliases_added", [])),
                )
            )
    out.sort(key=lambda m: m.merged_at, reverse=True)
    return out


def undo_merge(merge_id: str) -> MergeSummary:
    """Split a recorded merge back into two concepts, as they were before it.

    Recreates the dropped concept under its own id with its label, definition, aliases and graph
    membership; takes back the surface forms, definition and graph membership the survivor gained;
    moves each placement, triage verdict and suggestion back. Anything the user changed on the
    survivor since the merge is left alone. Rebuild the skeleton afterwards, as after the merge.

    Raises:
        MergeUndoError: no such merge, already undone, the dropped id exists again, the survivor
            is gone, or restoring a placement would close a cycle added since.
    """
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import TaxonomyCycleError

    with session_scope() as session:
        record = session.get(ConceptMerge, merge_id)
        if record is None:
            raise MergeUndoError(f"No merge is recorded with id {merge_id}.")
        if record.undone_at is not None:
            raise MergeUndoError(f"The merge of {record.drop_label!r} was already undone.")
        if session.get(Concept, record.drop_id) is not None:
            raise MergeUndoError(
                f"A concept with the id of {record.drop_label!r} exists again, so it cannot be "
                "restored."
            )
        keep = presence_node(session, record.keep_id)
        if keep is None:
            raise MergeUndoError(
                f"{record.keep_label!r} no longer exists, so there is nothing to split."
            )
        data = json.loads(record.record_json)
        dropped = data["dropped"]

        drop = Concept(
            id=record.drop_id,
            label=dropped["label"],
            definition=dropped["definition"],
            source=dropped["source"],
            kind="concept",
            graph_include=dropped["graph_include"],
            folder_id=dropped["folder_id"],
        )
        drop.aliases = [ConceptAlias(alias=a) for a in dropped["aliases"]]
        session.add(drop)

        taken = set(data["aliases_added"])
        for alias in [a for a in keep.aliases if a.alias in taken]:
            keep.aliases.remove(alias)
        if data["definition_taken"] and keep.definition == dropped["definition"]:
            keep.definition = None
        if data["graph_include_taken"]:
            keep.graph_include = False
        session.flush()

        try:
            for p in data["placements"]:
                if not p["repointed"]:
                    continue
                if p["survivor_origin_before"] is None:
                    source, target = p["repointed"]
                    remove_hierarchy_edge(session, source, target, p["type"])
                else:
                    row = session.execute(
                        select(ConceptHierarchy).where(
                            ConceptHierarchy.source_id == p["repointed"][0],
                            ConceptHierarchy.target_id == p["repointed"][1],
                            ConceptHierarchy.type == p["type"],
                        )
                    ).scalar_one_or_none()
                    if row is not None:
                        row.origin = p["survivor_origin_before"]
            session.flush()
            for p in data["placements"]:
                add_hierarchy_edge(
                    session, p["source_id"], p["target_id"], p["type"], origin=p["origin"]
                )
        except TaxonomyCycleError as e:
            raise MergeUndoError(
                f"Restoring the placements of {record.drop_label!r} would close a cycle added "
                "since the merge."
            ) from e

        for t in data["triage"]:
            if t["moved"]:
                moved = session.get(GapTriage, (keep.id, t["kind"]))
                if moved is not None and moved.status == t["status"]:
                    session.delete(moved)
            session.add(GapTriage(concept_id=drop.id, kind=t["kind"], status=t["status"]))
        for gap_id in data["suggestion_gap_ids"]:
            gap = session.get(GapRow, gap_id)
            if gap is not None and gap.concept_id == keep.id:
                gap.concept_id = drop.id

        record.undone_at = datetime.now(timezone.utc)
        session.flush()
        log.info("concept_merge_undone", merge_id=record.id, drop_id=record.drop_id)
        return MergeSummary(
            id=record.id,
            keep_label=record.keep_label,
            drop_label=record.drop_label,
            merged_at=record.merged_at,
            undone_at=record.undone_at,
            n_placements=len(data["placements"]),
            n_aliases_added=len(data["aliases_added"]),
        )
