"""What a turn shows besides its answer: its claims, what it looked up, what it carried (A8b).

The contract is `docs/specs/chat-interface-pass.md` § "The contract — A8b". Every view here is read
from what the answer path already records; nothing new is stored:

- :class:`TurnClaim` — one sentence of an answer, as `answer_claims` holds it: its citations, its
  retrieval marker (`ok`, `weak`, `unsupported`) and the decision a person recorded on it. The
  window colours a claim by its marker only (G4): the marker measures whether a sentence cites a
  passage and how that passage ranked, never whether the sentence is true.
- :class:`RecordsView` — a rule-answered turn's lookup (A6, `answer_records.records_json`): its
  name, arguments, row count and its first rows named, each a document, folder or concept.
- :class:`ReviewView` — the reviewer's verdict as recorded (`answer_reviews`), when there is one.

What the answer carried (A5's counts, A8a's carried exchanges) is `context_json` itself, which the
wire passes as it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from sqlalchemy import select

from doc_assistant.db.models import AnswerClaim, AnswerReview, Concept, Document, Folder
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)

#: How many of a lookup's rows the records card names before "N more". A reading bound for a
#: card in a chat turn, not tuned to a corpus.
RECORD_ROWS_SHOWN = 5


@dataclass(frozen=True)
class TurnClaim:
    """One claim of an answer. ``sources`` are the passage numbers it cites, in order; ``n`` is
    its 1-based place in the answer; ``decision`` is ``pending`` until a person decides."""

    claim_id: str
    n: int
    text: str
    marker: str
    sources: list[int] = field(default_factory=list)
    decision: str = "pending"
    edited_text: str | None = None


def _sources_of(citations_json: str | None) -> list[int]:
    try:
        items = json.loads(citations_json or "[]")
    except ValueError:
        return []
    out: list[int] = []
    for item in items if isinstance(items, list) else []:
        n = item.get("source_number") if isinstance(item, dict) else None
        if isinstance(n, int) and not isinstance(n, bool) and n not in out:
            out.append(n)
    return out


def claim_of(row: AnswerClaim) -> TurnClaim:
    return TurnClaim(
        claim_id=str(row.id),
        n=row.claim_index + 1,
        text=row.claim_text,
        marker=row.marker,
        sources=_sources_of(row.citations_json),
        decision=row.decision,
        edited_text=row.edited_text,
    )


def claims_for(record_ids: list[str]) -> dict[str, list[TurnClaim]]:
    """Each answer's claims in order, with their recorded decisions; one query for them all."""
    ids = [i for i in dict.fromkeys(record_ids) if i]
    if not ids:
        return {}
    out: dict[str, list[TurnClaim]] = {i: [] for i in ids}
    with session_scope() as session:
        rows = session.execute(
            select(AnswerClaim)
            .where(AnswerClaim.answer_record_id.in_(ids))
            .order_by(AnswerClaim.answer_record_id, AnswerClaim.claim_index)
        ).scalars()
        for row in rows:
            out[str(row.answer_record_id)].append(claim_of(row))
    return out


@dataclass(frozen=True)
class ReviewView:
    """The reviewer's verdict on an answer, as recorded (the newest review when there are
    several). Scores are 1 to 5; ``None`` where the reviewer gave none."""

    reviewer_kind: str
    faithfulness: int | None = None
    citation_density: int | None = None
    hedging_adequacy: int | None = None
    unsupported_claims: int | None = None
    failure_tag: str | None = None
    error: str | None = None


def reviews_for(record_ids: list[str]) -> dict[str, ReviewView]:
    """The newest recorded review per answer, for the answers that have one; one query."""
    ids = [i for i in dict.fromkeys(record_ids) if i]
    if not ids:
        return {}
    out: dict[str, ReviewView] = {}
    with session_scope() as session:
        rows = session.execute(
            select(AnswerReview)
            .where(AnswerReview.answer_record_id.in_(ids))
            .order_by(AnswerReview.created_at.desc())
        ).scalars()
        for row in rows:
            rid = str(row.answer_record_id)
            if rid in out:
                continue
            out[rid] = ReviewView(
                reviewer_kind=row.reviewer_kind,
                faithfulness=row.faithfulness,
                citation_density=row.citation_density,
                hedging_adequacy=row.hedging_adequacy,
                unsupported_claims=row.unsupported_claims_count,
                failure_tag=row.failure_tag,
                error=row.error,
            )
    return out


@dataclass(frozen=True)
class RecordItem:
    """One row a lookup named: a ``document``, ``folder`` or ``concept``, by its id and name."""

    kind: Literal["document", "folder", "concept"]
    id: str
    label: str


@dataclass(frozen=True)
class RecordsView:
    """A rule-answered turn's lookup (A6), ready for *What it looked up*. ``more`` is how many
    of the named rows are not shown; ``via`` is ``rule`` or ``command``."""

    lookup: str
    args: dict[str, Any] = field(default_factory=dict)
    rows: int = 0
    items: list[RecordItem] = field(default_factory=list)
    more: int = 0
    via: str = "rule"
    asked: bool = False


def _labels(ids: list[str]) -> dict[str, RecordItem]:
    """Each id named: a document by its title (else its file name), a folder or a concept by its
    name. An id no table holds (deleted since) is left out. Three queries."""
    if not ids:
        return {}
    out: dict[str, RecordItem] = {}
    with session_scope() as session:
        for did, title, filename in session.execute(
            select(Document.id, Document.title, Document.filename).where(Document.id.in_(ids))
        ).all():
            out[str(did)] = RecordItem("document", str(did), str(title or filename))
        for fid, name in session.execute(
            select(Folder.id, Folder.name).where(Folder.id.in_(ids))
        ).all():
            out[str(fid)] = RecordItem("folder", str(fid), str(name))
        for cid, label in session.execute(
            select(Concept.id, Concept.label).where(Concept.id.in_(ids))
        ).all():
            out[str(cid)] = RecordItem("concept", str(cid), str(label))
    return out


def describe_records(
    raw: str | dict[str, Any] | None, shown: int = RECORD_ROWS_SHOWN
) -> RecordsView | None:
    """A records answer's lookup from its ``records_json`` (text or parsed), or ``None`` for a
    turn that was not answered from the records. Text that cannot be read gives ``None`` (the
    card is then not shown) and is logged."""
    if raw is None:
        return None
    value: Any = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            log.warning("records_json_unreadable")
            return None
    if not isinstance(value, dict) or not isinstance(value.get("lookup"), str):
        return None
    ids = [str(i) for i in value.get("ids") or [] if isinstance(i, str | int)]
    head = ids[:shown]
    try:
        named = _labels(head)
    except Exception as e:  # the card is a sidecar: a read failure shows no rows, never breaks
        log.warning("records_labels_failed", error=str(e))
        named = {}
    rows = value.get("rows")
    args = value.get("args")
    return RecordsView(
        lookup=value["lookup"],
        args=args if isinstance(args, dict) else {},
        rows=rows if isinstance(rows, int) and not isinstance(rows, bool) else len(ids),
        items=[named[i] for i in head if i in named],
        more=max(0, len(ids) - len(head)),
        via=str(value.get("via") or "rule"),
        asked=bool(value.get("asked")),
    )


def read_context(raw: str | None) -> dict[str, Any] | None:
    """A turn's ``context_json`` as a dict (A5's counts, A8a's carried exchanges), or ``None``
    for a turn that recorded none."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
