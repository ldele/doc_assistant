"""The text check, once per document after extraction (A2b, docs/security.md S5).

A2 marks a retrieved passage that addresses an AI at answer time (`source_checks`). This module
runs the same check over each document's whole extracted text once, when it is indexed or when
*Re-run… > Text checks* asks, and keeps the result in `DocumentTextCheck`. So the Library can say
which documents carry text written to an AI before anyone asks a question.

Enrichment-Layer Pattern: reads the cached markdown, writes only `document_text_checks`, never
the chunk store and never the text (ADR-043: a mark informs, nothing is removed). Deterministic, no
model. Idempotent: checking a document again replaces its row.

**Current or stale.** A row records the text it read (`doc_hash`) and the patterns it used
(`source_checks.CHECKS_VERSION`). When either has moved since, the row is *stale*: the reads report
it as such and never as a current finding, and the Summary offers the re-run that refreshes it.

**Scale.** Batches of `_BATCH` documents: one read of their hashes, one read of their texts, one
write. A whole-library re-check (~10,000 documents under the robustness contract) never holds more
than one batch of texts in memory. The reads for the Summary are one query over the table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_assistant.source_checks import CHECKS_VERSION, Finding, find_in_document

log = structlog.get_logger(__name__)

#: How many findings a row keeps for the document page. A paper about prompt injection can quote
#: dozens of attacks; the page shows the first ones and says how many there are in all. Structural
#: (what a page can usefully list), not tuned to a corpus.
KEPT_FINDINGS = 20

#: Documents per read-and-write. Structural, as in `metadata_enrich._FILL_BATCH`.
_BATCH = 100

CheckState = Literal["current", "stale"]
#: What a document page reports: a check's state, or that the document was never checked.
ViewState = Literal["current", "stale", "not_checked"]


@dataclass(frozen=True)
class CheckRun:
    """What one call did: documents checked, of which marked, and those with no cached text."""

    checked: int = 0
    marked: int = 0
    no_text: int = 0


@dataclass(frozen=True)
class DocumentCheck:
    """One document's check as the document page reports it."""

    state: ViewState
    checked_at: datetime | None
    n_marked: int
    findings: list[Finding] = field(default_factory=list)


def _encode(findings: list[Finding]) -> str:
    return json.dumps(
        [
            {"page": f.page, "labels": list(f.labels), "excerpt": f.excerpt}
            for f in findings[:KEPT_FINDINGS]
        ],
        ensure_ascii=False,
    )


def _decode(raw: str) -> list[Finding]:
    try:
        items = json.loads(raw or "[]")
    except ValueError:
        return []
    out: list[Finding] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            page = item.get("page")
            out.append(
                Finding(
                    page=page if isinstance(page, int) else None,
                    labels=tuple(str(x) for x in item.get("labels") or []),
                    excerpt=str(item.get("excerpt") or ""),
                )
            )
    return out


def check_documents(document_ids: Iterable[str]) -> CheckRun:
    """Check these documents' extracted text and keep what was found, replacing earlier rows.

    An archived document, or one whose cached text cannot be found, is skipped and counted under
    ``no_text``; it keeps whatever row it had, which the reads then report by its state."""
    from doc_assistant.db.models import Document, DocumentTextCheck
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.keywords import load_document_texts

    ids = sorted(set(document_ids))
    checked = marked = no_text = 0
    for start in range(0, len(ids), _BATCH):
        batch = ids[start : start + _BATCH]
        with session_scope() as session:
            hashes = {
                str(doc_id): str(doc_hash)
                for doc_id, doc_hash in session.execute(
                    select(Document.id, Document.doc_hash).where(Document.id.in_(batch))
                ).all()
            }
        texts = load_document_texts(batch)
        no_text += len(hashes) - len(texts)
        results = [(doc_id, find_in_document(text)) for doc_id, _name, text in texts]
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            for doc_id, findings in results:
                if doc_id not in hashes:  # deleted between the two reads
                    continue
                row = session.get(DocumentTextCheck, doc_id)
                if row is None:
                    row = DocumentTextCheck(document_id=doc_id)
                    session.add(row)
                row.doc_hash = hashes[doc_id]
                row.checks_version = CHECKS_VERSION
                row.n_marked = len(findings)
                row.findings_json = _encode(findings)
                row.checked_at = now
                checked += 1
                marked += 1 if findings else 0
    if checked or no_text:
        log.info("document_text_checked", checked=checked, marked=marked, no_text=no_text)
    return CheckRun(checked=checked, marked=marked, no_text=no_text)


def _state(row_hash: str, row_version: str, doc_hash: str | None) -> CheckState:
    return "current" if row_hash == doc_hash and row_version == CHECKS_VERSION else "stale"


def check_states(session: Session) -> dict[str, tuple[CheckState, int]]:
    """``document id -> (state, paragraphs marked)`` for every checked document, in one query.

    A document absent from the result has never been checked."""
    from doc_assistant.db.models import Document, DocumentTextCheck

    rows = session.execute(
        select(
            DocumentTextCheck.document_id,
            DocumentTextCheck.doc_hash,
            DocumentTextCheck.checks_version,
            DocumentTextCheck.n_marked,
            Document.doc_hash,
        ).join(Document, Document.id == DocumentTextCheck.document_id)
    ).all()
    return {
        str(doc_id): (_state(row_hash, version, doc_hash), int(n))
        for doc_id, row_hash, version, n, doc_hash in rows
    }


def document_check(document_id: str) -> DocumentCheck | None:
    """One document's check with its findings; ``None`` for a document that does not exist.

    A document never checked reports ``not_checked`` with no findings."""
    from doc_assistant.db.models import Document, DocumentTextCheck
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        row = session.get(DocumentTextCheck, document_id)
        if row is None:
            return DocumentCheck(state="not_checked", checked_at=None, n_marked=0)
        return DocumentCheck(
            state=_state(row.doc_hash, row.checks_version, doc.doc_hash),
            checked_at=row.checked_at,
            n_marked=int(row.n_marked),
            findings=_decode(row.findings_json),
        )
