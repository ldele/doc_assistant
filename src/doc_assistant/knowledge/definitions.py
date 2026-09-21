"""A concept's definition, chosen from candidates that keep their source (ADR-053, ROADMAP 93).

ADR-052 made a definition the thing that says what a concept means; ADR-053 says where one comes
from and what may replace what. The answer to the second is **nothing replaces anything**: a
sentence found in the library, a model's text with the inputs it was handed, and the user's own
words are rows side by side in ``concept_definitions``, and the user chooses one. Choosing is the
only write to ``Concept.definition``, it is recorded, and it is undone by putting the previous
choice back. This module is the only writer of either — keep it that way, or the mirror drifts.

**Passages.** A candidate from the library is a sentence copied **verbatim** — a substring of the
chunk it came from, whitespace and markup included (ADR-043; the UI collapses whitespace when it
renders). Three forms are looked for, and the form is part of the evidence:

- ``coined`` — the author names the term ("we call our model SPECTER", "we denote as 'hard
  negatives'"): the primary source of the word;
- ``named`` — the sentence describes something, then gives the term as its name ("… is called a
  'cross-encoder'", "… commonly referred to as passage retrieval");
- ``defining`` — the sentence is shaped as a definition ("knowledge distillation refers to…",
  "DBS is a neurosurgical therapy…");
- ``first_mention`` — the first sentence that uses the term in a document that uses it a lot, which
  is where a survey introduces it even when no pattern fires.

About half of the definition shapes are not definitions (measured on the 19 priority concepts,
``tests/eval/baselines/definition_sources_2026-09-21.md``), which is why a candidate carries its
reasons and a coarse grade rather than a verdict, and why nothing here chooses for the user.

**Reliability is evidence, not a score** — the rule the rest of this layer follows
(``docs/knowledge-layer.md`` §6). The grade is derived from the form and from where the sentence
sits, both shown as reasons; no model rates its own output here.

Zero LLM, zero network. The runner is ``scripts/extract_definitions.py`` (dry-run default); the app
reaches the same functions through ``/api/concepts/{id}/definitions``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import (
    Concept,
    ConceptDefinition,
    ConceptDefinitionEvent,
    Document,
)
from doc_assistant.knowledge.concept_skeleton import compile_boundary_pattern

log = structlog.get_logger(__name__)

SOURCES: frozenset[str] = frozenset({"passage", "user", "model"})
STATUSES: frozenset[str] = frozenset({"suggested", "chosen", "dismissed"})

#: How many passage candidates one concept keeps from one extraction. A readability bound on a
#: review list (a person reads these), not a corpus-tuned threshold.
PASSAGES_PER_CONCEPT = 5
#: A first mention is looked for only in the documents that use the concept most — where an
#: introduction is likely — and this many of them. Structural, like the bound above.
FIRST_MENTION_DOCS = 3
#: Sentence length bounds for a candidate: shorter is a heading or a fragment, longer is a
#: paragraph the splitter failed to cut. Readability bounds, not tuned on this corpus.
MIN_SENTENCE, MAX_SENTENCE = 25, 500

# A sentence ends at . ! or ? followed by whitespace and something that can start one. Run on the
# original text, so every span it yields is a substring of the chunk.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z#(\[*_“\"])")
# A full stop that belongs to an abbreviation, not to the end of a sentence (searched on the text
# up to a candidate boundary, so ``$`` is that boundary).
_ABBREVIATION_END = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs|\bfigs?|\beqs?|\bsec|\brefs?|\bno|\bapprox|\bresp"
    r"|\bdr|\bprof|\bst)\.$",
    re.IGNORECASE,
)
# A sentence that was allowed to finish: terminal punctuation, then at most closing quotes,
# brackets or emphasis markup.
_CLOSED = re.compile("[.!?][\"'\u201d\u2019)\\]*_]*$")
# A markdown heading, or a bold run-in label ("**Takeaway Lessons.**"), glued to the start of a
# sentence. Cut off the front so the candidate starts where the prose does — still a substring.
_LEADING_LABEL = re.compile(
    r"^(?:#{1,6}[^\n]*\n\s*"  # a heading line
    r"|#{1,6}\s+\*\*[^*\n]{1,160}\*\*\s+"  # a bold heading glued to the text after it
    r"|\*\*[^*\n]{1,60}\*\*\s+)"  # a bold run-in label
)
# Shapes that mark a bibliography entry or a title block rather than prose.
_REFERENCE_SHAPE = re.compile(
    r"\d+\s*\(\d+\)\s*:\s*\d+"  # volume(issue):pages
    r"|\bIn\s+(?:_|\*|Proceedings|Advances)"  # "In _Proceedings…"
    r"|arxiv|doi\.org|https?://|\bdoi:"
    "|[\u2217\u2020\u2021]",  # affiliation marks (asterisk operator, daggers): a title block
    re.IGNORECASE,
)
# An author-affiliation line from a title page: an institution followed, in the same span, by a
# five-digit postal code — or an e-mail address. A paper whose *title* holds the term
# ("Neuroanatomy goes viral!") otherwise offers its address block as the first mention.
# Five digits, not four: "the Institute showed in 2015" is prose, and a year is four.
_AFFILIATION = re.compile(
    r"@\w"
    r"|\b(?:Department|Dept|Institute|University|Laboratory|Hospital|School|Faculty|Center"
    r"|Centre)\b.{0,160}?\b\d{5}\b"  # "." allowed: an address has "Ave." in it
)
# An optional quote around the term, straight or curly (escaped: ruff's RUF001 reads the curly
# ones as look-alikes, and here they are the point).
_QUOTE_OPEN = "[\u201c\"'\u2018]?"
_QUOTE_CLOSE = "[\u201d\"'\u2019]?"


# ============================================================
# Finding passages — pure
# ============================================================


@dataclass(frozen=True)
class PassageHit:
    """One sentence that may define a concept, and what the evidence about it is."""

    concept_id: str
    text: str  # verbatim: a substring of the chunk's text
    document_id: str
    chunk_key: str
    form: str  # "coined" | "named" | "defining" | "first_mention"
    doc_mentions: int  # sentences in this document that mention the concept
    doc_rank: int  # 1 = the document that mentions it most
    concept_docs: int = 1  # documents that mention the concept at all
    shaped_docs: int = 0  # of those, documents with a coined or definition-shaped sentence
    opens: bool = False  # the sentence starts with the term (after at most an article)


def sentence_spans(text: str) -> list[str]:
    """Prose sentences of one chunk, each a **substring** of ``text``.

    A full stop after an abbreviation ("et al.", "e.g.", "Fig.") does not end a sentence. A chunk
    boundary can cut a sentence, so a first span that starts in lower case (the tail of the
    previous chunk's sentence) and a last span with no closing punctuation are dropped rather than
    offered half-said. A leading heading or bold label is cut off the front; a span that then looks
    like a bibliography entry or a title block, or falls outside the length bounds, is dropped.
    """
    out: list[str] = []
    start = 0
    ends = [
        m.start()
        for m in _SENTENCE_END.finditer(text)
        # Only the few characters before the boundary can hold the abbreviation; searching the
        # whole prefix would make every chunk quadratic in its length.
        if not _ABBREVIATION_END.search(text, max(0, m.start() - 12), m.start())
    ] + [len(text)]
    for n, end in enumerate(ends):
        span = text[start:end].strip()
        start = end
        if n == 0 and span[:1].islower():
            continue  # the tail of a sentence the previous chunk began
        if n == len(ends) - 1 and not _CLOSED.search(span):
            continue  # cut off by the chunk boundary
        label = _LEADING_LABEL.match(span)
        if label:
            span = span[label.end() :].strip()
        if not (MIN_SENTENCE <= len(span) <= MAX_SENTENCE):
            continue
        if "#" in span or span.startswith(("**", "_", "|", "- ", "* ")):
            continue  # a heading or title block the label cut did not remove, a table, a list
        if _REFERENCE_SHAPE.search(span) or _AFFILIATION.search(span):
            continue
        out.append(span)
    return out


def _body_chunks(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One document's ``(chunk_key, text)`` in reading order, with the bibliography cut off.

    A reference list is full of titles that mention a concept ("Efficient … re-ranking for large
    web datasets") and would pass for its first mention. The cut is the keyword layer's own
    (``keywords.strip_reference_section``: a whole-line References heading past the document's
    early part), applied to the joined text and mapped back onto the chunks, so a chunk that
    straddles the heading keeps its prefix — still a substring of the chunk.
    """
    from doc_assistant.knowledge.keywords import strip_reference_section

    full = "\n".join(text for _, text in rows)
    keep = len(strip_reference_section(full))
    out: list[tuple[str, str]] = []
    position = 0
    for chunk_key, text in rows:
        if position >= keep:
            break
        out.append((chunk_key, text[: keep - position]))
        position += len(text) + 1
    return out


def _form_patterns(form: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """``(coined, named, defining)`` patterns for one casefolded surface form."""
    f = re.escape(form)
    b, e = r"(?<![a-z0-9])", r"(?![a-z0-9])"
    term = rf"{_QUOTE_OPEN}{b}{f}{e}{_QUOTE_CLOSE}"
    # Coining is first person: "denoted as BM25 + RM3" names a *notation*, not the term.
    coined = re.compile(
        rf"\bwe\s+(?:call|term|name|denote|dub)\b[^.]{{0,80}}{term}"
        rf"|\bwe\s+(?:refer\s+to|define)\b[^.]{{0,80}}\bas\s+(?:an?\s+|the\s+)?{term}"
    )
    # Naming: the sentence describes something, then gives the term as its name ("... is called
    # a 'cross-encoder'", "... commonly referred to as passage retrieval"). The term comes last by
    # the grammar of it, so it is not held to the opens-with-the-term rule a definition is.
    named = re.compile(
        rf"\b(?:is|are)\s+(?:called|known\s+as|referred\s+to\s+as|termed)\s+"
        rf"(?:an?\s+|the\s+)?{term}"
        rf"|\b(?:commonly|also|often|usually)\s+(?:called|known\s+as|referred\s+to\s+as)\s+"
        rf"{term}"
    )
    defining = re.compile(
        rf"{b}{f}{e}{_QUOTE_CLOSE}\s*(?:\([^)]{{1,20}}\)\s*)?,?\s*"
        rf"(?:is|are)\s+(?:a|an|the|one|defined\s+as)\s"
        rf"|{b}{f}{e}{_QUOTE_CLOSE}\s*(?:\([^)]{{1,20}}\)\s*)?(?:refers?\s+to|denotes)\s"
    )
    return coined, named, defining


def find_passages(
    concepts: Sequence[tuple[str, str]],
    chunks: Iterable[tuple[str, str, str]],
    *,
    per_concept: int = PASSAGES_PER_CONCEPT,
    first_mention_docs: int = FIRST_MENTION_DOCS,
) -> dict[str, list[PassageHit]]:
    """Candidate definition sentences for each ``(id, label)``, from ``(chunk_key, doc_id, text)``.

    Pure. Matches the **label only** — aliases are different phrases and bring their own meanings
    in (``tests/eval/baselines/isa_head_suffix_2026-09-20.md`` measured that for heads; the
    passage scan found the same). Order within a concept: coined, then defining, then first
    mentions from the documents that use the concept most; duplicates of one text are kept once.
    A concept that appears nowhere is absent from the result.
    """
    # Every document's prose sentences, in reading order (the parent index in the chunk key).
    ordered: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for chunk_key, document_id, text in chunks:
        try:
            index = int(chunk_key.rsplit(":p", 1)[1])
        except (IndexError, ValueError):
            index = 0
        ordered[document_id].append((index, chunk_key, text))
    sentences: dict[str, list[tuple[str, str, str]]] = {}  # doc -> [(chunk_key, span, low)]
    for document_id, rows in ordered.items():
        rows.sort()
        spans: list[tuple[str, str, str]] = []
        for chunk_key, text in _body_chunks([(k, t) for _, k, t in rows]):
            spans.extend((chunk_key, span, span.casefold()) for span in sentence_spans(text))
        sentences[document_id] = spans

    out: dict[str, list[PassageHit]] = {}
    for concept_id, label in concepts:
        form = label.casefold().strip()
        if not form:
            continue
        mention = compile_boundary_pattern(form)
        coined_re, named_re, defining_re = _form_patterns(form)
        per_doc: Counter[str] = Counter()
        first: dict[str, tuple[str, str]] = {}
        shaped: list[tuple[str, str, str, str]] = []  # (form, doc, chunk_key, span)
        for document_id, spans in sentences.items():
            for chunk_key, span, low in spans:
                if form not in low or not mention.search(low):
                    continue
                per_doc[document_id] += 1
                first.setdefault(document_id, (chunk_key, span))
                if coined_re.search(low):
                    shaped.append(("coined", document_id, chunk_key, span))
                elif named_re.search(low):
                    shaped.append(("named", document_id, chunk_key, span))
                elif defining_re.search(low):
                    shaped.append(("defining", document_id, chunk_key, span))
        if not per_doc:
            continue
        # Most mentions first; a tie goes to the lower document id, not to whichever document the
        # chunks happened to arrive from first — so the keyword-index path and the full read (and
        # two runs of either) pick the same documents.
        doc_order = sorted(per_doc, key=lambda d: (-per_doc[d], d))
        rank = {d: i + 1 for i, d in enumerate(doc_order)}
        shaped_docs = len({s[1] for s in shaped})
        opens_re = re.compile(rf"(?:(?:the|an?)\s+)?{_QUOTE_OPEN}{re.escape(form)}(?![a-z0-9])")
        picked = list(shaped)
        for document_id in doc_order[:first_mention_docs]:
            chunk_key, span = first[document_id]
            picked.append(("first_mention", document_id, chunk_key, span))
        candidates: dict[str, PassageHit] = {}
        for kind, document_id, chunk_key, span in picked:
            if span in candidates:
                continue  # a first mention that is also a shaped sentence keeps its shape
            candidates[span] = PassageHit(
                concept_id=concept_id,
                text=span,
                document_id=document_id,
                chunk_key=chunk_key,
                form=kind,
                doc_mentions=per_doc[document_id],
                doc_rank=rank[document_id],
                concept_docs=len(per_doc),
                shaped_docs=shaped_docs,
                opens=bool(opens_re.match(span.casefold())),
            )
        # The cap keeps the best: by grade, then by how much the document uses the concept, so a
        # run of sentences that only look like definitions cannot push out a real introduction.
        ranked = sorted(
            candidates.values(),
            key=lambda h: (_GRADE_ORDER[passage_evidence(h)["grade"]], h.doc_rank),
        )
        out[concept_id] = ranked[:per_concept]
    return out


# ============================================================
# Evidence — pure
# ============================================================


def passage_evidence(hit: PassageHit) -> dict[str, Any]:
    """The reasons a passage candidate can be trusted, and the grade they add up to.

    ``strong`` — the author coins the term, or a definition-shaped sentence that *opens with* the
    term, so the sentence is about it. ``some`` — a definition shape where the term is not what the
    sentence is about ("… within SPECTER is an item of future work" has the shape and says
    nothing), or the first mention in the document that uses the concept most. ``thin`` — a first
    mention anywhere else. The rule is the reasons, spelled out, so the grade can be checked by
    reading them; no count here is tuned on this corpus.
    """
    reasons: list[str] = []
    if hit.form == "coined":
        reasons.append("The author names the term here")
    elif hit.form == "named":
        reasons.append("Describes something, then gives the term as its name")
    elif hit.form == "defining" and hit.opens:
        reasons.append("Worded as a definition, and the sentence opens with the term")
    elif hit.form == "defining":
        reasons.append("Has a definition's wording, but the sentence is not about the term")
    else:
        reasons.append("The first sentence that uses it in this document")
    times = "once" if hit.doc_mentions == 1 else f"{hit.doc_mentions} times"
    if hit.doc_rank == 1:
        reasons.append(f"From the document that mentions it most ({times})")
    else:
        reasons.append(f"From a document that mentions it {times}")
    if hit.shaped_docs >= 2:
        reasons.append(
            f"{hit.shaped_docs} of the {hit.concept_docs} documents that use it word a definition"
        )
    elif hit.concept_docs == 1:
        reasons.append("It appears in only one document")

    if hit.form in ("coined", "named") or (hit.form == "defining" and hit.opens):
        grade = "strong"
    elif hit.form == "defining" or hit.doc_rank == 1:
        grade = "some"
    else:
        grade = "thin"
    return {"grade": grade, "reasons": reasons, "form": hit.form}


def _fingerprint(text: str) -> str:
    """A short content fingerprint for idempotency keys — recognising the same text twice.

    Not a security use (nothing is authenticated or kept secret by it), which is what
    ``usedforsecurity=False`` says; the digest is the same either way.
    """
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _passage_key(hit: PassageHit) -> str:
    return f"{hit.chunk_key}:{_fingerprint(hit.text)}"


def _user_key(text: str) -> str:
    return "user:" + _fingerprint(text.strip())


# ============================================================
# Writing — session-scoped; the only writers of Concept.definition
# ============================================================


class DefinitionError(ValueError):
    """A definition action that does not fit the candidate's current state."""


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def write_passages(session: Session, hits_by_concept: dict[str, list[PassageHit]]) -> int:
    """Store passage candidates. Returns how many rows were **added**.

    Idempotent on ``(concept, source, chunk + text)``. A candidate already stored keeps its status
    — a re-run cannot resurrect a dismissed one or unchoose a chosen one — but its evidence is
    recomputed, since that is derived from the library as it is now.
    """
    added = 0
    for concept_id, hits in hits_by_concept.items():
        if not hits:
            continue
        existing = {
            row.provenance_key: row
            for row in session.execute(
                select(ConceptDefinition).where(
                    ConceptDefinition.concept_id == concept_id,
                    ConceptDefinition.source == "passage",
                )
            ).scalars()
        }
        for hit in hits:
            evidence = passage_evidence(hit)
            key = _passage_key(hit)
            row = existing.get(key)
            if row is not None:
                row.evidence_json = json.dumps(evidence)
                continue
            session.add(
                ConceptDefinition(
                    concept_id=concept_id,
                    text=hit.text,
                    source="passage",
                    provenance_key=key,
                    provenance_json=json.dumps(
                        {"document_id": hit.document_id, "chunk_key": hit.chunk_key}
                    ),
                    evidence_json=json.dumps(evidence),
                    status="suggested",
                )
            )
            added += 1
    session.flush()
    return added


def add_user_definition(
    session: Session, concept_id: str, text: str, *, choose: bool = True
) -> ConceptDefinition:
    """Store the user's own definition, and by default choose it.

    Writing the same text twice returns the existing row rather than adding a second one. Writing
    new text never touches any other candidate — the previous choice goes back to ``suggested``
    through :func:`choose`, where it stays for the user to pick again.
    """
    body = text.strip()
    if not body:
        raise DefinitionError("a definition cannot be empty")
    if session.get(Concept, concept_id) is None:
        raise DefinitionError(f"no concept with id {concept_id!r}")
    key = _user_key(body)
    row = session.execute(
        select(ConceptDefinition).where(
            ConceptDefinition.concept_id == concept_id,
            ConceptDefinition.source == "user",
            ConceptDefinition.provenance_key == key,
        )
    ).scalar_one_or_none()
    if row is None:
        row = ConceptDefinition(
            concept_id=concept_id,
            text=body,
            source="user",
            provenance_key=key,
            provenance_json="{}",
            evidence_json=json.dumps({"grade": "user", "reasons": ["Written by you"]}),
            status="suggested",
        )
        session.add(row)
        session.flush()
    if choose and row.status != "chosen":
        choose_definition(session, row.id)
    return row


def _chosen(session: Session, concept_id: str) -> ConceptDefinition | None:
    return session.execute(
        select(ConceptDefinition).where(
            ConceptDefinition.concept_id == concept_id, ConceptDefinition.status == "chosen"
        )
    ).scalar_one_or_none()


def _mirror(session: Session, concept_id: str, text: str | None) -> None:
    concept = session.get(Concept, concept_id)
    if concept is not None:
        concept.definition = text


def sync_mirror(session: Session, concept_id: str) -> None:
    """Set ``Concept.definition`` from the chosen candidate — for a caller that moved candidates.

    A concept with no candidates at all is left alone: its column may hold text written before
    candidates existed that the boot migration has not turned into one yet (a test database, a
    library opened by an older build), and blanking it would lose that text.
    """
    has_any = session.execute(
        select(ConceptDefinition.id).where(ConceptDefinition.concept_id == concept_id).limit(1)
    ).first()
    if has_any is None:
        return
    chosen = _chosen(session, concept_id)
    _mirror(session, concept_id, chosen.text if chosen is not None else None)


def _candidate(session: Session, definition_id: str) -> ConceptDefinition:
    row = session.get(ConceptDefinition, definition_id)
    if row is None:
        raise DefinitionError(f"no definition candidate with id {definition_id!r}")
    return row


def _record(
    session: Session, concept_id: str, definition_id: str, action: str, previous_id: str | None
) -> None:
    """Append one event to the concept's history, numbered after the last one."""
    last = session.execute(
        select(func.max(ConceptDefinitionEvent.seq)).where(
            ConceptDefinitionEvent.concept_id == concept_id
        )
    ).scalar_one_or_none()
    session.add(
        ConceptDefinitionEvent(
            concept_id=concept_id,
            definition_id=definition_id,
            action=action,
            previous_id=previous_id,
            seq=(last or 0) + 1,
        )
    )
    session.flush()


def choose_definition(session: Session, definition_id: str) -> ConceptDefinition:
    """Make one candidate the concept's definition. The previous one goes back to ``suggested``."""
    row = _candidate(session, definition_id)
    previous = _chosen(session, row.concept_id)
    if previous is not None and previous.id == row.id:
        return row
    if previous is not None:
        previous.status = "suggested"
        previous.decided_at = _now()
    row.status = "chosen"
    row.decided_at = _now()
    _mirror(session, row.concept_id, row.text)
    _record(session, row.concept_id, row.id, "chose", previous.id if previous else None)
    return row


def dismiss_definition(session: Session, definition_id: str) -> ConceptDefinition:
    """Hide a candidate. Dismissing the chosen one leaves the concept with no definition."""
    row = _candidate(session, definition_id)
    if row.status == "dismissed":
        return row
    was_chosen = row.status == "chosen"
    row.status = "dismissed"
    row.decided_at = _now()
    if was_chosen:
        _mirror(session, row.concept_id, None)
    _record(session, row.concept_id, row.id, "dismissed", row.id if was_chosen else None)
    return row


def restore_definition(session: Session, definition_id: str) -> ConceptDefinition:
    """Bring a dismissed candidate back as a suggestion."""
    row = _candidate(session, definition_id)
    if row.status != "dismissed":
        raise DefinitionError("only a dismissed candidate can be restored")
    row.status = "suggested"
    row.decided_at = _now()
    _record(session, row.concept_id, row.id, "restored", None)
    return row


def undo_last(session: Session, concept_id: str) -> ConceptDefinitionEvent | None:
    """Take back the most recent choose / dismiss / restore on a concept. ``None`` if none is left.

    Undo walks back one event at a time, so undoing twice takes back two steps — the ADR-052
    condition that every step can be taken back one by one.
    """
    event = session.execute(
        select(ConceptDefinitionEvent)
        .where(
            ConceptDefinitionEvent.concept_id == concept_id,
            ConceptDefinitionEvent.undone_at.is_(None),
        )
        .order_by(ConceptDefinitionEvent.seq.desc())
        .limit(1)
    ).scalar_one_or_none()
    if event is None:
        return None
    row = session.get(ConceptDefinition, event.definition_id) if event.definition_id else None
    if event.action == "chose":
        if row is not None and row.status == "chosen":
            row.status = "suggested"
        previous = session.get(ConceptDefinition, event.previous_id) if event.previous_id else None
        if previous is not None:
            previous.status = "chosen"
        _mirror(session, concept_id, previous.text if previous is not None else None)
    elif event.action == "dismissed" and row is not None:
        row.status = "chosen" if event.previous_id == row.id else "suggested"
        if row.status == "chosen":
            _mirror(session, concept_id, row.text)
    elif event.action == "restored" and row is not None:
        row.status = "dismissed"
    event.undone_at = _now()
    session.flush()
    return event


# ============================================================
# Reading
# ============================================================


@dataclass(frozen=True)
class DefinitionCandidate:
    id: str
    text: str
    source: str
    status: str
    grade: str
    reasons: tuple[str, ...]
    document_id: str | None = None
    document_title: str | None = None
    chunk_key: str | None = None


@dataclass(frozen=True)
class ConceptDefinitions:
    """Everything the concept panel shows about a concept's definition."""

    concept_id: str
    label: str
    chosen_id: str | None
    candidates: tuple[DefinitionCandidate, ...]
    can_undo: bool
    # No definition-shaped passage and not chosen: the panel says the evidence is thin.
    thin: bool
    extracted: bool  # any passage candidate on record (else: never looked, or nothing found)
    notes: tuple[str, ...] = field(default_factory=tuple)


_GRADE_ORDER = {"user": 0, "strong": 1, "some": 2, "thin": 3}
_STATUS_ORDER = {"chosen": 0, "suggested": 1, "dismissed": 2}


def load_definitions(session: Session, concept_id: str) -> ConceptDefinitions | None:
    """The panel's read model for one concept, or ``None`` for an unknown id or a field node."""
    concept = session.get(Concept, concept_id)
    if concept is None or concept.kind != "concept":
        return None
    rows = list(
        session.execute(
            select(ConceptDefinition).where(ConceptDefinition.concept_id == concept_id)
        ).scalars()
    )
    doc_ids = set()
    parsed: list[tuple[ConceptDefinition, dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        provenance = json.loads(row.provenance_json or "{}")
        evidence = json.loads(row.evidence_json or "{}")
        if provenance.get("document_id"):
            doc_ids.add(provenance["document_id"])
        parsed.append((row, provenance, evidence))
    titles = {
        str(i): (t or f)
        for i, t, f in session.execute(
            select(Document.id, Document.title, Document.filename).where(Document.id.in_(doc_ids))
        ).all()
    }
    candidates = [
        DefinitionCandidate(
            id=row.id,
            text=row.text,
            source=row.source,
            status=row.status,
            grade=str(evidence.get("grade", "thin")),
            reasons=tuple(evidence.get("reasons", [])),
            document_id=provenance.get("document_id"),
            document_title=titles.get(provenance.get("document_id", "")),
            chunk_key=provenance.get("chunk_key"),
        )
        for row, provenance, evidence in parsed
    ]
    candidates.sort(
        key=lambda c: (_STATUS_ORDER.get(c.status, 3), _GRADE_ORDER.get(c.grade, 4), c.text)
    )
    chosen = next((c for c in candidates if c.status == "chosen"), None)
    passages = [c for c in candidates if c.source == "passage"]
    shaped = [c for c in passages if c.grade in ("strong", "some")]
    can_undo = (
        session.execute(
            select(ConceptDefinitionEvent.id)
            .where(
                ConceptDefinitionEvent.concept_id == concept_id,
                ConceptDefinitionEvent.undone_at.is_(None),
            )
            .limit(1)
        ).first()
        is not None
    )
    return ConceptDefinitions(
        concept_id=concept_id,
        label=concept.label,
        chosen_id=chosen.id if chosen else None,
        candidates=tuple(candidates),
        can_undo=can_undo,
        thin=chosen is None and not shaped,
        extracted=bool(passages),
    )


# ============================================================
# Extraction runs
# ============================================================


def chunks_mentioning(
    labels: Sequence[str], *, index_file: Path | None = None
) -> list[tuple[str, str, str]] | None:
    """Every parent chunk of the documents that mention any of ``labels``, via the keyword index.

    The one-concept path ("Look in my library" in the panel). Reading the whole vector store costs
    about ten seconds at 104 documents and grows with the corpus — minutes at the 10,000-document
    contract — while the on-disk keyword index (``sparse_index``, ADR-036) answers "which
    documents say this phrase" in milliseconds and holds every parent block. Only the *documents*
    are chosen here; ``find_passages`` still reads all of each one, in order, so a first mention
    and the per-document counts are the same as a full scan would give.

    Returns ``None`` when the index is missing or unreadable (a CLI run before the app first
    built it), so the caller falls back to the full read rather than finding nothing.
    """
    import sqlite3

    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.keywords import tokenize

    if index_file is None:
        from doc_assistant.config import PC_CHROMA_PATH
        from doc_assistant.sparse_index import index_path

        index_file = index_path(PC_CHROMA_PATH)
    if not index_file.exists():
        return None
    # A prefix query on the last word: the index keeps a hyphenated word whole ("actor-critic",
    # "Rbp4-Cre", "filter-based"), where the mention matcher sees the bare term inside it. The
    # prefix over-selects ("actors") and ``find_passages`` re-checks every mention, so reading a
    # few extra documents is the only cost.
    phrases = [
        '"' + " ".join(tokens) + '"*' for tokens in (tokenize(label) for label in labels) if tokens
    ]
    if not phrases:
        return []
    try:
        con = sqlite3.connect(str(index_file))
        try:
            hashes = sorted(
                {
                    str(row[0])
                    for row in con.execute(
                        "SELECT DISTINCT c.doc_hash FROM chunks_fts "
                        "JOIN chunks c ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ?",
                        (" OR ".join(phrases),),
                    )
                }
            )
            placeholders = ",".join("?" * len(hashes))
            parents = (
                con.execute(
                    "SELECT doc_hash, parent_index, text FROM parents "  # nosec B608
                    f"WHERE doc_hash IN ({placeholders})",
                    hashes,
                ).fetchall()
                if hashes
                else []
            )
        finally:
            con.close()
    except sqlite3.Error as e:
        log.warning("definitions_keyword_index_unreadable", error=str(e))
        return None
    if not parents:
        return []
    with session_scope() as session:
        ids = {
            str(doc_hash): str(doc_id)
            for doc_id, doc_hash in session.execute(
                select(Document.id, Document.doc_hash).where(Document.doc_hash.in_(hashes))
            ).all()
        }
    return [(f"{ids[h]}:p{int(i)}", ids[h], str(text)) for h, i, text in parents if str(h) in ids]


@dataclass(frozen=True)
class ExtractResult:
    """What one extraction found and (with ``apply``) stored."""

    n_concepts: int
    n_with_passages: int
    n_hits: int
    hits: dict[str, list[PassageHit]]
    applied: bool = False
    n_added: int = 0


def extract_definitions(
    *,
    concept_ids: Sequence[str] | None = None,
    apply: bool = False,
    chunks: Iterable[tuple[str, str, str]] | None = None,
) -> ExtractResult:
    """Find passage candidates for concepts (all by default); with ``apply``, store them.

    Reads every parent chunk from the vector store unless ``chunks`` is given (tests, and the
    single-concept path, pass their own). Without ``apply`` nothing is written.
    """
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        stmt = select(Concept.id, Concept.label).where(Concept.kind == "concept")
        if concept_ids is not None:
            stmt = stmt.where(Concept.id.in_(list(concept_ids)))
        concepts = [(str(i), str(label)) for i, label in session.execute(stmt).all()]
    if chunks is None:
        from doc_assistant.knowledge.concept_skeleton import load_presence_inputs

        chunks = load_presence_inputs(None)
    hits = find_passages(concepts, chunks)
    result = ExtractResult(
        n_concepts=len(concepts),
        n_with_passages=sum(1 for h in hits.values() if h),
        n_hits=sum(len(h) for h in hits.values()),
        hits=hits,
    )
    if not apply:
        return result
    with session_scope() as session:
        added = write_passages(session, hits)
    log.info("definitions_extracted", concepts=len(concepts), added=added)
    return ExtractResult(
        n_concepts=result.n_concepts,
        n_with_passages=result.n_with_passages,
        n_hits=result.n_hits,
        hits=hits,
        applied=True,
        n_added=added,
    )
