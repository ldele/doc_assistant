"""How the library writes each concept's surface forms — case included (ADR-053 decision 3).

``Concept.label`` is curated and stored lower-cased, which loses a distinction the text keeps:
``dIN`` (an interneuron) and ``Din`` (the join in "vitamin Din"), ``Cre`` (Cre recombinase) and
``CRE`` (cAMP response element) are different words. This module derives, from the prose, the
spelling the library uses for each surface form, and keeps it **beside** the label — never a
rewrite of it (ADR-043). Matching reads it through ``concept_skeleton.form_matcher``, so presence,
gaps, definitions and epistemics all apply one rule; the Graph tab can show it.

**The vote.** A form's whole-word, case-insensitive occurrences in **body prose**
(``definitions.document_sentences``: the bibliography cut, headings, tables and lists dropped) that
are **not at the start of a sentence** — a capital there says nothing about the word. Each document
votes for the spelling it uses most (a tie goes to lower case); the written form is the spelling
most documents vote for, a tie broken by uses overall, then by lower case. Documents vote rather
than occurrences because one paper that repeats a dataset's name a hundred times (``PERSONA``) must
not outvote a library that uses the word ``persona``. Measured on the working library before it
was built (``tests/eval/baselines/written_forms_2026-09-30.md``).

Zero LLM, zero network. The only writer is the full skeleton build
(``concept_skeleton.build_concept_skeleton`` on apply), which scans every chunk already.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import structlog

from doc_assistant.knowledge.concept_skeleton import is_cased, surface_forms

log = structlog.get_logger(__name__)

_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class WrittenForm:
    """The spelling the library uses for one surface form, and the evidence for it."""

    concept_id: str
    form: str  # casefolded, as ``concept_skeleton.surface_forms`` gives it
    written: str
    uses: int  # mid-sentence uses in body prose
    documents: int  # documents with at least one such use
    variants: dict[str, int] = field(default_factory=dict)  # spelling -> uses
    votes: dict[str, int] = field(default_factory=dict)  # spelling -> documents voting for it

    @property
    def cased(self) -> bool:
        """Whether matching this form depends on case (``concept_skeleton.is_cased``)."""
        return is_cased(self.written)


def _favourite(counts: Counter[str]) -> str:
    """The most used spelling; a tie goes to the lower-case one, then alphabetically."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] != kv[0].lower(), kv[0]))[0][0]


def derive_written_forms(
    concepts: Sequence[tuple[str, str]],
    aliases: Mapping[str, list[str]],
    chunks: Iterable[tuple[str, str, str]],
) -> list[WrittenForm]:
    """The written form of every surface form that the body prose uses mid-sentence. Pure.

    ``concepts`` = ``(concept_id, label)``; ``aliases`` maps ``concept_id`` → aliases; ``chunks``
    = ``(chunk_key, document_id, text)``. A form with no mid-sentence use has no written form, and
    matches case-folded — an empty library yields ``[]`` (the robustness contract).
    """
    from doc_assistant.knowledge.definitions import document_sentences

    sentences = document_sentences(chunks)
    out: list[WrittenForm] = []
    for concept_id, label in concepts:
        for form in surface_forms(label, list(aliases.get(concept_id, []))):
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])", re.IGNORECASE
            )
            per_doc: dict[str, Counter[str]] = defaultdict(Counter)
            for document_id, spans in sentences.items():
                for _chunk_key, span, low in spans:
                    if form not in low:
                        continue
                    for m in pattern.finditer(span):
                        if _ALNUM.search(span, 0, m.start()):  # not the sentence's first word
                            per_doc[document_id][m.group(0)] += 1
            if not per_doc:
                continue
            variants: Counter[str] = Counter()
            for counts in per_doc.values():
                variants.update(counts)
            votes = Counter(_favourite(counts) for counts in per_doc.values())
            written = sorted(votes, key=lambda s: (-votes[s], -variants[s], s != s.lower(), s))[0]
            out.append(
                WrittenForm(
                    concept_id=concept_id,
                    form=form,
                    written=written,
                    uses=sum(variants.values()),
                    documents=len(per_doc),
                    variants=dict(variants.most_common()),
                    votes=dict(votes.most_common()),
                )
            )
    return out


def spelling_map(forms: Iterable[WrittenForm]) -> dict[tuple[str, str], str]:
    """``(concept_id, form)`` → written spelling, the shape ``match_presence`` takes."""
    return {(f.concept_id, f.form): f.written for f in forms}


def cased_label(label: str, written: str | None) -> str | None:
    """How to show ``label`` when the library writes it in a case, else ``None``.

    ``None`` when the written form is lower case, missing, or no longer a spelling of the label
    (the label was edited since the last build) — the caller shows the label as stored."""
    if not written or written.casefold() != label.strip().casefold():
        return None
    return written if is_cased(written) else None


# ============================================================
# Impure boundary — the vocabulary in, the table out and back
# ============================================================


def load_vocabulary() -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Every text-bearing concept and its aliases — the whole vocabulary, not just the graph's.

    Definitions and the vocabulary search reach concepts off the graph, so the written forms cover
    them all. Read through the taxonomy's kind guard (``presence_query``, ADR-028 D4)."""
    from sqlalchemy import select

    from doc_assistant.db.models import ConceptAlias
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import presence_query

    concepts: list[tuple[str, str]] = []
    aliases: dict[str, list[str]] = defaultdict(list)
    with session_scope() as session:
        for row in session.execute(presence_query()).scalars():
            concepts.append((str(row.id), str(row.label)))
        ids = {cid for cid, _ in concepts}
        for arow in session.execute(select(ConceptAlias)).scalars():
            if str(arow.concept_id) in ids:
                aliases[str(arow.concept_id)].append(str(arow.alias))
    concepts.sort()
    return concepts, dict(aliases)


def replace_written_forms(forms: Sequence[WrittenForm]) -> int:
    """Replace the whole ``concept_written_forms`` table with ``forms``; return the row count.

    Whole-table, like the other derived skeleton tables: a form the library stopped using, or a
    concept since deleted, leaves nothing behind. Never touches ``concepts`` (ADR-043)."""
    from sqlalchemy import delete

    from doc_assistant.db.models import ConceptWrittenForm
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(ConceptWrittenForm))
        session.add_all(
            ConceptWrittenForm(
                concept_id=f.concept_id,
                form=f.form,
                written=f.written,
                uses=f.uses,
                documents=f.documents,
                variants_json=json.dumps(f.variants, ensure_ascii=False),
                votes_json=json.dumps(f.votes, ensure_ascii=False),
            )
            for f in forms
        )
    log.info("written_forms_replaced", rows=len(forms), cased=sum(1 for f in forms if f.cased))
    return len(forms)


def load_written_forms(concept_ids: Iterable[str] | None = None) -> dict[tuple[str, str], str]:
    """``(concept_id, form)`` → written spelling, from the last full build (``{}`` before one)."""
    from sqlalchemy import select

    from doc_assistant.db.models import ConceptWrittenForm
    from doc_assistant.db.session import session_scope

    stmt = select(
        ConceptWrittenForm.concept_id, ConceptWrittenForm.form, ConceptWrittenForm.written
    )
    if concept_ids is not None:
        stmt = stmt.where(ConceptWrittenForm.concept_id.in_(list(concept_ids)))
    with session_scope() as session:
        return {(str(c), str(f)): str(w) for c, f, w in session.execute(stmt).all()}


def label_spellings(concepts: Sequence[tuple[str, str]]) -> dict[str, str]:
    """``concept_id`` → the written spelling of its **label** form, for label-only matchers
    (definitions, epistemics). Concepts without one are absent."""
    stored = load_written_forms([cid for cid, _ in concepts])
    out: dict[str, str] = {}
    for concept_id, label in concepts:
        written = stored.get((concept_id, label.strip().casefold()))
        if written:
            out[concept_id] = written
    return out
