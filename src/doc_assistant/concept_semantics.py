"""Semantic concept layer (#2) — title+abstract candidates + concept embedding distance.

Two capabilities the DF-based keyword extractor lacks, both grounding vocabulary in *meaning*
rather than frequency (the RG-001 lesson: statistics pick boilerplate, not concepts):

1. **Title+abstract candidate extraction (scientific papers).** A paper's title + abstract are an
   author-curated concentration of its key concepts with almost no boilerplate — the opposite of
   full-text, where generic academic vocab dominates. Extracting candidates from just the
   title+abstract surfaces the real concepts. **Papers only:** books / markdown notes have no
   abstract, so `extract_abstract` returns None and the caller falls back to full-text keywords or
   manual curation. The load-bearing assumption (yours): a paper's chunks are all *about* its
   title+abstract, so those two fields are a faithful concept summary of the whole document.

2. **Concept semantic distance.** We embed concept labels (+ definitions) and compute pairwise
   cosine — the first place the project measures concept↔concept distance (doc↔doc lives in
   `doc_vectors.py`). Used to suggest near-duplicate concepts to merge (dedup a curated glossary),
   and as a future semantic edge signal for the graph.

Pure core (`extract_abstract`, `abstract_candidates`, `nearest_pairs`) is unit-testable with no
model; the embedding boundary (`embed_texts`) reuses the retrieval embedder (loads the model).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import structlog

from doc_assistant.keywords import candidate_terms, tokenize

log = structlog.get_logger(__name__)

# `## **Abstract**` / `Abstract` heading line, tolerant of markdown emphasis.
_ABSTRACT_HEAD = re.compile(r"(?im)^#{0,4}\s*[*_]*\s*abstract\s*[*_]*\s*$")
# The next markdown heading (usually `## **1 Introduction**`) bounds the abstract body.
_NEXT_HEAD = re.compile(r"(?m)^#{1,4}\s")
_EMPH = re.compile(r"[*_`#>]+")
_FOOTNOTE = re.compile(r"\[\d+\]")


def extract_abstract(markdown: str) -> str | None:
    """The abstract text from a scientific paper's cached markdown, or None if there is none.

    Finds an ``Abstract`` heading and returns the text up to the next markdown heading. Markdown
    emphasis + ``[n]`` footnote markers stripped, whitespace collapsed. None for documents without
    an abstract heading (books, notes) — the caller chooses the fallback.
    """
    m = _ABSTRACT_HEAD.search(markdown)
    if m is None:
        return None
    rest = markdown[m.end() :]
    nxt = _NEXT_HEAD.search(rest)
    body = rest[: nxt.start()] if nxt else rest[:4000]
    body = _FOOTNOTE.sub("", _EMPH.sub("", body))
    body = " ".join(body.split())
    return body or None


def abstract_candidates(
    title: str | None,
    abstract: str | None,
    *,
    top_k: int,
    ngram_max: int = 3,
    min_chars: int = 3,
) -> list[str]:
    """Candidate concept phrases from a paper's title + abstract (concept-dense, low boilerplate).

    Ranks by (frequency, phrase length, term) so multi-word domain phrases surface first. The
    abstract is short, so TF-IDF is degenerate — plain frequency + a multi-word bias is the right
    ranking here. Returns ``[]`` when both title and abstract are missing (non-paper documents).
    """
    text = ". ".join(part for part in (title, abstract) if part)
    if not text.strip():
        return []
    terms = candidate_terms(tokenize(text), ngram_max=ngram_max, min_chars=min_chars)
    counts = Counter(terms)
    ranked = sorted(counts, key=lambda t: (-counts[t], -len(t.split()), t))
    return ranked[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class ConceptPair:
    """Two curated concepts and their embedding cosine (a merge/dedup candidate)."""

    label_a: str
    label_b: str
    cosine: float


def nearest_pairs(
    labels: list[str], vectors: list[list[float]], *, threshold: float
) -> list[ConceptPair]:
    """All concept pairs with cosine ≥ ``threshold``, most-similar first (pure; inject vectors).

    A high cosine between two curated concepts means they may be synonyms that should be one node
    (e.g. ``dense retrieval`` vs ``dense passage retrieval``) — the semantic merge-suggestion
    primitive. Deterministic: ties break by (label_a, label_b).
    """
    pairs: list[ConceptPair] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = _cosine(vectors[i], vectors[j])
            if sim >= threshold:
                pairs.append(ConceptPair(labels[i], labels[j], sim))
    pairs.sort(key=lambda p: (-p.cosine, p.label_a, p.label_b))
    return pairs


# --------------------------------------------------------------------------- #
# Impure boundary — embeddings + DB (host only; loads the retrieval model)
# --------------------------------------------------------------------------- #


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed strings with the active retrieval model (bge-base). Loads the model — not free."""
    from doc_assistant.embeddings import get_embeddings

    vectors = get_embeddings().embed_documents(texts)
    return [[float(x) for x in vec] for vec in vectors]


def suggest_from_abstracts(
    document_ids: list[str] | None = None, *, top_k: int
) -> list[tuple[str, str, list[str]]]:
    """Per-document ``(document_id, filename, candidate concepts)`` from title + abstract.

    Papers only — a document with no extractable abstract yields ``[]`` candidates (the caller
    falls back to full-text keywords / manual curation). Zero LLM; no embedding needed.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.keywords import _find_cached_text

    with session_scope() as session:
        stmt = select(Document).where(Document.is_archived.is_(False))
        if document_ids is not None:
            stmt = stmt.where(Document.id.in_(document_ids))
        rows = [
            (d.id, d.filename, d.title, _find_cached_text(d.source_cache, d.source_original))
            for d in session.execute(stmt).scalars()
        ]

    out: list[tuple[str, str, list[str]]] = []
    for doc_id, filename, title, markdown in rows:
        abstract = extract_abstract(markdown) if markdown else None
        candidates = abstract_candidates(title, abstract, top_k=top_k)
        if not candidates:
            log.info("no_title_abstract", document_id=doc_id, filename=filename)
        out.append((doc_id, filename, candidates))
    return out


def concept_merge_suggestions(*, threshold: float) -> list[ConceptPair]:
    """Embed curated concepts (label + definition) and return near-duplicate pairs to merge.

    The first concept↔concept distance in the project. Definitions, when present, enrich the
    embedding beyond the bare label. Returns ``[]`` for fewer than two concepts.
    """
    from doc_assistant.concept_skeleton import load_glossary

    entries = load_glossary()
    if len(entries) < 2:
        return []
    labels = [e.label for e in entries]
    texts = [f"{e.label}. {e.definition}" if e.definition else e.label for e in entries]
    vectors = embed_texts(texts)
    return nearest_pairs(labels, vectors, threshold=threshold)
