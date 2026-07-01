"""Deterministic, zero-LLM keyword extraction — the concept-skeleton vocabulary seed.

Fixes KI-13: the concept-skeleton promote seam (``scripts/seed_concepts.py``) mines
``Keyword`` rows, but nothing in the codebase produced them, so the vocabulary path was
dead on real data. This module extracts keyphrases from each document's cached markdown by
corpus **TF-IDF** — pure Python, no LLM, no new dependency — and writes them as
``Keyword(source="extracted")`` rows linked to their documents. Additive, idempotent, and
it never touches the chunk store (Enrichment-Layer Pattern), exactly like the
``extract_citations`` / ``extract_doc_metadata`` sidecar runners.

TF-IDF over a same-domain corpus down-weights ubiquitous terms (``model``, ``bert``) and
surfaces each document's distinctive phrases — which also mitigates the broad-hub density
blow-up the RG-001/008 skeleton run measured (those hubs get a low IDF, so they rank below
the distinctive per-paper terms a curator actually wants to promote).

Pure core (``tokenize`` → ``candidate_terms`` → ``tf_idf_keywords``) has no DB/LLM imports
and is fully unit-testable on toy inputs; the impure boundary (``load_document_texts`` →
``extract_keywords``) reads cached markdown and writes ``Keyword`` rows on the host (KI-5).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# Tech-token aware: keeps internal +/- so "bm25", "cross-encoder", "gpt-4", "specter2"
# survive as single tokens. Splits on everything else (whitespace, punctuation, markdown).
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-+][a-z0-9]+)*")

#: English function words + academic-paper boilerplate. Kept deliberately compact — the
#: user still curates (a Keyword is a *candidate only*, redesign Decision 1); over-pruning
#: here would silently drop promotable domain terms.
STOPWORDS: frozenset[str] = frozenset(
    {
        # articles / conjunctions / prepositions / pronouns
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "from",
        "into",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "us",
        "they",
        "their",
        "them",
        "he",
        "she",
        "his",
        "her",
        "you",
        "your",
        "i",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "over",
        "under",
        "between",
        "both",
        "up",
        "down",
        "out",
        "off",
        "about",
        "above",
        "below",
        "again",
        "further",
        "there",
        "here",
        "also",
        "however",
        "thus",
        "hence",
        "therefore",
        "while",
        "during",
        "before",
        "after",
        "because",
        "via",
        "per",
        "within",
        "without",
        "across",
        # academic boilerplate / units / citation cruft
        "et",
        "al",
        "eg",
        "ie",
        "cf",
        "etc",
        "figure",
        "fig",
        "table",
        "tab",
        "section",
        "sec",
        "eq",
        "equation",
        "appendix",
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
        "acknowledgements",
        "paper",
        "papers",
        "work",
        "works",
        "method",
        "methods",
        "approach",
        "approaches",
        "result",
        "results",
        "show",
        "shows",
        "shown",
        "showed",
        "using",
        "used",
        "use",
        "uses",
        "based",
        "propose",
        "proposed",
        "proposes",
        "present",
        "presented",
        "given",
        "toward",
        "towards",
        "arxiv",
        "preprint",
        "doi",
        "http",
        "https",
        "www",
        "pdf",
        "url",
        "isbn",
        "vol",
        "pp",
        "page",
        "pages",
        "dataset",
        "datasets",
        "model",
        "models",
        "task",
        "tasks",
        "one",
        "two",
        "three",
        "first",
        "second",
        "third",
        "new",
        "different",
        "several",
        "example",
        "examples",
        "number",
        "set",
        "sets",
        "case",
        "cases",
        "note",
        "see",
    }
)


@dataclass(frozen=True)
class ScoredKeyword:
    """A candidate keyphrase and its corpus TF-IDF score (higher = more distinctive)."""

    term: str
    score: float
    tf: int  # occurrences of the term in this document
    df: int  # number of documents in the corpus containing the term


def tokenize(text: str) -> list[str]:
    """Case-folded tech-aware word tokens (``BM25`` → ``bm25``, ``cross-encoder`` intact)."""
    return _TOKEN_RE.findall(text.casefold())


def candidate_terms(
    tokens: list[str],
    *,
    ngram_max: int,
    min_chars: int,
    stopwords: frozenset[str] = STOPWORDS,
) -> list[str]:
    """Candidate uni/bi/tri-grams from a token stream (one entry per occurrence → TF).

    A candidate is rejected if any of its tokens is a stopword (so no ``model of the``
    junk and no phrase padded by function words), if it is shorter than ``min_chars``
    (letters + digits, spaces excluded), or if it contains no alphabetic character (pure
    numbers / IDs are not keywords). Order/repetition is preserved so the caller can count
    term frequency directly.
    """
    terms: list[str] = []
    n = len(tokens)
    for size in range(1, ngram_max + 1):
        for i in range(n - size + 1):
            gram = tokens[i : i + size]
            if any(tok in stopwords for tok in gram):
                continue
            term = " ".join(gram)
            if len(term.replace(" ", "")) < min_chars:
                continue
            if not any(ch.isalpha() for ch in term):
                continue
            terms.append(term)
    return terms


def tf_idf_keywords(
    doc_terms: dict[str, list[str]], *, top_k: int
) -> dict[str, list[ScoredKeyword]]:
    """Rank each document's candidate terms by TF-IDF; return the top ``top_k`` per doc.

    ``doc_terms`` maps ``document_id`` → its candidate-term stream (with repeats).
    Weighting is ``(1 + ln tf) * idf`` with smoothed ``idf = ln((N + 1)/(df + 1)) + 1`` —
    the log-damped TF stops a term repeated hundreds of times from dominating, and the
    smoothed IDF keeps corpus-ubiquitous terms positive but low. Fully deterministic:
    ties break by term ascending, so the output is byte-stable across runs.
    """
    n_docs = len(doc_terms)
    per_doc_counts: dict[str, Counter[str]] = {
        doc_id: Counter(terms) for doc_id, terms in doc_terms.items()
    }
    doc_freq: Counter[str] = Counter()
    for counts in per_doc_counts.values():
        for term in counts:
            doc_freq[term] += 1

    ranked: dict[str, list[ScoredKeyword]] = {}
    for doc_id, counts in per_doc_counts.items():
        scored: list[ScoredKeyword] = []
        for term, tf in counts.items():
            df = doc_freq[term]
            idf = math.log((n_docs + 1) / (df + 1)) + 1.0
            score = (1.0 + math.log(tf)) * idf
            scored.append(ScoredKeyword(term=term, score=score, tf=tf, df=df))
        scored.sort(key=lambda s: (-s.score, s.term))
        ranked[doc_id] = scored[:top_k]
    return ranked


def corpus_band_keywords(
    doc_terms: dict[str, list[str]], *, min_df: int, max_df: int, top_k: int
) -> list[ScoredKeyword]:
    """Select a single corpus vocabulary of shared *mid-document-frequency* terms.

    The counterpart to :func:`tf_idf_keywords`. Where per-doc TF-IDF surfaces each document's
    *distinctive* terms (which are df≈1 and form per-paper cliques in the concept graph), this
    selects terms whose corpus document-frequency falls in ``min_df..max_df`` — the shared band
    that produces cross-document co-occurrence edges. Below the band = paper-specific singletons;
    above it = ubiquitous hubs that saturate the graph. Both are excluded.

    Score is ``df * (1 + ln total_tf)`` — breadth (how many documents) first, substance (how
    often) second. Deterministic: ties break by term ascending. ``ScoredKeyword.tf`` here is the
    *corpus-total* frequency (not per-document, unlike the TF-IDF path). Returns the top ``top_k``.
    """
    per_doc_counts = {doc_id: Counter(terms) for doc_id, terms in doc_terms.items()}
    doc_freq: Counter[str] = Counter()
    total_tf: Counter[str] = Counter()
    for counts in per_doc_counts.values():
        for term, tf in counts.items():
            doc_freq[term] += 1
            total_tf[term] += tf

    scored: list[ScoredKeyword] = []
    for term, df in doc_freq.items():
        if df < min_df or df > max_df:
            continue
        tf = total_tf[term]
        score = df * (1.0 + math.log(tf))
        scored.append(ScoredKeyword(term=term, score=score, tf=tf, df=df))
    scored.sort(key=lambda s: (-s.score, s.term))
    return scored[:top_k]


# --------------------------------------------------------------------------- #
# Impure boundary — reads cached markdown, writes Keyword rows (host only, KI-5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DocKeywords:
    """Per-document extraction result (for reporting)."""

    document_id: str
    filename: str
    keywords: list[ScoredKeyword]
    written: int


@dataclass(frozen=True)
class KeywordExtractionResult:
    """Whole-run result across the corpus."""

    docs: list[DocKeywords]
    n_documents: int
    n_distinct_keywords: int
    total_written: int


def _find_cached_text(source_cache: str | None, source_original: str) -> str | None:
    """Locate a document's cached markdown (mirrors the extract_citations resolver)."""
    from pathlib import Path

    from doc_assistant.config import CACHE_PATH, DOCS_PATH

    if source_cache:
        p = Path(source_cache)
        if p.exists():
            return p.read_text(encoding="utf-8")
    original = Path(source_original)
    if original.exists():
        try:
            relative = original.relative_to(DOCS_PATH)
            derived = CACHE_PATH / relative.with_suffix(".md")
            if derived.exists():
                return derived.read_text(encoding="utf-8")
        except ValueError:
            pass
    for candidate_path in (source_cache, source_original):
        if not candidate_path:
            continue
        stem = Path(candidate_path.replace("\\", "/")).stem
        derived = CACHE_PATH / f"{stem}.md"
        if derived.exists():
            return derived.read_text(encoding="utf-8")
    return None


def load_document_texts(
    document_ids: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``[(document_id, filename, cached_markdown)]`` for the corpus.

    Reads every non-archived ``Document``'s cached markdown (documents without a resolvable
    cache are skipped with a warning). ``document_ids`` restricts *which* rows are returned,
    but the caller should still compute IDF over the whole corpus for stable statistics.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    out: list[tuple[str, str, str]] = []
    with session_scope() as session:
        stmt = select(Document).where(Document.is_archived.is_(False))
        if document_ids is not None:
            stmt = stmt.where(Document.id.in_(document_ids))
        rows = list(session.execute(stmt).scalars())
        for doc in rows:
            text = _find_cached_text(doc.source_cache, doc.source_original)
            if text is None:
                log.warning("no_cached_markdown", document_id=doc.id, filename=doc.filename)
                continue
            out.append((doc.id, doc.filename, text))
    return out


def _persist_keywords(document_id: str, terms: list[str], *, force: bool) -> int:
    """Write ``Keyword(source="extracted")`` rows for one doc + link them. Idempotent.

    Skips a doc that already has extracted keywords unless ``force`` (then its existing
    *extracted* links are cleared first; author/manual keywords are left untouched). A
    Keyword is get-or-create by unique name, so a term shared across docs is one row with
    multiple document links. Returns the number of (keyword ← document) links added.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Document, Keyword
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return 0
        has_extracted = any(k.source == "extracted" for k in doc.keywords)
        if has_extracted and not force:
            return 0
        if force:
            doc.keywords = [k for k in doc.keywords if k.source != "extracted"]

        linked = {k.name for k in doc.keywords}
        added = 0
        for term in terms:
            if term in linked:
                continue
            keyword = session.execute(
                select(Keyword).where(Keyword.name == term)
            ).scalar_one_or_none()
            if keyword is None:
                keyword = Keyword(name=term, source="extracted")
                session.add(keyword)
            doc.keywords.append(keyword)
            linked.add(term)
            added += 1
        return added


def extract_keywords(
    *,
    apply: bool,
    force: bool = False,
    document_id: str | None = None,
    top_k: int,
    ngram_max: int,
    min_chars: int,
    mode: str = "per_doc",
    min_df: int = 2,
    max_df_frac: float = 0.7,
) -> KeywordExtractionResult:
    """Extract keyphrases (per-doc TF-IDF or corpus mid-DF band); optionally persist.

    ``mode="per_doc"`` ranks each document's distinctive terms by TF-IDF (``top_k`` per doc).
    ``mode="corpus_band"`` selects ONE corpus vocabulary of shared terms whose document-frequency
    is in ``min_df .. floor(max_df_frac * N)`` (``top_k`` total), then links each to the documents
    it appears in — the vocabulary that yields cross-document concept edges. Statistics are always
    computed over the whole cached corpus; ``document_id`` only restricts what is reported/written.
    Deterministic and free (no LLM). ``apply=False`` writes nothing (dry run).
    """
    corpus = load_document_texts()  # whole corpus → stable statistics
    filenames = {doc_id: fname for doc_id, fname, _ in corpus}
    doc_terms = {
        doc_id: candidate_terms(tokenize(text), ngram_max=ngram_max, min_chars=min_chars)
        for doc_id, _, text in corpus
    }

    ranked: dict[str, list[ScoredKeyword]]
    if mode == "corpus_band":
        max_df = max(min_df, int(max_df_frac * len(doc_terms)))
        selected = corpus_band_keywords(doc_terms, min_df=min_df, max_df=max_df, top_k=top_k)
        by_term = {s.term: s for s in selected}
        chosen = set(by_term)
        ranked = {
            doc_id: sorted(
                (by_term[t] for t in set(terms) & chosen),
                key=lambda s: (-s.score, s.term),
            )
            for doc_id, terms in doc_terms.items()
        }
    else:
        ranked = tf_idf_keywords(doc_terms, top_k=top_k)

    targets = [document_id] if document_id is not None else list(ranked)
    docs: list[DocKeywords] = []
    distinct: set[str] = set()
    total_written = 0
    for doc_id in targets:
        keywords = ranked.get(doc_id, [])
        distinct.update(k.term for k in keywords)
        written = 0
        if apply:
            written = _persist_keywords(doc_id, [k.term for k in keywords], force=force)
            total_written += written
        docs.append(
            DocKeywords(
                document_id=doc_id,
                filename=filenames.get(doc_id, doc_id),
                keywords=keywords,
                written=written,
            )
        )
    if apply:
        log.info(
            "keywords_extracted",
            documents=len(docs),
            distinct=len(distinct),
            links_written=total_written,
        )
    return KeywordExtractionResult(
        docs=docs,
        n_documents=len(docs),
        n_distinct_keywords=len(distinct),
        total_written=total_written,
    )
