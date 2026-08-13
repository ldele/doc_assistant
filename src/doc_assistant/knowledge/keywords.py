"""Deterministic, zero-LLM keyword extraction — the concept-skeleton vocabulary seed.

Fixes KI-13: the concept-skeleton promote seam (``scripts/seed_concepts.py``) mines
``Keyword`` rows, but nothing in the codebase produced them, so the vocabulary path was
dead on real data. This module extracts keyphrases from each document's cached markdown —
zero LLM — and writes them as ``Keyword(source="extracted")`` rows linked to their
documents. Additive, idempotent, and it never touches the chunk store (Enrichment-Layer
Pattern), exactly like the ``extract_citations`` / ``extract_doc_metadata`` sidecar runners.
Three modes: ``per_doc`` TF-IDF and ``corpus_band`` (pure Python), and ``contrastive``
(R3 / ADR-006) — C-value nested discount * reference-corpus weirdness, using ``wordfreq``
as the general-English reference.

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
from collections import Counter, defaultdict
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# Tech-token aware: keeps internal +/- so "bm25", "cross-encoder", "gpt-4", "specter2"
# survive as single tokens. Splits on everything else (whitespace, punctuation, markdown).
#
# D5 — `.` and `/` are also kept, but **only when the next character is a digit**. That is the
# whole of what separates a designator from prose: `16p11.2` and `c57bl/6` are one term each,
# while `e.g` / `i.e` / `arxiv.org` split as they always did (the character after the separator
# is a letter). Without this the tokeniser truncated at the separator — `16p11.2` became
# `16p11`, `C57BL/6` became `c57bl` — silently renaming a locus and a mouse strain.
# The residual leak this opens (`fig.2`, whose head is a stopword) is closed in
# `candidate_terms` by checking the pre-separator head against the stopword sets.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-+][a-z0-9]+|[./]\d[a-z0-9]*)*")

#: A token's head — the part before its first `.` or `/`. `arxiv.org` -> `arxiv`, `fig.2` -> `fig`.
_TOKEN_HEAD = re.compile(r"^[^./]+")

# D4 — citation artifacts by SHAPE, never by vocabulary.
#
# ⚠ The rule that governs this whole section: **a surname is not junk.** `cajal`, `cre`, `dbs`,
# `16p11`, `c57bl` are real specialist vocabulary in this corpus and have been mistaken for noise
# twice. So there is no name list here and there must never be one — author names are removed by
# deleting the *place* they pile up (the reference section, below), not by deciding which words
# look like people. A surname that survives that appears in the document's own prose, which is
# exactly when it is a real term for that document ("Shadmehr's model").
#
# These two patterns are pure shape and match nothing in the protected list above:
#   `2014a` / `2015b` — a citation year with a disambiguating letter.
#   `e04250`          — a publisher article id (eLife-style: one letter, then >=4 digits).
#   `10.18653`        — a DOI registrant prefix. Only visible once D5 stopped splitting on `.`,
#                       which is why it is fixed here rather than having been there all along.
_CITATION_YEAR_SUFFIX = re.compile(r"^\d{4}[a-z]$")
_ARTICLE_ID = re.compile(r"^[a-z]\d{4,}$")
_DOI_PREFIX = re.compile(r"^10\.\d{4,5}$")

#: Headings that begin a bibliography. Matched as a *whole line* only, so the word appearing
#: mid-sentence never triggers a cut.
#:
#: The emphasis markers are load-bearing, not defensive: a first pass without them fired on only
#: **25 of 97** documents, because PyMuPDF4LLM's dominant rendering is ``## **References**`` (32
#: of the sampled headings) with ``_REFERENCES_`` behind it. Matching the plain form alone would
#: have looked like a working fix while missing three quarters of the corpus.
_REFERENCE_HEADING = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:\d+\.?[ \t]*)?[*_]{0,2}[ \t]*"
    r"(?:references|bibliography|works[ \t]+cited|literature[ \t]+cited)"
    r"[ \t]*[*_]{0,2}[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: **Structural, not corpus-tuned.** A bibliography sits in the back of a document; requiring the
#: heading past the halfway mark is what stops a paper that merely *discusses* references early
#: from having its body amputated. Cheap insurance against the one catastrophic failure this
#: function has available to it.
REFERENCE_SECTION_MIN_POSITION = 0.5

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

# Scholarly-metadata artifacts: publisher / journal / preprint-server / repository names and
# identifier labels that leak out of reference lists and page headers. They are *rare* in
# general English, so contrastive weirdness scores them as "distinctive" even though they carry
# no topical meaning (RG-001/R3 flagged this as the contrastive mode's known publisher-artifact
# limitation). Filtered exactly like STOPWORDS — a candidate is dropped if ANY of its tokens is
# here, so "7554 elife" (the eLife DOI registrant + journal) goes out with "elife".
# Deliberately EXCLUDES words that double as domain concepts in a neuroscience/ML corpus
# (`cell`, `neuron`, `nature`, `science`) — those must survive as real keywords.
VENUE_STOPWORDS: frozenset[str] = frozenset(
    {
        # preprint servers / repositories / code hosts
        "arxiv",
        "biorxiv",
        "medrxiv",
        "chemrxiv",
        "ssrn",
        "osf",
        "zenodo",
        "figshare",
        "dryad",
        "github",
        # publishers / imprints
        "elsevier",
        "springer",
        "wiley",
        "plos",
        "frontiers",
        "frontiersin",
        "ieee",
        "acm",
        "sage",
        "bmc",
        "mdpi",
        "hindawi",
        # journal / venue abbreviations (as they appear casefolded in headers + refs)
        "elife",
        "jneurosci",
        "neurosci",
        "neuroimage",
        "neurobiol",
        "neurophysiol",
        "physiol",
        "biophys",
        "fnana",
        "pnas",
        "jmlr",
        "neurips",
        "nips",
        "iclr",
        "icml",
        "cvpr",
        "eccv",
        "iccv",
        # identifier / bibliographic labels
        "doi",
        "pmid",
        "pmc",
        "pubmed",
        "issn",
        "isbn",
        "orcid",
        "http",
        "https",
        "www",
        "preprint",
        "et",
        "al",
    }
)


@dataclass(frozen=True)
class ScoredKeyword:
    """A candidate keyphrase and its corpus TF-IDF score (higher = more distinctive)."""

    term: str
    score: float
    tf: int  # occurrences of the term in this document
    df: int  # number of documents in the corpus containing the term


# --------------------------------------------------------------------------- #
# D1 — page furniture. Running headers/footers dominate TF before anything else runs.
# --------------------------------------------------------------------------- #

#: The page marker the chunker writes into cached markdown. Imported by value rather than from
#: ``ingest.chunking`` to keep this module's pure core free of ingest imports (ADR-023).
_PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")

#: Runs of digits collapse to one placeholder when deciding whether two lines are "the same"
#: line of furniture. "Page 3 of 12" and "Page 4 of 12" are one running footer, not two lines.
_DIGITS = re.compile(r"\d+")

#: **Structural, not corpus-tuned** (the robustness contract). Furniture is defined by *repeating
#: across a document's own pages*, so the threshold is a fraction of that document's page count and
#: carries no assumption about corpus size, domain, or publisher. Half is deliberately
#: conservative: a running header appears on essentially every page, while a genuine sentence
#: repeated on half a paper's pages does not exist.
FURNITURE_PAGE_FRACTION = 0.5

#: ...and a floor, because "repeats on half the pages" is meaningless at 1-2 pages, where it would
#: describe an ordinary two-page paper's every shared line.
FURNITURE_MIN_PAGES = 3


def _furniture_key(line: str) -> str:
    """Normalised identity of a line for repetition counting (digit- and case-blind)."""
    return _DIGITS.sub("#", line.strip().casefold())


def split_pages(text: str) -> list[str]:
    """Split cached markdown on ``<!-- page:N -->`` markers into per-page blocks.

    Returns a single block for input with no markers (EPUB/HTML/MD, and any PDF whose extractor
    did not emit them) — which is what makes :func:`strip_page_furniture` a no-op there rather
    than a guess.
    """
    parts = _PAGE_MARKER.split(text)
    if len(parts) == 1:
        return [text]
    # re.split with one capture group yields [before, page_no, block, page_no, block, ...].
    return [parts[0], *parts[2::2]]


def strip_page_furniture(
    text: str,
    *,
    page_fraction: float = FURNITURE_PAGE_FRACTION,
    min_pages: int = FURNITURE_MIN_PAGES,
) -> str:
    """Remove lines that repeat across a document's pages — running headers, journal stamps.

    **This is the single largest defect in the keyword layer** (measured 2026-08-11 on the live
    97-document corpus): on ``nihms-66884.pdf``, **11 of 15** keyword slots were shingles of the
    PMC running header *"Exp Brain Res. Author manuscript; available in PMC 2008 September 26"*.
    It repeats on every page, so its term frequency beats the paper's actual subject matter.

    The signal is **position and repetition, not vocabulary** — which is why this is not another
    stopword list. ``VENUE_STOPWORDS`` cannot scale to it: every publisher has a different stamp,
    and the words in them (``brain``, ``september``) are not junk anywhere else.

    Degrades to the identity function when the text has no page markers or too few pages, because
    with nothing to repeat *across*, "repeated" has no meaning (0-document robustness in
    miniature: the honest answer to an unanswerable question is to change nothing).
    """
    pages = split_pages(text)
    if len(pages) < min_pages:
        return text
    seen_on: dict[str, set[int]] = defaultdict(set)
    for i, page in enumerate(pages):
        for line in page.splitlines():
            key = _furniture_key(line)
            if key:  # blank lines are structure, not furniture
                seen_on[key].add(i)
    threshold = max(min_pages, math.ceil(page_fraction * len(pages)))
    furniture = {key for key, seen in seen_on.items() if len(seen) >= threshold}
    if not furniture:
        return text
    kept = [ln for ln in text.splitlines() if _furniture_key(ln) not in furniture]
    log.debug("page_furniture_stripped", pages=len(pages), lines=len(furniture))
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# D2 — overlapping shingles. One phrase must not eat a document's whole budget.
# --------------------------------------------------------------------------- #


def _is_contiguous_subspan(inner: tuple[str, ...], outer: tuple[str, ...]) -> bool:
    """Whether ``inner`` appears as a contiguous run of tokens inside ``outer``."""
    n, m = len(inner), len(outer)
    if n > m:
        return False
    return any(outer[i : i + n] == inner for i in range(m - n + 1))


def suppress_nested(scored: list[ScoredKeyword], *, top_k: int) -> list[ScoredKeyword]:
    """Take the top ``top_k``, skipping any term that overlaps one already taken.

    **The second largest defect** (measured 2026-08-11): ``transformer_vaswani_2017.pdf`` spent
    **9 of its 15 slots** on ``eos`` · ``eos pad`` · ``pad`` · ``pad br`` · ``eos pad br`` … —
    five slots for one artifact in one figure. n-gram candidate generation emits every window, so
    a frequent phrase necessarily produces a frequent sub-phrase, and both score well.

    Greedy over the existing rank order, so the **highest-scoring** member of each overlapping
    family wins and the rest are dropped: a term is skipped if it is a contiguous sub-span of an
    accepted term *or* contains one. That direction matters — dropping only sub-spans would let
    ``eos pad br`` in after ``eos``, and dropping only super-spans would keep the fragments.

    Deliberately **not** applied to the ``contrastive`` mode, which already discounts nested terms
    through C-value, nor to ``corpus_band``, whose exposure to this is via page furniture and is
    fixed at source by :func:`strip_page_furniture`.
    """
    accepted: list[ScoredKeyword] = []
    accepted_tokens: list[tuple[str, ...]] = []
    for cand in scored:
        tokens = tuple(cand.term.split())
        if any(
            _is_contiguous_subspan(tokens, other) or _is_contiguous_subspan(other, tokens)
            for other in accepted_tokens
        ):
            continue
        accepted.append(cand)
        accepted_tokens.append(tokens)
        if len(accepted) >= top_k:
            break
    return accepted


def strip_reference_section(
    text: str, *, min_position: float = REFERENCE_SECTION_MIN_POSITION
) -> str:
    """Cut everything from a ``References`` / ``Bibliography`` heading to the end of the document.

    **This is D4's real fix, and it is deliberately structural.** Author surnames dominate keyword
    output because a bibliography repeats fifty of them, several times each — not because surnames
    are inherently junk. Removing the *place* they accumulate keeps every surname that earns its
    keep in the document's own prose, and needs no opinion about which words are people's names.
    That matters: ``cajal``, ``cre``, ``dbs``, ``16p11`` and ``c57bl`` are real vocabulary in this
    corpus and have twice been mistaken for noise.

    Only a **whole-line** heading counts, and only past ``min_position`` of the way through, so a
    document that discusses references in prose cannot lose its body. Returns ``text`` unchanged
    when no qualifying heading exists — including for any document with no bibliography at all.
    """
    if not text:
        return text
    cutoff = int(min_position * len(text))
    for match in _REFERENCE_HEADING.finditer(text):
        if match.start() >= cutoff:
            log.debug("reference_section_stripped", cut_at=match.start(), total=len(text))
            return text[: match.start()]
    return text


def tokenize(text: str) -> list[str]:
    """Case-folded tech-aware word tokens (``BM25`` → ``bm25``, ``cross-encoder`` intact).

    Designators keep an internal ``.``/``/`` when a digit follows it (``16p11.2``, ``c57bl/6``);
    everything else splits there as before (``e.g``, ``arxiv.org``). See :data:`_TOKEN_RE`.
    """
    return _TOKEN_RE.findall(text.casefold())


def is_citation_artifact(token: str) -> bool:
    """Whether ``token`` is a citation *shape*: ``2014a``, ``e04250``, or a DOI prefix.

    Shape only. Nothing here inspects vocabulary, so no real term can be caught by adding a word
    to a list later — the failure mode this function is written to avoid.
    """
    return bool(
        _CITATION_YEAR_SUFFIX.match(token) or _ARTICLE_ID.match(token) or _DOI_PREFIX.match(token)
    )


def candidate_terms(
    tokens: list[str],
    *,
    ngram_max: int,
    min_chars: int,
    stopwords: frozenset[str] = STOPWORDS,
    venue_stopwords: frozenset[str] = VENUE_STOPWORDS,
) -> list[str]:
    """Candidate uni/bi/tri-grams from a token stream (one entry per occurrence → TF).

    A candidate is rejected if any of its tokens is a stopword (so no ``model of the``
    junk and no phrase padded by function words) or a ``venue_stopwords`` scholarly-metadata
    artifact (publisher / journal / repository / ID token — so ``elife`` and ``7554 elife``
    both go), if it is shorter than ``min_chars`` (letters + digits, spaces excluded), or if
    it contains no alphabetic character (pure numbers / IDs are not keywords). Order/repetition
    is preserved so the caller can count term frequency directly.

    Two D4/D5 rules join those. A token is also rejected when its **head** — the part before its
    first ``.``/``/`` — is a stopword, which is what keeps ``fig.2`` and ``arxiv.org`` out now
    that :data:`_TOKEN_RE` can hold a separator; and when it is a citation *shape*
    (:func:`is_citation_artifact`). Neither rule consults a vocabulary of names.
    """
    terms: list[str] = []
    n = len(tokens)
    for size in range(1, ngram_max + 1):
        for i in range(n - size + 1):
            gram = tokens[i : i + size]
            if any(tok in stopwords or tok in venue_stopwords for tok in gram):
                continue
            if any(is_citation_artifact(tok) for tok in gram):
                continue
            # `fig.2` / `arxiv.org` survive tokenisation as one token, so the stopword check
            # above misses them — the head is what carries the meaning, so test that too.
            if any(
                (head := _TOKEN_HEAD.match(tok).group()) != tok  # type: ignore[union-attr]
                and (head in stopwords or head in venue_stopwords)
                for tok in gram
            ):
                continue
            # A repeated single token ("outflux outflux outflux") is an OCR/extraction
            # artifact, never a keyphrase — weirdness would otherwise rank it highly (RG-001/R3).
            if size > 1 and len(set(gram)) == 1:
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

    The top ``top_k`` are taken through :func:`suppress_nested` (D2), so one phrase cannot spend
    a document's whole budget on its own shingles.
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
        ranked[doc_id] = suppress_nested(scored, top_k=top_k)
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
# R3 — contrastive termhood: C-value nested discount + reference-corpus weirdness
# --------------------------------------------------------------------------- #


def _contiguous_subgrams(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    """Every *proper* contiguous sub-n-gram of ``tokens`` (deduplicated)."""
    subs: set[tuple[str, ...]] = set()
    n = len(tokens)
    for size in range(1, n):  # proper: size < n
        for i in range(n - size + 1):
            subs.add(tokens[i : i + size])
    return subs


def c_value_scores(term_freqs: dict[str, int]) -> dict[str, float]:
    """C-value nested-term termhood (Frantzi & Ananiadou), the ``+1`` length variant.

    ``C(a) = log2(|a| + 1) * (f(a) - mean_{b ⊋ a} f(b))`` where ``|a|`` is ``a``'s token
    length and ``b ⊋ a`` ranges over the *observed candidate terms* that strictly contain
    ``a`` as a contiguous sub-n-gram. A term appearing only inside longer terms scores ~0
    (its count equals its containers'), so nested fragments are discounted; a term that also
    occurs standalone keeps a positive score. The ``+1`` keeps unigrams eligible
    (``log2 2 = 1``, not ``log2 1 = 0``). Deterministic; O(V · ngram_max²).
    """
    tokens_of = {t: tuple(t.split()) for t in term_freqs}
    # For each candidate, collect the frequencies of the longer candidates containing it.
    super_freqs: dict[str, list[int]] = defaultdict(list)
    for b, b_tok in tokens_of.items():
        fb = term_freqs[b]
        for sub in _contiguous_subgrams(b_tok):
            key = " ".join(sub)
            if key in term_freqs:  # only real candidates count as nesting containers
                super_freqs[key].append(fb)
    scores: dict[str, float] = {}
    for term, tok in tokens_of.items():
        supers = super_freqs.get(term, [])
        mean_super = sum(supers) / len(supers) if supers else 0.0
        scores[term] = math.log2(len(tok) + 1) * (term_freqs[term] - mean_super)
    return scores


def _zipf(token: str) -> float:
    """General-English frequency of ``token`` on wordfreq's zipf scale (0=unseen … ~8)."""
    from wordfreq import zipf_frequency

    return float(zipf_frequency(token, "en"))


def weirdness(term: str, *, ref_ceiling: float) -> float:
    """Domain-specificity of ``term`` by contrast against general English (R3).

    Per token, ``max(0, ref_ceiling - zipf(token))`` — high when the token is rare in
    general English; an out-of-vocabulary technical token (``bm25`` → zipf 0) reaches the
    full ceiling, i.e. maximally weird (the desired smoothing of reference OOV). A phrase
    takes the **min** over its tokens, so it is only as domain-specific as its most-common
    word (``dense passage retrieval`` is bounded by ``passage``). Deterministic; ``wordfreq``
    ships its frequency table, so this is offline after install.
    """
    tokens = term.split()
    if not tokens:
        return 0.0
    return min(max(0.0, ref_ceiling - _zipf(tok)) for tok in tokens)


def contrastive_keywords(
    doc_terms: dict[str, list[str]],
    *,
    top_k: int,
    ref_ceiling: float,
    min_cvalue: float,
) -> list[ScoredKeyword]:
    """Select a corpus vocabulary by **termhood**: C-value nested discount * weirdness.

    Neither existing mode separates domain concepts from academic boilerplate: per-doc
    TF-IDF picks df≈1 per-paper cliques, and ``corpus_band``'s ``df·(1+ln tf)`` is monotone
    in df so it grabs the most-shared = most-generic terms. This mode scores by contrast
    against a reference corpus instead. A term is dropped if its C-value is ``≤ min_cvalue``
    (a nested fragment with no standalone occurrences) or its weirdness is 0 (pure common
    English). Otherwise ``score = (1 + ln tf_corpus) · weirdness(term)`` — the pre-registered
    R3 formula. ``tf`` is the corpus-total frequency, ``df`` the document frequency.
    Deterministic: ties break by term ascending.
    """
    per_doc_counts = {doc_id: Counter(terms) for doc_id, terms in doc_terms.items()}
    doc_freq: Counter[str] = Counter()
    total_tf: Counter[str] = Counter()
    for counts in per_doc_counts.values():
        for term, tf in counts.items():
            doc_freq[term] += 1
            total_tf[term] += tf

    cvals = c_value_scores(dict(total_tf))
    scored: list[ScoredKeyword] = []
    for term, tf in total_tf.items():
        if cvals[term] <= min_cvalue:  # nested fragment / no net standalone frequency
            continue
        w = weirdness(term, ref_ceiling=ref_ceiling)
        if w <= 0.0:  # pure common-English term (no contrast signal)
            continue
        score = (1.0 + math.log(tf)) * w
        scored.append(ScoredKeyword(term=term, score=score, tf=tf, df=doc_freq[term]))
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
    removed_orphans: int = 0  # extracted Keyword rows swept after a --force re-extract (R3)


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


def _sweep_orphan_keywords() -> int:
    """Delete extracted ``Keyword`` rows with no document links and no promoted concept (R3).

    A ``--force`` re-extract clears each doc's *extracted* links but leaves the ``Keyword``
    rows behind; a term that no longer appears in any document then lingers forever and
    pollutes ``seed_concepts`` candidates. This sweeps rows where ``source == "extracted"``,
    ``documents`` is empty, and the name does not match a promoted ``Concept`` label or
    ``ConceptAlias`` (never delete a curated concept's surface form). Returns the count.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, ConceptAlias, Keyword
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        curated = {
            label.casefold() for label in session.execute(select(Concept.label)).scalars() if label
        }
        curated |= {
            alias.casefold()
            for alias in session.execute(select(ConceptAlias.alias)).scalars()
            if alias
        }
        removed = 0
        rows = session.execute(select(Keyword).where(Keyword.source == "extracted")).scalars()
        for kw in rows:
            if kw.documents:  # still linked to a live document
                continue
            if kw.name.casefold() in curated:  # a promoted concept's surface form
                continue
            session.delete(kw)
            removed += 1
        return removed


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
    ref_ceiling: float = 8.0,
    min_cvalue: float = 0.0,
) -> KeywordExtractionResult:
    """Extract keyphrases (per-doc TF-IDF, corpus mid-DF band, or contrastive); persist opt.

    ``mode="per_doc"`` ranks each document's distinctive terms by TF-IDF (``top_k`` per doc).
    ``mode="corpus_band"`` selects ONE corpus vocabulary of shared terms whose document-frequency
    is in ``min_df .. floor(max_df_frac * N)`` (``top_k`` total), then links each to the documents
    it appears in. ``mode="contrastive"`` (R3) selects ONE corpus vocabulary by termhood —
    C-value nested discount * reference-corpus weirdness (``ref_ceiling`` / ``min_cvalue``) — the
    mode that separates domain concepts from academic boilerplate. Statistics are always computed
    over the whole cached corpus; ``document_id`` only restricts what is reported/written.
    Deterministic and free (no LLM). ``apply=False`` writes nothing (dry run). After a
    ``force`` re-extract, orphaned extracted ``Keyword`` rows are swept (R3).
    """
    corpus = load_document_texts()  # whole corpus → stable statistics
    filenames = {doc_id: fname for doc_id, fname, _ in corpus}
    # D1 runs *before* tokenisation, per document: furniture is defined by repeating across that
    # document's own pages, so it cannot be judged on the corpus-wide token stream. Doing it here
    # rather than at ingest keeps the primary chunk store untouched (Enrichment-Layer Pattern) —
    # the answer path still sees the document exactly as extracted.
    # D1 then D4: furniture first (it is judged per *page*, so it must see the whole document),
    # then the bibliography. Order matters only in that direction — cutting the references first
    # would remove pages that the furniture pass needs in order to count repetition.
    doc_terms = {
        doc_id: candidate_terms(
            tokenize(strip_reference_section(strip_page_furniture(text))),
            ngram_max=ngram_max,
            min_chars=min_chars,
        )
        for doc_id, _, text in corpus
    }

    ranked: dict[str, list[ScoredKeyword]]
    if mode in ("corpus_band", "contrastive"):
        if mode == "contrastive":
            selected = contrastive_keywords(
                doc_terms, top_k=top_k, ref_ceiling=ref_ceiling, min_cvalue=min_cvalue
            )
        else:
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
    # After a force re-extract, sweep extracted Keyword rows now orphaned (R3). Only when
    # writing over the whole corpus — a single-document run would wrongly sweep other docs'
    # terms that are momentarily unlinked in this pass.
    removed_orphans = 0
    if apply and force and document_id is None:
        removed_orphans = _sweep_orphan_keywords()

    if apply:
        log.info(
            "keywords_extracted",
            documents=len(docs),
            distinct=len(distinct),
            links_written=total_written,
            removed_orphans=removed_orphans,
        )
    return KeywordExtractionResult(
        docs=docs,
        n_documents=len(docs),
        n_distinct_keywords=len(distinct),
        total_written=total_written,
        removed_orphans=removed_orphans,
    )
