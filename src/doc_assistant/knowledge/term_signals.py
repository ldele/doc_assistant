"""What the prose says about a term, each signal with what it rests on (ADR-053 decision 4).

A term is a string the library uses; a concept is one the user has taken on, with a name and
forms (ADR-054). Four things can be read off the body prose about either, with no model:

- **spelled out / abbreviated** — the library gives a long form beside a short one ("deep brain
  stimulation (DBS)", "DPR (dense passage retriever)"). Found with Schwartz & Hearst's rule
  (2003): the letters of the short form occur in order in the words just before the bracket, the
  first of them at the start of a word, and the shortest such run of words is the long form. A
  label that is the short form gets its long form; a label that is the long form gets its short
  one.
- **the word beside it** — the content word that most often sits directly before or after the
  label ("pose" then "estimation"): the front or the tail of a longer term.
- **a cited author** — how often the label is followed by "et al.".
- **in prose** — how often, and in how many documents, the body prose uses the label at all. None
  means it lives in headings, tables, reference lists or figure descriptions only.

**A signal is offered, never applied.** Nothing here renames a concept, adds or marks a form,
promotes a term or deletes one: classifying a form is the user's act (ADR-054), and a long form
found here is a form to *offer*. Each signal keeps the counts it rests on and, for a pair, the
sentence it was read from, copied verbatim (ADR-043).

**Case is identity** (ADR-053 decision 3). The library spells ``CRE`` out as "cAMP response
element"; the concept it writes ``Cre`` is another word, so that long form is not offered to it.
A pair belongs to a label only when the label's own matcher accepts the short form as written.

**Measured** on the user's answers (``tests/eval/baselines/term_signals_2026-10-14.md``): the
long forms offered were right in their sentence 44 of 44 times; "et al." named a cited author
5 of 5 times; the word beside a label could not be judged as asked. Schwartz & Hearst accept any
in-order spelling, so "deep learning approaches (DNNs)" is a pair; ``initials`` says whether every
capital of the short form lands on the start of a word or on a capital of the long one. It is
evidence, never a filter: 10 of the 44 right pairs were without it.

**What the app shows** (:func:`load_concept_signals`, the concept panel): the offers, each with
its sentence and the two forms marked as written; a cited author with one "et al." sentence; and,
for a label the prose never uses, the places it does occur (:func:`where_it_occurs`). Not the
word beside a label.

Zero LLM, zero network. Reads the sentences ``definitions.document_sentences`` yields — body
prose, the bibliography cut — so it reads what written forms and definition candidates read.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from doc_assistant.knowledge.concept_skeleton import (
    FormMatcher,
    form_key,
    form_matcher,
    spelling_key,
    surface_forms,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A span of a string, ``(start, end)``: what a reader marks in it.
Span = tuple[int, int]

#: Schwartz & Hearst's bounds on a short form, in characters and in words. Part of the published
#: rule, not tuned on this library.
SHORT_MIN, SHORT_MAX, SHORT_WORDS = 2, 10, 2

# A capitalised word on its own in brackets is a name — "(Sporns, 2011)" — not a short form.
_CAPITALISED_WORD = re.compile(r"[A-Z][a-z]+")
# What may wrap a long form in the sentence without being part of it: quotes (straight and the
# four curly ones, built by code point so the source stays unambiguous), emphasis markup, brackets
# and the punctuation a clause ends on.
_CURLY_QUOTES = "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019))
_WRAPPING = " \t\r\n\"'*_[]{}.,;:" + _CURLY_QUOTES
_TOKEN = re.compile(r"\S+")
# The word joined to a mention by one space or a hyphen. Anything else between them — a comma, a
# bracket, a full stop — and the mention stands alone on that side. An apostrophe, straight or
# curly, stays inside its word ("Cajal's accessory body").
_LETTERS = "[A-Za-z][A-Za-z'" + chr(0x2019) + "]*"
_WORD_AFTER = re.compile(rf"[ \-]({_LETTERS})")
_WORD_BEFORE = re.compile(rf"({_LETTERS})[ \-]$")
_AL_AFTER = re.compile(r"\s+al\b")
#: How far back a neighbouring word is looked for. Longer than any word; a bound on work per
#: mention, not a setting.
_BEFORE_WINDOW = 48


# ============================================================
# Short form / long form pairs — Schwartz & Hearst, pure
# ============================================================


@dataclass(frozen=True)
class AbbreviationPair:
    """One short form and the long form one sentence gives beside it."""

    short: str  # as written
    long: str  # verbatim: a substring of the sentence
    initials: bool  # see :func:`spells_initials`


def is_short_form(text: str) -> bool:
    """Whether ``text`` can be a short form.

    Schwartz & Hearst's test — 2 to 10 characters, at most two words, a letter in it, a letter or
    digit first — and two structural additions: a capital or a digit somewhere, so "(see below)"
    is not a candidate, and not one capitalised word, which in brackets is a cited name."""
    if not SHORT_MIN <= len(text) <= SHORT_MAX or len(text.split()) > SHORT_WORDS:
        return False
    if not text[0].isalnum() or not any(c.isalpha() for c in text):
        return False
    if not any(c.isupper() or c.isdigit() for c in text):
        return False
    return _CAPITALISED_WORD.fullmatch(text) is None


def long_form_before(short: str, before: str) -> str | None:
    """The shortest tail of ``before`` that spells ``short`` in order, or ``None``.

    Schwartz & Hearst's search: the letters and digits of the short form are matched right to
    left, and the first of them must start a word. The result runs from the start of that word to
    the end of ``before``, untrimmed."""
    letters = [c.lower() for c in short if c.isalnum()]
    pos = len(before) - 1
    for n in range(len(letters) - 1, -1, -1):
        letter = letters[n]
        while pos >= 0 and (
            before[pos].lower() != letter or (n == 0 and pos > 0 and before[pos - 1].isalnum())
        ):
            pos -= 1
        if pos < 0:
            return None
        pos -= 1
    start = pos + 1
    while start > 0 and not before[start - 1].isspace():
        start -= 1
    return before[start:]


def spells_initials(short: str, long: str) -> bool:
    """Whether every capital of ``short`` lands, in order, on the start of a word or on a capital
    of ``long``; its lower-case letters and its digits need only follow in order.

    "DNNs" is spelled by "deep neural networks" and not by "deep learning approaches", which
    Schwartz & Hearst's rule accepts all the same. Evidence about a pair, never a filter."""
    pos = 0
    for c in short:
        if not c.isalnum():
            continue
        strong = c.isupper()
        found = -1
        for i in range(pos, len(long)):
            if long[i].lower() != c.lower():
                continue
            if strong and not (long[i].isupper() or i == 0 or not long[i - 1].isalnum()):
                continue
            found = i
            break
        if found < 0:
            return False
        pos = found + 1
    return True


def _window(before: str, short: str) -> str:
    """The words a long form may be found in: at most ``min(n + 5, 2n)`` of them, ``n`` the
    letters and digits of the short form (Schwartz & Hearst)."""
    n = sum(c.isalnum() for c in short)
    tokens = list(_TOKEN.finditer(before))
    limit = min(n + 5, n * 2)
    return before[tokens[-limit].start() :] if len(tokens) > limit else before


def _pair(short: str, candidate: str | None) -> AbbreviationPair | None:
    """``candidate`` trimmed to a long form and checked against ``short``, or ``None``."""
    if candidate is None:
        return None
    long = candidate.strip(_WRAPPING)
    if len(long) <= len(short):
        return None
    if any(word.strip(_WRAPPING).casefold() == short.casefold() for word in long.split()):
        return None  # "the BM25 score (BM25)": the short form is not its own long form
    return AbbreviationPair(short=short, long=long, initials=spells_initials(short, long))


def _lead(text: str) -> int:
    """How many wrapping characters ``text.strip(_WRAPPING)`` takes off its front."""
    return len(text) - len(text.lstrip(_WRAPPING))


def _scan_pairs(sentence: str) -> list[tuple[AbbreviationPair, Span, Span]]:
    """Every pair in ``sentence`` with where its short and its long form stand in it."""
    out: list[tuple[AbbreviationPair, Span, Span]] = []
    position = 0
    while True:
        open_ = sentence.find("(", position)
        if open_ < 0:
            return out
        position = open_ + 1
        close = sentence.find(")", position)
        if close < 0:
            return out
        first = re.split(r"[;,(]", sentence[position:close], maxsplit=1)[0]
        head = first.strip()
        head_at = position + len(first) - len(first.lstrip())
        before = sentence[:open_]
        found: tuple[AbbreviationPair, Span, Span] | None = None
        if is_short_form(head):
            candidate = long_form_before(head, _window(before, head))
            pair = _pair(head, candidate)
            if pair is not None and candidate is not None:
                # The candidate is a tail of `before`, which ends at the bracket.
                long_at = open_ - len(candidate) + _lead(candidate)
                found = (pair, (head_at, head_at + len(head)), (long_at, long_at + len(pair.long)))
        if found is None and len(head.split()) >= 2:
            last = list(_TOKEN.finditer(before))[-1:]
            short = last[0].group().strip(_WRAPPING) if last else ""
            if is_short_form(short):
                candidate = long_form_before(short, head)
                # The whole bracket is the long form, or it is a remark that happens to hold the
                # letters ("BM25 (a strong lexical baseline, much more than ...)").
                if candidate is not None and candidate.strip(_WRAPPING) == head.strip(_WRAPPING):
                    pair = _pair(short, candidate)
                    if pair is not None:
                        short_at = last[0].start() + _lead(last[0].group())
                        # The candidate is a tail of the bracket's head.
                        long_at = head_at + len(head) - len(candidate) + _lead(candidate)
                        found = (
                            pair,
                            (short_at, short_at + len(short)),
                            (long_at, long_at + len(pair.long)),
                        )
        if found is not None:
            out.append(found)


def abbreviation_pairs(sentence: str) -> list[AbbreviationPair]:
    """Every short form / long form pair ``sentence`` gives, in reading order.

    Two orders. *Long first*: "deep brain stimulation (DBS)" — the bracket opens on the short
    form, which may be followed by a citation ("(PDDL; Ghallab et al. (1998))"). *Short first*:
    "DPR (dense passage retriever)" — the bracket holds the long form and the word before it is
    the short one. Each ``long`` is a substring of ``sentence``."""
    return [pair for pair, _short, _long in _scan_pairs(sentence)]


@dataclass(frozen=True)
class SpelledOut:
    """A short form and a long form the library gives side by side, and where."""

    short: str  # as written
    long: str  # as the example sentence writes it
    documents: int  # documents that give this pair
    initials: bool  # some sentence writes the long form so that it spells the initials
    document_id: str  # the example: the first sentence, in document and reading order
    chunk_key: str
    sentence: str  # verbatim


def _long_key(long: str) -> str:
    """One key per long form: :func:`concept_skeleton.form_key`, so "state-space" and
    "state space" are one long form, as they are one form to the matcher (KI-63)."""
    return form_key(long.strip())


def spelled_out(
    sentences: Mapping[str, Sequence[tuple[str, str, str]]],
) -> list[SpelledOut]:
    """Every pair the library's prose gives, counted in documents. Pure.

    ``sentences`` is ``definitions.document_sentences``' shape: ``doc -> [(chunk_key, span,
    low)]``. A pair is one short form **as written** with one long form, case and spacing of the
    long form aside: ``CRE`` and ``Cre`` are two short forms. Sorted by documents, then by
    spelling, so two runs agree."""
    documents: dict[tuple[str, str], set[str]] = defaultdict(set)
    example: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    initials: dict[tuple[str, str], bool] = defaultdict(bool)
    for document_id in sorted(sentences):
        for chunk_key, span, _low in sentences[document_id]:
            if "(" not in span:
                continue
            for pair in abbreviation_pairs(span):
                key = (pair.short, _long_key(pair.long))
                documents[key].add(document_id)
                initials[key] = initials[key] or pair.initials
                example.setdefault(key, (pair.long, document_id, chunk_key, span))
    out = [
        SpelledOut(
            short=key[0],
            long=example[key][0],
            documents=len(docs),
            initials=initials[key],
            document_id=example[key][1],
            chunk_key=example[key][2],
            sentence=example[key][3],
        )
        for key, docs in documents.items()
    ]
    out.sort(key=lambda p: (-p.documents, p.short.casefold(), _long_key(p.long), p.short))
    return out


# ============================================================
# Signals per term — pure
# ============================================================


@dataclass(frozen=True)
class FormOffer:
    """A form the library pairs with one of a term's own, offered and never added."""

    form: str  # the form found, as the library writes it
    beside: str  # which of the term's forms it stands beside (casefolded)
    beside_written: str  # that form as the example sentence writes it
    kind: str  # "long": the term's form is the short one · "short": it is the long one
    documents: int
    initials: bool
    known: bool  # already one of the term's forms
    document_id: str
    chunk_key: str
    sentence: str  # verbatim


@dataclass(frozen=True)
class Neighbour:
    """The content word most often joined to a label on one side."""

    word: str  # lower-cased
    count: int  # mentions with this word on that side


@dataclass(frozen=True)
class Example:
    """A piece of the library's text a signal was read from, and what in it to mark."""

    document_id: str
    chunk_key: str
    text: str  # verbatim: a substring of the chunk's text
    marks: tuple[Span, ...] = ()  # spans of ``text``, as written; none when they cannot be placed


@dataclass(frozen=True)
class TermSignals:
    """What the body prose says about one term."""

    concept_id: str
    label: str
    mentions: int  # whole-word uses of the label
    documents: int  # documents with at least one — the spread
    offers: tuple[FormOffer, ...] = ()
    after: Neighbour | None = None
    before: Neighbour | None = None
    et_al: int = 0  # mentions followed by "et al."
    et_al_example: Example | None = None  # the first such sentence, in document order

    @property
    def one_word(self) -> bool:
        return not re.search(r"[\s\-]", self.label.strip())

    @property
    def cited(self) -> bool:
        """Mostly a cited author's name: "et al." follows at least half of its mentions."""
        return majority(self.et_al, self.mentions)


def majority(count: int, mentions: int) -> bool:
    """At least half of the mentions — a majority, not a tuned share. ``False`` for none."""
    return mentions > 0 and count * 2 >= mentions


def offer_example(offer: FormOffer) -> Example:
    """The sentence an offer was read from, its two forms marked (:func:`offer_marks`)."""
    return Example(
        document_id=offer.document_id,
        chunk_key=offer.chunk_key,
        text=offer.sentence,
        marks=offer_marks(offer),
    )


def offer_marks(offer: FormOffer) -> tuple[Span, ...]:
    """Where the offer's two forms stand in its sentence, short and long, in reading order.

    Read again from the sentence by the rule that found the pair, so a mark lands on the short
    form *as written* and on a word edge — never on the ``co`` inside "communication" (the
    review page's fault, baseline §1). Empty if the sentence no longer gives the pair."""
    if offer.kind == "long":
        short, long = offer.beside_written, offer.form
    else:
        short, long = offer.form, offer.beside_written
    for pair, short_at, long_at in _scan_pairs(offer.sentence):
        if pair.short == short and pair.long == long:
            return tuple(sorted((short_at, long_at)))
    return ()


def _mentions(matcher: FormMatcher, text: str, low: str) -> Iterator[tuple[str, int, int]]:
    """Each mention as ``(string, start, end)``. The string is ``low`` for a case-folded form
    and ``text`` for a cased one: casefolding can change a string's length, so a span is only
    good in the string it was found in."""
    if matcher.probe not in low:
        return
    if matcher.cased is None:
        for m in matcher.folded.finditer(low):
            yield low, m.start(), m.end()
        return
    anycase, key = matcher.cased
    for m in anycase.finditer(text):
        if spelling_key(m[0]) == key:
            yield text, m.start(), m.end()


def _offers(
    forms: Sequence[str],
    spellings: Mapping[str, str],
    by_short: Mapping[str, Sequence[SpelledOut]],
    by_long: Mapping[str, Sequence[SpelledOut]],
) -> tuple[FormOffer, ...]:
    """The pairs that hold one of ``forms``, as offers. ``spellings`` maps a form to how the
    library writes it. A pair is offered once: when both its sides are already the term's forms
    it is reported from the side met first, the label's before an alias's."""
    known = {_long_key(form) for form in forms}
    out: list[FormOffer] = []
    seen: set[tuple[str, str]] = set()
    for form in forms:
        matcher = form_matcher(form, spellings.get(form))
        for pair in by_short.get(form, ()):
            if not matcher.search(pair.short, pair.short.casefold()):
                continue  # written in another case: another word (CRE is not Cre)
            key = (pair.short.casefold(), _long_key(pair.long))
            if key not in seen:
                seen.add(key)
                out.append(_offer(pair, pair.long, form, "long", known))
        for pair in by_long.get(_long_key(form), ()):
            key = (pair.short.casefold(), _long_key(pair.long))
            if key not in seen:
                seen.add(key)
                out.append(_offer(pair, pair.short, form, "short", known))
    out.sort(key=lambda o: (-o.documents, o.kind, o.form.casefold(), o.form))
    return tuple(out)


def _offer(pair: SpelledOut, form: str, beside: str, kind: str, known: set[str]) -> FormOffer:
    return FormOffer(
        form=form,
        beside=beside,
        beside_written=pair.short if kind == "long" else pair.long,
        kind=kind,
        documents=pair.documents,
        initials=pair.initials,
        known=_long_key(form) in known,
        document_id=pair.document_id,
        chunk_key=pair.chunk_key,
        sentence=pair.sentence,
    )


def _is_content(word: str, stopwords: frozenset[str]) -> bool:
    """A neighbour worth counting: not a function word, and not a single letter (the "G" of
    "VSV-G", a list marker, a stray initial)."""
    return len(word) > 1 and word not in stopwords


def _top(counts: Counter[str]) -> Neighbour | None:
    if not counts:
        return None
    word, count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return Neighbour(word=word, count=count)


def term_signals(
    concepts: Sequence[tuple[str, str]],
    aliases: Mapping[str, list[str]],
    chunks: Iterable[tuple[str, str, str]],
    written: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, TermSignals]:
    """The signals of every term in ``concepts``, keyed by concept id. Pure.

    ``concepts`` = ``(concept_id, label)``; ``aliases`` maps ``concept_id`` → its other forms;
    ``chunks`` = ``(chunk_key, document_id, text)``; ``written`` maps ``(concept_id, form)`` → how
    the library writes that form (``written_forms.spelling_map``). Mentions, neighbours and
    "et al." are counted on the **label**; pairs are looked up for every form, so a long form the
    user already added is found and marked ``known``. A term the prose never uses still has an
    entry, with zeros — an empty library yields one per concept (the robustness contract).
    """
    from doc_assistant.knowledge.definitions import document_sentences
    from doc_assistant.knowledge.keywords import STOPWORDS

    sentences = document_sentences(chunks)
    by_short: dict[str, list[SpelledOut]] = defaultdict(list)
    by_long: dict[str, list[SpelledOut]] = defaultdict(list)
    for pair in spelled_out(sentences):
        by_short[pair.short.casefold()].append(pair)
        by_long[_long_key(pair.long)].append(pair)

    spellings = written or {}
    out: dict[str, TermSignals] = {}
    for concept_id, label in concepts:
        forms = surface_forms(label, list(aliases.get(concept_id, [])))
        own = {
            form: spellings[(concept_id, form)]
            for form in forms
            if (concept_id, form) in spellings
        }
        low_label = label.strip().casefold()
        mentions = et_al = 0
        cited: Example | None = None
        with_mention: set[str] = set()
        after: Counter[str] = Counter()
        before: Counter[str] = Counter()
        if low_label:
            matcher = form_matcher(low_label, own.get(low_label))
            # Document order, so the example sentence is the same on every run.
            for document_id, spans in sorted(sentences.items()):
                for chunk_key, span, low in spans:
                    for text, start, end in _mentions(matcher, span, low):
                        mentions += 1
                        with_mention.add(document_id)
                        nxt = _WORD_AFTER.match(text, end)
                        if nxt:
                            word = nxt.group(1).lower()
                            if word == "et" and _AL_AFTER.match(text, nxt.end()):
                                et_al += 1
                                if cited is None:
                                    cited = Example(
                                        document_id=document_id,
                                        chunk_key=chunk_key,
                                        text=span,
                                        marks=_marks_in(span, text, ((start, end),)),
                                    )
                            elif _is_content(word, STOPWORDS):
                                after[word] += 1
                        prv = _WORD_BEFORE.search(text, max(0, start - _BEFORE_WINDOW), start)
                        if prv and _is_content(prv.group(1).lower(), STOPWORDS):
                            before[prv.group(1).lower()] += 1
        out[concept_id] = TermSignals(
            concept_id=concept_id,
            label=label,
            mentions=mentions,
            documents=len(with_mention),
            offers=_offers(forms, own, by_short, by_long),
            after=_top(after),
            before=_top(before),
            et_al=et_al,
            et_al_example=cited,
        )
    return out


def _marks_in(text: str, found_in: str, spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """``spans`` as marks on ``text``, given the string they were found in (:func:`_mentions`).

    A span found in the casefolded copy is good in ``text`` only when casefolding kept the
    length ("ß" and a ligature do not); then no mark is given rather than a misplaced one."""
    return spans if found_in is text or len(found_in) == len(text) else ()


def signal_payload(signals: TermSignals) -> dict[str, Any]:
    """One term's signals as plain data, for a runner's JSON or an API response."""
    return {
        "concept_id": signals.concept_id,
        "label": signals.label,
        "one_word": signals.one_word,
        "mentions": signals.mentions,
        "documents": signals.documents,
        "et_al": signals.et_al,
        "after": asdict(signals.after) if signals.after else None,
        "before": asdict(signals.before) if signals.before else None,
        "offers": [asdict(offer) for offer in signals.offers],
    }


# ============================================================
# Where a label occurs, when the prose never uses it — pure
# ============================================================


#: The places a label the sentences never use can occur, as the keyword extractor read it.
#: ``sentence``: a body sentence that writes it another way — another case, or other characters
#: between its words ("Comput. Vis.") — which no reader counts as a use. ``elsewhere``: a line of
#: prose that is not a sentence — a list item, a short line, a page header or footer, code.
PLACES = ("sentence", "table", "heading", "figure", "references", "elsewhere")
#: Characters kept on each side of a mention in a place's example. A bound on what is shown.
_CONTEXT = 80
#: What joins a figure's own text to the copy of its citing passage in the figure's parent
#: (``ingest.figures.figure_parent_text``; a unit test holds the two together). Not imported:
#: the ingest package loads the embedding stack, and this module runs on a panel open.
_FIGURE_CONTEXT_JOIN = "\n\n---\n\n"


@dataclass(frozen=True)
class Place:
    """Where a label occurs, how often, and one piece of text that shows it."""

    place: str  # one of PLACES
    mentions: int
    documents: int
    example: Example


def _keyword_pattern(label: str) -> re.Pattern[str] | None:
    """The label as the keyword extractor finds it in casefolded text: its words in order, any run
    of characters that are not letters or digits between them (``keywords.tokenize`` splits there,
    and a keyword is adjacent tokens). Looser than the mention matcher on purpose: it answers
    where a keyword row came from, not whether the library uses the term."""
    words = re.findall(r"[a-z0-9]+", label.casefold())
    if not words:
        return None
    return re.compile(
        r"(?<![a-z0-9])" + "[^a-z0-9]+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    )


def _line_place(text: str, start: int) -> str:
    """``table``, ``heading`` or ``elsewhere`` for a mention at ``start``, from how its line
    begins in the extracted markdown."""
    line = text[text.rfind("\n", 0, start) + 1 : start + 1].lstrip()
    if line.startswith("|"):
        return "table"
    if line.startswith("#"):
        return "heading"
    return "elsewhere"


def _sentence_starts(body: str) -> tuple[list[int], list[int]]:
    """Where each body sentence of a chunk starts and ends (``definitions.sentence_spans``, the
    sentences every reader reads), as two sorted lists for a bisect."""
    from doc_assistant.knowledge.definitions import sentence_spans

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for span in sentence_spans(body):
        at = body.find(span, cursor)
        if at < 0:
            continue
        starts.append(at)
        ends.append(at + len(span))
        cursor = at + len(span)
    return starts, ends


def _parent_index(chunk_key: str) -> int:
    try:
        return int(chunk_key.rsplit(":p", 1)[1])
    except (IndexError, ValueError):
        return 0


def where_it_occurs(
    label: str,
    prose: Iterable[tuple[str, str, str]],
    figures: Iterable[tuple[str, str, str]] = (),
) -> tuple[Place, ...]:
    """Every place ``label`` occurs as the keyword extractor read it, most mentions first. Pure.

    For a label no body sentence uses, it says where the string comes from. ``prose`` and
    ``figures`` are ``(chunk_key, document_id, text)`` parents. A prose mention past the
    bibliography cut (``definitions.body_chunks``, the cut every reader shares) is in the
    references; before it, a mention inside a body sentence is one written another way, and
    otherwise its line says whether it is a table row, a heading, or a line that is not a
    sentence. A figure mention is one in the figure's own text — its caption or the model's
    description — never in the copy of the citing passage its parent carries.
    """
    import bisect

    from doc_assistant.knowledge.definitions import body_chunks

    pattern = _keyword_pattern(label)
    if pattern is None:
        return ()
    counts: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, Example] = {}

    def note(place: str, document_id: str, chunk_key: str, text: str, low: str, hit: Span) -> None:
        start, end = hit
        counts[place] += 1
        documents[place].add(document_id)
        if place not in examples:
            # Where casefolding changed the length the window is approximate and goes unmarked.
            a, b = max(0, start - _CONTEXT), end + _CONTEXT
            examples[place] = Example(
                document_id=document_id,
                chunk_key=chunk_key,
                text=text[a:b],
                marks=_marks_in(text, low, ((start - a, end - a),)),
            )

    by_document: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for chunk_key, document_id, text in prose:
        by_document[document_id].append((_parent_index(chunk_key), chunk_key, text))
    for document_id in sorted(by_document):
        rows = sorted(by_document[document_id])
        kept = {key: len(body) for key, body in body_chunks([(k, t) for _, k, t in rows])}
        for _, chunk_key, text in rows:
            low = text.casefold()
            body_end = kept.get(chunk_key, 0)
            sentences: tuple[list[int], list[int]] | None = None
            for m in pattern.finditer(low):
                if m.start() >= body_end:
                    place = "references"
                else:
                    if sentences is None:
                        sentences = _sentence_starts(text[:body_end])
                    starts, ends = sentences
                    i = bisect.bisect_right(starts, m.start()) - 1
                    inside = i >= 0 and m.start() < ends[i]
                    place = "sentence" if inside else _line_place(low, m.start())
                note(place, document_id, chunk_key, text, low, (m.start(), m.end()))
    for chunk_key, document_id, text in sorted(figures):
        own = text.split(_FIGURE_CONTEXT_JOIN, 1)[0]
        low = own.casefold()
        for m in pattern.finditer(low):
            note("figure", document_id, chunk_key, own, low, (m.start(), m.end()))
    return tuple(
        Place(place=p, mentions=counts[p], documents=len(documents[p]), example=examples[p])
        for p in sorted(counts, key=lambda p: (-counts[p], PLACES.index(p)))
    )


# ============================================================
# Impure boundary — the vocabulary and the prose in; nothing out
# ============================================================


def load_term_signals() -> dict[str, TermSignals]:
    """The signals of the whole vocabulary, concepts and terms, from the library as it stands.

    Reads the vocabulary, the written forms of the last graph build and the prose parents. Writes
    nothing: there is no table for a signal, and no caller that applies one."""
    from doc_assistant.knowledge.concept_skeleton import load_presence_inputs
    from doc_assistant.knowledge.written_forms import load_vocabulary, load_written_forms

    concepts, aliases = load_vocabulary()
    return term_signals(concepts, aliases, load_presence_inputs(), load_written_forms())


#: The most documents one concept's signals read: those with the most matching blocks. A bound on
#: one panel open — about 15 ms a document on the working library (2026-10-17), so a few seconds
#: at the bound — not a setting tuned to what is found. Above it, the answer says how many were
#: read of how many, and every count is a count over the documents read.
SIGNAL_DOCUMENTS = 200


@dataclass(frozen=True)
class ConceptSignals:
    """What the concept panel shows about one row's name: the measured signals only."""

    concept_id: str
    label: str
    # False when the keyword index is missing: nothing was read, which is not "never used".
    available: bool
    documents_read: int = 0
    # Documents the keyword index matched. Its prefix query over-selects ("combination" for
    # `comb`), so this bounds the documents that use the label; it is not their number.
    documents_matched: int = 0
    signals: TermSignals | None = None
    places: tuple[Place, ...] = ()  # only for a label no sentence uses
    titles: Mapping[str, str] = field(default_factory=dict)  # document_id -> title or filename

    @property
    def silent(self) -> bool:
        """No sentence of the documents read uses the label."""
        return self.available and self.signals is not None and self.signals.mentions == 0


def load_concept_signals(
    concept_id: str, *, index_file: Path | None = None
) -> ConceptSignals | None:
    """One row's signals for the concept panel, or ``None`` for an unknown id or a field node.

    Concepts and terms alike: a signal is offered, so a term has one as much as a concept. The
    keyword index picks the documents that mention any of the row's forms, a figure's text
    included (:func:`definitions.parents_mentioning`), at most :data:`SIGNAL_DOCUMENTS` of them,
    and each is read whole, so the bibliography cut and the counts are a full read's for those
    documents. Writes nothing."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, ConceptAlias, Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import parents_mentioning
    from doc_assistant.knowledge.written_forms import load_written_forms

    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None or concept.kind != "concept":
            return None
        label = str(concept.label)
        aliases = [
            str(alias)
            for alias in session.execute(
                select(ConceptAlias.alias).where(ConceptAlias.concept_id == concept_id)
            ).scalars()
        ]
    read = parents_mentioning(
        surface_forms(label, aliases), index_file=index_file, top_docs=SIGNAL_DOCUMENTS
    )
    if read is None:
        return ConceptSignals(concept_id=concept_id, label=label, available=False)
    written = load_written_forms([concept_id])
    signals = term_signals([(concept_id, label)], {concept_id: aliases}, read.prose, written)[
        concept_id
    ]
    places = where_it_occurs(label, read.prose, read.figures) if signals.mentions == 0 else ()
    shown = {o.document_id for o in signals.offers} | {p.example.document_id for p in places}
    if signals.et_al_example is not None:
        shown.add(signals.et_al_example.document_id)
    with session_scope() as session:
        titles = {
            str(i): str(t or f)
            for i, t, f in session.execute(
                select(Document.id, Document.title, Document.filename).where(
                    Document.id.in_(sorted(shown))
                )
            ).all()
        }
    return ConceptSignals(
        concept_id=concept_id,
        label=label,
        available=True,
        documents_read=read.documents_read,
        documents_matched=read.documents_matched,
        signals=signals,
        places=places,
        titles=titles,
    )
