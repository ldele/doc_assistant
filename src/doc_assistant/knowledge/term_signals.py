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

**Not known.** How often each signal is right. Schwartz & Hearst accept any in-order spelling, so
"deep learning approaches (DNNs)" is a pair; ``initials`` says whether every capital of the short
form lands on the start of a word or on a capital of the long one, and is shown as evidence. It
is not a filter. Measured on a labelled sample before the app shows anything.

Zero LLM, zero network. Reads the sentences ``definitions.document_sentences`` yields — body
prose, the bibliography cut — so it reads what written forms and definition candidates read.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from doc_assistant.knowledge.concept_skeleton import (
    FormMatcher,
    form_key,
    form_matcher,
    spelling_key,
    surface_forms,
)

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


def abbreviation_pairs(sentence: str) -> list[AbbreviationPair]:
    """Every short form / long form pair ``sentence`` gives, in reading order.

    Two orders. *Long first*: "deep brain stimulation (DBS)" — the bracket opens on the short
    form, which may be followed by a citation ("(PDDL; Ghallab et al. (1998))"). *Short first*:
    "DPR (dense passage retriever)" — the bracket holds the long form and the word before it is
    the short one. Each ``long`` is a substring of ``sentence``."""
    out: list[AbbreviationPair] = []
    position = 0
    while True:
        open_ = sentence.find("(", position)
        if open_ < 0:
            return out
        position = open_ + 1
        close = sentence.find(")", position)
        if close < 0:
            return out
        head = re.split(r"[;,(]", sentence[position:close], maxsplit=1)[0].strip()
        before = sentence[:open_]
        pair: AbbreviationPair | None = None
        if is_short_form(head):
            pair = _pair(head, long_form_before(head, _window(before, head)))
        if pair is None and len(head.split()) >= 2:
            last = _TOKEN.findall(before)[-1:]
            short = last[0].strip(_WRAPPING) if last else ""
            if is_short_form(short):
                found = long_form_before(short, head)
                # The whole bracket is the long form, or it is a remark that happens to hold the
                # letters ("BM25 (a strong lexical baseline, much more than ...)").
                if found is not None and found.strip(_WRAPPING) == head.strip(_WRAPPING):
                    pair = _pair(short, found)
        if pair is not None:
            out.append(pair)


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

    @property
    def one_word(self) -> bool:
        return not re.search(r"[\s\-]", self.label.strip())


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
        with_mention: set[str] = set()
        after: Counter[str] = Counter()
        before: Counter[str] = Counter()
        if low_label:
            matcher = form_matcher(low_label, own.get(low_label))
            for document_id, spans in sentences.items():
                for _chunk_key, span, low in spans:
                    for text, start, end in _mentions(matcher, span, low):
                        mentions += 1
                        with_mention.add(document_id)
                        nxt = _WORD_AFTER.match(text, end)
                        if nxt:
                            word = nxt.group(1).lower()
                            if word == "et" and _AL_AFTER.match(text, nxt.end()):
                                et_al += 1
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
        )
    return out


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
