"""Checks on retrieved passages and on an answer that hold on any model (A2, docs/security.md S5).

A small local model can follow an instruction written inside a document however the prompt is
worded (A2's measurement, ``tests/eval/baselines/a2_prompt_fence_2026-11-16.md``). So the app
also checks the text itself, the same way whatever the model:

- :func:`addressed_to_ai` reads a passage for text written to an AI assistant rather than to a
  person: a chat turn's markup, a "system notice", a request to ignore instructions or to reveal
  the prompt. The answer surface marks such a source so the person sees it; nothing is blocked
  and nothing is removed from the document (ADR-043, inform-don't-block).
- :func:`links_in` lists the web addresses in a text. An answer's link counts as sourced only
  when its address is in one of the passages the answer was given; the renderer shows any other
  link as plain text, so a link a document told the model to write cannot be clicked.
- :func:`find_in_document` runs the same check over a whole document once, after extraction
  (A2b): `document_checks` keeps the result, and the Library shows it before any question.

The patterns are about how text addresses a model, not about any corpus: they are structural and
need no tuning to the library (the robustness contract). A pattern can still miss a phrasing or
mark an innocent passage that quotes one (a paper about prompt injection will); a mark is a
prompt to read the passage, not a verdict on it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: (label, pattern). A label says what kind of address it is; the renderer shows one plain
#: sentence for any of them, and the label is kept for the record and the tests.
_ADDRESS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "asks to ignore instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?"
            r"(previous|prior|above|earlier|preceding|your)\s+(instructions|prompts?|rules)",
            re.I,
        ),
    ),
    (
        "speaks to the assistant",
        re.compile(
            r"\b(note|notice|message|instruction|reminder)\s+(to|for)\s+(the|any|an)?\s*"
            r"(ai|assistant|language\s+model|llm|model|chatbot)\b",
            re.I,
        ),
    ),
    (
        "speaks to the assistant",
        re.compile(
            r"(^|[\n.:;!?]\s*)((dear|to\s+the|to\s+any)\s+(ai\s+)?(assistant|model|language\s+model|llm)"
            r"|ai\s+assistant|assistant)(\s+(generating|reading|answering)\s+[^,:\n]{0,40})?\s*:",
            re.I,
        ),
    ),
    (
        "imitates a system message",
        # A paper about language models names "system prompts" in ordinary prose, so the
        # phrase alone is not enough: it has to be addressed (followed by "to" or a colon).
        re.compile(
            r"\bsystem\s+(notice|message|instruction|prompt)s?\s*(to\b|:)"
            r"|\[\s*assistant\s+instruction\s*\]",
            re.I,
        ),
    ),
    (
        "imitates a system message",
        re.compile(r"(^|\n)\s*SYSTEM\s*:"),
    ),
    (
        "imitates a chat turn",
        re.compile(r"</?\s*(system|user|assistant)\s*>", re.I),
    ),
    (
        "asks for the prompt",
        re.compile(
            r"\b(print|reveal|repeat|output|show)\s+(the\s+)?(full\s+text\s+of\s+)?"
            r"(your|the)\s+(system\s+prompt|instructions|prompt)",
            re.I,
        ),
    ),
    (
        "dictates the answer",
        re.compile(
            r"\b(your|the)\s+answer\s+must\b|\bwhatever\s+the\s+user\s+asks\b|"
            r"\banswer\s+(entirely|only)\s+in\s+[a-z]+\b|\bfrom\s+now\s+on,?\s+(write|answer|reply|respond)\b|"
            r"\bdo\s+not\s+answer\s+the\s+question\b",
            re.I,
        ),
    ),
)

#: Names the patterns above: a document checked under another version is checked again. Derived
#: from the patterns themselves, so editing one cannot leave the old marks looking current.
CHECKS_VERSION: str = hashlib.sha256(
    "\n".join(f"{label}\t{p.pattern}\t{p.flags}" for label, p in _ADDRESS_PATTERNS).encode()
).hexdigest()[:12]

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.I)
_TRAILING = ".,;:!?*_"


def addressed_to_ai(text: str) -> list[str]:
    """The kinds of address to an AI the text carries, each label once, in pattern order.

    An empty list means no pattern matched, not that the text is safe."""
    found: list[str] = []
    for label, pattern in _ADDRESS_PATTERNS:
        if label not in found and pattern.search(text):
            found.append(label)
    return found


#: How much of a marked paragraph a finding keeps: enough to recognise it on the page. Structural
#: (a reading length), not tuned to any corpus.
EXCERPT_CHARS = 280

_PAGE_RE = re.compile(r"<!--\s*page:(\d+)\s*-->")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Finding:
    """One paragraph of a document that addresses an AI: where, what kind, and how it starts."""

    page: int | None
    labels: tuple[str, ...]
    excerpt: str


def find_in_document(text: str) -> list[Finding]:
    """Every paragraph of a document's extracted text that :func:`addressed_to_ai` marks.

    Paragraphs are split at blank lines, and each carries the page of the last
    ``<!-- page:N -->`` marker before it (``None`` for a format without pages). Read once per
    document after extraction, so the Library can show the mark before any question is asked."""
    found: list[Finding] = []
    # `split` with a capturing group alternates text and page numbers: [text, "1", text, "2", …].
    pieces = _PAGE_RE.split(text)
    pages: list[int | None] = [None] + [int(n) for n in pieces[1::2]]
    for page, segment in zip(pages, pieces[0::2], strict=True):
        for paragraph in _PARAGRAPH_SPLIT.split(segment):
            body = paragraph.strip()
            labels = addressed_to_ai(body) if body else []
            if labels:
                flat = " ".join(body.split())
                excerpt = (
                    flat
                    if len(flat) <= EXCERPT_CHARS
                    else flat[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"
                )
                found.append(Finding(page=page, labels=tuple(labels), excerpt=excerpt))
    return found


def links_in(text: str) -> set[str]:
    """The web addresses in ``text``, without the punctuation that ends a sentence."""
    return {m.group(0).rstrip(_TRAILING) for m in _URL_RE.finditer(text)}


def sourced_links(passages: Iterable[str]) -> list[str]:
    """Every web address that appears in the passages, sorted: the links an answer may carry."""
    out: set[str] = set()
    for p in passages:
        out |= links_in(p)
    return sorted(out)


def unsourced_links(answer: str, passages: Iterable[str]) -> list[str]:
    """The answer's web addresses that appear in none of the passages, sorted."""
    allowed = set(sourced_links(passages))
    return sorted(links_in(answer) - allowed)
