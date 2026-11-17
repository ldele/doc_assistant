"""What a turn sends the model, fitted to its window, and what gave way (ROADMAP 101, A5).

The contract is `docs/specs/context-handling.md` (accepted 2026-11-17). Every model call of a turn
is measured by one estimate (:func:`estimate_tokens`) against the model's window
(:func:`window_for`), fitted in a fixed order, and recorded (:class:`ContextRecord`); when
anything gave way the answer says so in one line written here, never by the model (:func:`note`).

**What gives way, in order** (the user's decision 2):

- the rewrite call: earlier messages, oldest first and whole; then the newest earlier answer,
  shortened from its end; and if the question alone does not fit, the rewrite is skipped;
- the answer call: the lowest-ranked passages, whole; the best passage is always kept, clipped
  from its end inside its fence if it alone does not fit.

The instructions and the question are never cut. Pure: this module calls no model and reads no
store, so the rules are tested without one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document

from doc_assistant import config

#: Floor on UTF-8 bytes per token, so the token count comes out high: an upper estimate leaves
#: room to spare rather than cut past the window. Structural, not tuned: byte-level BPE tokenizers
#: spend more than 3 bytes per token on English and about 3 on CJK, where a character is 3 bytes,
#: so counting bytes keeps the estimate above the true count for every script. Measured 2026-10-18
#: on an 87-paper prompt: 3.50 bytes per token (qwen2.5, qwen3.5), 3.63 (llama3.1), so the estimate
#: runs 14-17% high. Moved here from the wiki (KI-65) so the two cannot disagree.
BYTES_PER_TOKEN = 3

#: Anthropic documents a 200,000-token window for its Claude models. A table of one row: the
#: spec's open question 2 keeps a provider here only when the floor would be wrong for it.
_DOCUMENTED_WINDOWS: dict[str, int] = {"anthropic": 200_000}

#: Kept free for a reply where the app sets no ``max_tokens`` (the Ollama chat path). Structural:
#: the answer prompt asks for a cited answer, and a reply past this is rare and only loses its end.
REPLY_RESERVE_TOKENS = 1024

WindowSource = Literal["set", "server", "documented", "floor"]
_SHORTENED = " …"


def estimate_tokens(text: str) -> int:
    """An upper estimate of ``text``'s token count, with no tokenizer."""
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)


@dataclass(frozen=True)
class Window:
    tokens: int
    source: WindowSource


def window_for(provider: str | None, server_window: int | None = None) -> Window:
    """The window a call to ``provider`` is fitted to, and where that number came from.

    Ollama's is the one the app sets (``num_ctx``); a server's own stated window comes next;
    then a window the provider documents; else the floor, which the turn then says it used
    (decision 3, ADR-060 choice 7: an unknown window is never guessed)."""
    if provider == "ollama":
        return Window(config.OLLAMA_CONTEXT_TOKENS, "set")
    if server_window:
        return Window(int(server_window), "server")
    documented = _DOCUMENTED_WINDOWS.get(provider or "")
    if documented:
        return Window(documented, "documented")
    return Window(config.OLLAMA_CONTEXT_TOKENS, "floor")


@dataclass(frozen=True)
class RewriteFit:
    """The earlier messages the rewrite call may carry, and what gave way."""

    messages: list[dict[str, str]]
    estimated_tokens: int
    history_used: int
    history_dropped: int
    answer_shortened: bool = False
    skipped: bool = False


@dataclass(frozen=True)
class AnswerFit:
    """The passages the answer call may carry, and what gave way."""

    docs: list[Document]
    estimated_tokens: int
    passages_used: int
    passages_dropped: int
    top_passage_clipped: bool = False


def _clip(text: str, budget_tokens: int) -> str:
    """``text`` cut from its end to about ``budget_tokens``, on a word, marked as shortened."""
    room = max(0, budget_tokens * BYTES_PER_TOKEN - len(_SHORTENED.encode("utf-8")))
    cut = text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
    head, sep, _ = cut.rpartition(" ")
    return (head if sep else cut).rstrip() + _SHORTENED


def _message_tokens(message: dict[str, str]) -> int:
    # The rewrite prompt writes each message as "Role: content" on its own line.
    return estimate_tokens(f"{message['role'].capitalize()}: {message['content']}\n")


def fit_history(
    history: list[dict[str, str]],
    *,
    fixed_tokens: int,
    window: Window,
    reserve: int,
    cap: int = 6,
) -> RewriteFit:
    """The newest earlier messages, up to ``cap``, that fit beside the rewrite prompt.

    ``fixed_tokens`` is the rewrite prompt with the question and no history. Whole messages are
    taken newest first; if not even the newest fits, it is shortened; if the prompt alone does
    not fit, the rewrite is skipped."""
    budget = window.tokens - reserve - fixed_tokens
    recent = history[-cap:]
    if budget <= 0:
        return RewriteFit([], fixed_tokens, 0, len(recent), skipped=True)
    kept: list[dict[str, str]] = []
    used = 0
    for message in reversed(recent):
        cost = _message_tokens(message)
        if used + cost > budget:
            break
        kept.insert(0, message)
        used += cost
    shortened = False
    if not kept and recent:
        newest = recent[-1]
        overhead = _message_tokens({**newest, "content": ""})
        clipped = _clip(newest["content"], budget - overhead)
        kept = [{**newest, "content": clipped}]
        used = _message_tokens(kept[0])
        shortened = True
    return RewriteFit(
        messages=kept,
        estimated_tokens=fixed_tokens + used,
        history_used=len(kept),
        history_dropped=len(recent) - len(kept),
        answer_shortened=shortened,
    )


def fit_passages(
    docs: list[Document],
    *,
    fixed_tokens: int,
    window: Window,
    reserve: int,
    fenced_tokens: Any,
) -> AnswerFit:
    """The best-ranked passages that fit beside the answer prompt, in rank order.

    ``fixed_tokens`` is the answer prompt with the question and no passages; ``fenced_tokens``
    measures one passage as the prompt writes it, fence included (``(doc, n) -> int``). The best
    passage is always kept, clipped from its end if it alone does not fit."""
    budget = window.tokens - reserve - fixed_tokens
    kept: list[Document] = []
    used = 0
    for doc in docs:
        cost = int(fenced_tokens(doc, len(kept) + 1)) + 1  # +1: the blank line between sources
        if used + cost > budget:
            break
        kept.append(doc)
        used += cost
    clipped = False
    if not kept and docs:
        top = docs[0]
        overhead = int(fenced_tokens(Document(page_content="", metadata=dict(top.metadata)), 1))
        short = Document(
            page_content=_clip(top.page_content, max(1, budget - overhead)),
            metadata=dict(top.metadata),
        )
        kept = [short]
        used = int(fenced_tokens(short, 1))
        clipped = True
    return AnswerFit(
        docs=kept,
        estimated_tokens=fixed_tokens + used,
        passages_used=len(kept),
        passages_dropped=len(docs) - len(kept),
        top_passage_clipped=clipped,
    )


@dataclass
class ContextRecord:
    """What one turn's calls carried, for provenance (`AnswerRecord.context_json`)."""

    window: int
    window_source: WindowSource
    rewrite: dict[str, Any] | None = None
    answer: dict[str, Any] | None = field(default=None)

    @classmethod
    def of(
        cls, window: Window, rewrite: RewriteFit | None, answer: AnswerFit | None
    ) -> ContextRecord:
        def _rewrite(r: RewriteFit) -> dict[str, Any]:
            return {
                "estimated_tokens": r.estimated_tokens,
                "history_used": r.history_used,
                "history_dropped": r.history_dropped,
                "answer_shortened": r.answer_shortened,
                "skipped": r.skipped,
            }

        def _answer(a: AnswerFit) -> dict[str, Any]:
            return {
                "estimated_tokens": a.estimated_tokens,
                "passages_used": a.passages_used,
                "passages_dropped": a.passages_dropped,
                "top_passage_clipped": a.top_passage_clipped,
            }

        return cls(
            window=window.tokens,
            window_source=window.source,
            rewrite=_rewrite(rewrite) if rewrite is not None else None,
            answer=_answer(answer) if answer is not None else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def note(record: ContextRecord) -> str | None:
    """The one line an answer carries when anything gave way; ``None`` when nothing did."""
    left: list[str] = []
    rw, an = record.rewrite or {}, record.answer or {}
    if rw.get("history_dropped"):
        left.append(_count(rw["history_dropped"], "earlier message", "earlier messages"))
    if an.get("passages_dropped"):
        left.append(_count(an["passages_dropped"], "passage", "passages"))
    parts: list[str] = []
    if left:
        verb = "was" if len(left) == 1 and left[0].startswith("1 ") else "were"
        parts.append(f"{' and '.join(left)} {verb} left out to fit the model's window.")
    if rw.get("answer_shortened"):
        parts.append("The previous answer was shortened to fit.")
    if rw.get("skipped"):
        parts.append("The conversation did not fit, so the question was searched as asked.")
    if an.get("top_passage_clipped"):
        parts.append("The best passage was shortened to fit.")
    if record.window_source == "floor":
        parts.append(
            f"This server does not state its window, so the turn was kept to "
            f"{record.window:,} tokens."
        )
    return " ".join(parts) or None
