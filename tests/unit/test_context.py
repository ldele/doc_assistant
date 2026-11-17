"""What a turn sends the model, fitted to its window (A5, docs/specs/context-handling.md).

The pure rules first, then real turns through the controller on a fake pipeline with a small
window: the oldest history gives way first, a dropped passage leaves the sources too, an unknown
window is fitted to the floor and said, and the record reaches provenance.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document

from doc_assistant import config, context
from doc_assistant.chat_controller import ChatController, Result, Session, TurnResult
from doc_assistant.pipeline import fence_passage

# --- the estimate and the window -------------------------------------------------------------- #


def test_the_estimate_is_an_upper_one_by_bytes() -> None:
    assert context.estimate_tokens("") == 0
    assert context.estimate_tokens("abc") == 1
    assert context.estimate_tokens("é" * 3) == 2, "two bytes a character are counted as bytes"


def test_each_provider_gets_a_window_and_says_where_it_came_from() -> None:
    assert context.window_for("ollama") == context.Window(config.OLLAMA_CONTEXT_TOKENS, "set")
    assert context.window_for("anthropic").source == "documented"
    assert context.window_for("lmstudio", server_window=32_000) == context.Window(32_000, "server")
    unknown = context.window_for("lmstudio")
    assert unknown == context.Window(config.OLLAMA_CONTEXT_TOKENS, "floor")


# --- the rewrite's history ----------------------------------------------------------------- #


def _history(n: int, words: int = 40) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"message {i} " + "word " * words})
    return out


def test_the_oldest_messages_give_way_first() -> None:
    history = _history(6)
    one = context._message_tokens(history[-1])
    fit = context.fit_history(
        history, fixed_tokens=100, window=context.Window(100 + 50 + 3 * one, "set"), reserve=50
    )
    assert [m["content"].split()[1] for m in fit.messages] == ["3", "4", "5"]
    assert (fit.history_used, fit.history_dropped) == (3, 3)
    assert not fit.answer_shortened and not fit.skipped


def test_the_newest_answer_is_shortened_when_nothing_else_fits() -> None:
    history = _history(2, words=500)
    fit = context.fit_history(
        history, fixed_tokens=100, window=context.Window(300, "set"), reserve=50
    )
    assert fit.answer_shortened and fit.history_used == 1
    assert fit.messages[0]["content"].endswith(" …")
    assert fit.estimated_tokens <= 300 - 50


def test_a_question_too_big_for_the_window_skips_the_rewrite() -> None:
    fit = context.fit_history(
        _history(2), fixed_tokens=900, window=context.Window(1000, "set"), reserve=200
    )
    assert fit.skipped and fit.messages == []


def test_history_is_capped_at_six_messages_even_with_room() -> None:
    fit = context.fit_history(
        _history(10, words=2), fixed_tokens=10, window=context.Window(100_000, "set"), reserve=0
    )
    assert fit.history_used == 6 and fit.history_dropped == 0


# --- the answer's passages ----------------------------------------------------------------- #


def _passages(n: int, words: int = 200) -> list[Document]:
    return [
        Document(
            page_content=f"passage {i} " + "evidence " * words, metadata={"filename": f"p{i}.pdf"}
        )
        for i in range(n)
    ]


def _fenced(doc: Document, n: int) -> int:
    return context.estimate_tokens(fence_passage(doc, n))


def test_the_lowest_ranked_passages_give_way_first() -> None:
    docs = _passages(5)
    each = _fenced(docs[0], 1) + 1
    fit = context.fit_passages(
        docs,
        fixed_tokens=200,
        window=context.Window(200 + 100 + 2 * each + 5, "set"),
        reserve=100,
        fenced_tokens=_fenced,
    )
    assert [d.page_content.split()[1] for d in fit.docs] == ["0", "1"]
    assert (fit.passages_used, fit.passages_dropped) == (2, 3)


def test_the_best_passage_is_kept_clipped_when_it_alone_does_not_fit() -> None:
    docs = _passages(3, words=2000)
    fit = context.fit_passages(
        docs,
        fixed_tokens=200,
        window=context.Window(800, "set"),
        reserve=100,
        fenced_tokens=_fenced,
    )
    assert fit.top_passage_clipped and fit.passages_used == 1
    assert fit.docs[0].page_content.startswith("passage 0") and fit.docs[0].page_content.endswith(
        " …"
    )
    assert fit.estimated_tokens <= 800 - 100
    assert docs[0].page_content.endswith("evidence "), "the stored passage is not changed"


# --- the note ------------------------------------------------------------------------------ #


def test_the_note_says_what_gave_way_and_nothing_when_nothing_did() -> None:
    window = context.Window(8192, "set")
    quiet = context.ContextRecord.of(
        window,
        context.RewriteFit([], 10, 2, 0),
        context.AnswerFit([], 10, 5, 0),
    )
    assert context.note(quiet) is None
    busy = context.ContextRecord.of(
        window, context.RewriteFit([], 10, 4, 2), context.AnswerFit([], 10, 4, 1)
    )
    assert context.note(busy) == (
        "2 earlier messages and 1 passage were left out to fit the model's window."
    )
    floor = context.ContextRecord.of(
        context.Window(8192, "floor"), None, context.AnswerFit([], 1, 1, 0)
    )
    assert context.note(floor) == (
        "This server does not state its window, so the turn was kept to 8,192 tokens."
    )


# --- real turns through the controller --------------------------------------------------- #


class _Rag:
    def __init__(self, scored: list[tuple[Document, float]], provider: str = "ollama") -> None:
        self.scored = scored
        self.provider = provider
        self.model = "llama3.1:8b"
        self.llm = types.SimpleNamespace(model=self.model)
        self.rewrite_history: list[list[dict[str, str]]] = []
        self.answer_docs: list[list[Document]] = []

    def chunk_count(self) -> int:
        return 10

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
        self.rewrite_history.append(list(history))
        return question

    def retrieve_with_scores(
        self, query: str, top_k: int = 5, **_: object
    ) -> list[tuple[Document, float]]:
        return list(self.scored)

    def stream_answer(
        self, question: str, docs: list[Document], counter: object = None, llm: object = None
    ) -> Iterator[str]:
        self.answer_docs.append(list(docs))
        yield "An answer [1]."


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant import app_settings
    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    yield tmp_path
    engine.dispose()


def _scored(n: int, words: int = 300) -> list[tuple[Document, float]]:
    return [
        (
            Document(
                page_content=f"passage {i} " + "evidence " * words,
                metadata={
                    "document_id": f"d{i}",
                    "chunk_index": i,
                    "filename": f"p{i}.pdf",
                    "page": 1,
                },
            ),
            0.9 - i / 10,
        )
        for i in range(n)
    ]


def _final(controller: ChatController, session: Session, text: str) -> TurnResult:
    results = [e.result for e in controller.handle_message(session, text) if isinstance(e, Result)]
    assert len(results) == 1
    return results[0]


def test_a_long_conversation_keeps_its_newest_turns_and_says_what_it_left_out(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "OLLAMA_CONTEXT_TOKENS", 2600)
    rag = _Rag(_scored(2, words=60))
    controller = ChatController(rag=rag)  # type: ignore[arg-type]
    session = Session()
    session.history = _history(12, words=250)
    result = _final(controller, session, "and what about the last one?")
    (sent,) = rag.rewrite_history
    assert 0 < len(sent) < 6
    assert sent[-1] == session.history[-3], "the newest earlier answer is carried"  # 2 appended
    assert "earlier messages" in (result.citation_note_md or "")
    assert "Fitted to the window" in (result.citation_note_md or "")


def test_a_passage_left_out_of_the_prompt_is_left_out_of_the_sources(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "OLLAMA_CONTEXT_TOKENS", 1800)
    rag = _Rag(_scored(5, words=150))
    result = _final(ChatController(rag=rag), Session(), "what does it say?")  # type: ignore[arg-type]
    (given,) = rag.answer_docs
    assert 0 < len(given) < 5
    assert len(result.sources) == len(given), "no source the model never saw"
    assert "passage" in (result.citation_note_md or "")


def test_an_unknown_window_is_fitted_to_the_floor_and_said(temp_db: Path) -> None:
    rag = _Rag(_scored(2, words=20), provider="lmstudio")
    result = _final(ChatController(rag=rag), Session(), "a question")  # type: ignore[arg-type]
    assert "does not state its window" in (result.citation_note_md or "")


def test_a_turn_that_fits_says_nothing_and_records_its_window(temp_db: Path) -> None:
    from sqlalchemy import select

    from doc_assistant.db.models import AnswerRecord
    from doc_assistant.db.session import session_scope

    rag = _Rag(_scored(2, words=20))
    result = _final(ChatController(rag=rag), Session(), "a question")  # type: ignore[arg-type]
    assert "Fitted to the window" not in (result.citation_note_md or "")
    with session_scope() as s:
        (raw,) = s.execute(select(AnswerRecord.context_json)).scalars().all()
    record = json.loads(raw)
    assert record["window"] == config.OLLAMA_CONTEXT_TOKENS and record["window_source"] == "set"
    assert record["answer"]["passages_used"] == 2 and record["answer"]["passages_dropped"] == 0
    assert record["rewrite"] is None


def test_the_prompt_version_covers_the_rewrite_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.prompts import ChatPromptTemplate

    from doc_assistant.chat_controller import controller as controller_mod

    before = ChatController(rag=_Rag([]))._answer_template_hash  # type: ignore[arg-type]
    changed = ChatPromptTemplate.from_template("Rewrite it differently. {history} {question}")
    monkeypatch.setattr(controller_mod, "REWRITE_PROMPT", changed)
    after = ChatController(rag=_Rag([]))._answer_template_hash  # type: ignore[arg-type]
    assert before != after
