"""What a turn shows besides its answer: claims, records, carried, a reopened chat's (A8b).

The contract is `docs/specs/chat-interface-pass.md` § "The contract — A8b". Deterministic: a fake
pipeline, a temp SQLite, no model, no network, no paid call.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from doc_assistant import chat_controller, conversations, turn_details
from doc_assistant.chat_controller import ChatController, RagOverrides, Result, Session
from doc_assistant.db import session as session_mod
from doc_assistant.db.models import AnswerRecord, AnswerReview, Base, Folder
from doc_assistant.db.models import Document as DocRow
from doc_assistant.db.session import session_scope
from doc_assistant.provenance import adjudicate_claim


@pytest.fixture(autouse=True)
def _no_paid_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_assistant.llm.reviewer_available", lambda provider=None: False)


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    engine = create_engine(f"sqlite:///{tmp_path / 'details.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield engine
    engine.dispose()


class FakeRAG:
    """Two passages, the second ranked low; the answer cites one well, one weakly, one not."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.llm = types.SimpleNamespace(model="fake-model")
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"

    def chunk_count(self) -> int:
        return 2

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
        return question

    def retrieve_with_scores(self, query: str, top_k: int = 10, **kw: object) -> list[Any]:
        return [
            (Document(page_content="Strong passage.", metadata={"filename": "a.pdf"}), 0.9),
            (Document(page_content="Weak passage.", metadata={"filename": "b.pdf"}), 0.01),
        ]

    def stream_answer(self, question: str, docs: list[Document], **kw: object) -> Iterator[str]:
        yield self.answer


ANSWER = "Release falls with age [1]. Density drops too [2]. This causes slowing."


def _turn(controller: ChatController, session: Session, text: str, **kw: Any) -> Any:
    events = list(controller.handle_message(session, text, **kw))
    return next(e.result for e in events if isinstance(e, Result))


def test_a_turn_carries_every_claim_with_its_marker_and_sources(temp_db) -> None:
    result = _turn(ChatController(rag=FakeRAG(ANSWER)), Session(session_id="c1"), "q")
    assert [(c.n, c.marker, c.sources) for c in result.claims] == [
        (1, "ok", [1]),
        (2, "weak", [2]),
        (3, "unsupported", []),
    ]
    assert result.claims[0].text == "Release falls with age [1]."
    assert {c.decision for c in result.claims} == {"pending"}
    # The flagged list under the answer is unchanged: the weak and the uncited ones.
    assert [c.n for c in result.flagged_claims] == [2, 3]


def test_the_carried_counts_and_mode_ride_the_result(temp_db) -> None:
    controller, session = ChatController(rag=FakeRAG(ANSWER)), Session(session_id="c2")
    _turn(controller, session, "first")
    result = _turn(controller, session, "second")
    assert result.context is not None
    assert result.context["window"] == 200_000 and result.context["window_source"] == "documented"
    assert result.context["rewrite"]["history_used"] == 2
    assert result.context["answer"]["passages_used"] == 2
    assert result.chat_mode == "Research assistant"
    assert result.prompt_version


def test_evidence_only_and_records_answers_have_no_claims(temp_db) -> None:
    controller = ChatController(rag=FakeRAG(ANSWER))
    human = _turn(
        controller, Session(session_id="c3"), "q", overrides=RagOverrides(synthesis_mode="human")
    )
    assert human.claims == [] and human.records is None
    records = _turn(controller, Session(session_id="c4"), "how many documents are in my library?")
    assert records.route == "library" and records.claims == []
    assert records.records is not None and records.records.lookup


def _docs_in_folder(n: int) -> str:
    with session_scope() as s:
        folder = Folder(name="Thesis ch.2")
        s.add(folder)
        s.flush()
        for i in range(n):
            d = DocRow(
                filename=f"p{i}.pdf",
                source_original=f"C:/x/p{i}.pdf",
                doc_hash=f"h{i}",
                format="pdf",
                title=f"Paper {i}",
            )
            d.folders.append(folder)
            s.add(d)
        return str(folder.id)


def test_the_records_card_reads_its_rows_from_records_json(temp_db) -> None:
    _docs_in_folder(7)
    result = _turn(
        ChatController(rag=FakeRAG(ANSWER)), Session(session_id="c5"), "/folder Thesis ch.2"
    )
    view = result.records
    assert view is not None and view.via == "command"
    assert view.rows == 7
    assert len(view.items) == turn_details.RECORD_ROWS_SHOWN
    assert view.more == 7 - turn_details.RECORD_ROWS_SHOWN
    assert {i.kind for i in view.items} == {"document"}
    assert all(i.label.startswith("Paper ") for i in view.items)


def test_describe_records_names_each_kind_and_drops_what_is_gone(temp_db) -> None:
    folder_id = _docs_in_folder(1)
    raw = json.dumps({"lookup": "list_folders", "args": {}, "ids": [folder_id, "gone"], "rows": 2})
    view = turn_details.describe_records(raw)
    assert view is not None
    assert [(i.kind, i.label) for i in view.items] == [("folder", "Thesis ch.2")]
    assert view.rows == 2 and view.more == 0
    assert turn_details.describe_records(None) is None
    assert turn_details.describe_records("{not json") is None
    assert turn_details.describe_records(json.dumps({"no": "lookup"})) is None


def test_a_reopened_chat_shows_its_claims_decisions_and_review(temp_db) -> None:
    controller = ChatController(rag=FakeRAG(ANSWER))
    result = _turn(controller, Session(session_id="c6"), "q")
    adjudicate_claim(result.claims[1].claim_id, "accepted")
    adjudicate_claim(result.claims[2].claim_id, "edited", edited_text="It may cause slowing.")
    with session_scope() as s:
        s.add(
            AnswerReview(
                answer_record_id=result.record_id,
                reviewer_kind="llm_haiku",
                faithfulness=4,
                citation_density=3,
                hedging_adequacy=5,
                unsupported_claims_count=1,
                created_at=datetime(2026, 11, 20, 10, 0),
            )
        )
    detail = conversations.get_conversation("c6")
    assert detail is not None
    turn = detail.turns[0]
    assert [(c.n, c.marker, c.decision) for c in turn.claims] == [
        (1, "ok", "pending"),
        (2, "weak", "accepted"),
        (3, "unsupported", "edited"),
    ]
    assert turn.claims[2].edited_text == "It may cause slowing."
    assert turn.review is not None and turn.review.faithfulness == 4
    assert turn.review.unsupported_claims == 1
    assert turn.context is not None and turn.prompt_version == result.prompt_version


def test_a_pre_a8b_record_replays_with_nothing_new(temp_db) -> None:
    with session_scope() as s:
        s.add(
            AnswerRecord(
                session_id="old",
                query="q",
                answer="a",
                retrieved_chunks_json="[]",
                created_at=datetime(2026, 11, 1),
            )
        )
    turn = conversations.get_conversation("old").turns[0]  # type: ignore[union-attr]
    assert turn.claims == [] and turn.records is None and turn.context is None
    assert turn.review is None


def test_the_wire_carries_claims_records_and_context(temp_db) -> None:
    client = TestClient(create_app(controller=ChatController(rag=FakeRAG(ANSWER))))  # type: ignore[arg-type]
    stream = client.post("/api/chat", json={"text": "q", "session_id": "w1"}).text
    result = json.loads(stream.split("event: result")[1].split("data: ", 1)[1].split("\n")[0])
    assert [c["marker"] for c in result["claims"]] == ["ok", "weak", "unsupported"]
    assert result["claims"][0]["sources"] == [1]
    assert result["context"]["answer"]["passages_used"] == 2
    assert result["chat_mode"] == "Research assistant"
    detail = client.get("/api/conversations/w1").json()
    assert [c["decision"] for c in detail["turns"][0]["claims"]] == ["pending"] * 3
    assert detail["turns"][0]["records"] is None and detail["turns"][0]["review"] is None
