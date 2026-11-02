"""A whole answer through an OpenAI-compatible server, at $0 (ADR-060, its tests 4 and 7).

The real ``ChatController``, the real ``pipeline.build_chat_model`` and
``RAGPipeline.stream_answer`` (so the real answer prompt and the real ``CompatibleChatModel``),
provenance on a temp SQLite, and the cost line, against the stdlib fake server on 127.0.0.1. One
of the server's route families demands a key, standing in for a cloud preset; every preset row is
driven through it with only its address changed.

The retriever is a stand-in that returns two passages: what is proved here is the provider path.
Retrieval over a real library is proved in the app (the ADR's test 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from doc_assistant import app_settings, chat_controller, credentials, providers
from doc_assistant.chat_controller import ChatController, Result, Session, Token
from doc_assistant.pipeline import RAGPipeline, build_chat_model
from tests.fixtures.compatible_server import FakeCompatibleServer


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod, "_SessionLocal", sessionmaker(bind=engine, autoflush=False, future=True)
    )
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    yield engine
    engine.dispose()


_PASSAGES = [
    (
        Document(
            page_content="Retrieval-augmented generation conditions the answer on passages.",
            metadata={"document_id": "d1", "chunk_index": 0, "filename": "rag.pdf", "page": 2},
        ),
        0.93,
    ),
    (
        Document(
            page_content="Grounding in retrieved text reduces unsupported claims.",
            metadata={"document_id": "d1", "chunk_index": 3, "filename": "rag.pdf", "page": 5},
        ),
        0.88,
    ),
]


class _Pipeline:
    """The real answer path of ``RAGPipeline`` over a stand-in retriever."""

    stream_answer = RAGPipeline.stream_answer
    rewrite = RAGPipeline.rewrite
    set_chat_model = RAGPipeline.set_chat_model

    def __init__(self, provider: str, model: str) -> None:
        self.set_chat_model(provider, model)

    def chunk_count(self) -> int:
        return len(_PASSAGES)

    def retrieve_with_scores(self, query: str, top_k: int = 10, **_: object):
        return list(_PASSAGES)


def _turn(provider: str, model: str = "fake-model"):
    controller = ChatController(rag=_Pipeline(provider, model))  # type: ignore[arg-type]
    session = Session()
    events = list(
        controller.handle_message(session, "What does retrieval add to a model's answers?")
    )
    tokens = "".join(e.text for e in events if isinstance(e, Token))
    (result,) = [e.result for e in events if isinstance(e, Result)]
    return tokens, result


def _route_of_record(engine, record_id: str) -> dict:
    from sqlalchemy import text

    with engine.connect() as conn:
        raw = conn.execute(
            text("SELECT answer_route_json, model_name FROM answer_records WHERE id = :i"),
            {"i": record_id},
        ).one()
    return {**json.loads(raw[0]), "model_name": raw[1]}


def test_a_cited_answer_through_a_server_of_your_own(temp_db) -> None:
    with FakeCompatibleServer() as server:
        sid = providers.save_server("Lab server", server.url("open")).id
        tokens, result = _turn(sid)
    assert "[1]" in tokens and "[2]" in tokens
    assert result.answer == tokens
    # The citation audit found nothing to say: both numbers point at a retrieved passage.
    assert "Citation check" not in result.citation_note_md
    # What was sent: the minimal body, the real answer prompt with the passages, no key.
    (request,) = server.chat_requests()
    assert set(request["body"]) == {"model", "messages", "stream"}
    prompt = request["body"]["messages"][0]["content"]
    assert "rag.pdf" in prompt and "What does retrieval add" in prompt
    assert "authorization" not in request["headers"]
    # Where the text went and who bills, in the provenance and under the answer.
    route = _route_of_record(temp_db, result.record_id)
    assert route["host"] == "127.0.0.1" and route["reach"] == "this_computer"
    assert route["billing"] == "unknown" and route["model_name"] == "fake-model"
    assert result.usage.cost_usd is None
    assert result.usage.billing == "Billing not known."
    assert "Stays on this computer" in result.usage_md
    assert "$" not in result.usage_md


def test_the_tick_turns_the_line_into_your_word(temp_db) -> None:
    with FakeCompatibleServer() as server:
        sid = providers.save_server("Lab server", server.url("open")).id
        app_settings.set_not_billed(sid, True)
        _, result = _turn(sid)
    assert result.usage.billing == "Not billed (your word)."


@pytest.mark.parametrize(
    "row", [p for p in providers.PRESETS if p.billed_by is not None], ids=lambda p: p.id
)
def test_every_cloud_preset_through_the_fake_server(
    temp_db, monkeypatch: pytest.MonkeyPatch, row: providers.Preset
) -> None:
    with FakeCompatibleServer(key="vendor-key") as server:
        from dataclasses import replace

        monkeypatch.setitem(
            providers._PRESETS_BY_ID, row.id, replace(row, address=server.url("cloud"))
        )
        credentials.set_bound_key(row.id, "vendor-key", server.url("cloud"))
        tokens, result = _turn(row.id)
    assert "[1]" in tokens
    (request,) = server.chat_requests()
    assert request["headers"]["authorization"] == "Bearer vendor-key"
    assert set(request["body"]) == {"model", "messages", "stream"}
    # A vendor bills it, and Provenote names the vendor rather than guessing a price.
    assert result.usage.billing == f"Billed by {row.billed_by}; Provenote does not know its price."
    assert result.usage.cost_usd is None
    assert _route_of_record(temp_db, result.record_id)["billed_by"] == row.billed_by


def test_a_key_saved_for_another_address_is_not_sent(
    temp_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    with FakeCompatibleServer(key="vendor-key") as server:
        row = providers._PRESETS_BY_ID["groq"]
        monkeypatch.setitem(
            providers._PRESETS_BY_ID, "groq", replace(row, address=server.url("cloud"))
        )
        credentials.set_bound_key("groq", "vendor-key", "https://api.groq.com/openai/v1")
        model = build_chat_model("groq", "fake-model")
        assert model.api_key is None
        events = list(
            ChatController(rag=_Pipeline("groq", "fake-model")).handle_message(  # type: ignore[arg-type]
                Session(), "q"
            )
        )
    (failure,) = [e for e in events if type(e).__name__ == "Failure"]
    assert "No answer from Groq" in failure.message and "401" in failure.message


def test_a_server_that_reads_half_the_prompt_is_warned_about(temp_db) -> None:
    # KI-66's case on a server that reports usage: far fewer prompt tokens than were sent.
    with FakeCompatibleServer(usage=(40, 20)) as server:
        sid = providers.save_server("Lab server", server.url("open")).id
        _, result = _turn(sid)
    assert "Prompt cut?" in result.citation_note_md


def test_a_server_that_is_down_ends_the_turn_with_a_sentence(temp_db) -> None:
    sid = providers.save_server("Lab server", "http://127.0.0.1:9/v1").id
    events = list(ChatController(rag=_Pipeline(sid, "m")).handle_message(Session(), "q"))  # type: ignore[arg-type]
    (failure,) = [e for e in events if type(e).__name__ == "Failure"]
    assert failure.message.startswith("No answer: Lab server is not answering at 127.0.0.1.")


def test_an_unknown_provider_never_falls_through_to_ollama() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_chat_model("nonesuch", "m")


def test_the_pipeline_records_the_model_like_the_other_adapters() -> None:
    assert build_chat_model("lmstudio", "qwen3").model == "qwen3"
