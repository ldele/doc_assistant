"""Integration tests for the A/B-compare endpoint (feature-ab-compare-sandbox.md, U6).

Drives the **real** ``ChatController.compare_retrieval`` through ``POST /api/compare`` with a fake
retriever (the ``ChatController(rag=…)`` seam, cpc §13). The fake adds a source when multi-query is
on (a membership diff); its ``llm``/``stream_answer`` raise, so any generation fails the test.
"""

from __future__ import annotations

from apps.api.main import create_app
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from doc_assistant import config
from doc_assistant.chat_controller import ChatController


class FakeRag:
    """A retrieval-only fake. Membership depends on ``use_multi_query`` so B can diverge from A."""

    provider = "anthropic"
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve_with_scores(
        self,
        text: str,
        top_k: int,
        *,
        use_multi_query: bool | None = None,
        scope: frozenset[str] | None = None,
    ) -> list[tuple[Document, float]]:
        self.calls.append({"text": text, "top_k": top_k, "use_multi_query": use_multi_query})
        docs = [
            (
                Document(page_content="alpha", metadata={"filename": "a.pdf", "doc_hash": "h1"}),
                0.9,
            ),
            (Document(page_content="beta", metadata={"filename": "b.pdf", "doc_hash": "h2"}), 0.8),
        ]
        if use_multi_query:  # B (override) surfaces an extra source -> a real membership diff
            docs.append(
                (
                    Document(
                        page_content="gamma", metadata={"filename": "c.pdf", "doc_hash": "h3"}
                    ),
                    0.7,
                )
            )
        return docs[:top_k]

    @property
    def llm(self) -> object:
        raise AssertionError("compare must not touch the LLM")

    def stream_answer(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("compare must not generate an answer")


def _client_and_rag() -> tuple[TestClient, FakeRag]:
    rag = FakeRag()
    controller = ChatController(rag=rag)  # type: ignore[arg-type]
    return TestClient(create_app(controller=controller)), rag


def test_compare_endpoint() -> None:
    top_k_default = config.TOP_K
    mq_default = config.USE_MULTI_QUERY
    client, rag = _client_and_rag()

    # --- override flips use_multi_query on -> B surfaces an extra source (membership diff) ---
    body = client.post(
        "/api/compare", json={"text": "what is x?", "overrides": {"use_multi_query": True}}
    ).json()
    assert len(body["sources_a"]) == 2  # A = defaults (mq follows global)
    assert len(body["sources_b"]) == 3  # B = mq on -> extra source
    statuses = {r["status"] for r in body["rows"]}
    assert "only_in_b" in statuses  # the extra source shows as B-only
    assert body["note"] == ""  # membership genuinely moved -> the diff speaks, no note
    assert body["eff_a"]["use_multi_query"] == mq_default
    assert body["eff_b"]["use_multi_query"] is True
    # A retrieves at the global default (use_multi_query=None); B forces the override value.
    assert rag.calls[0]["use_multi_query"] is None
    assert rag.calls[1]["use_multi_query"] is True

    # --- no override -> both sides identical, honest no-op note ---
    body = client.post("/api/compare", json={"text": "q", "overrides": None}).json()
    assert len(body["sources_a"]) == len(body["sources_b"]) == 2
    assert all(r["status"] == "in_both" for r in body["rows"])
    assert "doesn't change retrieval" in body["note"]

    # --- guard: no module-global was mutated (ADR-010 isolation) ---
    assert top_k_default == config.TOP_K
    assert mq_default == config.USE_MULTI_QUERY
