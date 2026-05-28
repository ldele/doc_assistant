"""RAG pipeline: retrieval, reranking, and answer generation."""

from collections.abc import Generator
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from doc_assistant.config import (
    ANTHROPIC_API_KEY,
    CHROMA_PATH,
    LLM_MODE,
    OLLAMA_HOST,
    PC_CHROMA_PATH,
    TOP_K,
    USE_MULTI_QUERY,
    USE_PARENT_CHILD,
)
from doc_assistant.embeddings import (
    get_active_model_name,
    get_collection_name,
    get_embeddings,
)
from doc_assistant.prompts import ANSWER_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT


class RAGPipeline:
    def __init__(self) -> None:
        active_model = get_active_model_name()
        collection = get_collection_name(active_model)
        print(f"Loading embeddings ({active_model})...")
        self.embeddings = get_embeddings(active_model)

        print("Loading vector store...")
        chroma_path = PC_CHROMA_PATH if USE_PARENT_CHILD else CHROMA_PATH
        print(f"Using vector store: {chroma_path} (collection: {collection})")
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name=collection,
        )

        print("Building keyword index...")
        data = self.db.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"], strict=True)
            if not (meta and meta.get("keep_for_retrieval") is False)
        ]
        excluded = len(data["documents"]) - len(all_docs)
        print(f"  BM25 index excludes {excluded} non-retrievable chunks")
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = 10
        vector = self.db.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"keep_for_retrieval": {"$ne": False}},
            }
        )
        self.ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])

        print("Loading reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        print("Loading LLM...")
        self.llm = self._build_llm()

    def _build_llm(self) -> Any:
        if LLM_MODE == "api":
            from langchain_anthropic import ChatAnthropic
            from pydantic import SecretStr

            secret_key = SecretStr(ANTHROPIC_API_KEY or "")
            return ChatAnthropic(  # type: ignore[call-arg]
                model="claude-haiku-4-5-20251001",
                api_key=secret_key,
                max_tokens=1024,
                streaming=True,
            )
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model="llama3", base_url=OLLAMA_HOST)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Document]:
        """Retrieve top-k documents for `query`. Reranker scores discarded."""
        return [doc for doc, _ in self.retrieve_with_scores(query, top_k)]

    def retrieve_with_scores(self, query: str, top_k: int = TOP_K) -> list[tuple[Document, float]]:
        """Retrieve top-k as ``(doc, reranker_score)`` pairs.

        Used by the provenance card to record per-chunk attribution and
        by anything that wants to inspect reranker confidence (e.g.,
        Phase 6 dual-interpretation gating).
        """
        # Multi-Query: generate variations if enabled
        queries = self.expand_query(query) if USE_MULTI_QUERY else [query]

        # Collect candidates from all queries
        all_candidates: list[Document] = []
        seen_ids: set[str] = set()
        for q in queries:
            candidates = self.ensemble.invoke(q)
            for doc in candidates:
                doc_id = doc.metadata.get("doc_hash", "") + "_" + doc.page_content[:50]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_candidates.append(doc)

        if not all_candidates:
            return []

        # Rerank against the original query
        pairs = [[query, doc.page_content] for doc in all_candidates]
        scores = self.reranker.predict(pairs)
        ranked: list[tuple[Document, float]] = sorted(
            zip(all_candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        # Parent-child: dedup by parent BEFORE applying top_k. The
        # reranker_score we return is the *child*'s score that won — the
        # parent is the LLM context, the child is the retrieval evidence.
        if USE_PARENT_CHILD:
            seen_parents: set[tuple[Any, ...]] = set()
            deduped: list[tuple[Document, float]] = []
            for doc, score in ranked:
                parent_text = doc.metadata.get("parent_text")
                parent_key = (
                    doc.metadata.get("filename"),
                    doc.metadata.get("parent_index"),
                )

                if parent_text and parent_key not in seen_parents:
                    seen_parents.add(parent_key)
                    new_doc = Document(
                        page_content=parent_text,
                        metadata={k: v for k, v in doc.metadata.items() if k != "parent_text"},
                    )
                    deduped.append((new_doc, float(score)))
                    if len(deduped) >= top_k:
                        break
            return deduped

        # No parent-child: just take top_k
        return [(doc, float(score)) for doc, score in ranked[:top_k]]

    def rewrite(
        self,
        question: str,
        history: list[dict[str, str]],
        counter: Any = None,
    ) -> str:
        if not history:
            return question
        chain = REWRITE_PROMPT | self.llm
        callbacks = [counter] if counter else []
        result = chain.invoke(
            {"history": format_history(history), "question": question},
            config={"callbacks": callbacks},
        )
        return result.content if hasattr(result, "content") else str(result)

    def stream_answer(
        self,
        question: str,
        docs: list[Document],
        counter: Any = None,
    ) -> Generator[str, None, None]:
        context = format_docs_for_prompt(docs)
        chain = ANSWER_PROMPT | self.llm
        callbacks = [counter] if counter else []
        for chunk in chain.stream(
            {"context": context, "question": question},
            config={"callbacks": callbacks},
        ):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    def chunk_count(self) -> int:
        return len(self.db.get(include=[])["ids"])

    def expand_query(self, query: str) -> list[str]:
        """Generate 3 alternative phrasings of the query.
        Returns a list including the original plus 3 variations.
        """
        chain = MULTI_QUERY_PROMPT | self.llm
        response = chain.invoke({"question": query})
        text = response.content if hasattr(response, "content") else str(response)

        # Parse the JSON array. Be defensive -- LLMs sometimes wrap in markdown.
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            import json

            variations = json.loads(text)
            if not isinstance(variations, list):
                variations = [query]
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, fall back to just the original query
            print("  Warning: MQ JSON parse failed, using original query only")
            variations = []

        # Always include the original query
        return [query] + [v for v in variations if isinstance(v, str) and v.strip()]


# ============================================================
# Formatting helpers
# ============================================================


def format_citation(doc: Document, idx: int) -> str:
    name = doc.metadata.get("filename", "unknown")
    page = doc.metadata.get("page")
    section = doc.metadata.get("section")
    parts = [f"[{idx}] {name}"]
    if page:
        parts.append(f"p.{page}")
    if section:
        parts.append(f'"{section}"')
    return " \xb7 ".join(parts)


def format_docs_for_prompt(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        header = f"[Source {i + 1}: {filename}"
        if page:
            header += f", page {page}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "(no prior messages)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[-6:])
