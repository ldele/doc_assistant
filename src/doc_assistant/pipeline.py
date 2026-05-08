"""RAG pipeline: retrieval, reranking, and answer generation."""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from doc_assistant.config import LLM_MODE, OLLAMA_HOST, ANTHROPIC_API_KEY, CHROMA_PATH
from doc_assistant.prompts import REWRITE_PROMPT, ANSWER_PROMPT


class RAGPipeline:
    def __init__(self):
        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        print("Loading vector store...")
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)

        print("Building keyword index...")
        data = self.db.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = 10
        vector = self.db.as_retriever(search_kwargs={"k": 10})
        self.ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])

        print("Loading reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        print("Loading LLM...")
        self.llm = self._build_llm()

    def _build_llm(self):
        if LLM_MODE == "api":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-haiku-4-5",
                api_key=ANTHROPIC_API_KEY,
                max_tokens=1024,
                streaming=True,
            )
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="llama3", base_url=OLLAMA_HOST)

    def retrieve(self, query: str, top_k: int = 5):
        candidates = self.ensemble.invoke(query)
        if not candidates:
            return []
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def rewrite(self, question: str, history: list[dict], counter=None) -> str:
        if not history:
            return question
        chain = REWRITE_PROMPT | self.llm
        callbacks = [counter] if counter else []
        result = chain.invoke(
            {"history": format_history(history), "question": question},
            config={"callbacks": callbacks},
        )
        return result.content if hasattr(result, "content") else str(result)

    def stream_answer(self, question: str, docs, counter=None):
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
    return " · ".join(parts)


def format_docs_for_prompt(docs):
    parts = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        header = f"[Source {i + 1}: {filename}"
        if page:
            header += f", page {page}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(messages: list[dict]) -> str:
    if not messages:
        return "(no prior messages)"
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in messages[-6:]
    )