"""Build a parent-child index alongside the existing Chroma store.

Child chunks are small and used for retrieval.
Each child carries its parent text in metadata, passed to the LLM at query time.

Run once to build, then pipeline switches between baseline and pc index via config.
"""
import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from doc_assistant.config import DATA_PATH, CACHE_PATH

# Where to store the new index. Sibling to data/chroma/.
PC_CHROMA_PATH = str(DATA_PATH / "chroma_pc")

# Sizing — these are the knobs to tune later
PARENT_SIZE = 2000
PARENT_OVERLAP = 200
CHILD_SIZE = 400
CHILD_OVERLAP = 50
EMBED_BATCH = 512

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_SIZE,
    chunk_overlap=PARENT_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_SIZE,
    chunk_overlap=CHILD_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def doc_hash(text: str, source: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def build_parent_child_chunks(text: str, doc_metadata: dict) -> list[Document]:
    """Produce child chunks, each carrying its parent text in metadata."""
    parents = _parent_splitter.split_text(text)
    all_children = []

    for parent_idx, parent_text in enumerate(parents):
        children = _child_splitter.split_text(parent_text)
        for child_idx, child_text in enumerate(children):
            meta = dict(doc_metadata)
            meta.update({
                "parent_text": parent_text,
                "parent_index": parent_idx,
                "child_index": child_idx,
            })
            all_children.append(Document(page_content=child_text, metadata=meta))

    return all_children


def main():
    # Wipe and rebuild
    if Path(PC_CHROMA_PATH).exists():
        print(f"Removing existing index at {PC_CHROMA_PATH}")
        shutil.rmtree(PC_CHROMA_PATH)
    Path(PC_CHROMA_PATH).mkdir(parents=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )
    db = Chroma(persist_directory=PC_CHROMA_PATH, embedding_function=embeddings)

    # Walk the cache (already extracted markdown — no need to re-extract)
    cache_files = list(Path(CACHE_PATH).rglob("*.md"))
    print(f"Found {len(cache_files)} cached documents")

    all_children: list[Document] = []
    for cache_path in tqdm(cache_files, desc="Chunking"):
        try:
            text = cache_path.read_text(encoding="utf-8")
            if not text.strip():
                continue

            relative = cache_path.relative_to(CACHE_PATH)
            original_path = (DATA_PATH / "sources" / relative).with_suffix(".pdf")

            doc_metadata = {
                "filename": cache_path.stem + ".pdf",
                "source_original": str(original_path),
                "source_cache": str(cache_path),
                "doc_hash": doc_hash(text, str(original_path)),
                "format": "pdf",
            }

            all_children.extend(build_parent_child_chunks(text, doc_metadata))
        except Exception as e:
            print(f"\nError on {cache_path.name}: {e}")

    print(f"Chunked {len(all_children)} child chunks. Embedding...")
    for i in tqdm(range(0, len(all_children), EMBED_BATCH), desc="Embedding"):
        db.add_documents(all_children[i : i + EMBED_BATCH])

    print(f"\nDone. Indexed {len(all_children)} child chunks.")


if __name__ == "__main__":
    main()