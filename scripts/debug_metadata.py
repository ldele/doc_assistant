from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from doc_assistant.config import CHROMA_PATH

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

data = db.get(include=["metadatas"])
print(f"Total chunks: {len(data['metadatas'])}")
print(f"\nFirst 3 metadata entries:")
for i, meta in enumerate(data["metadatas"][:3]):
    print(f"\n  Chunk {i}: {meta}")

# Count metadata completeness
with_filename = sum(1 for m in data["metadatas"] if m and m.get("filename"))
print(f"\nChunks with filename: {with_filename} / {len(data['metadatas'])}")