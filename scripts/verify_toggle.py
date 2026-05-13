"""Verify that USE_PARENT_CHILD actually routes retrieval to the right store.

We force the config value, instantiate the pipeline, and inspect which 
store it loaded.
"""
import os
import sys


def check_with(use_pc: bool):
    # Set env var before importing
    os.environ["USE_PARENT_CHILD"] = "true" if use_pc else "false"
    
    # Force re-import of config and pipeline
    for mod in list(sys.modules.keys()):
        if "doc_assistant" in mod:
            del sys.modules[mod]
    
    from doc_assistant.config import CHROMA_PATH, PC_CHROMA_PATH, USE_PARENT_CHILD
    from doc_assistant.pipeline import RAGPipeline
    
    assert USE_PARENT_CHILD == use_pc, f"Config didn't pick up env var: {USE_PARENT_CHILD}"
    
    rag = RAGPipeline()
    
    # Inspect the active store path
    active_path = rag.db._collection._client._system.settings.persist_directory \
        if hasattr(rag.db, '_collection') else None
    
    # Simpler check: see how many chunks are in the active store
    chunk_count = rag.chunk_count() if hasattr(rag, 'chunk_count') else None
    
    expected_path = PC_CHROMA_PATH if use_pc else CHROMA_PATH
    
    print(f"\nWith USE_PARENT_CHILD={use_pc}:")
    print(f"  Expected store: {expected_path}")
    print(f"  Active path:    {active_path}")
    print(f"  Chunk count:    {chunk_count}")
    
    # Try a sample retrieval
    test_query = "neuron"
    docs = rag.retrieve(test_query, top_k=3)
    print(f"  Sample retrieval for 'neuron': {len(docs)} chunks")
    if docs:
        first = docs[0]
        content_length = len(first.page_content)
        print(f"    First chunk content length: {content_length} (expect >=1500 if parent-child)")
        print(f"    First chunk filename: {first.metadata.get('filename')}")


def main():
    print("=" * 60)
    print("Toggle verification")
    print("=" * 60)
    
    check_with(use_pc=False)
    check_with(use_pc=True)
    
    print("\n" + "=" * 60)
    print("Both modes inspected. Check that:")
    print("  1. The active path matches the expected path for each mode")
    print("  2. Both modes return docs for the test query")
    print("  3. With USE_PARENT_CHILD=true, retrieved chunks have parent_text")
    print("  4. With USE_PARENT_CHILD=false, retrieved chunks may not have parent_text")
    print("=" * 60)


if __name__ == "__main__":
    main()