"""CLI version."""
from doc_assistant.pipeline import RAGPipeline, format_citation


def main():
    rag = RAGPipeline()
    history = []
    
    print(f"\nReady. {rag.chunk_count()} chunks indexed. Type 'exit' to quit.\n")
    while True:
        question = input("Ask: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        
        standalone = rag.rewrite(question, history)
        docs = rag.retrieve(standalone)
        
        print("\nAnswer: ", end="", flush=True)
        full_answer = ""
        for token in rag.stream_answer(standalone, docs):
            print(token, end="", flush=True)
            full_answer += token
        print("\n")
        
        print("Sources:")
        for i, doc in enumerate(docs):
            print(f"  {format_citation(doc, i + 1)}")
        print()
        
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    # Required for current pyproject.toml to work
    # Also wanted this for debugging
    main()