"""CLI version — thin shell over doc_assistant core.

Same dispatch order as the Chainlit UI:
    1. Slash command  -> commands.execute_command()
    2. Library metadata question -> query_router.answer_library_query()
    3. Otherwise -> RAG pipeline
"""

from doc_assistant.commands import execute_command, parse_command
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.query_router import answer_library_query, is_library_query


def main() -> None:
    rag = RAGPipeline()
    history: list[dict[str, str]] = []

    print(f"\nReady. {rag.chunk_count()} chunks indexed. Type 'exit' to quit.\n")
    while True:
        question = input("Ask: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        # --- Slash commands ---
        parsed = parse_command(question)
        if parsed is not None:
            cmd, arg = parsed
            print("\n" + execute_command(cmd, arg) + "\n")
            continue

        # --- Library metadata questions (SQLite-answered) ---
        if is_library_query(question):
            print("\n" + answer_library_query(question) + "\n")
            continue

        # --- RAG ---
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
    main()
