"""CLI — thin renderer over ``doc_assistant.chat_controller``.

Same turn orchestration as the Chainlit UI (one ``ChatController``); the CLI just
renders the ``TurnEvent`` stream to stdout: stream tokens as they arrive, then print
the finished ``TurnResult`` (answer + the pre-rendered markdown blocks). Dispatch
order (slash command → library query → RAG) now lives in the controller, so the CLI
gains the same commands the web UI has (including ``/export``).
"""

from doc_assistant.chat_controller import ChatController, Result, Session, Token, TurnResult
from doc_assistant.config import LOG_JSON, LOG_LEVEL
from doc_assistant.logging_config import configure_logging


def _render_result(result: TurnResult, *, streamed: bool) -> None:
    """Print the finished turn. If the answer was already streamed token-by-token, only
    the trailing blocks are printed; otherwise the full answer + blocks."""
    if not streamed:
        print("\n" + result.answer)
    blocks = (
        result.sources_md
        + result.usage_md
        + result.provenance_card_md
        + result.claim_review_md
        + result.citation_note_md
    )
    if blocks.strip():
        print(blocks)
    print()


def main() -> None:
    configure_logging(json=LOG_JSON, level=LOG_LEVEL)
    controller = ChatController()
    session = Session()

    print(f"\nReady. {controller.chunk_count()} chunks indexed. Type 'exit' to quit.\n")
    while True:
        question = input("Ask: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        streamed = False
        for event in controller.handle_message(session, question):
            if isinstance(event, Token):
                if not streamed:
                    print("\nAnswer: ", end="", flush=True)
                    streamed = True
                print(event.text, end="", flush=True)
            elif isinstance(event, Result):
                if streamed:
                    print()
                _render_result(event.result, streamed=streamed)
        # Step events are advisory; the CLI ignores them.


if __name__ == "__main__":
    main()
