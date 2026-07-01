"""CLI entrypoint: ``python -m doc_assistant.ingest [--rebuild] [--skip-cleanup] [--path P]``."""

from __future__ import annotations

import argparse

from doc_assistant import config
from doc_assistant.ingest import main


def _cli() -> None:
    # `python -m doc_assistant.ingest` is a program entrypoint (not a library import),
    # so it configures logging here — the only place src/ does, and only when run as a
    # program. Without it the converted progress events would be silenced (ADR-003).
    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the vector store and re-embed everything",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip the orphan cleanup pass",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Limit ingest to one file or subdirectory. Accepts an absolute path, "
            "a path relative to CWD, or a path relative to DOCS_PATH. "
            "Orphan cleanup is skipped when --path is set."
        ),
    )
    args = parser.parse_args()
    main(force_rebuild=args.rebuild, skip_cleanup=args.skip_cleanup, scope=args.path)


if __name__ == "__main__":
    _cli()
