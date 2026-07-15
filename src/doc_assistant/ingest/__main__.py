"""CLI entrypoint for ingest: ``--rebuild`` | ``--path P`` | ``--files P…`` plus ``--dry-run``."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        metavar="P",
        help=(
            "Ingest exactly these files (absolute paths). An explicit selection: it overrides a "
            "file's `excluded` flag, skips orphan cleanup, and is mutually exclusive with "
            "--path and --rebuild."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the ingest plan (would_add / would_reembed / skip_unchanged / excluded) and "
            "exit WITHOUT loading the embedding model or opening Chroma."
        ),
    )
    args = parser.parse_args()
    # --files / --path / --rebuild are mutually exclusive (main() re-checks; fail fast here too).
    chosen = [
        name
        for name, on in (
            ("--files", args.files),
            ("--path", args.path),
            ("--rebuild", args.rebuild),
        )
        if on
    ]
    if len(chosen) > 1:
        parser.error(f"{' and '.join(chosen)} are mutually exclusive")
    file_paths = [Path(p) for p in args.files] if args.files else None
    main(
        force_rebuild=args.rebuild,
        skip_cleanup=args.skip_cleanup,
        scope=args.path,
        files=file_paths,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
