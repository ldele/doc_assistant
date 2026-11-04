"""Where a path stored in the library may be acted on (ADR-061 choice 4, security S-7, review G1).

A library stores absolute paths: each document's source file and cached text, each figure's image.
They are right on the computer and in the folder that made the library, and wrong anywhere else.
A library copied with a file manager still names the original's files, and before this module a
re-run, a delete or the orphan sweep would act on them: a re-run of a copy's text deleted the
original's cache file (2026-11-01, a scratch copy).

So every place that **deletes, overwrites or serves** a file through a stored path asks
:func:`confine` first, and gets the path back only when it lies where that kind of file belongs:

* ``cache``: under the open library's cache folder (``config.CACHE_PATH``);
* ``figure``: under its figures folder (``ingest.figures.FIGURE_DIR``);
* ``source``: under its own sources folder, the source folder in its settings, or a root it has
  registered (ADR-046).

Anything else is refused, logged, and left alone; the caller says so in words. It holds for every
library, the app's own included. Reads that only take text for a derived pass (keywords, concept
semantics) are not gated: on a copy they read the same text, and refusing would only break them.

A root registered by a *reference* add is the folder of a file the person chose, so a row added
that way passes the source check. That is what S9 says a root check alone cannot stop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import structlog

from doc_assistant import config

log = structlog.get_logger(__name__)

Purpose = Literal["cache", "figure", "source"]


def _key(path: Path) -> str | None:
    """The path's resolved, case-folded spelling, or None when it cannot be resolved.

    Resolved so a junction or a symlink inside an allowed folder that points out of it is judged
    by where it lands, and ``..`` cannot walk out.
    """
    try:
        return os.path.normcase(str(path.resolve()))
    except (OSError, RuntimeError, ValueError):
        return None


def _inside(path: Path, base: Path) -> bool:
    p, b = _key(path), _key(base)
    if p is None or b is None:
        return False
    return p == b or p.startswith(b.rstrip(os.sep) + os.sep)


def source_bases() -> list[Path]:
    """Where a document's source file may be: the library's own sources folder, the source folder
    its settings name, and every root it has registered."""
    from sqlalchemy import select

    from doc_assistant import app_settings
    from doc_assistant.db.models import SourceRoot
    from doc_assistant.db.session import session_scope

    bases = [Path(config.DOCS_PATH)]
    try:
        bases.append(app_settings.get_source_dir())
    except Exception as e:  # unreadable settings must not open every path
        log.warning("source_dir_unreadable", error=str(e))
    try:
        with session_scope() as session:
            bases.extend(Path(p) for p in session.scalars(select(SourceRoot.path)) if p)
    except Exception as e:  # a registry that will not answer leaves the library's own folders
        log.warning("source_roots_unreadable", error=str(e))
    return bases


def bases_for(purpose: Purpose) -> list[Path]:
    if purpose == "cache":
        return [Path(config.CACHE_PATH)]
    if purpose == "figure":
        # Asked of the module that writes figure images, which owns the layout: every image sits
        # in `figure_dir(doc_hash)`, so the figures folder is that directory's parent.
        from doc_assistant.ingest.figures import figure_dir

        return [figure_dir("_").parent]
    return source_bases()


def is_confined(stored: str | Path, purpose: Purpose) -> bool:
    """Whether ``stored`` lies where a ``purpose`` file of the open library belongs."""
    path = Path(stored)
    return any(_inside(path, base) for base in bases_for(purpose))


def confine(stored: str | Path | None, purpose: Purpose) -> Path | None:
    """``stored`` as a path when it may be acted on, else None (logged once per call).

    None for an empty value too, so callers need one branch: no path to act on.
    """
    if stored is None or not str(stored).strip():
        return None
    path = Path(stored)
    if is_confined(path, purpose):
        return path
    log.warning("stored_path_outside_library", purpose=purpose, file=path.name)
    return None
