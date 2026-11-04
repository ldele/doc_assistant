"""What a library folder holds, read before the app switches to it (ADR-061 choice 3).

A library is a data home: ``library.db``, the two vector stores, the cache, settings. ADR-061 lets
the desktop app open one that was indexed elsewhere on this computer, and this module is the check
before *Open*: it reports what the folder holds and whether opening it would work, **and writes
nothing**. Every file is opened read-only through SQLite's URI mode, never through the app's
engine or the vector-store library: opening a store with the library can migrate it, which is a
write to a folder the person has not chosen yet.

What blocks *Open*, each with its sentence: a folder that is not a library; a path with a
character outside ASCII (the vector-store library cannot read it there, KI-11, ADR-061 choice 5);
the library already open. What only informs: no chunks for this app's embedding model, sources
not found where the library says, cached text that names another folder (a copy, ADR-061
*In practice*, step 4).
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from doc_assistant import config
from doc_assistant.embeddings import get_active_model_name, get_collection_name

log = structlog.get_logger(__name__)

LIBRARY_DB = "library.db"
#: The store the answer path searches, so its count is the "chunks" Settings shows.
SEARCHED_STORE = "chroma_pc"


@dataclass(frozen=True)
class LibraryInspection:
    """One folder, as the check before *Open* sees it."""

    path: str
    is_library: bool
    documents: int = 0
    chunks: int = 0
    embedding_model: str = ""
    sources_found: int = 0
    #: The library folder the stored cache paths name, when it is not this one (a copy or a move).
    cache_named_folder: str | None = None
    is_current: bool = False
    #: Sentences that stop *Open*; empty means it may open.
    problems: list[str] = field(default_factory=list)
    #: Sentences the person should read before *Open*.
    notes: list[str] = field(default_factory=list)

    @property
    def can_open(self) -> bool:
        return self.is_library and not self.problems


def _ro(path: Path) -> sqlite3.Connection:
    """A read-only connection: the URI's ``mode=ro`` refuses every write, and creates no file."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _same(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


def _inside(path: Path, base: Path) -> bool:
    p, b = os.path.normcase(str(path)), os.path.normcase(str(base))
    return p == b or p.startswith(b.rstrip(os.sep) + os.sep)


def _chunk_count(folder: Path, collection: str) -> int:
    """Embeddings in ``collection`` of the searched store, read from its SQLite file."""
    store = folder / SEARCHED_STORE / "chroma.sqlite3"
    if not store.is_file():
        return 0
    try:
        with _ro(store) as db:
            row = db.execute(
                "SELECT COUNT(e.id) FROM collections c "
                "JOIN segments s ON s.collection = c.id "
                "JOIN embeddings e ON e.segment_id = s.id WHERE c.name = ?",
                (collection,),
            ).fetchone()
    except sqlite3.Error as e:
        log.warning("library_inspect_store_unreadable", folder=str(folder), error=str(e))
        return 0
    return int(row[0]) if row else 0


def _cache_folder_named(cache_path: str) -> Path | None:
    """The library folder a stored cache path names: the parent of its ``cache`` folder."""
    parts = Path(cache_path).parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].lower() == "cache":
            return Path(*parts[:i])
    return None


def inspect_library(folder: Path) -> LibraryInspection:
    """Read what ``folder`` holds as a library. Writes nothing, raises nothing for a bad folder."""
    path = str(folder)
    if not folder.is_dir():
        return LibraryInspection(path, False, problems=["This folder does not exist."])
    db_path = folder / LIBRARY_DB
    if not db_path.is_file():
        return LibraryInspection(
            path, False, problems=["This folder is not a library: it holds no library.db."]
        )
    try:
        with _ro(db_path) as db:
            rows = db.execute(
                "SELECT filename, source_original, source_cache FROM documents"
            ).fetchall()
    except sqlite3.Error as e:
        log.warning("library_inspect_db_unreadable", folder=path, error=str(e))
        return LibraryInspection(
            path, False, problems=[f"Its library.db could not be read ({type(e).__name__})."]
        )

    model = get_active_model_name()
    collection = get_collection_name(model)
    # Counted where the app would read the stores once this folder is open: inside it, or under
    # %PROGRAMDATA% when its path is not ASCII on Windows (KI-11, `config.chroma_base_for`).
    base = config.chroma_base_for(folder.resolve())
    chunks = _chunk_count(base, collection)
    # Stores made inside an accented folder by a development backend that did not relocate them:
    # the app would read the empty relocated place instead (ADR-061 choice 5).
    stranded = base != folder.resolve() and chunks == 0 and _chunk_count(folder, collection) > 0
    found = sum(
        1
        for filename, original, _ in rows
        if (original and Path(original).exists()) or (folder / "sources" / filename).exists()
    )
    named = Counter(
        named
        for _, _, cache in rows
        if cache and not _inside(Path(cache), folder / "cache")
        for named in [_cache_folder_named(cache)]
        if named is not None
    )
    elsewhere = str(named.most_common(1)[0][0]) if named else None

    problems: list[str] = []
    notes: list[str] = []
    current = _same(folder, config.DATA_PATH)
    if current:
        problems.append("This is the library already open.")
    if stranded:
        problems.append(
            "Its folder path has a character outside plain ASCII (an accent, for instance), and "
            "its vector stores are inside it, where they cannot be read. Move the folder to a "
            "path without one, then open it."
        )
    elif rows and chunks == 0:
        notes.append(
            f"It has no chunks for this app's embedding model ({model}), so search will find "
            "nothing in it until it is indexed again."
        )
    if rows and found < len(rows):
        notes.append(
            f"{len(rows) - found} of {len(rows)} source files are not where the library says. "
            "Their documents stay searchable; opening or re-running one will say it is missing."
        )
    if elsewhere is not None:
        notes.append(
            f"Its cached text names another folder ({elsewhere}): it was copied or moved. "
            "Opening it is safe, nothing outside it will be written, but re-running a document's "
            "text needs the library where it was made."
        )
    if not rows:
        notes.append("It holds no documents yet.")
    return LibraryInspection(
        path=path,
        is_library=True,
        documents=len(rows),
        chunks=chunks,
        embedding_model=model,
        sources_found=found,
        cache_named_folder=elsewhere,
        is_current=current,
        problems=problems,
        notes=notes,
    )
