"""Read a Zotero library and return neutral records (ADR-049). The only vendor-aware module.

Everything Zotero-shaped stops here: `linkMode`, `itemAttachments`, `storage:` paths, the
`itemData`/`itemDataValues` indirection. What leaves is `ExternalDocument` — a path and the fields
any reference manager has. That is the spec's ADR-3 boundary, made structural rather than promised.

**It never writes, and it never opens the live file.** Zotero keeps `zotero.sqlite` open while it
runs, and a reader that trips its lock — or, worse, that a future edit could make write to it —
would put the user's own library at risk for a feature they can live without. So the database is
**copied** to a temporary file (with its `-wal`/`-shm` companions, or recent edits would be
invisible) and the copy is opened read-only. A copy is a few megabytes and takes milliseconds.

**Written against the documented Zotero 5/6/7 schema, and not yet run against a real library** —
there is no Zotero on the machine this was built on. The synthetic fixture in
`tests/unit/adapters/test_zotero.py` is built to that schema, so the tests prove the mapping, not
the schema. Every query is therefore defensive: a missing table or column produces
`CatalogueUnavailable` with a sentence in it, and an optional part that will not read (collections,
creators) degrades to "no collections" rather than failing the import.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import structlog

from doc_assistant.extractors import is_supported

from .catalogue import CatalogueUnavailable, ExternalDocument, ExternalScan

log = structlog.get_logger(__name__)

#: The label recorded in `ExternalMetadata.source`. A string, not a type.
SOURCE = "zotero"

#: Zotero's `itemAttachments.linkMode`. 0/1 keep the file inside the data directory; 2 points at a
#: file the user keeps elsewhere; 3 is a bare URL with no file at all.
_IMPORTED_FILE = 0
_IMPORTED_URL = 1
_LINKED_FILE = 2

#: Where a stored attachment lives: `<data dir>/storage/<attachment key>/<filename>`.
_STORAGE_PREFIX = "storage:"
#: A linked attachment relative to Zotero's "Linked Attachment Base Directory" preference.
_BASE_PREFIX = "attachments:"

_YEAR_RE = re.compile(r"(1[4-9]\d{2}|20\d{2}|21\d{2})")


def default_data_dir() -> Path:
    """Zotero's out-of-the-box data directory: `~/Zotero` on every platform it ships for.

    A guess, offered so the common case needs no folder picker. The caller always gets to
    override it, and `read_library` says plainly when nothing is there.
    """
    return Path.home() / "Zotero"


@contextmanager
def _open_copy(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Copy the catalogue aside and open the copy read-only.

    The `-wal` and `-shm` companions come too when they exist: with WAL journalling the main file
    can be minutes behind, so copying it alone would silently import a stale library.
    """
    with tempfile.TemporaryDirectory(prefix="provenote-zotero-") as tmp:
        target = Path(tmp) / db_path.name
        try:
            shutil.copy2(db_path, target)
            for suffix in ("-wal", "-shm"):
                companion = db_path.with_name(db_path.name + suffix)
                if companion.exists():
                    shutil.copy2(companion, target.with_name(target.name + suffix))
        except OSError as e:
            raise CatalogueUnavailable(
                f"Could not read the Zotero database at {db_path} ({type(e).__name__}). "
                "If Zotero is mid-sync, try again in a moment."
            ) from e
        connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()


def _year_from(date_value: str | None) -> int | None:
    """The first plausible year in a Zotero date string.

    Zotero stores dates as free text with a parsed suffix — `2020-04-12 2020-04-12`, `April 2020`,
    `2020`, `n.d.` — so a four-digit match is the only thing that reads all of them. Anything
    outside 1400-2199 is not a publication year.
    """
    if not date_value:
        return None
    match = _YEAR_RE.search(date_value)
    return int(match.group(1)) if match else None


def _query(connection: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    """Run a query, turning a schema mismatch into a readable failure rather than a crash."""
    try:
        return list(connection.execute(sql))
    except sqlite3.DatabaseError as e:
        raise CatalogueUnavailable(
            f"This does not look like a Zotero database — {e}. "
            "Point at the folder containing zotero.sqlite."
        ) from e


def _optional(connection: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    """A query whose failure costs a nice-to-have, not the import.

    Collections and creators are enrichment: an older or newer Zotero that renamed one of those
    tables should cost the user their authors list, never their documents.
    """
    try:
        return list(connection.execute(sql))
    except sqlite3.DatabaseError as e:
        log.warning("zotero_optional_query_failed", error=str(e))
        return []


def _fields_by_item(connection: sqlite3.Connection) -> dict[int, dict[str, str]]:
    """`{itemID: {fieldName: value}}` for the fields worth having.

    Zotero stores every field value once in `itemDataValues` and points at it, so this is a
    three-way join rather than a column read.
    """
    rows = _query(
        connection,
        """
        SELECT itemData.itemID AS item_id, fields.fieldName AS name, itemDataValues.value AS value
        FROM itemData
        JOIN fields ON fields.fieldID = itemData.fieldID
        JOIN itemDataValues ON itemDataValues.valueID = itemData.valueID
        WHERE fields.fieldName IN ('title', 'date', 'DOI', 'publicationTitle')
        """,
    )
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        out.setdefault(int(row["item_id"]), {})[str(row["name"])] = str(row["value"])
    return out


def _authors_by_item(connection: sqlite3.Connection) -> dict[int, str]:
    """`{itemID: "First Last, First Last"}` — authors in Zotero's own order.

    `fieldMode = 1` means a single-field name (an institution, "World Health Organization"), which
    lives in `lastName` with `firstName` empty; joining the two blindly would emit a leading space.

    Authors only where an item has any: an edited volume whose creators are all editors would
    otherwise report no one at all.
    """
    rows = _optional(
        connection,
        """
        SELECT itemCreators.itemID AS item_id, creators.firstName AS first,
               creators.lastName AS last, creators.fieldMode AS mode,
               creatorTypes.creatorType AS role, itemCreators.orderIndex AS ord
        FROM itemCreators
        JOIN creators ON creators.creatorID = itemCreators.creatorID
        LEFT JOIN creatorTypes ON creatorTypes.creatorTypeID = itemCreators.creatorTypeID
        ORDER BY itemCreators.itemID, itemCreators.orderIndex
        """,
    )
    everyone: dict[int, list[str]] = {}
    authors: dict[int, list[str]] = {}
    for row in rows:
        last = (row["last"] or "").strip()
        first = (row["first"] or "").strip()
        name = last if int(row["mode"] or 0) == 1 or not first else f"{first} {last}".strip()
        if not name:
            continue
        item_id = int(row["item_id"])
        everyone.setdefault(item_id, []).append(name)
        if str(row["role"] or "") == "author":
            authors.setdefault(item_id, []).append(name)
    return {
        item_id: ", ".join(authors.get(item_id) or names) for item_id, names in everyone.items()
    }


def _collections_by_item(connection: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    """`{itemID: (collection name, …)}`, each name a full path like `Thesis / Chapter 2`.

    The path rather than the leaf, because a bare `Chapter 2` means nothing on its own and two
    collections may share a leaf name.
    """
    collections = {
        int(r["collectionID"]): (str(r["collectionName"]), r["parentCollectionID"])
        for r in _optional(
            connection, "SELECT collectionID, collectionName, parentCollectionID FROM collections"
        )
    }

    def full_path(collection_id: int) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        current: int | None = collection_id
        while current is not None and current in collections and current not in seen:
            seen.add(current)  # a cycle in the parent chain must not hang the import
            name, parent = collections[current]
            parts.append(name)
            current = int(parent) if parent is not None else None
        return " / ".join(reversed(parts))

    out: dict[int, list[str]] = {}
    for row in _optional(connection, "SELECT collectionID, itemID FROM collectionItems"):
        path = full_path(int(row["collectionID"]))
        if path:
            out.setdefault(int(row["itemID"]), []).append(path)
    return {item_id: tuple(sorted(set(names))) for item_id, names in out.items()}


def _attachment_path(
    storage_dir: Path, attachment_key: str, raw: str | None, link_mode: int, base_dir: Path | None
) -> Path | None:
    """Turn Zotero's stored path into a real one, or None when it cannot be resolved here."""
    if not raw:
        return None
    if link_mode in (_IMPORTED_FILE, _IMPORTED_URL):
        if not raw.startswith(_STORAGE_PREFIX):
            return None
        return storage_dir / attachment_key / raw[len(_STORAGE_PREFIX) :]
    if link_mode == _LINKED_FILE:
        if raw.startswith(_BASE_PREFIX):
            if base_dir is None:
                return None
            return base_dir / raw[len(_BASE_PREFIX) :]
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else None
    return None


def read_library(
    data_dir: Path | None = None,
    *,
    base_dir: Path | None = None,
    include_snapshots: bool = False,
) -> ExternalScan:
    """Read a Zotero data directory and return every attachment we could use, plus what was not.

    `base_dir` is Zotero's "Linked Attachment Base Directory" — a *preference*, not a database
    value, so it cannot be read from here and has to be supplied by the caller when it is set.
    Without it, linked attachments stored relative to that base are counted as skipped rather than
    guessed at.

    `include_snapshots` is off by default. A Zotero snapshot is a saved copy of a web page, and a
    library of any age holds hundreds; importing them silently would swamp the corpus with page
    furniture. They are counted so the number is visible rather than absent.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    db_path = root / "zotero.sqlite"
    if not db_path.exists():
        raise CatalogueUnavailable(
            f"No Zotero library at {root} — there is no zotero.sqlite in that folder. "
            "Choose the Zotero data directory (it contains zotero.sqlite and a storage folder)."
        )
    storage_dir = root / "storage"

    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    documents: list[ExternalDocument] = []
    with _open_copy(db_path) as connection:
        trashed = {
            int(r["itemID"]) for r in _optional(connection, "SELECT itemID FROM deletedItems")
        }
        fields = _fields_by_item(connection)
        authors = _authors_by_item(connection)
        collections = _collections_by_item(connection)
        types = {
            int(r["itemID"]): str(r["typeName"])
            for r in _query(
                connection,
                """
                SELECT items.itemID, itemTypes.typeName
                FROM items JOIN itemTypes ON itemTypes.itemTypeID = items.itemTypeID
                """,
            )
        }
        keys = {
            int(r["itemID"]): str(r["key"])
            for r in _query(connection, "SELECT itemID, key FROM items")
        }

        rows = _query(
            connection,
            """
            SELECT itemID, parentItemID, linkMode, contentType, path
            FROM itemAttachments
            """,
        )

    for row in rows:
        item_id = int(row["itemID"])
        parent_id = int(row["parentItemID"]) if row["parentItemID"] is not None else None
        link_mode = int(row["linkMode"] or 0)

        if item_id in trashed or (parent_id is not None and parent_id in trashed):
            skip("in the Zotero trash")
            continue
        if link_mode not in (_IMPORTED_FILE, _IMPORTED_URL, _LINKED_FILE):
            skip("a saved link with no file")
            continue

        is_snapshot = link_mode == _IMPORTED_URL and str(row["contentType"] or "") == "text/html"
        if is_snapshot and not include_snapshots:
            skip("a web-page snapshot")
            continue

        path = _attachment_path(
            storage_dir, keys.get(item_id, ""), row["path"], link_mode, base_dir
        )
        if path is None:
            skip("stored somewhere this app cannot resolve")
            continue
        if not path.exists():
            # Common and worth naming: with Zotero's sync set to metadata-only, the entry is real
            # and the file is simply not on this machine.
            skip("not downloaded to this computer")
            continue
        if not is_supported(path):
            skip(f"an unsupported file type ({path.suffix.lower().lstrip('.') or 'no extension'})")
            continue

        # The metadata belongs to the *parent* item — an attachment row is the file, the item is
        # the work. A standalone attachment (no parent) is its own item and describes itself.
        meta_id = parent_id if parent_id is not None else item_id
        item_fields = fields.get(meta_id, {})
        documents.append(
            ExternalDocument(
                path=path.resolve(),
                title=(item_fields.get("title") or "").strip() or None,
                authors=authors.get(meta_id),
                year=_year_from(item_fields.get("date")),
                doi=(item_fields.get("DOI") or "").strip() or None,
                item_type=types.get(meta_id),
                collections=collections.get(meta_id, ()),
                external_key=keys.get(meta_id),
            )
        )

    log.info(
        "zotero_library_read",
        data_dir=str(root),
        found=len(documents),
        skipped=sum(skipped.values()),
    )
    return ExternalScan(
        root=storage_dir if storage_dir.is_dir() else root,
        documents=tuple(documents),
        skipped=skipped,
        label="Zotero",
    )
