"""Reading a Zotero library (ADR-049, ROADMAP 17).

**What these tests prove, and what they cannot.** There is no Zotero on the machine this was built
on, so the fixture below *constructs* a database to the documented Zotero 5/6/7 schema. That makes
these tests a proof of the **mapping** — attachment rows to file paths, `itemData` indirection to
title/date/DOI, creators to an author string, the trash and snapshot filters — and **not** a proof
of the schema. If a real library ever disagrees, the fixture is what to correct first; every query
in the adapter is written to fail with a sentence rather than a stack trace precisely because that
day may come.

The other thing worth pinning here is the boundary: nothing that leaves this module carries a
`linkMode`, a `storage:` prefix or an `itemID`. What comes back is `ExternalDocument`, and the rest
of the app has never heard of Zotero.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from doc_assistant.adapters import zotero
from doc_assistant.adapters.catalogue import CatalogueUnavailable

# The subset of Zotero's schema the adapter reads. Column order and types follow the real thing.
_SCHEMA = """
CREATE TABLE items (
    itemID INTEGER PRIMARY KEY, itemTypeID INT NOT NULL, dateAdded TEXT, dateModified TEXT,
    libraryID INT NOT NULL DEFAULT 1, key TEXT NOT NULL, version INT DEFAULT 0
);
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT UNIQUE);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value UNIQUE);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT, PRIMARY KEY (itemID, fieldID));
CREATE TABLE creators (
    creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT, fieldMode INT
);
CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT UNIQUE);
CREATE TABLE itemCreators (
    itemID INT, creatorID INT, creatorTypeID INT, orderIndex INT,
    PRIMARY KEY (itemID, creatorID, creatorTypeID, orderIndex)
);
CREATE TABLE itemAttachments (
    itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT, contentType TEXT,
    charsetID INT, path TEXT, syncState INT DEFAULT 0
);
CREATE TABLE collections (
    collectionID INTEGER PRIMARY KEY, collectionName TEXT NOT NULL, parentCollectionID INT,
    libraryID INT NOT NULL DEFAULT 1, key TEXT NOT NULL
);
CREATE TABLE collectionItems (
    collectionID INT, itemID INT, orderIndex INT DEFAULT 0, PRIMARY KEY (collectionID, itemID)
);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY, dateDeleted TEXT);
"""

_FIELD_IDS = {"title": 1, "date": 2, "DOI": 3, "publicationTitle": 4}


class _Library:
    """A Zotero data directory under construction."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.storage = root / "storage"
        self.storage.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "zotero.sqlite")
        self.db.executescript(_SCHEMA)
        self.db.executemany(
            "INSERT INTO fields (fieldID, fieldName) VALUES (?, ?)",
            [(v, k) for k, v in _FIELD_IDS.items()],
        )
        self.db.executemany(
            "INSERT INTO itemTypes (itemTypeID, typeName) VALUES (?, ?)",
            [(1, "journalArticle"), (2, "book"), (3, "attachment")],
        )
        self.db.executemany(
            "INSERT INTO creatorTypes (creatorTypeID, creatorType) VALUES (?, ?)",
            [(1, "author"), (2, "editor")],
        )
        self._next = 1
        self._value = 1
        self._creator = 1

    def item(self, key: str, *, type_id: int = 1, **fields: str) -> int:
        item_id = self._next
        self._next += 1
        self.db.execute(
            "INSERT INTO items (itemID, itemTypeID, key) VALUES (?, ?, ?)", (item_id, type_id, key)
        )
        for name, value in fields.items():
            self.db.execute(
                "INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)", (self._value, value)
            )
            self.db.execute(
                "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
                (item_id, _FIELD_IDS[name], self._value),
            )
            self._value += 1
        return item_id

    def creator(
        self, item_id: int, first: str, last: str, *, role: int = 1, mode: int = 0, order: int = 0
    ) -> None:
        creator_id = self._creator
        self._creator += 1
        self.db.execute(
            "INSERT INTO creators (creatorID, firstName, lastName, fieldMode) VALUES (?, ?, ?, ?)",
            (creator_id, first, last, mode),
        )
        self.db.execute(
            "INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex) "
            "VALUES (?, ?, ?, ?)",
            (item_id, creator_id, role, order),
        )

    def attachment(
        self,
        key: str,
        *,
        parent: int | None,
        filename: str | None = None,
        link_mode: int = 0,
        content_type: str = "application/pdf",
        raw_path: str | None = None,
        on_disk: bool = True,
    ) -> int:
        item_id = self.item(key, type_id=3)
        path = raw_path if raw_path is not None else f"storage:{filename}"
        self.db.execute(
            "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, parent, link_mode, content_type, path),
        )
        if on_disk and filename and link_mode in (0, 1):
            folder = self.storage / key
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_bytes(b"%PDF-1.4 fixture")
        return item_id

    def collection(self, collection_id: int, name: str, *, parent: int | None = None) -> None:
        self.db.execute(
            "INSERT INTO collections (collectionID, collectionName, parentCollectionID, key) "
            "VALUES (?, ?, ?, ?)",
            (collection_id, name, parent, f"COLL{collection_id}"),
        )

    def file_in(self, collection_id: int, item_id: int) -> None:
        self.db.execute(
            "INSERT INTO collectionItems (collectionID, itemID) VALUES (?, ?)",
            (collection_id, item_id),
        )

    def trash(self, item_id: int) -> None:
        self.db.execute(
            "INSERT INTO deletedItems (itemID, dateDeleted) VALUES (?, '2026-01-01')", (item_id,)
        )

    def close(self) -> None:
        self.db.commit()
        self.db.close()


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A small but realistic library: two papers, a book, and four things to skip."""
    lib = _Library(tmp_path / "Zotero")

    paper = lib.item(
        "PAPER001",
        title="Retrieval-Augmented Generation",
        date="2020-04-12 2020-04-12",
        DOI="10.1000/rag",
    )
    lib.creator(paper, "Patrick", "Lewis", order=0)
    lib.creator(paper, "Ethan", "Perez", order=1)
    lib.attachment("ATTACH01", parent=paper, filename="lewis_2020.pdf")

    book = lib.item("BOOK0001", type_id=2, title="The Organization of Behavior", date="1949")
    lib.creator(book, "Donald", "Hebb", order=0)
    lib.attachment(
        "ATTACH02", parent=book, filename="hebb.epub", content_type="application/epub+zip"
    )

    lib.collection(1, "Reading")
    lib.collection(2, "RAG", parent=1)
    lib.file_in(2, paper)

    # Four entries that must not become documents, one per reason.
    snapshot = lib.item("SNAP0001", title="A blog post")
    lib.attachment(
        "ATTACH03", parent=snapshot, filename="page.html", link_mode=1, content_type="text/html"
    )
    binned = lib.item("BIN00001", title="A retracted preprint")
    lib.attachment("ATTACH04", parent=binned, filename="retracted.pdf")
    lib.trash(binned)
    absent = lib.item("GONE0001", title="Synced as metadata only")
    lib.attachment("ATTACH05", parent=absent, filename="not_here.pdf", on_disk=False)
    weblink = lib.item("LINK0001", title="Just a URL")
    lib.attachment(
        "ATTACH06", parent=weblink, link_mode=3, raw_path=None, content_type="text/html"
    )

    lib.close()
    return lib.root


# --- the mapping ------------------------------------------------------------------------------ #


def test_a_stored_attachment_becomes_a_document_with_its_item_s_metadata(library: Path) -> None:
    """The metadata belongs to the parent *item*; the attachment row is only the file."""
    scan = zotero.read_library(library)
    by_name = {d.path.name: d for d in scan.documents}
    assert set(by_name) == {"lewis_2020.pdf", "hebb.epub"}

    paper = by_name["lewis_2020.pdf"]
    assert paper.title == "Retrieval-Augmented Generation"
    assert paper.authors == "Patrick Lewis, Ethan Perez"
    assert paper.year == 2020
    assert paper.doi == "10.1000/rag"
    assert paper.item_type == "journalArticle"
    assert paper.path.exists()


def test_the_file_is_found_under_its_own_attachment_key(library: Path) -> None:
    """Zotero files live at `storage/<attachment key>/<name>`, not `storage/<item key>/`."""
    scan = zotero.read_library(library)
    paper = next(d for d in scan.documents if d.path.name == "lewis_2020.pdf")
    assert paper.path.parent.name == "ATTACH01"
    assert paper.path.parent.parent == (library / "storage").resolve()


def test_collections_come_through_as_full_paths(library: Path) -> None:
    """A bare `RAG` says nothing and two collections may share a leaf name."""
    scan = zotero.read_library(library)
    paper = next(d for d in scan.documents if d.path.name == "lewis_2020.pdf")
    assert paper.collections == ("Reading / RAG",)


def test_the_root_offered_is_the_storage_folder(library: Path) -> None:
    """It is the folder that gets registered as a source root, so it must be the one with files."""
    scan = zotero.read_library(library)
    assert scan.root == library / "storage"
    assert scan.label == "Zotero"


# --- what it declines, and why it says so ------------------------------------------------------ #


def test_every_declined_entry_is_counted_under_a_readable_reason(library: Path) -> None:
    """A filter that drops entries silently reads as a broken catalogue."""
    scan = zotero.read_library(library)
    assert scan.total_skipped == 4
    assert set(scan.skipped) == {
        "a web-page snapshot",
        "in the Zotero trash",
        "not downloaded to this computer",
        "a saved link with no file",
    }
    assert all(count == 1 for count in scan.skipped.values())


def test_snapshots_can_be_asked_for(library: Path) -> None:
    """Off by default because a library of any age holds hundreds — but they are real files."""
    scan = zotero.read_library(library, include_snapshots=True)
    assert {d.path.name for d in scan.documents} == {"lewis_2020.pdf", "hebb.epub", "page.html"}
    assert "a web-page snapshot" not in scan.skipped


def test_an_unsupported_file_type_names_the_type(library: Path) -> None:
    lib = _Library(library.parent / "Zotero2")
    item = lib.item("ITEM0001", title="A spreadsheet")
    lib.attachment("ATTACH99", parent=item, filename="data.xlsx", content_type="application/xlsx")
    lib.close()
    scan = zotero.read_library(lib.root)
    assert scan.documents == ()
    assert "an unsupported file type (xlsx)" in scan.skipped


# --- linked files ------------------------------------------------------------------------------ #


def test_an_absolute_linked_file_is_imported_where_it_lives(tmp_path: Path) -> None:
    elsewhere = tmp_path / "Papers"
    elsewhere.mkdir()
    target = elsewhere / "linked.pdf"
    target.write_bytes(b"%PDF-1.4")

    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="Kept outside Zotero")
    lib.attachment("ATTACH01", parent=item, link_mode=2, raw_path=str(target))
    lib.close()

    scan = zotero.read_library(lib.root)
    assert [d.path for d in scan.documents] == [target.resolve()]


def test_a_base_directory_link_is_skipped_rather_than_guessed(tmp_path: Path) -> None:
    """The base directory is a Zotero *preference*, not a database value. Without it there is no
    honest way to resolve the path, and inventing one would import the wrong file."""
    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="Relative to a base directory")
    lib.attachment("ATTACH01", parent=item, link_mode=2, raw_path="attachments:sub/linked.pdf")
    lib.close()

    scan = zotero.read_library(lib.root)
    assert scan.documents == ()
    assert "stored somewhere this app cannot resolve" in scan.skipped


def test_a_base_directory_link_resolves_once_the_base_is_supplied(tmp_path: Path) -> None:
    base = tmp_path / "Attachments"
    (base / "sub").mkdir(parents=True)
    target = base / "sub" / "linked.pdf"
    target.write_bytes(b"%PDF-1.4")

    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="Relative to a base directory")
    lib.attachment("ATTACH01", parent=item, link_mode=2, raw_path="attachments:sub/linked.pdf")
    lib.close()

    scan = zotero.read_library(lib.root, base_dir=base)
    assert [d.path for d in scan.documents] == [target.resolve()]


# --- the awkward shapes ------------------------------------------------------------------------ #


def test_a_standalone_attachment_describes_itself(tmp_path: Path) -> None:
    """An attachment with no parent is its own item — reading a null parent as "no metadata"
    would silently drop the title of every file dragged straight into Zotero."""
    lib = _Library(tmp_path / "Zotero")
    item = lib.item("LOOSE001", title="Dragged straight in", date="2019")
    lib.db.execute(
        "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path) "
        "VALUES (?, NULL, 0, 'application/pdf', 'storage:loose.pdf')",
        (item,),
    )
    folder = lib.storage / "LOOSE001"
    folder.mkdir(parents=True)
    (folder / "loose.pdf").write_bytes(b"%PDF-1.4")
    lib.close()

    scan = zotero.read_library(lib.root)
    assert len(scan.documents) == 1
    assert scan.documents[0].title == "Dragged straight in"
    assert scan.documents[0].year == 2019


def test_an_institutional_author_is_not_prefixed_with_a_space(tmp_path: Path) -> None:
    """`fieldMode = 1` is a single-field name; joining it to an empty first name leaves a space."""
    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="A report")
    lib.creator(item, "", "World Health Organization", mode=1)
    lib.attachment("ATTACH01", parent=item, filename="report.pdf")
    lib.close()

    scan = zotero.read_library(lib.root)
    assert scan.documents[0].authors == "World Health Organization"


def test_editors_are_used_only_when_there_are_no_authors(tmp_path: Path) -> None:
    """An edited volume would otherwise report nobody at all."""
    lib = _Library(tmp_path / "Zotero")
    edited = lib.item("ITEM0001", title="An edited volume")
    lib.creator(edited, "Ada", "Lovelace", role=2)
    lib.attachment("ATTACH01", parent=edited, filename="volume.pdf")
    lib.close()

    scan = zotero.read_library(lib.root)
    assert scan.documents[0].authors == "Ada Lovelace"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("2020-04-12 2020-04-12", 2020),
        ("April 2020", 2020),
        ("2020", 2020),
        ("n.d.", None),
        ("", None),
        (None, None),
        ("12/04/99", None),  # a two-digit year is not a year we can trust
    ],
)
def test_the_year_is_read_out_of_zotero_s_free_text_dates(
    stored: str | None, expected: int | None
) -> None:
    assert zotero._year_from(stored) == expected


def test_a_cycle_in_the_collection_tree_does_not_hang(tmp_path: Path) -> None:
    """Zotero should never produce one; a corrupted or hand-edited database can."""
    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="In a loop")
    lib.attachment("ATTACH01", parent=item, filename="paper.pdf")
    lib.collection(1, "A", parent=2)
    lib.collection(2, "B", parent=1)
    lib.file_in(1, item)
    lib.close()

    scan = zotero.read_library(lib.root)
    assert len(scan.documents) == 1
    assert scan.documents[0].collections  # some path came back, and the read terminated


# --- failing honestly ------------------------------------------------------------------------ #


def test_a_folder_with_no_zotero_database_says_which_folder(tmp_path: Path) -> None:
    with pytest.raises(CatalogueUnavailable) as excinfo:
        zotero.read_library(tmp_path / "not-zotero")
    assert "zotero.sqlite" in str(excinfo.value)


def test_a_database_that_is_not_zotero_is_refused_not_crashed(tmp_path: Path) -> None:
    root = tmp_path / "Zotero"
    root.mkdir()
    connection = sqlite3.connect(root / "zotero.sqlite")
    connection.execute("CREATE TABLE something_else (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(CatalogueUnavailable):
        zotero.read_library(root)


def test_a_missing_optional_table_costs_the_extra_not_the_import(tmp_path: Path) -> None:
    """Collections and creators are enrichment. Losing them must not lose the documents."""
    lib = _Library(tmp_path / "Zotero")
    item = lib.item("ITEM0001", title="Still importable")
    lib.creator(item, "Ada", "Lovelace")
    lib.attachment("ATTACH01", parent=item, filename="paper.pdf")
    lib.db.execute("DROP TABLE collections")
    lib.db.execute("DROP TABLE creators")
    lib.close()

    scan = zotero.read_library(lib.root)
    assert len(scan.documents) == 1
    assert scan.documents[0].title == "Still importable"
    assert scan.documents[0].authors is None
    assert scan.documents[0].collections == ()


def test_reading_never_writes_to_the_users_library(library: Path) -> None:
    """The one guarantee that matters most: their catalogue is not ours to modify."""
    db = library / "zotero.sqlite"
    before = (db.read_bytes(), db.stat().st_mtime)
    zotero.read_library(library)
    assert (db.read_bytes(), db.stat().st_mtime) == before
