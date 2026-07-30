"""The sparse (keyword) retrieval arm, on disk instead of on the Python heap (ADR-036, KI-32).

**Why this exists.** The BM25 arm used to be built in memory at every launch: a `Document` per
chunk, a token list per chunk, and a `BM25Okapi` frequency dict per chunk, all resident for the
life of the process. Measured at 97 documents / 33,105 chunks that was **195 MB of heap**, linear
in corpus size, which put the app's practical ceiling near 5,000 documents on a 16 GB machine
against a 10,000-document robustness contract. Deduplicating the parent text (KI-32 step 1) took
26% off it; attribution then showed the rest is the index's own structures plus per-chunk Python
object overhead, i.e. things that only go away if the index itself moves off the heap.

**What replaces it.** One SQLite database beside the Chroma store, holding the chunk text and
metadata in a plain table and an **FTS5 full-text index** over the *same token stream*
`keywords.tokenize` produces. A query touches the index and materialises `Document` objects for the
top-K rows only, so steady-state memory is O(K) rather than O(corpus), and folder scoping (ADR-025
F2) becomes a `WHERE doc_hash IN (...)` instead of rebuilding an index over a subset.

**Term parity is enforced, not assumed.** The FTS5 table is declared with
``tokenize="unicode61 tokenchars '-+' remove_diacritics 0"`` and fed the output of
`keywords.tokenize` joined by spaces. That tokenizer reproduces exactly those tokens (the token
regex is ``[a-z0-9]+(?:[-+][a-z0-9]+)*``, already casefolded), so index and query see the same
vocabulary the in-RAM arm did — including ``cross-encoder`` staying one term.

**Ranking is NOT identical, and that is the honest cost.** FTS5's `bm25()` is Okapi BM25 with
k1=1.2, b=0.75 and no IDF floor; `rank_bm25.BM25Okapi` used k1=1.5, b=0.75, epsilon=0.25. Measured
on the live corpus over the ten public eval queries, the two arms agree on **84% of the top-20
candidates**. What the user receives is unchanged far more often than that, because the
cross-encoder re-scores the union of both arms — the end-to-end effect is measured in ADR-036 and
`tests/eval/baselines/sparse_index_2026-07-30.md`, not assumed here.

**Failure policy differs from `bm25_cache` on purpose.** That module was a cache over a structure
that could always be rebuilt in memory, so every unhappy path fell back. This *is* the arm: a
missing or stale database is rebuilt (once, from the store), and a corrupt one is rebuilt too, but
there is no "serve it anyway" path. If the rebuild fails, the caller is told and falls back to the
legacy in-RAM arm (`DOC_SPARSE_INDEX=0` forces that path).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document

from doc_assistant.knowledge.keywords import tokenize

log = structlog.get_logger(__name__)

#: Bumped when the schema or the indexed representation changes. The fingerprint (which includes
#: this) decides staleness, so an older database is rebuilt rather than misread.
_SCHEMA_VERSION = 1

INDEX_FILENAME = "sparse_index.sqlite3"

#: Rows per executemany batch during a build. Structural (bounded memory per batch), not tuned.
_WRITE_BATCH = 2000

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    rowid    INTEGER PRIMARY KEY,
    doc_hash TEXT NOT NULL,
    text     TEXT NOT NULL,
    meta     TEXT NOT NULL
);
CREATE INDEX chunks_doc_hash ON chunks(doc_hash);
CREATE TABLE parents (
    doc_hash     TEXT NOT NULL,
    parent_index INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (doc_hash, parent_index)
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    tok,
    content='',
    tokenize="unicode61 tokenchars '-+' remove_diacritics 0"
);
"""


def index_path(chroma_path: str) -> Path:
    """Sit beside the store it indexes, so the two travel (and are deleted) together."""
    return Path(chroma_path).parent / INDEX_FILENAME


def fingerprint_from_pages(collection_name: str, id_pages: Iterable[Sequence[str]]) -> str:
    """The same identity as :func:`fingerprint`, computed **without holding the ids**.

    Measured, and the reason this exists: accumulating the id list to hash it cost **159 MB of
    working set at 33,105 chunks** (chromadb's paged `get()` plus the Python list), which after
    ADR-036 was the *only* corpus-linear memory left in the launch path — the on-disk index itself
    adds 1 MB and answers queries in 0. Streaming the pages makes launch memory bounded by one
    page.

    **Order-independent by construction, not by sorting.** Sorting is what forced the whole list
    into memory; instead each id's digest is *summed* modulo 2^128, and addition does not care
    what order Chroma pages rows in. The chunk **count** is hashed in as well, so an add and a
    removal cannot cancel out. This is a change detector over UUIDs, not a security primitive.
    """
    from doc_assistant.knowledge import keywords

    total = 0
    count = 0
    for page in id_pages:
        for chunk_id in page:
            total = (total + int.from_bytes(hashlib.sha256(chunk_id.encode()).digest(), "big")) % (
                1 << 128
            )
            count += 1

    digest = hashlib.sha256()
    digest.update(
        f"{_SCHEMA_VERSION}\x00{collection_name}\x00{count}\x00{total:032x}\x00".encode()
    )
    try:
        digest.update(inspect.getsource(keywords.tokenize).encode("utf-8"))
    except (OSError, TypeError):
        digest.update(b"no-source")
    return digest.hexdigest()


def fingerprint(collection_name: str, chunk_ids: list[str]) -> str:
    """Identify the corpus + tokeniser this index was built from.

    The convenience form for callers that already hold every id (tests, small collections). It
    delegates to :func:`fingerprint_from_pages` rather than reimplementing the digest, so the two
    entry points can never drift apart.

    Components, and why each is here — the three ADR-035 arrived at, for the same measured reasons,
    plus this module's schema version:

    * the **chunk ids** — an add, a removal or a `--rebuild` (fresh UUIDs) all move the set, which
      a bare `count()` would miss. **Not** the store file's mtime: *opening* a
      `chromadb.PersistentClient` rewrites it even for a pure read, so an mtime fingerprint
      invalidates on every launch (measured; it never hit once);
    * the **collection name** — an embedding-model switch points retrieval at a different one;
    * the **tokeniser's source** — the index is tokens, so a tokeniser change must invalidate it
      without anyone remembering to bump a constant.

    The tokeniser is imported **inside** the implementation on purpose: a module-level
    `from ... import tokenize` binds the function object at import time, so a later rebinding of
    `keywords.tokenize` would be invisible and the fingerprint would keep vouching for an index
    built with a different tokeniser. Same separate-binding trap `src/doc_assistant/CLAUDE.md`
    records; caught by a guard test rather than by reasoning.
    """
    return fingerprint_from_pages(collection_name, [chunk_ids])


def enabled() -> bool:
    """``DOC_SPARSE_INDEX=0`` restores the legacy in-RAM BM25 arm without a code change."""
    return os.getenv("DOC_SPARSE_INDEX", "1").strip().lower() not in {"0", "false", "no"}


def match_expression(query: str) -> str | None:
    """Turn a user query into an FTS5 MATCH expression, or ``None`` if it has no terms.

    Terms are OR-ed, because the in-RAM arm scored any document containing *any* query term and a
    bare FTS5 term list means **AND** — silently returning far fewer candidates. Each term is
    double-quoted so it is a string literal: an unquoted token could otherwise be read as an FTS5
    operator (``NOT``, ``OR``, ``*``) and change or break the query.
    """
    terms = tokenize(query)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


class SparseIndex:
    """Read handle on the on-disk sparse arm. Construct via :func:`open_index`."""

    def __init__(self, con: sqlite3.Connection, chunks: int) -> None:
        self._con = con
        self.chunks = chunks

    # -- queries ------------------------------------------------------------------------------ #

    def search(self, query: str, k: int, *, scope: frozenset[str] | None = None) -> list[Document]:
        """The ``k`` best-matching chunks, highest BM25 first, as `Document`s.

        ``scope`` (ADR-025 F2) restricts to a set of ``doc_hash`` values. It is applied **inside**
        the ranked query, before ``LIMIT``, so a scoped turn gets its own top-k rather than the
        whole-corpus top-k filtered down to whatever survives — the difference between "the best k
        in this folder" and "however many of the global best k happen to be in this folder".

        Only the returned rows are turned into Python objects, which is the whole point of the
        module: memory is O(k), not O(corpus).
        """
        expression = match_expression(query)
        if expression is None or k <= 0:
            return []
        if scope is not None and not scope:
            return []

        sql = (
            "SELECT c.text, c.meta, bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
        )
        params: list[Any] = [expression]
        if scope is not None:
            placeholders = ",".join("?" * len(scope))
            sql += f" AND c.doc_hash IN ({placeholders})"
            params.extend(sorted(scope))
        # bm25() is negative, most-relevant first at the bottom of the scale, so ASC is "best".
        sql += " ORDER BY score LIMIT ?"
        params.append(k)

        try:
            rows = self._con.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            # A malformed MATCH expression must not take a turn down with it: the arm returns
            # nothing, the vector arm still answers, and the failure is visible in the log.
            log.warning("sparse_index_query_failed", error=str(e))
            return []
        return [Document(page_content=text, metadata=json.loads(meta)) for text, meta, _ in rows]

    def parent_text(self, doc_hash: str, parent_index: int) -> str | None:
        """The parent block a child belongs to (KI-32 step 1's map, now a table)."""
        row = self._con.execute(
            "SELECT text FROM parents WHERE doc_hash = ? AND parent_index = ?",
            (doc_hash, parent_index),
        ).fetchone()
        return str(row[0]) if row else None

    def doc_hashes(self) -> set[str]:
        """Every ``doc_hash`` present. Used to answer "is this scope empty?" without a scan."""
        return {str(r[0]) for r in self._con.execute("SELECT DISTINCT doc_hash FROM chunks")}

    def close(self) -> None:
        self._con.close()

    # -- construction -------------------------------------------------------------------------- #

    @classmethod
    def build(
        cls,
        path: Path,
        fingerprint: str,
        pages: Iterator[tuple[str, dict[str, Any]]],
    ) -> SparseIndex:
        """Build the database from a **stream** of ``(text, metadata)`` and return a read handle.

        Streaming is not a style choice: accumulating the corpus to build an index whose purpose is
        to keep the corpus out of memory would pay the peak it exists to avoid. Rows are written in
        batches and the parent text is deduplicated on the way past, so the build holds one batch
        plus the parent map for the document currently being read.

        Written to a temporary file and moved into place, so an interrupted build cannot leave a
        half-index that the next launch has to detect.
        """
        tmp = path.with_suffix(".building")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        try:
            con.executescript(_SCHEMA)
            n = _fill(con, pages)
            con.execute(
                "INSERT INTO meta(key, value) VALUES ('fingerprint', ?), ('chunks', ?)",
                (fingerprint, str(n)),
            )
            con.commit()
        except BaseException:
            con.close()
            tmp.unlink(missing_ok=True)
            raise
        con.close()

        path.unlink(missing_ok=True)
        tmp.replace(path)
        log.info("sparse_index_built", chunks=n, mb=round(path.stat().st_size / 1e6, 1))
        return cls(_connect(path), n)


def _fill(con: sqlite3.Connection, pages: Iterator[tuple[str, dict[str, Any]]]) -> int:
    """Insert every chunk, its tokens and its parent text. Returns the row count."""
    chunk_rows: list[tuple[int, str, str, str]] = []
    fts_rows: list[tuple[int, str]] = []
    parent_rows: dict[tuple[str, int], str] = {}
    rowid = 0

    def flush() -> None:
        if chunk_rows:
            con.executemany(
                "INSERT INTO chunks(rowid, doc_hash, text, meta) VALUES (?,?,?,?)", chunk_rows
            )
            con.executemany("INSERT INTO chunks_fts(rowid, tok) VALUES (?,?)", fts_rows)
            chunk_rows.clear()
            fts_rows.clear()
        if parent_rows:
            con.executemany(
                "INSERT OR IGNORE INTO parents(doc_hash, parent_index, text) VALUES (?,?,?)",
                [(h, i, t) for (h, i), t in parent_rows.items()],
            )
            parent_rows.clear()

    for text, meta in pages:
        metadata = dict(meta or {})
        doc_hash = str(metadata.get("doc_hash", ""))
        parent = metadata.pop("parent_text", None)
        if parent and doc_hash and metadata.get("parent_index") is not None:
            parent_rows[(doc_hash, int(metadata["parent_index"]))] = str(parent)
        rowid += 1
        chunk_rows.append((rowid, doc_hash, text, json.dumps(metadata)))
        fts_rows.append((rowid, " ".join(tokenize(text))))
        if len(chunk_rows) >= _WRITE_BATCH:
            flush()
    flush()
    return rowid


def _connect(path: Path) -> sqlite3.Connection:
    """Open read-only-ish: the answer path never writes, and `check_same_thread=False` because the
    FastAPI shell serves requests from a threadpool while holding one pipeline."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con


def open_index(path: Path, fingerprint: str) -> SparseIndex | None:
    """Open an existing index if it matches ``fingerprint``; ``None`` if absent, stale or corrupt.

    ``None`` means "build it", never "serve it anyway" — unlike `bm25_cache`, this is the arm
    itself, so a wrong answer here is a wrong answer to the user.
    """
    if not path.exists():
        return None
    try:
        con = _connect(path)
        rows = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if rows.get("fingerprint") != fingerprint:
            con.close()
            log.info("sparse_index_stale", hint="corpus or tokenizer changed; rebuilding")
            return None
        chunks = int(rows.get("chunks", 0))
        log.info("sparse_index_opened", chunks=chunks)
        return SparseIndex(con, chunks)
    except (sqlite3.Error, ValueError, TypeError) as e:
        log.warning("sparse_index_unreadable", error=str(e), path=str(path))
        return None
