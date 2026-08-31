"""Optional producers for the source registry — reference managers, e-book catalogues (ADR-049).

**An adapter is never a dependency.** Provenote's SQLite is the system of record (user, 2026-07-02)
and every feature here works with nothing installed; an adapter only offers a faster way to *point
at* files the user already has, plus the metadata their catalogue already holds.

The package is split so the vendor boundary is structural rather than a promise:

- `catalogue` — **neutral**. `ExternalDocument`, the shape every adapter returns, and the reading
  and writing of `ExternalMetadata`. Nothing here knows a vendor schema, and everything downstream
  (the API, the library, the UI) talks only to this half.
- `zotero` — **vendor**. The one module that knows what a `linkMode` is. It reads and returns
  neutral records; it writes nothing.

Adding Calibre means adding one module beside `zotero` and one route; it means changing nothing in
`catalogue`, `registry`, or the client. That is the spec's ADR-3
(`docs/specs/feature-selective-ingestion.md`) made concrete.
"""

from .catalogue import (
    ExternalDocument,
    ExternalScan,
    apply_external_metadata,
    external_for_path,
    record_external,
)

__all__ = [
    "ExternalDocument",
    "ExternalScan",
    "apply_external_metadata",
    "external_for_path",
    "record_external",
]
