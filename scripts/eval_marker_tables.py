"""Compare table engines on the caption-gated candidate pages (RTX-machine eval).

Marker is ML-based and CPU-slow / GPU-fast — run this on the RTX box. It
emits three artifacts per document so you can judge which engine to trust:

  1. The caption-gated candidate pages (cheap; works anywhere).
  2. pdfplumber's extracted tables, rendered as markdown.
  3. Marker's full-document markdown (if `marker-pdf` is installed),
     saved to disk so you can read off its rendering of the candidate pages.

Marker is NOT a project dependency and is never imported in-process — it
cannot co-resolve with our pinned torch/transformers stack. Instead it runs
out-of-process in an isolated environment, fetched on demand by ``uvx``:

    uvx --from marker-pdf marker_single ...   # no project install needed

(If you have a ``marker_single`` already on PATH it is used directly.) Marker
runs only on the caption-gated candidate pages via ``--page_range``, so it is
far cheaper than a full-document conversion.

Usage:
    python -m scripts.eval_marker_tables --doc <hash>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select

from doc_assistant.config import DATA_PATH, MARKER_PYTHON
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.regions import analyze_pages
from doc_assistant.ingest.tables import extract_tables_from_pages, render_table_markdown
from scripts.extract_tables import _resolve_pdf_path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = DATA_PATH / "tables_debug"


# Marker is ML-slow on CPU; the gated pages still need a generous ceiling.
_MARKER_TIMEOUT_S = 3600


def _marker_command() -> list[str] | None:
    """Resolve how to invoke Marker *out-of-process*.

    Marker cannot co-resolve with our pinned torch/transformers stack, so it is
    never imported in-process (see the module docstring). Prefer a
    ``marker_single`` already on PATH; otherwise fetch it on demand via ``uvx``.
    Returns the command prefix, or ``None`` if neither is available.
    """
    local = shutil.which("marker_single")
    if local is not None:
        return [local]
    uvx = shutil.which("uvx")
    if uvx is not None:
        return [uvx, "--python", MARKER_PYTHON, "--from", "marker-pdf", "marker_single"]
    return None


def _to_marker_page_range(pages: list[int]) -> str:
    """Map 1-based classifier pages → Marker's 0-based, comma-separated --page_range."""
    return ",".join(str(p - 1) for p in sorted(set(pages)))


def _marker_to_markdown(
    pdf_path: Path, pages: list[int], out_dir: Path, *, paginate: bool = False
) -> str:
    """Run isolated Marker on the gated ``pages`` → markdown.

    Shells out to ``marker_single`` (never imports ``marker``), confined to the
    caption-gated candidate pages via ``--page_range``. Returns the markdown
    Marker wrote; raises ``RuntimeError`` if Marker is unavailable or fails.

    ``paginate`` adds ``--paginate_output`` so per-page boundaries survive (the 4a
    ingest path needs them for page attribution). NOTE: confirm the flag name
    against the pinned marker-pdf version at build time (see the 4a spec).
    """
    base = _marker_command()
    if base is None:
        raise RuntimeError("neither 'marker_single' nor 'uvx' found on PATH")
    marker_out = out_dir / "marker_out"
    marker_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        *base,
        str(pdf_path),
        "--output_format",
        "markdown",
        "--output_dir",
        str(marker_out),
        "--page_range",
        _to_marker_page_range(pages),
    ]
    if paginate:
        cmd.append("--paginate_output")
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, timeout=_MARKER_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"marker_single exited with code {proc.returncode}")
    md_files = sorted(marker_out.glob("**/*.md"))
    if not md_files:
        raise RuntimeError(f"Marker produced no markdown under {marker_out}")
    return md_files[0].read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=str, required=True, help="doc_hash or id prefix")
    parser.add_argument(
        "--skip-marker", action="store_true", help="Only emit candidate pages + pdfplumber tables"
    )
    args = parser.parse_args()

    with session_scope() as session:
        row = session.execute(
            select(Document.filename, Document.source_original).where(
                Document.doc_hash.startswith(args.doc), Document.format == "pdf"
            )
        ).first()
    if row is None:
        print(f"No PDF document matching '{args.doc}'.")
        return 1

    filename, source_original = row
    pdf_path = _resolve_pdf_path(source_original, filename)
    if pdf_path is None:
        print(f"Source PDF not found for {filename}.")
        return 1

    out_dir = OUT_DIR / Path(filename).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Candidate pages (cheap) — via the page classifier.
    classifications = analyze_pages(str(pdf_path))
    pages = [c.page for c in classifications if c.is_table_candidate]
    print(f"{filename}: table-candidate pages = {pages}")
    for c in classifications:
        if c.is_table_candidate or c.kind in ("chart", "photo", "figure"):
            print(f"  p{c.page}: {c.kind} ({c.reason})")

    # 2. pdfplumber tables on those pages.
    pp_tables = extract_tables_from_pages(str(pdf_path), pages)
    pp_md = "\n\n".join(render_table_markdown(t) for t in pp_tables) or "(none)"
    pp_path = out_dir / "pdfplumber_tables.md"
    pp_path.write_text(pp_md, encoding="utf-8")
    print(f"\npdfplumber extracted {len(pp_tables)} table(s) -> {pp_path}")

    # 3. Marker's markdown (slow; runs out-of-process on the gated pages only).
    if args.skip_marker:
        print("\n--skip-marker set; not running Marker.")
        return 0
    if not pages:
        print("\nNo table-candidate pages — nothing for Marker to convert.")
        return 0
    if _marker_command() is None:
        print(
            "\nMarker unavailable — skipping engine comparison.\n"
            "  This script shells out to an isolated Marker; it needs either\n"
            "  'marker_single' on PATH or 'uvx' (ships with uv) to fetch it:\n"
            "    uvx --from marker-pdf marker_single ...\n"
            f"  Then re-run. Compare Marker's rendering of pages {pages}\n"
            f"  against {pp_path.name}."
        )
        return 0

    print(f"\nRunning isolated Marker on pages {pages} (slow on CPU)...")
    try:
        marker_md = _marker_to_markdown(pdf_path, pages, out_dir)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"Marker run failed: {e}")
        return 1
    marker_path = out_dir / "marker_gated.md"
    marker_path.write_text(marker_md, encoding="utf-8")
    print(f"Marker markdown -> {marker_path}")
    print(
        f"\nCompare by hand: open {marker_path.name} and read its rendering of "
        f"pages {pages} against {pp_path.name}. Which engine reproduces the "
        f"table structure (rows/columns/headers) faithfully?"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
