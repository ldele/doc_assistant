"""Describe figures with a VLM and persist the description (Feature 4c).

Reads the 4b ``Figure`` sidecar rows (caption + PNG crop on disk), calls Claude
vision with a schema-first tool-use prompt, and writes a natural-language
description back to ``Figure.vlm_description``. A re-``ingest`` then turns each
described figure into a retrievable ``chunk_type='figure'`` chunk
(``caption + description``).

This is the project's only **paid, API-only** enrichment, so it is gated:
  * only figures with a rendered PNG are eligible (caption-only rows are skipped);
  * a figure whose caption is already long enough is skipped as self-describing;
  * a per-document call budget (``MAX_VLM_CALLS_PER_DOC``) caps cost.
Every skip records a ``vlm_call_skipped_reason`` so the decision is auditable.

Enrichment-Layer Pattern: writes the ``Figure`` sidecar only — never the chunk
store (ingest does that on the next run). Idempotent: a figure that already has
a description is left alone unless ``--force``.

Usage:
    python -m scripts.describe_figures                 # dry-run (no API calls)
    python -m scripts.describe_figures --apply         # describe + persist
    python -m scripts.describe_figures --doc <hash>    # one doc only
    python -m scripts.describe_figures --apply --force # re-describe everything
    python -m scripts.describe_figures --apply --max-calls 10   # tighter budget
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import or_, select, update

from doc_assistant.config import FIGURE_VLM_MODEL, MAX_VLM_CALLS_PER_DOC
from doc_assistant.db.models import Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.figures import (
    SKIP_BUDGET_EXHAUSTED,
    SKIP_IMAGE_MISSING,
    AnthropicVisionDescriber,
    FigureDescriber,
    describe_figure,
    should_describe,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_figures(document_id: str) -> list[tuple[str, str | None, str | None, str | None]]:
    """Read ``(id, caption, image_path, vlm_description)`` for a doc's figures."""
    with session_scope() as session:
        rows = session.execute(
            select(Figure.id, Figure.caption, Figure.image_path, Figure.vlm_description)
            .where(Figure.document_id == document_id)
            .order_by(Figure.page, Figure.id)
        ).all()
    return [(str(r[0]), r[1], r[2], r[3]) for r in rows]


def _write_updates(updates: dict[str, dict[str, object]]) -> None:
    with session_scope() as session:
        for fig_id, values in updates.items():
            session.execute(update(Figure).where(Figure.id == fig_id).values(**values))


def _describe_doc(
    document_id: str,
    filename: str,
    *,
    apply: bool,
    force: bool,
    describer: FigureDescriber | None,
    model: str,
    max_calls: int,
) -> dict[str, object]:
    """Gate + (optionally) describe one document's figures. Returns a stats row."""
    figs = _load_figures(document_id)
    row: dict[str, object] = {
        "filename": filename,
        "figures": len(figs),
        "described": 0,
        "skipped": 0,
        "errors": 0,
        "note": "",
    }

    updates: dict[str, dict[str, object]] = {}
    planned = 0
    described = skipped = errors = 0

    for fig_id, caption, image_path, existing in figs:
        if existing and not force:
            continue  # already described — idempotent, left untouched

        do_describe, reason = should_describe(caption, image_path)
        if not do_describe:
            updates[fig_id] = {"vlm_call_skipped_reason": reason}
            skipped += 1
            continue

        if planned >= max_calls:
            updates[fig_id] = {"vlm_call_skipped_reason": SKIP_BUDGET_EXHAUSTED}
            skipped += 1
            continue
        planned += 1

        if not apply or describer is None:
            described += 1  # dry-run: would describe
            continue

        png = Path(str(image_path))
        if not png.exists():
            updates[fig_id] = {"vlm_call_skipped_reason": SKIP_IMAGE_MISSING}
            skipped += 1
            planned -= 1  # never actually called — don't spend the budget on it
            continue
        try:
            desc = describe_figure(png, caption, describer, model=model)
            updates[fig_id] = {
                "vlm_description": desc.to_text(),
                "vlm_call_skipped_reason": None,
            }
            described += 1
        except Exception as e:  # per-figure isolation: one bad call doesn't abort the doc
            updates[fig_id] = {"vlm_call_skipped_reason": f"error: {type(e).__name__}"}
            errors += 1

    if apply and updates:
        _write_updates(updates)

    row["described"] = described
    row["skipped"] = skipped
    row["errors"] = errors
    return row


def _format_report(rows: list[dict[str, object]], *, apply: bool, model: str) -> str:
    total_figs = sum(int(r["figures"]) for r in rows)
    total_described = sum(int(r["described"]) for r in rows)
    total_skipped = sum(int(r["skipped"]) for r in rows)
    total_errors = sum(int(r["errors"]) for r in rows)

    out: list[str] = []
    out.append("=" * 76)
    out.append(f"VLM model:                 {model}")
    out.append(f"Documents with figures:    {len(rows)}")
    out.append(f"Total figures:             {total_figs}")
    out.append(f"  {'Described' if apply else 'Would describe'}:{'':>14}{total_described}")
    out.append(f"  Skipped (gated/budget):  {total_skipped}")
    out.append(f"  Errors:                  {total_errors}")
    out.append("=" * 76)
    out.append("")
    out.append(f"{'filename':<48} {'figs':>4} {'desc':>4} {'skip':>4} {'err':>4}")
    out.append("-" * 76)
    for r in sorted(rows, key=lambda x: (-int(x["described"]), str(x["filename"]))):
        out.append(
            f"{str(r['filename'])[:47]:<48} "
            f"{int(r['figures']):>4} "
            f"{int(r['described']):>4} "
            f"{int(r['skipped']):>4} "
            f"{int(r['errors']):>4}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Call the VLM and persist descriptions"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-describe figures that already have a description"
    )
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=MAX_VLM_CALLS_PER_DOC,
        help="Per-document VLM call budget (default %(default)s)",
    )
    parser.add_argument(
        "--model", type=str, default=FIGURE_VLM_MODEL, help="Vision model (default %(default)s)"
    )
    args = parser.parse_args()

    if args.apply:
        # The VLM is Anthropic-only by ADR (vision + tool-use; no local path), so
        # this run always bills. Route it through the shared cost guard for the
        # loud warning + abort window it otherwise lacked.
        from doc_assistant.llm import ProviderCostError, assert_provider_intent

        try:
            assert_provider_intent(
                "anthropic",
                operation="figure VLM description (paid, API-only)",
                model=args.model,
                scope="each eligible figure (caption- and budget-gated)",
            )
        except ProviderCostError:
            print("ANTHROPIC_API_KEY is not set — cannot run --apply. Set it in .env first.")
            return 1

    with session_scope() as session:
        stmt = (
            select(Document.id, Document.filename)
            .join(Figure, Figure.document_id == Document.id)
            .where(Document.is_archived.is_(False))
            .distinct()
        )
        if args.doc:
            stmt = stmt.where(
                or_(Document.doc_hash.startswith(args.doc), Document.id.startswith(args.doc))
            )
        docs = [(str(r[0]), str(r[1])) for r in session.execute(stmt).all()]

    if not docs:
        print("No documents with figures matched. Run `extract_figures --apply` first.")
        return 1

    describer: FigureDescriber | None = AnthropicVisionDescriber() if args.apply else None
    print(
        f"Processing {len(docs)} document(s) with figures... "
        f"(apply={args.apply}, force={args.force}, model={args.model})"
    )
    rows = [
        _describe_doc(
            doc_id,
            filename,
            apply=args.apply,
            force=args.force,
            describer=describer,
            model=args.model,
            max_calls=args.max_calls,
        )
        for doc_id, filename in docs
    ]
    print(_format_report(rows, apply=args.apply, model=args.model))
    if not args.apply:
        print("\nDry run. Pass --apply to call the VLM and persist descriptions (paid API).")
    else:
        print("\nNote: re-run `ingest` to pull described figures into retrieval as figure chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
