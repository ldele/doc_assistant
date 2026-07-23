"""Seed the ANZSRC 2020 Fields of Research trunk into the taxonomy (ADR-028, increment 1).

Writes the bundled `data/anzsrc_2020_for.json` — the ANZSRC divisions (+ groups, when present)
— as abstract field nodes (`Concept(kind="domain", source="anzsrc")`) and their group→division
`in_field` hierarchy edges. **$0, zero-LLM, deterministic, idempotent**: each field node keys on a
stable UUID derived from its ANZSRC code, so a re-run matches (no duplicates); edges go through
`knowledge.taxonomy.add_hierarchy_edge`, which is idempotent on its unique key and rejects cycles.

The seed is CC-BY data; this runner prints the required attribution on every run (a licence
obligation, ADR-028 D7). The About/Settings UI surfacing is a later frontend increment.

Usage:
    python -m scripts.seed_taxonomy              # dry-run: report what would be written
    python -m scripts.seed_taxonomy --apply      # write the domain nodes + in_field edges
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from doc_assistant.db.models import Concept
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.taxonomy import add_hierarchy_edge

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# The bundled seed data file (committed; not a runtime artifact under the gitignored data/ paths).
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "anzsrc_2020_for.json"

# ANZSRC's official linked-data URI stem — a stable, principled namespace for the derived node
# ids, so a code always maps to the same Concept.id across runs and machines.
_URI_STEM = "http://linked.data.gov.au/def/anzsrc-for/2020/"


def anzsrc_node_id(code: str) -> str:
    """The stable Concept.id for an ANZSRC field code (deterministic UUID5, code-derived)."""
    return str(uuid5(NAMESPACE_URL, _URI_STEM + code))


def load_seed(path: Path = DATA_FILE) -> dict:
    """Read + parse the bundled ANZSRC seed data file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _print_attribution(meta: dict) -> None:
    print(
        meta.get("attribution", "ANZSRC 2020 FoR — © Australian Bureau of Statistics, CC BY 4.0.")
    )
    print(f"  Licence: {meta.get('license', 'CC BY 4.0')}  <{meta.get('license_url', '')}>")


def seed_taxonomy(*, apply: bool, path: Path = DATA_FILE) -> tuple[int, int, int]:
    """Seed the ANZSRC trunk. Returns ``(n_domains, n_groups, n_edges)`` written-or-would-write.

    Idempotent: re-running with ``--apply`` is a no-op (same stable ids, same edge keys). A dry
    run (``apply=False``) reports the counts without touching the DB.
    """
    data = load_seed(path)
    divisions = data.get("divisions", [])
    groups = data.get("groups", [])

    _print_attribution(data.get("_meta", {}))

    if not apply:
        print(
            f"\nDry run — would seed {len(divisions)} division domain(s) + {len(groups)} group "
            f"domain(s) + {len(groups)} in_field edge(s). Pass --apply to write."
        )
        return len(divisions), len(groups), len(groups)

    n_domains = 0
    n_edges = 0
    with session_scope() as session:
        # Field nodes (divisions + groups) — upsert by stable id.
        for row in [*divisions, *groups]:
            node_id = anzsrc_node_id(row["code"])
            if session.get(Concept, node_id) is None:
                session.add(
                    Concept(id=node_id, label=row["label"], kind="domain", source="anzsrc")
                )
            n_domains += 1
        session.flush()  # so the group→division edge inserts see the domain rows

        # group --in_field--> division edges (idempotent + acyclic via add_hierarchy_edge).
        for group in groups:
            add_hierarchy_edge(
                session,
                anzsrc_node_id(group["code"]),
                anzsrc_node_id(group["division"]),
                "in_field",
            )
            n_edges += 1

    n_groups = len(groups)
    print(
        f"\nSeeded {n_domains} domain node(s) ({len(divisions)} divisions + {n_groups} groups) "
        f"+ {n_edges} in_field edge(s). Idempotent — re-run is a no-op."
    )
    return n_domains, n_groups, n_edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the domain nodes + in_field edges (default: dry-run).",
    )
    args = parser.parse_args()

    from doc_assistant.config import LOG_JSON, LOG_LEVEL
    from doc_assistant.logging_config import configure_logging

    configure_logging(json=LOG_JSON, level=LOG_LEVEL)

    seed_taxonomy(apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
