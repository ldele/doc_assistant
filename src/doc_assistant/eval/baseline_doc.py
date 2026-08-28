"""A baseline document that carries its own provenance, instead of prose that can be wrong.

``tests/eval/baselines/`` holds this project's committed reference results, and until now
everything about *what produced them* lived in hand-written prose — "Generator: local Ollama
``llama3.1:8b``", "Corpus: the working library, 97 documents". Those sentences are exactly the kind
RG-029 showed can be wrong: the Haiku-vs-llama split across the 2026-08-08 arms existed **only** in
``evals/README.md`` prose, recoverable by a human reading a document and by nothing else. Worse,
the run store the numbers came from (``data/eval.duckdb``) is gitignored, so a fresh clone holds
the conclusions and none of the evidence.

This renders the mechanical half of a baseline **from the run record** — the run-defining settings,
the corpus composition, the generator, and the aggregate table — plus a machine-readable provenance
block. A later run can then be checked against the committed *document*, with no DuckDB required.

**A visible fenced block, not an HTML comment.** The reader sees exactly what the checker reads.
A hidden block invites the two to drift, and this whole layer exists because a claim and its
evidence drifted apart.

**The judgement stays human, and is marked as owed.** The emitter writes the numbers and the setup;
it writes ``TODO`` where a baseline needs a person — what the result means, what would settle it,
which caveats apply. A baseline that only contains numbers is a table, not a record. Compare the
existing hand-written ones: their value is in the caveats (*"this generator scores 36% citation
coverage against Haiku's 81%, so treat the ordering as provisional"*), which no emitter can derive.

**Older baselines parse to nothing, on purpose.** :func:`parse_provenance` returns ``{}`` for the
~30 documents written before this existed, so a check against one reports *unknown* rather than
inventing agreement — the same three-state rule the rest of the layer follows.

Generic by construction: dicts and strings, no ``doc_assistant`` import (ADR-003 Decision 8).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Marks the fenced block below it as the machine-readable record. A literal marker rather than
#: "the first json block" so a baseline may carry other JSON (a config excerpt, a payload) without
#: the checker mistaking it for provenance.
PROVENANCE_MARKER = "<!-- eval-baseline-provenance -->"

#: Bumped when the block's shape changes. A reader that does not recognise the version says so
#: rather than guessing at the fields.
PROVENANCE_SCHEMA = 1

_BLOCK_RE = re.compile(
    re.escape(PROVENANCE_MARKER) + r"\s*```json\s*(?P<body>.*?)```",
    re.DOTALL,
)


def render_provenance(
    *,
    settings: Mapping[str, Any],
    run_ids: Sequence[str],
    scores: Mapping[str, Mapping[str, Any]],
) -> str:
    """The machine-readable block: what produced these numbers, in the run record's own vocabulary.

    ``settings`` is the ``config_json`` the runs agreed on — copied, never re-derived, so the
    document cannot claim a setting the run did not record. A key the runs never recorded is
    simply absent here, and a later check reads that as unknown.
    """
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "run_ids": list(run_ids),
        "n_trials": len(run_ids),
        "settings": dict(sorted(settings.items())),
        "scores": {name: dict(sorted(stats.items())) for name, stats in sorted(scores.items())},
    }
    return "\n".join(
        [
            PROVENANCE_MARKER,
            "```json",
            json.dumps(payload, indent=2, sort_keys=False, default=str),
            "```",
        ]
    )


def parse_provenance(text: str) -> dict[str, Any]:
    """Read a provenance block back, or ``{}`` when the document has none.

    Tolerant by design. Most committed baselines predate this block, and a checker that raised on
    them would simply not be used; returning ``{}`` lets the comparison report *unknown*, which is
    the truthful verdict about a document that never said what produced it. A malformed block is
    also ``{}`` — a half-parsed record is worse than an absent one, because it looks authoritative.
    """
    match = _BLOCK_RE.search(text)
    if match is None:
        return {}
    try:
        loaded = json.loads(match.group("body"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def baseline_settings(text: str) -> dict[str, Any]:
    """Just the settings from a baseline document — ``{}`` when it carries no provenance."""
    provenance = parse_provenance(text)
    settings = provenance.get("settings")
    return dict(settings) if isinstance(settings, dict) else {}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-" if value is None else str(value)


#: A results row: ``| `contains_all` | 0.777 | 0.004 | 105 | 0 |``
_RESULT_ROW_RE = re.compile(r"^\|\s*`(?P<scorer>[a-z_]+)`\s*\|\s*(?P<mean>[-\d.]+)\s*\|", re.M)


def table_drift(text: str) -> list[str]:
    """Rows where the document's visible table disagrees with its own provenance block.

    The failure this catches is small and nasty: someone tidies a number in the pretty table — or
    copies a document and edits it — while the machine-readable block still carries the original.
    The table is what a human quotes and the block is what the checker reads, so a drift between
    them turns this whole layer into a second, better-hidden version of the problem it exists to
    solve.

    Compares only the mean, and only for scorers the block names. Returns ``[]`` for a document
    with no provenance (there is nothing to disagree with) — absence is handled by
    :func:`parse_provenance`, not here.
    """
    scores = parse_provenance(text).get("scores")
    if not isinstance(scores, dict):
        return []
    problems: list[str] = []
    for match in _RESULT_ROW_RE.finditer(text):
        scorer = match.group("scorer")
        recorded = scores.get(scorer)
        if not isinstance(recorded, dict):
            continue
        shown, expected = match.group("mean"), _fmt(recorded.get("mean"))
        if shown != expected:
            problems.append(
                f"{scorer}: the table shows {shown} but the provenance block records {expected}"
            )
    return problems


def _setup_rows(settings: Mapping[str, Any]) -> list[str]:
    """The setup table, grouped so a reader meets the corpus before the knobs.

    ``not recorded`` is printed for a missing key rather than the row being dropped: a baseline
    whose corpus is unknown must say so on its face, since that is precisely the defect that made
    the 2026-06-06 sweep unauditable for two months.
    """
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("Corpus", ("index_doc_count", "index_doc_digest")),
        ("Cases", ("n_cases",)),
        ("Generator", ("llm_provider", "llm_model")),
        ("Judge", ("judge_provider", "judge_model")),
        ("Embedder", ("embedding_model",)),
        (
            "Chunking",
            (
                "parent_chunk_size",
                "parent_chunk_overlap",
                "child_chunk_size",
                "child_chunk_overlap",
            ),
        ),
        (
            "Retrieval",
            (
                "use_parent_child",
                "use_multi_query",
                "top_k",
                "candidate_k",
                "bm25_weight",
                "rerank_candidate_cap",
            ),
        ),
    ]
    rows = ["| Group | Setting | Value |", "|---|---|---|"]
    for label, keys in groups:
        for i, key in enumerate(keys):
            # `.get` with a default keeps the two absences apart, which matters: a key that
            # is present and null is a run that ran and could not tell (index composition
            # on a degraded keyword arm), while a missing key is a run from before it existed.
            value = settings.get(key, "**not recorded**")
            rows.append(f"| {label if i == 0 else ''} | `{key}` | {_fmt(value)} |")
    return rows


def _results_rows(scores: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = [
        "| Scorer | Mean | Trial-mean std | n_scored | n_skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(scores.items()):
        rows.append(
            f"| `{name}` | {_fmt(stats.get('mean'))} | {_fmt(stats.get('trial_mean_std'))} | "
            f"{_fmt(stats.get('n_scored'))} | {_fmt(stats.get('n_skipped'))} |"
        )
    return rows


def render_baseline(
    *,
    title: str,
    date: str,
    settings: Mapping[str, Any],
    run_ids: Sequence[str],
    scores: Mapping[str, Mapping[str, Any]],
    notes: Sequence[str] = (),
) -> str:
    """The full document: what ran, what it scored, and what a human still owes it.

    ``date`` is passed in rather than read from the clock so the caller decides what the document
    is dated — and so rendering is deterministic and testable.
    """
    unknown = [k for k in ("index_doc_digest", "llm_model") if k not in settings]
    lines = [
        f"# {title} ({date})",
        "",
        "> **TODO — the judgement.** What does this result mean, and what would change it? A"
        " baseline whose value is only its numbers is a table; the caveats are the record. Say"
        " which scorer to trust here and why, and what would settle what it leaves open.",
        "",
    ]
    if unknown:
        lines += [
            "> **⚠ This baseline does not pin " + ", ".join(f"`{k}`" for k in unknown) + ".**"
            " The runs behind it never recorded those, so nothing here — and nothing later —"
            " can show what corpus or model produced these numbers. Treat it as a reading, not"
            " as a reference to compare against.",
            "",
        ]
    lines += ["## Setup", "", *_setup_rows(settings), "", "## Results", "", *_results_rows(scores)]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    lines += [
        "",
        "## Provenance",
        "",
        "Copied from the run record, not re-derived. `scripts/compare_runs.py --against` reads the"
        " block below to check a later run against this document — the run store it came from is"
        " gitignored, so this block is the only evidence a fresh clone has.",
        "",
        render_provenance(settings=settings, run_ids=run_ids, scores=scores),
        "",
    ]
    return "\n".join(lines)
