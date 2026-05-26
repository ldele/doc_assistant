"""Synthetic citation-corpus generator for Phase 4 testing.

Produces fake-paper markdown that the real `metadata_extractor` and
`citations` modules can chew on. The caller supplies a planned citation
graph; the generator produces N markdown strings whose References sections
implement that graph.

Supports four reference-section formats so the tier-1 regex extractor gets
exercised against all the messy cases observed in the real corpus:

    "clean_apa"     — `- Author, A. B. (year). Title. Journal.`
    "lncs_colon"    — `- Author A.: Title. Journal pp (year)` (real failure)
    "multi_column"  — refs concatenated on one line (Fornito-style)
    "no_heading"    — no "References" heading at all (textbook-style)

Each generated paper has title / authors / year / DOI in the header so the
metadata extractor can populate the Document row. Cross-citations match by
DOI (highest-confidence path), so the test asserts both extractor and
matcher behavior end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RefFormat = Literal["clean_apa", "lncs_colon", "multi_column", "no_heading"]


@dataclass
class FakePaper:
    """Spec for one synthetic paper."""

    paper_id: str  # short stable id, used to build filename
    title: str
    authors: str  # display form, e.g. "Smith, J., Doe, J."
    year: int
    doi: str
    cites: list[str] = field(default_factory=list)  # paper_ids cited
    ref_format: RefFormat = "clean_apa"


# ============================================================
# Per-format reference renderers
# ============================================================


def _render_ref_clean_apa(p: FakePaper) -> str:
    """- Authors. (year). Title. _Journal_. DOI: ...`"""
    return (
        f"- {p.authors} ({p.year}). {p.title}. "
        f"_Journal of Synthetic Studies_, 1, 1-10. "
        f"DOI: {p.doi}"
    )


def _render_ref_lncs_colon(p: FakePaper) -> str:
    """- Authors: Title. Venue pp (year) — known tier-1 weakness."""
    return f"- {p.authors}: {p.title}. In: Synthetic Conf., pp. 1-10 ({p.year})"


def _render_ref_minimal(p: FakePaper) -> str:
    """Minimal ref used inside multi-column lines."""
    return f"{p.authors} ({p.year}). {p.title}. DOI: {p.doi}"


# ============================================================
# Per-format header renderers
# ============================================================


def _render_header(p: FakePaper) -> str:
    return (
        f"## **{p.title}**\n\n"
        f"**{p.authors}**\n\n"
        f"Published {p.year}. DOI: {p.doi}\n\n"
        f"## Abstract\n\n"
        f"This synthetic paper exists to exercise the citation pipeline. "
        f"Any resemblance to real work is intentional structural mimicry.\n\n"
    )


# ============================================================
# Per-format References-section renderers
# ============================================================


def _render_refs_section(
    fmt: RefFormat,
    cited_papers: list[FakePaper],
) -> str:
    if not cited_papers:
        # Even with the "References" heading, an empty section is realistic
        return "## References\n\n"

    if fmt == "no_heading":
        # No "References" heading at all — tier-1 should NOT detect this.
        # Refs still appear as numbered list at the document tail.
        lines = [
            f"{i + 1}. {_render_ref_minimal(p)}"
            for i, p in enumerate(cited_papers)
        ]
        return "\n\n".join(lines) + "\n"

    if fmt == "multi_column":
        # Refs concatenated on a single line, mimicking PDF multi-column extraction.
        # Tier-1 has an `_INLINE_REF_BREAK` fallback for exactly this case.
        numbered = " ".join(
            f"{i + 1}. {_render_ref_minimal(p)}"
            for i, p in enumerate(cited_papers)
        )
        return f"## References\n\n{numbered}\n"

    if fmt == "lncs_colon":
        rendered = "\n\n".join(_render_ref_lncs_colon(p) for p in cited_papers)
        return f"## References\n\n{rendered}\n"

    # Default: clean APA
    rendered = "\n\n".join(_render_ref_clean_apa(p) for p in cited_papers)
    return f"## References\n\n{rendered}\n"


# ============================================================
# Top-level generator
# ============================================================


def render_paper(p: FakePaper, by_id: dict[str, FakePaper]) -> str:
    """Render one paper's full markdown given the corpus dict for citation lookups."""
    header = _render_header(p)
    body = (
        "## Introduction\n\n"
        "This is the body text. It would normally have prose about something.\n\n"
        "## Conclusion\n\n"
        "We did the work and these were the results.\n\n"
    )
    cited = [by_id[cid] for cid in p.cites if cid in by_id]
    refs = _render_refs_section(p.ref_format, cited)
    return header + body + refs


def render_corpus(papers: list[FakePaper]) -> dict[str, str]:
    """Return {filename: markdown} for the whole synthetic corpus.

    Filenames mirror the convention used by the real cache: `<paper_id>.md`.
    """
    by_id = {p.paper_id: p for p in papers}
    return {f"{p.paper_id}.md": render_paper(p, by_id) for p in papers}


# ============================================================
# Convenience: pre-built scenarios
# ============================================================


def chain_scenario() -> list[FakePaper]:
    """A -> B -> C -> D linear citation chain. Tests transitive structure."""
    return [
        FakePaper("paper_a", "Foundations of Foo", "Alpha, A.", 2010, "10.9999/a"),
        FakePaper(
            "paper_b", "Extensions to Foo", "Beta, B.", 2015, "10.9999/b",
            cites=["paper_a"],
        ),
        FakePaper(
            "paper_c", "Applications of Foo", "Gamma, G.", 2020, "10.9999/c",
            cites=["paper_b"],
        ),
        FakePaper(
            "paper_d", "Surveys of Foo", "Delta, D.", 2024, "10.9999/d",
            cites=["paper_c"],
        ),
    ]


def cycle_scenario() -> list[FakePaper]:
    """Two papers citing each other plus a third citing both. Tests dedup + bidirectional."""
    return [
        FakePaper(
            "ring_x", "Ring Theory X", "Xavier, X.", 2018, "10.9999/x",
            cites=["ring_y"],
        ),
        FakePaper(
            "ring_y", "Ring Theory Y", "Yorke, Y.", 2019, "10.9999/y",
            cites=["ring_x"],
        ),
        FakePaper(
            "ring_z", "Ring Theory Survey", "Zane, Z.", 2022, "10.9999/z",
            cites=["ring_x", "ring_y"],
        ),
    ]


def mixed_format_scenario() -> list[FakePaper]:
    """Same graph topology rendered in all four ref formats. Tests format robustness."""
    base_cites = ["paper_a"]  # everyone cites paper_a
    return [
        FakePaper("paper_a", "The Original", "Origin, O.", 2005, "10.9999/orig"),
        FakePaper(
            "paper_clean", "Clean Style", "Clean, C.", 2020, "10.9999/clean",
            cites=base_cites, ref_format="clean_apa",
        ),
        FakePaper(
            "paper_lncs", "LNCS Style", "Lncs, L.", 2020, "10.9999/lncs",
            cites=base_cites, ref_format="lncs_colon",
        ),
        FakePaper(
            "paper_mcol", "Multi Column", "Mcol, M.", 2020, "10.9999/mcol",
            cites=base_cites, ref_format="multi_column",
        ),
        FakePaper(
            "paper_nohdr", "No Heading", "Nohdr, N.", 2020, "10.9999/nohdr",
            cites=base_cites, ref_format="no_heading",
        ),
    ]


def isolated_scenario() -> list[FakePaper]:
    """One paper that cites nothing and is cited by nothing. Tests empty-edge case."""
    return [
        FakePaper("loner", "Solitude", "Hermit, H.", 2024, "10.9999/loner"),
    ]
