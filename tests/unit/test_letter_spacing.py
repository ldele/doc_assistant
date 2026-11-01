"""Letter-spacing the page does not show closes at extraction; every other space stays (ADR-056).

PyMuPDF writes a space wherever two glyphs are drawn apart and marks each one it writes, so a title
set with letter-spacing came out as ``N E U R O N S`` (Cajal's 1906 lecture, 33 runs). Each PDF
here is drawn in the test with one known mechanism, so the suite needs no private file: a run that
must close, and every look-alike that must not. The label row 0.4-0.7 em apart is pinned as
*closing*: it is the case the file cannot tell from letter-spacing, and the ignore list is what
stops it (ADR-056, amendment 2026-10-23).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from doc_assistant.extractors import (
    KEEP_ALL_SPACED,
    apply_letter_spacing,
    close_letter_spacing,
    extract_to_markdown,
    letter_spacing_page_joins,
    user_spacing_is_valid,
)
from doc_assistant.ingest.cache import locate_spacing_runs

SIZE = 10.0
FONT = pymupdf.Font("helv")
WORD_SPACE = 0.28  # Helvetica's space, in em


def _page_joins(tmp_path: Path, pieces: list[tuple[str, float]], *, name: str = "p") -> list:
    """The joins on a page drawn as ``pieces``: (text, gap in em after it), on one baseline."""
    path = _draw(tmp_path, pieces, name=name)
    with pymupdf.open(path) as doc:
        return letter_spacing_page_joins(doc[0])


def _draw(tmp_path: Path, pieces: list[tuple[str, float]], *, name: str = "p") -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    for text, gap in pieces:
        page.insert_text((x, 100), text, fontsize=SIZE, fontname="helv")
        x += FONT.text_length(text, fontsize=SIZE) + gap * SIZE
    path = tmp_path / f"{name}.pdf"
    doc.save(path)
    doc.close()
    return path


def _spaced(word: str, tracking: float) -> list[tuple[str, float]]:
    """``word`` drawn one letter at a time, ``tracking`` em apart, nothing between the letters."""
    return [(ch, tracking) for ch in word]


def _joined(joins: list) -> list[str]:
    return [joined for _, joined, _ in joins]


# --- what closes ---------------------------------------------------------------------------------


def test_a_letter_spaced_title_closes_and_keeps_its_word_break(tmp_path: Path) -> None:
    pieces = [*_spaced("NEURON", 0.3), ("S", 0.3 + WORD_SPACE), *_spaced("STRUCTURE", 0.3)]
    joins = _page_joins(tmp_path, pieces)
    assert joins == [("N E U R O N S S T R U C T U R E", "NEURONS STRUCTURE", "tracked")]


def test_a_spaced_word_in_a_sentence_closes_where_the_pdf_holds_the_word_spaces(
    tmp_path: Path,
) -> None:
    # "Das ist " ends in a space glyph, as a typesetter that writes its spaces draws it
    pieces = [("Das ist ", 0.0), *_spaced("gesperr", 0.25), ("t", 0.25), (" gesetzt", 0.0)]
    assert _joined(_page_joins(tmp_path, pieces)) == ["gesperrt"]


def test_touching_glyphs_with_space_glyphs_between_them_close(tmp_path: Path) -> None:
    # Each letter is drawn with a space glyph after it, and the next letter is drawn over that
    # space: the page shows "https://arxiv", the text layer holds "h t t p s : / / a r x i v".
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    for ch in "https://arxiv":
        page.insert_text((x, 100), ch + " ", fontsize=SIZE, fontname="helv")
        x += FONT.text_length(ch, fontsize=SIZE)
    path = tmp_path / "url.pdf"
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as pdf:
        joins = letter_spacing_page_joins(pdf[0])
    assert joins == [("h t t p s : / / a r x i v", "https://arxiv", "url")]


def test_a_row_of_labels_inside_the_gray_zone_closes(tmp_path: Path) -> None:
    """Pinned as closing on purpose: 0.6 em is inside the range real tracking reaches, so only the
    ignore list can stop it (see the next test)."""
    assert _joined(_page_joins(tmp_path, _spaced("ABCD", 0.6))) == ["ABCD"]


def test_the_ignore_list_keeps_a_run_spaced() -> None:
    joins = [("A B C D", "ABCD", "tracked"), ("N E U", "NEU", "tracked")]
    text, applied = close_letter_spacing("A B C D and N E U", joins, frozenset({"A B C D"}))
    assert text == "A B C D and NEU"
    assert applied == [("N E U", "NEU", "tracked")]


def test_a_whole_document_entry_keeps_every_run_spaced() -> None:
    joins = [("N E U", "NEU", "tracked")]
    text, applied = close_letter_spacing("N E U", joins, frozenset({KEEP_ALL_SPACED}))
    assert (text, applied) == ("N E U", [])


def _touching(tmp_path: Path, before: str, run: str, *, name: str) -> list:
    """The joins on a page holding ``before`` then ``run`` drawn glyph by glyph, each followed by a
    space glyph the next glyph is drawn over (the space has no visible width)."""
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    if before:
        page.insert_text((x, 100), before + " ", fontsize=SIZE, fontname="helv")
        x += FONT.text_length(before + " ", fontsize=SIZE)
    for ch in run:
        page.insert_text((x, 100), ch + " ", fontsize=SIZE, fontname="helv")
        x += FONT.text_length(ch, fontsize=SIZE)
    path = tmp_path / f"{name}.pdf"
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as pdf:
        return letter_spacing_page_joins(pdf[0])


def test_a_run_that_continues_a_url_on_its_line_closes_as_a_url(tmp_path: Path) -> None:
    """A query string holds "=", which alone would read as maths; the URL before it decides."""
    joins = _touching(tmp_path, "https://x.org/wiki.php?", "n=Downloads", name="query")
    assert joins == [("n = D o w n l o a d s", "n=Downloads", "url")]


def test_maths_stays_as_extracted(tmp_path: Path) -> None:
    joins = _touching(tmp_path, "", "a=b+c", name="maths")
    assert joins == [("a = b + c", "a = b + c", "maths")]
    text, applied = close_letter_spacing("so a = b + c here", joins)
    assert (text, applied) == ("so a = b + c here", [])


# --- what stays ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pieces",
    [
        pytest.param([("a r t i c l e  i n f o", 0.0)], id="visible space glyphs"),
        pytest.param(
            [
                ("In", 0.18),
                ("tro", 0.18),
                ("d", 0.18),
                ("u", 0.18),
                ("c", 0.18),
                ("tio", 0.18),
                ("n", 0.0),
            ],
            id="an OCR layer's word fragments",
        ),
        pytest.param(
            [(t, WORD_SPACE) for t in ["d", "=", "m", "+", "n", "+", "1"]],
            id="maths at word spacing",
        ),
        pytest.param(
            [(t, WORD_SPACE) for t in ["for", "x", "y", "z", "we"]], id="variables at word spacing"
        ),
        pytest.param(_spaced("ABCD", 1.2), id="labels far apart (separate lines)"),
        pytest.param(
            [(t, WORD_SPACE) for t in ["The", "structure", "of", "neurons"]], id="a plain sentence"
        ),
    ],
)
def test_spaces_the_page_shows_stay(tmp_path: Path, pieces: list[tuple[str, float]]) -> None:
    assert _page_joins(tmp_path, pieces) == []


def test_sub_and_superscripts_stay(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    for ch, dy, size in (("W", 0, SIZE), ("V", -4, 7.0), ("i", 3, 7.0), ("Q", 0, SIZE)):
        page.insert_text((x, 100 + dy), ch, fontsize=size, fontname="helv")
        x += FONT.text_length(ch, fontsize=size) + 0.2 * size
    path = tmp_path / "scripts.pdf"
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as pdf:
        assert letter_spacing_page_joins(pdf[0]) == []


def test_a_spaced_word_in_a_pdf_without_space_glyphs_stays_spaced(tmp_path: Path) -> None:
    """A known miss, pinned so it stays visible: with no space glyph anywhere, a spaced word in a
    sentence spaced like its words cannot be told from one-letter words."""
    pieces = [("Das", WORD_SPACE), ("ist", 0.25), *_spaced("gesperr", 0.25), ("t", 0.25)]
    assert _page_joins(tmp_path, [*pieces, ("gesetzt", 0.0)]) == []


def _with_image(tmp_path: Path, share: float, name: str) -> Path:
    """A letter-spaced title over an image covering ``share`` of the page's height (full width)."""
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.clear_with(230)
    page.insert_image(pymupdf.Rect(0, 0, page.rect.width, page.rect.height * share), pixmap=pix)
    x = 72.0
    for ch in "NEURONS":
        page.insert_text((x, 100), ch, fontsize=SIZE, fontname="helv")
        x += FONT.text_length(ch, fontsize=SIZE) + 0.3 * SIZE
    path = tmp_path / f"{name}.pdf"
    doc.save(path)
    doc.close()
    return path


def test_a_scanned_page_is_left_alone(tmp_path: Path) -> None:
    """One image under the whole page: an OCR layer, whose glyph positions are estimates."""
    with pymupdf.open(_with_image(tmp_path, 1.0, "scan")) as pdf:
        assert letter_spacing_page_joins(pdf[0]) == []


def test_a_page_with_a_large_figure_is_not_a_scan(tmp_path: Path) -> None:
    with pymupdf.open(_with_image(tmp_path, 0.6, "figure")) as pdf:
        assert _joined(letter_spacing_page_joins(pdf[0])) == ["NEURONS"]


@pytest.mark.parametrize(
    ("box", "closes"), [("formula", False), ("table", False), ("picture", False), ("text", True)]
)
def test_formula_table_and_picture_boxes_are_left_alone(
    tmp_path: Path, box: str, closes: bool
) -> None:
    path = _draw(tmp_path, _spaced("NEURONS", 0.3), name=f"box-{box}")
    with pymupdf.open(path) as pdf:
        page = pdf[0]
        # The layout analysis is pymupdf4llm's model; a box set beforehand stands in for it
        # (`get_layout` keeps information already present).
        page.layout_information = [(50.0, 80.0, 300.0, 110.0, box)]
        assert bool(letter_spacing_page_joins(page)) is closes


# --- what the text pass may do -------------------------------------------------------------------


def test_the_text_pass_only_ever_removes_spaces() -> None:
    page = "# N E U R O N S :  S T R U C T U R E\n\nBody with h t t p s : / / x and A B C D."
    joins = [
        ("N E U R O N S : S T R U C T U R E", "NEURONS: STRUCTURE", "tracked"),
        ("h t t p s : / / x", "https://x", "invisible"),
        ("A B C D", "ABCD", "tracked"),
    ]
    text, applied = close_letter_spacing(page, joins)
    assert text == "# NEURONS: STRUCTURE\n\nBody with https://x and ABCD."
    assert len(applied) == 3
    assert text.replace(" ", "") == page.replace(" ", "")


def test_a_run_is_not_matched_into_a_letter_that_follows_it() -> None:
    text, applied = close_letter_spacing("A B CD", [("A B C", "ABC", "tracked")])
    assert (text, applied) == ("A B CD", [])


@pytest.mark.parametrize(
    ("page", "closed"),
    [
        ("## **N E U** text", "## **NEU** text"),
        ("_N E U_ text", "_NEU_ text"),
        ("`N E U` text", "`NEU` text"),
        ("~~N E U~~ text", "~~NEU~~ text"),
    ],
)
def test_a_run_closes_inside_the_emphasis_around_it(page: str, closed: str) -> None:
    text, applied = close_letter_spacing(page, [("N E U", "NEU", "tracked")])
    assert text == closed
    assert applied == [("N E U", "NEU", "tracked")]


def test_a_run_is_not_matched_out_of_the_word_before_it() -> None:
    text, applied = close_letter_spacing("XA B C", [("A B C", "ABC", "tracked")])
    assert text == "XA B C"
    assert applied == []


@pytest.mark.parametrize("font", ["hebo", "heit"])
def test_a_bold_or_italic_title_closes_in_the_extracted_text(tmp_path: Path, font: str) -> None:
    """The markdown puts the emphasis against the run: it must not read as a word before it."""
    doc = pymupdf.open()
    page = doc.new_page()
    face = pymupdf.Font(font)
    x = 72.0
    for ch in "NEURONS":
        page.insert_text((x, 100), ch, fontsize=SIZE, fontname=font)
        x += face.text_length(ch, fontsize=SIZE) + 0.3 * SIZE
    page.insert_text((72, 140), "A sentence of body text.", fontsize=SIZE, fontname="helv")
    path = tmp_path / f"{font}.pdf"
    doc.save(path)
    doc.close()
    record: list[dict] = []
    text = extract_to_markdown(path, record=record)
    assert "NEURONS" in text
    assert [(r["state"], r["in_text"]) for r in record] == [("closed", True)]


def test_a_title_wrapped_over_two_pdf_lines_closes_both_halves() -> None:
    """pymupdf4llm writes a paragraph's lines on one markdown line, so the second line's spaced run
    follows the first's directly; each join still closes its own glyphs."""
    joins = [("S A N T I A G O", "SANTIAGO", "tracked"), ("R A M Ó N", "RAMÓN", "tracked")]
    text, applied = close_letter_spacing("S A N T I A G O R A M Ó N", joins)
    assert (text, len(applied)) == ("SANTIAGO RAMÓN", 2)


def test_a_joined_form_is_written_literally() -> None:
    """OCR noise can hold a backslash; a replacement string would read it as an escape."""
    text, _ = close_letter_spacing("\\ d x j", [("\\ d x j", "\\dxj", "tracked")])
    assert text == "\\dxj"


def test_a_page_with_nothing_to_close_is_returned_unchanged() -> None:
    page = "Plain text, x y z, and a table | a | b |"
    assert close_letter_spacing(page, []) == (page, [])


# --- what the pass records (ADR-056 points 7 and 10) -----------------------------------------


def test_every_run_comes_back_with_what_became_of_it() -> None:
    runs = [
        {"spaced": "N E U", "joined": "NEU", "kind": "tracked"},
        {"spaced": "A B C", "joined": "ABC", "kind": "tracked"},
        {"spaced": "X Y Z", "joined": "XYZ", "kind": "tracked"},
        {"spaced": "a = b", "joined": "a = b", "kind": "maths"},
        {"spaced": "Q R S", "joined": "QRS", "kind": "tracked"},
    ]
    page = "N E U then A B C then X Y Z then a = b"
    text, outcomes = apply_letter_spacing(page, runs, frozenset({"A B C"}), {"X Y Z": "XY Z"})
    assert text == "NEU then A B C then XY Z then a = b"
    got = [(o["spaced"], o["state"], o["written"], o["in_text"]) for o in outcomes]
    assert got == [
        ("N E U", "closed", "NEU", True),
        ("A B C", "kept", "A B C", True),
        ("X Y Z", "yours", "XY Z", True),
        ("a = b", "maths", "a = b", True),
        ("Q R S", "closed", "QRS", False),
    ]


@pytest.mark.parametrize(
    ("chosen", "valid"),
    [
        ("NEU RONS", True),
        ("NEURONS", True),
        ("NEU  RONS", False),
        (" NEURONS", False),
        ("NEURONZ", False),
        ("NEURON", False),
        ("", False),
    ],
)
def test_a_spacing_may_only_move_spaces(chosen: str, valid: bool) -> None:
    assert user_spacing_is_valid("N E U R O N S", chosen) is valid


def test_a_run_left_in_place_records_the_text_as_it_stands(tmp_path: Path) -> None:
    """Markdown can hold a kept run's word break as three spaces: `written` is what a reader of
    the text finds there, so the run's span can be located."""
    page = "<!-- page:1 -->\nA B   C D here"
    run = {"spaced": "A B C D", "joined": "AB CD", "kind": "tracked", "page": 1}
    text, [kept] = apply_letter_spacing(page, [run], frozenset({"A B C D"}))
    assert text == page
    assert (kept["state"], kept["written"]) == ("kept", "A B   C D")

    runs = [kept]
    locate_spacing_runs(text, runs)
    assert text[runs[0]["start"] : runs[0]["end"]] == "A B   C D"


def test_a_spacing_that_changes_a_letter_is_refused_and_the_rule_decides() -> None:
    runs = [{"spaced": "N E U", "joined": "NEU", "kind": "tracked"}]
    text, [outcome] = apply_letter_spacing("N E U", runs, frozenset(), {"N E U": "NEW"})
    assert (text, outcome["state"]) == ("NEU", "closed")


# --- end to end ----------------------------------------------------------------------------------


def test_extraction_closes_the_title_and_honours_the_ignore_list(tmp_path: Path) -> None:
    path = _draw(tmp_path, [*_spaced("NEURON", 0.3), ("S", 0.0)], name="title")
    assert "NEURONS" in extract_to_markdown(path)
    kept = extract_to_markdown(path, keep_spaced=frozenset({"N E U R O N S"}))
    assert "NEURONS" not in kept
    assert "N E U R O N S" in kept


def test_a_recorded_run_points_at_its_text_in_the_cache(tmp_path: Path) -> None:
    """The record a real extraction writes locates each run in the cached text (ADR-056 points 7
    and 8): what keyword selection will check before it drops the run."""
    from doc_assistant.ingest.cache import read_spacing_record, write_cache

    path = _draw(tmp_path, [*_spaced("NEURON", 0.3), ("S", 0.0)], name="span")
    record: list = []
    text = extract_to_markdown(path, record=record)
    cached = tmp_path / "span.md"
    write_cache(cached, text, source=path, spacing_record=record)
    [run] = read_spacing_record(cached)
    assert text[run["start"] : run["end"]] == run["written"] == "NEURONS"


def test_a_page_whose_glyphs_cannot_be_read_still_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair that failed the document would be worse than the spacing it repairs."""
    from doc_assistant import extractors

    def _broken(page: object) -> list:
        raise RuntimeError("a font PyMuPDF cannot read")

    monkeypatch.setattr(extractors, "letter_spacing_page_runs", _broken)
    path = _draw(tmp_path, [*_spaced("NEURON", 0.3), ("S", 0.0)], name="broken")
    assert "N E U R O N S" in extract_to_markdown(path)


# --- the extraction fingerprint ------------------------------------------------------------------


def _fingerprint(suffix: str) -> str:
    from doc_assistant.ingest.cache import extraction_fingerprint

    extraction_fingerprint.cache_clear()
    return extraction_fingerprint(suffix)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("_HALF_SPACE_EM", 0.2),
        ("_TOUCH_EM", 0.1),
        ("_BASELINE_EM", 0.3),
        ("_CLOSING_MARKS", ":"),
        ("_SCANNED_PAGE_MIN", 0.5),
        ("_NOT_TEXT_BOXES", ("table",)),
        ("_MATH_GLYPHS", "="),
        # the functions extraction runs (`letter_spacing_joins` is a wrapper it does not call)
        ("_line_runs", lambda line: []),
        ("_scanned_page", lambda page: True),
        (
            "apply_letter_spacing",
            lambda page_md, runs, keep=frozenset(), spacing=None: (page_md, []),
        ),
    ],
)
def test_a_change_to_the_rule_re_extracts_pdfs_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    """KI-40: the rule lives in `extractors`, so the fingerprint walk sees its code and its
    constants. A rule kept in another module would change without re-extracting anything."""
    from doc_assistant import extractors

    before = {s: _fingerprint(s) for s in (".pdf", ".epub", ".html")}
    monkeypatch.setattr(extractors, name, value)
    after = {s: _fingerprint(s) for s in (".pdf", ".epub", ".html")}
    assert after[".pdf"] != before[".pdf"]
    assert (after[".epub"], after[".html"]) == (before[".epub"], before[".html"])


def test_no_constant_in_the_pdf_fingerprint_depends_on_the_hash_seed() -> None:
    """The fingerprint hashes each constant's repr. A set's repr order changes with the hash seed,
    so a set (or a dict) among them would make every cache read stale in the next process: that
    is why `_CLOSING_MARKS` is a string."""
    from doc_assistant import extractors
    from doc_assistant.ingest.cache import _extraction_closure

    _, constants, _ = _extraction_closure(frozenset(), ("extract_pdf_pymupdf",))
    assert {
        "_HALF_SPACE_EM",
        "_TOUCH_EM",
        "_BASELINE_EM",
        "_CLOSING_MARKS",
        "_SCANNED_PAGE_MIN",
        "_NOT_TEXT_BOXES",
        "_MATH_GLYPHS",
        "_URL_SHAPE",
    } <= set(constants)
    unordered = {
        name: type(getattr(extractors, name)).__name__
        for name in constants
        if isinstance(getattr(extractors, name), (set, frozenset))
    }
    assert unordered == {}
