"""One-shot regenerator for the committed extraction fixtures. NOT a test — run by hand.

    uv run --no-sync python tests/fixtures/documents/make_fixtures.py

Why the outputs are committed rather than built at test time: a document written by the same
library that reads it can only prove a round-trip. These files are *frozen artifacts* — once
committed they stop tracking `ebooklib`'s output, so a test against them keeps asserting what a
real file on disk does, which is the variance a round-trip cannot reach. Regenerate only when you
mean to change what is being asserted, and say so in the DEVLOG.

`article.html` is hand-authored and lives next to this script; it is not generated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Deliberately exercised, because each one has broken an extractor somewhere:
#   - a nav/TOC document (ebooklib generates one; the extractor currently emits it as prose)
#   - non-ASCII in body, heading and metadata
#   - an HTML entity, which must arrive decoded rather than literal
#   - nested markup inside a paragraph (the text must survive the tags)
#   - a heading level below h1, to prove level mapping is not hardcoded
CHAPTER_ONE = """<html><body>
<h1>Cortical Microcircuits</h1>
<p>The layer-five pyramidal neuron integrates input across a dendritic tree that
spans several hundred micrometres.</p>
<h2>M&#233;thodes</h2>
<p>Recordings were made at 32&#176;C in artificial cerebrospinal fluid.</p>
<p>Emphasis <em>inside</em> a paragraph must not <strong>split</strong> the sentence.</p>
<blockquote>A quoted passage from Ram&#243;n y Cajal.</blockquote>
<ul><li>First listed item</li><li>Second listed item</li></ul>
</body></html>"""

CHAPTER_TWO = """<html><body>
<h1>R&#233;sultats</h1>
<p>Na&#239;ve estimates overstated the effect by a factor of two.</p>
<h3>A third-level heading</h3>
<p>Closing paragraph of the second chapter.</p>
</body></html>"""


def build_epub(path: Path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("provenote-test-fixture-001")
    book.set_title("A Treatise on Cortical Microcircuits")
    book.set_language("en")
    book.add_author("Delacroix, Renée")

    chapters = []
    for i, (name, title, content) in enumerate(
        [
            ("ch1.xhtml", "Cortical Microcircuits", CHAPTER_ONE),
            ("ch2.xhtml", "Résultats", CHAPTER_TWO),
        ],
        start=1,
    ):
        ch = epub.EpubHtml(title=title, file_name=name, lang="en")
        ch.content = content
        book.add_item(ch)
        chapters.append(ch)
        del i

    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)


def main() -> int:
    target = HERE / "treatise.epub"
    build_epub(target)
    size = target.stat().st_size
    print(f"wrote {target.relative_to(HERE.parents[2])} ({size:,} bytes)")
    if not (HERE / "article.html").exists():
        print("WARNING: article.html is missing and is NOT generated here (hand-authored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
