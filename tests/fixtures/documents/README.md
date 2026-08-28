# Extraction fixtures — real documents on disk

Committed document files that the extractors are run against, asserted by
[`tests/unit/test_extraction_fixtures.py`](../../unit/test_extraction_fixtures.py).

## Why these exist

`tests/unit/test_extractors_formats.py` builds each fixture with the same library that reads it
back. That proves the round-trip and the structure heuristics, and it says in its own docstring
that it **cannot** prove anything about files produced by anything else.

These files close that gap. They are **frozen artifacts**: once committed they stop tracking what
`ebooklib` happens to emit today, so a test against them keeps asserting what a real file on disk
does. That is the variance a round-trip cannot reach.

| File | Kind | What it exercises |
|---|---|---|
| `treatise.epub` | binary (zip) | Dublin Core title, two chapters, `h1`/`h2`/`h3`, a generated nav/TOC document, entity-encoded accents, blockquote, list, inline `<em>`/`<strong>` |
| `article.html` | hand-authored text | `<head><title>`, `<style>`, `<nav>`, `<script>`, `<footer>`, article body with `h1`/`h2`/`h3`, byline, a table with a caption, an ordered reference list, entity-encoded accents and maths symbols, inline `<em>`/`<strong>`/`<a>` |

`article.html` carries unique markers (`NAVIGATION_CHROME_MARKER`, `FOOTER_CHROME_MARKER`,
`SCRIPT_BODY_MARKER`) so a chrome-removal failure names the tag that survived instead of just
reporting that something did.

## Three defects are pinned here

The fixtures found three real extraction bugs. They are recorded as `xfail(strict=True)` — the test
**fails when the bug is fixed**, so whoever fixes it is told to come back and update the
expectation, rather than the test quietly passing and leaving a stale comment behind.

1. **EPUB emits its own navigation document as prose** — `get_items_of_type(ITEM_DOCUMENT)`
   returns the generated `nav.xhtml` alongside real chapters.
2. **HTML leaks `<head><title>`** — it lands above the real `<h1>`, indistinguishable from an
   opening sentence.
3. **Inline markup fragments sentences** — `get_text(separator="\n")` breaks a line at every tag
   boundary, so `Emphasis <em>inside</em> a sentence` becomes three lines. This is the one with
   teeth: scientific prose italicises constantly (gene names, species, emphasis).

None is fixed, because each changes extraction *content* and therefore `doc_hash` for every
affected document (ADR-042) — a decision, not a patch. Full write-up in `docs/DEVLOG.md`
(2026-08-20).

## Regenerating

`article.html` is hand-authored — edit it directly, never generate it.

`treatise.epub` comes from:

```bash
uv run --no-sync python tests/fixtures/documents/make_fixtures.py
```

Regenerate **only** when you mean to change what is asserted, and say so in the DEVLOG. `*.epub` is
marked `binary` in `.gitattributes` so line-ending normalisation cannot corrupt the container; the
test suite asserts the committed file still starts with `PK`.
