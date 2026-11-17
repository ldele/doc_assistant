<!-- status: active · updated: 2026-11-17 (A3: a stand-in AI client on the read tools, twice; the changes between the runs) · class: baseline -->

# Are the read tools enough? A stand-in client tried them (A3)

**The first run showed gaps the tests could not see, and the second ran with them fixed.** A
Claude Code subagent (Sonnet) played an outside AI program. It answered nine realistic tasks on
the 28 public demo papers using only the read tools, and reported where they got in its way.

| | Run 1 (ADR-066's seven tools) | Run 2 (nine tools, pages fixed) |
|---|---|---|
| Tasks done | 4 of 9 | 6 of 9 |
| Partly done | 4 | 2 |
| Could not do | 1 | 1 (no concept exists in this library: the correct answer) |
| Tool calls | 30 | 24 |
| Bytes per search passage | 3,200 | 2,081 |

One client and nine tasks. Treat the table as a list of what blocked, not a score: another client
or other tasks would block elsewhere.

## Method

- **Library.** A copy of the public eval home: 28 papers, one folder (*Demo corpus*, 18
  documents), and 578 conversations left by eval runs. *Re-run › References* was run on the copy
  first (1,131 references, 23 of them linked to papers in the library), so `get_document` had a
  reference list to show. The copy has no concepts.
- **Client.** A Claude Code subagent (Sonnet). It saw only each tool's description and results,
  called the tools through a local test server shaped like A4's endpoint, and was forbidden from
  reading any file, database or source code. Run 2 used a fresh subagent with no memory of run 1.
- **The nine tasks:**
  1. compare two methods with pages;
  2. find papers by an author;
  3. check a reported number against its paper;
  4. describe a folder;
  5. read the text around a passage;
  6. follow a paper's references into the library;
  7. recall a past conversation;
  8. use the person's own definition;
  9. ask about something the library does not hold.
- **Each call was logged** with its tool, arguments and result size.

## What run 1 found, and what changed

| Finding | Cause | Change before run 2 |
|---|---|---|
| `page` was null on every passage and every conversation citation | A parent passage stores its position as character offsets, not a page | Search derives the page from the document's page markers (`library.source_view.page_from_meta`, the source viewer's rule). A conversation's cited excerpt is found in the document's text and given that page. On this copy, 471 of 480 cited passages in the GPipe conversations are placed |
| Every passage came twice, as plain text and fenced | `Passage` carried both | `text` removed; `fenced` holds the same text between its tags |
| No way to read around a passage | No tool for it | `get_passage(passage_id, around)`: the passage with fenced text before and after, read from the document (`library.chunks.get_chunk_context`) |
| No way to tell "no concept" from "wrong name" | No list of concepts | `list_concepts`: the concepts the person took on, never terms |
| "Have I asked about GPipe?" meant paging through 578 conversations | No text filter | `list_conversations(text=…)` keeps conversations whose questions contain it |
| `get_document` closed the connection | The trial copy had not been migrated (the engine migrates at start, the test server did not) | None: a fault of the trial's setup, not of the tool |

The user added the two tools and the filter on 2026-11-17, after confirming that none had been
left out for a security reason. Run 2's report also showed conversation pages as null. That too was
the copy: its stored paths named the original home, and the path guard rightly refused to read
outside the open library. With the paths repointed, the GPipe citations were placed as the table
says.

## What is still open, from run 2

- **Author metadata.** `authors` is null on 15 of the 28 documents and wrong on some (a citation
  string, an affiliation, a truncated list). This is extraction (KI-54, KI-89), not the tools: no
  shape can supply missing metadata. Task 2 stays partly done until it improves.
- **No author filter on `list_documents`.** A field can be added later without a new version.
- **No "nothing relevant" signal.** The AlphaFold search returned weak passages with `note` null.
  A score floor would be a number tuned to this corpus unless it is measured, so the tool returns
  the scores and leaves the judgement to the caller.
- **Near-duplicate conversations.** Most of the 39 GPipe conversations repeat one eval question.
  That is this home's eval history; a person's library would not look like it.
- **Damaged passage text.** Some ColBERT passages read "Qery" and "Te", and DPR's equation is
  missing. This is extraction (release B), not the tools.
