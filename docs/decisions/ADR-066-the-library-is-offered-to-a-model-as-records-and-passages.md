<!-- status: active · updated: 2026-11-13 (accepted as proposed: all nine choices, the seven tools as written); 2026-11-12 (proposed; every choice decided by the user in a grill the same day; not built) · class: append-only -->

# ADR-066 — The library is offered to a model as records and passages: an MCP endpoint in the engine, lookups by rule in the chat, and modes that only change the framing

- **Status:** proposed (Claude Code, 2026-11-12); **accepted as proposed by the user on
  2026-11-13**: option A, the nine choices and the seven read tools as written. Nothing is built.
  *As proposed:* The nine choices below were each put to the user
  on 2026-11-12 and each is recorded with the user's answer. Not accepted until the user has read
  this file; nothing is built. This is ROADMAP rows 65 (the MCP server), 102 (the chat looks
  things up in the library) and 37 (chat modes), designed together because they share one read
  surface.
- **Date:** 2026-11-12
- **Deciders:** user + Claude Code
- **Extends** [ADR-054](ADR-054-concepts-and-terms-names-and-forms.md) (a concept is a name with
  its forms; `get_concept` returns exactly that) and
  [ADR-062](ADR-062-the-app-stays-repairable.md) (one engine owns the library; a second program on
  it is the fault *in use*). **Corrects** ROADMAP row 65, which described *a local stdio server
  over `pipeline.py`*: choice 2 puts the server inside the engine. **Depends on** security step
  S-9, the prompt fence, which lands before any of this.
- **In one sentence:** an AI program outside Provenote can search the library and read its
  records through seven read-only tools served by the running engine (MCP, the Model Context
  Protocol, is the standard such programs use to call tools), switched off until the person turns
  it on; the chat answers questions about the library's records by rule and by command, the same
  on every model; a chat mode changes only the framing above the citing block; and each of these
  is the small first form of something meant to grow.

## To decide — nine choices, each as the user decided it

1. **What a search returns to an outside AI.** *Decided:* passages, each with its document, page
   and a stable id. There is no tool that asks Provenote's own model for an answer.
   - The outside AI writes the answer, so nothing on this side calls a model or bills one (KI-4).
   - *Cost:* the integrity layer (evidence apart from interpretation, the reviewer) does not run
     on what an outside AI writes. The tool's description says so.
   - *Not taken:* an `ask` tool returning the app's cited answer. The user: *"I think ask is not
     useful."* It would spend the configured provider's tokens at an outside program's request.
2. **Where the server runs.** *Decided:* inside the running engine, on its local address. A small
   bridge program speaks stdio (the client starts it as a command and talks over its input and
   output) and forwards to the engine, for clients that can only start a command.
   - One process owns the library, and the switch in Settings takes effect at once.
   - *Cost:* Provenote must be running for an outside AI to reach the library, and the bridge
     needs its own access token, since it is not the app's window (choice 9).
   - *Not taken:* a separate server process that opens the library itself. It would load its own
     embedding model (about 1.8 GB of memory, as KI-84 measured for an ingest run) and be a second
     program on one library.
3. **The read tools.** *Decided:* seven. The names are working names until the build.

   | Tool | Returns |
   |---|---|
   | `search_library` | ranked passages for a question, optionally within a folder or a list of documents |
   | `list_folders` | the folders, with their document counts |
   | `list_documents` | documents with title, authors, year and type, a page at a time, optionally one folder |
   | `get_document` | one document's metadata, its reference list and its figures' captions |
   | `list_conversations` | past conversations with their titles and derived labels (choice 7) |
   | `get_conversation` | one conversation's turns, each answer with the passages it cited |
   | `get_concept` | a concept's name, its exact and broad forms, and the definition the person chose |

   - Every result names its source (a document, a page, an id) and is honest on an empty library:
     an empty list and a count of zero, never an error and never an invented row.
   - *Not taken, for now:* the gap list and the library summary. The gap list's signals are of
     uneven trust (`docs/knowledge-layer.md`), and an outside AI would repeat them as fact.
4. **Writes.** *Decided:* none in this release. The next release adds a second switch, *Allow
   changes*, with three tools: correct a document's metadata, put a document in a folder,
   re-extract one document. There is never a delete tool.
   - The fence has one release of use before a tool can change anything.
   - Re-extraction waits on KI-43 (a document's identity is its extracted content, so
     re-extracting orphans its concepts and marks) and on the slow extraction track.
5. **How the chat decides a question is about records.** *Decided:* by rule and by command. The
   router that already answers *how many documents* without a model (`query_router.py`) is
   extended to documents' metadata, folders and concepts, and commands name a lookup outright.
   The model is never asked to choose.
   - The same on every model, no extra model call, testable at no cost.
   - An answer that came from records says so, in the answer and in its provenance.
   - *Cost:* a record question phrased outside the rules goes to normal retrieval, which may
     answer a count from passages. The eval's records family measures how often.
   - *Not taken, for now:* one routing call that returns strict JSON (one more model call per
     turn), and tool calling by the model (two code paths, different per model).
6. **Chat modes.** *Decided:* a mode is a name and the framing text above the citing block. Two
   are built in: today's, and one more. A person can write their own.
   - The citing block is not editable: `synthesis._CITATION_RE` parses it.
   - The template is hashed per turn, so provenance names the mode that answered.
   - A person's own mode is marked *not evaluated* in an answer's provenance. A built-in mode has
     a run on every model of the eval panel before it ships.
   - *Not taken, for now:* skills, meaning reusable recipes of several steps. Nothing in this
     release needs a mode to do more than change the framing.
7. **Labels on a conversation.** *Decided:* a conversation shows the documents and folders its
   answers cited. They are read from the answer records each time; nothing new is stored.
   - The chat list and `list_conversations` can filter by a document or a folder.
   - One command asks a question within a concept's documents or a folder, and the label follows
     because it is derived.
   - *Not taken, for now:* tags a person types, and tags from extracted keywords. A library the
     installer builds has no keywords (ROADMAP row 109), and the keyword index can be stale after
     a delete (KI-82).
8. **What is frozen.** *Decided:* the tools' names, parameters and result shapes, under a version
   number the server reports. The app's own HTTP routes stay free to change.
   - A result may gain a field in a later version. It never loses one or changes a field's
     meaning without a new version.
9. **Switches and safety.** *Decided:*
   - The endpoint is off until the person turns it on, in Settings.
   - Turning it on shows an access token and a client setup to paste. The token can be replaced,
     which cuts off every client holding the old one.
   - Reading conversations has its own switch, off by default. With it off, a client sees five
     tools.
   - Passages in a tool result carry the same fence as in the chat's prompt (S-9).
   - Every call is logged: the tool, the time and the number of results. The question's text is
     never logged.

## In practice — one question, three ways (designed, not run)

Nothing here is built, so this is the intended behaviour, written on the public demo corpus (the
papers `tests/eval/corpus_manifest.yaml` pins). The question is the first public eval case:
*What are the two RAG formulations introduced by Lewis et al., and how do they differ?*

1. **In the chat, on any model.** No rule matches: this is a question about content. Retrieval
   runs as today and the model answers with citations. Nothing changes for this question.
2. **From an outside AI.** The person has turned the endpoint on and pasted the setup into their
   AI program. They ask it the same question. It calls
   `search_library(question="RAG formulations Lewis", top_k=5)` and receives, for each passage:
   the text inside the fence, the document's title and authors, the page, and an id. It writes
   its own answer and cites the pages it was given. Provenote called no model. The app log gains
   one line: `search_library`, the time, 5 results.
3. **A record question, in the chat.** The person asks *which documents are in the Demo corpus
   folder?* A rule matches. The answer is the folder's list, read from the library's records, and
   it opens with a line saying it came from the records and not from passages. It reads the same
   on a cloud model and on `llama3.1:8b`, because no model wrote it.
4. **The same record question, phrased outside the rules:** *what have I got on retrieval?* No
   rule matches, so it goes to retrieval and is answered from passages, with citations. That is a
   correct answer to a different question. Whether this happens too often is what the eval's
   records family counts.
5. **On an empty library.** `list_documents` returns an empty list and a count of zero.
   `search_library` returns no passages and says the library holds no documents. The chat's rule
   answers *there are no documents yet*.

## Context

- **Three roadmap rows share one need:** a read-only surface over the library, in
  `src/doc_assistant/`, that both an outside AI and the chat can use (rows 65 and 102).
- **The chat has no tool calling.** The only structured model output in the code is the figure
  description (`ingest/figures.py`). It does have a rule-based router, `query_router.py`, in use
  since the first releases.
- **Local models differ.** Citation coverage on the same prompt was 81% on a small cloud model
  and 36% and 14% on two local ones (KI-36). A design that leans on the model choosing a tool
  would behave differently on each.
- **The knowledge layer is uneven.** Curated concepts and chosen definitions are the person's own
  and can be trusted. Extracted keywords and several gap signals cannot yet
  (`docs/knowledge-layer.md`), and an installed library has no keywords at all (row 109).
- **Prior art read on 2026-11-12:** one open-source RAG server's MCP documentation. It offers
  three tools, all read-only: retrieval, a list of datasets and a list of chats. Its code was not
  read. The user asked for that shape as the starting point.
- **The user's direction, 2026-11-12:** *"I'd like to build progressively toward something that
  is ideal … for now, I'd like to have something that stays more manageable but that we can
  improve later on."* Keywords, tags and concepts could be known to the tools once keyword
  extraction and epistemics improve; they are *"still lacking today"*.

## Options

- **A — records and passages, served by the engine, lookups by rule.** The nine choices above.
- **B — the model drives.** The chat's model calls the tools itself, and an `ask` tool gives an
  outside AI the app's cited answer. More capable on a strong model. It behaves differently per
  model, costs a model call per step, and puts a tool in reach of document text in its first
  release.
- **C — a standalone server.** As row 65 was written: a separate process over `pipeline.py`. It
  works with the app closed. It costs a second embedding model in memory and a second program on
  the library.

## Decision

**Accepted 2026-11-13 (user): A, as proposed.** A smaller first set (three or four tools, without
the conversation tools and their switch) was offered and not taken: the seven tools stay. The
user's steer stands: the endpoint is not meant to be complete in its first form and grows through
the seams below. *As proposed:*

**A**, with the nine choices as the user decided them. Build order: the read surface (session
A3), then the endpoint and its bridge (A4), then the chat's rules and commands (A6), then modes
(A7). The eval's cases for records and for hostile passages are written against the seven tools
and the rules before A6.

**What each piece is the first form of.** The user asked for something manageable now that can
grow. Each choice leaves a named seam:

| Now | The seam | What it can grow into |
|---|---|---|
| Seven read tools | versioned result shapes that may gain fields (choice 8) | the gap list and the summary, once their signals are measured; three write tools behind *Allow changes* |
| `get_concept` returns curated concepts only | the same tool and shape | extracted keywords and epistemic markers on a concept, once they can be trusted |
| Labels derived from answer records | the filter on the chat list and on `list_conversations` | tags a person types, and tags from keywords, beside *every instance of what the user chose* (row 106) |
| Lookups by rule and command | the lookup functions are the tools' own, so a caller can change | one routing call, then tool calling by the model, if the eval's records family shows the rules miss too much |
| Modes change the framing | the mode is hashed per turn | a mode with a first step, then skills |
| The endpoint needs the app running | the tools live in `src/doc_assistant/`, not in the server | a standalone mode, if the library must be reachable with the app closed |

## Consequences

- **Easier:** an outside AI can use the library with no copy of it leaving the computer unless
  that AI sends it on; a record question gets the same answer on every model; a new tool is a
  function in `src/doc_assistant/` and a thin wrapper.
- **Harder:** the tools' shapes are a promise once a client is written against them. Rules need
  upkeep as phrasing is found that they miss.
- **New reach, to state in `docs/security.md`:** an MCP client receives document text and may
  act on it with tools of its own. The fence marks that text as data, but this app cannot make
  another program respect it. The endpoint is outside the launch token that guards the window's
  writes, which is why it has its own.
- **A new dependency or a hand-written protocol:** no MCP package is in `pyproject.toml` today.
  Whichever is chosen is audited like any other.
- **To revisit:** the rules against a routing call, when the records family has a number;
  writes, when the next release opens.

## Enforced by

Planned; nothing is built yet.

- One test per tool on an empty library: an empty result, a zero count, no error.
- A test that every tool result names a source for each item it returns.
- A test that the endpoint refuses a call when the switch is off, and when the token is wrong or
  has been replaced.
- A test that a client sees five tools while the conversations switch is off.
- A test that a passage in a tool result is inside the fence, sharing the chat's own check.
- A snapshot of each tool's result shape under its version: a removed or renamed field fails.
- A test that a rule-answered turn records *from the library's records* in its provenance, and
  that a person's own mode records *not evaluated*.
- The eval's records, not-in-the-library and hostile-passage families (ROADMAP row 113).

## Confidence

- **Read in the code, 2026-11-12:** the router and its patterns; the absence of tool calling in
  the chat; that an answer record holds the passages and documents an answer used; the routes
  the tools would stand on.
- **Not verified:**
  - that the engine can serve MCP beside its own routes;
  - that a stdio bridge starts from an installed Windows build;
  - how many record questions the rules miss. No number exists; the records family supplies it.
- **Decided on judgment, not measurement:** seven tools and not five or nine; two built-in
  modes; a second switch for conversations. Each is cheap to change before a client depends on
  it, and costly after.
