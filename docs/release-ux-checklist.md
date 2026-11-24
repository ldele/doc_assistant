<!-- status: active · updated: 2026-11-24 (0.8.0 refresh: §1 one list for every ingestion and an add while the engine starts, A8″a; §2 the letter-spacing block's place, A8″b, and the Text addressed to an AI block, A2b; §3 the start screen, a chat's setup, Continue in a new chat, the chat list, what a turn shows, chat modes, answers from the records, scoped questions, long chats, the source card's mark, A2-A8b, and which rows use the real library; §5 AI programs, A4; §7 the 0.7.3 to 0.8.0 upgrade, the downgrade refusal and the reinstall page, Tauri 2.12); 2026-11-20 (§0: the provenance fold is named How this answer was made, A8b); 2026-11-11 (§7: from 0.7.2 to 0.7.3 every PDF reads *changed*, KI-91; 0.7.3 refresh: rows for indexing filling blank metadata, selection in both lists, the library summary, the concept view's graph, Settings' door and its confirms, buttons and dialogs; §6: a failing block reads as a 500, the engine gone as *Failed to fetch*; 2026-11-06: §6: a row for the title bar, ADR-065; 2026-11-04: 0.7.2 refresh: rows for the first-run checklist, the indexing screen, the Re-index confirmation, Settings closing during a run, Provider & model's presets and servers, opening another library, and the upgrade's one-time re-read; 2026-11-01: §2: the letter-spacing block, ADR-056; 2026-10-31: §7: the upgrade from the previous installer, KI-80; §2: a referenced file reaches the bin only by choice; 2026-10-30: §1: keyword search survives an add, KI-77; §2: a deleted document is not cited, KI-78; §3: the override's real lifetime; §1: the setup card and the count follow an add, KI-75; §0: a question at 0 documents reaches no model, KI-74; 2026-10-26: §6: how to check the sidecar ends with the window, KI-71; §6: the freeze behind the answer-engine row was KI-67; §7: RG-012 asks as the window does; §6: the answer-engine-unreachable row, the user's feedback; 2026-10-20: 0.7.0 refresh: rows for the extraction pool, Manage keywords at size, proposed placements, ranked keywords, the KL1 labels, names and written case, the Definition card, term signals, no console window, S-4/S-5/S-6; 2026-09-17: §3 hostile-markup row, S-3) · class: runbook -->

# Release UX/UI walkthrough — what to drive before every release

The gate that `docs/RELEASE.md` §5b points at. `preflight` and the test suites prove the code;
this proves the **product**: every surface a user can reach is driven once, in the **frozen
build**, by a person, with the outcome written down. It exists because the same three things
kept happening — a feature "construction-verified" in July was still unproven in September
(`docs/ui-checklist.md` §3), the 2026-08-28 walkthrough found three defects in the first feature
it drove (KI-51), and the app showed a *zero-state* instead of a connection error when its
backend was dead (REVIEWS 2026-08-28), which no test can see.

**Rules of the walkthrough**

1. **Drive the installed artifact, not the dev loop.** Vite HMR and the dev sidecar hide packaging
   defects (KI-34 shipped a build that could not read a PDF while every gate was green).
2. **$0.** Ollama for generation; force it — `.env` defaults are all-Anthropic (KI-4). Name the model
   in the record: citation coverage is model-dependent (KI-36).
3. **Drive the failure branch too.** A surface is not verified until its empty state, its error
   state and its "unavailable" state have each been seen once. Three states, never two (ADR-044).
4. **A blocker is a data-loss or a lie.** Anything that deletes, hides or mislabels what the user
   has → fix before the tag. Anything else → a KI, listed under "Known limits" in the CHANGELOG (§2
   of `RELEASE.md`), and the release proceeds.
5. **Record it.** One row per surface in the table at the foot of this file *for the release being
   cut* (copy the table into `.claude/release-notes-X.Y.Z.md`), then a log line in
   `.claude/REVIEWS.md` row 6 stating what was **not** driven. An unstated omission reads as
   coverage six months later.

**This file is stale until refreshed** (`docs/RELEASE.md` §0). Before driving it: list what shipped
since the previous tag, add or amend a row per surface — including its empty and failed states —
and bump the header. `release_preflight` fails the release if this file has no commit since the
last tag, on purpose: an untouched checklist is evidence that nobody looked, not that nothing changed.

Two libraries are needed: an **empty data home** (first run) and the **real library** (~100
documents, folders, keywords, a taxonomy with placements, at least one referenced root and one
scanned PDF). Most rows say which.

---

## 0 · First run (empty data home)

- [ ] Launch → the readiness banner names every outstanding step (corpus **and** provider), not just one.
- [ ] Provider setup: no key → `not configured`; wrong key → the verification fails with a sentence, the key is not echoed back anywhere (`key_hint` shows 4 characters at most); Ollama down → `unreachable`, with the action named.
- [ ] Set the data home; the empty-corpus banner offers *Add documents* and the sample-question chips are clickable.
- [ ] Ask a question with 0 documents: an honest "no documents" answer, no traceback, no spinner that never ends. No model is called (KI-74): *How this answer was made* says so, and Ollama's log shows no `/api/generate`. Ask once more in the same chat, and once in an empty folder: the same, naming the folder.
- [ ] Getting started is a checklist (0.7.2): two rows, **Model** and **Documents**, each with its status and a link to its tab, and no key field; with no model the Model row says so and links to Provider & model; stop the engine and the Model row turns red; both green once a model answers and a document is indexed.
- [ ] Settings → About shows the version that `preflight` checked; Updates → *Check now* returns one of exactly three states (`newer` / `current` / `unknown`), and `unknown` is what you get offline.

## 1 · Add documents (both libraries)

- [ ] Picker: one file · several files · a folder. Drag-and-drop the same three (never driven on 2026-08-28 — File Explorer needs full-tier access).
- [ ] Review sheet: a duplicate reads as a duplicate, of the library or of another file in the same selection (KI-76); an unsupported format says why; a scanned PDF is warned about **before** indexing and stays ticked, not discovered as a 0-chunk document. The warning needs a box with no `tesseract` on `PATH`: where one is, the scan is read and there is nothing to warn about.
- [ ] Size caps (security S-1): add a file renamed to `.epub` that is not a zip → the sheet says it is damaged, in a sentence; with `DOC_MAX_INGEST_BYTES=1000000` set before launch, a larger PDF is refused with its size and the limit. Nothing crashes and nothing is indexed.
- [ ] The limit is stated up front: the Add documents dialog, the empty-library card and the drop overlay each show the formats line **and** "Up to 1 GB per file" (it follows `DOC_MAX_INGEST_BYTES` if set). These three render only in the desktop window — never verified in a browser.
- [ ] Walk cap (security S-2): with `DOC_MAX_ADD_FILES=5` set before launch, drop a folder of 6+ files → the sheet shows the one-sentence refusal naming 5 where the rows would be; Cancel works.
- [ ] Both placement modes: **copy** lands in the library folder; **reference** registers the root and the file stays where it was.
- [ ] Batch header with > 50 files: the `shown` cap and "and N more" render; make one file fail (rename it mid-run) → `stopped_early`, "Keep the N", `applyError` all render.
- [ ] Index now → the ingest chip shows progress; the document is answerable **before** the chip says done (controller reload).
- [ ] After the first add the chat's setup card is gone and the status bar shows the new count, with no relaunch; after **Undo all** the count follows (KI-75).
- [ ] With documents already in the library, add one more: Settings still shows the keyword index on disk, not *unavailable* (KI-77: on Windows each such add used to turn it off until a relaunch, which an add from empty never shows).
- [ ] Add five or more PDFs at once with the default workers → extraction finishes and the chip completes; nothing hangs (the pool spawns instead of forking since 0.7.0, `16d1a4b`; the hang was seen on Linux only, so this row proves the installed Windows build still runs it).
- [ ] **Undo all** after index-now removes the document row, its chunks *and* the referenced root (KI-51, all three parts).
- [ ] **The indexing screen** (0.7.2, ROADMAP 31): during a run the status bar reads *Indexing… m/n done* beside its bar; it and Settings' *View indexing* open the screen, which lists every file with its status and what happened. *Pause* (amber) waits for the file in progress and names it, then *Paused at m of n*; searching and chatting work meanwhile; *Resume* goes on. *Stop* (red) asks first, with the counts, and Escape closes only the question; a stopped run lists *Not started* files and offers *Index the remaining N*. Include one unreadable file: it shows *Failed* with its reason, and *Retry the one that failed* takes it alone. Closing the screen leaves the run going; a stopped run stays on the status bar until dismissed. Empty state: the screen opened before any run says so.
- [ ] **Re-index asks first** (0.7.2): Settings → Documents → *Re-index* opens a confirmation counting new, changed and already indexed files; before any run on this computer it says there is no estimate yet, and after one it gives a duration and where it was measured. *Only index new files* changes the counts and the button; with no new file it cannot start. A first *Index folder* (empty library) starts at once.
- [ ] **One list for every ingestion** (0.8.0, A8″a): start an index of several files, then on the indexing screen use *Add to queue* in the header → the new files join the list as waiting rows and run after the current ones, in order; a re-run started from a document while the list runs waits too, and its dialog says how many items run first and is not locked. *Remove* on a waiting row takes it out (it is never counted); a row already being read cannot be removed. Nothing is refused because something else runs, except *Undo all* once one of its files is being read: record what the window says then (the engine answers 409; the spec names no sentence, so a raw error here is a KI). While the list runs the chat line reads *Indexing: answers take longer*, and a question still gets its answer. A finished re-run shows a `Done` pill.
- [ ] **An add while the engine starts** (0.8.0, A8″a, T-007): add files in the first seconds after launch → the status bar reads *Waiting for the engine · N papers to add*, and the review sheet opens once the engine is ready. Make the engine fail to start (port 8001 held by another program) → *… not added: the engine did not start* with *Try again* and *Dismiss*; nothing is lost.
- [ ] **Indexing fills blank metadata** (0.7.3): add two PDFs whose first page prints a title and authors → once indexed, each shows its title, authors and year (and DOI if printed) in the grid and in *Edit metadata*, with no *Re-run… › Metadata*. No author list is a sentence of body text (KI-89). Edit one title, change its file, Re-index: the edit stays (ADR-013), and an extracted value is not replaced either. A document whose text yields nothing keeps its blanks and the library summary counts it as missing metadata. A document indexed before 0.7.3 is not filled by a Re-index; *Re-run… › Metadata* fills it.

## 2 · Library (real library)

- [ ] Grid ⇄ list; the filter strip; the keyword-filter overlay; search scoped to a collection with "Search all".
- [ ] **Selection** (0.7.3, ROADMAP 78): in the grid and the list, a Shift- or Ctrl-click outside select mode starts it; a click or Ctrl-click toggles one document; a Shift-click gives the range from the last document clicked the clicked one's new state. The bar reads *Select all* / *Select none* and ends with *Done* (filled), which leaves select mode; nothing in the bar deletes.
- [ ] **The library summary** (0.7.3, ROADMAP 112): the *Summary* layout, for the whole library and for an open folder, which has its way back. *Need attention* leads; each count opens exactly that many documents in the grid, and a zero is plain text; meters read "N of M". Each *From indexing* row names its fix: files that failed in this session's run open the indexing screen, other files Settings › Documents, document issues the grid. The Metadata card says indexing fills new or changed documents and *Re-run… › Metadata* reads older ones. At 0 documents, and in an empty folder, it says so in a sentence; what was never looked for reads *not looked for*, not 0. The layout is not remembered: the Library reopens on documents. Light, dark and 375 px.
- [ ] Folders: create · rename · assign · delete; a folder-scoped chat only cites that folder (ADR-025 F2) and the provenance card says so.
- [ ] Open a document: the five blocks with the jump index; chunks stay unfetched until opened; the block you left open is remembered.
- [ ] Metadata editor: an edit survives a per-part **metadata** re-ingest (ADR-013 wins over ADR-049 wins over the extractor).
- [ ] Figures panel + full-size viewer with zoom; references block; Connections panel (`1st/2nd/3rd`, no raw score); taxonomy placement modal (attach ticks the rollup; a cycle is refused with a sentence).
- [ ] Re-ingest dialog: each part states its cost before it runs; `text` re-chunks without a duplicate document; a batch from *Select* mode continues past one failing document.
- [ ] Delete: asks; defaults to library-only; a **referenced** file goes to the bin only if you choose that, and the dialog names its real path (ADR-046). Then ask about what you deleted: it is not cited (KI-78), and the status bar's count has moved.
- [ ] Source viewer: opens at the cited page; *Fit page* is the default; zoom and the split are draggable; a non-PDF degrades to extracted text and says why; an unreachable file is one sentence naming the path.
- [ ] Manage keywords: the graph-include toggle changes the "On the graph (N)" count immediately.
- [ ] Manage keywords at library size (ROADMAP 96): it opens at once on the real library; a form typed into *Add a form* lands on the row; a concept's form can be marked *exact* or *broad* and a term's cannot; Delete shows what it removes first and, for a concept, offers *Take off the graph instead*; every door to the view says *Manage keywords* (GLOSSARY U-001) — Library filter, Graph tab button, a concept's *Edit*, a term's link.
- [ ] Taxonomy view → *Proposed placements* (ROADMAP 51): the document and concept proposals are listed; **Accept** turns one into a placement that survives a rebuild, **Reject** removes it; with no `is_a` proposal (the case on the development library) the pane says there is none instead of showing a blank.
- [ ] A document's keywords (ROADMAP 99): listed main topic first, by score; none of the known junk — table residue (`br`), letter-spaced fragments, dates or bare numbers, venue fragments, cited surnames; the re-ingest dialog says where keywords are redone.
- [ ] **Where the letter-spacing block sits** (0.8.0, A8″b, T-009): it is at the foot of the document page, after References, with a fold in its heading. It opens by itself while the document has runs and no rule of yours, or a change waits for a re-run; once a rule is set it is folded, and a folded block with a waiting change says how many in its heading. Your fold or unfold holds until another document is shown.
- [ ] **Text addressed to an AI** (0.8.0, A2b, GLOSSARY U-005): ingest a document holding a line written to an AI (a request to ignore instructions, a fake system message) → its page shows the *Text addressed to an AI* block with the line; the library summary counts it; nothing in the text is changed or removed. A paper that only quotes such text is marked too, and the block says a mark asks you to look, not that the document is bad. A document with none shows no block. *Re-run… › Text checks* redoes it.
- [ ] A document with letter-spaced text (ADR-056): *Letter-spacing on this document* lists each run beside a crop of its page; *Show on page*, *Edit spacing* and *Keep spaced* each do what they say, and the row says what the next re-run will change; *Re-run text now…* opens with *Text & chunks* ticked, and when it finishes the block shows what the text now holds without reopening the document. A document with none shows no block. *All entries in Settings* lands on the list; *Remove all*, then *Undo*.
- [ ] A document with **no keywords**, a **0-chunk** document and a document whose root is **unavailable** each render honestly (badge, not a blank).

## 3 · Chat (real library, Ollama)

*0.8.0 (decided 2026-11-24): the rows about how the chat is set up, what a turn shows and the
records run on the real library; the rows that judge answer and citation quality run on the
public papers, as RG-012 does.*

- [ ] **The start screen** (0.8.0, A8a, wireframe `[s2]`-`[s4]`): New chat shows the mode chips (*Research assistant*, *Critical reading*; a person's own modes would show too, but no screen creates one yet), *Search in* with *All documents* set apart and one tickbox per folder, the union line ("N documents"), the sample questions for the chosen mode, and *Make this my default*. Ticking a folder unticks *All documents*; unticking the last folder ticks it again. In the dark theme the ticks and the chosen chip follow the accent. At 0 documents the start screen still shows, under the first-run steps. Defaults whose folders were all deleted say so, and ticking a folder replaces them; they never read as the whole library.
- [ ] **A chat keeps its setup** (A8a): ask in a two-folder chat → the provenance line reads "Searched **A**, **B** only: N documents."; reopen the chat later → the same setup is shown and used. In a chat with turns `/mode <name>` changes nothing and offers *Continue in a new chat* with that mode; on an empty chat it sets the chip.
- [ ] **Continue in a new chat** (A8a, `[s5]`, `[s6]`): from a chat line's menu, *Continue in a new chat…* opens a new chat that says where it continued from and folds the carried exchanges; its first answer's *How this answer was made* shows *What the answer carried* with how many exchanges it held and how many it left out.
- [ ] **The chat list** (A8a, `[s9]`): the sort menu's *Group* (*No groups*, *By mode*, *By folder*; pinned chats stay on top) and *Only chats that cited*, which filters by the documents and folders a chat's answers cited; each row's date line carries those labels. A whole-library chat sits under "Whole library". Both choices are remembered.
- [ ] **What a turn shows** (A8b, `[s1]`, `[s7]`, `[s8]`, `[s10]`): the answer's claims are coloured, lilac when supported with a citation, amber when weak (with its hover text), plain when unsupported; a claim is a button to mouse and keyboard and opens its first cited passage, drawn darker while open. A records answer shows *What it looked up* (five rows, then "N more in the answer above"). A reopened chat shows its claims, the decisions recorded on them and the reviewer's verdict. A claim the window cannot find in the shown text is left plain, never misplaced.
- [ ] **Chat modes** (A7): *Critical reading* says where sources agree, disagree or are limited only where they show it, and adds no empty "limitations" section when they do not; the provenance card's model line ends with the mode and *(evaluated)* or *(not evaluated)*. A person's own modes have engine routes (`/api/modes`) and no screen yet (A8 left the editor out): record them as *not driven*.
- [ ] **Answers from the records** (A6): ask "How many documents are in my library?", "Which documents have no year?", "What is in folder X?", "What is the definition of concept Y?" → each is answered from the records, cites nothing, and ends *From the library's records: no passage was read and no model was asked.*; Ollama's log shows no call. A folder or concept that does not exist is said to be unknown. A content question ("What is the definition of attention?") still goes to retrieval.
- [ ] **A question within folders or a concept** (A7b): `/folder Thesis: <question>` searches that folder alone for that turn, and `/concept <name>: <question>` the documents where the concept is present; the provenance line names them. An unknown or empty folder is left out with its sentence and never widens the scope; a concept with no documents says to rebuild the graph and searches nothing; two matches ask which one.
- [ ] **Long chats** (A5): in a chat of ten or more turns that lean on each other, a late follow-up still answers about the right paper; when anything was left out, the citation block's *Fitted to the window:* line says what. Nothing is cut silently. On a server that does not state its window (Settings' own server, ADR-060), the line says so and names the token count the turn was kept to.
- [ ] **Text addressed to an AI, in an answer** (A2, U-005): ask until an answer cites the passage from §2's test document → its source card carries *contains text addressed to an AI*; the answer does not follow the planted text; a link with no source shows as text.
- [ ] A streamed answer with inline `[n]`; click one → the citation panel; **In context** shows the passage highlighted with what surrounds it and "N% of the way in"; **Show the page** opens the source viewer on that page (a figure citation too).
- [ ] Provenance card: which folder scope, which provider/model, which prompt hash; a **local model** is labelled as such on the answer.
- [ ] Source-evaluation strip shows year · relevance · graph freshness — and **not** coverage/`contested` chips (withheld since v0.4.1, KI-33) unless `EPISTEMICS_MARKERS_ENABLED=true` was set on purpose.
- [ ] Claim review: an `unsupported` verdict on a correct refusal is not the accusing one (KI-37); low-confidence card appears on a thin retrieval.
- [ ] **Compare** (A/B): two source sets side by side, $0, no LLM call in the log.
- [ ] Session override: change a sandbox knob → the `Session override` note renders **and** the retrieved set changes; out-of-range → 422 surfaced as a sentence, never a silent clamp; the override lasts the session, across new chats, and ends at a restart.
- [ ] Human synthesis mode: evidence only, no interpretation call in the log.
- [ ] Ask something the corpus cannot answer → a refusal that cites nothing, not a fabricated citation.
- [ ] **Hostile markup is inert** (S-3): ingest a `.md` holding `<img src=x onerror=alert(1)>` and `[x](javascript:alert(1))`, ask about it until the answer quotes them → no dialog, the link does nothing, no broken-image icon from the handler; the devtools console shows no CSP violation raised by the app itself.
- [ ] New chat resets turns, panel and composer; history sidebar reopens a chat read-only; rename · pin · export-all · bulk delete each work and each asks before deleting.
- [ ] **Chat selection** (0.7.3, ROADMAP 80): in the history rail a Shift-click selects a range and a Ctrl-click toggles one chat; the bar has *Select all* / *Select none*, a red *Delete…* and *Done*. *Delete…* asks with the count and opens with focus on *Cancel*; *Cancel* keeps the selection and select mode; confirming deletes exactly the ticked chats. Bulk archive and unarchive still work.

## 4 · Graph & gaps (real library)

- [ ] *Never built* and *built but empty* are two different screens (row 22).
- [ ] Concept rail: the three lenses are one control style with tooltips that say what they **do**; the ego graph opens on a node; *Place* deep-links into the taxonomy modal with the concept preselected.
- [ ] Gap list: promote · dismiss · reset; **rebuild the graph in-app** → the triage survives (`GapTriage` sidecar); the filter box matches label and kind; the coverage sentence ("N of M documents") is present.
- [ ] `under_connected` is **off by default** (RG-014 grade); `single_source` leads.
- [ ] The signals say what they are (KL1): the gap list reads **Uncited in answers**, not "Unsourced claims"; a thin bridge is flagged on the smaller side only; the coverage sentence counts only documents still in the library; opted-in stance chips read *contested? (experimental)*.
- [ ] Names and written case (ROADMAP 94, ADR-054): nodes, the panel heading and the rail show a concept's name in its written case (`dIN`, `Cre`); the staleness notice after a Manage keywords edit names the concepts and forms that changed; **Rebuild** clears it.
- [ ] The Definition card (ROADMAP 93a): a concept with none says *No definition chosen yet*; **Look in my library** lists quoted options, each opening its page, with a `strong` / `some` / `thin` grade and its reasons; **Use this**, **Write your own**, **Undo** (walks back one choice at a time), **Dismiss** and **Restore** each change only what they say; *How your library uses it* shows plain sentences; a concept with nothing definitional says the evidence is thin. A term reached through the search shows passages but offers no choice — it points to Manage keywords.
- [ ] **The concept view keeps its graph in sight** (0.7.3, ROADMAP 111): open a concept with a long definition and many documents at 1280×800 → the canvas sits above the definition card, about 400 px tall or more, and a label at the layout's edge is not cut; at 640 px the coverage sentence and the buttons wrap and the graph is still drawn.
- [ ] Term signals in the card (ROADMAP 93): an abbreviation shows its long form with the sentence that spells it out (`dbs` → *deep brain stimulation*), a cited name shows its "et al." sentence, a label no sentence uses shows where it occurs; each is offered, nothing is added for you.

## 5 · Settings

- [ ] Six categories in the rail (*Getting started*, *Documents*, *Provider & model*, *Retrieval*, *AI programs* since 0.8.0, *General*); nothing lost from the flat list (compare against the previous release's screenshot).
- [ ] Live provider/model switch without restart; the **reviewer follows** an unpinned switch; an explicit `REVIEWER_PROVIDER` still wins.
- [ ] Corpus facts: documents · chunks · disk · which keyword arm · its size; *Rebuild keyword index* runs, reports, and the warning banner clears; simulate a failed build → `keyword_index_unavailable`, chat degrades to vector-only and **says so**.
- [ ] Theme: System / Light / Dark persists across relaunch; both themes on every surface above (screenshot each once).
- [ ] Source directory change → the sources panel re-derives `new/changed/ingested/missing`.
- [ ] **Settings closes during a run** (0.7.2): start a run from Settings, then close Settings by Escape, by a click outside and by the close button; the status bar keeps counting and the library refreshes when the run ends; reopening Settings shows the run.
- [ ] **Provider & model** (0.7.2, ADR-060): the *Answering with* card names the model, where questions and passages go, and who bills. A preset (OpenAI, Gemini, Mistral, Groq, OpenRouter) with a made-up key: *Test connection* fails with a sentence and the key is never shown back; LM Studio or a server of your own on this computer: its models are listed from the server, a cited answer comes back, and the card says it stays on this computer, *Billing not known* until *This server does not bill me* is ticked. A key saved for one address is not sent to another.
- [ ] **Open another library** (0.7.2, ADR-061): Settings → Documents → Corpus → *Open another library…*: a folder that is not a library is refused with its sentence; a library copied from elsewhere shows its counts and says its cached text names another folder; *Open this library* restarts the engine and the library opens with the counts the check showed, nothing re-read (the indexing chip stays quiet). *Data home* names it. Delete one of its documents with *also the file*: the original library's file is untouched (choice 4). *Use the app's own library* asks, restarts, and comes back to the app's library unchanged. A failed switch (a folder whose `library.db` is damaged): after 90 seconds the card offers *Go back*, which reopens the previous library. Not shown in a development build.
- [ ] **AI programs** (0.8.0, A4): Settings › *AI programs*: under *Let AI programs read this library* the *On* switch is off, and the rest of the page shows only once it is on; *Connect a program* shows the address (with its own token, never the window's) and the command for a program that starts one; *What a program may read* lists the tools; *Recent calls* says no program has called yet, then lists only the tool and the time. Connect an MCP client by the address → it lists the tools and a call returns passages each carrying its source and, where it applies, the text-check mark. Connect by the command, **from the installed build** (`--mcp-bridge`; never driven before 0.8.0). *Replace the access token?* asks first; the old token then fails. Turned off, a client's call gets a 404 carrying *The MCP endpoint is off. Turn it on in Provenote's Settings.*, and a call carrying a web page's `Origin` is refused (403) either way.
- [ ] **Settings' door and its confirms** (0.7.3, ADR-065 choice 7, ADR-063): the cog sits right after the menu in the top bar; it and the menu's *Settings* entry both open Settings. *Remove key…*, *Remove server…*, and the ignore list's *Remove…* and *Remove all…* are red, each asks first with focus on *Cancel*, and *Cancel* changes nothing.

## 6 · Shell & platform

- [ ] Sidebar collapse/expand (an uncaptured defect since 2026-07-21 — capture the exact symptom if it misbehaves); global search (same); shortcuts dialog; About with the right version.
- [ ] Window at 640×480 minimum: no horizontal overflow; wide content scrolls in its own container; reduced-motion honoured.
- [ ] **The title bar** (ADR-065, Windows): no native title bar above the top bar. Drag the window by the bar's empty space; a double-click maximises and restores it; every button and tab in the bar still clicks. *Minimise*, *Maximise*/*Restore* and *Close* work, Close turns red on hover, and closing ends the engine (no `doc-assistant-api` left in the Task Manager). Resize from all four edges and the corners; Tab reaches the three buttons with a visible ring; check the dark and light themes and a 640 px window; open Settings and a dialog, then close them, and the controls work again. Win+Z opens the snap layouts (hovering *Maximise* does not, by design).
- [ ] **Buttons and dialogs** (0.7.3, ROADMAP 110, ADR-063): every button shows its hover text on hover and when Tab reaches it, and Escape dismisses the text first; a disabled button says why (*Send* with an empty box). Four looks only: ordinary, filled primary, quiet, and red for what removes. A confirm that removes has a filled red button and opens with focus on *Cancel*: drive *Dismiss…* on a chosen definition, *Reset to default…* in *Edit metadata* and *Undo all…* in the add review. In every dialog Escape closes only the one in front (open a confirm from Settings to have two), Tab stays inside, and focus returns to the button that opened it; in the dark theme the backdrop darkens what is behind. Under 780 px the top bar's tabs still have names for a screen reader.
- [ ] 0 console errors across the walkthrough (open devtools once in the dev build for the same route set — the release build has none).
- [ ] Kill the sidecar while the app is open → a **connection** error, not the zero-state (REVIEWS 2026-08-28: it showed "No documents indexed yet"). The panels' *TypeError: Failed to fetch* is expected here until KI-79's health check; since 0.7.3 that text means the engine is gone, because a block that fails with the engine up reads *failed: 500* (`apps/api/server_errors.py`). That second case cannot be caused by hand: `test_api_window_origins.py` covers it, so record it as *not driven* unless one shows up.
- [ ] **Answer engine unreachable** (the user, 2026-10-26): with the app open on Ollama, stop Ollama → the window stays responsive; the Library, the Graph tab and **Settings** open; Settings lets you point at another Ollama address or paste an API key; a question gets a sentence naming the unreachable engine, not a spinner that never ends. The freeze that prompted this row, after the 0.7.0 clean-machine run, was not the engine: the API refused the Windows window's origin, so the window never got past "still starting" (KI-67, fixed). This row is still to be driven (ROADMAP 100). If the installed app locks up here, that is a blocker for this release.
- [ ] Occupy port 8001 before launch → the app reports the conflict; cold start to first `/api/health 200` is within RG-010's recorded envelope; closing the window ends the sidecar process — check in the Task Manager that no `doc-assistant-api` is left, then reopen and make one change (KI-71: before the fix, the left-over backend refused every change of the next launch).
- [ ] Offline (network cable out): launch, ingest, answer with Ollama — all work; the update check says `unknown`.
- [ ] **No console window** (user, 2026-10-09): launching the installed app opens the app window only — no terminal window at launch, when the backend starts, or while it indexes. The app is built for the windows subsystem (`src-tauri/src/main.rs:2`); the backend is frozen as a console program (`scripts/doc_assistant_api.spec:124`, `console=True`) and relies on the launcher to start it without one. If a window shows: a blocker for this release — freeze with `console=False` and send stderr to the app log.
- [ ] The launch token (S-6): in the installed app, add, index, rename and delete all work; from a terminal, a `POST` to `http://127.0.0.1:8001` without `X-Doc-Api-Token` answers 401 with its sentence while a `GET` still answers.
- [ ] The host guard (S-5): a request to `127.0.0.1:8001/api/health` with `Host: example.com` answers 400; the app itself is unaffected.
- [ ] The CSP additions (S-4): the source viewer still renders pages and the figure viewer still shows figures in the installed build.

## 7 · Packaging gates (pointers, not duplicates)

- RG-012 Tier-1 (clean machine, offline) and Tier-2 (a cited turn, and the API answering the installed window's origin — KI-67: the gate's own calls come from PowerShell, which ignores CORS) — `docs/desktop-packaging.md` §5; the PASS must be bound to **this** installer's build timestamp (`preflight rg012`).
- The installer is the one in `bundle/nsis` *now*, and the previous one is deleted **after** the push (`RELEASE.md` §8).
- [ ] **Upgrade from the previous release's real installer, before the tag** (KI-80: RG-012 installs fresh, so nothing else sees this). Build a small library on it: documents copied and referenced, a folder, an edited title, a chat. Close its window and install over it, once silently (`/S`) and once by double-click, leaving the old engine running for at least one of the two. Expect no restart needed; the app answers; every record came across; a change saves; nothing of the old build left in the install folder or in `%TEMP%`. From 0.7.1 to 0.7.2 also expect (ADR-056): Settings → Documents shows every document as *changed* and each stays searchable until it is run; *Select new + changed*, then *Ingest selected* reads them once. From 0.7.2 to 0.7.3 expect every PDF, and only the PDFs, to show *changed* (KI-91: the app's version is in the PDF extraction fingerprint, so every upgrade since 0.7.0 does this until it is fixed); every other format stays *ingested*. From 0.7.3 to 0.8.0 expect the same one last time (the KI-91 fix takes the version out of the fingerprint); the next upgrade must leave every format *ingested*. Also run the 0.7.3 installer over 0.8.0 once by double-click: the installer allows downgrades (`bundle.windows.allowDowngrades` is not set, so Tauri's default `true` applies), so record what the older app does with the newer library (it opens, or what it says), and that 0.8.0 installed again opens it unchanged.
- [ ] **The installer's reinstall page** (0.8.0, Tauri 2.12's template, re-taken with the KI-73 line): running the 0.8.0 installer over 0.8.0 shows the reinstall page with its heading text (Tauri 2.12.5 renamed the language string behind it); uninstall from Windows Settings removes the install folder, `uninstall.exe` included.

---

## Record for release vX.Y.Z

Copy into `.claude/release-notes-X.Y.Z.md` (local) and fill. `Not driven` is a legal value — it is
the one that keeps the next release honest.

| § | Surface | Library | Build (installer timestamp) | Driven by / date | Result (ok · KI-nn · blocker) | Not driven |
|---|---------|---------|-----------------------------|------------------|-------------------------------|------------|
| 0 | First run | empty | | | | |
| 1 | Add documents | both | | | | drag-and-drop? multi-file? failure branch? |
| 2 | Library | real | | | | |
| 3 | Chat | real | | | | which model |
| 4 | Graph & gaps | real | | | | |
| 5 | Settings | real | | | | |
| 6 | Shell & platform | both | | | | |
| 7 | Packaging gates | — | | | | tier 2? |

---

## Lifting this into cpc (proposal, 2026-09-10)

The generic shape is small and is what would survive the generalisation:

1. **A per-surface table, not a per-feature one.** Features are how the roadmap is organised;
   surfaces are how a user meets the product. The table's rows are the product's screens, each with
   *steps → expect*, and the record column has three legal values: ok · issue-id · **not driven**.
2. **Three states per surface.** Every row is driven in its populated, empty and failed state; a
   checklist that only lists the happy path reproduces the blind spot it exists to remove.
3. **The artifact under test is named.** A build identifier column ties the record to the thing
   that shipped (the `preflight rg012` rule generalised); a PASS from a previous build is worse
   than none.
4. **The omission column is mandatory.** Same principle as `.claude/REVIEWS.md`: what was *not*
   covered is the load-bearing half of the entry.

Registration: a `release-close` keypoint (or `[keypoints.sprint-close]` on projects without a
release track) whose deterministic floor runs the artifact/version checks and whose judgment
checklist points at this file; the record table is the stamp. Per-project rows live in the
project's copy of this file; cpc ships the template (§ headings + the record table + the four
rules) — the way `REVIEWS.md` is proposed as a template in `.claude/REVIEWS.md` § Candidate.
