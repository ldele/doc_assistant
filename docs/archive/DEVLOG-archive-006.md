<!-- status: archived · updated: 2026-09-04 · class: append-only -->

# DEVLOG — archive 006 (2026-08-12 (1) → 2026-08-13)

Older entries, moved verbatim from `docs/DEVLOG.md` on 2026-09-04 so the working log stays
about recent work. Newest-first, same format, unedited. Rotated because the live log had
reached 4,011 lines against the 4,000-line cap in `tests/unit/test_doc_sizes.py`, which fails
before it can grow further. Cut on a date boundary so no day is split across two files.

---
## 2026-08-13 — ADR-045 (taxonomy display rule), and the auto-propose run that says the scope is the bug

**What changed.** New `docs/decisions/ADR-045-taxonomy-display-rule-and-document-identity.md` + its
index row; RG-015 in `.claude/RIGOR_TODO.md` gains a third evidence section. No code.

**ADR-045 fills a gap ADR-028 names in its own "Must revisit": *"the display rule is unspecified."***
A document shows its attached nodes **minus any that is an ancestor of another** (so a concept placed
under both `optics` and `neuroscience` shows both — neither subsumes the other); search and filter
match the **full ancestor closure**; attachment happens only at the most specific node and ancestors
are derived, never stored. Both rules degrade to identity on an unplaced vocabulary, which is not a
future edge case — it is the path 344 of 357 concepts take today.

**Why it matters beyond tidiness:** this is the actual fix for the partitioning failure measured
twice on 2026-08-12. `rag` is `df=1` and always will be; `machine learning` is not. Under closure a
97%-singleton vocabulary still filters cleanly, with **no change to extraction and no re-ingest**.
Two further decisions, both taken to keep something out of the vocabulary: a document identity key
(`rag_lewis_2020`) is a **computed field, not a `Concept` row** — it is an identifier, `df=1`
forever, and admitting it would manufacture 97 permanent singletons in the facet being repaired; and
bibliographic type/origin are **metadata columns**, because "is a journal article" is not "belongs to
a research field" and modelling it as `in_field` gives that edge two meanings.

**Then the measurement, because ADR-045 makes placement load-bearing.** Under closure a wrong
placement silently *widens a filter*, so `propose_taxonomy --apply --all-concepts --limit 25` ran on
`qwen3.5:9b` (49 calls, ~2 min, $0) before anything bulk. Judged against each term's own source
document: **~4 right · ~8 coarse · ~12 wrong**, and the wrong ones are not near-misses —
`acdc`→**Music** (the ACDC cardiac-MRI benchmark, from a Mamba-UNet segmentation paper),
`alpha`→**Analytical chemistry** (an EEG band), `actor`→**Performing arts** (actor–critic RL),
`accessory`→**built environment** (from a Cajal neuroanatomy paper).

**The finding is not "the model is bad".** On the **13 curated `graph_include` concepts** a *weaker*
model (llama3.1:8b, 2026-07-25) was **13/13 plausible**. On raw keyword rows a *stronger* model is
half wrong. **The variable is scope, not capability** — `--all-concepts` crosses the ADR-018
boundary on purpose and hands the classifier 344 rows nobody curated, ~40% of which are not concepts
at all (venue artifacts `aclweb`/`aclanthology`, orphans with no document links, and bare fragments
like `alpha`/`actor` that only mean something inside a phrase). So `graph_include` is a
**precondition for auto-propose working**, not bookkeeping.

**Confidence is still not a signal, third independent confirmation:** 0.80–0.95 across the sample,
`acdc`→Music at 0.80 and a correct placement at 0.80.

**A correction worth more than the run.** The taxonomy has **never had a human-accepted concept
placement**: all 37 concept→field links were `origin='proposed'`; the 213 `curated` edges are the
ANZSRC trunk itself (domain→domain). Every earlier statement of the form "13 concepts are placed"
should have said *proposed*. State which, always — they have never meant the same thing here.

**Disposition.** The 24 rows were deleted via `taxonomy.remove_hierarchy_edge` (back to 213 curated
+ 13 older proposals); backup `data/library.db.bak-20260813-pretaxonomy`. The remaining 333 were not
run. ⚠ **`--limit N` takes the alphabetically-first N**, so this was an `a*` sample — enough to
answer "usable in bulk?" (no), not enough to quote as a precision figure.

**Rejected.** Running the remaining 333 (would have written ~330 rows of this quality into a layer
ADR-045 just made load-bearing). Keeping the 24 as a labelled sample (the taxonomy view would show
`acdc → Music`, and a wrong placement that is *visible* is how a feature teaches something false —
the exact failure the 2026-08-12 relabels were about).

**What it opens.** The sequence is now clear and is upstream of the classifier: curate the
vocabulary, clean the D4/D5 residue out of it, then place a small set — by hand or with a stronger
model — and only then trust coverage. RG-015's original debt (per-kind detector precision) is
untouched and still needs the detectors to exist.

---
## 2026-08-12 (4) — keyword quality D4 + D5: the bibliography is where surnames come from, and the tokeniser was renaming genes

**What changed.** `knowledge/keywords.py` gains `strip_reference_section`, `is_citation_artifact`,
a rewritten `_TOKEN_RE`, and a head-check in `candidate_terms`; 15 more unit tests (44 in that file
now). Re-applied to the live 97-document library.

**D5 — the tokeniser was silently renaming things.** `_TOKEN_RE` split on `.` and `/`, so
`16p11.2` became `16p11` (a different locus), `C57BL/6` became `c57bl`, and `gpt-3.5`/`gpt-4.5`
became `gpt-3`/`gpt-4`. The fix keeps a separator **only when a digit follows it** — which is the
whole of what distinguishes a designator from prose. `e.g`, `i.e` and `arxiv.org` still split
exactly as before, because a *letter* follows their separator. The corpus now carries `16p11.2`,
`dlight1.1`, `gpt-3.5` and `gpt-4.5` whole, where before it carried truncations of them.

**D4 — surnames come from the bibliography, so remove the bibliography, not the surnames.**
⚠ The obvious fix here is a name filter and it is the wrong one: `cajal`, `cre`, `dbs`, `16p11`
and `c57bl` are real vocabulary in this corpus and have been mistaken for noise twice. So D4 is
structural — cut from a whole-line `References` heading to the end of the document — plus three
pure *shape* rules (`2014a` year-suffixes, `e04250` article ids, `10.xxxx` DOI prefixes). A
surname that survives that appears in the document's own prose, which is exactly when it is a real
term for that document. `shadmehr` and `wolpert` are still keywords of Shadmehr's own review, and
should be.

**The measurement that saved the fix from looking finished.** The first regex matched only a plain
`References` line and fired on **25 of 97** documents. Sampling the actual cache showed why:
PyMuPDF4LLM's dominant rendering is `## **References**` (32 of the sampled headings), with
`_REFERENCES_` behind it. Allowing markdown emphasis took it to **79 of 97** — a fix that would
otherwise have shipped working on a quarter of the corpus while reporting success.

**Effect on the two probes** (dry run, live corpus): year-suffix citations **2 → 0**;
`rag_lewis_2020.pdf` lost `10 18653 v1` and `aclweb org anthology` (DOI + URL fragments) and gained
`dpr` and `generator`; `transformer_vaswani_2017.pdf` lost `eos` / `my opinion` / `pad br` and
gained `attention function` and `convs2s`.

**A nuance worth recording, because it looks like a regression and is not.** Keywords on ≥2
documents went **48 → 42** and on ≥5 went **2 → 0**. Removing bibliographies removes terms that
were "shared" across documents *because they were citation artifacts* — spurious cross-document
links, not corpus structure. Partitioning is not what D4/D5 were for, and it remains the open
question from DEVLOG 2026-08-12 (3): per-document TF-IDF cannot produce a facet.

**Rejected.** A surname/author stopword list (the trap above). Rejecting any n-gram containing a
bare decimal — it would kill `gpt 3.5`, a real term, to remove `0.0 true`. Cutting at the *last*
References heading rather than the first past the halfway mark (an appendix after the bibliography
is smaller than the risk of matching a heading in the body).

**What it opens.** Residual noise this did not touch and did not create: `br` is an HTML
line-break artifact leaking out of the markdown (`1020 br`, `br 32`, `0.01 br`), and bare decimals
still ride inside bigrams (`0.0 true`). Both are small, both are their own fix, neither is D4/D5.

---
## 2026-08-12 (3) — the four release-readiness relabels: a rank instead of a score, keywords that are about the paper, an experimental label, and a hidden tab

**What changed.** The user's decision on `docs/REVIEW_2026-08-12_release-readiness.md` §2b was
**"fix all four, hide the graph tab"**. Four independent changes, one theme: *a built feature that
teaches a first-time user something false is worse than one that is missing.*

**R1 · Connections shows a rank, never a score.** New `lib/library/connections.ts` (+5 tests);
`DocConnections.svelte` prints `1st / 2nd / 3rd` and a one-line caveat instead of `score.toFixed(2)`.
Measured on the live corpus while verifying: one document's top four neighbours score **0.982,
0.965, 0.962, 0.956** — the old UI rendered that as "0.98 / 0.97 / 0.96 / 0.96", which reads as a
precise claim about how alike two papers are. It is not one: `doc_vectors` mean-pools every chunk,
so same-field papers collapse (750 edges, **median 0.918**, against a 0.5 threshold that therefore
excludes almost nothing). The ordering survives that; the distance does not. The caveat is a
sentence, not a tooltip, because the user who never hovers is exactly the one who would over-read it.

**R2 · Keywords — D1 (page furniture) + D2 (shingle suppression), and the honest half of the result.**
New pure functions `split_pages` / `strip_page_furniture` / `suppress_nested` in
`knowledge/keywords.py` (+14 tests), wired into the shipped `per_doc` path, then **applied to the
live 97-document library** (`extract_keywords --apply --force`, $0, deterministic; `library.db`
backed up first).

| | before | after |
|---|---|---|
| documents with ≥1 keyword | 82 / 97 | **96 / 97** |
| nested/shingle slots | **334 / 1230 (27%)** | **0 / 1440 (0%)** |
| distinct keywords | 1192 | 1375 |
| on ≥2 documents | 30 | 48 |
| on ≥5 documents | 1 | 2 |
| on ≤1 document | 1162 (**97%**) | 1327 (**97%**) |

The eyeball test is decisive. `nihms-66884.pdf` went from **11 of 15 slots** being shingles of the
PMC running header (`author manuscript` · `exp brain res` · `2008 september 26` …) to
`motor commands` · `optimal control` · `proprioceptive` · `state estimation` · `saccades` — the PMC
stamp is *gone*. `transformer_vaswani_2017.pdf` went from **9 of 15** slots on one figure's
`<eos> <pad>` artifact to `multi-head attention` · `self-attention` · `scaled dot-product` ·
`sequence transduction`. `rag_lewis_2020.pdf` went from **zero** keywords to `rag` · `rag-sequence` ·
`rag-token` · `retriever` · `non-parametric memory`.

**But the number the plan said to report did not move, and that is the finding.** Singletons are
**97% before and after**. D1/D2 fixed *what the keywords say*; they did not make the layer
**partition** the corpus — and they cannot, because per-document TF-IDF selects `df≈1` distinctive
terms *by construction* (this module's own docstring says so). **So keywords are a good per-document
descriptor and are still not a facet.** Partitioning would need the `corpus_band` or `contrastive`
mode, which is a different decision, not a bug fix. The Library overlay already tells the truth here
(PR-2.7 F4 collapses the 1-document tail behind *"on 1 document — search still finds them"*), so no
UI change was needed — but the open question survives this session intact.
*(The coverage jump 82 → 96 is **D3**, not D1/D2: it is simply what re-running a stale enrichment
does. One document still yields nothing.)*

**R3 · The epistemics chips say `experimental`, in both places they are met.** The chip carries an
inline `experimental` tag (inside the pill, so a line wrap cannot separate the qualifier from the
claim) and a tooltip that says it is not a measurement and can be wrong; the Settings section gains
a `(experimental)` heading and a plain-words **Known limitation** paragraph. The default was already
`false` (flipped 2026-08-03, KI-33) — what was missing was *saying so*: a silent default tells a user
who turns it on nothing, and a user who leaves it off nothing either. Rebuild is ADR-041.

**R4 · The Graph tab is hidden.** New `lib/core/features.ts` with a single documented
`GRAPH_TAB_ENABLED = false`; the tab is `{#if}`-gated in `Topbar.svelte` and `selectMode` coerces
`graph → chat` defensively, because that function is also the nav-history restore path and landing
in a mode with no exit is the one failure a hidden tab can still cause. **Nothing was deleted** —
`/api/concepts/*` stays mounted, the gap list keeps its triage writes, and `GraphIndex` /
`ConceptGraph` / `GapList` are untouched and still tested. Checked before hiding: the Taxonomy view
is a global overlay opened from the **Library** rail, so it is not orphaned by this.

**Rejected.** Deleting the graph code (throws away working, tested work to solve a *placement*
problem). Keeping the similarity number in a tooltip (if the number is not meaningful, hiding it one
hover away is not honesty). Growing `VENUE_STOPWORDS` for D1 — the signal is position and
repetition, not vocabulary; every publisher has a different stamp and the words in them
(`brain`, `september`) are not junk anywhere else. Applying D2 to `contrastive` (it already
discounts nesting via C-value) or `corpus_band` (its exposure is via furniture, fixed at source).

**What it opens.** The facet question above. Whether corpus growth should re-trigger keyword
enrichment automatically (**D3**, which is KI-44's question in a different costume — answer them
together). And the one document that still extracts zero keywords.

---
## 2026-08-12 (2) — the graph rail's three small fixes, and two checklist rows that were already done

**What changed.** `gaps.ts` gained `filterGapRows` (+5 tests); `GapList.svelte` gained a filter box
and turned its two checkboxes into pressed-state `.lens` buttons; `GraphIndex.svelte` did the same
to `Include under-connected` and lost the now-unused `.toggle` rule. Both `Gaps only` and
`Include under-connected` got tooltips that say what the control *does* rather than restating its
own label. `docs/ui-checklist.md` updated.

**Why.** Three rows the checklist marked "no dependencies", and they were: `Gaps only` was a button
while `Include under-connected` next to it was a checkbox — two different kinds of thing doing one
job — and the Gaps tab had no filter while the concept rail beside it did. The filter matches the
**gap kind** as well as the concept label, because "single" is the list's own vocabulary for a class
of problem; matching labels only would return nothing and read as *"no such gaps"*.

**The other half is a finding, not a change.** The two "review findings" in §3 —
`reviewer_kind="llm_haiku"` hardcoded, and `set_llm_selection` accepting an empty model — are
**both already fixed in code**, with a test pinning the first
(`test_chat_controller.py:1052`). The rows were stale. §3 now says so, because the cost here was
only ten minutes of reading, and the cost of *believing* it would have been a redundant "fix" to
working code.

**Verified live** (dev server, $0): 15 gap rows → 3 on `single`, all Single-source; a no-match query
shows the honest empty line; both rails show two matching lens buttons and zero leftover checkboxes.

---
## 2026-08-12 (1) — the app can now tell you a new version exists, and that is deliberately all it can do (ADR-044)

**What changed.** New `docs/decisions/ADR-044-update-notification-not-delivery.md` + its index row.
New `src/doc_assistant/update_check.py`, `apps/api/{models,routers}/updates.py` (three routes),
`apps/desktop/src/lib/core/{types,api}/updates.ts`, `apps/desktop/src/lib/settings/updates.ts`
(pure display logic) and an **Updates** section in `Settings.svelte`. `app_settings.py` gained
four accessors. **`src/doc_assistant/__init__.py` gained `__version__`** — it was empty, and the
app had no runtime knowledge of its own version at all. Tests: 26 unit + 12 integration + 9
frontend, all passing; `release_preflight` now checks **six** version strings, not five.

**Why.** The app ships as an NSIS installer with no store and no package manager behind it, so
every install is frozen at the version it shipped with — including one carrying a bug we have
already fixed. The user's framing was explicit: do it like calibre — signal, link, and let the
user install — and **an integrated updater is too ambitious for now** because several features
are not stable enough to push at people automatically. That rejection is the load-bearing half:
an in-app updater must replace a running binary, verify a signature and roll back a bad write,
and none of that is earned while the release process has not yet produced a verified artifact for
the tag it cut.

**The design decision that is not implementation detail: three states, never two.** A failed check
reports `unknown`, never `current`. Saying "you are up to date" because the network was down is
the one failure mode that would make this feature worse than not having it, so every network,
parse and decode error becomes `unknown` with a plain-words reason the UI shows verbatim. Verified
against the real endpoint: the repo has **no published releases**, and the app says *"no published
release to compare against"* rather than going quiet or claiming currency.

**Two defects found while building it, both worth recording.**
1. **My first cut cached only the check timestamp, not the observed version** — reasoning that a
   stored verdict would go stale. It does, but the cure was worse: a GET inside the 24 h window
   then *forgot* the update the last check found, so the banner vanished on the next page load.
   Fixed by storing the observed **version** and recomputing the verdict against the running one
   on every read — which self-corrects in both directions (after the user updates, the same cached
   observation reads `current` with no further request). Two regression tests pin it.
2. **mypy caught a real portability bug**: `datetime.UTC` is 3.11+, and this package declares
   `requires-python = ">=3.10"`. Now `timezone.utc`. CI runs the same interpreter as the dev box,
   so nothing else would have caught it.

**Rejected.** In-app download/install (user's call, and it needs a code-signing decision first).
Default-on checking (this app makes no outbound calls the user did not ask for; the toggle is off
by default, and the **manual** "Check now" runs regardless, because gating an explicit press would
leave a user who declined background traffic no way to answer the question at all). A configurable
repository URL — pointing an update banner at an arbitrary host is a way to get someone to install
something they did not choose. Notifying on `unknown` outside Settings — an offline machine would
nag forever about a check it cannot run.

**What it opens.** This project now has a **network code path in `src/`, permanently**, and it
inherits the frozen-build OS-trust obligation (KI-10) for a second caller. It also creates a
release-process coupling that is now real and currently unmet: **the banner is only truthful if
GitHub *release objects* are cut for tags**, and none exist — `v0.5.0` is not even pushed. Cutting
a release is now a step in `docs/RELEASE.md`, not a courtesy. Not built, and deliberately: any
surface outside Settings (a topbar dot, a startup banner) — `shouldNotify()` exists and is tested,
but nothing calls it yet, so today the only place an update appears is the Settings panel.
