<!-- status: active · updated: 2026-07-30 · class: append-only -->

# ADR-037 — Answer the scale question with corpus facts, not with performance knobs

- **Status:** accepted (built 2026-07-30)
- **Date:** 2026-07-30
- **Deciders:** user (three decisions, `grill-me` session) + Claude Code
- **Closes:** ROADMAP **PF2**, filed 2026-07-30 as "which cost knobs become user-facing, and how"
- **Supersedes:** the three-tier preset proposal in `docs/performance.md` §5, written before ADR-036

## Context

PF2 was filed on a real user question: *"we may want to give the user the option to change those
settings if they prefer working differently … a few hundred documents is not a lot."* The knob
inventory (`docs/performance.md` §4) proposed three tiers, with named presets for the
**output-neutral** knobs — the ones that trade time, space or money but cannot change what an answer
says.

**ADR-036 dissolved most of that premise the same day.** Checking the proposed Tier-A list against
the code after the sparse arm moved on disk:

| Proposed knob | State now |
|---|---|
| BM25 snapshot (`DOC_BM25_CACHE`) | **legacy-arm only** — the on-disk index replaced the thing it cached |
| Scoped-ensemble cache size | the ~20 µs/chunk rebuild it traded against **no longer exists** (scoping is a `WHERE` clause) |
| Ingest workers (`MARKER_MAX_WORKERS`), VLM budget, figure DPI | used **only by `scripts/`** — CLI runners that already take flags; the in-app ingest never reads them |
| `DOC_SPARSE_INDEX` (new) | **not output-neutral** — the two arms rank differently, so it fails the test that made Tier A safe |
| Reranker eager/lazy | the one genuine in-app, output-neutral trade left (4.4 s launch vs ~5 s first answer) |

So the exposable set had collapsed to a single minor toggle, while **the need behind the request was
already met in engineering**: backend RAM is flat, and the question "will this hold my big library?"
now has a measured answer rather than a knob.

## Decision

**Ship the answer, not the controls.** The existing Settings → **Corpus** section reports what this
corpus costs on this machine: documents, chunks, disk (total and per document), which keyword-index
implementation is serving, its size and when it was built, plus one sentence about memory. No
performance presets, no new switches.

Three sub-decisions, each the user's call in the grill:

1. **Corpus facts in Settings**, not a knob panel and not documentation-only. It answers the
   question that was actually being asked, and it needs no restart semantics — which the knobs would
   have, since every one of them is decided at pipeline construction and there is no live rebuild
   path.
2. **One bounded action: rebuild the keyword index.** Not the full re-index. A rebuild is derived
   data that the next launch would regenerate anyway (the fingerprint moves with the corpus), so it
   is non-destructive, needs no confirmation, and is 2.8 s here / minutes at 10k documents. A *full*
   re-index is hours at that size with no progress or resumability yet — shipping a button for it
   would be the worst possible reading of inform-don't-block.
3. **State the shape of memory, never a live number.** "Memory does not grow with your library."
   A live RSS figure needs a new dependency or three per-OS implementations, fluctuates, and is
   dominated by model weights — a user would read 2 GB as the cost of *their documents* when it is
   the embedder.

**The memory sentence is frontend copy, and `keyword_index.mode` is what crosses the wire.** The two
arms have opposite answers: on-disk keeps memory flat, the legacy in-RAM arm grows at ~5.9 KB/chunk
(KI-32's ceiling). A pre-rendered sentence from the backend would be one refactor away from
reassuring a user whose process is doing the opposite. `corpus_stats` therefore reports the mode of
the **live pipeline** — not whether an index file happens to exist, which a stale file beside the
legacy arm would answer wrongly.

**The knobs stay where they are.** `DOC_SPARSE_INDEX` and friends remain environment variables
documented in `docs/performance.md` §4: a developer rollback surface, not a product one. ADR-010's
per-turn sandbox keeps the *quality* knobs, and the eval-locked settings stay locked.

## Consequences

**Good.** The scale question is answerable inside the app, in the user's own numbers, without
inviting anyone to tune retrieval by hand. The panel also surfaces two states that were previously
invisible: which sparse arm is live (a `DOC_SPARSE_INDEX=0` install now says so) and when the index
was last built. `POST /api/settings/reindex-keywords` returns the refreshed settings, so the panel
never needs a second request.

**Costs, stated plainly.**
- **A directory walk per panel open.** `os.scandir` over the data home's artifacts — a few thousand
  stats, no reads, errors swallowed per entry. Fine at any size seen so far; if it ever isn't, it
  becomes a cached figure with a timestamp, not a spinner.
- **A synchronous route.** The rebuild blocks its request for seconds-to-minutes. Bounded work with
  a definite end, so a job runner with polling would be machinery for nothing — but if the index
  ever stops being cheap to rebuild, this becomes a `202` + status route like the graph rebuild.
- **One more wire contract to keep in sync** (`corpus_stats.py` → `models`-shaped payload →
  `types/settings.ts`), which is the standing cost of every panel in this app.

**What this does not decide.** Whether the app should ever expose retrieval-quality knobs to users
(ADR-010's answer — per-turn, non-persistent, recorded in provenance — still stands), and whether
the legacy in-RAM arm survives (ADR-036 keeps it until the A/B can be repeated on a discriminating
case set). Both are recorded, neither is reopened here.

**Reopens if:** the in-app ingest grows progress and resumability (then a full re-index button is
worth revisiting), or a user-facing knob appears that is genuinely output-neutral and cannot be set
at install time.
