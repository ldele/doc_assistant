<!-- status: archived · updated: 2026-08-26 · class: append-only -->

# DEVLOG — archive 003 (2026-08-02 (1) → 2026-08-03 (3))

Older entries, moved verbatim from `docs/DEVLOG.md` on 2026-08-26 so the working log stays about
recent work. Newest-first, same format, unedited. Rotated because the live log had reached
4043 lines against the 4,000-line cap in `tests/unit/test_doc_sizes.py`, which fails before
it can grow further.

---
## 2026-08-03 (3) — v0.4.1: the first installer since June, KI-33 contained before it ships — and **RG-012 Tier-2 still has no evidence**

**What changed.** KI-33 containment (`config.py` default + `SourceEvaluation.svelte`), version 0.4.1
across **seven** strings, a CHANGELOG entry, `@tauri-apps/cli` added as a devDependency, and two
release artifacts. User priority for this stretch: *"focus on the binary release … to finally have a
beta-release"*, with the KI-33 surfacing fix landed first.

**The containment, and why it is a default rather than a deletion.** `EPISTEMICS_MARKERS_ENABLED`
**true → false** — the same lever R7 used for KI-7 containment, now for a defect one layer down —
and the strip's coverage + `superseded` chips commented out with the reason at the line, markup and
CSS together so restoring is one contiguous uncomment. **All three coverage values go, not just
`contested`:** `ns` and `nc` both derive from `stance_by_doc`, so `corroborated` and `unique` inherit
the same defect. The strip keeps what is sound — year, relevance score, graph freshness.

**Non-vacuous by construction.** The flip failed **exactly one** test —
`test_markers_enabled_by_default`, the one encoding the old default. Renamed to
`test_markers_disabled_by_default` and paired with a new
`test_markers_still_available_when_explicitly_enabled`, so the suite now pins **both** the new
default and that the opt-in still works. A containment nobody can prove reversible is a deletion.

**The release.** Sidecar re-frozen on a CPU sync (KI-3): **1545.5 MB**, replacing a **2026-06-24**
build — pre-rename, pre-icon, pre-ADR-034. Smoke-tested standalone *before* bundling, which is the
step that catches what tests cannot: `/api/health` in ~30 s, **chunk_count 33,105**, no frozen-import
failures. Then `Provenote_0.4.1_x64-setup.exe` (**1555.4 MB**) and `Provenote_0.4.1_x64_en-US.msi`
(1546.7 MB) — the first installers since June and the first carrying the Provenote identity.

**A root cause worth naming: the build recipe had rotted because it was never declared.**
`npx tauri build` failed outright — `@tauri-apps/cli` was not a devDependency, not global, not
anywhere. The June installer was built against undocumented machine state, so "how to build the
installer" was unreproducible the moment that state changed. Now pinned at `^2.11.4` in
`devDependencies`. **This is the same class of failure as the `uv.lock` miss** — a build input that
nothing declared and nothing checked.

**⚠ RG-012 Tier-2 IS STILL OPEN, and this session produced no evidence about it.** Two Windows
Sandbox launches, ~30 minutes: the `.wsb` parses, all four mapped folders resolve, the output folder
is host-writable, the VM boots and burns CPU — and `LogonCommand` writes **nothing**, even hardened
to sleep 25 s and then immediately create a file before doing anything else. It is not executing in
this Sandbox configuration. Root cause unknown. **Nothing here licenses any claim about the
installer on a clean box**, and the CHANGELOG's *Known limits* says so in the release itself. The
harness is left at `C:\rg012-host\` (ASCII path on purpose — `.wsb` parsing is unreliable with the
accented profile path) with a self-contained `rg012-run.ps1` that installs silently, seeds three
PDFs, ingests, asks one question and writes a PASS/FAIL verdict.

**Machine state touched and restored.** Ollama was rebound to `0.0.0.0` (user-approved) so a sandbox
could reach it, and is back to **127.0.0.1** with the env var cleared. The venv was CPU for the
freeze and is back to **`cu130`, CUDA available**.

**Rejected.** Shipping the binary as 0.4.0 — the user's choice, made when the delta was docs-only;
landing KI-33 first made the binary behave differently from the `v0.4.0` tag, so it was re-raised and
became **0.4.1**. Fixing the `postcss` advisory (build-time only, via `vite@6`, processes only
first-party CSS, and a clean fix exists) — deferred rather than applied because it would have changed
the toolchain underneath the gate run testing that exact build.

**What it opens.** RG-012 Tier-2 needs a path that does not depend on `LogonCommand`: a hand-run in
the sandbox, computer-use driving the VM, or a real second machine. Until one of them produces a
cited turn, **this is a beta by its own CHANGELOG**. Then: `npm audit fix`, and the KL1–KL4 plan.

---
## 2026-08-03 (2) — Full review of the knowledge layer against the stated goal: **the acquisition half has no implementation**, and the one suggestion engine runs on the detector graded noise

**What changed.** No source code. New `docs/PLAN_2026-08-03_knowledge-layer-to-goal.md` (local-only,
ADR-029) — the in-depth review; new **§6b State of play** in the tracked `docs/knowledge-layer.md`;
new ROADMAP rows **KL1–KL4** so the plan's items are visible in git rather than only in a gitignored
file.

**Method.** Read every governing artifact rather than working from memory: ROADMAP rows S1–S2 ·
G1–G8 · E0–E5 · TX1–TX3 · MM1–MM3 · PF1–PF4; ADR-004/008/015/017/018/027/028/030–033/036–041; the
concept-graph, gap-detection and taxonomy specs; RG-014/015/019; KI-18/19/33.

**The goal decomposed into five testable capabilities** — C1 unsubstantiated claims · C2 per-concept
classification · C3 gap exposure · C4 **acquisition direction** · C5 navigation — then mapped onto
every built component.

**Finding 1 — C4 is the goal's operative capability and has no sound implementation anywhere.** The
goal's verb is *"should find more resources and documents."* **Every built detector looks inward** at
what the corpus already holds. ADR-004 named the outward half precisely and deferred it — Tier-2b
needs *"a representation of 'outside the known space'"* — and ADR-032 has been a stub since. No
amount of repairing the inward detectors closes this; it is the largest distance between the product
and the goal.

**Finding 2 — the one suggestion engine runs on the one detector graded noise.** `gap_suggest` (G5)
is the closest thing to C4 in the tree, and it fires one LLM call per **`under_connected`** concept —
the kind RG-014 graded ❌ *"mostly noise… measures graph degree, dominated by vocabulary sparsity"*.
Meanwhile **`single_source`**, the kind RG-014 graded *"TRUE POSITIVE — the product thesis"*, gets
**no suggestion pass at all**. Re-pointing it is hours of work and the cheapest real progress toward
the stated goal.

**Finding 3 — C1 already works, on the wrong layer.** The answer path classifies claims by
retrieval-derived support and says outright why (*markers never come from model confidence*). The
concept layer ignores it and asks an LLM about labels. That is ADR-041 option 6, now KL1.

**Finding 4 — C3 and C5 are genuinely strong**, which is worth stating plainly after two sessions of
finding defects: `single_source`, the graph, ego navigation, the gap list with durable triage, the
Connections panel and the taxonomy view are all built and sound. The per-document map (MM1–MM3,
absorbing PR-G2c) is the one navigation piece missing, still gated on the ADR-030 stub.

**The plan, phased so nothing is built on top of something that lies.** **A** make what exists tell
the truth (the surfacing deadline · option 6 · `unsourced_claim` contamination · encode RG-014's
grades in the gap list) → **B** the acquisition half (re-point `gap_suggest` · grill ADR-032 · the
taxonomy as reference class) → **C** the per-file map (grill ADR-030 → MM1 → MM2 → MM3) → **D**
Node-B rebuilt on evidence, ground-truth study as the gate → **E** measure the two unmeasured shipped
layers (RG-015 taxonomy placement, RG-018 wiki flip). **B before C** deliberately: B1 is hours and
moves the operative capability, C is weeks and improves navigation that already works.

**Rejected.** Putting the whole review in the tracked docs (the dated-PLAN convention is local-only,
ADR-029) — mirrored as §6b + KL1–KL4 instead, so the conclusions survive outside a gitignored file.
Attaching effort estimates (the ordering is by dependency and goal-value; guessing hours would have
dressed judgement as measurement).

**What it opens.** Five things the review did **not** verify, listed in the plan's §6 — most
importantly that `gap_suggest`'s restriction to `under_connected` is taken from CONTEXT.md and
ADR-004 and **not re-read in `gap_suggest.py`** (confirm before doing B1), and that RG-014's verdict
dates from 76 docs / 26 concepts against today's 97 / 13 — direction stable, numbers not.

---
## 2026-08-03 (1) — ADR-041 (rebuild-or-retire Node B) + the knowledge layer finally has a map with a trust table

**What changed.** No source code. New **ADR-041**, new **`docs/knowledge-layer.md`**, a new
**"Read before you touch — never assume"** section in `.claude/CONTEXT.md`, plus pointers from
`AGENTS.md`, `architecture.md` and `src/doc_assistant/CLAUDE.md` so none of it depends on knowing
it exists.

**The CONTEXT.md table is the durable half.** Eight rows mapping *area of the app* → *what to read
first* → *what assuming instead has actually cost* (the epistemics threshold chase; reverting the
lazy reranker or the `_sparse is None` guard; page markers in the evidence block; curated structure
in `concept_edges`; a "local" run billing the API). Plus two standing rules: **a spec's surface
description is not its purpose** — with today's retire recommendation as the worked example, and the
corollary that *months of deliberate work on a feature is evidence of intent, so ask what it is for
before proposing to remove it* — and **re-measure per box; check a "known" fact before inheriting
it** (three inherited claims were false this week: the private eval set, retrieval determinism, and
`gh`/Docker being absent).

**Also answered, since it came up as "I don't know if it was done":** the *explore concepts within a
given file* feature is **not built**. It was **PR-G2c** (Library entry, doc → its concepts);
`feature-concept-graph.md:19` records E4 shipping a related-papers Connections panel instead, and it
has since been absorbed into **ROADMAP MM2** (`knowledge/doc_map.py` + `GET
/api/library/documents/{id}/map`), gated on **ADR-030** — still a proposed stub needing `grill-me`,
and already flagged in the baton as the one to do first because it blocks MM1.

> **Date correction.** The three entries below are headed `2026-08-02`; that is wrong — **all of that
> work was done on 2026-08-03**, in the same session as this entry, and the two baselines it
> committed carry `2026-08-02` in their filenames for the same reason. Left as-is rather than
> rewritten: the entries are append-only and already committed (`40888b1`), and a rename would
> break the links pointing at them. Corrected here so the record is not silently off by a day.

**Why the doc, and why now.** The user's read was that the docs are behind what we have on the
concept graph — *"it is not clear what we are doing and why."* Checking rather than agreeing: the
purpose **is** written down, and well. `docs/specs/feature-concept-graph.md` § *The job* (locked with
the user 2026-07-17) states it as three questions — **corroboration** (*"is this concept backed by
more than one source?"*), **coverage**, **navigation** — plus ADR-004's north star, *"the graph is
the substrate; the gaps are the payload."* The mechanism is in `architecture.md`; the decisions are
in nine ADRs. **Nothing connected them**, so nothing noticed when an output stopped matching the
purpose. That is not hypothetical — it is precisely how `contested` shipped saturated and how
RG-019's prescription went two weeks unchallenged. The missing artifact was never a description; it
was a **trust table**.

`docs/knowledge-layer.md` is that page: the job · the one-vocabulary rule · the two graph layers ·
the end-to-end flow · who consumes what · **a per-signal trust table** · how to run it · the ADR
reading order. It grades `single_source` ✅ (RG-014's "TRUE POSITIVE — the product thesis"),
`unsourced_claim` ⚠️ (~33% contaminated), `under_connected` ❌ (noise at small vocabularies), and
`contested`/`superseded_trend` ❌ **not a corpus measurement** (KI-33).

**The finding that shaped ADR-041, and it came from re-reading the spec rather than the code.**
**All three of B1's jobs are answered by counting documents** — corroboration is `len(doc_ids) >= 2`,
coverage is presence per field, navigation is `node → doc_ids → chunk_keys`. All Node-A,
deterministic, zero-LLM. **Stance answers a question B1 never asked** ("do the sources *disagree*?"),
and it arrived later, via the 7d currency work and ADR-027's strip. The spec's own grounding note
from design-lock records *"The epistemic dimension is empty … `contested_edges()` → `[]`"* — **the
feature was designed, locked and graded useful before stance existed at all.**

**So ADR-041 is not only "how to rebuild Node B" — it is whether to.** Five options: rebuild with the
co-occurrence passages + a neutral label + one pair per call · **retire stance, keep Node A** ·
keep the relation verb only · a deterministic tension proxy · park it. **Recommendation: retire**,
reopening as a properly-scoped feature (with the ground-truth study budgeted from the start) only on
an explicit product decision. Option 3 is explicitly warned against as a false compromise — the
relation verb comes from the same text-free, position-sensitive prompt (the position probe produced
*is used with · uses · is improved by · compared to · is compared to · improves on* for one pair) and
`relation_by_pair` keeps whichever document answered first.

**Costs stated honestly rather than waved through.** Retiring takes `contested` and
`superseded_trend` out of the product — the CHANGELOG feature list, ADR-027 D3's strip column, the
reviewer's `contested_evidence` tag — and **G3/G6's year-aware `superseded_trend` is collateral**: it
is deterministic and correct in itself but rides on stance-derived direction, so it goes too unless
re-based on `doc_years` alone. Whether that re-basing is feasible **has not been checked in code**,
and ADR-041 says so in its Confidence section.

**Rejected.** Writing a new "what is the concept graph" doc from scratch (the purpose was already
written; duplicating it would have created a second source to drift). Agreeing that the docs were
missing without looking — they were **scattered and stale, not absent**, and the fix for those is
different.

**Then the user supplied the decision input the ADR was waiting on, and it flipped the
recommendation.** The stated intent: *"see which claims are unsubstantiated and where are the
knowledge gaps … classify knowledge per concept in order to find the gaps where the user, for a given
subject, should find more resources … We want epistemics feature. That is the idea."*

**My "retire" recommendation was wrong, and the way it was wrong is worth keeping.** I read B1's spec
text — corroboration/coverage/navigation, all document counts — and concluded stance "answers a
question B1 never asked". That is true of the *text* and false of the *intent*. **A spec's surface
description is not its purpose**, and I had just written a whole page arguing that the docs' problem
was exactly this kind of disconnection. The measurement's finding is untouched (the current Node B
cannot serve the goal); what changed is that "this implementation is invalid" and "the feature is
unwanted" are different claims and only the first was ever supported.

**Two things the correction produced that the retire framing had hidden.**
1. **A unit mismatch nobody had named.** The goal is knowledge classified *per concept*; today's
   epistemics classifies **edges** (concept pairs) and reaches concepts only by aggregation.
2. **A cheaper first move — new ADR-041 option 6.** *"Which claims are unsubstantiated"* is a
   **support** question, not a polarity one, and the project already has a working, deterministic,
   retrieval-derived claim layer (`AnswerClaim`, `weakly grounded`/`unsupported`, `unsourced_claim`)
   built on the principle `how-answers-work.md` states outright — markers come from retrieval
   signals, never the model's own confidence. Re-basing per-concept status on it serves the headline
   goal with no new LLM pass. It cannot do polarity, so option 1 still owns "these sources disagree".

**Recommendation now: 6 → 1**, with option 4 (deterministic structure from the taxonomy) kept in view,
because the user's *"concepts are linked in general to predictable things"* makes a gap **a deviation
from expected structure** — which needs a source for the expectation, and ADR-028's curated taxonomy
is the one already in the tree. Same reference-class argument ADR-040 reached from the other side.

**What it opens.** The build sequence is open; the ADR no longer is. One thing keeps its deadline
regardless: **the surfaces must stop presenting stance-derived output as an epistemic finding** until
6 or 1 lands. Unscoped: whether `superseded_trend` survives on years alone.

---
## 2026-08-02 (3) — ADR-040 option 5 executed: Node-B stance is judged **without the document** and flips with **list position**. `contested` is not measuring the corpus (KI-33)

**What changed.** No source code. New instrument `scripts/validate_node_b_stance.py`, new baseline
`tests/eval/baselines/node_b_stance_validity_2026-08-02.md`, **KI-33** filed, ADR-040 given an
*Update* section that blocks every surfacing option behind a Node-B fix.

**Why.** Entry (2) concluded `contested` was a surfacing problem and put "validate the stance
extractor" first because every other option's value depended on the answer. It ran. The answer is
worse than expected: the signal is not a measurement of the corpus at all.

**Two structural facts, from the code, before any measurement.**
1. **The model never sees the document.** `build_messages(present_labels, pair_labels)` composes the
   entire user turn from concept labels and a numbered pair list — `annotate_relations`' own
   docstring says so — while the system prompt asks for stance *"from the document's apparent
   framing"*. There is no document in the prompt to have a framing.
2. **There is no neutral stance.** `POLARITIES` = supports/refines/contradicts/supersedes, two of
   them opposing, all mandatory. Citation-polarity corpora put neutral above 60% as the *majority*
   class. The vocabulary cannot express the common case, and its boundary is a hair wide: `refines`
   ("improves") supports, `supersedes` ("replaces") opposes, and the prompt's own example verb is
   *"improves on"*.

**The controlled experiment — one variable, four verdicts.** One document, same 7 present concepts,
same 17 pairs, `llama3.1:8b`, temperature **0.0** (all shipped settings), varying **only the target
pair's index** in the numbered list:

| index | 0 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| stance | supports | supports | **supersedes** | **contradicts** | **supersedes** | refines |

Four distinct verdicts **crossing the supporting/opposing boundary**, from a list position. Replaying
the five real documents' actual prompts reproduced **5/5** recorded stances, so this is the shipped
pipeline's deterministic behaviour, not sampling noise.

**A hypothesis refuted en route, kept because it cost a run.** The first guess was that the
*co-present concept set* drove the variation. Nine realistic contexts with the pair held at index 0
returned **`supports` 9/9** — stable and deterministic. It was position, not context. Two experiments
to get there; *reasoning proposed, measurement decided* — the third time that pattern has paid this
week.

**Supporting evidence from the artifact.** The whole integrity layer is **65 stance assignments**,
**30.8% opposing**, and **14 of 19 annotated edges carry more than one stance** (`re-ranking <-> BM25`
takes all four across 5 documents). Relation and stance contradict each other: `is a component of` →
`supersedes` ×2, `uses` → `contradicts` ×3, `builds upon` → all four.

**This subsumes entry (2)'s domain confound rather than competing with it.** Pair-list length scales
with a document's concept count, so dense documents → long lists → deep indices → opposing →
`contested`; sparse documents → one pair → index 0 → `supports`. That *is* the 7/9-vs-0/4
parent-field table. `cre`, `dbs`, `ntsr1`, `pddl` were never settled — their documents yield one pair.

**Rejected.** Swapping the model (facts 1 and 2 are structural — any model inherits them); raising
`max_tokens` (the JSON parses; KI-28 was a different failure); tuning the threshold (measured inert
in entry 2). Also rejected: implementing option 2 now — it is still correct that `ns=0` is not
contested, but the node it fixes came from a single `supersedes` on a one-pair prompt, so it was an
artifact, not the "one real defect" entry (2) called it. **That correction is the honest cost of
having measured the input second instead of first.**

**What it opens.** A **Node-B redesign** with its own ADR: pass the passages where the two concepts
co-occur, add a neutral/no-stance option, remove the position dependence (one pair per call or
equivalent), and carry a hand-labelled ground-truth study on the RG-015 template — which is the only
way to make an accuracy claim, since nothing here scores against ground truth. Until then `contested`
should not be presented as an epistemic signal, and ADR-040's options 1/2/3/4/6 cannot be evaluated.
Unquantified and worth knowing: what *share* of the 30.8% opposing is position-driven rather than
prior-driven.

---
## 2026-08-02 (2) — RG-019 measured: the `contested` floor everyone planned to add is **inert**, and the saturation is a surfacing problem (ADR-040)

**What changed.** No source code, and deliberately so. New instrument
`scripts/measure_contested_density.py`, new baseline
`tests/eval/baselines/contested_density_2026-08-02.md`, new **ADR-040** with its decision left open,
and RG-019 rewritten from an untested hypothesis into a measured negative result.

**Why.** The v0.4.0 walkthrough recorded the integrity strip — the product thesis — reading as
noise: **53.3% of assessed chunks marked `contested`** against 3.9% `corroborated`, 8 of 10 sources
in a live turn marked contested on a question that is not a controversy. Two records already agreed
on the cause and the cure. RG-019: *"triggers on `nc >= 1` … derive a named floor (min disputing
docs and/or an agreement-ratio band — the `MIN_DATED_DOCS_PER_SIDE` pattern)"*. ADR-027 shipped the
always-on strip without it, noting the strip would otherwise "ship saturated". **Neither had been
measured, and both are wrong.**

**Measured, $0, every counterfactual re-projected onto the real 18,831 chunk segments.**

| lever | density |
|---|---|
| shipped (`nc >= 1`) | **53.3%** — reproduces the walkthrough's 396/743 exactly, which is what validates the instrument |
| `nc >= 2` — *the prescribed fix* | **53.0%** |
| `nc >= 3` | 52.9% |
| `agreement_ratio < 0.70` | 53.2% |
| chunk rule "majority of claims contested" | 53.2% |

**The prescription accounts for two chunks.** Only **1 of 7** contested nodes has `nc == 1`. The
other six are the corpus's core vocabulary — BM25, dense retrieval, passage retrieval, contrastive
learning, re-ranking, hard negatives — each with genuine two-sided stance across **5–11 documents**.
`contested` is not misfiring; it is firing correctly on ordinary scholarly disagreement that the UI
then presents as cautionary.

**Three findings that outlive this item.**
1. **The denominator was half-quoted.** 53.3% is of *assessed* chunks, and only **3.9% of the store
   carries any claim** — marked chunks are **2.1%** of the store. Both true; the first alone
   overstates the marker's reach ~25x. It is still the number a user sees, because retrieval returns
   the chunks that carry claims.
2. **There are two stacked `>= 1` thresholds, not one.** The node rule, and `derive_markers` marking
   a chunk if *any* claim is contested. But **89% of assessed chunks carry exactly one claim**, so
   any/majority/all are the same rule here (53.3 / 53.2 / 53.0%). `n_contested >= 2` does cut to
   7.9% — by requiring two contested concepts in a chunk that mostly mentions one. A structural
   silencer, not an epistemic threshold.
3. **`agreement_ratio` is the only lever with range and the one that must not be used.** `<0.60` →
   27.1%, `<0.50` → 0.5%. But the seven observed values sit in **0.545–0.714**: every effective
   threshold is fitted to seven points on a 13-concept vocabulary. That is the
   over-optimise-on-the-current-corpus failure KI-19 exists to forbid.

**One real defect found.** `knowledge distillation` — `ns=0, nc=1, agreement=0.000`: zero supporting
sources, one disputing. Coverage is decided **contested-first**, so the unique-source neutrality
rule (Decision 4, "a sole source is never contested") never gets to judge it. A node with no support
is not contested, it is unsourced. Structural, no constant, fixable in a line — **not** done here,
because it lands with whichever ADR-040 option is chosen.

**Rejected.** Landing `nc >= 2` to close the item (it would record a fix that changes 0.3 points and
spend the corpus-tuning budget doing it). Picking a surfacing option unilaterally — the measurement
kills option 1, it does not choose between "surface the ratio", "re-frame the label" and "validate
the extractor first"; that is the user's call and ADR-040 says so in its status line.

**The finding that reframed all of it, found by asking what the threshold was measuring *against*.**
Every graph concept carries exactly one ANZSRC parent field (ADR-028) — **13/13 placed**. Joined:

| parent field | concepts | contested |
|---|---|---|
| Machine learning | 6 | **4** |
| Data management and data science | 3 | **3** |
| Artificial intelligence · Neurosciences · Med. chemistry · Biochemistry | 1 each | **0** |

**7 of 9 concepts in the two IR/ML fields are contested; 0 of 4 outside them.** `cre`, `dbs`,
`ntsr1`, `pddl` are not uncontested because they are settled — they have **one source each**. The
marker tracks *how densely the corpus covers a field*, not whether a claim is disputed, and all
three levers operate on a per-concept rate whose dominant term is a variable that rate does not
contain. **That is why no cut point works, and it is a stronger statement than "the levers are
inert".**

**The literature was checked, and it is unfavourable to every lever.** `agreement_ratio` is raw
percent agreement — the statistic Cohen's κ and Krippendorff's α exist to replace, with Landis &
Koch's bands as the standard cautionary tale about arbitrary cut points. Meta-analysis, the
discipline that actually owns "do sources disagree" (Cochran's Q, Higgins' I²), offered its
25/50/75% bands as tentative, is cautioned against mechanical application by the Cochrane Handbook,
and finds I² non-discriminative for prevalence meta-analyses — the closest analogue; practice
reports it with τ² and a prediction interval. Citation-polarity corpora put neutral citations above
60% with contrasting/negative the **rarest** class, against ~45% opposing sources per node here —
independent evidence that Node-B, not the corpus, produces this. Partial pooling / empirical-Bayes
shrinkage toward a parent mean is the named method for the hierarchy, with James–Stein dominating
raw group means for k≥3.

**ADR-040 gained a sixth option — score contestedness against the parent field's base rate.** It is
the only option that addresses the confound rather than routing around it, it removes the tunable
(the reference class is derived from data), and **this project already made the same argument once**:
ADR-006 rejected absolute keyword frequency for contrastive termhood against a background
distribution. Deferred rather than rejected, for a falsifiable reason: four of six fields hold one
concept, so no field base rate is estimable yet — a blocker that expires as `graph_include` grows
past 13. The instrument prints the cross-tab, so the re-test is a re-run.

**A framing kept even if every option is rejected:** *insufficient evidence is a state, not a low
score.* The schema already encodes it (`unique` = sole source, held NEUTRAL, Decision 4) and
contested-first precedence is what steals it — which makes the `ns=0` fix the first instance of a
principle rather than a one-node patch.

**What it opens.** The recommendation recorded in ADR-040 is **5 → 2 → 3, with 6 as the target
shape**: validate the Node-B
stance extractor first, because `llama3.1:8b` biased toward disagreement would reproduce this entire
picture and nothing yet separates the two — and its calibration is already recorded as suspect
(flat `rating` output, `gap_suggest_ollama_2026-07-08.md`). Then the `ns=0` gate regardless. Then
prefer the continuous surface over a rename. Still owed and unchanged: RG-019's precision
spot-check (this run argues structural correctness, it never reads the chunks) and density at a
second corpus size — **monotonicity in corpus size, the original worry, is still untested**.

---
## 2026-08-02 (1) — the v0.4.0 release commit left `uv.lock` at 0.3.0; **CI has been red on `main` since**, and the Docker build could never have worked

**What changed.** Two config lines, no source code. `uv.lock`'s own project entry
**0.3.0 → 0.4.0** (produced by `uv lock`, not hand-edited — the diff is exactly one line, zero
dependency churn), and **`README.md` removed from `.dockerignore`**.

**Why — and this is the part worth carrying.** The session opened on the baton's item 1,
`docker compose build`, the verification owed for `a052703`. The build never ran (Docker Desktop on
this box will not start its engine, below), but the two things that would have failed it were found
anyway, and the first is much larger than the Docker item.

**1. `uv sync --locked` fails at the v0.4.0 tag.** `47aabdd` bumped five version strings —
`pyproject.toml` · `package.json` · `tauri.conf.json` · `Cargo.toml` · `Cargo.lock` — and
**`uv.lock` was not one of them**. A uv lockfile records the project's *own* version, so it went
stale the moment `pyproject.toml` said 0.4.0. Both `.github/workflows/ci.yml:36`
(`uv sync --locked --extra cpu --extra dev`) and `Dockerfile:34` (`uv sync --locked --extra cpu`)
pass `--locked`, whose entire job is to fail rather than silently re-resolve.

**Confirmed against GitHub, not inferred** (`gh` is installed on this box now — the baton says it is
not; that fact is stale). The public Actions API says CI went red **exactly at the release commit**
and stayed red:

| run | sha | conclusion |
|---|---|---|
| 2026-08-01T21:09Z | `a052703` | **failure** |
| 2026-08-01T20:45Z | `47aabdd` | **failure** |
| 2026-08-01T14:10Z | `0cc2c3d` | success |

In both failed runs the failing step is **5, "Install dependencies"**, and steps 6–12 — ruff, ruff
format, mypy, pytest, bandit, pip-audit, detect-secrets — are all **`skipped`**. So **no lint, type,
test or security gate has run on `main` since the release**, and the v0.4.0 tag is CI-unverified.

**Reproduced and fixed on Linux, with the exact commands.** In the `~/pv-clean` clean-room tree left
warm by the 08-01 session, at `47aabdd` with the tag's shipped lockfile restored:
`uv sync --locked --extra cpu --extra dev` → **exit 1**, `error: The lockfile at uv.lock needs to be
updated, but --locked was provided`. After `uv lock`: the same command → **exit 0**, and the
Dockerfile's `uv sync --locked --extra cpu` → **exit 0**.

**Why five green local gate batteries missed it, which is the real lesson.** Nothing run locally
passes `--locked`: `just`/`uv run`/pre-commit all use the plain form, which re-resolves in silence.
**And the 08-01 clean-room run — the one ceremony designed to catch exactly this — used
`uv sync --extra cpu --extra dev`, not CI's `uv sync --locked …`.** It therefore *repaired* the
lockfile inside its own clone instead of failing on it. That is not a reconstruction: `~/pv-clean`
still carried an uncommitted `M uv.lock` whose whole diff is `-version = "0.3.0"` /
`+version = "0.4.0"`. **A clean-room check that does not run the shipped command validates a path
nobody ships.**

**2. `.dockerignore` excluded `README.md`, which the Dockerfile copies.** `pyproject.toml` declares
`readme = "README.md"`, so setuptools needs the file present to build this project's own metadata
during `uv sync`, and `Dockerfile:32` copies it for that reason. Docker matches `.dockerignore`
exactly and drops excluded paths from the build context, so the `COPY` fails before uv ever runs.
Both halves arrived together in the same unbuilt commit (`a052703` created `.dockerignore`; before
it the file was inert under the wrong name, so every earlier build had `README.md`). The exclusion
is now removed with the reason written at the line, so it is not re-added as tidy-up.

**Verified / not verified — stated separately, because they are not the same.** The lockfile fix is
verified end-to-end on Linux with both shipped commands. **The `.dockerignore` fix is reasoned, not
built** — the same caveat entry (6) carries for the rest of the Dockerfile, and it does not clear
yet. `docker compose build` is **still owed**.

**Docker Desktop 4.84.0 will not start on this box.** Installed per-user
(`%LOCALAPPDATA%\Programs\DockerDesktop`), CLI 29.6.2 / Compose v5.3.1. `docker version`, `docker
info` and `docker desktop status` all **hang on the named pipe** rather than erroring. Diagnosed:
WSL 2.7.11 is healthy and the `docker-desktop` distro boots by hand (kernel 6.18.33.2) but contains
**no `dockerd`** — only `/init`; **no Docker Windows service exists at all** (`Get-Service` and
`HKLM:\SYSTEM\CurrentControlSet\Services` both empty of docker entries); and `com.docker.backend` is
alive and answering, with the GUI polling `ErrorReportAPI GET /diagnostics/status` once a second —
the pattern of a startup-error screen waiting for a human. A clean kill-and-restart did not change
it. Needs eyes on the window; not fixable from a shell.

**Rejected.** Hand-editing the version line in `uv.lock` (ran `uv lock` instead — it is the
canonical producer, and it proves no dependency churn rode along). Installing Docker Engine natively
inside WSL Ubuntu to get a build (a sudo-level system change nobody asked for). Pre-emptively
"fixing" the Dockerfile further while it remains unbuildable — the whole point of this item is that
unbuilt Docker changes are how the repo got here.

**A third stale version, found by checking the rest of the class.** `apps/desktop/package-lock.json`
recorded `doc-assistant-desktop` at **0.1.0** against `package.json`'s 0.4.0 — stale since before
0.2.0, and unlike `uv.lock` **harmless**: there is no frontend job in CI at all, so nothing gates on
it, and npm reads the version from `package.json` regardless. Aligned anyway (both the root and
`packages[""]` fields, the two npm itself writes) so the release ritual has no exceptions to
remember.

**What it opens.** **A release bumps seven version strings, not five** — the checklist is missing
`uv.lock` and `package-lock.json`, and the first of those is the one that takes CI down. More
useful than the checklist: **`uv lock --check` is the cheap gate that would have caught this before
the tag** (it runs in ~1 s and needs no network), and it belongs either in the pre-commit battery or
at the release keypoint. Worth pairing with the wider lesson — a local battery that never runs the
*shipped* command can be green while `main` is red.
