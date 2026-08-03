<!-- status: active · updated: 2026-08-02 · class: append-only -->

# Node-B stance validity — the input to `contested` encodes list position, not documents (2026-08-02)

**Question.** ADR-040 option 5: before choosing how to *surface* `contested`, check the signal it
rests on. Every `contested` verdict traces back to Node-B stance annotations
(`concept_skeleton_enrich.annotate_relations`). Are they measuring the corpus?

**Answer: no.** Two code facts and three measurements, ending in a controlled experiment where the
verdict changes with nothing but a list index.

## Setup

| | |
|---|---|
| Corpus | 97 documents · skeleton **13 nodes / 19 edges** (`graph_include`-scoped, ADR-018) |
| Model | `llama3.1:8b` via local Ollama — `CONCEPT_SKELETON_LLM_MODEL`, the shipped Node-B default |
| Temperature | **0.0** — the shipped setting (`annotate_relations(temperature=0.0)`) |
| Instrument | `scripts/validate_node_b_stance.py` (`--replay`, `--positions`) |
| Cost | **$0** — local only, read-only, no writes to skeleton or DB |

## 1. Two facts from the code, before any measurement

**The model never sees the document.** `build_messages(present_labels, pair_labels)` composes the
entire user turn from concept labels and a numbered pair list. `annotate_relations`' own docstring:
*"the LLM is handed only that document's present concepts and the subset of skeleton edges among
them."* The system prompt nonetheless asks it to decide stance *"from the document's apparent
framing"* — of a document it is never shown. Stance can only come from the model's prior about the
concept pair, plus the shape of the prompt.

**There is no neutral stance.** `POLARITIES = ("supports", "refines", "contradicts", "supersedes")`,
of which `OPPOSING_POLARITIES = {"contradicts", "supersedes"}`. Every co-present pair must take one
of four labels and two of them are opposing. Citation-polarity corpora put neutral above 60% as the
majority class and contrasting/negative as the rarest; this vocabulary cannot express the majority
case at all.

The boundary is also thin by construction: `refines` ("extends, specialises, or improves") counts as
**supporting**, `supersedes` ("replacing or obsoleting") as **opposing** — and the prompt's own
example relation verb is *"improves on"*, which sits exactly on that line.

## 2. The layer is 65 judgments, a third of them opposing

| stance | n | share | side |
|---|---|---|---|
| supports | 28 | 43.1% | supporting |
| refines | 17 | 26.2% | supporting |
| contradicts | 12 | 18.5% | **opposing** |
| supersedes | 8 | 12.3% | **opposing** |
| **total** | **65** | | **30.8% opposing** |

The entire research-integrity layer — the product thesis — rests on **65 LLM judgments**.

## 3. Stance is unstable across documents, and relation verbs cross the boundary

**14 of 19 annotated edges carry more than one stance.** The extreme case, `re-ranking <-> BM25`,
takes **all four** across 5 documents: `{supersedes: 2, supports: 1, contradicts: 1, refines: 1}`.

Relation verb × stance shows the incoherence directly:

| relation | supporting | opposing | breakdown |
|---|---|---|---|
| improves on | 12 | 4 | refines 7, supports 5, contradicts 3, supersedes 1 |
| is a component of | 6 | 2 | supports 6, **supersedes 2** |
| uses | 4 | 3 | supports 4, **contradicts 3** |
| builds upon | 2 | 3 | supersedes 2, supports 1, contradicts 1, refines 1 |

*"X is a component of Y"* being `supersedes`, and *"X uses Y"* being `contradicts`, are not
defensible readings under any framing. The relation and the stance disagree with each other.

## 4. The replay is faithful — so the instability is real, not a reconstruction artifact

Rebuilding each document's actual prompt from `concept_presence` and re-running it:

| document | #present | #pairs | index | recorded | replayed |
|---|---|---|---|---|---|
| 068a28e5 | 3 | 3 | 2 | supersedes | supersedes |
| 15233bbd | 2 | 1 | **0** | **supports** | supports |
| 2d1a2037 | 5 | 9 | 8 | supersedes | supersedes |
| 4f2b71e1 | 7 | 14 | 13 | contradicts | contradicts |
| c495b879 | 7 | 17 | 16 | refines | refines |

**5/5 reproduced.** The pipeline is deterministic and the recorded stances are exactly what this
prompt produces. Note the covariate: the one document where the pair is **alone at index 0** is the
one that says `supports`.

## 5. The controlled experiment — position alone changes the verdict

One document. Same 7 present concepts, same 17 pairs, same model, temperature 0.0. **Only the
target pair's index in the numbered list varies:**

| index | relation emitted | stance |
|---|---|---|
| 0 | is used with | supports |
| 2 | uses | supports |
| 4 | is improved by | **supersedes** |
| 8 | compared to | **contradicts** |
| 12 | is compared to | **supersedes** |
| 16 | improves on | refines |

**Four distinct verdicts, crossing the supporting/opposing boundary, from a list index.** Nothing
about the document, the pair, or the model changed.

A refuted hypothesis on the way to this, kept because it cost a run: the first guess was that the
*co-present concept set* drove the variation. Holding the pair at index 0 and varying the context
across nine realistic sets gave **`supports` 9/9** — stable and deterministic. It is position, not
context. *Reasoning proposed; measurement decided.*

## Conclusion — the chain, end to end

1. Node B sees **no document text**; stance cannot reflect what a document claims.
2. **No neutral option** forces every co-present pair onto a four-way scale, two of which oppose.
3. Pair-list length scales with the document's concept count (`combinations(present, 2)` ∩ edges).
4. **Verdicts move with list position** (§5); a pair alone at index 0 reliably reads `supports`.
5. Therefore dense-coverage documents → long lists → deep indices → opposing → `contested`;
   sparse documents → one pair → index 0 → `supports` → not contested.
6. `node_weights_for_epistemics` reads step 5 as *documents disagreeing* → **53.3% contested**.

**This subsumes the parent-field confound** measured the same day
([`contested_density_2026-08-02.md`](contested_density_2026-08-02.md) §6): 7/9 concepts contested in
Machine learning + Data management, **0/4** outside. `cre`, `dbs`, `ntsr1` and `pddl` are not
uncontested because they are settled — their documents produce one-pair prompts. The domain confound
is the downstream shadow of the position artifact.

**Consequently every ADR-040 option except 5 is downstream of an invalid input.** No threshold
(option 1), precedence fix (2), continuous surface (3), relabel (4) or field-relative reference
class (6) can rescue a signal whose disagreement is manufactured by prompt composition. Filed as
**KI-33**.

## Caveats — what this does not establish

- ⚠ **One model.** `llama3.1:8b` only. A stronger model may be more stable, and the position
  sensitivity may shrink — but facts §1 (no document text, no neutral label) are **structural** and
  hold for any model. Do not read this as "swap the model and it is fixed".
- ⚠ **One pair probed for position**, the least stable one, chosen deliberately as the sharpest
  case. The layer-wide instability figure (14/19 edges) is what generalises; the six-position table
  demonstrates the mechanism rather than measuring its prevalence.
- ⚠ **Prevalence of the artifact is not quantified.** How much of the 30.8% opposing share is
  position-driven versus prior-driven is unmeasured. The claim is that the signal is *contaminated
  and unusable as an epistemic marker*, not that every opposing stance is an artifact.
- ⚠ **No ground truth was built.** Nobody read the source documents to establish what the true
  stance is. This shows the extractor cannot be measuring the documents (it never sees them); it
  does not score accuracy against a hand-labelled set. That remains owed if Node B is rebuilt —
  and RG-015's hand-labelled taxonomy study is the template.
