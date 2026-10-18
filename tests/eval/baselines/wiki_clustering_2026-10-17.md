<!-- status: active · updated: 2026-10-17 (RG-018 measured: both wiki clustering paths on the working library, dry run and an Ollama --apply run read by hand; the default stays; KI-65 found) · class: baseline -->

# Wiki clustering on the working library: neither path partitions it (RG-018)

**The default, cosine union-find at `WIKI_MIN_SIMILARITY` 0.90, puts 87 of 104 documents in one
topic.** The concept-community path, which the monolith recorded as the right primitive, puts 75
of 104 in topics of one document each, because only 29 documents mention one of the graph's 13
concepts. Read by hand after an `--apply` run on `llama3.1:8b`, the community path's five small
topics hold 23 of their 29 documents on one subject, and the cosine path's two small topics are
coherent (8 documents). The 87-document topic is not a topic, and its note was written from a
prompt Ollama cut from 15,245 tokens to 2,050 (**KI-65**).

**Verdict on the default: keep it.** Flipping `WIKI_USE_CONCEPT_COMMUNITIES` would trade one blob
for 75 single-paper notes and would scatter the six political-science papers the cosine path groups
correctly. Neither path is usable as a topic map of this library. What would change the verdict is
in §4.

These are counts and one reader's judgement on one library of 104 documents, read on one day. They
say what each path does here; they are not a quality score for either primitive, and they will move
with the vocabulary and the corpus. The wiki is a CLI sidecar (`scripts/build_wiki.py`, Obsidian
notes under `data/wiki/`); no screen in the app shows it.

## 1. What was asked

RIGOR_TODO RG-018: the community path is built but off by default, validated only on the 10-paper
public corpus, while the default threshold was tuned on that same corpus. Owed: cluster count and
size distribution of both paths on the current corpus, a hand-read of note coherence, then a
decision on the default.

## 2. The dry run: how many topics, how big

Both paths over the same 104 documents and the same stored similarity edges (`DocSimilarity`:
10 neighbours per document, 1,040 edges, `bge-base`).

| | cosine at 0.90 (default) | concept communities |
|---|---|---|
| topics | 12 | 80 |
| topics of 2+ documents | 3 (87, 6, 2) | 5 (15, 5, 4, 3, 2) |
| single-document topics | 9 | 75 |
| largest topic, share of the library | 87 (84%) | 15 (14%) |
| documents in a single-document topic | 9 (9%) | 75 (72%) |
| `[[links]]` per note, median (max) | 1.5 (11) | 12 (43) |

**Why the cosine path makes one blob.** The edges are dense at the top: their median score is
0.922, the 25th percentile 0.897, the 5th 0.850. A 0.90 threshold keeps about three quarters of the
edges, and union-find joins anything they chain together. Swept on the same edges:

| threshold | topics | of 2+ documents | largest |
|---|---|---|---|
| 0.85 | 2 | 1 | 103 |
| 0.88 | 3 | 2 | 97 |
| **0.90** | **12** | **3** | **87** |
| 0.92 | 26 | 4 | 76 |
| 0.93 | 35 | 3 | 68 |
| 0.94 | 47 | 9 | 18 |
| 0.95 | 61 | 10 | 11 |
| 0.96 | 79 | 10 | 6 |

The blob breaks between 0.93 and 0.94, and any value picked from this table would be fitted to
this library (the KI-19 failure). That is the "wrong primitive" the monolith recorded, measured.

**Why the community path makes singletons.** It groups documents by the Louvain community of the
graph concepts they mention: 13 concepts in 5 communities (graph `dbf0e870aea6fed1`), present in 29
of 104 documents. A document that mentions none of them is its own topic, by construction
(`concept_skeleton.doc_clusters_from_skeleton`). Its topic count is bounded by the graph's
vocabulary, not by the library.

**Links.** Both paths derive `[[links]]` from the stored similarity edges that cross two topics.
On the community path almost every edge crosses, so a note links to a median 12 others.

## 3. The `--apply` run, read by hand

`llama3.1:8b` on Ollama, temperature 0, the shipped prompt; 12 notes in 92 s and 80 notes in 327 s;
no call failed and no note came back empty. Each topic of two or more documents was read against
its member titles: is it one subject, and does the summary describe the members?

**Cosine path**

| docs | note title | on one subject | the summary |
|---|---|---|---|
| 87 | Automatic discovery of cell types and microcircuitry from neural connectomics | no: transformers, retrieval, pose estimation, neuroanatomy, medical imaging, physiology | describes one paper ("This paper presents a method…") — see below |
| 6 | Analyzing Political Theory and Polarization | 6 of 6 | faithful |
| 2 | ECG Arrhythmia Detection and Classification | 2 of 2 | faithful |

**The 87-document note was written from a cut prompt.** Its material is 55,165 characters, 15,245
tokens. `OllamaClient` sets no `num_ctx`, Ollama's VRAM-based default here is 4,096, and its log
reads `truncating input prompt limit=2050 prompt=15245 keep=4 new=2050`: roughly the last dozen
documents survived, and the note is named after one of them, the 76th of 87. The next-largest
prompt of the session (15 documents, 2,703 tokens) was not cut. Filed as **KI-65**: about 175
tokens a document means a topic past about 20 documents loses material on this model, silently.

**Community path**

| docs | note title | on one subject | the summary |
|---|---|---|---|
| 15 | Document Representation Learning and Retrieval | 12 of 15 — off: a motion-capture primer, a latent-embedding paper for behaviour and neural data, an attribution-map paper (they reach the retrieval community through `contrastive` and embedding vocabulary) | faithful to the 12 |
| 5 | Genetic Tools for Studying Brain Circuits | 4 of 5 — the Parkinson's disease review is tangential | faithful to the 4 |
| 4 | Multi-scale Neural Network Architectures | 2 of 4 — Res2Net and the distillation survey; RNN regularisation and Sentence-BERT are not | fits 2; its tags came back as one run-on string |
| 3 | Hypothalamic Stimulation and Brain Networks | 3 of 3 | attributes one paper's patients to all three |
| 2 | Temporal Planning and Hierarchical Planning | 2 of 2 | faithful |

So 23 of the 29 documents in the community path's topics sit with their subject, against 8 of the
cosine path's 95 documents in multi-document topics (its two small topics, with the blob counted as
off).

**Single-document notes.** Nine on the cosine path, all read; ten of the 75 on the community path,
sampled with seed 17. Each summary describes its paper. All of them call a single paper "this topic
cluster" or "the papers in this cluster", which the prompt invites. The community path's singletons
include all six political-science papers, which mention none of the 13 concepts.

## 4. The verdict, and what would change it

**Keep `WIKI_USE_CONCEPT_COMMUNITIES=false` and `WIKI_MIN_SIMILARITY=0.90`.** Flipping moves the
failure from one blob to 72% singletons, loses the one subject the cosine path finds that the
graph does not know (political science), and gives every note a dozen links. Retuning the threshold
would fit a number to this library. Neither is a fix; the measurement says the wiki is not a usable
topic map of this library on either path, and the recorded refinement (links from inter-community
skeleton edges) would not change that.

**Re-measure when** (a) the graph's vocabulary covers most of the library: the community path's
singleton share is set by `graph_include`, which KL2 and the vocabulary work grow; the same
re-test trigger as RG-019's (`graph_include` past 13); or (b) a relative primitive replaces the
absolute threshold (mutual nearest neighbours, or a per-document percentile of its own edges), which
is design work with its own ADR, not a flip; or (c) KI-65 is fixed, since a topic of more than about
20 documents cannot be read for coherence through its note until then.

## 5. Environment and method

- **Library:** the working library, 104 documents, 357 vocabulary rows (13 on the graph), graph
  `dbf0e870aea6fed1`, at commit `93edbe6`.
- **Dry run:** `python -m scripts.build_wiki --no-use-concept-communities` and
  `--use-concept-communities` (read-only), and a probe that calls the same functions to record the
  shape, the edge distribution and the sweep: `.claude/probes/2026-10-17-rg018/measure.py` →
  `measure.json`.
- **Apply run:** `.claude/probes/2026-10-17-rg018/apply.py llama3.1:8b`: the same `build_wiki` the
  CLI calls, provider fixed to `ollama` (KI-4), `wiki_dir` pointed at the probe folder so
  `data/wiki/` was not overwritten (its notes still date from 2026-07-05). It records each topic's
  prompt size. Output: `apply.json`, and the 92 notes in `wiki_notes.zip` (zipped: the docs gate
reads every `.md` under `.claude/`). Ollama's log:
  `%LOCALAPPDATA%\Ollama\server.log`, 2026-10-17 13:45 local.
- **Cost:** $0; 92 local calls.
- **Reproduce:** the two dry-run commands above; for the notes,
  `python -m scripts.build_wiki --apply --provider ollama --model llama3.1:8b` (writes `data/wiki/`).
  The shipped default model `llama3` is not installed on this box.

## 6. Bounds

One library, 104 documents, heavily machine learning and neuroscience; one reader, no second
rating; one model at temperature 0. "On one subject" was judged from titles, with the summary
checked against them, not from the papers' text. The single-document sample on the community path
is 10 of 75. The cosine sweep is on stored edges of one embedder (`bge-base`, 10 neighbours each);
another `k` changes it.
