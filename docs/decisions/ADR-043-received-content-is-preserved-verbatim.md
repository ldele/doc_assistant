<!-- status: active · updated: 2026-08-11 · class: append-only -->

# ADR-043 — Content we received is preserved verbatim; normalisation is a choice, never a silent default

- **Status:** accepted (the vendored-artifact half is enforced today; the user-facing half is a direction, not a built feature)
- **Date:** 2026-08-11
- **Deciders:** user + Claude Code

## Context

A routine hygiene run surfaced a decision that is not routine.

`pre-commit run --all-files` failed on `data/anzsrc-2020-for-20210429.ttl`, and the
`trailing-whitespace` hook offered to fix it across ~28 lines. Read as a formatting nit, that is an
obvious yes. It is not a formatting nit. **All 28 of those lines sit inside Turtle triple-quoted
literals** — `skos:definition` and `skos:scopeNote` bodies. Zero are structural:

```
   18: '• fisheries sciences; '
  152: 'd) Agricultural water management is included in Group 3099 Other agricultural… '
 3855: '  skos:definition """This group covers macromolecular and materials chemistry. '
```

Inside `"""…"""` a trailing space is **part of the literal's value**. The hook was not proposing to
reformat the file; it was proposing to edit the text of a classification published by the Australian
Bureau of Statistics.

Three facts settled it, and each is checkable rather than asserted:

1. **The file is vendored, not authored here.** It is the official ANZSRC 2020 Fields of Research
   SKOS/Turtle from `linked.data.gov.au`, used under **CC BY 4.0**.
2. **Its whole job is to be a faithful copy.** The derived `data/anzsrc_2020_for.json` names it in
   `_meta.source_file` as the artifact it was generated from. That citation is the attribution
   chain; it is only worth anything if the cited bytes are the published bytes.
3. **Nothing parses it at runtime.** There is no `.ttl`, `rdflib`, `turtle`, or `skos` reference in
   `src/`, `scripts/`, `tests/`, or `apps/` — rdflib is not even a dependency. The runtime reads the
   JSON. So the edit buys **nothing** operationally and costs the one property the file has.

**The general shape, which is why this is an ADR and not a config comment.** An automated formatter
cannot distinguish *layout* from *content*. It is safe only where those coincide — in files we
author, in languages where whitespace is insignificant. It is unsafe wherever whitespace is
load-bearing (Turtle/YAML/Markdown literals, fixture files pinning exact bytes) **or** wherever the
bytes are someone else's record.

**And this project already does the unsafe thing to its users, by design.** Ingest normalises source
text on the way in: `strip_image_placeholders` (KI-14), page-marker stripping out of the evidence
block (KI-29), and the queued **de-hyphenation** pass (RG-025). Each has the same shape as the hook
— a transformation that improves the common case and silently destroys information in some cases,
applied by default, with no record of what changed and no way for the reader to see the original.
For a tool whose entire pitch is provenance and citation integrity, "we quietly rewrote your source
text and cited the result" is the failure mode that matters most.

## Options

1. **Strip the whitespace.** One command, `--all-files` goes green, no config. Trade-off: mutates a
   third-party CC-BY dataset so it no longer matches what was published, breaking the provenance
   claim the derived JSON makes about it — in exchange for nothing, since nothing reads it.
2. **Exclude the vendored artifact from content-mutating hooks.** The file keeps its bytes; the hook
   keeps its value everywhere else. Trade-off: an exclude pattern is a standing exception, and
   exceptions rot unless the reason travels with them — so the reason must be written at the
   exclusion site, not just here.
3. **Delete the `.ttl`** and keep only the derived JSON. Smallest repo, no exception. Trade-off:
   destroys the attribution chain and the ability to re-derive or audit the JSON against its source
   — the JSON's `_meta.source_file` would then cite a file nobody can produce.
4. **Move it out of the repo** (fetch on demand, like the eval corpus manifest). Keeps the tree clean
   and the provenance honest. Trade-off: a network dependency and a new failure mode for a 926 KB
   file that never changes; the eval-corpus pattern exists because those PDFs are large and
   redistribution is legally fraught, neither of which applies to a CC-BY vocabulary.

## Decision

**Adopt option 2, and generalise it into a rule: content that was received rather than authored here
is preserved byte-for-byte, and any tool that would rewrite it is excluded rather than obeyed.**

The deciding reason is that options 1 and 3 both trade a real property (this is what was published;
here is where it came from) for a cosmetic one (a green hook run; a smaller tree), and the project's
value proposition *is* the real property. A research-integrity tool that silently edits a source
document has refuted itself in miniature.

**Two tests tell the classes apart.** Both must pass before a formatter may touch a file:

- **Did we author it?** If the bytes came from somewhere else — a standards body, an upstream
  release, a user's document — they are a record, not a draft.
- **Is whitespace load-bearing?** Inside string literals, in fixtures pinning exact bytes, in
  significant-whitespace formats, the formatter cannot see the difference between layout and value.

**The exclusion carries its reason inline** (`.pre-commit-config.yaml`), because an unexplained
`exclude:` is indistinguishable from someone silencing an inconvenient hook, and the next reader
will "clean it up".

**The user-facing half — recorded here as direction, not as a built feature.** The same rule applies
to the documents a user ingests, where today it is *not* followed. The direction:

- **The original is never edited.** Normalisation produces a derived layer, which is already this
  project's law for everything else (Enrichment-Layer Pattern, non-negotiable #4). Extraction is the
  one place that law is bent, and it is bent on the user's own data.
- **Normalisation is inspectable and attributable.** A reader who sees a citation should be able to
  ask what was done to that passage between the PDF and the quote — de-hyphenation, marker
  stripping, ligature folding — and get an answer.
- **Where a transformation is genuinely a judgement call, it becomes an option** rather than a
  hardcoded default. De-hyphenation (RG-025) is the first real candidate: it is right for a
  line-wrapped PDF and wrong for a document where hyphens are meaningful (chemical names, code,
  hyphenated compounds), and that is a property of the corpus, not of the code.
- **Defaults stay opinionated.** "Make it an option" is not a licence to push every decision onto the
  user (`ux-no-friction-inform-dont-block`). The default should be the right answer for most
  documents; the option exists for the reader who can see it was wrong for theirs.

**What would reverse it:** a vendored artifact whose upstream itself changes shape often enough that
byte-fidelity stops being meaningful, or a case where a formatter exclusion hides a real defect in a
file we do in fact own. Neither applies to a frozen 2021 classification release.

## Consequences

**Easier.** `pre-commit run --all-files` is green, so a failure there is information again rather
than noise to be scrolled past — which is the whole reason the nuisance was worth fixing. The
attribution chain (`_meta.source_file` → the actual published bytes) holds. Future vendored data has
a decided policy instead of a per-file argument.

**Harder.** The exclude list is a thing that must be maintained and justified; each new entry needs
its reason written at the exclusion site. And the rule is now on record as applying to ingest, where
the code does not yet follow it — this ADR therefore creates a visible gap between stated policy and
shipped behaviour. That is intentional (the gap already existed; it was just unnamed), but it means
the de-hyphenation work inherits a constraint it did not have yesterday: it must be reversible and
inspectable, not merely correct on average.

**Must revisit.** What "inspectable normalisation" costs in storage and complexity — keeping the
pre-normalisation text for every chunk is not obviously affordable at 10,000 documents, and the
cheaper form (record the *transformations applied*, re-derive the original on demand) needs
designing. Also whether the settings surface can carry per-corpus ingestion options at all without
becoming the knob-farm ADR-010 deliberately avoided.

## Confidence

- ✓ **The literal-vs-structural split is measured, not assumed** — a classifier walked the file
  tracking `"""` state: **28 inside a literal, 0 structural**. Re-runnable.
- ✓ **"Nothing parses it" is verified** — no `.ttl`/`rdflib`/`turtle`/`skos` reference in `src/`,
  `scripts/`, `tests/`, `apps/`; `importlib.util.find_spec("rdflib")` is `None`.
- ✓ **The file is unchanged and structurally intact** — byte-identical to HEAD (EOL-normalised),
  284 balanced `"""` delimiters, 5 `@prefix` declarations with no undeclared prefix in use,
  2,203 `skos:Concept`, terminating `.`, final newline. Attribution chain present in the derived
  JSON's `_meta`.
- ⚠ **The user-facing half is unbuilt and unmeasured.** No design exists for how a normalisation
  record would be stored, surfaced, or paid for at scale, and no user has asked for the option — it
  is inferred from the product's own integrity claim plus the user's note that these choices may
  become user options. Treat it as direction; it needs its own ADR before anything is built.
- ⚠ **"Upstream is frozen" is an assumption.** ANZSRC 2020 is a published release and the file name
  pins a 2021-04-29 snapshot, but nothing in the repo checks it against upstream, and nothing would
  notice if it were edited locally by something other than a hook.
