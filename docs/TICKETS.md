<!-- status: active · updated: 2026-10-14 · class: living -->

# TICKETS — issues reported against doc_assistant

One section per report, newest first (cpc ADR-047). A ticket is something somebody **reported** —
a consumer running a gate, a reader of a doc, a user of the app — and it is owed an answer. Open
one with `cpc-ticket new --title "…" --from "who (version) · date"`; close it with
`cpc-ticket close --id N --as fixed|declined|duplicate --note "where the answer went"`.
`cpc-ticket check` fails when this file stops being a ledger; `cpc-digest` lists the open ones.

Not a known issue: that is a weakness this project found in itself, logged the second time it bit
(`.claude/KNOWN_ISSUES.md`). A ticket may become one, or a fix, or a deliberate no — the
`Resolution` line says which. Nothing here is rewritten or deleted; a declined ticket keeps its
reason.

<!-- Tickets below, newest first. The five lines are read by cpc-ticket and cpc-digest:
     ## T-NNN — title
     - **Status:** open | triaged | fixed | declined | duplicate  · date
     - **From:** who reported it, at what version · when
     - **Symptom:** one line: what was seen, and where
     - **Reproduce:** the command, or the file and line
     - **Resolution:** — (while open) | a DEVLOG date, a CHANGELOG version, KI-N, or why not -->

## T-002 — Update cpc from 1.12.0 to 1.18.1 at the start of the next session
- **Status:** fixed · 2026-10-14
- **From:** cpc session, Code (unknown) · 2026-10-12
- **Symptom:** This project vendors cpc 1.12.0, and v1.18.1 was tagged on 2026-10-12. Measured from the cpc checkout on this tree, 1.12.0 against 1.18.1 over docs_check, init_check, sprint_check and integrity_check: +0 new findings and one new NOTE, which is not a finding: the entry context holds about 8,290 tokens and no token cap is set. The owner asked on 2026-10-12 that each project update first thing in its next session.
- **Reproduce:** From this root, where CPC is the cpc checkout and DIR a scratch folder: git -C CPC worktree add --detach DIR v1.18.1, then PYTHONPATH=DIR/src python -m cpc.init --root . with no --profile, then git -C CPC worktree remove DIR. Afterwards regenerate the generated docs, run the gates and write the DEVLOG line cpc-init prints. How it is run, and what is optional afterwards, is in the cpc checkout under migrations: 2026-10-01-revendor-and-ticket-triage, 2026-10-11-known-issues-entry-tokens-and-counts, and 2026-09-30-shared-launcher-and-commands for the launcher this version lays. This tree held 6 uncommitted paths when this ticket was written, so the work in progress comes first.
- **Resolution:** DEVLOG 2026-10-14

## T-001 — A cited number and a bibliography line need their record: where a figure was measured, on which version and data, and which source a reference line names
- **Status:** open · 2026-10-01
- **From:** llm-technical-writing-poc (ADR-052) · 2026-10-01
- **Symptom:** Reviewing 20 flagged figures in other projects' reports, the owner could not rule on 13: a number's meaning depended on context the passage did not carry, and a reference line was judged as a paragraph
- **Reproduce:** C:/Projects/llm-technical-writing-poc/docs/decisions/ADR-052-a-line-is-judged-as-what-it-is-and-a-number-by-its-record.md, parts 3 and 4: one context window for every judge (title, section path, neighbouring paragraphs, table header), and a record per measured number. ProveNote holds the source records a reference line can be checked against, and its retrieval chunks can carry the same kind, section and role metadata
- **Resolution:** —
