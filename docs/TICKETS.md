<!-- status: active · updated: 2026-10-01 · class: living -->

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

## T-001 — A cited number and a bibliography line need their record: where a figure was measured, on which version and data, and which source a reference line names
- **Status:** open · 2026-10-01
- **From:** llm-technical-writing-poc (ADR-052) · 2026-10-01
- **Symptom:** Reviewing 20 flagged figures in other projects' reports, the owner could not rule on 13: a number's meaning depended on context the passage did not carry, and a reference line was judged as a paragraph
- **Reproduce:** C:/Projects/llm-technical-writing-poc/docs/decisions/ADR-052-a-line-is-judged-as-what-it-is-and-a-number-by-its-record.md, parts 3 and 4: one context window for every judge (title, section path, neighbouring paragraphs, table header), and a record per measured number. ProveNote holds the source records a reference line can be checked against, and its retrieval chunks can carry the same kind, section and role metadata
- **Resolution:** —
