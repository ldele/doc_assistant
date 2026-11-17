<!-- status: active · updated: 2026-11-17 (built; the gate run and its control recorded under As built); 2026-11-17 (accepted by the user as written, the same day; written after their four answers) · class: living -->

# Spec — What a turn sends the model: fitted to the window, and said (ROADMAP 101, session A5)

> The executor brief. Target executor: **Claude Code**. Session A5 of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`, local). It reads, and must not re-litigate:
> ADR-060 choice 7 (a server's unknown window is shown as unknown, never guessed), ADR-066 (modes
> change only the framing; A7 builds them on this), the fence of S-9 (`pipeline.fence_passage`),
> and `docs/specs/eval-harness-chat.md` (A5's gate). **Accepted by the user on 2026-11-17; buildable.**

## In one sentence

Every model call of a turn is measured before it is sent, fitted to the model's window by a fixed
order of what gives way, recorded in the turn's provenance, and, when something was left out, said
on the answer: no prompt is cut silently by a server again.

## Today (read in the code, 2026-11-17)

- **The answer step** sends `ANSWER_PROMPT` with two inputs: the fenced passages and the
  rewritten question (`pipeline.stream_answer`, `prompts.py`). It never sees earlier turns.
- **The rewrite step** (`pipeline.rewrite`) sends `REWRITE_PROMPT` with the last 6 messages,
  verbatim, full answers included (`pipeline.format_history`), whenever the session has history.
  `Session.history` itself grows without a cap.
- **Windows:**
  - On Ollama the app sets `num_ctx` to `OLLAMA_CONTEXT_TOKENS` (8,192). A longer prompt is cut by
    the server, keeping its end, and the call succeeds silently (KI-66's cause).
  - On an OpenAI-compatible server the app cannot set a window. Ollama's `/v1` cuts the same way
    (KI-94), and the app warns only when the server reports usage (`providers.prompt_cut_note`).
  - Anthropic refuses an oversized request, which becomes a failed turn.
- **Nothing is fitted** by the app on the chat path, and nothing records a prompt's size, the
  window, or what a call carried. `prompt_version` hashes `ANSWER_PROMPT` only, not
  `REWRITE_PROMPT`.

## Decisions — the user, 2026-11-17

| # | Question | Decided | Why |
|---|---|---|---|
| 1 | Should the answer step see earlier turns? | **No, in this first form.** It stays passages + rewritten question; A5 makes the rewrite's history deliberate instead | Smallest change with a measurement behind it: the conversation family says whether the rewrite carries the thread. Turns in the answer prompt stay the next seam |
| 2 | When a call does not fit, what gives way first? | **Oldest history first:** whole earlier messages, oldest first; then the newest earlier answer is shortened; only then the lowest-ranked passages. The instructions, the question and the best passage are never cut | The thread costs least to lose at its oldest end; evidence is what the answer stands on |
| 3 | A server whose window is unknown? | **Fit to a stated floor and say so:** the floor is the window the app sets on Ollama; the answer says the server's window is unknown, so the turn was kept to that size | Never cut silently, never claim a window the server has not stated (ADR-060 choice 7). The person can state the real window |
| 4 | Where is "left out" shown? | **A note on the answer** when anything was dropped, plus the full record in provenance; silent when nothing was | A cut that changes what the answer could see is something the reader needs at the answer, not only in a record |

## The contract

### What a call is measured against

- **The window.** One value per call, with where it came from (`window_source`):
  - `set`: a window the app itself sets (Ollama's `num_ctx`).
  - `server`: a window the server documents (`ModelList.windows`, read by ADR-060's adapter).
  - `documented`: a window the provider documents for the model, from a table in code.
  - `floor`: none of the above. The window is unknown, and the turn is fitted to the floor
    (`config.OLLAMA_CONTEXT_TOKENS`) and says so (decision 3).
- **The reserve.** The tokens kept free for the reply: the call's own `max_tokens` where the app
  sets one, else a structural reserve named in `config`.
- **The estimate.** One shared estimator, `context.estimate_tokens(text)`: an upper estimate with
  no tokenizer. It moves out of `knowledge/wiki.py` (UTF-8 bytes / 3), so the wiki and the chat
  cannot disagree about how big a text is. An upper estimate leaves room to spare rather than cut
  past the window.

### What gives way, in order (decision 2)

For the **rewrite call**, the budget is the window minus the reserve minus the fixed prompt and the
question:

1. Earlier messages are taken newest first, whole, up to the existing cap of 6 messages, while they
   fit.
2. If not even the newest exchange fits, its answer is shortened from its end, marked with `…`,
   to what fits.
3. If the question alone does not fit, the rewrite is skipped and the raw question is searched.
   This is recorded and said, never an error.

For the **answer call**, the budget is the window minus the reserve minus the fixed prompt and the
rewritten question:

1. Passages are taken in rank order, whole and fenced, while they fit.
2. The best passage is always kept. If it alone does not fit, it is clipped from its end inside
   its fence, and that is recorded.
3. Passages that do not fit are left out of the prompt, and of the answer's sources: an answer must
   not cite a passage its model never saw.

### What is recorded

A new nullable `AnswerRecord.context_json` (additive column), one object per turn:

```
{
  "window": int, "window_source": "set" | "server" | "documented" | "floor",
  "rewrite": {"estimated_tokens": int, "history_used": int, "history_dropped": int,
              "answer_shortened": bool, "skipped": bool} | null,
  "answer":  {"estimated_tokens": int, "passages_used": int, "passages_dropped": int,
              "top_passage_clipped": bool}
}
```

`prompt_version` hashes `REWRITE_PROMPT` too, so a rewrite change is visible in provenance. That
moves every `prompt_version`, once.

### What is said (decision 4)

When anything was dropped, shortened, clipped or skipped, or the window is a floor, the answer
carries one plain line, built from the record and never from the model. For example:

- "2 earlier messages and 1 passage were left out to fit the model's window."
- "This server does not state its window, so the turn was kept to 8,192 tokens."

Nothing is said when nothing gave way. The source panel shows the record.

## Hard constraints

- **Model- and budget-neutral** (CONTEXT rule 10): the rules above hold on any model. Which models
  the gate runs on is the plan's and the run's.
- **No tokenizer, no new dependency.**
- **No silent cut.** A call the app sent must fit its window by the app's own estimate, or the
  turn says why it did not.
- **The fence is unchanged:** fitting works on whole fenced passages, and a clip happens inside a
  fence, never across one.
- **0 documents:** the answer call is not made (KI-74), and the record says so.

## Guard tests

- A conversation longer than the window on a fake model: the oldest messages are dropped first, the
  record counts them, and the answer carries the note.
- The best passage survives a window too small for the rest; a dropped passage is in neither the
  prompt nor the sources.
- An unknown window fits to the floor and says so; a `set` or `server` window says nothing when
  everything fits.
- `prompt_version` changes when `REWRITE_PROMPT` changes.
- The wiki's budget tests still pass on the moved estimator.
- No call built by the controller exceeds its window by the shared estimate (a property over the
  eval's cases and synthetic long ones).

## Done when

- The guard tests pass.
- A conversation of 12 or more turns on `llama3.1:8b` keeps its thread with no silent cut. "Keeps
  its thread" means the last turn's rewrite names the conversation's subject, checked on the long
  cases of the conversation family.
- The gate of `docs/specs/eval-harness-chat.md` § The gates passes for A5 (`conversation` and
  `single`; citation coverage and `contains_all` on the last turn) against the H4 baseline on each
  column of the panel, or the run's record says why it does not.

## Out of scope

- Earlier turns in the answer prompt (decision 1's seam).
- A running summary of the conversation.
- What the library knows (A6's lookups).
- Modes (A7).
- A window the person types for a server: the record and the note make the floor visible first.
- KI-94's routing of local Ollama to the native adapter.

## As built (2026-11-17)

**Code.** `src/doc_assistant/context.py` holds the estimate (moved from the wiki, which now imports
it), the window (`window_for`), the two fits (`fit_history`, `fit_passages`), the record
(`ContextRecord`) and the note. The controller fits the rewrite call, then the answer call after
retrieval (not in the human mode, which calls no model), trims the sources to the passages the
model was given, writes the record to `AnswerRecord.context_json` (a new nullable column, migrated
additively) and appends the note to the citation block as **Fitted to the window:**.
`prompt_version` hashes `ANSWER_PROMPT` and `REWRITE_PROMPT` together. Guard tests:
`tests/unit/test_context.py` (14). The property over the eval's cases was checked on the run's
records instead of in a test: see below.

**The gate run** (eval home of the 28 public papers; panel `llama3.1:8b` and `qwen2.5:7b` through
the controller; `--repeat 3`; store notes `A5 context: …`, `A5 context r2: conversation`,
`A5 control HEAD: …`). Against H4, conversation on llama read as a measured drop in citation
coverage (−0.133, interval [−0.254, −0.017]), while H4 was measured before A2's fence and A2 was
gated on `hostile` and `single` only. A same-day control on the commit before A5 settled it:

| conversation, HEAD → A5 (3 → 6 trials) | citation coverage | `contains_all` |
|---|---|---|
| llama3.1:8b | −0.030 [−0.153, +0.091], no difference | −0.049 [−0.170, +0.052], no difference |
| qwen2.5:7b | −0.108 [−0.229, +0.009], no difference | +0.056 [−0.076, +0.167], no difference |

`single` against H4: llama +0.045 coverage and +0.028 `contains_all`, no difference at this n;
qwen +0.091 coverage, higher at this n, and −0.023 `contains_all`, no difference. No turn failed.

**Nothing gave way on the eval's cases.** All 126 recorded turns of the conversation rerun and all
78 of the long case carried every passage and every message the cap allows; the window came from
`set` every time. The largest answer call was estimated at 7,041 tokens against a budget of 7,168
(8,192 less the reserve): the fit is one long passage away from acting on the public set. So on
these cases A5 sent the model what it sent before, and the gate's movements are not A5's.

**Done when, as measured.** The 13-turn GPipe case (scratch, one paper, every follow-up leaning on
the ones before) kept its thread on llama in 3 of 3 trials on A5 and 3 of 3 on the control, and
in 0 of 2 in the day's first two A5 runs; qwen in 1 of 3 on each. Where it is lost, the last
rewrite turns "this library's paper" into a search that does not name GPipe: the rewrite's
quality, not the window (all six messages were carried). Decision 1's seam, earlier turns in the
answer prompt, is where that is addressed.

**Found on the way.** The eval harness never migrated its data home, so after a new column every
answer record failed, said only as a line on the answer, never logged. `run_eval` now runs
`init_db` before the controller adapter, and the controller logs `provenance_capture_failed`.

## Open questions

| # | Question | State |
|---|---|---|
| 1 | The structural reserve where the app sets no `max_tokens` (the Ollama chat path) | settled in the build: 1,024 tokens (`context.REPLY_RESERVE_TOKENS`); where the app sets `max_tokens`, that number |
| 2 | Whether `documented` needs a table at all, or whether the floor serves every provider without a stated window | settled in the build: one row, Anthropic at 200,000, because the floor would cut a Claude turn to a 25th of its window |
