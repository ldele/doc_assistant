<!-- status: active · updated: 2026-08-07 · class: baseline -->

# Citation coverage — before/after the unbracketed source header (2026-08-06/07)

**Question.** The 2026-08-06 prompt change (`format_docs_for_prompt` stopped bracketing the source
header, to remove the shape models were copying as a citation) fixed the *format* failure that
failed RG-012. Did it also move **coverage** — the fraction of an answer's sentences carrying a
resolving citation? And do the figures the app publishes in its provider picker still hold on the
code that actually shipped?

**Answer: no, and yes.** Coverage did not move meaningfully on any provider; the published figures
remain in range.

## Method

- 35-case private set → **the 8 cases whose source document is degraded or absent were excluded**
  (`hh_*`, `hebb_*`, `hubel_wiesel_*`, `middleton_*`, `not_in_library_rlhf`). A case whose document
  is unreachable has nothing to cite; mixing it in measures retrieval, not citation.
- 27 remaining cases → the real `ChatController` on the live 97-document / 33,105-chunk corpus.
- Shipped settings; provider forced through `app_settings.get_llm_selection` **and**
  `config.REVIEWER_PROVIDER` (KI-38: env vars cannot do it — `load_dotenv(override=True)`).
- **Coverage** = cited sentences ÷ total sentences, pooled across all 27 answers (not a mean of
  per-answer ratios, which over-weights short answers).
- **1 repeat per arm.** This is the measurement's main limitation — see *Precision* below.

## Result

| provider / model | before (bracketed header) | after (shipped) | Δ | answers citing nothing |
|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` | 155/191 = **81.2%** | 157/188 = **83.5%** | +2.4 pp | 0/27 → 0/27 |
| `ollama/llama3.1:8b` (shipped local default) | 79/217 = **36.4%** | 77/205 = **37.6%** | +1.2 pp | 11/27 → **12/27** |
| `ollama/qwen2.5:7b` | 40/296 = **13.5%** | 43/239 = **18.0%** | +4.5 pp | 18/27 → 18/27 |

Paired by case:
- `llama3.1:8b` — **no** answer that cited nothing started citing; one that *had* cited stopped
  (`rajpurkar_arrhythmia`).
- `qwen2.5:7b` — one started (`deeplabcut_overview`), one stopped (`comparative_connectomics`).
- `haiku-4.5` — unchanged at zero.

## Reading it

**All three deltas are small, same-signed, and inside the noise this measurement can resolve.** One
repeat per arm, n=27, against a retrieval path with **~3% case-level non-determinism already on the
record** (DEVLOG 2026-08-01). Same-signed movement across three models is mildly suggestive of a
real, tiny effect, but it is not separable from noise here and **must not be reported as an
improvement**. What is solid is the *negative*: nothing moved enough to change any conclusion.

**This is the second independent line of evidence that KI-36 is a capability floor, not a prompt
artifact.** The first was cross-provider — the same prompt gets 81% from Haiku and 36% from
llama3.1:8b. This is the same-provider version: changing the prompt cured a format failure outright
(header-copies 6 → 0, RG-012 FAIL → PASS) and left coverage where it was. Prompt engineering fixed
what prompt engineering can fix, and did not touch the rest.

**Precision.** The two runs bracket each arm and give a rough sense of spread: llama 36.4–37.6,
qwen 13.5–18.0, haiku 81.2–83.5. Treat the published figures as ±2 pp for the paid arm and local
llama, and ±5 pp for qwen — which is why `ProviderSetup.svelte` states them as indicative context
for a choice, not as a benchmark result.

## Consequence for the shipped app

`ProviderSetup.svelte` publishes **36% / 14% / 81%**, measured on the *previous* prompt.
**Deliberately not updated in v0.4.1**: every delta is within the noise above, so churning a tagged
artifact would trade a real risk (an unverified rebuild) for a cosmetic gain. Refresh the card at
the next release that rebuilds for another reason, using the *shipped-prompt* column.

## Provenance

Runners are disposable (scratchpad); the method above is what to re-create. Raw per-case rows were
captured in the session that produced this file. Related: KI-36, DEVLOG 2026-08-05 (2),
2026-08-06 (3), 2026-08-07 (1).
