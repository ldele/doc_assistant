<!-- status: active · updated: 2026-11-02 · class: append-only -->

# ADR-060 — Other answer providers come through one OpenAI-compatible adapter, and the address says where the text goes

- **Status:** proposed (Claude Code, 2026-11-02). Not accepted, not built. The direction is the
  user's, 2026-11-01: *"make sure we give more options than claude and ollama in Provider Model. I
  am sure there are norms now to do this."* The shape below is Claude's proposal, for 0.7.2's
  piece 3 (`docs/plans/PLAN_2026-10-31_next-two-releases.md`, local).
- **Date:** 2026-11-02
- **Deciders:** user + Claude Code
- **Extends** [ADR-011](ADR-011-desktop-provider-apikey-management.md) (the live switch, the
  reviewer following it) and [ADR-034](ADR-034-in-app-provider-setup.md) (the key store, readiness,
  and its amendment of 2026-11-01: where the text goes comes from the address). Adds a third adapter
  to both call shapes of [`docs/specs/llm-provider-isolation.md`](../specs/llm-provider-isolation.md).
- **In one sentence:** every answer provider other than Claude and Ollama goes through one client
  of our own that speaks the OpenAI-compatible chat API, set up from a table of presets or a server
  you add. It sends only the model, the messages and `stream`, lists each server's models when you
  open its form, and works out from the address alone whether your text stays on this computer.
  Anything that leaves this computer counts as paid.

## To decide — nine choices, each with a recommendation

1. **The client library.** *Recommended:* our own client over `httpx`, which is already a
   dependency (0.28.1) and already in the frozen build. It is a `langchain_core` chat model for the
   streaming path and an `LLMClient` for the one-shot path, both over one request function.
   `langchain-openai` is rejected: its current release needs a newer `langchain_core` than the
   whole app runs on, and brings `openai` and `tiktoken` (Options). **The frozen build is proved
   by behaviour, not by import:** the RG-012 harness answers one cited turn through a fake
   compatible server inside the sandbox. TLS uses the OS trust store as KI-10's path does, and a
   unit test proves that with `sys.frozen` set.
2. **How presets are stored.** *Recommended:* one row per preset in a table in a new module
   (`doc_assistant/providers.py`): id, name, address, where to get a key, who bills, a note, the
   date its address was checked. No code branches on a preset's id. A preset's address is fixed;
   another address, such as a proxy or a regional endpoint, is a server of your own. Your servers
   live in `settings.json` (name, address; no secret), as the selection does (ADR-011 fork H). Not
   a JSON file: the build freezes `doc_assistant` as code only (`collect_submodules`,
   `scripts/doc_assistant_api.spec:56`), and this project has already shipped a build that
   imported fine and then failed at runtime for want of a data file (the spec's comment on
   `pymupdf`, RG-012 2026-08-05).
3. **The route that takes a secret.** *Recommended:* the setup router stays the only module that
   accepts a secret. Its two Anthropic routes become routes keyed by provider id:
   `POST/DELETE /api/setup/providers/{id}/key`, and `POST /api/setup/servers` for *Test
   connection* on a server of your own (name, address, optional key). Verification is the free
   model list (choice 8), with ADR-034's three outcomes unchanged: refused → 400 and nothing
   stored; unreachable → stored, with the reason; accepted → stored. **The key is bound to its
   address.** It is sent to the address it was saved with and nowhere else. Changing a server's
   address removes its key, and the form says so first. An `http://` address off this computer
   warns at the key field and does not block. Anthropic keeps `.env`-first (ADR-034 B2); the new
   providers read the app's store only. That rule exists so the app and the CLI agree on one key,
   and no CLI uses these providers. Without it, an `OPENAI_API_KEY` exported for another tool would
   be used without the user typing it here.
4. **Where the text goes.** *Recommended:* one function, from the address only, with three
   outcomes. *This computer* is `localhost`, `*.localhost` or a loopback address. *Your local
   network* is a private or link-local IP literal, and the text still leaves this computer. Every
   other address, **including every name but `localhost`**, reads *goes to \<host\>*. A name is
   never looked up to decide this: the sentence is a promise, and a name can point anywhere. An
   address that does not parse counts as off this computer. The sentence shows on the *Answering
   with* card, on each provider's form and in *Test connection*'s result. Each answer's provenance
   records it with the host.
5. **What counts as paid (KI-4).** *Recommended:* `config.PAID_PROVIDERS`, a set of names, gives
   way to one test: paid unless the address is on this computer. That covers every cloud preset, a
   server on your network, an address that does not parse, and an Ollama whose `OLLAMA_HOST` is
   another machine. Nothing unknown is free. The screens say *Leaves this computer* rather than
   *paid*, because a lab server bills nobody. The CLI guard's banner names the host. The cost chip
   stops pricing every non-Ollama turn at Haiku's rates (`controller.py:899-902`). On any provider
   but Claude it reads *Billed by \<vendor\>; Provenote does not know its price.*
6. **What is sent, and what never is.** *Recommended:* the body is `model`, `messages` and
   `stream`, and nothing else. The headers are `Content-Type`, plus `Authorization: Bearer` when a
   key is stored. Never sent: `temperature`, a token cap under either name, `stream_options`,
   `response_format`, `tools`, `n`, `seed`, `logprobs`, `user`, or any header that names this app
   to a third party. The reader takes `choices[0].delta.content` when it is a string and ignores
   everything else: role-only chunks, usage chunks, reasoning fields, SSE comments.
7. **When a server's context window is unknown (KI-66).** *Recommended:* the window is shown as
   unknown, never guessed, and this ADR does not fit or cut the prompt (row 101 owns that). A
   preset whose model list carries a window under a documented field records it. A server on this
   computer or your network gets one line on its form. It says Provenote cannot set that server's
   window, that an answer prompt runs to about 5,000 tokens on 10 passages (KI-66's measurement),
   and that the model needs at least the 8,192 tokens Provenote asks of Ollama
   (`config.OLLAMA_CONTEXT_TOKENS`). A server that reports `usage.prompt_tokens` below half of what
   was sent gets a warning on that answer.
8. **Model lists, Claude included.** *Recommended:* each provider's form lists the models its
   server reports. Claude's come from `GET /v1/models` through the SDK, which is free, and its
   `capabilities`. Ollama's come from `/api/tags`, which is built. A compatible server's come from
   `GET {address}/models`. A list is fetched on opening the form, on saving a key, on *Test
   connection* and on *Refresh*: never per settings read, never in the background, never on the
   answer path. It is held in memory with the time it was fetched. A preset or server on this
   computer, such as LM Studio's *Not running*, is probed alongside Ollama on
   `/api/setup?probe=true`, within the same 2 s budget. Nothing off this computer is probed that
   way. A refused key reads *Key
   refused* until it is changed. A typed model that is not in the list is named and still saved. A
   list proves the key and the model, never the balance, so *Ready* says when the key was accepted
   and that the balance is not checked.
9. **The tests that prove it at $0.** *Recommended:* before any paid call, in this order: the
   request body and the stream reader on recorded streams (`httpx.MockTransport`); the address
   table; every preset driven through one fake server with only its row changed; keys and lists
   on mocked failures. Then a real turn through the real `ChatController` against a fake
   compatible server; a real local server driven in the app on a scratch data copy; and the frozen
   build's RG-012 turn. A paid message on a preset comes last, one preset at a time, each at the
   user's word (memory *cost-discipline-prove-llm-logic-offline-first*). The list in *Consequences*
   names each test.

## In practice — one question, a cloud preset, then a server of your own

The question is *"What does retrieval add to a language model's answers?"*, asked as a follow-up in
a conversation over a library that holds the RAG paper (Lewis et al., 2020). Model names, counts and
the lab address are examples; the code paths and the measured numbers are cited.

**Before.** Provider & model, *Answering with*: Claude Haiku 4.5 · *Ready* · *Your questions and the
passages found for them go to api.anthropic.com.*

**1. Groq, a preset.** The row reads *Needs a key*. Its form:

> **Groq** — Your own key, billed by Groq.
> Your questions and the passages found for them go to api.groq.com.
> Address `https://api.groq.com/openai/v1` (the preset's; not editable)

The user pastes a key and presses *Save key*. The app sends one request, which bills nothing:
`GET https://api.groq.com/openai/v1/models` with `Authorization: Bearer <key>`.

| The server answers | The form says | Stored |
|---|---|---|
| 200, 19 models | *Key accepted at 14:02. 19 models. Groq's balance is not checked.* | key + address, under `groq` |
| 401 or 403 | *Groq refused this key.* | nothing (400) |
| no network | *Could not reach api.groq.com to check the key. Saved.* | key + address |

The model box offers the 19 names and accepts a typed one. *Use for answers* writes
`llm_provider: "groq"` to `settings.json`. *Answering with* now reads: \<model\> on Groq · *Ready* ·
*Your questions and the passages found for them go to api.groq.com.*

The user asks the question:

| Step | Runs on | What leaves this computer |
|---|---|---|
| The follow-up is rewritten (`pipeline.rewrite`) | api.groq.com | the question and up to six earlier messages, answers included (`format_history`, `[-6:]`) |
| Keyword and vector search, rerank | this computer | nothing |
| The answer prompt | api.groq.com | the instructions, 10 passages with their file name and page (`TOP_K`, `format_docs_for_prompt`), the question |
| The answer streams back; citations are checked | this computer | nothing |
| A flagged answer is reviewed (the reviewer follows the switch, ADR-011 fork C) | api.groq.com | the question, the passages, the answer |

The answer request, in full:

```text
POST https://api.groq.com/openai/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <the key saved for this address>

{"model": "<model>", "messages": [<the answer prompt's messages>], "stream": true}
```

Under the answer: *Sent to api.groq.com. Billed by Groq; Provenote does not know its price.* The
provenance records `groq`, the model and the host.

**2. A server of your own, on this computer.** *Add an OpenAI-compatible server*: name *Lab
server*, address `http://localhost:1234/v1`, no key. *Test connection* sends
`GET http://localhost:1234/v1/models`:

> Reached it. 3 models found.
> Stays on this computer: this address is on your computer.
> Provenote cannot set this server's context window. An answer prompt runs to about 5,000 tokens;
> give the model at least 8,192, or the start of the prompt, which asks for citations, may be cut
> without an error.

*Use for answers*. *Answering with*: \<model\> on Lab server · *Ready* · *Your questions and the
passages found for them stay on this computer.* The same question runs the same five steps, and
every row reads *this computer*. Under the answer: *Stays on this computer. No metered cost.*

**3. The same server, moved to the lab machine.** The address becomes
`http://192.168.1.40:1234/v1`. Before saving, the form says:

> Your questions and the passages found for them leave this computer: they go to 192.168.1.40, on
> your local network.
> This address is not encrypted: the text crosses the network readable.

Had the server held a key, the form would add: *Changing the address removes the saved key. Enter
it again for the new address.* For the credit guard this server now counts as paid. Written as
`http://gpu-box:1234/v1`, the sentence reads *they go to gpu-box*, because the name is not looked
up.

## Context

- **Two providers, each native, named in eight places.** Claude is `ChatAnthropic` plus
  `AnthropicClient`; Ollama is `OllamaLLM` plus `OllamaClient`. The plan counts five places:
  `llm.make_client`, `pipeline.build_chat_model`, `readiness.PROVIDERS`, `credentials.keyed_providers`
  and the `/api/setup/anthropic-key` route, plus the wire models and `core/types`. Three more are
  named here. `app_settings.set_llm_selection:107` refuses any other name.
  `chat_controller.helpers._is_local:52` decides *Local model, no metered token cost* from the
  name `ollama`. `controller.py:899-902` prices every other turn at $1 and $5 per million tokens.
  `pipeline.build_chat_model:121` builds an Ollama model for **any** name that is not `anthropic`,
  so a third provider wired only halfway would answer on Ollama without a word.
- **The norm.** An OpenAI-compatible `/v1/chat/completions` with `/v1/models`, a bearer key and an
  editable address. The plan checked these documented addresses against the vendors' docs on
  2026-11-01: Gemini `https://generativelanguage.googleapis.com/v1beta/openai/`, Groq
  `https://api.groq.com/openai/v1`, OpenRouter `https://openrouter.ai/api/v1`, LM Studio
  `http://localhost:1234/v1` and Ollama `http://localhost:11434/v1/`. Mistral's chat path is
  documented, but its model list path was inferred. OpenAI's own address was not fetched.
- **Compatible is not identical** (the plan's check). Gemini ignores parameters it does not list;
  Groq returns 400 on some and rewrites `temperature` 0. Claude refuses `temperature` from Opus 4.7
  on, which already cost one fix in 0.7.2 (`llm.is_older_claude`, DEVLOG 2026-11-01 (4)).
- **Ollama's compatible endpoint cannot set the window.** Ollama's page (fetched 2026-11-02):
  *"The OpenAI API does not have a way of setting the context size for a model."* Through `/v1`,
  Ollama would cut long prompts again, which is KI-66, fixed only on the native path by `num_ctx`.
- **Where the text goes already comes from the address** for Claude and Ollama
  (`readiness._destination`, ADR-034's amendment), with two outcomes: this computer, or a host.
- **What the pipeline needs.** It composes `PROMPT | self.llm` and calls `.stream` and `.invoke`
  (`pipeline.py:673, 694, 709`), so the chat model must be a LangChain runnable; the reviewer and
  the judge need `LLMClient.complete`.
- **A model list proves less than it seems.** On 2026-11-01 this machine's key listed 14 Claude
  models for free, and a message on the same key was refused for lack of credit (the plan; DEVLOG
  2026-11-01). Today *Ready* on Claude means only that a key is present, so a dead key reads ready.
- **Measured, or says so.** Citation quality is measured on Claude Haiku and two local models
  (KI-36, probably confounded by KI-66), nowhere else.

## Options (the client library, choice 1)

1. **A thin client of our own over `httpx`** (recommended). It adds no package, and the body sent is
   a literal in our code that a test pins. *Trade-off:* we own an SSE reader and the error mapping.
   That is a small surface because choice 6 sends and reads so little.
2. **`langchain-openai`.** It fits the LangChain stack. *Trade-off, from PyPI on 2026-11-02:*
   1.7.3 requires `langchain-core>=1.7.0`, while `uv.lock` holds 1.4.9, so every LangChain package
   in the app would move with it (anthropic, ollama, chroma, huggingface, community, classic). That
   is a change of its own, with its own eval. It also requires `openai>=2.45.0`; `openai` 3.42.1
   requires `httpx2`, a second HTTP stack beside `httpx` 0.28.1, and `jiter>=0.16` against the
   locked 0.15.0. It requires `tiktoken`, which depends on `requests` to fetch its encoding files
   by address on first use (`tiktoken_ext/openai_public.py`; not re-read for this ADR). Whatever
   counts tokens through it would call a host the user never chose. And the body becomes the
   library's, version by version, where choice 6 needs it to be ours. An older release that fits
   `langchain-core` 1.4 would pin an old client for a body we would override anyway.
3. **The `openai` SDK alone.** *Trade-off:* the same `httpx2` and `jiter` moves, and it still
   needs a LangChain wrapper.
4. **LiteLLM.** Ruled out: its PyPI package was compromised on 2026-03-24 (1.82.7, 1.82.8), and it
   is far more than this needs (the plan).

For choice 2, a bundled data file was weighed and set aside for the freeze reason given there. A
catalogue fetched at runtime breaks the offline promise. For choice 4, looking up a name's address
was set aside: a lookup can change between the check and the request, and the sentence must hold
for both.

## Decision

Option 1, with the nine recommendations. **The deciding reason:** compatibility lives in what is
sent. Owning the body turns *"compatible is not identical"* into one tested literal, and it adds
nothing to the frozen build.

**What would reverse it:** presets needing more vendor differences than a row can hold, or row
102's tools needing a full client. Then `langchain-openai` comes in after a `langchain-core` upgrade
of its own, measured on its own.

## Consequences

- **Easier:** a new vendor is one row. Any server that speaks the norm works without a release.
  Claude's form lists its models, and a key that was checked and refused stops reading *Ready*.
- **Harder: what changes together.** The eight places above;
  `controller.py:495`'s reviewer kind (`llm_haiku` or `llm_<provider>`); `services.py:96`'s
  provider list and `models/settings.py:24`'s `Literal["anthropic", "ollama"]`; the wire models,
  `core/types/setup.ts`, `core/api/setup.ts` and `ProviderSetup.svelte`. `credentials.json` gains
  `keys: {id: {key, address}}`, and its `anthropic_api_key` field is still read: additive, no
  rewrite. The answer's provenance gains the host and where the text went (additive).
  `build_chat_model` and `make_client` raise on an unknown id instead of falling through.
- **The tests (choice 9).** (1) The body's keys are exactly `model`, `messages` and `stream`, and
  `Authorization` goes only to the saved address. (2) The stream reader on recorded streams: a
  role-only chunk, a usage chunk with empty `choices`, reasoning fields, `:` comments, `[DONE]`, an
  error object mid-stream, 4xx with JSON, 5xx with HTML, a dropped connection. (3) The address
  table, covering `localhost`, `127.0.0.1`, `::1`, `0.0.0.0`, `10.0.0.5`, `172.20.1.1`,
  `192.168.1.40`, `169.254.1.1`, `fe80::1`, a public IP, `localhost.example.com`, `gpu-box`,
  `api.groq.com`, an empty address and one that does not parse, each with its sentence and its
  paid flag. (4) Every preset row: its address parses, it is `https` unless the row is local, and
  it has a key page when it takes a key; one parametrized test drives every row through the same
  fake server. (5) Keys: an exported `GROQ_API_KEY` does not reach the app, an address change drops
  the key, the old Anthropic field still reads, and no log line carries a key. (6) Model lists on
  200, 401, 403, 404, a timeout and bad JSON, with Claude's through a mocked SDK. (7) One turn
  through the real `ChatController` and retrieval on the test corpus against a stdlib fake server
  that streams a cited answer. One of its routes demands a key, standing in for a cloud preset.
  The answer cites, the provenance names the host, and the chip names no price. (8) A real local
  server on this machine, with one thinking model, driven in the app on a scratch data copy.
  (9) The RG-012 turn in the frozen build. Then a paid message, one preset at a time, at the user's
  word. A real key's model list is free and may come first.
- **Unmeasured, and the panel says so:** citation quality on every new provider (KI-36's line
  stays on each form), and the reviewer's verdicts there. A flagged answer's review also bills the
  chosen provider, because the reviewer follows the switch unless `REVIEWER_PROVIDER` is pinned.
  Without `temperature`, these reviews sample at the server's default.
- **Thinking models on these servers:** their reasoning fields are not shown, so the first visible
  token can wait (memory *thinking-models-stall-the-stream*). Some local servers put the reasoning
  inline in the content. Test 8 checks for that; if it happens, it is filed as a known issue, not
  stripped quietly.
- **Robustness contract:** the corpus does not shape providers. At 0 documents a turn asks no model
  (KI-74), as today. A list of several hundred models, as a router returns, needs a model box that
  filters as you type.
- **Left out:** tools and function calling (row 102, after S-9). Open Responses, launched 2026-01-15
  and adopted by Ollama and vLLM (the plan), belongs to row 102 if it comes at all. A bundled model
  catalogue of names, windows and prices comes later, and a fetched one would break the offline
  promise. Also left out: LiteLLM; native adapters for vendors other than Claude and Ollama; APIs
  that authenticate other than by bearer or put the model in the path (a server of your own cannot
  reach those); prices; an OS keychain (ADR-034's file store stands). **What stays where it is:**
  the embedder and reranker on this computer, figure descriptions, keyword scans and every
  enrichment runner (`--provider` still takes `anthropic` and `ollama`), and the eval judge, which
  is a pinned instrument (the isolation spec).
- **Must revisit:** a preset whose vendor moves its address (a release fixes the row, and the row
  carries the date it was checked); Claude's `capabilities` replacing the `is_older_claude` name
  rule; Open Responses if compatible servers converge on it for chat.

## Confidence

- ✓ The documented addresses: the vendors' docs, 2026-11-01 (the plan). Gemini, Groq, OpenRouter,
  LM Studio and Ollama are confirmed.
- ⚠ OpenAI's address was not fetched, and Mistral's model list path is inferred. Both are confirmed
  against the vendors' docs before their rows ship.
- ✓ The `langchain-openai` dependency facts: PyPI JSON, 2026-11-02, against `uv.lock`.
- ✓ Ollama's `/v1` cannot set the window: Ollama's OpenAI-compatibility page, fetched 2026-11-02.
- ✓ A Claude model list is free and proves the key, not the balance: the 2026-11-01 measurement.
- ⚠ The minimal body (`model`, `messages`, `stream`) is accepted by each preset: this is the norm's
  minimum, but no preset has been driven yet. A free model list and one paid message per preset at
  the user's word prove it.
- ⚠ Cloud vendors refuse an over-long prompt with an error rather than cutting it. No vendor has been
  checked, and a fake server cannot prove it.
- ⚠ The usage check in choice 7 helps only where a server reports usage unasked, and how many do is
  unmeasured. Its *below half* is a guess set against a rough estimate (characters / 4, as the cost
  chip uses). It would have caught KI-66's cut, about 5,000 tokens down to 2,050, and it is not
  tuned beyond that.
