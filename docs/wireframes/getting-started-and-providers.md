<!-- status: active · updated: 2026-11-02 · class: living -->

# Wireframe: Getting started and Provider & model

The scene is `getting-started-and-providers.excalidraw` beside this file; open it on the canvas with
`excalidraw-canvas import docs/wireframes/getting-started-and-providers.excalidraw`, edit, and export
it back over the same file. Four screens, approved by the user on 2026-11-01 at rev 176, after one
round of review in chat and no markup on the canvas: Getting started on a fresh install, Getting
started when ready, Provider & model, and a server of your own. All four are built: screens 3 and 4's
other providers and server form with ADR-060, 2026-11-02.

**What the drawing decides.**

- Getting started is two rows, Model and Documents. Each has a status and a link to the tab that does
  the job, and neither holds a form. Other rows and options wait until they are needed.
- Ready means both rows are green: the model with *Engine online*, and the number of documents
  indexed. An engine that goes offline turns the Model row red, and the same link fixes it.
- Everything about answer models lives in Provider & model, keys included.
- The top card says which model is answering and where the text goes, worked out from the address.
- Providers are grouped by where the text goes: on this computer, in the cloud with your own key, and
  your own server.
- Each cloud provider has the same form: its address is filled in, you add the key and choose a model
  from its list. Claude and Ollama keep their own connections.
- A server of your own needs a name, an address and an optional key. *Test connection* lists its
  models and says whether the text stays on this computer. An address off this computer receives the
  key, the questions and the passages found for them, and the panel says so.
- The panel says which models citation quality was measured on, and that others were not.

**What the backend needs before this can be built.** `/api/setup` already carries what Getting started
shows. Provider & model needs one OpenAI-compatible adapter behind the presets, decided in an ADR first.

The model name, the document count and the server are examples.

## Wireframe

Score at rev 176: `fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, 4 screens, 173 components.

```text
## Wireframe reading
4 screens, 173 components, 3 connections, 0 annotations.
Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).
### Screen 1 "Getting started: a fresh install"  [s1] rectangle 1160x820
  1. header  [s1-hdr] rectangle 1160x72 · fill #f8f7f3
    1.1. heading "Getting started: a fresh install"  [s1-title] text 800x34 · 26px
    1.2. button "Close"  [s1-close] rectangle 88x44 · fill #ffffff
  2. sidebar  [s1-rail] rectangle 220x748 · fill #f8f7f3
    2.1. button "Getting started"  [s1-rail0] rectangle 188x44 · fill #4a3fa6
    2.2. button "Documents"  [s1-rail1] rectangle 188x44 · fill #ffffff
    2.3. button "Provider & model"  [s1-rail2] rectangle 188x44 · fill #ffffff
    2.4. button "Retrieval"  [s1-rail3] rectangle 188x44 · fill #ffffff
    2.5. button "General"  [s1-rail4] rectangle 188x44 · fill #ffffff
  3. heading "2 things left before Provenote can answer"  [s1-h] text 800x26 · 20px
  4. text "Searching your documents runs on this computer.…"  [s1-sub] text 800x20 · 13px
  5. card  [s1-c1] rectangle 876x104 · fill #ffffff
    5.1. icon  [s1-c1-i] ellipse 24x24 · fill #ffffff
    5.2. text "Model"  [s1-c1-t] text 400x24 · 16px
    5.3. text "No model chosen yet."  [s1-c1-d] text 460x20 · 13px
    5.4. text "Not set"  [s1-c1-s] rectangle 130x30 · fill #f8f7f3
    5.5. button "Provider & model →"  [link-model1] rectangle 196x40
  6. card  [s1-c2] rectangle 876x104 · fill #ffffff
    6.1. icon  [s1-c2-i] ellipse 24x24 · fill #ffffff
    6.2. text "Documents"  [s1-c2-t] text 400x24 · 16px
    6.3. text "Nothing indexed yet."  [s1-c2-d] text 460x20 · 13px
    6.4. text "Not set"  [s1-c2-s] rectangle 130x30 · fill #f8f7f3
    6.5. button "Documents →"  [link-docs1] rectangle 196x40
  7. text "Each row links to the tab that does the job, an…"  [s1-note] text 860x40 · 12px
### Screen 2 "Getting started: ready"  [s2] rectangle 1160x820
  1. header  [s2-hdr] rectangle 1160x72 · fill #f8f7f3
    1.1. heading "Getting started: ready"  [s2-title] text 800x34 · 26px
    1.2. button "Close"  [s2-close] rectangle 88x44 · fill #ffffff
  2. sidebar  [s2-rail] rectangle 220x748 · fill #f8f7f3
    2.1. button "Getting started"  [s2-rail0] rectangle 188x44 · fill #4a3fa6
    2.2. button "Documents"  [s2-rail1] rectangle 188x44 · fill #ffffff
    2.3. button "Provider & model"  [s2-rail2] rectangle 188x44 · fill #ffffff
    2.4. button "Retrieval"  [s2-rail3] rectangle 188x44 · fill #ffffff
    2.5. button "General"  [s2-rail4] rectangle 188x44 · fill #ffffff
  3. heading "Provenote is ready"  [s2-h] text 800x26 · 20px
  4. text "It can answer from your documents."  [s2-sub] text 800x20 · 13px
  5. card  [s2-c1] rectangle 876x104 · fill #ffffff
    5.1. icon  [s2-c1-i] ellipse 24x24 · fill #dcefe1
    5.2. text "Model"  [s2-c1-t] text 400x24 · 16px
    5.3. text "Claude Haiku 4.5"  [s2-c1-d] text 460x20 · 13px
    5.4. text "Engine online"  [s2-c1-s] rectangle 130x30 · fill #dcefe1
    5.5. button "Provider & model →"  [link-model2] rectangle 196x40
  6. card  [s2-c2] rectangle 876x104 · fill #ffffff
    6.1. icon  [s2-c2-i] ellipse 24x24 · fill #dcefe1
    6.2. text "Documents"  [s2-c2-t] text 400x24 · 16px
    6.3. text "30 documents indexed"  [s2-c2-d] text 460x20 · 13px
    6.4. text "Indexed"  [s2-c2-s] rectangle 130x30 · fill #dcefe1
    6.5. button "Documents →"  [link-docs2] rectangle 196x40
  7. text "If the engine goes offline, the Model row turns…"  [s2-note] text 860x40 · 12px
### Screen 3 "Provider & model"  [s3] rectangle 1160x820
  1. header  [s3-hdr] rectangle 1160x72 · fill #f8f7f3
    1.1. heading "Provider & model"  [s3-title] text 800x34 · 26px
    1.2. button "Close"  [s3-close] rectangle 88x44 · fill #ffffff
  2. sidebar  [s3-rail] rectangle 220x748 · fill #f8f7f3
    2.1. button "Getting started"  [s3-rail0] rectangle 188x44 · fill #ffffff
    2.2. button "Documents"  [s3-rail1] rectangle 188x44 · fill #ffffff
    2.3. button "Provider & model"  [s3-rail2] rectangle 188x44 · fill #4a3fa6
    2.4. button "Retrieval"  [s3-rail3] rectangle 188x44 · fill #ffffff
    2.5. button "General"  [s3-rail4] rectangle 188x44 · fill #ffffff
  3. card  [s3-now] rectangle 876x92 · fill #ffffff
    3.1. text "Answering with"  [s3-now-l] text 300x16 · 12px
    3.2. text "Claude Haiku 4.5"  [s3-now-m] text 600x22 · 16px
    3.3. text "Ready"  [s3-now-s] rectangle 102x28 · fill #dcefe1
    3.4. text "Your questions and the passages found for them …"  [s3-now-w] text 740x18 · 13px
  4. text "On this computer"  [s3-g1] text 320x16 · 12px
  5. list-item  [s3-p-ollama] rectangle 330x42 · fill #ffffff
    5.1. text "Ollama"  [s3-p-ollama-n] text 180x20 · 14px
    5.2. text "Ready"  [s3-p-ollama-s] rectangle 112x26 · fill #dcefe1
  6. list-item  [s3-p-lmstudio] rectangle 330x42 · fill #ffffff
    6.1. text "LM Studio"  [s3-p-lmstudio-n] text 180x20 · 14px
    6.2. text "Not running"  [s3-p-lmstudio-s] rectangle 112x26 · fill #f8f7f3
  7. text "Cloud, with your own key"  [s3-g2] text 320x16 · 12px
  8. list-item  [s3-p-claude] rectangle 330x42 · fill #e4e1f5
    8.1. text "Claude"  [s3-p-claude-n] text 180x20 · 14px
    8.2. text "Ready"  [s3-p-claude-s] rectangle 112x26 · fill #dcefe1
  9. list-item  [s3-p-openai] rectangle 330x42 · fill #ffffff
    9.1. text "OpenAI"  [s3-p-openai-n] text 180x20 · 14px
    9.2. text "Needs a key"  [s3-p-openai-s] rectangle 112x26 · fill #f8f7f3
  10. list-item  [s3-p-gemini] rectangle 330x42 · fill #ffffff
    10.1. text "Google Gemini"  [s3-p-gemini-n] text 180x20 · 14px
    10.2. text "Needs a key"  [s3-p-gemini-s] rectangle 112x26 · fill #f8f7f3
  11. list-item  [s3-p-mistral] rectangle 330x42 · fill #ffffff
    11.1. text "Mistral"  [s3-p-mistral-n] text 180x20 · 14px
    11.2. text "Needs a key"  [s3-p-mistral-s] rectangle 112x26 · fill #f8f7f3
  12. list-item  [s3-p-groq] rectangle 330x42 · fill #ffffff
    12.1. text "Groq"  [s3-p-groq-n] text 180x20 · 14px
    12.2. text "Needs a key"  [s3-p-groq-s] rectangle 112x26 · fill #f8f7f3
  13. list-item  [s3-p-openrouter] rectangle 330x42 · fill #ffffff
    13.1. text "OpenRouter"  [s3-p-openrouter-n] text 180x20 · 14px
    13.2. text "Needs a key"  [s3-p-openrouter-s] rectangle 112x26 · fill #f8f7f3
  14. text "Your own server"  [s3-g3] text 320x16 · 12px
  15. button "Add an OpenAI-compatible server"  [add-server] rectangle 330x42 · fill #ffffff
  16. card  [s3-detail] rectangle 530x500 · fill #ffffff
    16.1. heading "Claude"  [s3-d-h] text 400x26 · 20px
    16.2. text "Your own key, billed by Anthropic."  [s3-d-a] text 494x18 · 13px
    16.3. text "Your questions and the passages found for them …"  [s3-d-w] text 494x38 · 13px
    16.4. text "API key"  [s3-key-l] text 340x16 · 12px
    16.5. input  [s3-key] rectangle 340x40 · fill #ffffff
      16.5.1. text "sk-ant-..."  [s3-key-v] text 316x18 · 13px
    16.6. button "Save key"  [s3-save] rectangle 142x40 · fill #ffffff
    16.7. text "Model"  [s3-model-l] text 340x16 · 12px
    16.8. input  [s3-model] rectangle 340x40 · fill #ffffff
      16.8.1. text "Claude Haiku 4.5"  [s3-model-v] text 316x18 · 13px
    16.9. button "Use for answers"  [s3-use] rectangle 142x40 · fill #4a3fa6
    16.10. divider  [s3-d-rule] rectangle 494x2 · fill #e5e1d8
    16.11. text "This app's citation checks ran on Claude Haiku …"  [s3-d-m] text 494x38 · 12px
    16.12. text "The key is stored on this machine only, in your…"  [s3-d-k] text 494x18 · 12px
    16.13. text "Each cloud provider here has the same form: its…"  [s3-d-c] text 494x38 · 12px
### Screen 4 "Provider & model: a server of your own"  [s4] rectangle 1160x820
  1. header  [s4-hdr] rectangle 1160x72 · fill #f8f7f3
    1.1. heading "Provider & model: a server of your own"  [s4-title] text 800x34 · 26px
    1.2. button "Close"  [s4-close] rectangle 88x44 · fill #ffffff
  2. sidebar  [s4-rail] rectangle 220x748 · fill #f8f7f3
    2.1. button "Getting started"  [s4-rail0] rectangle 188x44 · fill #ffffff
    2.2. button "Documents"  [s4-rail1] rectangle 188x44 · fill #ffffff
    2.3. button "Provider & model"  [s4-rail2] rectangle 188x44 · fill #4a3fa6
    2.4. button "Retrieval"  [s4-rail3] rectangle 188x44 · fill #ffffff
    2.5. button "General"  [s4-rail4] rectangle 188x44 · fill #ffffff
  3. card  [s4-now] rectangle 876x92 · fill #ffffff
    3.1. text "Answering with"  [s4-now-l] text 300x16 · 12px
    3.2. text "Claude Haiku 4.5"  [s4-now-m] text 600x22 · 16px
    3.3. text "Ready"  [s4-now-s] rectangle 102x28 · fill #dcefe1
    3.4. text "Your questions and the passages found for them …"  [s4-now-w] text 740x18 · 13px
  4. text "On this computer"  [s4-g1] text 320x16 · 12px
  5. list-item  [s4-p-ollama] rectangle 330x42 · fill #ffffff
    5.1. text "Ollama"  [s4-p-ollama-n] text 180x20 · 14px
    5.2. text "Ready"  [s4-p-ollama-s] rectangle 112x26 · fill #dcefe1
  6. list-item  [s4-p-lmstudio] rectangle 330x42 · fill #ffffff
    6.1. text "LM Studio"  [s4-p-lmstudio-n] text 180x20 · 14px
    6.2. text "Not running"  [s4-p-lmstudio-s] rectangle 112x26 · fill #f8f7f3
  7. text "Cloud, with your own key"  [s4-g2] text 320x16 · 12px
  8. list-item  [s4-p-claude] rectangle 330x42 · fill #ffffff
    8.1. text "Claude"  [s4-p-claude-n] text 180x20 · 14px
    8.2. text "Ready"  [s4-p-claude-s] rectangle 112x26 · fill #dcefe1
  9. list-item  [s4-p-openai] rectangle 330x42 · fill #ffffff
    9.1. text "OpenAI"  [s4-p-openai-n] text 180x20 · 14px
    9.2. text "Needs a key"  [s4-p-openai-s] rectangle 112x26 · fill #f8f7f3
  10. list-item  [s4-p-gemini] rectangle 330x42 · fill #ffffff
    10.1. text "Google Gemini"  [s4-p-gemini-n] text 180x20 · 14px
    10.2. text "Needs a key"  [s4-p-gemini-s] rectangle 112x26 · fill #f8f7f3
  11. list-item  [s4-p-mistral] rectangle 330x42 · fill #ffffff
    11.1. text "Mistral"  [s4-p-mistral-n] text 180x20 · 14px
    11.2. text "Needs a key"  [s4-p-mistral-s] rectangle 112x26 · fill #f8f7f3
  12. list-item  [s4-p-groq] rectangle 330x42 · fill #ffffff
    12.1. text "Groq"  [s4-p-groq-n] text 180x20 · 14px
    12.2. text "Needs a key"  [s4-p-groq-s] rectangle 112x26 · fill #f8f7f3
  13. list-item  [s4-p-openrouter] rectangle 330x42 · fill #ffffff
    13.1. text "OpenRouter"  [s4-p-openrouter-n] text 180x20 · 14px
    13.2. text "Needs a key"  [s4-p-openrouter-s] rectangle 112x26 · fill #f8f7f3
  14. text "Your own server"  [s4-g3] text 320x16 · 12px
  15. button "Add an OpenAI-compatible server"  [s4-add] rectangle 330x42 · fill #e4e1f5
  16. card  [s4-detail] rectangle 530x500 · fill #ffffff
    16.1. heading "OpenAI-compatible server"  [s4-d-h] text 480x26 · 20px
    16.2. text "Any server that speaks the chat-completions for…"  [s4-d-a] text 494x38 · 13px
    16.3. text "Name"  [s4-name-l] text 160x16 · 12px
    16.4. text "Server address"  [s4-addr-l] text 322x16 · 12px
    16.5. input  [s4-name] rectangle 160x40 · fill #ffffff
      16.5.1. text "Lab server"  [s4-name-v] text 136x18 · 13px
    16.6. input  [s4-addr] rectangle 322x40 · fill #ffffff
      16.6.1. text "http://localhost:1234/v1"  [s4-addr-v] text 298x18 · 13px
    16.7. text "API key"  [s4-key-l] text 340x16 · 12px
    16.8. input  [s4-key] rectangle 340x40 · fill #ffffff
      16.8.1. text "Optional for most local servers"  [s4-key-v] text 316x18 · 13px
    16.9. button "Test connection"  [s4-test] rectangle 142x40 · fill #ffffff
    16.10. card  [s4-result] rectangle 494x70 · fill #dcefe1
      16.10.1. text "Reached it. 3 models found."  [s4-r-1] text 460x18 · 13px
      16.10.2. text "Stays on this computer: this address is on your…"  [s4-r-2] text 460x18 · 13px
    16.11. text "Model"  [s4-model-l] text 340x16 · 12px
    16.12. input  [s4-model] rectangle 340x40 · fill #ffffff
      16.12.1. text "Choose from the list, or type a name"  [s4-model-v] text 316x18 · 13px
    16.13. button "Use for answers"  [s4-use] rectangle 142x40 · fill #4a3fa6
    16.14. text "An address outside this computer receives your …"  [s4-warn] text 494x38 · 12px
### Navigation (3)
  button "Provider & model →" [link-model1] → screen "Provider & model" [s3]   (arrow [f-open])
  button "Provider & model →" [link-model2] → screen "Provider & model" [s3]   (arrow [f-fix])
  button "Add an OpenAI-compatible server" [add-server] → screen "Provider & model: a server of y…" [s4]   (arrow [f-add])
```
