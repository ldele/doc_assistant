<script lang="ts">
  // First-run setup: the two ways to get answers (a Claude API key, or a local Ollama server),
  // with the state of each and the exact next action. ADR-034.
  //
  // Everything shown here is computed by the backend (`GET /api/setup` -> doc_assistant.readiness),
  // so this component renders a verdict rather than forming one — the same wording the docs use.
  // The only local logic is presentation (`./setup.ts`, unit-tested under `npm test`).
  import type { ProviderReadiness, SetupState } from '../core/types/setup'
  import { clearAnthropicKey, getSetup, saveAnthropicKey, setLlmProvider } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { costLabel, keyPrecedenceNote, providerStatusLabel, setupSummary, suggestOllamaModel } from './setup'

  // `onProviderChanged` lets the parent re-pull /api/settings so the "Active:" line and the status
  // bar agree with the switch made here (one source of truth per surface, refreshed together).
  let { onProviderChanged }: { onProviderChanged: () => void } = $props()

  const CONSOLE_URL = 'https://console.anthropic.com/settings/keys'
  const OLLAMA_URL = 'https://ollama.com/download'

  let setup = $state<SetupState | null>(null)
  let loadError = $state<string | null>(null)
  let busy = $state(false)
  let note = $state<string | null>(null)
  let noteKind = $state<'ok' | 'warn' | 'err'>('ok')
  let keyInput = $state('')
  let ollamaModel = $state('')
  // Set once the user saves a key in this session — the only way to know a .env key is shadowing
  // what they just typed (the backend reports which source wins, not that both exist).
  let savedHere = $state(false)
  let copied = $state<string | null>(null)
  // Targets for the copy fallback: if the clipboard API is unavailable, the text gets selected
  // instead so Ctrl+C still works.
  let consoleEl = $state<HTMLElement | null>(null)
  let ollamaEl = $state<HTMLElement | null>(null)
  let pullEl = $state<HTMLElement | null>(null)

  const anthropic = $derived(setup?.providers.find((p) => p.id === 'anthropic') ?? null)
  const ollama = $derived(setup?.providers.find((p) => p.id === 'ollama') ?? null)

  void refresh()

  async function refresh(probe = true): Promise<void> {
    try {
      const state = await getSetup(probe)
      setup = state
      loadError = null
      const o = state.providers.find((p) => p.id === 'ollama')
      // Seed the model picker once, and re-seed if the current pick is no longer installed —
      // offering a model the server does not have would fail on the next question.
      if (o && (!ollamaModel || (o.models.length > 0 && !o.models.includes(ollamaModel)))) {
        ollamaModel = suggestOllamaModel(o.models) ?? state.active_model
      }
    } catch (e) {
      loadError = String(e)
    }
  }

  function say(kind: 'ok' | 'warn' | 'err', message: string): void {
    noteKind = kind
    note = message
  }

  async function saveKey(): Promise<void> {
    const key = keyInput.trim()
    if (!key || busy) return
    busy = true
    note = null
    try {
      const result = await saveAnthropicKey(key)
      setup = result.setup
      savedHere = true
      keyInput = '' // never leave key material sitting in a form field
      if (result.verification === 'ok') {
        say('ok', 'Key saved and verified. Ask a question whenever you like.')
      } else {
        // Stored but unverifiable (offline / proxy). Saying "saved" alone would over-promise.
        say('warn', `Key saved, but it could not be checked: ${result.detail}`)
      }
      onProviderChanged()
    } catch (e) {
      say('err', String(e))
    } finally {
      busy = false
    }
  }

  async function removeKey(): Promise<void> {
    if (busy) return
    busy = true
    note = null
    try {
      setup = await clearAnthropicKey()
      savedHere = false
      say('ok', 'Key removed from this app.')
      onProviderChanged()
    } catch (e) {
      say('err', String(e))
    } finally {
      busy = false
    }
  }

  async function use(provider: string, model: string): Promise<void> {
    if (busy || !model.trim()) return
    busy = true
    note = null
    try {
      await setLlmProvider(provider, model.trim())
      say('ok', `Answers now come from ${provider} / ${model.trim()}.`)
      onProviderChanged()
      await refresh()
    } catch (e) {
      say('err', String(e))
    } finally {
      busy = false
    }
  }

  // Copy without needing a Tauri clipboard capability: try the async API, and if it is
  // unavailable (or refused), select the text so Ctrl+C still works. Inform, don't block.
  // `el: … = null`, never `el?:` — the Svelte TS-strip keeps the `?` and emits `function f(x?)`,
  // a SyntaxError that blanks the whole app mount while svelte-check passes it
  // (apps/desktop/CLAUDE.md).
  async function copy(text: string, label: string, el: HTMLElement | null = null): Promise<void> {
    try {
      await navigator.clipboard.writeText(text)
      copied = label
      setTimeout(() => (copied = copied === label ? null : copied), 1600)
    } catch {
      const range = document.createRange()
      if (el) {
        range.selectNodeContents(el)
        const sel = window.getSelection()
        sel?.removeAllRanges()
        sel?.addRange(range)
      }
      say('warn', 'Could not reach the clipboard — the text is selected, press Ctrl+C.')
    }
  }

  function pillClass(p: ProviderReadiness | null): string {
    if (!p) return 'unknown'
    return p.ready ? 'ok' : 'todo'
  }
</script>

<section class="setup">
  <h3>
    Getting started
    <span class="muted">({setupSummary(setup)})</span>
  </h3>

  {#if loadError}
    <p class="err" role="alert">Couldn't read the setup state: {loadError}</p>
  {:else if !setup}
    <p class="muted">Checking…</p>
  {:else}
    <!-- The checklist is the honest summary: what is done, what is not, and what fixes it. -->
    <ol class="steps">
      {#each setup.steps as step (step.id)}
        <li class:done={step.done}>
          <span class="tick" aria-hidden="true">
            <Icon name={step.done ? 'check' : 'arrow-right'} size={14} />
          </span>
          <div>
            <strong>{step.title}</strong>
            <p>{step.detail}{#if step.action}<br /><em>{step.action}</em>{/if}</p>
          </div>
        </li>
      {/each}
    </ol>

    <p class="hint">
      Two ways to get answers. Pick either — you can switch any time, and retrieval always runs
      locally on your machine regardless.
    </p>

    <!-- ---------------------------------------------------------------- Claude API (BYOK) -->
    <div class="card">
      <header>
        <strong>Claude API</strong>
        <span class="pill {pillClass(anthropic)}">{anthropic ? providerStatusLabel(anthropic) : '—'}</span>
      </header>
      <p class="detail">
        Best answer quality. Your own key, {anthropic ? costLabel(anthropic) : 'metered'}; the app
        never proxies it anywhere else.
      </p>

      {#if anthropic?.configured}
        <p class="ok-line">
          <Icon name="check" size={13} /> Key <code>{anthropic.key_hint}</code>
          {anthropic.key_source === 'env' ? 'from your .env file' : 'saved in this app'}.
        </p>
        {#if keyPrecedenceNote(anthropic, savedHere)}
          <p class="warn">{keyPrecedenceNote(anthropic, savedHere)}</p>
        {/if}
        {#if anthropic.key_source === 'app'}
          <button class="ghost" onclick={removeKey} disabled={busy}>Remove key</button>
        {/if}
      {:else}
        <ol class="howto">
          <li>
            Create a key in the Anthropic Console:
            <span class="copyrow">
              <code bind:this={consoleEl}>{CONSOLE_URL}</code>
              <button class="ghost sm" onclick={() => copy(CONSOLE_URL, 'console', consoleEl)}>
                {copied === 'console' ? 'Copied' : 'Copy'}
              </button>
            </span>
          </li>
          <li>Paste it here. It is stored on this machine only, in your data folder.</li>
        </ol>
      {/if}

      <label for="anthropic-key">{anthropic?.configured ? 'Replace the key' : 'API key'}</label>
      <input
        id="anthropic-key"
        type="password"
        bind:value={keyInput}
        placeholder="sk-ant-..."
        spellcheck="false"
        autocomplete="off"
        disabled={busy}
      />
      <div class="row">
        <button class="primary" onclick={saveKey} disabled={busy || keyInput.trim() === ''}>
          {busy ? 'Checking…' : 'Save key'}
        </button>
        {#if anthropic?.ready && setup.active_provider !== 'anthropic'}
          <button class="ghost" onclick={() => use('anthropic', 'claude-haiku-4-5-20251001')} disabled={busy}>
            Use Claude for answers
          </button>
        {/if}
      </div>
    </div>

    <!-- ---------------------------------------------------------------- Ollama (local) -->
    <div class="card">
      <header>
        <strong>Ollama <span class="muted">(local)</span></strong>
        <span class="pill {pillClass(ollama)}">{ollama ? providerStatusLabel(ollama) : '—'}</span>
      </header>
      <p class="detail">
        No key, no metering, nothing leaves your machine. Needs about 5 GB of disk for an 8B model,
        and is happiest with a GPU.
      </p>
      <p class="detail muted">Looking for a server at <code>{setup.ollama_host}</code>.</p>

      {#if ollama?.ready}
        <p class="ok-line"><Icon name="check" size={13} /> {ollama.detail}</p>
        <label for="ollama-model">Model</label>
        <input
          id="ollama-model"
          list="ollama-models"
          bind:value={ollamaModel}
          spellcheck="false"
          disabled={busy}
        />
        <!-- A datalist, not a <select>: the installed list is a suggestion, and a model the probe
             missed must still be typeable (inform-don't-block). -->
        <datalist id="ollama-models">
          {#each ollama.models as m (m)}
            <option value={m}></option>
          {/each}
        </datalist>
        <div class="row">
          {#if setup.active_provider !== 'ollama' || setup.active_model !== ollamaModel}
            <button class="primary" onclick={() => use('ollama', ollamaModel)} disabled={busy || !ollamaModel}>
              Use this model
            </button>
          {:else}
            <span class="ok-line"><Icon name="check" size={13} /> In use for answers.</span>
          {/if}
          <button class="ghost" onclick={() => refresh()} disabled={busy}>Re-check</button>
        </div>
      {:else}
        <ol class="howto">
          {#if ollama?.reachable === false}
            <li>
              Install Ollama, then start it (it usually starts on its own after installing):
              <span class="copyrow">
                <code bind:this={ollamaEl}>{OLLAMA_URL}</code>
                <button class="ghost sm" onclick={() => copy(OLLAMA_URL, 'ollama', ollamaEl)}>
                  {copied === 'ollama' ? 'Copied' : 'Copy'}
                </button>
              </span>
            </li>
          {/if}
          <li>
            Download a model:
            <span class="copyrow">
              <code bind:this={pullEl}>ollama pull llama3.1:8b</code>
              <button class="ghost sm" onclick={() => copy('ollama pull llama3.1:8b', 'pull', pullEl)}>
                {copied === 'pull' ? 'Copied' : 'Copy'}
              </button>
            </span>
          </li>
          <li>Come back and press Re-check.</li>
        </ol>
        <button class="ghost" onclick={() => refresh()} disabled={busy}>Re-check</button>
      {/if}
    </div>

    <div aria-live="polite">
      {#if note}
        <p class={noteKind === 'ok' ? 'ok-line' : noteKind === 'warn' ? 'warn' : 'err'} role={noteKind === 'err' ? 'alert' : undefined}>
          {note}
        </p>
      {/if}
    </div>
  {/if}
</section>

<style>
  .setup {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .steps {
    list-style: none;
    margin: 0 0 0.8rem;
    padding: 0;
    display: grid;
    gap: 0.45rem;
  }
  .steps li {
    display: grid;
    grid-template-columns: 1rem 1fr;
    gap: 0.5rem;
    align-items: start;
    font-size: 0.82rem;
  }
  .steps .tick {
    color: var(--fg-2);
    display: inline-flex;
    padding-top: 0.15rem;
  }
  .steps li.done .tick {
    color: var(--ok-fg);
  }
  .steps strong {
    font-size: 0.82rem;
  }
  .steps p {
    margin: 0.1rem 0 0;
    color: var(--fg-2);
  }
  .steps em {
    color: var(--fg);
    font-style: normal;
  }
  .card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.75rem;
    margin-top: 0.7rem;
    background: var(--surface);
  }
  .card header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
  }
  .pill {
    font-size: 0.72rem;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    border: 1px solid var(--border);
    color: var(--fg-2);
  }
  .pill.ok {
    color: var(--ok-fg);
    border-color: var(--ok-fg);
  }
  .pill.todo {
    color: var(--warn-fg);
    border-color: var(--warn-border);
  }
  .detail {
    margin: 0 0 0.35rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .howto {
    margin: 0.4rem 0 0.6rem;
    padding-left: 1.1rem;
    font-size: 0.78rem;
    color: var(--fg-2);
    display: grid;
    gap: 0.35rem;
  }
  .copyrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.2rem;
    flex-wrap: wrap;
  }
  code {
    font-family: ui-monospace, monospace;
    font-size: 0.74rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.1rem 0.3rem;
    word-break: break-all;
  }
  label {
    display: block;
    font-size: 0.8rem;
    color: var(--fg-2);
    margin: 0.6rem 0 0.3rem;
  }
  input {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
  }
  input:disabled {
    opacity: 0.6;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
  }
  .primary {
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
    padding: 0.35rem 0.9rem;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ghost {
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
    padding: 0.3rem 0.7rem;
  }
  .ghost.sm {
    font-size: 0.72rem;
    padding: 0.15rem 0.45rem;
  }
  .ghost:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .ok-line {
    color: var(--ok-fg);
    font-size: 0.78rem;
    margin: 0.4rem 0 0;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.8rem;
    margin: 0.5rem 0 0;
  }
  .warn {
    color: var(--warn-fg);
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    padding: 0.4rem 0.55rem;
    font-size: 0.78rem;
    margin: 0.5rem 0 0;
  }
  .hint {
    font-size: 0.78rem;
    color: var(--fg-2);
    margin: 0.2rem 0 0;
  }
</style>
