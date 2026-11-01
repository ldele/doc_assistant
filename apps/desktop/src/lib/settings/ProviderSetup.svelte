<script lang="ts">
  // Settings > Provider & model: which model answers and where the text goes, then the providers
  // grouped by that, and one provider's form at a time. ADR-034, amended 2026-11-01: Getting started
  // holds no form, so the keys and the Ollama probe live here.
  //
  // Everything shown here is computed by the backend (`GET /api/setup` -> doc_assistant.readiness),
  // so this component renders a verdict rather than forming one — the same wording the docs use.
  // The only local logic is presentation (`./setup.ts`, unit-tested under `npm test`).
  import type { ProviderReadiness, SetupState } from '../core/types/setup'
  import { clearAnthropicKey, getSetup, saveAnthropicKey, setLlmProvider } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { shell } from '../shell/shell.svelte'
  import {
    DEFAULT_CLAUDE_MODEL,
    answeringWith,
    keyPrecedenceNote,
    keyRefusal,
    modelForProvider,
    providerGroups,
    providerStatusLabel,
  } from './setup'

  // `onProviderChanged` lets the parent re-pull /api/settings so the status bar and the Retrieval
  // tab agree with the switch made here (one source of truth per surface, refreshed together).
  let { onProviderChanged }: { onProviderChanged: () => void } = $props()

  const CONSOLE_URL = 'https://console.anthropic.com/settings/keys'
  const OLLAMA_URL = 'https://ollama.com/download'

  let setup = $state<SetupState | null>(null)
  let loadError = $state<string | null>(null)
  let busy = $state(false)
  // True while the Ollama probe is out. The panel draws from local state first (`probe=false`)
  // and fills in Ollama's verdict after: with nothing listening, the probe takes seconds, and the
  // whole panel used to read "Checking…" until it came back.
  let probing = $state(false)
  // The provider whose form is open: the one in use, until the user picks another from the list.
  let picked = $state<string | null>(null)
  // One result line per card, drawn under that card's own buttons: a key's verdict used to land
  // at the foot of the panel, under the Ollama card, where nobody saw it.
  type Card = 'claude' | 'ollama'
  type Kind = 'ok' | 'warn' | 'err'
  let notes = $state<Record<Card, { kind: Kind; text: string } | null>>({ claude: null, ollama: null })
  let keyInput = $state('')
  let claudeModel = $state('')
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
  const open = $derived(picked ?? setup?.active_provider ?? null)
  const now = $derived(setup ? answeringWith(setup, probing) : null)
  const claudeInUse = $derived(
    setup?.active_provider === 'anthropic' && setup.active_ready && setup.active_model === claudeModel.trim(),
  )

  void refresh()

  async function refresh(probe = true): Promise<void> {
    try {
      if (probe && setup === null) {
        // Local state first, in well under a second; the probe's verdict replaces it below.
        show(await getSetup(false))
      }
      probing = probe
      const state = await getSetup(probe)
      show(state)
      // A Re-check that finds Ollama back has to reach the rail badge and the chat banner too.
      if (probe) shell.setup = state
    } catch (e) {
      loadError = message(e)
    } finally {
      probing = false
    }
  }

  function show(state: SetupState): void {
    setup = state
    loadError = null
    const o = state.providers.find((p) => p.id === 'ollama')
    // Seed the model picker with a model Ollama can serve, and re-seed when the pick is not
    // installed (or is a Claude name carried over from a switch) — offering a model the server
    // does not have would fail on the next question.
    const current = ollamaModel || (state.active_provider === 'ollama' ? state.active_model : '')
    if (o) ollamaModel = modelForProvider('ollama', current, [...o.models])
    // Only an empty box is filled: this field is free text, so a refresh must not undo typing.
    if (!claudeModel) {
      claudeModel = modelForProvider('anthropic', state.active_provider === 'anthropic' ? state.active_model : '')
    }
  }

  const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))

  function say(card: Card, kind: Kind, text: string): void {
    notes[card] = { kind, text }
  }

  async function saveKey(): Promise<void> {
    const key = keyInput.trim()
    if (!key || busy) return
    busy = true
    notes.claude = null
    try {
      const result = await saveAnthropicKey(key)
      show(result.setup)
      savedHere = true
      keyInput = '' // never leave key material sitting in a form field
      if (result.verification === 'ok') {
        say('claude', 'ok', 'Key saved and verified.')
      } else {
        // Stored but unverifiable (offline / proxy). Saying "saved" alone would over-promise.
        say('claude', 'warn', `Key saved, but it could not be checked: ${result.detail}`)
      }
      onProviderChanged()
    } catch (e) {
      say('claude', 'err', keyRefusal(message(e)))
    } finally {
      busy = false
    }
  }

  async function removeKey(): Promise<void> {
    if (busy) return
    busy = true
    notes.claude = null
    try {
      show(await clearAnthropicKey())
      savedHere = false
      say('claude', 'ok', 'Key removed from this app.')
      onProviderChanged()
    } catch (e) {
      say('claude', 'err', message(e))
    } finally {
      busy = false
    }
  }

  async function use(provider: string, model: string): Promise<void> {
    if (busy || !model.trim()) return
    const card: Card = provider === 'anthropic' ? 'claude' : 'ollama'
    busy = true
    notes[card] = null
    try {
      await setLlmProvider(provider, model.trim())
      say(card, 'ok', `Answers now come from ${provider} / ${model.trim()}.`)
      onProviderChanged()
      // The card follows at once; with the probe alone it read the old model for seconds.
      probing = true
      show(await getSetup(false))
      await refresh()
    } catch (e) {
      say(card, 'err', message(e))
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
      say(label === 'console' ? 'claude' : 'ollama', 'warn', 'Could not reach the clipboard — the text is selected, press Ctrl+C.')
    }
  }

  function pillClass(p: ProviderReadiness): string {
    return p.ready ? 'ok' : 'todo'
  }
</script>

<section class="setup">
  {#if loadError}
    <p class="err" role="alert">Couldn't read the setup state: {loadError}</p>
  {:else if !setup || !now}
    <p class="muted">Checking…</p>
  {:else}
    <div class="card now">
      <div class="now-head">
        <div class="now-what">
          <span class="eyebrow">Answering with</span>
          {#if now.model}<strong class="model">{now.model}</strong>{/if}
        </div>
        <span class="pill {now.tone}">{now.status}</span>
      </div>
      <p class="detail">{now.line}</p>
      {#if now.action}<p class="detail fix">{now.action}</p>{/if}
    </div>

    <div class="split">
      <!-- Grouped by where the text goes. A new provider takes its row from the payload, its group
           included, so this list does not change when one is added. -->
      <div class="plist" role="group" aria-label="Providers">
        {#each providerGroups(setup.providers) as group (group.id)}
          <h4>{group.label}</h4>
          {#each group.providers as p (p.id)}
            <button
              type="button"
              class="prow"
              class:on={open === p.id}
              aria-current={open === p.id ? 'true' : undefined}
              onclick={() => (picked = p.id)}
            >
              <span class="pname">{p.name}</span>
              <span class="pill {pillClass(p)}">{providerStatusLabel(p, probing)}</span>
            </button>
          {/each}
        {/each}
      </div>

      {#if open === 'ollama'}
        <!-- ---------------------------------------------------------------- Ollama (local) -->
        <div class="card form">
          <h4 class="form-title">{ollama?.name}</h4>
          <p class="detail">
            No key and no metering. Needs about 5 GB of disk for an 8B model, and is happiest with a GPU.
          </p>
          <p class="detail">{ollama?.destination}</p>
          <!-- The trade-off, stated where the choice is made (KI-36). Local models cite far less than a
               hosted one on the SAME prompt and retrieval, so a local user meets a wall of "uncited"
               claims with no explanation unless we give one here. No figures: the measurement ran on
               the private library, and a release says nothing about it. It also predates KI-66 (Ollama
               cut long prompts to their end, instructions to cite included), so the sentence says so
               until the re-measure KI-36 asks for. -->
          <p class="detail">
            <strong>Expect fewer citations.</strong> On the same prompt and retrieval, local models
            carry fewer inline citations than Claude Haiku 4.5. That was measured before 0.7.0, which
            stopped long prompts losing their instructions to cite, so the gap is likely smaller now.
            Answers stay grounded in your documents; more claims will simply show as <em>uncited</em>.
          </p>
          <p class="detail muted">Looking for a server at <code>{setup.ollama_host}</code>.</p>

          {#if ollama?.ready}
            <p class="ok-line"><Icon name="check" size={13} /> {ollama.detail}</p>
            <label for="ollama-model">Model</label>
            <div class="row">
              <input
                id="ollama-model"
                list="ollama-models"
                bind:value={ollamaModel}
                spellcheck="false"
                disabled={busy}
              />
              <!-- A datalist, not a <select>: the installed list is a suggestion, and a model the
                   probe missed must still be typeable (inform-don't-block). -->
              <datalist id="ollama-models">
                {#each ollama.models as m (m)}
                  <option value={m}></option>
                {/each}
              </datalist>
              {#if setup.active_provider !== 'ollama' || setup.active_model !== ollamaModel}
                <button class="primary" onclick={() => use('ollama', ollamaModel)} disabled={busy || !ollamaModel}>
                  Use for answers
                </button>
              {:else}
                <span class="ok-line"><Icon name="check" size={13} /> In use for answers.</span>
              {/if}
            </div>
            <button class="ghost again" onclick={() => refresh()} disabled={busy}>Re-check</button>
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
            <button class="ghost" onclick={() => refresh()} disabled={busy || probing}>Re-check</button>
          {/if}
          <div aria-live="polite">
            {#if notes.ollama}
              <p class={notes.ollama.kind === 'ok' ? 'ok-line' : notes.ollama.kind} role={notes.ollama.kind === 'err' ? 'alert' : undefined}>
                {notes.ollama.text}
              </p>
            {/if}
          </div>
        </div>
      {:else}
        <!-- ---------------------------------------------------------------- Claude API (BYOK) -->
        <div class="card form">
          <h4 class="form-title">{anthropic?.name}</h4>
          <p class="detail">Your own key, billed by Anthropic.</p>
          <p class="detail">{anthropic?.destination} Your documents stay here.</p>

          {#if anthropic?.configured}
            <p class="ok-line">
              <Icon name="check" size={13} /> Key <code>{anthropic.key_hint}</code>
              {anthropic.key_source === 'env' ? 'from your .env file' : 'saved in this app'}.
            </p>
            {#if keyPrecedenceNote(anthropic, savedHere)}
              <p class="warn">{keyPrecedenceNote(anthropic, savedHere)}</p>
            {/if}
            {#if anthropic.key_source === 'app'}
              <button class="ghost again" onclick={removeKey} disabled={busy}>Remove key</button>
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
              <li>Paste it here.</li>
            </ol>
          {/if}

          <label for="anthropic-key">{anthropic?.configured ? 'Replace the key' : 'API key'}</label>
          <div class="row">
            <input
              id="anthropic-key"
              type="password"
              bind:value={keyInput}
              placeholder="sk-ant-..."
              spellcheck="false"
              autocomplete="off"
              disabled={busy}
            />
            <button class="ghost" onclick={saveKey} disabled={busy || keyInput.trim() === ''}>
              {busy ? 'Checking with Anthropic…' : 'Save key'}
            </button>
          </div>

          <!-- Free text on purpose: any model name, no list to keep current. A name Claude cannot
               serve is refused with its reason on Use for answers. -->
          <label for="claude-model">Model</label>
          <div class="row">
            <input
              id="claude-model"
              bind:value={claudeModel}
              placeholder={DEFAULT_CLAUDE_MODEL}
              spellcheck="false"
              disabled={busy}
            />
            {#if claudeInUse}
              <span class="ok-line"><Icon name="check" size={13} /> In use for answers.</span>
            {:else}
              <button
                class="primary"
                onclick={() => use('anthropic', claudeModel)}
                disabled={busy || !anthropic?.configured || !claudeModel.trim()}
              >
                Use for answers
              </button>
            {/if}
          </div>
          <div aria-live="polite">
            {#if notes.claude}
              <p class={notes.claude.kind === 'ok' ? 'ok-line' : notes.claude.kind} role={notes.claude.kind === 'err' ? 'alert' : undefined}>
                {notes.claude.text}
              </p>
            {/if}
          </div>

          <hr />
          <p class="fine">
            This app's citation checks ran on Claude Haiku and two local models. Other models work;
            their citations are not measured.
          </p>
          <p class="fine">The key is stored on this machine only, in your data folder.</p>
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  .setup {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.75rem;
    background: var(--surface);
  }
  .now-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .now-what {
    min-width: 0;
  }
  .eyebrow {
    display: block;
    font-size: 0.72rem;
    color: var(--fg-2);
  }
  .model {
    display: block;
    font-size: 0.95rem;
    overflow-wrap: anywhere;
  }
  .now .detail {
    margin: 0.3rem 0 0;
    overflow-wrap: anywhere;
  }
  .now .fix {
    color: var(--fg);
  }
  /* Two columns while both fit, one when they do not: the list keeps its width and the form takes
     what is left, so the drawer's own width decides and no breakpoint has to guess it. */
  .split {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 0.8rem;
    margin-top: 0.8rem;
  }
  .plist {
    flex: 1 1 10.5rem;
    min-width: 0;
  }
  .form {
    flex: 999 1 19rem;
    min-width: 0;
  }
  h4 {
    margin: 0.7rem 0 0.3rem;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--fg-2);
  }
  h4:first-child {
    margin-top: 0;
  }
  .form-title {
    margin: 0 0 0.35rem;
    font-size: 1rem;
    color: var(--fg);
  }
  .prow {
    width: 100%;
    font: inherit;
    font-size: 0.84rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.4rem 0.55rem;
    margin-bottom: 0.3rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--fg);
  }
  .prow:hover {
    border-color: var(--accent);
  }
  .prow.on {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 14%, var(--bg));
    font-weight: 600;
  }
  .pname {
    min-width: 0;
  }
  .pill {
    flex: none;
    font-size: 0.72rem;
    font-weight: 400;
    white-space: nowrap;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    border: 1px solid var(--border);
    color: var(--fg-2);
  }
  .pill.ok {
    color: var(--ok-fg);
    border-color: var(--ok-fg);
  }
  .pill.down {
    color: var(--danger);
    border-color: var(--danger);
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
    flex: 1 1 10rem;
    min-width: 0;
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
  }
  .primary {
    flex: none;
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
    flex: none;
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
  .ghost.again {
    margin-top: 0.5rem;
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
    flex-wrap: wrap;
  }
  .row .ok-line {
    margin: 0;
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
  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.7rem 0 0.5rem;
  }
  .fine {
    margin: 0.25rem 0 0;
    font-size: 0.74rem;
    color: var(--fg-2);
  }
</style>
