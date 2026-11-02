<script lang="ts">
  // Settings > Provider & model: which model answers, where its text goes and who bills, then the
  // providers grouped by where the text goes, and one provider's form at a time. ADR-034, amended
  // 2026-11-01: Getting started holds no form, so the keys and the Ollama probe live here. ADR-060:
  // every provider but Claude and Ollama is a preset or a server of your own, on one form.
  //
  // Everything shown here is computed by the backend (`GET /api/setup` -> doc_assistant.readiness),
  // so this component renders a verdict rather than forming one — the same wording the docs use.
  // The only local logic is presentation (`./setup.ts`, unit-tested under `npm test`).
  import type { ProviderReadiness, SetupState } from '../core/types/setup'
  import {
    clearProviderKey,
    fetchProviderModels,
    getSetup,
    removeServer,
    saveProviderKey,
    setLlmProvider,
    setNotBilled,
    testServer,
  } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { shell } from '../shell/shell.svelte'
  import {
    DEFAULT_CLAUDE_MODEL,
    addressChangeNote,
    answeringWith,
    clockTime,
    keyPrecedenceNote,
    keyRefusal,
    modelForProvider,
    providerGroups,
    providerStatusLabel,
    unlistedModel,
  } from './setup'

  // `onProviderChanged` lets the parent re-pull /api/settings so the status bar and the Retrieval
  // tab agree with the switch made here (one source of truth per surface, refreshed together).
  let { onProviderChanged }: { onProviderChanged: () => void } = $props()

  const OLLAMA_URL = 'https://ollama.com/download'
  const NEW_SERVER = 'new-server'

  let setup = $state<SetupState | null>(null)
  let loadError = $state<string | null>(null)
  let busy = $state(false)
  // True while the probe is out. The panel draws from local state first (`probe=false`) and fills
  // in the probe's verdict after: with nothing listening, it takes seconds.
  let probing = $state(false)
  // True while the open form's model list is being fetched from its server.
  let listing = $state(false)
  // Which button's request is out, so only that button says it is working (`busy` disables all).
  let working = $state<'key' | 'connect' | null>(null)
  // The provider whose form is open: the one in use, until the user picks another from the list.
  let picked = $state<string | null>(null)
  // One result line per form, drawn under that form's own buttons.
  type Kind = 'ok' | 'warn' | 'err'
  let notes = $state<Record<string, { kind: Kind; text: string } | null>>({})
  let keyInput = $state('')
  // The model box of each form, by provider id. Free text over the server's list.
  let models = $state<Record<string, string>>({})
  // Set once the user saves a Claude key in this session — the only way to know a .env key is
  // shadowing what they just typed (the backend reports which source wins, not that both exist).
  let savedHere = $state(false)
  let copied = $state<string | null>(null)
  // The server form: the one being added, or the open server's saved name and address.
  let server = $state({ name: '', address: '', key: '' })
  // Targets for the copy fallback: if the clipboard API is unavailable, the text gets selected
  // instead so Ctrl+C still works.
  let copyEls = $state<Record<string, HTMLElement | null>>({})

  const open = $derived(picked ?? setup?.active_provider ?? null)
  const current = $derived(setup?.providers.find((p) => p.id === open) ?? null)
  const now = $derived(setup ? answeringWith(setup, probing) : null)
  const groups = $derived(setup ? providerGroups(setup.providers) : [])

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
      // A Re-check that finds a server back has to reach the rail badge and the chat banner too.
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
    // Seed each model box with a model its provider can serve; only an empty box is filled, so a
    // refresh never undoes typing. Ollama's is re-seeded when its pick is not installed: offering
    // a model the server does not have would fail on the next question.
    for (const p of state.providers) {
      const mine = state.active_provider === p.id ? state.active_model : ''
      if (p.id === 'ollama') {
        models[p.id] = modelForProvider('ollama', models[p.id] || mine, [...p.models])
      } else if (!models[p.id]) {
        models[p.id] = modelForProvider(p.id, mine, [...p.models])
      }
    }
  }

  const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))

  function say(id: string, kind: Kind, text: string): void {
    notes[id] = { kind, text }
  }

  // Opening a form asks its server for its models (ADR-060, choice 8): never per settings read,
  // never in the background. Ollama's list comes with the probe, and a provider with no key yet
  // has nothing to ask with.
  let lastOpened: string | null = null
  $effect(() => {
    const id = open
    if (!id || id === lastOpened || !setup) return
    lastOpened = id
    keyInput = ''
    const p = setup.providers.find((x) => x.id === id)
    if (p?.kind === 'server') server = { name: p.name, address: p.address, key: '' }
    if (id === NEW_SERVER) server = { name: '', address: '', key: '' }
    if (p && p.kind !== 'ollama' && p.configured) void listModels(id)
  })

  async function listModels(id: string): Promise<void> {
    listing = true
    try {
      const result = await fetchProviderModels(id)
      show(await getSetup(false))
      if (result.status !== 'ok' && result.detail) say(id, 'warn', result.detail)
    } catch (e) {
      say(id, 'err', message(e))
    } finally {
      listing = false
    }
  }

  async function saveKey(p: ProviderReadiness): Promise<void> {
    const key = keyInput.trim()
    if (!key || busy) return
    busy = true
    working = 'key'
    notes[p.id] = null
    try {
      const result = await saveProviderKey(p.id, key)
      show(result.setup)
      if (p.kind === 'claude') savedHere = true
      keyInput = '' // never leave key material sitting in a form field
      if (result.verification === 'ok') {
        say(p.id, 'ok', 'Key saved and accepted.')
      } else {
        // Stored but unverifiable (offline / proxy). Saying "saved" alone would over-promise.
        say(p.id, 'warn', `Key saved, but it could not be checked: ${result.detail}`)
      }
      onProviderChanged()
      if (p.kind === 'claude') await listModels(p.id)
    } catch (e) {
      say(p.id, 'err', keyRefusal(p.name, message(e)))
    } finally {
      busy = false
      working = null
    }
  }

  async function removeKey(p: ProviderReadiness): Promise<void> {
    if (busy) return
    busy = true
    notes[p.id] = null
    try {
      show(await clearProviderKey(p.id))
      savedHere = false
      say(p.id, 'ok', 'Key removed from this app.')
      onProviderChanged()
    } catch (e) {
      say(p.id, 'err', message(e))
    } finally {
      busy = false
    }
  }

  async function connect(): Promise<void> {
    if (busy || !server.name.trim() || !server.address.trim()) return
    const id = current?.kind === 'server' ? current.id : null
    const card = id ?? NEW_SERVER
    busy = true
    working = 'connect'
    notes[card] = null
    try {
      const result = await testServer({
        id,
        name: server.name.trim(),
        address: server.address.trim(),
        key: server.key.trim() || null,
      })
      server.key = '' // never leave key material sitting in a form field
      show(result.setup)
      picked = result.id
      lastOpened = result.id
      const found = result.listing
      const count = `${found.models.length} model${found.models.length === 1 ? '' : 's'} found`
      if (found.status === 'ok') say(result.id, 'ok', `Reached it. ${count}.`)
      else say(result.id, 'warn', `Saved. ${found.detail}`)
      models[result.id] = modelForProvider(result.id, models[result.id] ?? '', found.models)
      onProviderChanged()
    } catch (e) {
      say(card, 'err', message(e))
    } finally {
      busy = false
      working = null
    }
  }

  async function forget(p: ProviderReadiness): Promise<void> {
    if (busy) return
    busy = true
    try {
      show(await removeServer(p.id))
      picked = null
      onProviderChanged()
    } catch (e) {
      say(p.id, 'err', message(e))
    } finally {
      busy = false
    }
  }

  async function tick(p: ProviderReadiness, notBilled: boolean): Promise<void> {
    if (busy) return
    busy = true
    try {
      show(await setNotBilled(p.id, notBilled))
      onProviderChanged()
    } catch (e) {
      say(p.id, 'err', message(e))
    } finally {
      busy = false
    }
  }

  async function use(p: ProviderReadiness): Promise<void> {
    const model = (models[p.id] ?? '').trim()
    if (busy || !model) return
    busy = true
    notes[p.id] = null
    try {
      await setLlmProvider(p.id, model)
      say(p.id, 'ok', `Answers now come from ${p.name} / ${model}.`)
      onProviderChanged()
      // The card follows at once; with the probe alone it read the old model for seconds.
      probing = true
      show(await getSetup(false))
      await refresh()
    } catch (e) {
      say(p.id, 'err', message(e))
    } finally {
      busy = false
    }
  }

  const inUse = (p: ProviderReadiness): boolean =>
    setup?.active_provider === p.id && !!setup.active_ready && setup.active_model === (models[p.id] ?? '').trim()

  // Copy without needing a Tauri clipboard capability: try the async API, and if it is
  // unavailable (or refused), select the text so Ctrl+C still works. Inform, don't block.
  // `card: … = null`-style defaults only, never `x?:` — the Svelte TS-strip keeps the `?` and
  // emits `function f(x?)`, a SyntaxError that blanks the whole app mount (apps/desktop/CLAUDE.md).
  async function copy(text: string, label: string, card: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text)
      copied = label
      setTimeout(() => (copied = copied === label ? null : copied), 1600)
    } catch {
      const el = copyEls[label]
      if (el) {
        const range = document.createRange()
        range.selectNodeContents(el)
        const sel = window.getSelection()
        sel?.removeAllRanges()
        sel?.addRange(range)
      }
      say(card, 'warn', 'Could not reach the clipboard — the text is selected, press Ctrl+C.')
    }
  }

  function pillClass(p: ProviderReadiness): string {
    if (p.key_status === 'refused') return 'down'
    return p.ready ? 'ok' : 'todo'
  }
</script>

{#snippet copyRow(text: string, label: string, card: string)}
  <span class="copyrow">
    <code bind:this={copyEls[label]}>{text}</code>
    <button class="ghost sm" onclick={() => copy(text, label, card)}>{copied === label ? 'Copied' : 'Copy'}</button>
  </span>
{/snippet}

{#snippet note(id: string)}
  <div aria-live="polite">
    {#if notes[id]}
      <p class={notes[id]?.kind === 'ok' ? 'ok-line' : notes[id]?.kind} role={notes[id]?.kind === 'err' ? 'alert' : undefined}>
        {notes[id]?.text}
      </p>
    {/if}
  </div>
{/snippet}

{#snippet facts(p: ProviderReadiness)}
  <!-- The two facts, kept apart (ADR-060's amendment): where the text goes, then who bills. -->
  <p class="detail">{p.destination}</p>
  <p class="detail bill">{p.billing}.</p>
  {#if p.can_tick}
    <label class="tick">
      <input
        type="checkbox"
        checked={p.billing_kind === 'your_word'}
        disabled={busy}
        onchange={(e) => tick(p, e.currentTarget.checked)}
      />
      This server does not bill me
    </label>
  {/if}
{/snippet}

{#snippet modelRow(p: ProviderReadiness, placeholder: string, canUse: boolean)}
  <div class="model-head">
    <label for="model-{p.id}">Model</label>
    {#if p.kind !== 'ollama'}
      <button class="link" onclick={() => listModels(p.id)} disabled={busy || listing || !p.configured}>
        {listing ? 'Asking the server…' : 'Refresh'}
      </button>
    {/if}
  </div>
  <div class="row">
    <!-- A datalist, not a <select>: the server's list is a suggestion that filters as you type,
         and a model it does not list must still be typeable (inform-don't-block). -->
    <input
      id="model-{p.id}"
      list="models-{p.id}"
      bind:value={models[p.id]}
      {placeholder}
      spellcheck="false"
      disabled={busy}
    />
    <datalist id="models-{p.id}">
      {#each p.models as m (m)}
        <option value={m}></option>
      {/each}
    </datalist>
    {#if inUse(p)}
      <span class="ok-line"><Icon name="check" size={13} /> In use for answers.</span>
    {:else}
      <button class="primary" onclick={() => use(p)} disabled={busy || !canUse || !(models[p.id] ?? '').trim()}>
        Use for answers
      </button>
    {/if}
  </div>
  {#if p.models_fetched_at && p.models.length}
    <p class="fine">
      {p.models.length} model{p.models.length === 1 ? '' : 's'} listed by the server at {clockTime(p.models_fetched_at)}.
    </p>
  {/if}
  {#if unlistedModel(models[p.id] ?? '', p.models)}
    <p class="fine">{unlistedModel(models[p.id] ?? '', p.models)}</p>
  {/if}
  {#if p.remote[(models[p.id] ?? '').trim()]}
    <p class="warn">
      This model runs on {p.remote[(models[p.id] ?? '').trim()]}, not on this computer, and Ollama bills it.
    </p>
  {/if}
{/snippet}

{#snippet sent(p: ProviderReadiness)}
  <details class="sends">
    <summary>What an answer sends</summary>
    <ul>
      {#each p.sends as item (item)}<li>{item}</li>{/each}
    </ul>
  </details>
{/snippet}

{#snippet measured()}
  <p class="fine">
    This app's citation checks ran on Claude Haiku and two local models. Other models work; their
    citations are not measured.
  </p>
{/snippet}

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
          {#if now.model}<strong class="model">{now.model} <span class="on">on {now.provider}</span></strong>{/if}
        </div>
        <span class="pill {now.tone}">{now.status}</span>
      </div>
      <p class="detail">{now.line}</p>
      {#if now.billing}<p class="detail">{now.billing}.</p>{/if}
      {#if now.action}<p class="detail fix">{now.action}</p>{/if}
    </div>

    <div class="split">
      <!-- Grouped by where the text goes. A new provider takes its row from the payload, its group
           included, so this list does not change when one is added. -->
      <div class="plist" role="group" aria-label="Providers">
        {#each groups as group (group.id)}
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
        {#if !groups.some((g) => g.id === 'server')}<h4>Your own server</h4>{/if}
        <button
          type="button"
          class="prow add"
          class:on={open === NEW_SERVER}
          aria-current={open === NEW_SERVER ? 'true' : undefined}
          onclick={() => (picked = NEW_SERVER)}
        >
          <span class="pname">Add an OpenAI-compatible server</span>
        </button>
      </div>

      {#if open === NEW_SERVER || current?.kind === 'server'}
        <!-- ------------------------------------------------------- A server of your own -->
        {@const p = current?.kind === 'server' ? current : null}
        {@const card = p?.id ?? NEW_SERVER}
        <div class="card form">
          <h4 class="form-title">{p ? p.name : 'OpenAI-compatible server'}</h4>
          <p class="detail">
            Any server that speaks the chat-completions format: LM Studio, llama.cpp, vLLM, or a
            lab machine. Its address, the questions and the passages found for them go to it.
          </p>
          <div class="pair">
            <div class="name">
              <label for="server-name">Name</label>
              <input id="server-name" bind:value={server.name} placeholder="Lab server" disabled={busy} />
            </div>
            <div class="addr">
              <label for="server-address">Server address</label>
              <input
                id="server-address"
                bind:value={server.address}
                placeholder="http://localhost:1234/v1"
                spellcheck="false"
                disabled={busy}
              />
            </div>
          </div>
          {#if p && addressChangeNote(p.address, server.address, p.key_source !== null)}
            <p class="warn">{addressChangeNote(p.address, server.address, p.key_source !== null)}</p>
          {/if}
          <label for="server-key">{p?.key_hint ? `API key (saved: ${p.key_hint})` : 'API key'}</label>
          <div class="row">
            <input
              id="server-key"
              type="password"
              bind:value={server.key}
              placeholder="Optional for most local servers"
              spellcheck="false"
              autocomplete="off"
              disabled={busy}
            />
            <button class="ghost" onclick={connect} disabled={busy || !server.name.trim() || !server.address.trim()}>
              {working === 'connect' ? 'Reaching it…' : 'Test connection'}
            </button>
          </div>
          {@render note(card)}
          {#if p}
            <div class="result" class:bad={!p.ready}>
              <p>{p.detail}</p>
              {@render facts(p)}
            </div>
            {#if p.action}<p class="detail fix">{p.action}</p>{/if}
            {@render modelRow(p, 'Choose from the list, or type a name', p.ready)}
            {#each p.notes as line (line)}<p class="fine">{line}</p>{/each}
            {@render sent(p)}
            <hr />
            {@render measured()}
            {#if p.key_source}
              <button class="ghost again" onclick={() => removeKey(p)} disabled={busy}>Remove key</button>
            {/if}
            <button class="ghost again" onclick={() => forget(p)} disabled={busy}>Remove server</button>
          {:else}
            <p class="fine">
              An address outside this computer receives your key, your questions and the passages
              found for them. Provenote sees only the address you give; a server that forwards
              requests sends your text on too.
            </p>
          {/if}
        </div>
      {:else if current?.kind === 'ollama'}
        <!-- ---------------------------------------------------------------- Ollama (local) -->
        {@const p = current}
        <div class="card form">
          <h4 class="form-title">{p.name}</h4>
          <p class="detail">
            No key. Needs about 5 GB of disk for an 8B model, and is happiest with a GPU.
          </p>
          {@render facts(p)}
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

          {#if p.ready}
            <p class="ok-line"><Icon name="check" size={13} /> {p.detail}</p>
            {@render modelRow(p, 'llama3.1:8b', true)}
            <button class="ghost again" onclick={() => refresh()} disabled={busy}>Re-check</button>
          {:else}
            <ol class="howto">
              {#if p.reachable === false}
                <li>
                  Install Ollama, then start it (it usually starts on its own after installing):
                  {@render copyRow(OLLAMA_URL, 'ollama', p.id)}
                </li>
              {/if}
              <li>
                Download a model:
                {@render copyRow('ollama pull llama3.1:8b', 'pull', p.id)}
              </li>
              <li>Come back and press Re-check.</li>
            </ol>
            <button class="ghost" onclick={() => refresh()} disabled={busy || probing}>Re-check</button>
          {/if}
          {#each p.notes as line (line)}<p class="fine">{line}</p>{/each}
          {@render note(p.id)}
          {@render sent(p)}
        </div>
      {:else if current}
        <!-- ------------------------------------------ Claude, or a preset (cloud or local) -->
        {@const p = current}
        {@const vendor = p.billing_kind === 'vendor' ? p.billing.replace(/^Billed by /, '') : ''}
        <div class="card form">
          <h4 class="form-title">{p.name}</h4>
          <p class="detail">
            {#if p.takes_key}Your own key, billed by {vendor}.{:else}No key and no metering.{/if}
          </p>
          <p class="detail">{p.destination}{p.reach !== 'this_computer' ? ' Your documents stay here.' : ''}</p>
          {#if p.kind === 'preset'}
            <p class="detail muted">Address <code>{p.address}</code> (the preset's; not editable)</p>
          {/if}

          {#if p.key_status === 'refused'}
            <p class="err" role="alert">{p.detail}</p>
          {:else if p.configured && p.takes_key}
            <p class="ok-line">
              <Icon name="check" size={13} /> Key <code>{p.key_hint}</code>
              {p.key_source === 'env' ? 'from your .env file' : 'saved in this app'}.
            </p>
            {#if p.key_status === 'accepted'}<p class="fine">{p.detail}</p>{/if}
            {#if keyPrecedenceNote(p, savedHere)}
              <p class="warn">{keyPrecedenceNote(p, savedHere)}</p>
            {/if}
            {#if p.key_source === 'app'}
              <button class="ghost again" onclick={() => removeKey(p)} disabled={busy}>Remove key</button>
            {/if}
          {:else if p.takes_key && p.key_page}
            <ol class="howto">
              <li>
                Create a key with {vendor}:
                {@render copyRow(p.key_page, `page-${p.id}`, p.id)}
              </li>
              <li>Paste it here.</li>
            </ol>
          {:else if !p.ready}
            <p class="detail fix">{p.detail}{#if p.action} {p.action}{/if}</p>
          {:else}
            <p class="ok-line"><Icon name="check" size={13} /> {p.detail}</p>
          {/if}

          {#if p.takes_key}
            <label for="key-{p.id}">{p.configured ? 'Replace the key' : 'API key'}</label>
            <div class="row">
              <input
                id="key-{p.id}"
                type="password"
                bind:value={keyInput}
                placeholder={p.kind === 'claude' ? 'sk-ant-...' : 'Paste the key'}
                spellcheck="false"
                autocomplete="off"
                disabled={busy}
              />
              <button class="ghost" onclick={() => saveKey(p)} disabled={busy || keyInput.trim() === ''}>
                {working === 'key' ? `Checking with ${vendor}…` : 'Save key'}
              </button>
            </div>
          {/if}

          {@render modelRow(p, p.kind === 'claude' ? DEFAULT_CLAUDE_MODEL : 'Choose from the list, or type a name', p.configured && p.key_status !== 'refused')}
          {@render note(p.id)}
          {#if p.kind === 'preset' && p.reach === 'this_computer'}
            <button class="ghost again" onclick={() => refresh()} disabled={busy || probing}>Re-check</button>
          {/if}
          <p class="detail">{p.billing}{#if p.kind !== 'claude' && p.billing_kind === 'vendor'}; Provenote does not know its price{/if}.</p>
          {#each p.notes as line (line)}<p class="fine">{line}</p>{/each}
          {@render sent(p)}

          <hr />
          {@render measured()}
          {#if p.takes_key}<p class="fine">The key is stored on this machine only, in your data folder, and sent only to this address.</p>{/if}
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
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.25rem 0.5rem;
    text-align: left;
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
    margin-left: auto;
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
  .on {
    font-weight: 400;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
  .prow.add {
    justify-content: flex-start;
    color: var(--fg-2);
    border-style: dashed;
  }
  .model-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .link {
    font: inherit;
    font-size: 0.76rem;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .link:disabled {
    color: var(--fg-2);
    cursor: default;
  }
  .pair {
    display: flex;
    flex-wrap: wrap;
    gap: 0 0.5rem;
  }
  .pair .name {
    flex: 1 1 8rem;
    min-width: 0;
  }
  .pair .addr {
    flex: 3 1 12rem;
    min-width: 0;
  }
  .result {
    margin: 0.6rem 0 0;
    padding: 0.5rem 0.6rem;
    border-radius: 8px;
    border: 1px solid var(--ok-fg);
    background: color-mix(in srgb, var(--ok-fg) 8%, var(--bg));
    font-size: 0.78rem;
  }
  .result.bad {
    border-color: var(--warn-border);
    background: var(--warn-bg);
  }
  .result p {
    margin: 0 0 0.3rem;
    overflow-wrap: anywhere;
  }
  .result p:last-child {
    margin-bottom: 0;
  }
  .tick {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0.3rem 0 0;
    color: var(--fg);
  }
  .tick input {
    width: auto;
    flex: none;
  }
  .sends {
    margin: 0.5rem 0 0;
    font-size: 0.76rem;
    color: var(--fg-2);
  }
  .sends summary {
    cursor: pointer;
  }
  .sends ul {
    margin: 0.3rem 0 0;
    padding-left: 1.1rem;
  }
  .detail.bill {
    color: var(--fg);
  }
</style>
