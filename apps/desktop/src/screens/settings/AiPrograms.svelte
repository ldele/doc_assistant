<script lang="ts">
  // Settings > AI programs (ADR-066 choice 9, A4): the MCP endpoint's switch, its access token and
  // the setup to paste, the conversations switch, and what outside programs have called. The
  // endpoint is off until the person turns it on here; the token is shown only when asked for,
  // and replacing it cuts off every program holding the old one.
  import { onMount } from 'svelte'
  import type { McpSetup, McpStatus } from '../../lib/core/types'
  import { getMcpSetup, getMcpStatus, replaceMcpToken, setMcpSettings } from '../../lib/core/api'
  import { callLine, toolsLine } from '../../lib/settings/mcp'
  import { tooltip } from '../../lib/shell/tooltip'
  import ConfirmDialog from '../../shell/ConfirmDialog.svelte'

  let status = $state<McpStatus | null>(null)
  let setup = $state<McpSetup | null>(null)
  let busy = $state(false)
  let error = $state<string | null>(null)
  let copied = $state<string | null>(null)
  let confirmReplace = $state(false)

  async function run(action: () => Promise<void>): Promise<void> {
    busy = true
    error = null
    try {
      await action()
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  onMount(() => {
    void run(async () => {
      status = await getMcpStatus()
    })
  })

  function toggle(change: { enabled?: boolean; conversations?: boolean }): void {
    void run(async () => {
      status = await setMcpSettings(change)
      if (change.enabled === false) setup = null
    })
  }

  function showSetup(): void {
    void run(async () => {
      setup = await getMcpSetup()
    })
  }

  function replace(): void {
    confirmReplace = false
    void run(async () => {
      setup = await replaceMcpToken()
      status = await getMcpStatus()
    })
  }

  async function copy(text: string, key: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text)
      copied = key
      setTimeout(() => {
        if (copied === key) copied = null
      }, 1500)
    } catch {
      error = 'Could not reach the clipboard: select the text and press Ctrl+C.'
    }
  }
</script>

<section class="mcp-block">
  <h3>Let AI programs read this library</h3>
  <p class="mcp-hint">
    An AI program on this computer that speaks MCP can then search your documents and read their
    records, through read-only tools. It writes its own answers: Provenote calls no model for it,
    and changes nothing in your library. Provenote must be running for the program to reach it.
  </p>
  {#if status}
    <label class="mcp-switch">
      <input
        type="checkbox"
        checked={status.enabled}
        disabled={busy}
        onchange={(e) => toggle({ enabled: (e.target as HTMLInputElement).checked })}
      />
      On
    </label>
  {:else if !error}
    <p class="muted" role="status">Loading…</p>
  {/if}
  {#if error}<p class="mcp-err" role="alert">{error}</p>{/if}
</section>

{#if status?.enabled}
  <section class="mcp-block">
    <h3>Connect a program</h3>
    <p class="mcp-hint">
      Paste one of these into the program's MCP settings. Both carry the access token, so treat
      them like a password.
    </p>
    {#if setup}
      <h4>A program that connects to an address</h4>
      <div class="mcp-copy">
        <code>{setup.url}</code>
        <button class="btn" type="button" onclick={() => copy(setup?.url ?? '', 'url')} use:tooltip={'Copy the address to the clipboard'}>
          {copied === 'url' ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div class="mcp-copy">
        <code>{setup.header}</code>
        <button class="btn" type="button" onclick={() => copy(setup?.header ?? '', 'header')} use:tooltip={'Copy the header, with the token, to the clipboard'}>
          {copied === 'header' ? 'Copied' : 'Copy'}
        </button>
      </div>
      <h4>A program that starts a command</h4>
      <pre class="mcp-config">{setup.config_json}</pre>
      <button class="btn" type="button" onclick={() => copy(setup?.config_json ?? '', 'config')} use:tooltip={'Copy this entry, with the token, to the clipboard'}>
        {copied === 'config' ? 'Copied' : 'Copy'}
      </button>
    {:else}
      <button class="btn mcp-below" type="button" disabled={busy} onclick={showSetup} use:tooltip={'Show the access token and what to paste into a program'}>
        Show setup
      </button>
    {/if}
    <p class="mcp-hint">
      Token {status.token_hint ?? 'not made yet'}. Replacing it cuts off every program that holds
      the old one until you give it the new setup.
    </p>
    <button class="btn" type="button" disabled={busy} onclick={() => (confirmReplace = true)} use:tooltip={'Make a new token, cutting off every program that holds the old one'}>
      Replace token…
    </button>
  </section>

  <section class="mcp-block">
    <h3>What a program may read</h3>
    <p class="mcp-hint">{toolsLine(status.tools, status.conversations)} Result shapes, version {status.surface_version}.</p>
    <label class="mcp-switch">
      <input
        type="checkbox"
        checked={status.conversations}
        disabled={busy}
        onchange={(e) => toggle({ conversations: (e.target as HTMLInputElement).checked })}
      />
      Also your past conversations
    </label>
    <p class="mcp-hint">
      A program may have tools of its own, such as the web, your files or sending messages. Text
      from your documents reaches it marked as document text, and text written to an AI is
      flagged, but Provenote cannot make another program respect either.
    </p>
  </section>

  <section class="mcp-block">
    <h3>Recent calls</h3>
    {#if status.recent_calls.length === 0}
      <p class="muted">No program has called yet. Only the tool and the time are kept, never what was asked.</p>
    {:else}
      <p class="mcp-hint">The latest calls since the engine started. Only the tool and the time are kept, never what was asked.</p>
      <ul class="mcp-calls">
        {#each status.recent_calls as call, i (i)}
          <li>{callLine(call)}</li>
        {/each}
      </ul>
      <button class="btn" type="button" disabled={busy} onclick={() => void run(async () => { status = await getMcpStatus() })} use:tooltip={'Read the latest calls again'}>
        Refresh
      </button>
    {/if}
  </section>
{/if}

{#if confirmReplace}
  <ConfirmDialog
    title="Replace the access token?"
    body="Every program that holds the current token loses access until you give it the new setup."
    confirmLabel="Replace"
    tone="default"
    onConfirm={replace}
    onClose={() => (confirmReplace = false)}
  />
{/if}

<style>
  .mcp-block {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  .mcp-block:last-of-type {
    border-bottom: none;
  }
  .mcp-block h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  .mcp-hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0.5rem;
    line-height: 1.45;
  }
  .mcp-below {
    margin-top: 0.4rem;
  }
  .mcp-err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
    line-height: 1.4;
  }
  .mcp-switch {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--fg);
    cursor: pointer;
    margin: 0.2rem 0;
  }
  .mcp-switch input {
    accent-color: var(--accent);
  }
  .mcp-block h4 {
    margin: 0.6rem 0 0.3rem;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
  .mcp-copy {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.35rem;
    min-width: 0;
  }
  .mcp-copy code {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: 0.78rem;
  }
  .mcp-config {
    margin: 0 0 0.4rem;
    padding: 0.5rem 0.6rem;
    max-height: 14rem;
    overflow: auto;
    font-size: 0.75rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .mcp-calls {
    margin: 0 0 0.5rem;
    padding-left: 1.1rem;
    font-size: 0.8rem;
    font-family: var(--font-mono, monospace);
  }
</style>
