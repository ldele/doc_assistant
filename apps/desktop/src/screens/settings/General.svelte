<script lang="ts">
  // Settings > General: *Display* (the theme), *Chat history* (every conversation exported to one
  // markdown file) and *Updates* (ADR-044: notification only, nothing is ever downloaded or
  // installed). Owns the theme and the requests behind its buttons. Settings loads the update
  // status with the window and holds the export's and the check's state, which it keeps while
  // another section shows.
  import type { UpdateStatus } from '../../lib/core/types'
  import { checkForUpdate, exportAllConversations, setAutoUpdateCheck } from '../../lib/core/api'
  import { describeUpdate } from '../../lib/settings/updates'
  import { getTheme, setTheme, applyTheme, type Theme } from '../../lib/core/theme'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    update = $bindable(),
    updateBusy = $bindable(),
    updateError = $bindable(),
    exporting = $bindable(),
    exportError = $bindable(),
  }: {
    /** The update status, or null until the first read. */
    update: UpdateStatus | null
    updateBusy: boolean
    updateError: string | null
    exporting: boolean
    exportError: string | null
  } = $props()

  let theme = $state<Theme>(getTheme())

  function onThemeChange(t: Theme): void {
    theme = t
    setTheme(t)
    applyTheme(t)
  }

  // Chat-history export. A failure is shown, not swallowed: the whole point of the button is to
  // be trusted before a delete, so a silent no-op would be the worst outcome it could have.
  async function exportHistory(): Promise<void> {
    exporting = true
    exportError = null
    try {
      await exportAllConversations()
    } catch (e) {
      exportError = `Export failed: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      exporting = false
    }
  }

  // ADR-044 — the update check. Notification only; nothing here can install anything. A failed
  // *check* is not an error to show in red — it comes back as state 'unknown' with a reason, and
  // only a transport failure lands in `updateError`.
  async function runUpdateCheck(): Promise<void> {
    if (updateBusy) return
    updateBusy = true
    updateError = null
    try {
      update = await checkForUpdate()
    } catch (e) {
      updateError = `Update check failed: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      updateBusy = false
    }
  }

  async function applyAutoCheck(enabled: boolean): Promise<void> {
    if (updateBusy) return
    updateBusy = true
    updateError = null
    try {
      update = await setAutoUpdateCheck(enabled)
      // Turning it on should answer the question it implies, not wait a day to do it.
      if (enabled) update = await checkForUpdate()
    } catch (e) {
      updateError = `Couldn't save that: ${e instanceof Error ? e.message : String(e)}`
    } finally {
      updateBusy = false
    }
  }
</script>

<section>
  <h3>Display</h3>
  <div class="segmented" role="radiogroup" aria-label="Theme">
    <button
      type="button"
      role="radio"
      aria-checked={theme === 'system'}
      class:active={theme === 'system'}
      onclick={() => onThemeChange('system')}
      use:tooltip={'Follow this computer’s light or dark setting'}
    >
      System
    </button>
    <button
      type="button"
      role="radio"
      aria-checked={theme === 'light'}
      class:active={theme === 'light'}
      onclick={() => onThemeChange('light')}
      use:tooltip={'Always use the light theme'}
    >
      Light
    </button>
    <button
      type="button"
      role="radio"
      aria-checked={theme === 'dark'}
      class:active={theme === 'dark'}
      onclick={() => onThemeChange('dark')}
      use:tooltip={'Always use the dark theme'}
    >
      Dark
    </button>
  </div>
</section>

<!-- Chat history (user request 2026-08-10). Export lives here rather than next to the
     sidebar's delete because it is a whole-history action, and because the order matters:
     the file is the copy you can act on, the soft delete is not. -->
<section>
  <h3>Chat history</h3>
  <p class="hint">
    Every conversation in one markdown file — uncapped, unlike the sidebar list. Delete chats
    from the sidebar’s select mode (✓).
  </p>
  <button class="btn below" onclick={exportHistory} disabled={exporting} type="button" use:tooltip={'Save every conversation to one file'}>
    {exporting ? 'Exporting…' : 'Export all conversations'}
  </button>
  {#if exportError}<p class="err">{exportError}</p>{/if}
</section>

<!-- Updates (ADR-044). Notification only: this app never downloads or installs anything.
     The toggle governs the *automatic* daily check; "Check now" always runs, because an
     explicit press is its own consent and gating it would leave a user who declined
     background traffic with no way to find out whether they're current. -->
<section>
  <h3>Updates</h3>
  <p class="hint">
    Provenote can check whether a newer version has been published, and link you to it.
    It never downloads or installs anything — you stay in control of what runs on your
    machine.
  </p>
  <label class="switch-row">
    <input
      type="checkbox"
      checked={update?.auto_check_enabled ?? false}
      disabled={updateBusy}
      onchange={(e) => void applyAutoCheck((e.target as HTMLInputElement).checked)}
    />
    Check for updates automatically <span class="muted">(once a day)</span>
  </label>
  <p class="hint">
    When on, Provenote asks GitHub for the latest release version once a day. Nothing about
    you or your documents is sent — no queries, no titles, no identifier.
  </p>

  <div class="update-status" aria-live="polite">
    {#if update}
      {@const line = describeUpdate(update)}
      <p class:ok={line.tone === 'ok'} class:muted={line.tone === 'muted'}>
        <strong>{line.headline}</strong>
      </p>
      <p class="hint">{line.detail}</p>
      {#if line.showLink}
        <a class="btn btn-primary below" href={update.release_url} target="_blank" rel="noreferrer noopener">
          Open the release page
        </a>
      {/if}
    {:else}
      <p class="muted">Provenote {'—'} version unknown until the first check.</p>
    {/if}
    {#if updateError}
      <p class="err" role="alert">{updateError}</p>
    {/if}
  </div>

  <button class="btn below" onclick={runUpdateCheck} disabled={updateBusy} type="button" use:tooltip={'Look for a newer version now'}>
    {updateBusy ? 'Checking…' : 'Check now'}
  </button>
</section>

<style>
  section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  section:last-child {
    border-bottom: none;
  }
  h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  label {
    display: block;
    font-size: 0.8rem;
    color: var(--fg-2);
    margin-bottom: 0.3rem;
  }
  input {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
  input:disabled {
    opacity: 0.6;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
  }
  /* Layout only: the roles' look is app.css's (ADR-063). */
  .below {
    margin-top: 0.7rem;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .ok {
    color: var(--ok-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  .segmented {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .segmented button {
    font: inherit;
    font-size: 0.82rem;
    cursor: pointer;
    border: none;
    background: var(--surface);
    color: var(--fg);
    padding: 0.35rem 0.8rem;
  }
  .segmented button + button {
    border-left: 1px solid var(--border);
  }
  .segmented button.active {
    background: var(--accent);
    color: var(--accent-fg);
  }
  section label {
    margin-top: 0.7rem;
  }
  section label:first-of-type {
    margin-top: 0;
  }
  .switch-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--fg);
    cursor: pointer;
  }
  .switch-row input {
    width: auto;
    accent-color: var(--accent);
  }
  /* ADR-044 — the update verdict. Boxed so the state reads as a standing fact about this
     install rather than as transient feedback from the button below it. */
  .update-status {
    margin-top: 0.7rem;
    padding: 0.6rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-2, transparent);
  }
  .update-status p {
    margin: 0;
  }
  .update-status p + p {
    margin-top: 0.25rem;
  }
  /* An <a> styled as the primary button: this is a link on purpose — it opens the release page
     in the browser, and must never look like an in-app install action (ADR-044). */
</style>
