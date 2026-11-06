<script lang="ts">
  // ADR-061 choice 7: a library switch whose backend never answered. Shown by App when the
  // start-up gate reaches *stalled* (STALLED_MS) with a switch pending. The shell kept the library
  // open before the switch, so *Go back* restarts the backend on it; nothing about the library
  // that failed is changed. *Keep waiting* hides this; the gate goes on polling either way.
  import { libraryState, markLibrarySwitch, openLibrary } from '../lib/core/tauri'
  import { goBackLabel, libraryName } from '../lib/settings/library'

  let { asked, onDismiss }: { asked: string; onDismiss: () => void } = $props()

  let previous = $state<string | null>(null)
  let known = $state(false)
  let going = $state(false)
  let error = $state<string | null>(null)

  void libraryState().then((s) => {
    previous = s?.previous ?? null
    known = s !== null
  })

  async function goBack(): Promise<void> {
    going = true
    error = null
    try {
      markLibrarySwitch(previous)
      await openLibrary(previous)
      window.location.reload()
    } catch (e) {
      error = String(e)
      going = false
    }
  }
</script>

<div class="card" role="alert">
  <strong>The library did not open</strong>
  <p>
    {libraryName(asked === '' ? null : asked)} has not answered for a minute and a half. A large
    library can take longer on its first start; if it never answers, go back to the library you
    had open.
  </p>
  {#if error}
    <p class="err">{error}</p>
  {/if}
  <div class="actions">
    <button class="btn" type="button" onclick={onDismiss} disabled={going}>Keep waiting</button>
    {#if known}
      <button class="btn btn-primary" type="button" onclick={() => void goBack()} disabled={going}>
        {going ? 'Going back…' : goBackLabel(previous)}
      </button>
    {/if}
  </div>
</div>

<style>
  .card {
    position: fixed;
    z-index: 45;
    left: 50%;
    bottom: 2.5rem;
    transform: translateX(-50%);
    width: min(92vw, 30rem);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    border: 1px solid var(--warn-border);
    background: var(--warn-bg);
    color: var(--warn-fg);
    box-shadow: var(--shadow-2);
  }
  p {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .err {
    color: var(--danger);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-2);
  }
</style>
