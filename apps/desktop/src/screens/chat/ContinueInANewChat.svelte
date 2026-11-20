<script lang="ts">
  // *Continue in a new chat* (A8a, `[s5]`): the same choices as the start screen, prefilled with
  // the current chat's (or with the mode `/mode <name>` asked for, G2), and *Carry this chat's
  // exchanges*. The whole chat is carried and fitted to the model's window like any earlier
  // message, oldest left out first, never summarised (G3). This chat stays as it is.
  import Dialog from '../../shell/Dialog.svelte'
  import FolderTicks from './FolderTicks.svelte'
  import ModeChoice from './ModeChoice.svelte'
  import type { ChatMode, LibraryFolder } from '../../lib/core/types'
  import type { FolderChoice } from '../../lib/chat/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    modes,
    folders,
    libraryCount,
    modeId = $bindable(),
    choice = $bindable(),
    carry = $bindable(),
    busy,
    error,
    onContinue,
    onClose,
  }: {
    modes: ChatMode[]
    folders: LibraryFolder[]
    libraryCount: number
    modeId: string
    choice: FolderChoice
    carry: boolean
    busy: boolean
    error: string
    onContinue: () => void
    onClose: () => void
  } = $props()
</script>

<Dialog labelledby="continue-new-title" width="32rem" {onClose}>
  <h2 class="dialog-title" id="continue-new-title">Continue in a new chat</h2>
  <p class="setup-sub muted">This chat stays as it is. The new one starts with the choices below.</p>
  <ModeChoice {modes} bind:modeId name="continue-mode" />
  <div class="setup-sep"></div>
  <p class="setup-head muted">Search in</p>
  <FolderTicks {folders} {libraryCount} bind:choice name="continue" />
  <div class="setup-sep"></div>
  <label class="carry">
    <input type="checkbox" bind:checked={carry} />
    <span>Carry this chat's exchanges</span>
  </label>
  <p class="setup-sub muted">
    Fitted to the model's window like any earlier message: when it cannot hold them all, the
    oldest are left out, and each answer says how many it carried. Never summarised.
  </p>
  {#if error}<p class="setup-sub setup-warn" role="alert">{error}</p>{/if}
  <div class="dialog-actions">
    <button class="btn" onclick={onClose} type="button" use:tooltip={'Stay in this chat'}>Cancel</button>
    <button class="btn btn-primary" onclick={onContinue} disabled={busy} type="button" use:tooltip={'Open a new chat with these choices; this chat is not changed'}>Continue</button>
  </div>
</Dialog>

<style>
  .carry {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-sm);
    cursor: pointer;
    margin-bottom: var(--space-1);
  }
  .carry input {
    accent-color: var(--accent);
    width: 1rem;
    height: 1rem;
    margin: 0;
  }
</style>
