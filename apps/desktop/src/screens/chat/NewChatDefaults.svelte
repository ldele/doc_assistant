<script lang="ts">
  // *New chat defaults* (A8a, W2, `[s4]`): the mode and folders every New chat starts with, opened
  // from the ⚙ beside *New chat*. Separate from Settings. Kept in the library's own settings, so
  // each library has its own. *Make this my default* on the start screen writes the same place.
  import Dialog from '../../shell/Dialog.svelte'
  import FolderTicks from './FolderTicks.svelte'
  import ModeChoice from './ModeChoice.svelte'
  import type { ChatMode, LibraryFolder, NewChatDefaults } from '../../lib/core/types'
  import { RESEARCH_ASSISTANT, type FolderChoice } from '../../lib/chat/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    modes,
    folders,
    libraryCount,
    defaults,
    onSave,
    onClose,
  }: {
    modes: ChatMode[]
    folders: LibraryFolder[]
    libraryCount: number
    defaults: NewChatDefaults | null
    onSave: (modeId: string, folders: FolderChoice) => Promise<boolean>
    onClose: () => void
  } = $props()

  // Edited here, saved on Save: the start screen of an open New chat does not move under it.
  // svelte-ignore state_referenced_locally
  let modeId = $state(defaults?.mode_id ?? RESEARCH_ASSISTANT)
  // svelte-ignore state_referenced_locally
  let choice = $state<FolderChoice>(defaults?.folder_ids ?? null)
  let saving = $state(false)
  let failed = $state(false)

  async function save(): Promise<void> {
    saving = true
    failed = false
    const ok = await onSave(modeId, choice)
    saving = false
    if (ok) onClose()
    else failed = true
  }
</script>

<Dialog labelledby="newchat-defaults-title" width="28rem" {onClose}>
  <h2 class="dialog-title" id="newchat-defaults-title">New chat defaults</h2>
  <p class="setup-sub muted">Every New chat starts with these. Separate from Settings.</p>
  {#if defaults && defaults.missing_folders.length > 0}
    <p class="setup-sub setup-warn">
      Left out, since they no longer exist: {defaults.missing_folders.join(', ')}.
    </p>
  {/if}
  <ModeChoice {modes} bind:modeId name="defaults-mode" />
  <div class="setup-sep"></div>
  <p class="setup-head muted">Search in</p>
  <FolderTicks {folders} {libraryCount} bind:choice name="defaults" />
  {#if failed}<p class="setup-sub setup-warn" role="alert">The defaults could not be saved. Try again.</p>{/if}
  <div class="dialog-actions">
    <button class="btn" onclick={onClose} type="button" use:tooltip={'Close without changing the defaults'}>Cancel</button>
    <button class="btn btn-primary" onclick={save} disabled={saving} type="button" use:tooltip={'Start every New chat with this mode and these folders'}>Save</button>
  </div>
</Dialog>
