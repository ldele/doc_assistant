<script lang="ts">
  // The start screen New chat opens (A8a, `[s2]`, `[s3]`): the mode chips, *Search in* with the
  // folder list, sample questions that follow the mode, and *Make this my default*. The first
  // question starts the chat with this choice, and the chat keeps it: changing it later means
  // *Continue in a new chat* (W1). Opens preselected from the New chat defaults (W2).
  import Icon from '../../shell/Icon.svelte'
  import FolderTicks from './FolderTicks.svelte'
  import type { ChatMode, LibraryFolder } from '../../lib/core/types'
  import { previewScope } from '../../lib/core/api'
  import { sampleQuestionsFor, searchInLabel, type FolderChoice } from '../../lib/chat/setup'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    modes,
    folders,
    libraryCount,
    modeId = $bindable(),
    choice = $bindable(),
    notice,
    savedNote,
    onUseSample,
    onMakeDefault,
  }: {
    modes: ChatMode[]
    folders: LibraryFolder[]
    libraryCount: number
    modeId: string
    choice: FolderChoice
    /** What the defaults could not keep (a folder or mode that no longer exists), or ''. */
    notice: string
    /** Said once *Make this my default* has saved, or ''. */
    savedNote: string
    onUseSample: (q: string) => void
    onMakeDefault: () => void
  } = $props()

  const mode = $derived(modes.find((m) => m.id === modeId) ?? null)
  const samples = $derived(sampleQuestionsFor(modeId))
  let union = $state<number | null>(null)
  let open = $state(false)

  // *Search in*'s count, asked of the backend whenever the choice changes, open or closed.
  let asked = 0
  $effect(() => {
    const ids = choice === null ? null : [...choice]
    const n = ++asked
    union = null
    if (ids === null) return
    previewScope(ids)
      .then((s) => {
        if (n === asked) union = s?.doc_count ?? 0
      })
      .catch(() => {
        if (n === asked) union = null
      })
  })
  let wrapEl = $state<HTMLElement | null>(null)

  // The list opens below the button inside the scrolling chat pane: bring all of it into view.
  function reveal(node: HTMLElement): void {
    node.scrollIntoView({ block: 'nearest' })
  }

  $effect(() => {
    if (!open) return
    function onDown(e: PointerEvent): void {
      if (wrapEl && !wrapEl.contains(e.target as Node)) open = false
    }
    function onKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') open = false
    }
    window.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  })
</script>

<div class="start">
  <span class="state-mark"><Icon name="book-open-text" size={26} /></span>
  <h2>Ask your library a question</h2>
  <p class="sub">
    Every answer is grounded in your own documents, with inline citations, provenance, and
    per-claim review.
  </p>

  <div class="modechips" role="group" aria-label="Chat mode">
    {#each modes as m (m.id)}
      <button
        class="modechip"
        class:on={m.id === modeId}
        aria-pressed={m.id === modeId}
        onclick={() => (modeId = m.id)}
        type="button"
        use:tooltip={m.built_in ? `Answer as ${m.name} for this whole chat` : `Answer as ${m.name}, your own mode, for this whole chat; its answers are marked not evaluated`}
      >{m.name}</button>
    {/each}
  </div>
  {#if mode && !mode.built_in}
    <p class="note muted">{mode.name} is yours: its answers are marked not evaluated.</p>
  {/if}

  <div class="searchin" bind:this={wrapEl}>
    <span class="label muted">Search in</span>
    <button
      class="scopebtn"
      class:scoped={choice !== null}
      aria-haspopup="dialog"
      aria-expanded={open}
      onclick={() => (open = !open)}
      type="button"
      use:tooltip={'Choose which documents this chat searches: the whole library or some folders'}
    >
      <Icon name="folder" size={13} />
      <span>{searchInLabel(choice, folders, libraryCount, union)}</span>
      <Icon name="chevron-down" size={13} />
    </button>
    {#if open}
      <div class="pop" role="dialog" aria-label="Search in" use:reveal>
        <FolderTicks {folders} {libraryCount} bind:choice {union} preview={false} name="start" />
      </div>
    {/if}
  </div>
  {#if notice}
    <p class="note warn">{notice}</p>
  {/if}

  <div class="samples">
    {#each samples as q (q)}
      <button class="sample" onclick={() => onUseSample(q)} type="button" use:tooltip={'Put this question in the box, ready to edit or send'}>{q}</button>
    {/each}
  </div>

  <button class="btn btn-quiet makedefault" onclick={onMakeDefault} type="button" use:tooltip={'Start every New chat with this mode and these folders; the ⚙ beside New chat changes it later'}>
    Make this my default
  </button>
  {#if savedNote}<p class="note muted" role="status">{savedNote}</p>{/if}
  <p class="foot muted">
    Your first question starts the chat with these; to change them later, continue in a new chat.
  </p>
</div>

<style>
  .start {
    max-width: 560px;
    margin: var(--space-6) auto 0;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
  }
  .state-mark {
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: var(--surface-2);
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: var(--space-2);
  }
  h2 {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--fg);
    margin: 0;
  }
  .sub {
    color: var(--fg-2);
    font-size: var(--text-sm);
    line-height: 1.6;
    max-width: 46ch;
    margin: 0 0 var(--space-2);
  }
  .modechips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--space-2);
  }
  .modechip {
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    color: var(--fg);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: var(--space-1) var(--space-3);
  }
  .modechip:hover {
    border-color: var(--accent);
  }
  /* The chosen mode, in both themes: the accent's tint, never a fixed colour. */
  .modechip.on {
    color: var(--accent);
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, var(--surface));
    font-weight: 600;
  }
  .note {
    margin: 0;
    font-size: 0.74rem;
  }
  .note.warn {
    color: var(--warn-fg);
  }
  .searchin {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .label {
    font-size: 0.78rem;
    white-space: nowrap;
  }
  .scopebtn {
    font: inherit;
    font-size: var(--text-sm);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    min-width: min(15rem, 62vw);
    justify-content: space-between;
    cursor: pointer;
    color: var(--fg);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.6rem;
  }
  .scopebtn.scoped {
    color: var(--accent);
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, var(--surface));
  }
  .pop {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: 30;
    width: 18rem;
    text-align: left;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-2);
    padding: var(--space-3);
  }
  .samples {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin-top: var(--space-3);
    width: min(100%, 22rem);
  }
  /* A suggested question, not an action: it fills the composer and sends nothing. */
  .sample {
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    color: var(--accent);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: var(--space-2) var(--space-3);
  }
  .sample:hover {
    border-color: var(--accent);
  }
  .makedefault {
    margin-top: var(--space-3);
  }
  .foot {
    margin: 0;
    font-size: 0.72rem;
    max-width: 46ch;
  }
</style>
