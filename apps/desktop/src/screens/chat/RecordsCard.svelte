<script lang="ts">
  // *What it looked up* (A8b, `[s7]`): a turn answered from the library's records (A6) says that
  // no model and no passage were used, and shows its lookup: the name, its arguments, how many rows
  // it returned and the first rows, each opening its document, folder or concept. Read from the
  // record's `records_json`; nothing new is stored.
  import Icon from '../../shell/Icon.svelte'
  import type { RecordsView } from '../../lib/core/types'
  import { RECORDS_TAG, recordsLine } from '../../lib/chat/turnDetails'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    records,
    onOpenRecord,
  }: {
    records: RecordsView
    onOpenRecord?: (kind: 'document' | 'folder' | 'concept', id: string) => void
  } = $props()

  const icon = { document: 'file-text', folder: 'folder', concept: 'tag' } as const
  const opens = { document: 'Open this document in the Library', folder: 'Open this folder in the Library', concept: 'Open this concept on the Graph tab' } as const
</script>

<p class="tag"><Icon name="library" size={13} /> {RECORDS_TAG}</p>
<section class="looked" aria-label="What it looked up">
  <h4>What it looked up</h4>
  <p class="call">{recordsLine(records)}</p>
  {#if records.items.length > 0}
    <ul>
      {#each records.items as item (item.kind + item.id)}
        <li>
          <button class="row" onclick={() => onOpenRecord?.(item.kind, item.id)} type="button" use:tooltip={opens[item.kind]}>
            <Icon name={icon[item.kind]} size={13} />
            <span class="label">{item.label}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
  {#if records.more > 0}
    <p class="more muted">{records.more} more, listed in the answer above</p>
  {/if}
</section>

<style>
  .tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0.5rem 0 0;
    font-size: 0.76rem;
    color: var(--fg-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
  }
  .looked {
    margin-top: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 0.6rem 0.8rem;
    max-width: 40rem;
  }
  h4 {
    margin: 0 0 0.2rem;
    font-size: 0.9rem;
  }
  .call {
    margin: 0 0 0.4rem;
    font-size: 0.8rem;
    color: var(--fg-2);
    font-family: ui-monospace, 'Cascadia Code', Consolas, monospace;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.15rem;
  }
  .row {
    font: inherit;
    font-size: 0.84rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    width: 100%;
    text-align: left;
    cursor: pointer;
    color: var(--fg);
    background: none;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 0.25rem 0.4rem;
  }
  .row:hover {
    border-color: var(--border);
    color: var(--accent);
  }
  .label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .more {
    margin: 0.35rem 0 0;
    font-size: 0.76rem;
  }
</style>
