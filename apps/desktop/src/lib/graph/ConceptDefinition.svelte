<script lang="ts">
  // A concept's definition, chosen from candidates that keep their source (ADR-053, ROADMAP 93).
  // Lives in the Graph tab's concept panel — the user's choice, 2026-09-21: a definition is read
  // where the concept is, so it is written there too.
  //
  // Nothing on this card overwrites anything. A passage from the library, the user's own words
  // and (93b) a model's text sit side by side; "Use this" picks one and the previous one goes back
  // to the suggestions; "Undo" walks back one step. Each passage shows where it came from and why
  // it is graded the way it is — the reasons are the evidence, never a verdict.
  //
  // Self-contained like GapList: it fetches its own data and owns its writes, and every write
  // answers with the refreshed candidates, so the card never patches its own copy.
  // `onOpenPassage` is the one host hook — it opens a passage where it was written.
  import type { ConceptDefinitions, DefinitionCandidate } from '../core/types'
  import {
    addDefinition,
    chooseDefinition,
    dismissDefinition,
    extractDefinitions,
    getDefinitions,
    restoreDefinition,
    undoDefinition,
  } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { displayText, gradeLabel, groupCandidates, sourceLabel } from './definitions'

  let {
    conceptId,
    onOpenPassage,
  }: {
    conceptId: string
    onOpenPassage: (chunkKey: string) => void
  } = $props()

  let view = $state<ConceptDefinitions | null>(null)
  let loading = $state(true)
  let busy = $state(false)
  let looking = $state(false)
  let error = $state<string | null>(null)
  let writing = $state(false)
  let draft = $state('')
  let showDismissed = $state(false)
  let openReasons = $state<string | null>(null)

  const groups = $derived(groupCandidates(view))

  // Reload whenever the selected concept changes; a stale answer for the previous concept is
  // dropped rather than painted over the new one.
  $effect(() => {
    const id = conceptId
    loading = true
    error = null
    writing = false
    draft = ''
    showDismissed = false
    getDefinitions(id)
      .then((v) => {
        if (id === conceptId) view = v
      })
      .catch((e) => {
        if (id === conceptId) error = e instanceof Error ? e.message : String(e)
      })
      .finally(() => {
        if (id === conceptId) loading = false
      })
  })

  async function run(op: () => Promise<ConceptDefinitions>): Promise<void> {
    if (busy) return
    busy = true
    error = null
    try {
      view = await op()
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  async function look(): Promise<void> {
    looking = true
    await run(() => extractDefinitions(conceptId))
    looking = false
  }

  async function save(): Promise<void> {
    const text = draft.trim()
    if (!text) return
    await run(() => addDefinition(conceptId, text))
    if (!error) {
      writing = false
      draft = ''
    }
  }

  function startWriting(from: DefinitionCandidate | null = null): void {
    draft = from ? displayText(from.text) : ''
    writing = true
  }
</script>

<section class="defcard" aria-label="Definition">
  <div class="defhead">
    <h3>Definition</h3>
    <div class="defactions">
      {#if view?.can_undo}
        <button class="ghost sm" onclick={() => run(() => undoDefinition(conceptId))} disabled={busy} type="button" title="Take back the last change">
          <Icon name="rotate-ccw" size={13} /> Undo
        </button>
      {/if}
      {#if !writing}
        <button class="ghost sm" onclick={() => startWriting()} disabled={busy} type="button">
          <Icon name="pencil" size={13} /> Write your own
        </button>
      {/if}
    </div>
  </div>

  {#if loading}
    <p class="muted">Loading…</p>
  {:else}
    {#if error}
      <p class="err">{error}</p>
    {/if}

    {#if groups.chosen}
      {@const c = groups.chosen}
      <blockquote class="chosen">
        <p>{displayText(c.text)}</p>
        <footer>
          <span class="src">{sourceLabel(c)}</span>
          {#if c.chunk_key}
            <button class="linkish" onclick={() => c.chunk_key && onOpenPassage(c.chunk_key)} type="button">Open passage →</button>
          {/if}
          <button class="linkish" onclick={() => run(() => dismissDefinition(conceptId, c.id))} disabled={busy} type="button">Clear</button>
        </footer>
      </blockquote>
    {:else}
      <p class="muted none">No definition chosen yet.</p>
    {/if}

    {#if writing}
      <div class="writer">
        <textarea bind:value={draft} rows="3" placeholder="What does this concept mean, in your library?" aria-label="Your definition"></textarea>
        <div class="writeractions">
          <button class="primary sm" onclick={save} disabled={busy || !draft.trim()} type="button">Save and use</button>
          <button class="ghost sm" onclick={() => (writing = false)} type="button">Cancel</button>
          <span class="muted hint">Saving keeps every other option — nothing is replaced.</span>
        </div>
      </div>
    {/if}

    {#if groups.suggested.length > 0}
      <h4>{groups.chosen ? 'Other options' : 'Options'} ({groups.suggested.length})</h4>
      <ul class="cands">
        {#each groups.suggested as c (c.id)}
          <li class="cand">
            <p class="ctext">{displayText(c.text)}</p>
            <div class="cmeta">
              <button
                class="grade {c.grade}"
                onclick={() => (openReasons = openReasons === c.id ? null : c.id)}
                aria-expanded={openReasons === c.id}
                type="button"
                title="Why this grade"
              >
                {gradeLabel(c.grade)}
              </button>
              <span class="src">{sourceLabel(c)}</span>
              {#if c.chunk_key}
                <button class="linkish" onclick={() => c.chunk_key && onOpenPassage(c.chunk_key)} type="button">Open passage →</button>
              {/if}
            </div>
            {#if openReasons === c.id && c.reasons.length > 0}
              <ul class="reasons">
                {#each c.reasons as r (r)}<li>{r}</li>{/each}
              </ul>
            {/if}
            <div class="cactions">
              <button class="ghost sm" onclick={() => run(() => chooseDefinition(conceptId, c.id))} disabled={busy} type="button">
                <Icon name="check" size={13} /> Use this
              </button>
              <button class="ghost sm" onclick={() => startWriting(c)} disabled={busy} type="button" title="Start your own definition from this wording">
                Edit into my own
              </button>
              <button class="linkish" onclick={() => run(() => dismissDefinition(conceptId, c.id))} disabled={busy} type="button">Dismiss</button>
            </div>
          </li>
        {/each}
      </ul>
    {/if}

    {#if view && !view.extracted}
      <p class="look">
        <button class="ghost sm" onclick={look} disabled={busy} type="button">
          <Icon name="search" size={13} /> {looking ? 'Looking…' : 'Look in my library'}
        </button>
        <span class="muted">Finds sentences that define it, with where each one is. Takes a few seconds.</span>
      </p>
    {:else if view && view.thin}
      <p class="muted thin">
        Your library has little to go on for this concept — no sentence that reads as a definition.
        Writing your own is a good option here.
      </p>
    {/if}

    {#if groups.dismissed.length > 0}
      <button class="linkish small" onclick={() => (showDismissed = !showDismissed)} aria-expanded={showDismissed} type="button">
        {showDismissed ? 'Hide' : 'Show'} dismissed ({groups.dismissed.length})
      </button>
      {#if showDismissed}
        <ul class="cands dismissed">
          {#each groups.dismissed as c (c.id)}
            <li class="cand">
              <p class="ctext">{displayText(c.text)}</p>
              <div class="cactions">
                <span class="src">{sourceLabel(c)}</span>
                <button class="linkish" onclick={() => run(() => restoreDefinition(conceptId, c.id))} disabled={busy} type="button">Restore</button>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  {/if}
</section>

<style>
  .defcard {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: var(--space-3);
    background: var(--surface);
  }
  .defhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--fg-2);
  }
  h4 {
    margin: var(--space-2) 0 0;
    font-size: var(--text-sm);
    color: var(--fg-2);
    font-weight: 600;
  }
  .defactions,
  .cactions,
  .writeractions,
  .cmeta {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .muted {
    color: var(--fg-2);
  }
  .none,
  .thin {
    margin: 0;
    font-size: var(--text-sm);
  }
  .err {
    margin: 0;
    color: var(--danger);
    font-size: var(--text-sm);
  }
  .chosen {
    margin: 0;
    border-left: 3px solid var(--accent);
    padding: var(--space-1) var(--space-3);
  }
  .chosen p {
    margin: 0 0 var(--space-1);
    line-height: 1.5;
  }
  .chosen footer {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    flex-wrap: wrap;
    font-size: var(--text-sm);
  }
  .src {
    font-size: var(--text-sm);
    color: var(--fg-2);
    overflow-wrap: anywhere;
  }
  .cands {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .cand {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: var(--space-2) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .dismissed .cand {
    opacity: 0.75;
  }
  .ctext {
    margin: 0;
    line-height: 1.45;
    font-size: var(--text-sm);
  }
  .grade {
    font: inherit;
    font-size: 0.72rem;
    border-radius: 999px;
    padding: 0 0.45rem;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    cursor: pointer;
  }
  .grade.strong {
    border-color: var(--accent);
    color: var(--accent);
  }
  .grade.thin {
    color: var(--fg-2);
  }
  .reasons {
    margin: 0;
    padding-left: 1.1rem;
    font-size: var(--text-sm);
    color: var(--fg-2);
  }
  .writer {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  textarea {
    font: inherit;
    font-size: var(--text-sm);
    color: var(--fg);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: var(--space-2);
    resize: vertical;
    width: 100%;
    box-sizing: border-box;
  }
  .hint {
    font-size: var(--text-sm);
  }
  .look {
    margin: 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
    font-size: var(--text-sm);
  }
  .ghost {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
    border-radius: 7px;
    padding: var(--space-1) var(--space-3);
    font: inherit;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .primary {
    border: 1px solid var(--accent);
    background: var(--accent);
    color: var(--accent-fg);
    border-radius: 7px;
    padding: var(--space-1) var(--space-3);
    font: inherit;
    cursor: pointer;
  }
  .sm {
    font-size: var(--text-sm);
  }
  .ghost:disabled,
  .primary:disabled {
    opacity: 0.55;
    cursor: default;
  }
  .linkish {
    border: none;
    background: none;
    color: var(--accent);
    cursor: pointer;
    font: inherit;
    padding: 0;
  }
  .linkish.small {
    font-size: var(--text-sm);
    align-self: flex-start;
  }
  .linkish:disabled {
    opacity: 0.6;
    cursor: default;
  }
</style>
