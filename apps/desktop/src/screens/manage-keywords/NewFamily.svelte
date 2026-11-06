<script lang="ts">
  // The *New family* section of Manage keywords (feature-tag-families.md, PR-1; PR-2.7 F2–F4): a
  // name field that offers to go to a family that already has the name, and the un-familied
  // keywords to pick as members — searchable, with the one-document tail behind a toggle. Owns the
  // pool search and the picks. The name is bound to the view, which clears it when it goes to a
  // row; going there and creating are the view's.
  import type { KeywordFamily } from '../../lib/core/types'
  import { RARE_MAX_DOCS, filterByQuery } from '../../lib/library/library'
  import Icon from '../../shell/Icon.svelte'
  import { tooltip } from '../../lib/shell/tooltip'

  let {
    families,
    unfamilied,
    keywordDocCounts,
    newCanonical = $bindable(),
    onGoToFamily,
    onCreate,
  }: {
    families: KeywordFamily[]
    unfamilied: string[] // keywords in no family, sorted; every row's add field offers them too
    keywordDocCounts: Map<string, number> // documents per raw keyword (PR-2.7 F4)
    newCanonical: string // what is typed in the name field
    onGoToFamily: (f: KeywordFamily) => void
    onCreate: (canonical: string, members: string[]) => void
  } = $props()

  // PR-2.7 F3/F4 — the flat pools stopped scanning at ~40 chips, and half of a real corpus's
  // keywords sit on one document. Both pools get a search box; the 1-doc tail is collapsed behind
  // a toggle (never deleted — search still reaches it, and so does the toggle).
  let poolQuery = $state('')
  let showRareMembers = $state(false)
  const poolMatches = $derived(filterByQuery(unfamilied, poolQuery, (k) => k))
  const poolSplit = $derived.by(() => {
    if (poolQuery.trim() !== '') return { common: poolMatches, rare: [] as string[] }
    const common: string[] = []
    const rare: string[] = []
    for (const k of poolMatches)
      ((keywordDocCounts.get(k) ?? 0) <= RARE_MAX_DOCS ? rare : common).push(k)
    return common.length === 0 ? { common: poolMatches, rare: [] as string[] } : { common, rare }
  })
  const poolShown = $derived(
    showRareMembers ? [...poolSplit.common, ...poolSplit.rare] : poolSplit.common,
  )

  // PR-2.7 F2 — the canonical was unchecked free text: no way to reach an existing family at
  // scale, and the hole D3 closed at the library boundary. Typing an existing canonical now offers
  // to *go to* that family instead of silently creating a second claim on the same keyword.
  const canonicalMatch = $derived.by(() => {
    const v = newCanonical.trim().toLowerCase()
    if (v === '') return null
    return families.find((f) => f.canonical.toLowerCase() === v) ?? null
  })
  const canonicalSuggestions = $derived.by(() => {
    const v = newCanonical.trim().toLowerCase()
    if (v === '' || canonicalMatch) return []
    return families.filter((f) => f.canonical.toLowerCase().includes(v)).slice(0, 5)
  })

  let newMembers = $state<string[]>([])
  function toggleNewMember(k: string): void {
    newMembers = newMembers.includes(k) ? newMembers.filter((x) => x !== k) : [...newMembers, k]
  }
  function submitCreate(): void {
    const canonical = newCanonical.trim()
    if (!canonical) return
    if (canonicalMatch) {
      onGoToFamily(canonicalMatch)
      return
    }
    onCreate(canonical, newMembers)
    newCanonical = ''
    newMembers = []
  }

  function autofocus(node: HTMLElement): void {
    node.focus()
  }
</script>

<section class="block">
  <h3>New family</h3>
  <p class="hint">
    Collapse near-duplicate keywords (e.g. “llm” / “llms”) into one canonical tag.
  </p>
  <div class="createrow">
    <input
      use:autofocus
      bind:value={newCanonical}
      placeholder="Canonical name, e.g. “large language models”"
      aria-label="New family canonical name"
      onkeydown={(e) => e.key === 'Enter' && submitCreate()}
    />
    <button class="btn btn-primary" onclick={submitCreate} disabled={!newCanonical.trim()} type="button" use:tooltip={!newCanonical.trim() ? 'Type a name first' : canonicalMatch ? 'Open the family that already has this name' : 'Make a family with this name and the keywords picked below'}>
      {canonicalMatch ? 'Go to family' : 'Create'}
    </button>
  </div>
  {#if canonicalMatch}
    <p class="hint match">
      <Icon name="triangle-alert" size={13} />
      “{canonicalMatch.canonical}” already exists ({canonicalMatch.doc_count} doc{canonicalMatch.doc_count ===
      1
        ? ''
        : 's'}) — creating it again would split the same keyword across two families.
    </p>
  {:else if canonicalSuggestions.length > 0}
    <div class="suggest" role="group" aria-label="Existing families matching what you typed">
      <span class="hint">Existing:</span>
      {#each canonicalSuggestions as f (f.id)}
        <button class="suggestbtn" onclick={() => onGoToFamily(f)} type="button" use:tooltip={'Open this existing family'}>
          {f.canonical}
        </button>
      {/each}
    </div>
  {/if}
  {#if unfamilied.length > 0}
    <div class="poolhead">
      <div class="searchrow small">
        <Icon name="search" size={13} />
        <input
          bind:value={poolQuery}
          placeholder="Search keywords"
          aria-label="Search un-familied keywords"
        />
      </div>
      {#if poolSplit.rare.length > 0}
        <button class="linkbtn" onclick={() => (showRareMembers = !showRareMembers)} type="button" use:tooltip={showRareMembers ? 'Hide the keywords found on only one document' : 'List the keywords found on only one document too'}>
          {showRareMembers ? 'Hide' : 'Show'} rare ({poolSplit.rare.length})
        </button>
      {/if}
    </div>
    <div class="pickrow" role="group" aria-label="Member keywords for the new family">
      {#each poolShown as k (k)}
        <button
          class="pick"
          class:on={newMembers.includes(k)}
          onclick={() => toggleNewMember(k)}
          type="button"
          aria-pressed={newMembers.includes(k)}
          use:tooltip={newMembers.includes(k) ? 'Leave this keyword out of the new family' : 'Put this keyword in the new family'}
        >
          {#if newMembers.includes(k)}<Icon name="check" size={11} />{/if}
          {k}
        </button>
      {:else}
        <p class="hint">No keywords match “{poolQuery.trim()}”.</p>
      {/each}
    </div>
  {/if}
</section>

<style>
  /* PR-2.7 F3 — a search row + a demote toggle above the pool that outgrew scanning. */
  .poolhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    margin-bottom: 0.4rem;
  }
  .searchrow.small {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: 1;
    min-width: 0;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm, 6px);
    padding: 0.2rem 0.45rem;
    color: var(--fg-2);
  }
  .searchrow.small input {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    color: var(--fg);
    font: inherit;
    font-size: 0.8rem;
    outline: none;
  }
  .linkbtn {
    flex: none;
    border: none;
    background: none;
    padding: 0;
    font: inherit;
    font-size: 0.75rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .linkbtn:hover {
    color: var(--fg);
  }
  /* PR-2.7 F2 — typing an existing canonical offers navigation, not a duplicate. */
  .hint.match {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--warn, var(--fg-2));
  }
  .suggest {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: 0.4rem;
  }
  .suggestbtn {
    border: 1px solid var(--line);
    border-radius: 999px;
    background: none;
    color: var(--fg);
    font: inherit;
    font-size: 0.75rem;
    padding: 0.1rem 0.5rem;
    cursor: pointer;
  }
  .suggestbtn:hover {
    border-color: var(--accent, var(--fg-2));
  }
  .block h3 {
    margin: 0 0 0.3rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
  }
  .hint {
    margin: 0 0 0.5rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .createrow {
    display: flex;
    gap: 0.4rem;
  }
  .createrow input {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--fg);
  }
  .createrow input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .pickrow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }
  .pick {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    color: var(--fg-2);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.14rem 0.5rem;
  }
  .pick:hover {
    border-color: var(--accent);
  }
  .pick.on {
    color: var(--accent-fg);
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }
</style>
