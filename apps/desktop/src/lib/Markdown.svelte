<script lang="ts">
  import { marked } from 'marked'

  // Content is our own backend's markdown (the answer + pre-rendered blocks) — trusted,
  // local, single-user. A hardened build would run it through DOMPurify.
  let { source = '' }: { source?: string } = $props()
  const html = $derived(marked.parse(source, { async: false }))
</script>

<div class="md">{@html html}</div>

<style>
  .md :global(p) {
    margin: 0.4em 0;
  }
  .md :global(code) {
    background: var(--surface-2);
    padding: 0.1em 0.3em;
    border-radius: 4px;
    font-size: 0.9em;
  }
  .md :global(h1),
  .md :global(h2),
  .md :global(h3) {
    margin: 0.6em 0 0.3em;
  }
</style>
