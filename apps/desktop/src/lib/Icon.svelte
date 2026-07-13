<script lang="ts">
  // Lucide inline-SVG icons (docs/specs/feature-visual-identity.md, SPRINT-016). One coherent,
  // stroke-consistent set replacing the app's chrome emoji glyphs — inline SVG means no icon-font,
  // no CDN, and `currentColor` theming (inherits the button's text color, both themes for free).
  // Decorative by default (aria-hidden); the labelled parent button owns the accessible name.
  // Paths are Lucide (ISC license), 24x24 viewBox, 2px stroke, rounded caps/joins. Rendered as
  // real <path> children under {#if} (SVG-namespace-safe — no {@html}).
  type IconName =
    | 'menu'
    | 'download'
    | 'settings'
    | 'arrow-left'
    | 'rotate-ccw'
    | 'x'
    | 'check'
    | 'pencil'
    | 'triangle-alert'

  let { name, size = 18 }: { name: IconName; size?: number } = $props()
</script>

<svg
  class="icon"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
  focusable="false"
>
  {#if name === 'menu'}
    <path d="M4 6h16" /><path d="M4 12h16" /><path d="M4 18h16" />
  {:else if name === 'download'}
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path
      d="M12 15V3"
    />
  {:else if name === 'settings'}
    <path
      d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
    /><circle cx="12" cy="12" r="3" />
  {:else if name === 'arrow-left'}
    <path d="M19 12H5" /><path d="M12 19l-7-7 7-7" />
  {:else if name === 'rotate-ccw'}
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" />
  {:else if name === 'x'}
    <path d="M18 6 6 18" /><path d="M6 6l12 12" />
  {:else if name === 'check'}
    <path d="M20 6 9 17l-5-5" />
  {:else if name === 'pencil'}
    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z" /><path d="m15 5 4 4" />
  {:else if name === 'triangle-alert'}
    <path
      d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"
    /><path d="M12 9v4" /><path d="M12 17h.01" />
  {/if}
</svg>

<style>
  .icon {
    display: inline-block;
    vertical-align: -0.15em;
    flex: none;
  }
</style>
