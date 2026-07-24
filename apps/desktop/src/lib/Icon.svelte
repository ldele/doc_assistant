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
    | 'book-open'
    | 'book-open-text'
    | 'library'
    | 'pin'
    | 'archive'
    | 'trash-2'
    | 'message-square'
    | 'ellipsis'
    | 'search'
    | 'arrow-up-down'
    | 'layout-grid'
    | 'list'
    | 'folder'
    | 'file-text'
    | 'calendar'
    | 'tag'
    | 'chevron-right'
    | 'waypoints'
    | 'panel-left'
    | 'plus'
    | 'square-check-big'

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
  {:else if name === 'book-open'}
    <path d="M12 7v14" /><path
      d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"
    />
  {:else if name === 'book-open-text'}
    <path d="M12 7v14" /><path d="M16 12h2" /><path d="M16 8h2" /><path
      d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"
    /><path d="M6 8h2" /><path d="M6 12h2" />
  {:else if name === 'library'}
    <path d="m16 6 4 14" /><path d="M12 6v14" /><path d="M8 8v12" /><path d="M4 4v16" />
  {:else if name === 'pin'}
    <path d="M12 17v5" /><path
      d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"
    />
  {:else if name === 'archive'}
    <rect width="20" height="5" x="2" y="3" rx="1" /><path
      d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"
    /><path d="M10 12h4" />
  {:else if name === 'trash-2'}
    <path d="M3 6h18" /><path
      d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
    /><line x1="10" x2="10" y1="11" y2="17" /><line x1="14" x2="14" y1="11" y2="17" />
  {:else if name === 'message-square'}
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  {:else if name === 'ellipsis'}
    <circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" />
  {:else if name === 'search'}
    <circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" />
  {:else if name === 'arrow-up-down'}
    <path d="m21 16-4 4-4-4" /><path d="M17 20V4" /><path d="m3 8 4-4 4 4" /><path d="M7 4v16" />
  {:else if name === 'layout-grid'}
    <rect width="7" height="7" x="3" y="3" rx="1" /><rect width="7" height="7" x="14" y="3" rx="1" /><rect
      width="7"
      height="7"
      x="14"
      y="14"
      rx="1"
    /><rect width="7" height="7" x="3" y="14" rx="1" />
  {:else if name === 'list'}
    <path d="M3 12h.01" /><path d="M3 18h.01" /><path d="M3 6h.01" /><path d="M8 12h13" /><path
      d="M8 18h13"
    /><path d="M8 6h13" />
  {:else if name === 'folder'}
    <path
      d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
    />
  {:else if name === 'file-text'}
    <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /><path
      d="M14 2v4a2 2 0 0 0 2 2h4"
    /><path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />
  {:else if name === 'calendar'}
    <path d="M8 2v4" /><path d="M16 2v4" /><rect width="18" height="18" x="3" y="4" rx="2" /><path
      d="M3 10h18"
    />
  {:else if name === 'tag'}
    <path
      d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"
    /><circle cx="7.5" cy="7.5" r=".5" fill="currentColor" />
  {:else if name === 'chevron-right'}
    <path d="m9 18 6-6-6-6" />
  {:else if name === 'waypoints'}
    <circle cx="12" cy="4.5" r="2.5" /><path d="m10.2 6.3-3.9 3.9" /><circle
      cx="4.5"
      cy="12"
      r="2.5"
    /><path d="M7 12h10" /><circle cx="19.5" cy="12" r="2.5" /><path
      d="m13.8 17.7 3.9-3.9"
    /><circle cx="12" cy="19.5" r="2.5" />
  {:else if name === 'panel-left'}
    <rect width="18" height="18" x="3" y="3" rx="2" /><path d="M9 3v18" />
  {:else if name === 'plus'}
    <path d="M5 12h14" /><path d="M12 5v14" />
  {:else if name === 'square-check-big'}
    <path d="M21 10.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h12.5" /><path d="m9 11 3 3L22 4" />
  {/if}
</svg>

<style>
  .icon {
    display: inline-block;
    vertical-align: -0.15em;
    flex: none;
  }
</style>
