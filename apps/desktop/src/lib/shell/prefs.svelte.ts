// Shell view preferences — the resizable/collapsible left sidebar.
//
// Client-only and localStorage-backed, the same class as `library/prefs.svelte.ts` and the theme
// toggle: never a backend setting. Collapse and width are independent — expanding restores the
// persisted width unchanged (collapse is not a resize to zero).
//
// The mobile off-canvas drawer (`sidebarOpen`) is NOT here: it is transient navigation state that
// half the app writes on selection, so it stays in App.svelte with the rest of the shell state.

export const SIDEBAR_MIN = 200
export const SIDEBAR_MAX = 480
const SIDEBAR_DEFAULT = 260

function loadWidth(): number {
  try {
    const v = Number(localStorage.getItem('sidebarWidth'))
    return v >= SIDEBAR_MIN && v <= SIDEBAR_MAX ? v : SIDEBAR_DEFAULT
  } catch {
    return SIDEBAR_DEFAULT
  }
}

function loadCollapsed(): boolean {
  try {
    return localStorage.getItem('sidebarCollapsed') === '1'
  } catch {
    return false
  }
}

export const sidebarPrefs = $state({
  width: loadWidth(),
  collapsed: loadCollapsed(),
})

export function toggleSidebarCollapsed(): void {
  sidebarPrefs.collapsed = !sidebarPrefs.collapsed
  try {
    localStorage.setItem('sidebarCollapsed', sidebarPrefs.collapsed ? '1' : '0')
  } catch {
    /* ignore — collapse state just won't persist */
  }
}

// Pointer-drag resize. Width is clamped so the rail can't be dragged uselessly narrow or wide,
// and only persisted on release (a write per pointermove would hammer localStorage).
export function startSidebarResize(e: PointerEvent): void {
  e.preventDefault()
  const onMove = (ev: PointerEvent): void => {
    sidebarPrefs.width = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, ev.clientX))
  }
  const onUp = (): void => {
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    try {
      localStorage.setItem('sidebarWidth', String(Math.round(sidebarPrefs.width)))
    } catch {
      /* ignore — width just won't persist */
    }
  }
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', onUp)
}
