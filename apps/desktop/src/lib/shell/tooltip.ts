// The shared tooltip: a button's hover text (ADR-063 choice 6, GLOSSARY U-003), shown when the
// pointer rests on the button and when the keyboard moves focus to it. One element serves the whole
// app, so every button's text looks and behaves the same. It meets WCAG 2.2 SC 1.4.13 (content on
// hover or focus), which the native `title` attribute does not:
//   - dismissible: Esc hides it without moving the pointer or the focus;
//   - hoverable: the pointer can cross from the button onto the text without it disappearing;
//   - persistent: it stays until the pointer and the focus have both left, or until Esc.
// It follows the WAI-ARIA Authoring Practices tooltip pattern: the text is `role="tooltip"` and the
// button's `aria-describedby` points at it while it shows. The text explains what the button will
// do; it never names it, so an icon-only button still needs its own `aria-label`.
//
// Use: `<button use:tooltip={'Leave select mode'}>Done</button>`. A null or empty text shows
// nothing. The placement maths and the text rule are pure and held by `tooltip.test.ts`; the rest
// touches `document` only inside the action, so the module imports cleanly under `node:test`.
import type { Action } from 'svelte/action'

export interface Box {
  left: number
  top: number
  width: number
  height: number
}

export interface Size {
  width: number
  height: number
}

export interface Placement {
  left: number
  top: number
  side: 'below' | 'above'
}

/** Space between a button and its text, and the least space kept to the window's edge (px). */
export const GAP = 6
export const EDGE = 8

/** The text to show for an action value: whitespace collapsed, or null when there is none. */
export function hoverText(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const text = value.replace(/\s+/g, ' ').trim()
  return text === '' ? null : text
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(value, Math.max(low, high)))
}

/**
 * Where the text goes: centred under the button, above it when it does not fit below and there is
 * more room above, and always EDGE px inside the window, so it is never cut off at 375 px.
 */
export function placeTooltip(anchor: Box, tip: Size, view: Size): Placement {
  const roomBelow = view.height - (anchor.top + anchor.height) - GAP - EDGE
  const roomAbove = anchor.top - GAP - EDGE
  const side = tip.height <= roomBelow || roomBelow >= roomAbove ? 'below' : 'above'
  const top = side === 'below' ? anchor.top + anchor.height + GAP : anchor.top - GAP - tip.height
  const left = anchor.left + anchor.width / 2 - tip.width / 2
  return {
    side,
    top: clamp(top, EDGE, view.height - tip.height - EDGE),
    left: clamp(left, EDGE, view.width - tip.width - EDGE),
  }
}

// ---- The DOM side: one element, one owner at a time ------------------------------------------

/** How long the pointer rests on a button before its text shows. */
const SHOW_DELAY_MS = 500
/** How long the text waits after the pointer leaves, so the pointer can cross onto it. */
const HIDE_GRACE_MS = 150
/** After one text hides, the next shows at once for this long (moving along a bar). */
const WARM_MS = 600
/** Focus shows the text only this soon after a navigation key: a click focuses a button too, and
 *  a dialog moves focus by script when it opens, and neither should pop a text up. */
const NAV_KEY_WINDOW_MS = 1000
const NAV_KEYS = new Set(['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'])

const TIP_ID = 'app-tooltip'

interface Anchor {
  text: string | null
  hovered: boolean
  focused: boolean
  /** Esc or a press hid the text; it stays hidden until the pointer and the focus leave. */
  dismissed: boolean
}

const anchors = new WeakMap<HTMLElement, Anchor>()
let tip: HTMLDivElement | null = null
let owner: HTMLElement | null = null
let pendingFor: HTMLElement | null = null
let showTimer: ReturnType<typeof setTimeout> | undefined
let hideTimer: ReturnType<typeof setTimeout> | undefined
let warmUntil = 0
let lastNavKey = -Infinity
let frame = 0
let listening = false
let tracking = false

function surface(): HTMLDivElement {
  if (tip?.isConnected) return tip
  const el = document.createElement('div')
  el.id = TIP_ID
  el.className = 'app-tooltip'
  el.setAttribute('role', 'tooltip')
  el.hidden = true
  el.addEventListener('pointerenter', () => clearTimeout(hideTimer))
  el.addEventListener('pointerleave', () => {
    if (owner && !anchors.get(owner)?.focused) scheduleHide()
  })
  document.body.append(el)
  tip = el
  return el
}

function describe(node: HTMLElement, on: boolean): void {
  const ids = (node.getAttribute('aria-describedby') ?? '').split(/\s+/).filter((id) => id && id !== TIP_ID)
  if (on) ids.push(TIP_ID)
  if (ids.length > 0) node.setAttribute('aria-describedby', ids.join(' '))
  else node.removeAttribute('aria-describedby')
}

function place(): void {
  frame = 0
  if (!owner || !tip) return
  const r = owner.getBoundingClientRect()
  const outOfView = r.width === 0 || r.bottom < 0 || r.top > window.innerHeight
  if (!owner.isConnected || outOfView) {
    hide()
    return
  }
  const p = placeTooltip(
    r,
    { width: tip.offsetWidth, height: tip.offsetHeight },
    { width: window.innerWidth, height: window.innerHeight },
  )
  tip.style.left = `${Math.round(p.left)}px`
  tip.style.top = `${Math.round(p.top)}px`
  tip.dataset.side = p.side
}

function schedulePlace(): void {
  if (!frame) frame = requestAnimationFrame(place)
}

function track(on: boolean): void {
  if (on === tracking) return
  tracking = on
  const method = on ? 'addEventListener' : 'removeEventListener'
  window[method]('scroll', schedulePlace, true)
  window[method]('resize', schedulePlace)
}

function show(node: HTMLElement): void {
  const a = anchors.get(node)
  clearTimeout(showTimer)
  clearTimeout(hideTimer)
  pendingFor = null
  if (!a || a.text === null || !node.isConnected) return
  const el = surface()
  if (owner && owner !== node) describe(owner, false)
  owner = node
  el.textContent = a.text
  el.hidden = false
  describe(node, true)
  track(true)
  place()
}

function hide(): void {
  clearTimeout(showTimer)
  clearTimeout(hideTimer)
  pendingFor = null
  if (!owner) return
  describe(owner, false)
  owner = null
  if (tip) tip.hidden = true
  track(false)
  warmUntil = performance.now() + WARM_MS
}

function scheduleHide(): void {
  clearTimeout(hideTimer)
  hideTimer = setTimeout(hide, HIDE_GRACE_MS)
}

function onKeydown(e: KeyboardEvent): void {
  if (NAV_KEYS.has(e.key)) lastNavKey = performance.now()
  if (e.key !== 'Escape' || !owner) return
  const a = anchors.get(owner)
  // Stays hidden while the pointer or focus is still on the button. When the pointer is on the
  // text instead, it has already left the button, and no leave would ever clear the flag.
  if (a && (a.hovered || a.focused)) a.dismissed = true
  hide()
  // The text is the top layer, so this Esc dismisses it alone and the next one reaches the dialog
  // or panel underneath (the WAI-ARIA tooltip pattern). Captured on the window, so it runs first.
  e.preventDefault()
  e.stopPropagation()
}

function listen(): void {
  if (listening) return
  listening = true
  window.addEventListener('keydown', onKeydown, true)
  window.addEventListener('blur', hide)
}

/** `use:tooltip={text}`: the button's hover text, on pointer hover and on keyboard focus. */
export const tooltip: Action<HTMLElement, string | null | undefined> = (node, value) => {
  const a: Anchor = { text: hoverText(value), hovered: false, focused: false, dismissed: false }
  anchors.set(node, a)
  listen()
  node.removeAttribute('title') // one hover text, never the native one as well

  const enter = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') return // no hover on touch; the label and the screen carry it
    a.hovered = true
    if (owner === node) {
      clearTimeout(hideTimer)
      return
    }
    if (a.dismissed || a.text === null) return
    clearTimeout(showTimer)
    pendingFor = node
    const warm = owner !== null || performance.now() < warmUntil
    showTimer = setTimeout(() => show(node), warm ? 0 : SHOW_DELAY_MS)
  }
  const leave = (): void => {
    a.hovered = false
    if (!a.focused) a.dismissed = false
    if (pendingFor === node) {
      clearTimeout(showTimer)
      pendingFor = null
    }
    if (owner === node && !a.focused) scheduleHide()
  }
  const focus = (): void => {
    if (!node.matches(':focus-visible')) return
    if (performance.now() - lastNavKey > NAV_KEY_WINDOW_MS) return
    a.focused = true
    if (!a.dismissed) show(node)
  }
  const blur = (): void => {
    a.focused = false
    if (!a.hovered) a.dismissed = false
    if (owner === node && !a.hovered) hide()
  }
  const press = (): void => {
    // Pressing the button answers what its text was for; it shows again after the pointer leaves.
    a.dismissed = true
    if (owner === node || pendingFor === node) hide()
  }

  node.addEventListener('pointerenter', enter)
  node.addEventListener('pointerleave', leave)
  node.addEventListener('pointerdown', press)
  node.addEventListener('focus', focus)
  node.addEventListener('blur', blur)

  return {
    update(next) {
      a.text = hoverText(next)
      if (owner !== node) return
      if (a.text === null) hide()
      else if (tip) {
        tip.textContent = a.text
        place()
      }
    },
    destroy() {
      if (owner === node || pendingFor === node) hide()
      node.removeEventListener('pointerenter', enter)
      node.removeEventListener('pointerleave', leave)
      node.removeEventListener('pointerdown', press)
      node.removeEventListener('focus', focus)
      node.removeEventListener('blur', blur)
      anchors.delete(node)
    },
  }
}
