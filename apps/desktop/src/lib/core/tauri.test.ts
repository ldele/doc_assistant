// The window half of the Tauri boundary (ADR-065): what the app's own title bar asks of the
// window, against a stand-in for `window.__TAURI__`. The rule under test is the module's: degrade,
// never throw, and draw no control that could not work.
import assert from 'node:assert/strict'
import { afterEach, test } from 'node:test'

import {
  closeWindow,
  drawsOwnWindowControls,
  isWindowMaximized,
  minimizeWindow,
  onWindowFocusChanged,
  onWindowResized,
  toggleMaximizeWindow,
} from './tauri.ts'

type Handler = (e: { payload: unknown }) => void

/** A window like Tauri's: methods on a class, so a detached call would lose `this`. */
class FakeWindow {
  calls: string[] = []
  decorated: boolean | Error = false
  maximized = false
  failCommands = false
  handlers: Record<string, Handler | null> = {}
  unlistened: string[] = []
  private listenGate: Promise<void> = Promise.resolve()
  private releaseGate: () => void = () => {}

  holdListens(): void {
    this.listenGate = new Promise((resolve) => (this.releaseGate = resolve))
  }
  releaseListens(): void {
    this.releaseGate()
  }

  private command(name: string): Promise<void> {
    if (!(this instanceof FakeWindow)) throw new Error('called without its window')
    this.calls.push(name)
    return this.failCommands ? Promise.reject(new Error('not allowed')) : Promise.resolve()
  }
  minimize(): Promise<void> {
    return this.command('minimize')
  }
  toggleMaximize(): Promise<void> {
    return this.command('toggleMaximize')
  }
  close(): Promise<void> {
    return this.command('close')
  }
  isMaximized(): Promise<boolean> {
    return Promise.resolve(this.maximized)
  }
  isDecorated(): Promise<boolean> {
    return this.decorated instanceof Error ? Promise.reject(this.decorated) : Promise.resolve(this.decorated)
  }
  private listen(event: string, handler: Handler): Promise<() => void> {
    this.handlers[event] = handler
    return this.listenGate.then(() => () => {
      this.unlistened.push(event)
      this.handlers[event] = null
    })
  }
  onResized(handler: () => void): Promise<() => void> {
    return this.listen('resized', () => handler())
  }
  onFocusChanged(handler: (e: { payload: boolean }) => void): Promise<() => void> {
    return this.listen('focus', (e) => handler(e as { payload: boolean }))
  }
}

const g = globalThis as { __TAURI__?: unknown }

function inTauri(win: FakeWindow): void {
  g.__TAURI__ = { window: { getCurrentWindow: () => win } }
}

const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0))

afterEach(() => {
  delete g.__TAURI__
})

test('in a browser there are no controls to draw and every call is a no-op', async () => {
  assert.equal(await drawsOwnWindowControls(), false)
  assert.equal(await isWindowMaximized(), false)
  minimizeWindow()
  toggleMaximizeWindow()
  closeWindow()
  const stop = onWindowResized(() => assert.fail('no window, no event'))
  assert.equal(typeof stop, 'function')
  stop()
})

test('controls are drawn only for a window with no native frame', async () => {
  const win = new FakeWindow()
  inTauri(win)
  win.decorated = false
  assert.equal(await drawsOwnWindowControls(), true)
  win.decorated = true
  assert.equal(await drawsOwnWindowControls(), false)
  win.decorated = new Error('core:window:allow-is-decorated not granted')
  assert.equal(await drawsOwnWindowControls(), false)
})

test('a Tauri global without the window API draws nothing', async () => {
  g.__TAURI__ = { core: {} }
  assert.equal(await drawsOwnWindowControls(), false)
  g.__TAURI__ = { window: { getCurrentWindow: () => { throw new Error('no window') } } }
  assert.equal(await drawsOwnWindowControls(), false)
})

test('the three buttons reach the window, on the window object', async () => {
  const win = new FakeWindow()
  inTauri(win)
  minimizeWindow()
  toggleMaximizeWindow()
  closeWindow()
  assert.deepEqual(win.calls, ['minimize', 'toggleMaximize', 'close'])
  win.maximized = true
  assert.equal(await isWindowMaximized(), true)
})

test('a refused command is swallowed, not an unhandled rejection', async () => {
  const win = new FakeWindow()
  win.failCommands = true
  inTauri(win)
  let unhandled = 0
  const count = (): void => {
    unhandled++
  }
  process.on('unhandledRejection', count)
  try {
    minimizeWindow()
    toggleMaximizeWindow()
    closeWindow()
    await tick()
  } finally {
    process.off('unhandledRejection', count)
  }
  assert.equal(unhandled, 0)
  assert.equal(win.calls.length, 3)
})

test('resize and focus events reach the handler, and unsubscribing stops them', async () => {
  const win = new FakeWindow()
  inTauri(win)
  let resized = 0
  const focus: boolean[] = []
  const stopResize = onWindowResized(() => resized++)
  const stopFocus = onWindowFocusChanged((f) => focus.push(f))
  await tick()
  win.handlers.resized?.({ payload: { width: 800, height: 600 } })
  win.handlers.focus?.({ payload: false })
  win.handlers.focus?.({ payload: true })
  assert.equal(resized, 1)
  assert.deepEqual(focus, [false, true])
  stopResize()
  stopFocus()
  assert.deepEqual(win.unlistened.sort(), ['focus', 'resized'])
})

test('unsubscribing before the listener is ready still removes it once it is', async () => {
  const win = new FakeWindow()
  win.holdListens()
  inTauri(win)
  const stop = onWindowResized(() => {})
  stop()
  assert.deepEqual(win.unlistened, [])
  win.releaseListens()
  await tick()
  assert.deepEqual(win.unlistened, ['resized'])
})
