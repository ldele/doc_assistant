import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte'

import { devCsp } from './src/lib/core/devCsp'

const DEV_PORT = 1420

// The production policy lives in tauri.conf.json, and Tauri injects it only into the built app —
// `tauri dev` loads this server directly. So the dev server sends it too, plus what the dev loop
// needs (src/lib/core/devCsp.ts; docs/security.md S2).
const tauriConf = JSON.parse(
  readFileSync(new URL('./src-tauri/tauri.conf.json', import.meta.url), 'utf-8'),
)

// The version the window's saved lists are stamped with, so an upgrade never draws a list written
// by an older app (src/lib/core/lastSession.ts, KI-99). package.json is one of the bumped files.
const appVersion: string = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf-8'),
).version

// The Tauri webview loads this dev server in dev; the FastAPI backend (PR-M2) runs
// separately on 127.0.0.1:8001. Proxy `/api` to it so the frontend stays same-origin in
// dev (no CORS) and the packaged build hits the absolute URL (see src/lib/core/api.ts).
export default defineConfig({
  plugins: [svelte({ preprocess: vitePreprocess() })],
  define: {
    __APP_VERSION__: JSON.stringify(appVersion),
  },
  clearScreen: false,
  server: {
    port: DEV_PORT,
    strictPort: true,
    headers: {
      'Content-Security-Policy': devCsp(tauriConf.app.security.csp, DEV_PORT),
    },
    proxy: {
      '/api': {
        // DOC_API_TARGET points the dev server at another engine, e.g. one on a scratch data
        // home while the installed app holds 8001 (memory: the installed app takes Vite's proxy).
        target: process.env.DOC_API_TARGET ?? 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
})
