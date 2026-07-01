import { defineConfig } from 'vite'
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte'

// The Tauri webview loads this dev server in dev; the FastAPI backend (PR-M2) runs
// separately on 127.0.0.1:8001. Proxy `/api` to it so the frontend stays same-origin in
// dev (no CORS) and the packaged build hits the absolute URL (see src/lib/api.ts).
export default defineConfig({
  plugins: [svelte({ preprocess: vitePreprocess() })],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
})
