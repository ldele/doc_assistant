import { mount } from 'svelte'
import './lib/core/fonts.css'
import './app.css'
import App from './App.svelte'
import { applyTheme, getTheme } from './lib/core/theme'

// Must run before mount() so the theme attribute is set before first paint (no
// flash-of-wrong-theme on a Dark/Light choice that differs from the OS preference).
applyTheme(getTheme())

const target = document.getElementById('app')
if (!target) throw new Error('#app mount point missing')
// index.html draws a start screen inside #app before any script runs; mount() appends, so clear it.
target.replaceChildren()

export default mount(App, { target })
