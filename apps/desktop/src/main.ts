import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'
import { applyTheme, getTheme } from './lib/theme'

// Must run before mount() so the theme attribute is set before first paint (no
// flash-of-wrong-theme on a Dark/Light choice that differs from the OS preference).
applyTheme(getTheme())

const target = document.getElementById('app')
if (!target) throw new Error('#app mount point missing')

export default mount(App, { target })
