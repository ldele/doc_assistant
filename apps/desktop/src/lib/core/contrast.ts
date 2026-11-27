// WCAG 2.x contrast between two `#rrggbb` colours. Pure, so a theme's pairs can be pinned by a
// test that reads `app.css` (src/theme-contrast.test.ts) instead of being judged by eye.

/** Relative luminance of `#rrggbb` (WCAG 2.x). */
export function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) throw new Error(`not a #rrggbb colour: ${hex}`)
  const channel = (i: number): number => {
    const c = parseInt(m[1].slice(i, i + 2), 16) / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * channel(0) + 0.7152 * channel(2) + 0.0722 * channel(4)
}

/** Contrast ratio, 1 to 21, the same whichever colour comes first. */
export function contrastRatio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}
