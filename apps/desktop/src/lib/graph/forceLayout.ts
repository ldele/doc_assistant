// Deterministic force-directed layout for the concept-graph ego view (feature-concept-graph.md
// PR-G2a, ADR-017). Pure + seeded: same input → identical positions, which is what makes the SVG
// DOM assertable with no frontend test runner (the layout is the risk; determinism is the safety
// net). Hand-rolled — no dependency (the frontend is a deliberate 1-dep artifact; a graph lib would
// be 4× the whole bundle, and one using eval would break only in the packaged Tauri build).
//
// Fruchterman–Reingold: all-pairs repulsion + per-edge attraction, cooled over a fixed iteration
// count, then fit to the viewBox. Ego graphs are tiny (median 6 nodes, hub 21), so O(n²) per tick
// is trivial. Every distance is floored at EPS and the init is a phyllotaxis spiral (no two nodes
// start coincident), so no cx/cy can ever be NaN — the classic force-layout bug, closed by design.

export interface Point {
  x: number
  y: number
}

export interface LayoutEdge {
  source: string
  target: string
}

export interface LayoutOptions {
  width: number
  height: number
  seed?: number
  iterations?: number
  padding?: number
}

const GOLDEN_ANGLE = 2.399963229728653 // radians — the phyllotaxis spread that avoids coincidence
const EPS = 0.01 // distance floor: no division by zero, so no NaN can propagate into a coordinate

/** mulberry32 — a tiny deterministic PRNG. Seeded so a fixed `seed` gives a fixed jitter, matching
 *  the skeleton's own `seed: 42`. Never `Math.random()` (it would make the layout non-reproducible
 *  and the DOM assertions flaky). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Lay out `nodeIds` connected by `edges` inside a `width`×`height` box.
 *
 * Returns a map from node id to a finite `{x, y}` inside `[padding, width-padding]` ×
 * `[padding, height-padding]`. Edges whose endpoints are not both in `nodeIds` are ignored, so an
 * ego subgraph can pass the full edge list unfiltered if convenient.
 */
export function forceLayout(
  nodeIds: readonly string[],
  edges: readonly LayoutEdge[],
  opts: LayoutOptions,
): Map<string, Point> {
  const { width, height } = opts
  const seed = opts.seed ?? 42
  const iterations = opts.iterations ?? 300
  const padding = opts.padding ?? 24
  const pos = new Map<string, Point>()
  const n = nodeIds.length
  if (n === 0) return pos

  const rand = mulberry32(seed)
  // Phyllotaxis spiral init around the centre — deterministic and, crucially, no two nodes share a
  // point (distinct radius+angle per index). A small seeded jitter honours `seed` without risking a
  // collision. This is what lets the EPS floor below stay a safety net, not the primary defence.
  const cx = width / 2
  const cy = height / 2
  const spread = Math.min(width, height) * 0.38
  nodeIds.forEach((id, i) => {
    const r = Math.sqrt((i + 0.5) / n) * spread
    const theta = i * GOLDEN_ANGLE
    pos.set(id, {
      x: cx + r * Math.cos(theta) + (rand() - 0.5),
      y: cy + r * Math.sin(theta) + (rand() - 0.5),
    })
  })

  if (n === 1) {
    pos.set(nodeIds[0], { x: cx, y: cy })
    return pos
  }

  const present = new Set(nodeIds)
  const realEdges = edges.filter((e) => present.has(e.source) && present.has(e.target))

  const area = width * height
  const k = Math.sqrt(area / n) // ideal separation
  let temp = Math.max(width, height) / 10
  const cooling = 0.95

  for (let it = 0; it < iterations; it++) {
    const disp = new Map<string, Point>()
    for (const id of nodeIds) disp.set(id, { x: 0, y: 0 })

    // Repulsion — every unordered pair pushes apart with force k²/d.
    for (let i = 0; i < n; i++) {
      const a = pos.get(nodeIds[i])!
      const da = disp.get(nodeIds[i])!
      for (let j = i + 1; j < n; j++) {
        const b = pos.get(nodeIds[j])!
        const db = disp.get(nodeIds[j])!
        const dx = a.x - b.x
        const dy = a.y - b.y
        const dist = Math.max(EPS, Math.hypot(dx, dy))
        const force = (k * k) / dist
        const ux = dx / dist
        const uy = dy / dist
        da.x += ux * force
        da.y += uy * force
        db.x -= ux * force
        db.y -= uy * force
      }
    }

    // Attraction — each edge pulls its endpoints together with force d²/k.
    for (const e of realEdges) {
      const a = pos.get(e.source)!
      const b = pos.get(e.target)!
      const da = disp.get(e.source)!
      const db = disp.get(e.target)!
      const dx = a.x - b.x
      const dy = a.y - b.y
      const dist = Math.max(EPS, Math.hypot(dx, dy))
      const force = (dist * dist) / k
      const ux = dx / dist
      const uy = dy / dist
      da.x -= ux * force
      da.y -= uy * force
      db.x += ux * force
      db.y += uy * force
    }

    // Apply, capped by the current temperature; cool down.
    for (const id of nodeIds) {
      const p = pos.get(id)!
      const d = disp.get(id)!
      const dl = Math.max(EPS, Math.hypot(d.x, d.y))
      p.x += (d.x / dl) * Math.min(dl, temp)
      p.y += (d.y / dl) * Math.min(dl, temp)
    }
    temp *= cooling
  }

  return fitToBox(pos, nodeIds, width, height, padding)
}

/** Rescale the converged positions to fill the box with a uniform margin. A degenerate axis (every
 *  node on one line) is centred rather than divided by zero. */
function fitToBox(
  pos: Map<string, Point>,
  nodeIds: readonly string[],
  width: number,
  height: number,
  padding: number,
): Map<string, Point> {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const id of nodeIds) {
    const p = pos.get(id)!
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  const spanX = maxX - minX
  const spanY = maxY - minY
  const innerW = width - 2 * padding
  const innerH = height - 2 * padding
  const scaleX = spanX > EPS ? innerW / spanX : 0
  const scaleY = spanY > EPS ? innerH / spanY : 0
  // Uniform scale (keep aspect); use the tighter axis so nothing spills outside the box.
  const scale = Math.min(scaleX || scaleY, scaleY || scaleX) || 1
  const drawnW = spanX * scale
  const drawnH = spanY * scale
  const offX = padding + (innerW - drawnW) / 2
  const offY = padding + (innerH - drawnH) / 2
  for (const id of nodeIds) {
    const p = pos.get(id)!
    p.x = spanX > EPS ? offX + (p.x - minX) * scale : width / 2
    p.y = spanY > EPS ? offY + (p.y - minY) * scale : height / 2
  }
  return pos
}
