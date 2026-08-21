import type { ValueMapPoint } from './api'

export function hexToRgb(hex: string): [number, number, number] {
  const m = hex.replace('#', '')
  return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)]
}

// "−0.3" not "-0.3" - the design uses a true minus sign, not a hyphen, for negative axis labels
export function signedTick(v: number): string {
  if (v === 0) return '0.0'
  return v > 0 ? `+${v.toFixed(1)}` : `−${Math.abs(v).toFixed(1)}`
}

// the value map is desktop-only and the app shell is a fixed 1180px container (not fluid), so the
// chart's pixel dimensions are deterministic - matches the two-column grid's actual column width
// (1180 - 2*44 padding - 56 gap) / 2 - which lets label placement reason in real pixel space
// instead of approximate percentages, same approach app/dashboard.py used for the Streamlit version
export const CHART_W = 518
export const CHART_H = 390
const MARGIN = 44

export interface XScale {
  domainMin: number
  domainMax: number
  ticks: number[]
}

// x axis (points per £M) - plain linear domain with ~8% headroom and a "nice" step, same
// rounding approach as ticks.ts' niceScale but kept local since the y axis below needs a
// zero-anchored variant that niceScale doesn't support
export function xScale(values: number[]): XScale {
  const lo = Math.min(...values, 0.5)
  const hi = Math.max(...values, lo + 1)
  const span = Math.max(hi - lo, 0.1)
  const pad = span * 0.12
  const domainMin = Math.max(0, lo - pad)
  const domainMax = hi + pad
  const rawStep = (domainMax - domainMin) / 4
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const candidates = [1, 2, 5, 10].map((m) => m * magnitude)
  const step = candidates.reduce((best, c) => (Math.abs(c - rawStep) < Math.abs(best - rawStep) ? c : best))
  const ticks: number[] = []
  for (let t = Math.ceil(domainMin / step) * step; t <= domainMax + 1e-9; t += step) {
    ticks.push(Math.round(t * 100) / 100)
  }
  return { domainMin, domainMax, ticks }
}

export interface YScale {
  domainMax: number
  ticks: number[]
}

// y axis (expected price move) - symmetric around zero so the "no price change" line sits
// exactly at the vertical midpoint regardless of how lopsided the actual data is. ticks stop at
// the smallest "nice" step that still covers the data, so the top tick isn't far past the actual max
export function yScale(values: number[]): YScale {
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 0.1)
  const rawStep = maxAbs / 2
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const candidates = [1, 2, 5, 10].map((m) => m * magnitude)
  const step = candidates.reduce((best, c) => (Math.abs(c - rawStep) < Math.abs(best - rawStep) ? c : best))
  const ticksPerSide = Math.max(1, Math.ceil(maxAbs / step))
  const domainMax = step * ticksPerSide
  const ticks: number[] = []
  for (let t = -ticksPerSide; t <= ticksPerSide; t++) ticks.push(Math.round(t * step * 100) / 100)
  return { domainMax, ticks }
}

type Rect = [number, number, number, number]
export type Side = 'right' | 'left' | 'below' | 'above' | 'upperRight' | 'upperLeft' | 'lowerRight' | 'lowerLeft'

export interface PlacedBubble extends ValueMapPoint {
  x: number
  y: number
  dia: number
  labelSide: Side
}

function overlapArea(a: Rect, b: Rect): number {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  return w > 0 && h > 0 ? w * h : 0
}

function edgeDist(cx: number, cy: number, ox: number, oy: number, odia: number): number {
  return Math.hypot(cx - ox, cy - oy) - odia / 2
}

// greedy label placement in real pixel space: try 8 directions around the bubble and take the
// first spot clear of every dot and already-placed label, placing selected/highest-points bubbles
// first so they win the preferred "beside" spot in dense clusters. When a cluster is so tight that
// every direction collides, fall back to whichever direction overlaps the least (rather than
// always "right", which just stacks every blocked label on top of each other) - direct port of
// app/dashboard.py's value_map() collision-avoidance loop, extended with diagonal candidates
export function layoutValueMap(rows: ValueMapPoint[], xd: XScale, yd: YScale): PlacedBubble[] {
  if (rows.length === 0) return []
  const ptsHi = Math.max(...rows.map((r) => r.points), 1)
  const xSpan = xd.domainMax - xd.domainMin || 1

  const positioned = rows.map((r) => {
    const move = r.move ?? 0
    const fx = (r.ppm - xd.domainMin) / xSpan
    const fy = 0.5 - move / (2 * yd.domainMax || 1)
    return {
      ...r,
      x: MARGIN + fx * (CHART_W - 2 * MARGIN),
      y: 14 + Math.max(0, Math.min(1, fy)) * (CHART_H - 44),
      dia: 16 + (r.points / ptsHi) * 24,
    }
  })

  const dotRects: Rect[] = positioned.map((r) => [r.x - r.dia / 2, r.y - r.dia / 2, r.x + r.dia / 2, r.y + r.dia / 2])
  const placedLabels: Rect[] = []
  const order = [...positioned].sort((a, b) => Number(b.selected) - Number(a.selected) || b.points - a.points)
  const bySide = new Map<string, Side>()

  for (const r of order) {
    const { x, y, dia, name } = r
    const lw = name.length * 6.4 + 6
    const lh = 14
    const diag = dia / 2 * 0.72 + 3
    const candidates: { side: Side; rect: Rect }[] = [
      { side: 'right', rect: [x + dia / 2 + 4, y - lh / 2, x + dia / 2 + 4 + lw, y + lh / 2] },
      { side: 'left', rect: [x - dia / 2 - 4 - lw, y - lh / 2, x - dia / 2 - 4, y + lh / 2] },
      { side: 'below', rect: [x - lw / 2, y + dia / 2 + 2, x + lw / 2, y + dia / 2 + 2 + lh] },
      { side: 'above', rect: [x - lw / 2, y - dia / 2 - 2 - lh, x + lw / 2, y - dia / 2 - 2] },
      { side: 'upperRight', rect: [x + diag, y - diag - lh, x + diag + lw, y - diag] },
      { side: 'upperLeft', rect: [x - diag - lw, y - diag - lh, x - diag, y - diag] },
      { side: 'lowerRight', rect: [x + diag, y + diag, x + diag + lw, y + diag + lh] },
      { side: 'lowerLeft', rect: [x - diag - lw, y + diag, x - diag, y + diag + lh] },
    ]
    const blockers = [...placedLabels, ...dotRects]
    const others = positioned.filter((o) => o.id !== r.id)

    // a label placed nearer to a different bubble than to its own is confusing even when it
    // doesn't physically overlap anything (e.g. a label sitting between two bubbles can read as
    // belonging to the wrong one) - penalize those candidates so an unambiguous spot wins first.
    // require a real margin, not just a nominal edge: a spot that's only marginally closer to its
    // own dot than to a neighbour (a couple px) still reads as ambiguous to the eye
    const AMBIGUITY_MARGIN = 12
    const score = (cand: { rect: Rect }) => {
      const [cx, cy] = [(cand.rect[0] + cand.rect[2]) / 2, (cand.rect[1] + cand.rect[3]) / 2]
      const ownDist = edgeDist(cx, cy, x, y, dia)
      const nearestOther = Math.min(...others.map((o) => edgeDist(cx, cy, o.x, o.y, o.dia)))
      const overlap = blockers.reduce((sum, other) => sum + overlapArea(cand.rect, other), 0)
      return overlap + (nearestOther < ownDist + AMBIGUITY_MARGIN ? 2000 : 0)
    }
    const chosen = candidates.reduce((best, cand) => (score(cand) < score(best) ? cand : best))
    placedLabels.push(chosen.rect)
    bySide.set(r.id, chosen.side)
  }

  return positioned.map((r) => ({ ...r, labelSide: bySide.get(r.id) ?? 'right' }))
}
