// palette - warm editorial light theme, matches app/dashboard.py exactly
export const INK = '#16150f'
export const MUTED = '#57534b'
export const MUTED2 = '#8d887c'
export const FAINT = '#b0aa9c'
export const GREEN = '#2f6a53'
export const REDNEG = '#a8412a'
export const CARD = '#f1efe8'
export const PANEL = '#e5e0d3'
export const PANEL_MOBILE = '#eae7de'  // the design uses a slightly different panel shade on mobile
export const LINE = 'rgba(22,21,15,.14)'
export const LINE_SOFT = 'rgba(22,21,15,.07)'
export const LINE_MED = 'rgba(22,21,15,.16)'
export const LINE_STR = 'rgba(22,21,15,.3)'
export const GREEN_TINT = 'rgba(47,106,83,.07)'
export const RED_TINT = 'rgba(201,52,28,.09)'
export const RED_BORDER = 'rgba(201,52,28,.28)'

export const FONT = 'Archivo, system-ui, sans-serif'
export const MONO = "'IBM Plex Mono', ui-monospace, monospace"

function hexToRgb(hex: string) {
  const clean = hex.replace('#', '')
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  }
}

// alpha-composite fgHex over bgHex - used to find the colour a translucent bar actually reads as
// on screen, so markers drawn "one step darker" start from what's visible, not the raw source hue
export function mix(fgHex: string, bgHex: string, alpha: number): string {
  const f = hexToRgb(fgHex)
  const b = hexToRgb(bgHex)
  const toHex = (x: number) => Math.round(x).toString(16).padStart(2, '0')
  const r = f.r * alpha + b.r * (1 - alpha)
  const g = f.g * alpha + b.g * (1 - alpha)
  const bl = f.b * alpha + b.b * (1 - alpha)
  return `#${toHex(r)}${toHex(g)}${toHex(bl)}`
}

export function relLuminance(hex: string): number {
  const { r, g, b } = hexToRgb(hex)
  const lin = (c: number) => {
    const v = c / 255
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

// WCAG-style contrast ratio between two colours (1 = identical, 21 = black on white)
export function contrastRatio(hexA: string, hexB: string): number {
  const la = relLuminance(hexA)
  const lb = relLuminance(hexB)
  const [hi, lo] = la > lb ? [la, lb] : [lb, la]
  return (hi + 0.05) / (lo + 0.05)
}

// one ramp-step darker than the given hex colour, hue preserved - used for markers that need to
// read as "the same colour, but a deliberate mark" rather than a fixed colour across every row
export function darken(hex: string, amount = 0.2): string {
  const clean = hex.replace('#', '')
  const r = parseInt(clean.slice(0, 2), 16) / 255
  const g = parseInt(clean.slice(2, 4), 16) / 255
  const b = parseInt(clean.slice(4, 6), 16) / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  let h = 0
  let s = 0
  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    if (max === r) h = (g - b) / d + (g < b ? 6 : 0)
    else if (max === g) h = (b - r) / d + 2
    else h = (r - g) / d + 4
    h /= 6
  }
  const l2 = Math.max(0, l - amount)
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  let r2: number
  let g2: number
  let b2: number
  if (s === 0) {
    r2 = g2 = b2 = l2
  } else {
    const q = l2 < 0.5 ? l2 * (1 + s) : l2 + s - l2 * s
    const p = 2 * l2 - q
    r2 = hue2rgb(p, q, h + 1 / 3)
    g2 = hue2rgb(p, q, h)
    b2 = hue2rgb(p, q, h - 1 / 3)
  }
  const toHex = (x: number) => Math.round(x * 255).toString(16).padStart(2, '0')
  return `#${toHex(r2)}${toHex(g2)}${toHex(b2)}`
}

// starts one modest step darker than baseHex, then keeps darkening until it clears a contrast
// ratio against baseHex. the target scales with baseHex's own lightness rather than using one
// fixed number: a pale colour (silvers, sky blue) needs a much bigger drop to read as a deliberate
// mark instead of "slightly smudged", while an already-dark neutral (the grey used for unselected
// rows) only needs a small nudge - forcing the same big drop on it just reproduces flat black. a
// fixed lightness step alone would also under-darken saturated reds/greens, whose HSL lightness
// barely maps to perceptual luminance, so the step size still adapts per hue on top of that.
export function tickColor(baseHex: string): string {
  const minContrast = 1.2 + relLuminance(baseHex) * 1.1
  let amount = 0.08
  let out = darken(baseHex, amount)
  while (contrastRatio(out, baseHex) < minContrast && amount < 0.5) {
    amount += 0.05
    out = darken(baseHex, amount)
  }
  return out
}
