export interface Scale {
  ticks: number[]
  domainMin: number
  domainMax: number
}

// picks a "nice" tick step (1/2/5/10x a power of ten) for an axis spanning [minValue, maxValue],
// then returns the tick values plus ~4% headroom on both ends. Driver P10s regularly run well
// negative (a bad DNF can cost -20), so the domain must NOT be pinned to zero or those bars would
// get clipped at the left edge - it has to track the real min/max across the column.
export function niceScale(minValue: number, maxValue: number, targetCount = 5): Scale {
  const span = Math.max(maxValue - minValue, 1)
  const rawStep = span / targetCount
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const candidates = [1, 2, 5, 10].map((m) => m * magnitude)
  const step = candidates.reduce((best, c) =>
    Math.abs(c - rawStep) < Math.abs(best - rawStep) ? c : best,
  )

  const pad = span * 0.04
  const domainMin = minValue - pad
  const domainMax = maxValue + pad

  const ticks: number[] = []
  const start = Math.ceil(domainMin / step) * step
  for (let t = start; t <= domainMax; t += step) {
    ticks.push(Math.round(t * 100) / 100)
  }

  return { ticks, domainMin, domainMax }
}
