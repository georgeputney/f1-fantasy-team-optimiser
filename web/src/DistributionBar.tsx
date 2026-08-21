import { INK } from './theme'
import type { Distribution } from './api'

interface Props {
  dist: Distribution | null
  points: number
  domainMin: number
  domainMax: number
  ticks: number[]
  color: string
  selected: boolean
  compact?: boolean
}

// graded distribution bar: faded ends reach P10/P90, the solid middle covers P25-P75, a caret
// marks the simulation's own median. selected rows render in the constructor colour at full
// strength; unselected rows fall back to a low-opacity neutral so selected picks stand out.
// compact shaves 2px off the row/caret height for the mobile ladder. the domain is NOT
// pinned to zero - P10 regularly runs negative (a bad DNF costs -20), so it tracks the
// real min/max across the column or those bars would get clipped at the left edge.
// the caret deliberately tracks dist.median, not the deterministic `points` estimate - that's a
// separately-computed value from a different model and can legitimately fall outside the P10-P90
// range entirely, which reads as a bug (a marker sitting outside the box it's drawn on). points is
// only a fallback for the rare row with no distribution to derive a median from.
export function DistributionBar({ dist, points, domainMin, domainMax, ticks, color, selected, compact }: Props) {
  const span = domainMax - domainMin || 1
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - domainMin) / span) * 100))
  const caretValue = dist ? dist.median : points
  const barColor = selected ? color : INK
  const fadedOpacity = selected ? 0.32 : 0.13
  const solidOpacity = selected ? 1 : 0.4
  const caretColor = selected ? INK : 'rgba(22,21,15,.5)'
  const rowHeight = compact ? 16 : 18
  const baselineTop = compact ? 7.5 : 8.5
  const fadedTop = compact ? 5.5 : 6.5
  const solidTop = compact ? 3.5 : 4.5
  const caretSize = compact ? 4 : 5

  return (
    <div style={{ position: 'relative', height: rowHeight }}>
      {ticks.map((t) => (
        <div
          key={t}
          style={{
            position: 'absolute', left: `${pct(t)}%`, top: 0, bottom: 0,
            width: 1, background: 'rgba(22,21,15,.07)',
          }}
        />
      ))}
      <div style={{ position: 'absolute', left: 0, right: 0, top: baselineTop, height: 1, background: 'rgba(22,21,15,.08)' }} />
      {dist && (
        <>
          <div
            style={{
              position: 'absolute', left: `${pct(dist.p10)}%`, top: fadedTop, height: 5,
              minWidth: compact ? 2 : 3, width: `${pct(dist.p25) - pct(dist.p10)}%`,
              background: barColor, opacity: fadedOpacity,
            }}
          />
          <div
            style={{
              position: 'absolute', left: `${pct(dist.p75)}%`, top: fadedTop, height: 5,
              minWidth: compact ? 2 : 3, width: `${pct(dist.p90) - pct(dist.p75)}%`,
              background: barColor, opacity: fadedOpacity,
            }}
          />
          <div
            style={{
              position: 'absolute', left: `${pct(dist.p25)}%`, top: solidTop, height: 9,
              minWidth: compact ? 5 : 6, width: `${pct(dist.p75) - pct(dist.p25)}%`,
              background: barColor, opacity: solidOpacity,
            }}
          />
        </>
      )}
      <div
        style={{
          position: 'absolute', left: `${pct(caretValue)}%`, top: 0, marginLeft: -3,
          width: 0, height: 0,
          borderLeft: '3px solid transparent', borderRight: '3px solid transparent',
          borderTop: `${caretSize}px solid ${caretColor}`,
        }}
      />
    </div>
  )
}
