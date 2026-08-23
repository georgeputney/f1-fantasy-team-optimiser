import { useLayoutEffect, useRef, useState } from 'react'
import { mix, PANEL, tickColor, INK } from './theme'
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

// graded distribution bar: faded ends reach P10/P90, the solid middle covers P25-P75. Two markers:
// a caret above the bar for the deterministic expected-points estimate (what the row is actually
// ranked/selected by, and matches the points figure shown elsewhere in the row - so it has to stay
// put even on the rare row where it lands outside the box), and a small rounded tick through the bar
// for the simulation's own median - it overhangs the box by a fixed amount top and bottom and is
// drawn on top, so it reads as a deliberate mark rather than a border artifact even when it sits
// right at a p25/p75 edge. The caret is pulled up with a fixed gap above the tick's own height so the
// two never merge into one blob when the median and the projected estimate land at the same x. The
// tick's colour is derived per row (one step darker than what the bar visually reads as, including its
// own opacity) and nudged darker again until it clears a minimum contrast ratio against that colour -
// a fixed HSL lightness step is invisible against saturated reds/greens and reads as flat black
// against the already-near-black neutral used for unselected rows, so the step size adapts per hue
// instead of guessing one constant. selected rows render in the constructor colour at full strength;
// unselected rows fall back to a low-opacity neutral so selected picks stand out. compact shaves 2px
// off the row/caret height for the mobile ladder. the domain is NOT pinned to zero - P10 regularly
// runs negative (a bad DNF costs -20), so it tracks the real min/max across the column or those bars
// would get clipped.
export function DistributionBar({ dist, points, domainMin, domainMax, ticks, color, selected, compact }: Props) {
  // percentage-based left positions put the median tick at a different fractional pixel offset on
  // every row (it tracks each row's own median value), so the browser anti-aliases it differently
  // row to row - some land clean on a pixel boundary, others straddle two and blur into a visibly
  // thicker mark even though the CSS width is identical everywhere. rounding the tick's position
  // relative to its own container isn't enough on its own - the container itself can sit at a
  // fractional viewport offset (grid/flex layouts routinely produce those), so two rows that are
  // "the same" relative offset can still land on different device pixels. measuring the container's
  // real viewport position via getBoundingClientRect and rounding the ABSOLUTE target position
  // (not just the offset within the row) is what actually guarantees every tick lands on a pixel
  // boundary regardless of where its row happens to sit on the page.
  const containerRef = useRef<HTMLDivElement>(null)
  const [geometry, setGeometry] = useState<{ width: number; left: number } | null>(null)
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => {
      const rect = el.getBoundingClientRect()
      setGeometry({ width: rect.width, left: rect.left })
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [])

  const span = domainMax - domainMin || 1
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - domainMin) / span) * 100))
  const barColor = selected ? color : INK
  const fadedOpacity = selected ? 0.32 : 0.13
  const solidOpacity = selected ? 1 : 0.4
  const caretColor = selected ? INK : 'rgba(22,21,15,.5)'
  // what the p25-p75 box actually reads as on screen, opacity included - darkening the raw
  // (often near-black) source colour instead would clip straight to black for unselected rows
  const effectiveBarColor = selected ? barColor : mix(INK, PANEL, solidOpacity)
  const medianColor = tickColor(effectiveBarColor)
  const rowHeight = compact ? 16 : 18
  const baselineTop = compact ? 7.5 : 8.5
  const fadedTop = compact ? 5.5 : 6.5
  const solidTop = compact ? 3.5 : 4.5
  const caretSize = compact ? 4 : 5

  // both markers keep fixed, full geometry on every row - the caret always rests right on the bar
  // and the tick always keeps its full overhang top and bottom. earlier attempts special-cased the
  // rare row where the median and the projected estimate coincide (lifting the caret, or shrinking
  // the tick's overhang) and both read worse than just letting the two crisp, thin marks sit close
  // together there - they're no longer the heavy blurred shapes that originally made touching look
  // like a rendering glitch.
  const caretTop = 0
  const solidHeight = 9
  const overhang = 2.5
  const medianTop = solidTop - overhang
  const medianHeight = solidHeight + overhang * 2
  const medianPct = dist ? pct(dist.median) : 0
  const medianWidth = 1
  // round the ABSOLUTE viewport target to a whole pixel, then convert back to a container-relative
  // offset - the result is very likely a non-integer CSS number, but that's fine: it's exactly the
  // fractional adjustment needed to cancel out the container's own fractional viewport position, so
  // the two sum to a true device pixel boundary instead of each independently rounding to one
  const medianLeftPx = geometry
    ? Math.round(geometry.left + (medianPct / 100) * geometry.width) - geometry.left
    : null

  return (
    <div ref={containerRef} style={{ position: 'relative', height: rowHeight }}>
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
              position: 'absolute', left: `${pct(dist.p25)}%`, top: solidTop, height: solidHeight,
              minWidth: compact ? 5 : 6, width: `${pct(dist.p75) - pct(dist.p25)}%`,
              background: barColor, opacity: solidOpacity,
            }}
          />
          {medianLeftPx !== null ? (
            // svg + shape-rendering="crispEdges" is the standard technique for a hairline that
            // stays exactly 1 device pixel regardless of where its fractional CSS position falls -
            // a plain div background blurs across two pixels at the wrong sub-pixel offset even
            // when the numbers going in are "correct", which is what made this worth doing properly
            <svg
              style={{ position: 'absolute', left: medianLeftPx, top: medianTop, overflow: 'visible' }}
              width={medianWidth} height={medianHeight}
            >
              <rect x={0} y={0} width={medianWidth} height={medianHeight} fill={medianColor} shapeRendering="crispEdges" />
            </svg>
          ) : (
            <div
              style={{
                position: 'absolute', left: `${medianPct}%`, transform: 'translateX(-50%)', top: medianTop,
                width: medianWidth, height: medianHeight, background: medianColor,
              }}
            />
          )}
        </>
      )}
      <div
        style={{
          position: 'absolute', left: `${pct(points)}%`, top: caretTop, marginLeft: -3,
          width: 0, height: 0,
          borderLeft: '3px solid transparent', borderRight: '3px solid transparent',
          borderTop: `${caretSize}px solid ${caretColor}`,
        }}
      />
    </div>
  )
}
