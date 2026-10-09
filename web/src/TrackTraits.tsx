import { FAINT, INK, LINE, LINE_MED, LINE_SOFT, MUTED, MUTED2, PANEL } from './theme'
import type { TrackTrait, TrackTraitsResponse } from './api'

interface Props {
  data: TrackTraitsResponse
}

const AXIS = 'rgba(22,21,15,.18)'
const GAP_LINE = 'rgba(22,21,15,.55)'

interface TraitCopy {
  label: string
  // short call-out for a track well above or below the calendar - neither direction is "good" in itself
  high: string
  low: string
}

const COPY: Record<TrackTrait['key'], TraitCopy> = {
  overtakes: { label: 'Overtakes per driver', high: 'Overtaking pays', low: 'Grid position matters' },
  pole_to_win: { label: 'Pole converts to the win', high: 'Qualifying decides', low: 'Pole is no lock' },
  top3_to_podium: { label: 'Top-3 starters reach the podium', high: 'Front row holds', low: 'Podium is open' },
  fp3_to_quali: { label: 'FP3 top 3 qualify top 3', high: 'Practice reads true', low: 'Practice misleads' },
  safety_car: { label: 'Races with a safety car or VSC', high: 'Expect neutralisations', low: 'Tends to run green' },
  dnf: { label: 'DNF rate per driver', high: 'Attrition risk', low: 'Reliable finishes' },
}

// within this much of the calendar average (relative) reads as "typical" rather than a lean
const TYPICAL_BAND = 0.1

// overtakes per driver is a count with no natural ceiling, so it's plotted on an axis of twice this
// season's average race, putting the average mid-way. everything else is a rate on a 0-100% axis
const OVERTAKE_AXIS_MULT = 2

const NUMBER_WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

interface Row {
  t: TrackTrait
  value: string
  avg: string
  gap: string
  frac: number
  avgFrac: number
  lean: string | null
}

// U+2212 minus so a negative gap is the same width as the plus sign
function signed(x: number, digits: number) {
  const s = Math.abs(x).toFixed(digits)
  if (Number(s) === 0) return s
  return x > 0 ? `+${s}` : `−${s}`
}

function toRow(t: TrackTrait): Row | null {
  if (t.value == null || t.calendar_avg == null) return null

  const rel = t.calendar_avg === 0 ? 0 : (t.value - t.calendar_avg) / t.calendar_avg
  const lean = rel > TYPICAL_BAND ? COPY[t.key].high : rel < -TYPICAL_BAND ? COPY[t.key].low : null

  if (t.key === 'overtakes') {
    const axis = t.calendar_avg * OVERTAKE_AXIS_MULT
    return {
      t, lean,
      value: t.value.toFixed(1), avg: t.calendar_avg.toFixed(1),
      gap: signed(t.value - t.calendar_avg, 1),
      frac: t.value / axis, avgFrac: t.calendar_avg / axis,
    }
  }

  // the gap is the difference in percentage points, shown as % to match the columns beside it
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`
  return {
    t, lean,
    value: pct(t.value), avg: pct(t.calendar_avg),
    gap: `${signed((t.value - t.calendar_avg) * 100, 1)}%`,
    frac: t.value, avgFrac: t.calendar_avg,
  }
}

// dot is this track, tick is the calendar average, the line between them is the gap
function DotPlot({ frac, avgFrac }: { frac: number; avgFrac: number }) {
  const x = Math.min(Math.max(frac, 0), 1) * 100
  const a = Math.min(Math.max(avgFrac, 0), 1) * 100
  return (
    <div style={{ position: 'relative', height: 14 }}>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 7, height: 1, background: AXIS }} />
      <div style={{ position: 'absolute', left: `${Math.min(x, a)}%`, width: `${Math.abs(x - a)}%`, top: 6, height: 3, background: GAP_LINE }} />
      <div style={{ position: 'absolute', left: `${a}%`, top: 0, width: 1.5, height: 14, marginLeft: -0.75, background: GAP_LINE }} />
      <div style={{ position: 'absolute', left: `${x}%`, top: 2, width: 10, height: 10, marginLeft: -5, borderRadius: '50%', background: INK }} />
    </div>
  )
}

function intro(data: TrackTraitsResponse) {
  const kind = data.is_street_circuit ? 'a street circuit' : data.is_street_circuit === false ? 'a permanent circuit' : 'a circuit'
  const history = data.n_seasons > 0
    ? `${NUMBER_WORDS[data.n_seasons] ?? data.n_seasons} season${data.n_seasons === 1 ? '' : 's'} of history at ${data.circuit}, ${kind}`
    : `${data.circuit} is new to the calendar, so there's no history here yet`
  const sprint = data.is_sprint ? ' This is a sprint weekend, so there is no FP3.' : ''
  return `${history}.${sprint} The dot is this track, the tick is the season average across this year's calendar, and the line between them is the gap. For overtakes, the season average is this year's average race so far. Traits that differ meaningfully from the season average are called out on the right.`
}

// type scale matches the value section's price-move table so this reads as part of the same page
const HEAD = { font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }
const COLS = '230px minmax(0, 1fr) 80px 80px 80px 200px'
// the number columns are right-aligned, so the space between them is a column width minus a value's
// width (~44px at 13.5px) plus the column gap - Leans is left-aligned, so it's indented to match
const LEANS_INDENT = 36

// desktop-only - the mobile layout leaves this section out
export function TrackTraits({ data }: Props) {
  if (!data.available) return null

  const rows = data.traits.map(toRow).filter((r): r is Row => r !== null)

  return (
    <div id="track" style={{ padding: '40px 44px 44px', borderTop: `1px solid ${LINE}`, background: PANEL, scrollMarginTop: 66 }}>
      <h3 style={{ margin: '0 0 8px', font: '500 22px/1 Archivo,sans-serif', letterSpacing: '-.015em', color: INK }}>
        Track traits
      </h3>
      <p style={{ margin: '0 0 26px', font: '400 14.5px/1.6 Archivo,sans-serif', color: MUTED }}>{intro(data)}</p>

      <div
        style={{
          display: 'grid', gridTemplateColumns: COLS, columnGap: 20, alignItems: 'baseline',
          paddingBottom: 8, borderBottom: `1px solid ${LINE_MED}`,
        }}
      >
        <span style={HEAD}>Trait</span>
        <span />
        <span style={{ ...HEAD, textAlign: 'right' }}>This track</span>
        <span style={{ ...HEAD, textAlign: 'right' }} title="Average across this season's calendar">Avg</span>
        <span style={{ ...HEAD, textAlign: 'right' }}>Gap</span>
        <span style={{ ...HEAD, paddingLeft: LEANS_INDENT }}>Leans</span>
      </div>

      {rows.map((r, i) => {
        const muted = !r.lean
        return (
          <div
            key={r.t.key}
            style={{
              display: 'grid', gridTemplateColumns: COLS, columnGap: 20, alignItems: 'center', padding: '12px 0',
              borderBottom: i < rows.length - 1 ? `1px solid ${LINE_SOFT}` : 'none',
            }}
          >
            <span style={{ font: '400 14.5px/1 Archivo,sans-serif', color: INK }}>{COPY[r.t.key].label}</span>
            <DotPlot frac={r.frac} avgFrac={r.avgFrac} />
            <span style={{ font: '500 13.5px/1 Archivo,sans-serif', color: INK, textAlign: 'right' }}>{r.value}</span>
            <span style={{ font: '400 13.5px/1 Archivo,sans-serif', color: MUTED2, textAlign: 'right' }}>{r.avg}</span>
            <span style={{ font: '400 13.5px/1 Archivo,sans-serif', color: muted ? FAINT : MUTED, textAlign: 'right' }}>{r.gap}</span>
            <span style={{ paddingLeft: LEANS_INDENT, font: `${muted ? 400 : 500} 13.5px/1 Archivo,sans-serif`, color: muted ? FAINT : INK }}>
              {r.lean ?? 'Typical'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
