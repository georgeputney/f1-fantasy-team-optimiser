import { useState } from 'react'
import { Select } from './Select'
import { CARD, FAINT, GREEN, GREEN_TINT, INK, LINE_SOFT, LINE_STR, MUTED, MUTED2 } from './theme'
import type { BreakdownResponse, BreakdownRow } from './api'

interface Props {
  data: BreakdownResponse
  onRoundChange: (round: number) => void
}

const GRID = '2px 132px repeat(8,1fr)'

type SortDir = 'asc' | 'desc'
type SortKey = keyof Pick<BreakdownRow,
  'name' | 'quali_position' | 'finish_position' | 'positions_gained' | 'overtakes' | 'prob_fl' | 'prob_dotd' | 'dnf_prob' | 'expected_points'
>

// P1 reads as "best" ascending, so quali/finish default that way on first click; everything else
// defaults to biggest-first, which is what you want the first time you click e.g. "DNF risk"
const COLUMNS: { key: SortKey; label: string; defaultDir: SortDir }[] = [
  { key: 'quali_position', label: 'Quali', defaultDir: 'asc' },
  { key: 'finish_position', label: 'Finish', defaultDir: 'asc' },
  { key: 'positions_gained', label: 'Pos. gained', defaultDir: 'desc' },
  { key: 'overtakes', label: 'Overtakes', defaultDir: 'desc' },
  { key: 'prob_fl', label: 'Fastest lap', defaultDir: 'desc' },
  { key: 'prob_dotd', label: 'DOTD', defaultDir: 'desc' },
  { key: 'dnf_prob', label: 'DNF risk', defaultDir: 'desc' },
  { key: 'expected_points', label: 'Expected', defaultDir: 'desc' },
]

export function DriverBreakdown({ data, onRoundChange }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('expected_points')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const roundOptions = data.available_rounds.map((r) => ({ id: String(r), name: `Round ${r}` }))

  const toggleSort = (key: SortKey, defaultDir: SortDir) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(defaultDir)
    }
  }

  const rows = [...data.rows].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number)
    return sortDir === 'asc' ? cmp : -cmp
  })

  return (
    <div id="breakdown" style={{ padding: '40px 44px 12px', background: CARD, scrollMarginTop: 66 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 32, marginBottom: 22 }}>
        <div>
          <h3 style={{ margin: '0 0 6px', font: '500 22px/1 Archivo,sans-serif', letterSpacing: '-.015em', color: INK }}>
            Driver breakdown
          </h3>
          <p style={{ margin: 0, font: '400 14.5px/1.6 Archivo,sans-serif', color: MUTED, maxWidth: '60em' }}>
            Where each expected-points total comes from. Quali and finish are the model's median predicted
            positions, so positions gained is the difference between them; overtakes, fastest lap and driver
            of the day are per-race averages, and DNF risk is the share of simulations in which the car fails
            to finish.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 20, alignItems: 'baseline', flexShrink: 0, paddingTop: 2 }}>
          <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>Season {data.season}</span>
          <Select
            value={String(data.round)}
            options={roundOptions}
            placeholder="Round"
            onChange={(id) => onRoundChange(Number(id))}
            fontSize={13}
            fitContent
          />
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 900 }}>
          <div style={{ display: 'grid', gridTemplateColumns: GRID, columnGap: 14, alignItems: 'center', padding: '0 12px 9px', borderBottom: `1px solid ${LINE_STR}` }}>
            <span />
            <span
              onClick={() => toggleSort('name', 'asc')}
              style={{ font: '400 11.5px/1 Archivo,sans-serif', color: sortKey === 'name' ? INK : FAINT, cursor: 'pointer', userSelect: 'none' }}
            >
              Driver{sortKey === 'name' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
            </span>
            {COLUMNS.map((c) => (
              <span
                key={c.key}
                onClick={() => toggleSort(c.key, c.defaultDir)}
                style={{
                  font: '400 11.5px/1.3 Archivo,sans-serif', color: sortKey === c.key ? INK : FAINT,
                  textAlign: 'right', cursor: 'pointer', userSelect: 'none',
                }}
              >
                {c.label}{sortKey === c.key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
              </span>
            ))}
          </div>

          {rows.map((row, i) => (
            <div
              key={row.id}
              style={{
                display: 'grid', gridTemplateColumns: GRID, columnGap: 14, alignItems: 'center', padding: '13px 12px',
                borderBottom: i === rows.length - 1 ? 'none' : `1px solid ${LINE_SOFT}`,
                background: row.selected ? GREEN_TINT : 'transparent',
              }}
            >
              <span style={{ height: 20, background: row.color }} />
              <span style={{ font: `${row.selected ? 500 : 400} 15px/1 Archivo,sans-serif`, color: INK }}>{row.name}</span>
              <span style={{ textAlign: 'right', font: `${row.selected ? 500 : 400} 14px/1 Archivo,sans-serif`, color: row.selected ? INK : MUTED }}>
                P{row.quali_position}
              </span>
              <span style={{ textAlign: 'right', font: `${row.selected ? 500 : 400} 14px/1 Archivo,sans-serif`, color: row.selected ? INK : MUTED }}>
                P{row.finish_position}
              </span>
              <span style={{ textAlign: 'right', font: '400 14px/1 Archivo,sans-serif', color: row.positions_gained > 0 ? GREEN : row.positions_gained < 0 ? '#a8412a' : MUTED }}>
                {row.positions_gained > 0 ? '+' : ''}{row.positions_gained}
              </span>
              <span style={{ textAlign: 'right', font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{row.overtakes.toFixed(1)}</span>
              <span style={{ textAlign: 'right', font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{Math.round(row.prob_fl * 100)}%</span>
              <span style={{ textAlign: 'right', font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{Math.round(row.prob_dotd * 100)}%</span>
              <span style={{ textAlign: 'right', font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{Math.round(row.dnf_prob * 100)}%</span>
              <span style={{ textAlign: 'right', font: `500 14px/1 Archivo,sans-serif`, color: INK }}>{row.expected_points.toFixed(1)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
