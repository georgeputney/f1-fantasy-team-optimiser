import type { ReactNode } from 'react'
import { CARD, FAINT, GREEN, INK, LINE_MED, LINE_SOFT, MUTED2 } from './theme'
import type { LineupRow } from './api'

interface Props {
  lineup: LineupRow[]
}

const GRID = '3px 1fr 62px 74px'

function Badge({ children, tone }: { children: ReactNode; tone: 'captain' | 'in' }) {
  return (
    <span
      style={{
        display: 'inline-flex', alignItems: 'center', height: 18, padding: '0 7px',
        font: '500 11px/1 Archivo,sans-serif', letterSpacing: '.04em',
        background: tone === 'captain' ? GREEN : 'transparent',
        border: tone === 'in' ? '1px solid rgba(22,21,15,.28)' : undefined,
        color: tone === 'captain' ? '#f2efe7' : '#57534b',
      }}
    >
      {children}
    </span>
  )
}

function Column({ label, rows }: { label: string; rows: LineupRow[] }) {
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: GRID, columnGap: 14, alignItems: 'baseline', padding: '0 0 9px', borderBottom: `1px solid ${LINE_MED}` }}>
        <span />
        <span style={{ font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>{label}</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Pts</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Price</span>
      </div>
      {rows.map((row, i) => (
        <div
          key={row.id}
          style={{
            display: 'grid', gridTemplateColumns: GRID, columnGap: 14, alignItems: 'center', padding: '13px 0',
            borderBottom: i === rows.length - 1 ? 'none' : `1px solid ${LINE_SOFT}`,
          }}
        >
          <span style={{ height: 20, background: row.color }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap' }}>
            <span style={{ font: '500 17px/1 Archivo,sans-serif', letterSpacing: '-.01em', color: INK }}>{row.name}</span>
            {row.captain && <Badge tone="captain">×2</Badge>}
            {row.in && <Badge tone="in">NEW</Badge>}
            {row.captain && (
              <span style={{ marginLeft: 2, font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>
                ({row.points.toFixed(1)} ×2)
              </span>
            )}
          </div>
          <span style={{ textAlign: 'right', font: '500 17px/1 Archivo,sans-serif', color: INK }}>
            {(row.doubled_points ?? row.points).toFixed(1)}
          </span>
          <span style={{ textAlign: 'right', font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>
            £{row.price.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  )
}

export function Lineup({ lineup }: Props) {
  const drivers = lineup.filter((r) => r.is_driver)
  const constructors = lineup.filter((r) => !r.is_driver)

  return (
    <div style={{ padding: '0 44px 44px', display: 'grid', gridTemplateColumns: '1.25fr 1fr', columnGap: 64, alignItems: 'start', background: CARD }}>
      <Column label="Five drivers" rows={drivers} />
      <Column label="Two constructors" rows={constructors} />
    </div>
  )
}
