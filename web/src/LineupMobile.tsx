import { CARD, INK, LINE_MED, MUTED2 } from './theme'
import type { LineupRow } from './api'

interface Props {
  lineup: LineupRow[]
}

const GRID = '2px 1fr auto 58px'

export function LineupMobile({ lineup }: Props) {
  return (
    <div style={{ padding: '0 22px 4px', background: CARD }}>
      {lineup.map((row, i) => (
        <div
          key={row.id}
          style={{
            display: 'grid', gridTemplateColumns: GRID, columnGap: 12, alignItems: 'center', padding: '14px 0',
            minHeight: 48,
            borderBottom: i === lineup.length - 1 ? 'none'
              : row.is_driver && !lineup[i + 1].is_driver ? `1px solid ${LINE_MED}`
              : '1px solid rgba(22,21,15,.08)',
          }}
        >
          <span style={{ height: 20, background: row.color }} />
          <span style={{ display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap', minWidth: 0 }}>
            <span style={{ font: '500 16px/1.2 Archivo,sans-serif', color: INK }}>{row.name}</span>
            {row.captain && (
              <span style={{ display: 'inline-flex', alignItems: 'center', height: 18, padding: '0 7px', background: '#2f6a53', font: '500 11px/1 Archivo,sans-serif', letterSpacing: '.04em', color: '#f2efe7' }}>
                ×2
              </span>
            )}
            {row.in && (
              <span style={{ display: 'inline-flex', alignItems: 'center', height: 18, padding: '0 7px', border: '1px solid rgba(22,21,15,.28)', font: '500 11px/1 Archivo,sans-serif', letterSpacing: '.04em', color: '#57534b' }}>
                NEW
              </span>
            )}
            {row.captain && (
              <span style={{ marginLeft: 2, font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>
                ({row.points.toFixed(1)} ×2)
              </span>
            )}
          </span>
          <span style={{ font: '500 16px/1 Archivo,sans-serif', color: INK, textAlign: 'right' }}>
            {(row.doubled_points ?? row.points).toFixed(1)}
          </span>
          <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2, textAlign: 'right' }}>
            £{row.price.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  )
}
