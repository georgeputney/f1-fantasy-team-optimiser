import { DistributionBar } from './DistributionBar'
import { niceScale } from './ticks'
import { FAINT, GREEN, GREEN_TINT, INK, LINE_MED, MONO, MUTED } from './theme'
import type { LadderRow } from './api'

interface Props {
  label: string
  rows: LadderRow[]
}

const GRID = '16px 40px 1fr 38px'

export function LadderColumnMobile({ label, rows }: Props) {
  const maxP90 = Math.max(...rows.map((r) => r.distribution?.p90 ?? r.points), 1)
  const minP10 = Math.min(...rows.map((r) => r.distribution?.p10 ?? r.points), 0)
  const { ticks, domainMin, domainMax } = niceScale(minP10, maxP90)

  return (
    <div>
      <div
        style={{
          display: 'grid', gridTemplateColumns: GRID, columnGap: 8, alignItems: 'baseline',
          padding: '0 0 7px', borderBottom: `1px solid ${LINE_MED}`,
        }}
      >
        <span />
        <span style={{ font: '400 11px/1 Archivo,sans-serif', color: FAINT }}>{label}</span>
        <div style={{ position: 'relative', height: 11 }}>
          {ticks.map((t) => (
            <span
              key={t}
              style={{
                position: 'absolute', left: `${((t - domainMin) / (domainMax - domainMin)) * 100}%`, top: 0,
                transform: 'translateX(-50%)', font: `400 9.5px/1 ${MONO}`, color: FAINT,
              }}
            >
              {t}
            </span>
          ))}
        </div>
        <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: FAINT, paddingRight: 2 }}>Pts</span>
      </div>

      {rows.map((row, i) => (
        <div
          key={row.id}
          style={{
            display: 'grid', gridTemplateColumns: GRID, columnGap: 8, alignItems: 'center',
            padding: '9px 0',
            borderBottom: i === rows.length - 1 ? 'none' : '1px solid rgba(22,21,15,.07)',
            background: row.selected ? GREEN_TINT : 'transparent',
          }}
        >
          <span style={{ font: `400 10.5px/1 ${MONO}`, color: row.selected ? GREEN : FAINT, paddingLeft: 4 }}>
            {i + 1}
          </span>
          <span
            style={{
              font: `${row.selected ? 500 : 400} 12px/1.15 ${MONO}`, letterSpacing: '.04em',
              color: row.selected ? INK : MUTED,
            }}
          >
            {row.fia_code}
          </span>
          <DistributionBar
            dist={row.distribution}
            points={row.points}
            domainMin={domainMin}
            domainMax={domainMax}
            ticks={ticks}
            color={row.color}
            selected={row.selected}
            compact
          />
          <span
            style={{
              textAlign: 'right', font: `${row.selected ? 500 : 400} 13px/1 Archivo,sans-serif`,
              color: INK, paddingRight: 2,
            }}
          >
            {row.points.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  )
}
