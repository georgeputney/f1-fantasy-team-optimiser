import { DistributionBar } from './DistributionBar'
import { niceScale } from './ticks'
import { FAINT, GREEN, GREEN_TINT, INK, LINE_MED, LINE_SOFT, MONO, MUTED, MUTED2 } from './theme'
import type { LadderRow } from './api'

interface Props {
  label: string
  rows: LadderRow[]
}

const GRID = '26px 128px 1fr 52px 104px 58px'

export function LadderColumn({ label, rows }: Props) {
  // each row's own points must be covered too, not just its distribution's p10/p90 - they're
  // different models and can disagree enough that points falls outside that row's simulated range,
  // which would otherwise clamp that row's caret to the edge instead of showing where it actually is
  const maxP90 = Math.max(...rows.map((r) => Math.max(r.distribution?.p90 ?? r.points, r.points)), 1)
  const minP10 = Math.min(...rows.map((r) => Math.min(r.distribution?.p10 ?? r.points, r.points)), 0)
  const { ticks, domainMin, domainMax } = niceScale(minP10, maxP90)

  return (
    <div>
      <div
        style={{
          display: 'grid', gridTemplateColumns: GRID, columnGap: 16, alignItems: 'baseline',
          padding: '0 0 8px', borderBottom: `1px solid ${LINE_MED}`,
        }}
      >
        <span />
        <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>{label}</span>
        <div style={{ position: 'relative', height: 12 }}>
          {ticks.map((t) => (
            <span
              key={t}
              style={{
                position: 'absolute', left: `${((t - domainMin) / (domainMax - domainMin)) * 100}%`, top: 0,
                font: `400 10px/1 ${MONO}`, color: FAINT, transform: 'translateX(-50%)',
              }}
            >
              {t}
            </span>
          ))}
        </div>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Pts</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>P10-P90</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Price</span>
      </div>

      {rows.map((row, i) => (
        <div
          key={row.id}
          style={{
            display: 'grid', gridTemplateColumns: GRID, columnGap: 16, alignItems: 'center',
            padding: '9px 0',
            borderBottom: i === rows.length - 1 ? 'none' : `1px solid ${LINE_SOFT}`,
            background: row.selected ? GREEN_TINT : 'transparent',
          }}
        >
          <span style={{ font: `400 11.5px/1 ${MONO}`, color: row.selected ? GREEN : FAINT, paddingLeft: 6 }}>
            {i + 1}
          </span>
          <span style={{ font: `${row.selected ? 500 : 400} 14px/1 Archivo,sans-serif`, color: row.selected ? INK : MUTED }}>
            {row.name}
          </span>
          <DistributionBar
            dist={row.distribution}
            points={row.points}
            domainMin={domainMin}
            domainMax={domainMax}
            ticks={ticks}
            color={row.color}
            selected={row.selected}
          />
          <span style={{ textAlign: 'right', font: `${row.selected ? 500 : 400} 14px/1 Archivo,sans-serif`, color: INK }}>
            {row.points.toFixed(1)}
          </span>
          <span style={{ textAlign: 'right', font: `400 11.5px/1 ${MONO}`, color: MUTED2 }}>
            {row.distribution ? `${row.distribution.p10.toFixed(1)} – ${row.distribution.p90.toFixed(1)}` : '-'}
          </span>
          <span style={{ textAlign: 'right', font: '400 13px/1 Archivo,sans-serif', color: MUTED2, paddingRight: 6 }}>
            £{row.price.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  )
}
