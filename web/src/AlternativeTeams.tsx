import {
  FAINT, GREEN, GREEN_TINT, INK, LINE_MED, MONO, MUTED2, RED_BORDER, RED_TINT,
} from './theme'
import type { AlternativeTeam } from './api'

interface Props {
  teams: AlternativeTeam[]
}

const GRID = '26px 1fr 210px 64px 66px'

function Chip({ code, color, differs, tag }: { code: string; color: string; differs: boolean; tag?: string }) {
  return (
    <span
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 8px 4px 6px',
        background: differs ? RED_TINT : 'transparent',
        border: `1px solid ${differs ? RED_BORDER : 'rgba(22,21,15,.12)'}`,
        borderRadius: 2,
      }}
    >
      <span style={{ width: 3, height: 13, background: color, borderRadius: 1 }} />
      <span style={{ font: `500 12.5px/1 ${MONO}`, letterSpacing: '.04em', color: differs ? '#a8412a' : '#33312b' }}>
        {code}
      </span>
      {tag && <span style={{ font: `500 10px/1 ${MONO}`, color: MUTED2 }}>{tag}</span>}
    </span>
  )
}

export function AlternativeTeams({ teams }: Props) {
  return (
    <div>
      <p style={{ margin: '0 0 6px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Alternative teams</p>
      <p style={{ margin: '0 0 16px', font: '400 13px/1.55 Archivo,sans-serif', color: MUTED2 }}>
        Every lineup the optimiser ranked next, by expected points. Row 1 is the recommended team; the
        bar on each chip is the constructor colour,{' '}
        <span style={{ color: '#a8412a' }}>red-outlined chips</span> are picks that differ from row 1,
        and ×2 marks the captain.
      </p>
      <div
        style={{
          display: 'grid', gridTemplateColumns: GRID, columnGap: 16, alignItems: 'baseline',
          padding: '0 0 8px', borderBottom: `1px solid ${LINE_MED}`,
        }}
      >
        <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: FAINT, paddingLeft: 6 }}>#</span>
        <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Drivers</span>
        <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Constructors</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT, paddingRight: 6 }}>Pts</span>
        <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT, paddingRight: 6 }}>Spend</span>
      </div>

      {teams.map((t) => (
        <div
          key={t.rank}
          style={{
            display: 'grid', gridTemplateColumns: GRID, columnGap: 16, alignItems: 'center',
            padding: '11px 0',
            borderBottom: t.rank === teams.length ? 'none' : '1px solid rgba(22,21,15,.07)',
            background: t.rank === 1 ? GREEN_TINT : 'transparent',
          }}
        >
          <span style={{ font: `400 12px/1 ${MONO}`, color: t.rank === 1 ? GREEN : FAINT, paddingLeft: 6 }}>{t.rank}</span>
          <span style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {t.drivers.map((d) => (
              <Chip
                key={d.id}
                code={d.fia_code}
                color={d.color}
                differs={t.rank !== 1 && d.differs}
                tag={d.captain ? '×2' : undefined}
              />
            ))}
          </span>
          <span style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {t.constructors.map((c) => (
              <Chip key={c.id} code={c.code} color={c.color} differs={t.rank !== 1 && c.differs} />
            ))}
          </span>
          <span style={{ textAlign: 'right', paddingRight: 6 }}>
            <div style={{ font: '500 14px/1 Archivo,sans-serif', color: INK }}>{t.total_points.toFixed(1)}</div>
            {t.rank !== 1 && (
              <div style={{ font: `400 10.5px/1 ${MONO}`, color: FAINT, marginTop: 3 }}>
                −{Math.abs(t.gap_to_best).toFixed(1)}
              </div>
            )}
          </span>
          <span style={{ textAlign: 'right', font: '400 13px/1 Archivo,sans-serif', color: MUTED2, paddingRight: 6 }}>
            £{t.spend.toFixed(1)}M
          </span>
        </div>
      ))}
    </div>
  )
}
