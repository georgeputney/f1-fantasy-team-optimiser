import { FAINT, GREEN, GREEN_TINT, INK, LINE_MED, MONO, MUTED, MUTED2, RED_BORDER, RED_TINT } from './theme'
import type { AlternativeTeam } from './api'

interface Props {
  teams: AlternativeTeam[]
}

function Chip({ code, color, differs, tag }: { code: string; color: string; differs: boolean; tag?: string }) {
  return (
    <span
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 6px 3px 4px',
        background: differs ? RED_TINT : 'transparent',
        border: `1px solid ${differs ? RED_BORDER : 'rgba(22,21,15,.12)'}`,
        borderRadius: 2,
      }}
    >
      <span style={{ width: 3, height: 11, background: color, borderRadius: 1 }} />
      <span style={{ font: `500 11px/1 ${MONO}`, letterSpacing: '.03em', color: differs ? '#a8412a' : '#33312b' }}>
        {code}
      </span>
      {tag && <span style={{ font: `500 9px/1 ${MONO}`, color: MUTED2 }}>{tag}</span>}
    </span>
  )
}

export function AlternativeTeamsMobile({ teams }: Props) {
  return (
    <div>
      <p style={{ margin: '0 0 14px', font: '400 12.5px/1.55 Archivo,sans-serif', color: MUTED }}>
        Ranked by expected points. Row 1 is the recommended team; <span style={{ color: '#a8412a' }}>red-outlined chips</span> differ
        from it, ×2 marks the captain, constructors sit after the divider.
      </p>
      <div style={{ borderTop: `1px solid ${LINE_MED}` }}>
        {teams.map((t) => (
          <div
            key={t.rank}
            style={{
              display: 'grid', gridTemplateColumns: '16px 1fr', columnGap: 8, padding: '11px 0',
              borderBottom: t.rank === teams.length ? 'none' : '1px solid rgba(22,21,15,.07)',
              background: t.rank === 1 ? GREEN_TINT : 'transparent',
            }}
          >
            <span style={{ font: `400 10.5px/1.6 ${MONO}`, color: t.rank === 1 ? GREEN : FAINT, paddingLeft: 4 }}>
              {t.rank}
            </span>
            <div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {t.drivers.map((d) => (
                  <Chip key={d.id} code={d.fia_code} color={d.color} differs={t.rank !== 1 && d.differs} tag={d.captain ? '×2' : undefined} />
                ))}
                {/* divider + both constructors grouped as one flex item, so they wrap to the next
                    line together instead of splitting mid-pair */}
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 1, height: 16, background: 'rgba(22,21,15,.14)', margin: '0 2px' }} />
                  {t.constructors.map((c) => (
                    <Chip key={c.id} code={c.code} color={c.color} differs={t.rank !== 1 && c.differs} />
                  ))}
                </span>
              </div>
              <p style={{ margin: '7px 0 0', textAlign: 'right', font: `400 11px/1 ${MONO}`, color: MUTED2 }}>
                <span style={{ color: INK, fontWeight: 500 }}>{t.total_points.toFixed(1)}</span>
                {t.rank !== 1 && <span style={{ color: FAINT }}> ({t.gap_to_best.toFixed(1)})</span>}
                {' '}pts · £{t.spend.toFixed(1)}M
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
