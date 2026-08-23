import { CARD, GREEN, INK, LINE, MUTED2 } from './theme'

interface Props {
  round: number
  circuit: string
  status: string
}

const NAV_ITEMS: [string, string][] = [
  ['team', 'Team'],
  ['ladder', 'Ladder'],
  ['breakdown', 'Breakdown'],
  ['value', 'Value'],
  ['record', 'Record'],
]

// sticky so the nav stays reachable on a long page - anchors scroll to each section's own id,
// smooth-scroll comes from index.css's `html { scroll-behavior: smooth }`
export function Header({ round, circuit, status }: Props) {
  return (
    <div
      style={{
        padding: '22px 44px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: `1px solid ${LINE}`, position: 'sticky', top: 0, background: CARD, zIndex: 5,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 18 }}>
        <span style={{ font: '600 14px/1 Archivo,sans-serif', color: INK }}>Pitwall</span>
        <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>
          Round {round} · {circuit}
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 26 }}>
        {NAV_ITEMS.map(([id, label]) => (
          <a key={id} href={`#${id}`} style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2, textDecoration: 'none' }}>
            {label}
          </a>
        ))}
        <span style={{ display: 'flex', alignItems: 'center', gap: 7, paddingLeft: 8 }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: GREEN }} />
          <span style={{ font: '400 13px/1 Archivo,sans-serif', color: INK }}>{status}</span>
        </span>
      </div>
    </div>
  )
}
