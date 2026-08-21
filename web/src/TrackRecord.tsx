import { Fragment } from 'react'
import { CARD, FAINT, GREEN, INK, LINE, LINE_MED, LINE_STR, MUTED, MUTED2, REDNEG } from './theme'
import type { TrackRecordResponse } from './api'

interface Props {
  data: TrackRecordResponse
}

const BAR_GREY = 'rgba(22,21,15,.2)'

// desktop-only per the brief - mobile drops the track record section entirely
export function TrackRecord({ data }: Props) {
  if (!data.available) return null

  const stats: [string, string, string][] = [
    [`Model points, R1–R${data.last_round}`, data.model_total.toLocaleString(), INK],
    ['Oracle (best possible)', data.oracle_total.toLocaleString(), INK],
    ['Average % of oracle', `${data.average_pct.toFixed(0)}%`, GREEN],
    ['Best round', `R${data.best_round} · ${data.best_pct.toFixed(0)}%`, INK],
    ['Worst round', `R${data.worst_round} · ${data.worst_pct.toFixed(0)}%`, INK],
  ]

  return (
    <div id="record" style={{ padding: '40px 44px 48px', borderTop: `1px solid ${LINE}`, background: CARD, scrollMarginTop: 66 }}>
      <h3 style={{ margin: '0 0 8px', font: '500 22px/1 Archivo,sans-serif', letterSpacing: '-.015em', color: INK }}>
        Track record
      </h3>
      <div style={{ marginBottom: 26 }}>
        <p style={{ margin: 0, font: '400 14.5px/1.6 Archivo,sans-serif', color: MUTED }}>
          How the model has actually done this season, measured against the oracle team (the best possible
          lineup for each race, chosen with hindsight). Every round is shown, including the ones it got wrong.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 56, alignItems: 'start' }}>
        <div>
          <div
            style={{
              display: 'grid', gridTemplateColumns: '1fr auto', columnGap: 20, rowGap: 14, alignItems: 'baseline',
              borderTop: `1px solid ${LINE_MED}`, paddingTop: 14,
            }}
          >
            {stats.map(([label, value, color]) => (
              <Fragment key={label}>
                <span style={{ font: '400 14.5px/1 Archivo,sans-serif', color: MUTED }}>{label}</span>
                <span style={{ font: '500 15px/1 Archivo,sans-serif', color, textAlign: 'right' }}>{value}</span>
              </Fragment>
            ))}
          </div>
        </div>

        <div>
          <p style={{ margin: '0 0 14px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>
            % of oracle achieved by round · average marked
          </p>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'flex-end', gap: 10, height: 170, borderBottom: `1px solid ${LINE_STR}` }}>
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: `${data.average_pct}%`, height: 1, background: 'rgba(47,106,83,.55)' }} />
            {data.rounds.map((r) => (
              <div
                key={r.round}
                title={`R${r.round}: ${r.pct.toFixed(0)}%`}
                style={{ flex: 1, height: `${r.pct}%`, background: r.is_worst ? REDNEG : BAR_GREY }}
              />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
            {data.rounds.map((r) => (
              <span key={r.round} style={{ flex: 1, font: '400 11.5px/1 Archivo,sans-serif', color: FAINT, textAlign: 'center' }}>
                {r.round}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
