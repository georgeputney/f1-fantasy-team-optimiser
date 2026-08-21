import { Fragment } from 'react'
import { CARD, GREEN, INK, LINE, MUTED, MUTED2 } from './theme'
import type { Transfers as TransfersData } from './api'

interface Props {
  transfers: TransfersData
}

export function Transfers({ transfers }: Props) {
  if (!transfers.has_state || transfers.rows.length === 0) return null

  const footer = transfers.paid
    ? `${transfers.free} free, ${transfers.paid} paid at -10 pts`
    : `${transfers.rows.length} transfer${transfers.rows.length !== 1 ? 's' : ''}, no penalty`

  return (
    <div style={{ padding: '0 44px 30px', display: 'grid', gridTemplateColumns: '1.25fr 1fr', columnGap: 64, alignItems: 'start', background: CARD }}>
      <div style={{ maxWidth: 620 }}>
        <p style={{ margin: '0 0 18px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Suggested transfers</p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 22px 1fr auto', columnGap: 12, rowGap: 14, alignItems: 'baseline', borderTop: `1px solid ${LINE}`, paddingTop: 14 }}>
          {transfers.rows.map((r, i) => (
            <Fragment key={i}>
              <span style={{ font: '400 16px/1 Archivo,sans-serif', color: MUTED2 }}>{r.out_name}</span>
              <span style={{ font: '400 15px/1 Archivo,sans-serif', color: GREEN, textAlign: 'center' }}>→</span>
              <span style={{ font: '500 16px/1 Archivo,sans-serif', color: INK }}>{r.in_name}</span>
              <span style={{ font: '400 14px/1 Archivo,sans-serif', color: r.delta >= 0 ? GREEN : '#a8412a', textAlign: 'right' }}>
                {r.delta >= 0 ? '+' : ''}{r.delta.toFixed(1)}
              </span>
            </Fragment>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 0 0', marginTop: 6, borderTop: `1px solid ${LINE}` }}>
          <span style={{ font: '400 14.5px/1 Archivo,sans-serif', color: MUTED }}>{footer}</span>
          <span style={{ font: '500 14.5px/1 Archivo,sans-serif', color: INK }}>
            Net {transfers.net >= 0 ? '+' : ''}{transfers.net.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  )
}
