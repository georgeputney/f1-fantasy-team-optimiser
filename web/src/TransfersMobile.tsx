import { Fragment } from 'react'
import { GREEN, INK, MUTED, MUTED2, PANEL_MOBILE } from './theme'
import type { Transfers as TransfersData } from './api'

interface Props {
  transfers: TransfersData
}

export function TransfersMobile({ transfers }: Props) {
  if (!transfers.has_state || transfers.rows.length === 0) return null

  const footer = transfers.paid
    ? `${transfers.free} free, ${transfers.paid} paid at -10`
    : `${transfers.rows.length} transfer${transfers.rows.length !== 1 ? 's' : ''}, no penalty`

  return (
    <div style={{ padding: '24px 22px 26px', background: PANEL_MOBILE, borderTop: '1px solid rgba(22,21,15,.14)' }}>
      <p style={{ margin: '0 0 16px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Suggested transfers</p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 20px 1fr auto', columnGap: 10, rowGap: 14, alignItems: 'baseline' }}>
        {transfers.rows.map((r, i) => (
          <Fragment key={i}>
            <span style={{ font: '400 15px/1 Archivo,sans-serif', color: MUTED2 }}>{r.out_name}</span>
            <span style={{ font: '400 14px/1 Archivo,sans-serif', color: GREEN, textAlign: 'center' }}>→</span>
            <span style={{ font: '500 15px/1 Archivo,sans-serif', color: INK }}>{r.in_name}</span>
            <span style={{ font: '400 13px/1 Archivo,sans-serif', color: r.delta >= 0 ? GREEN : '#a8412a', textAlign: 'right' }}>
              {r.delta >= 0 ? '+' : ''}{r.delta.toFixed(1)}
            </span>
          </Fragment>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 14, marginTop: 14, borderTop: '1px solid rgba(22,21,15,.14)' }}>
        <span style={{ font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{footer}</span>
        <span style={{ font: '500 14px/1 Archivo,sans-serif', color: INK }}>
          Net {transfers.net >= 0 ? '+' : ''}{transfers.net.toFixed(1)}
        </span>
      </div>
    </div>
  )
}
