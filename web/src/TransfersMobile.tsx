import { Fragment } from 'react'
import { FAINT, GREEN, INK, MUTED, MUTED2, PANEL_MOBILE, REDNEG } from './theme'
import { deltaTone, signedPoints, signedValue } from './format'
import type { Transfers as TransfersData } from './api'

interface Props {
  transfers: TransfersData
}

export function TransfersMobile({ transfers }: Props) {
  if (!transfers.has_state || transfers.rows.length === 0) return null

  // typeof, not `!== null` - an API older than this bundle sends no net_value at all, and
  // `undefined !== null` would render the whole column as £NaN instead of hiding it
  const netValue = typeof transfers.net_value === 'number' ? transfers.net_value : null
  const grid = netValue !== null ? '1fr 20px 1fr 48px 52px' : '1fr 20px 1fr auto'

  const footer = transfers.paid
    ? `${transfers.free} free, ${transfers.paid} paid at -10`
    : `${transfers.rows.length} transfer${transfers.rows.length !== 1 ? 's' : ''}, no penalty`

  return (
    <div style={{ padding: '24px 22px 26px', background: PANEL_MOBILE, borderTop: '1px solid rgba(22,21,15,.14)' }}>
      <p style={{ margin: '0 0 16px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Suggested transfers</p>
      <div style={{ display: 'grid', gridTemplateColumns: grid, columnGap: 10, rowGap: 14, alignItems: 'baseline' }}>
        {netValue !== null && (
          <>
            <span />
            <span />
            <span />
            <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: FAINT }}>Pts</span>
            <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: FAINT }}>Value</span>
          </>
        )}
        {transfers.rows.map((r, i) => (
          <Fragment key={i}>
            <span style={{ font: '400 15px/1 Archivo,sans-serif', color: MUTED2 }}>{r.out_name}</span>
            <span style={{ font: '400 14px/1 Archivo,sans-serif', color: GREEN, textAlign: 'center' }}>→</span>
            <span style={{ font: '500 15px/1 Archivo,sans-serif', color: INK }}>{r.in_name}</span>
            <span style={{ font: '400 13px/1 Archivo,sans-serif', color: r.delta >= 0 ? GREEN : REDNEG, textAlign: 'right' }}>
              {signedPoints(r.delta)}
            </span>
            {netValue !== null && (
              <span style={{ font: '400 13px/1 Archivo,sans-serif', color: deltaTone(r.value_delta), textAlign: 'right' }}>
                {signedValue(r.value_delta)}
              </span>
            )}
          </Fragment>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, paddingTop: 14, marginTop: 14, borderTop: '1px solid rgba(22,21,15,.14)' }}>
        <span style={{ font: '400 14px/1 Archivo,sans-serif', color: MUTED }}>{footer}</span>
        <span style={{ display: 'flex', gap: 10, font: '500 14px/1 Archivo,sans-serif', color: INK }}>
          <span>Net {signedPoints(transfers.net)}</span>
          {netValue !== null && <span style={{ color: deltaTone(netValue) }}>{signedValue(netValue)}</span>}
        </span>
      </div>
    </div>
  )
}
