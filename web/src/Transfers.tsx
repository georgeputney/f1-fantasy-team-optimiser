import { Fragment } from 'react'
import { CARD, FAINT, GREEN, INK, LINE, MUTED, MUTED2, REDNEG } from './theme'
import { deltaTone, signedPoints, signedValue } from './format'
import type { Transfers as TransfersData } from './api'

interface Props {
  transfers: TransfersData
}

export function Transfers({ transfers }: Props) {
  if (!transfers.has_state || transfers.rows.length === 0) return null

  // typeof, not `!== null` - an API older than this bundle sends no net_value at all, and
  // `undefined !== null` would render the whole column as £NaN instead of hiding it
  const netValue = typeof transfers.net_value === 'number' ? transfers.net_value : null
  const grid = netValue !== null ? '1fr 22px 1fr 62px 66px' : '1fr 22px 1fr auto'

  const footer = transfers.paid
    ? `${transfers.free} free, ${transfers.paid} paid at -10 pts`
    : `${transfers.rows.length} transfer${transfers.rows.length !== 1 ? 's' : ''}, no penalty`

  return (
    <div style={{ padding: '0 44px 30px', display: 'grid', gridTemplateColumns: '1.25fr 1fr', columnGap: 64, alignItems: 'start', background: CARD }}>
      <div style={{ maxWidth: 620 }}>
        <p style={{ margin: '0 0 18px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Suggested transfers</p>
        <div style={{ display: 'grid', gridTemplateColumns: grid, columnGap: 12, rowGap: 14, alignItems: 'baseline', borderTop: `1px solid ${LINE}`, paddingTop: 14 }}>
          {netValue !== null && (
            <>
              <span />
              <span />
              <span />
              <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Pts</span>
              <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Value</span>
            </>
          )}
          {transfers.rows.map((r, i) => (
            <Fragment key={i}>
              <span style={{ font: '400 16px/1 Archivo,sans-serif', color: MUTED2 }}>{r.out_name}</span>
              <span style={{ font: '400 15px/1 Archivo,sans-serif', color: GREEN, textAlign: 'center' }}>→</span>
              <span style={{ font: '500 16px/1 Archivo,sans-serif', color: INK }}>{r.in_name}</span>
              <span style={{ font: '400 14px/1 Archivo,sans-serif', color: r.delta >= 0 ? GREEN : REDNEG, textAlign: 'right' }}>
                {signedPoints(r.delta)}
              </span>
              {netValue !== null && (
                <span style={{ font: '400 14px/1 Archivo,sans-serif', color: deltaTone(r.value_delta), textAlign: 'right' }}>
                  {signedValue(r.value_delta)}
                </span>
              )}
            </Fragment>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, padding: '16px 0 0', marginTop: 6, borderTop: `1px solid ${LINE}` }}>
          <span style={{ font: '400 14.5px/1 Archivo,sans-serif', color: MUTED }}>{footer}</span>
          <span style={{ display: 'flex', gap: 14, font: '500 14.5px/1 Archivo,sans-serif', color: INK }}>
            <span>Net {signedPoints(transfers.net)} pts</span>
            {netValue !== null && <span style={{ color: deltaTone(netValue) }}>{signedValue(netValue)} value</span>}
          </span>
        </div>
      </div>
    </div>
  )
}
