import { GREEN, INK, LINE_MED, LINE_SOFT, MONO, MUTED, MUTED2, PANEL_MOBILE } from './theme'
import { MobileSheet } from './MobileSheet'
import type { ValueResponse } from './api'

interface Props {
  data: ValueResponse
}

const WBAR_GREY = 'rgba(22,21,15,.16)'

// mobile drops the value-map scatter (too dense for the viewport) and keeps the waterfall plus
// the price-move table, matching the design's #5a-m variant - a MobileSheet like every other
// collapsible mobile section, badged with the projected next-round budget
export function ValueBuyingPowerMobile({ data }: Props) {
  const { waterfall: w, price_moves } = data
  const wmax = Math.max(w.team_value, w.budget_next, 1)
  const rises = w.expected_rises

  const gainNote = rises >= 0
    ? `This lineup is expected to gain £${rises.toFixed(1)}M by round ${data.next_round}.`
    : `This lineup is expected to lose £${Math.abs(rises).toFixed(1)}M by round ${data.next_round}.`

  return (
    <div style={{ padding: '0 22px 22px', background: PANEL_MOBILE }}>
    <MobileSheet title="Value & buying power" count={`£${w.budget_next.toFixed(1)}M`}>
      <p style={{ margin: '0 0 20px', font: '400 12.5px/1.55 Archivo,sans-serif', color: MUTED }}>
        Budget is cash plus team value, so each pick is also an investment. {gainNote}
      </p>

      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, height: 120, marginBottom: 8 }}>
        {[
          { label: 'Team value', value: w.team_value, color: WBAR_GREY, text: INK },
          { label: 'Cash', value: w.cash, color: WBAR_GREY, text: INK },
          { label: 'Rises', value: rises, color: GREEN, text: GREEN, rises: true },
          { label: `R${data.next_round}`, value: w.budget_next, color: 'rgba(47,106,83,.3)', text: INK, borderTop: `2px solid ${GREEN}` },
        ].map((bar) => (
          <div key={bar.label} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: '100%' }}>
            <span style={{ font: '500 14px/1 Archivo,sans-serif', color: bar.text, marginBottom: 6 }}>{bar.value.toFixed(1)}</span>
            <div
              style={{
                height: `${bar.rises ? Math.abs(bar.value) / wmax * 100 + 2 : Math.max(Math.max(bar.value, 0) / wmax * 100, 3)}%`,
                background: bar.color, borderTop: bar.borderTop,
              }}
            />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 10, borderTop: `1px solid ${LINE_MED}`, paddingTop: 8 }}>
        <span style={{ flex: 1, font: '400 10.5px/1.3 Archivo,sans-serif', color: MUTED2 }}>Team value</span>
        <span style={{ flex: 1, font: '400 10.5px/1.3 Archivo,sans-serif', color: MUTED2 }}>Cash</span>
        <span style={{ flex: 1, font: '400 10.5px/1.3 Archivo,sans-serif', color: GREEN }}>Rises</span>
        <span style={{ flex: 1, font: '400 10.5px/1.3 Archivo,sans-serif', color: MUTED2 }}>R{data.next_round}</span>
      </div>

      <p style={{ margin: '24px 0 10px', font: '500 12px/1 Archivo,sans-serif', color: MUTED2 }}>
        Expected price move · your assets
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '3px 1fr 46px 38px 40px', columnGap: 8, alignItems: 'baseline', padding: '0 0 7px' }}>
        <span />
        <span style={{ font: '400 11px/1 Archivo,sans-serif', color: MUTED2 }}>Asset</span>
        <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: MUTED2 }}>Price</span>
        <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: MUTED2 }}>PPM</span>
        <span style={{ textAlign: 'right', font: '400 11px/1 Archivo,sans-serif', color: MUTED2 }}>Move</span>
      </div>
      <div style={{ borderTop: `1px solid ${LINE_MED}` }}>
        {price_moves.map((row, i) => (
          <div
            key={row.id}
            style={{
              display: 'grid', gridTemplateColumns: '3px 1fr 46px 38px 40px', columnGap: 8, alignItems: 'center', padding: '10px 0',
              borderBottom: i === price_moves.length - 1 ? 'none'
                : row.is_driver && !price_moves[i + 1].is_driver ? `1px solid ${LINE_MED}`
                : `1px solid ${LINE_SOFT}`,
            }}
          >
            <span style={{ height: 16, background: row.color }} />
            <span style={{ font: '400 13.5px/1 Archivo,sans-serif', color: INK }}>{row.name}</span>
            <span style={{ textAlign: 'right', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED }}>£{row.price.toFixed(1)}</span>
            <span style={{ textAlign: 'right', font: `400 12.5px/1 ${MONO}`, color: MUTED2 }}>{row.ppm.toFixed(2)}</span>
            <span style={{ textAlign: 'right', font: '500 12.5px/1 Archivo,sans-serif', color: row.move == null ? MUTED2 : row.move > 0 ? GREEN : row.move < 0 ? '#a8412a' : MUTED2 }}>
              {row.move == null ? '-' : `${row.move > 0 ? '+' : ''}${row.move.toFixed(1)}`}
            </span>
          </div>
        ))}
      </div>
    </MobileSheet>
    </div>
  )
}
