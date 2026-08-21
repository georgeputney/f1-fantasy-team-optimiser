import type { CSSProperties } from 'react'
import { CHART_H, CHART_W, hexToRgb, layoutValueMap, signedTick, xScale, yScale } from './valueMap'
import { FAINT, GREEN, INK, LINE_MED, LINE_SOFT, LINE_STR, MONO, MUTED, MUTED2, PANEL } from './theme'
import type { ValueResponse } from './api'

interface Props {
  data: ValueResponse
}

const WBAR_GREY = 'rgba(22,21,15,.16)'

function WaterfallBar({ value, heightPct, barColor, textColor, flex = 1, borderTop }: {
  value: number; heightPct: number; barColor: string; textColor: string; flex?: number; borderTop?: string
}) {
  return (
    <div style={{ flex, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: '100%' }}>
      <span style={{ font: '500 16px/1 Archivo,sans-serif', color: textColor, marginBottom: 8 }}>{value.toFixed(1)}</span>
      <div style={{ height: `${heightPct}%`, background: barColor, borderTop }} />
    </div>
  )
}

export function ValueBuyingPower({ data }: Props) {
  const { waterfall: w, price_moves, value_map } = data
  const wmax = Math.max(w.team_value, w.budget_next, 1)

  const lamNote = data.have_price_data
    ? `The optimiser also credits each pick with λ = ${data.price_lambda.toFixed(1)} points for every £1M it's expected to rise, so it can trade a little raw score now for more buying power later.`
    : 'Price projections are unavailable in this environment.'

  const xd = xScale(value_map.map((r) => r.ppm))
  const yd = yScale(value_map.map((r) => r.move ?? 0))
  const bubbles = layoutValueMap(value_map, xd, yd)

  const xPct = (v: number) => ((v - xd.domainMin) / (xd.domainMax - xd.domainMin || 1)) * 100
  const yPct = (v: number) => 50 - (v / (yd.domainMax || 1)) * 50

  return (
    <div id="value" style={{ padding: '40px 44px 44px', borderTop: '1px solid rgba(22,21,15,.14)', background: PANEL, scrollMarginTop: 66 }}>
      <h3 style={{ margin: '0 0 6px', font: '500 22px/1 Archivo,sans-serif', letterSpacing: '-.015em', color: INK }}>
        Value &amp; buying power
      </h3>
      <p style={{ margin: '0 0 26px', font: '400 14.5px/1.6 Archivo,sans-serif', color: MUTED }}>
        Budget is cash plus team value, so each pick is also an investment. {lamNote}
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 56 }}>
        <div>
          <p style={{ margin: '0 0 16px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>
            Where round {data.next_round}'s budget comes from
          </p>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 14, height: 150, marginBottom: 10 }}>
            <WaterfallBar value={w.team_value} heightPct={Math.max(w.team_value / wmax * 100, 3)} barColor={WBAR_GREY} textColor={INK} />
            <WaterfallBar value={w.cash} heightPct={Math.max(Math.max(w.cash, 0) / wmax * 100, 3)} barColor={WBAR_GREY} textColor={INK} />
            <WaterfallBar value={w.expected_rises} heightPct={Math.abs(w.expected_rises) / wmax * 100 + 2} barColor={GREEN} textColor={GREEN} />
            <WaterfallBar
              value={w.budget_next}
              heightPct={Math.max(w.budget_next / wmax * 100, 3)} barColor="rgba(47,106,83,.3)" textColor={INK}
              flex={1.15} borderTop={`2px solid ${GREEN}`}
            />
          </div>
          <div style={{ display: 'flex', gap: 14, borderTop: `1px solid ${LINE_MED}`, paddingTop: 9 }}>
            <span style={{ flex: 1, font: '400 11.5px/1.4 Archivo,sans-serif', color: MUTED2 }}>Team value</span>
            <span style={{ flex: 1, font: '400 11.5px/1.4 Archivo,sans-serif', color: MUTED2 }}>Cash</span>
            <span style={{ flex: 1, font: '400 11.5px/1.4 Archivo,sans-serif', color: GREEN }}>Expected rises</span>
            <span style={{ flex: 1.15, font: '400 11.5px/1.4 Archivo,sans-serif', color: INK }}>Budget at R{data.next_round}</span>
          </div>

          <p style={{ margin: '22px 0 14px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>
            Expected price move · your assets
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '3px 1fr 56px 52px 52px', columnGap: 12, alignItems: 'baseline', padding: '0 0 8px' }}>
            <span />
            <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Asset</span>
            <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Price</span>
            <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>PPM</span>
            <span style={{ textAlign: 'right', font: '400 11.5px/1 Archivo,sans-serif', color: FAINT }}>Move</span>
          </div>
          <div style={{ borderTop: `1px solid ${LINE_MED}` }}>
            {price_moves.map((row, i) => (
              <div
                key={row.id}
                style={{
                  display: 'grid', gridTemplateColumns: '3px 1fr 56px 52px 52px', columnGap: 12, alignItems: 'center', padding: '10px 0',
                  borderBottom: i === price_moves.length - 1 ? 'none'
                    : row.is_driver && !price_moves[i + 1].is_driver ? `1px solid ${LINE_MED}`
                    : `1px solid ${LINE_SOFT}`,
                }}
              >
                <span style={{ height: 18, background: row.color }} />
                <span style={{ font: '400 14.5px/1 Archivo,sans-serif', color: INK }}>{row.name}</span>
                <span style={{ textAlign: 'right', font: '400 13.5px/1 Archivo,sans-serif', color: MUTED }}>£{row.price.toFixed(1)}</span>
                <span style={{ textAlign: 'right', font: '400 13.5px/1 Archivo,sans-serif', color: MUTED2 }}>{row.ppm.toFixed(2)}</span>
                <span style={{ textAlign: 'right', font: '500 13.5px/1 Archivo,sans-serif', color: row.move == null ? MUTED2 : row.move > 0 ? GREEN : row.move < 0 ? '#a8412a' : MUTED2 }}>
                  {row.move == null ? '—' : `${row.move > 0 ? '+' : ''}${row.move.toFixed(1)}`}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <p style={{ margin: '0 0 8px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Value map</p>
          <p style={{ margin: '0 0 34px', font: '400 13px/1.55 Archivo,sans-serif', color: MUTED2 }}>
            Points per £M across, expected price move up, bubble size is expected points. Solid bubbles are your
            picks, faint ones are alternatives. Up and to the right is best: scores well <em>and</em> pays for itself.
          </p>

          <div style={{ display: 'flex', justifyContent: 'flex-end', width: CHART_W, marginBottom: 6 }}>
            <span style={{ font: '400 11.5px/1 Archivo,sans-serif', color: GREEN }}>high points, rising price</span>
          </div>
          <div style={{ position: 'relative', width: CHART_W, height: CHART_H, borderLeft: `1px solid ${LINE_STR}`, borderBottom: `1px solid ${LINE_STR}` }}>
            {xd.ticks.map((t) => (
              <div key={t} style={{ position: 'absolute', left: `${xPct(t)}%`, top: 0, bottom: 0, width: 1, background: 'rgba(22,21,15,.06)' }} />
            ))}
            {yd.ticks.filter((t) => t !== 0).map((t) => (
              <div key={t} style={{ position: 'absolute', left: 0, right: 0, top: `${yPct(t)}%`, height: 1, background: 'rgba(22,21,15,.06)' }} />
            ))}
            <div
              style={{
                position: 'absolute', left: 0, right: 0, top: '50%', height: 1,
                background: 'repeating-linear-gradient(90deg,rgba(22,21,15,.16) 0 4px,transparent 4px 9px)',
              }}
            />
            {yd.ticks.map((t) => (
              <span
                key={t}
                style={{ position: 'absolute', left: 6, top: `calc(${yPct(t)}% - 12px)`, font: `400 11px/1 ${MONO}`, color: t === 0 ? MUTED2 : FAINT }}
              >
                {signedTick(t)}
              </span>
            ))}
            <span style={{ position: 'absolute', right: 6, top: 'calc(50% + 5px)', font: `400 11px/1 ${MONO}`, color: MUTED2 }}>no price change</span>

            {bubbles.map((b) => {
              const [r, g, bl] = hexToRgb(b.color)
              const fill = b.selected ? `rgba(${r},${g},${bl},0.42)` : `rgba(${r},${g},${bl},0.2)`
              const border = b.selected ? `2px solid ${b.color}` : `1px solid rgba(${r},${g},${bl},.45)`
              const labelStyle: CSSProperties = { position: 'absolute', whiteSpace: 'nowrap', font: `${b.selected ? 500 : 400} 12.5px/1 Archivo,sans-serif`, color: b.selected ? INK : MUTED2 }
              const diag = b.dia / 2 * 0.72 + 3
              if (b.labelSide === 'right') { labelStyle.left = b.x + b.dia / 2 + 4; labelStyle.top = b.y; labelStyle.transform = 'translateY(-50%)' }
              else if (b.labelSide === 'left') { labelStyle.left = b.x - b.dia / 2 - 4; labelStyle.top = b.y; labelStyle.transform = 'translate(-100%,-50%)' }
              else if (b.labelSide === 'below') { labelStyle.left = b.x; labelStyle.top = b.y + b.dia / 2 + 2; labelStyle.transform = 'translateX(-50%)' }
              else if (b.labelSide === 'above') { labelStyle.left = b.x; labelStyle.top = b.y - b.dia / 2 - 16; labelStyle.transform = 'translateX(-50%)' }
              else if (b.labelSide === 'upperRight') { labelStyle.left = b.x + diag; labelStyle.top = b.y - diag; labelStyle.transform = 'translateY(-100%)' }
              else if (b.labelSide === 'upperLeft') { labelStyle.left = b.x - diag; labelStyle.top = b.y - diag; labelStyle.transform = 'translate(-100%,-100%)' }
              else if (b.labelSide === 'lowerRight') { labelStyle.left = b.x + diag; labelStyle.top = b.y + diag }
              else { labelStyle.left = b.x - diag; labelStyle.top = b.y + diag; labelStyle.transform = 'translateX(-100%)' }
              return (
                <div key={b.id}>
                  <div
                    style={{
                      position: 'absolute', left: b.x - b.dia / 2, top: b.y - b.dia / 2, width: b.dia, height: b.dia,
                      borderRadius: '50%', background: fill, border,
                    }}
                  />
                  <span style={labelStyle}>{b.name}</span>
                </div>
              )
            })}
          </div>
          <div style={{ position: 'relative', height: 36 }}>
            {xd.ticks.map((t) => (
              <span
                key={t}
                style={{ position: 'absolute', left: `${xPct(t)}%`, top: 6, transform: 'translateX(-50%)', font: `400 11px/1 ${MONO}`, color: FAINT }}
              >
                {t.toFixed(1)}
              </span>
            ))}
            <span style={{ position: 'absolute', left: 0, top: 22, font: '400 11.5px/1 Archivo,sans-serif', color: MUTED2 }}>points per £M →</span>
            <span style={{ position: 'absolute', right: 0, top: 22, font: '400 11.5px/1 Archivo,sans-serif', color: MUTED2 }}>↑ expected price move £M</span>
          </div>
        </div>
      </div>
    </div>
  )
}
