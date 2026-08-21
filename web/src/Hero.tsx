import { DistributionBar } from './DistributionBar'
import { niceScale } from './ticks'
import { CARD, GREEN, INK, MUTED2, REDNEG } from './theme'
import type { Hero as HeroData } from './api'

interface Props {
  hero: HeroData
}

export function Hero({ hero }: Props) {
  const { likely_range: r } = hero
  const { domainMin, domainMax } = niceScale(r.p10, r.p90, 3)
  const netPositive = (hero.net_after_hit ?? 0) >= 0

  return (
    <div id="team" style={{ padding: '44px 44px 38px', display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 64, alignItems: 'end', background: CARD, scrollMarginTop: 66 }}>
      <div>
        <p style={{ margin: '0 0 10px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Projected points</p>
        <p style={{ margin: 0, font: '600 148px/.8 Archivo,sans-serif', letterSpacing: '-.05em', color: INK }}>
          {hero.projected_points}
        </p>
      </div>
      <div style={{ paddingBottom: 14, display: 'grid', gridTemplateColumns: '1fr auto auto', columnGap: 52, alignItems: 'end' }}>
        <div>
          <p style={{ margin: '0 0 14px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Likely range · 10,000 simulations</p>
          <div style={{ maxWidth: 340 }}>
            <DistributionBar
              dist={r} points={r.median} domainMin={domainMin} domainMax={domainMax} ticks={[]}
              color={GREEN} selected
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', maxWidth: 340, marginTop: 6 }}>
            <span style={{ font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>{r.p10.toFixed(0)}</span>
            <span style={{ font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>{r.p90.toFixed(0)}</span>
          </div>
        </div>
        <div>
          <p style={{ margin: '0 0 6px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Spend</p>
          <p style={{ margin: 0, font: '500 21px/1 Archivo,sans-serif', color: INK }}>£{hero.spend.toFixed(1)}M</p>
        </div>
        {hero.net_after_hit != null && (
          <div>
            <p style={{ margin: '0 0 6px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Net after transfer hit</p>
            <p style={{ margin: 0, font: '500 21px/1 Archivo,sans-serif', color: netPositive ? GREEN : REDNEG }}>
              {netPositive ? '+' : ''}{hero.net_after_hit.toFixed(0)} pts
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
