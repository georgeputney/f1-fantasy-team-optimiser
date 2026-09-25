import { DistributionBar } from './DistributionBar'
import { niceScale } from './ticks'
import { CARD, GREEN, INK, MUTED2, REDNEG } from './theme'
import type { Hero as HeroData } from './api'

interface Props {
  hero: HeroData
}

export function Hero({ hero }: Props) {
  const { likely_range: r } = hero
  // the domain must cover the caret (projected_points) too, not just the simulation's own p10/p90 -
  // the two are different models and can disagree enough that projected_points falls outside the
  // simulated range, which would otherwise clamp the caret to the edge and make it look like it's
  // sitting exactly at p10/p90 when it's actually well beyond either
  const { domainMin, domainMax } = niceScale(
    Math.min(r.p10, hero.projected_points), Math.max(r.p90, hero.projected_points), 3,
  )
  const netPositive = (hero.net_after_hit ?? 0) >= 0
  // the p10/p90 captions have to be placed on the same scale the bar is drawn on, not pinned to the
  // container's edges - niceScale pads the domain ~4% past both ends, so a space-between pair sits
  // that padding's width outside the bar ends it names (~13px on a 340px bar). That gap read as the
  // caret overrunning the bar whenever projected_points landed near p90, when it was really the p90
  // caption sitting too far right and crowding it
  const span = domainMax - domainMin || 1
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - domainMin) / span) * 100))

  return (
    <div id="team" style={{ padding: '44px 44px 38px', display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 64, alignItems: 'end', background: CARD, scrollMarginTop: 66 }}>
      <div>
        <p style={{ margin: '0 0 10px', font: '500 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Projected points</p>
        <p style={{ margin: 0, font: '600 148px/.8 Archivo,sans-serif', letterSpacing: '-.05em', color: INK }}>
          {hero.projected_points}
        </p>
        {/* the headline scores every driver as finishing - compose_drivers carries no DNF term, by
            design (discounting it backtested worse both for selection and for display accuracy). The
            simulation beside it does price retirement in, so its median always sits lower and the caret
            rides high in the band. Saying so turns that gap from a glitch into the point */}
        <p style={{ margin: '14px 0 0', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>assumes a clean race</p>
      </div>
      <div style={{ paddingBottom: 14, display: 'grid', gridTemplateColumns: '1fr auto auto', columnGap: 52, alignItems: 'end' }}>
        <div>
          <p style={{ margin: '0 0 14px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Likely range · 10,000 simulations</p>
          <div style={{ maxWidth: 340 }}>
            <DistributionBar
              dist={r} points={hero.projected_points} domainMin={domainMin} domainMax={domainMax} ticks={[]}
              color={GREEN} selected
            />
          </div>
          <div style={{ position: 'relative', height: 13, maxWidth: 340, marginTop: 6 }}>
            {([r.p10, r.p90] as const).map((v) => (
              <span
                key={v}
                style={{
                  position: 'absolute', left: `${pct(v)}%`, transform: 'translateX(-50%)',
                  font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2, whiteSpace: 'nowrap',
                }}
              >
                {v.toFixed(0)}
              </span>
            ))}
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
