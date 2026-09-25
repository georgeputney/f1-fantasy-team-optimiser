import { CARD, INK, MUTED2 } from './theme'
import type { Hero as HeroData } from './api'

interface Props {
  hero: HeroData
}

export function HeroMobile({ hero }: Props) {
  const { likely_range: r } = hero

  return (
    <div style={{ padding: '28px 22px 26px', background: CARD }}>
      <p style={{ margin: '0 0 8px', font: '500 12px/1 Archivo,sans-serif', color: MUTED2 }}>Projected points</p>
      <p style={{ margin: '0 0 6px', font: '600 104px/.8 Archivo,sans-serif', letterSpacing: '-.05em', color: INK }}>
        {hero.projected_points}
      </p>
      <p style={{ margin: 0, font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>
        {r.p10.toFixed(0)} – {r.p90.toFixed(0)} likely · £{hero.spend.toFixed(1)}M spent
      </p>
      {/* same caveat the desktop hero carries: the headline assumes every driver finishes, the
          likely range beside it prices retirement in, so the two never quite line up */}
      <p style={{ margin: '7px 0 0', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>assumes a clean race</p>
    </div>
  )
}
