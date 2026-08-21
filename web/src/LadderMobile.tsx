import { LadderColumnMobile } from './LadderColumnMobile'
import { AlternativeTeamsMobile } from './AlternativeTeamsMobile'
import { MobileSheet } from './MobileSheet'
import { MUTED } from './theme'
import type { LadderResponse } from './api'

interface Props {
  data: LadderResponse
}

export function LadderMobile({ data }: Props) {
  return (
    <div style={{ padding: '8px 22px 0', background: '#eae7de', borderTop: '1px solid rgba(22,21,15,.14)' }}>
      <MobileSheet title="The ladder" count={`${data.drivers.length} drivers`} defaultOpen>
        <p style={{ margin: '0 0 18px', font: '400 12.5px/1.55 Archivo,sans-serif', color: MUTED }}>
          Solid bar = middle half of 10,000 simulations, faded ends reach the 10th and 90th
          percentiles (P10-P90), caret = expected points. Your picks carry their team colours.
        </p>
        <LadderColumnMobile label="Driver" rows={data.drivers} />
        <div style={{ height: 24 }} />
        <LadderColumnMobile label="Constructor" rows={data.constructors} />
      </MobileSheet>

      <MobileSheet title="Alternative teams" count={`${data.alternative_teams.length} lineups`}>
        <AlternativeTeamsMobile teams={data.alternative_teams} />
      </MobileSheet>
    </div>
  )
}
