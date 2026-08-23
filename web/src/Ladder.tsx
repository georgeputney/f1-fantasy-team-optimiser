import { LadderColumn } from './LadderColumn'
import { AlternativeTeams } from './AlternativeTeams'
import { INK, LINE, MUTED, PANEL } from './theme'
import type { LadderResponse } from './api'

interface Props {
  data: LadderResponse
}

export function Ladder({ data }: Props) {
  return (
    <div id="ladder" style={{ padding: '40px 44px 44px', borderTop: `1px solid ${LINE}`, background: PANEL, scrollMarginTop: 66 }}>
      <div style={{ marginBottom: 22 }}>
        <h3 style={{ margin: '0 0 8px', font: '500 22px/1 Archivo,sans-serif', letterSpacing: '-.015em', color: INK }}>
          The ladder
        </h3>
        <p style={{ margin: 0, font: '400 14.5px/1.6 Archivo,sans-serif', color: MUTED }}>
          Every driver and constructor ranked by expected points. The solid bar covers the middle half
          of 10,000 simulated race outcomes; the faded ends reach the 10th and 90th percentiles, so a
          wide bar means a volatile weekend. The caret marks expected points and the line through the
          bar marks the simulation's median - they're two different estimates and can disagree.
          Gridlines mark the points scale, and the P10-P90 column gives the same range as numbers so
          close-run rows near the bottom can still be told apart. Your seven picks carry their team
          colours.
        </p>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 34 }}>
        <LadderColumn label="Driver" rows={data.drivers} />
        <LadderColumn label="Constructor" rows={data.constructors} />
        <AlternativeTeams teams={data.alternative_teams} />
      </div>
    </div>
  )
}
