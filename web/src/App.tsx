import { useEffect, useState } from 'react'
import { useTeam } from './useTeam'
import { Ladder } from './Ladder'
import { LadderMobile } from './LadderMobile'
import { SquadControls } from './SquadControls'
import { SquadControlsMobile } from './SquadControlsMobile'
import { Hero } from './Hero'
import { HeroMobile } from './HeroMobile'
import { Lineup } from './Lineup'
import { LineupMobile } from './LineupMobile'
import { Transfers } from './Transfers'
import { TransfersMobile } from './TransfersMobile'
import { DriverBreakdown } from './DriverBreakdown'
import { ValueBuyingPower } from './ValueBuyingPower'
import { ValueBuyingPowerMobile } from './ValueBuyingPowerMobile'
import { TrackRecord } from './TrackRecord'
import { Header } from './Header'
import { Footer } from './Footer'
import { CARD, GREEN, INK, LINE, MUTED2 } from './theme'

const MOBILE_BREAKPOINT = 820

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < MOBILE_BREAKPOINT)
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return isMobile
}

export default function App() {
  const isMobile = useIsMobile()
  const {
    team, ladder, breakdown, value, trackRecord, error, loading, dirty, budget, squadDrivers, squadConstructors, freeTransfers,
    setBudget, setDriver, setConstructor, setFreeTransfers, recalculate, setBreakdownRound,
  } = useTeam()

  if (error) {
    return <div style={{ padding: 44, font: '400 14px/1.5 Archivo,sans-serif', color: '#a8412a' }}>{error}</div>
  }
  if (!ladder || !team) {
    return (
      <div
        style={{
          maxWidth: isMobile ? 390 : 1180, minHeight: '100vh', margin: '0 auto', background: CARD,
          border: `1px solid ${LINE}`, display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', gap: 14,
        }}
      >
        <span style={{ font: '600 15px/1 Archivo,sans-serif', color: INK, letterSpacing: '-.01em' }}>Pitwall</span>
        <div style={{ display: 'flex', gap: 6 }}>
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="pw-loading-dot"
              style={{ width: 6, height: 6, borderRadius: '50%', background: GREEN, animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
        <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>Loading this round's data…</span>
      </div>
    )
  }

  return (
    <div
      style={{
        maxWidth: isMobile ? 390 : 1180, margin: '0 auto', background: CARD,
        border: `1px solid ${LINE}`,
      }}
    >
      {isMobile ? (
        <div
          style={{
            padding: '18px 22px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            borderBottom: `1px solid ${LINE}`,
          }}
        >
          <span style={{ font: '600 14px/1 Archivo,sans-serif', color: INK }}>Pitwall</span>
          <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>
            Round {team.round} · {team.circuit}
          </span>
        </div>
      ) : (
        <Header round={team.round} circuit={team.circuit} status={team.status} />
      )}

      {isMobile ? (
        <>
          <SquadControlsMobile
            controls={team.controls} budget={budget ?? team.controls.budget}
            driverOptions={team.driver_options} constructorOptions={team.constructor_options}
            squadDrivers={squadDrivers} squadConstructors={squadConstructors} freeTransfers={freeTransfers}
            dirty={dirty} loading={loading}
            onBudgetChange={setBudget} onDriverChange={setDriver} onConstructorChange={setConstructor}
            onFreeTransfersChange={setFreeTransfers} onRecalculate={recalculate}
          />
          <HeroMobile hero={team.hero} />
          <LineupMobile lineup={team.lineup} />
          <TransfersMobile transfers={team.transfers} />
          <LadderMobile data={ladder} />
          {value && <ValueBuyingPowerMobile data={value} />}
          <Footer mobile />
        </>
      ) : (
        <>
          <SquadControls
            controls={team.controls} budget={budget ?? team.controls.budget}
            driverOptions={team.driver_options} constructorOptions={team.constructor_options}
            squadDrivers={squadDrivers} squadConstructors={squadConstructors} freeTransfers={freeTransfers}
            dirty={dirty} loading={loading}
            onBudgetChange={setBudget} onDriverChange={setDriver} onConstructorChange={setConstructor}
            onFreeTransfersChange={setFreeTransfers} onRecalculate={recalculate}
          />
          <Hero hero={team.hero} />
          <Lineup lineup={team.lineup} />
          <Transfers transfers={team.transfers} />
          <Ladder data={ladder} />
          {breakdown && <DriverBreakdown data={breakdown} onRoundChange={setBreakdownRound} />}
          {value && <ValueBuyingPower data={value} />}
          {trackRecord && <TrackRecord data={trackRecord} />}
          <Footer />
        </>
      )}
    </div>
  )
}
