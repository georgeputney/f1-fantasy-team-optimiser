import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchTeam, fetchLadder, fetchBreakdown, fetchValue, fetchTrackRecord,
  type TeamResponse, type LadderResponse, type BreakdownResponse, type ValueResponse, type TrackRecordResponse,
} from './api'

// drives /api/team, /api/ladder and /api/breakdown from the same squad-controls state, so the
// ladder's selection highlighting, alternative teams, and the breakdown's highlighted row all
// match whatever the user is exploring above. each of those requests re-solves the optimiser (the
// ladder also solves 5 alternative-team ILPs), so edits don't auto-fetch - they just update local
// state and mark it dirty; a "Recalculate" button fires the actual request once the user is done
// adjusting things, instead of on every keystroke/slot change. the breakdown's round is browsed
// independently of that - switching rounds re-fetches immediately using whatever squad state was
// last committed (not pending edits), tracked in a ref so `load` doesn't need it as a dependency
export function useTeam() {
  const [team, setTeam] = useState<TeamResponse | null>(null)
  const [ladder, setLadder] = useState<LadderResponse | null>(null)
  const [breakdown, setBreakdown] = useState<BreakdownResponse | null>(null)
  const [value, setValue] = useState<ValueResponse | null>(null)
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [dirty, setDirty] = useState(false)

  const [budget, setBudgetState] = useState<number | null>(null)
  const [squadDrivers, setSquadDrivers] = useState<string[]>([])
  const [squadConstructors, setSquadConstructors] = useState<string[]>([])
  const [freeTransfers, setFreeTransfersState] = useState(2)

  const initialised = useRef(false)
  const breakdownRound = useRef<number | undefined>(undefined)
  // StrictMode double-invokes the initial effect in dev, and switching rounds fires a new request
  // before the previous one necessarily resolves - without a sequence guard, whichever response
  // resolves LAST wins regardless of which was issued last, so a stale response can silently
  // clobber a newer one (this is what made round-switching look broken: the request was correct,
  // but an earlier in-flight request's response overwrote it moments later)
  const requestSeq = useRef(0)

  const load = useCallback((params: {
    budget: number | null
    squadMode: 'model' | 'custom'
    drivers: string[]
    constructors: string[]
    freeTransfers: number
  }) => {
    const seq = ++requestSeq.current
    setLoading(true)
    const requestParams = {
      budget: params.budget ?? undefined,
      squadMode: params.squadMode,
      drivers: params.squadMode === 'custom' ? params.drivers.filter(Boolean) : undefined,
      constructors: params.squadMode === 'custom' ? params.constructors.filter(Boolean) : undefined,
      freeTransfers: params.freeTransfers,
    }
    Promise.all([
      fetchTeam(requestParams),
      fetchLadder(requestParams),
      fetchBreakdown(undefined, breakdownRound.current, requestParams),
      fetchValue(requestParams),
      fetchTrackRecord(),
    ])
      .then(([teamRes, ladderRes, breakdownRes, valueRes, trackRecordRes]) => {
        if (seq !== requestSeq.current) return // a newer request has since been issued - drop this one
        setTeam(teamRes)
        setLadder(ladderRes)
        setBreakdown(breakdownRes)
        setValue(valueRes)
        setTrackRecord(trackRecordRes)
        breakdownRound.current = breakdownRes.round
        setDirty(false)
        if (!initialised.current) {
          initialised.current = true
          setBudgetState(teamRes.controls.budget)
          setSquadDrivers(teamRes.controls.current_drivers)
          setSquadConstructors(teamRes.controls.current_constructors)
          setFreeTransfersState(teamRes.controls.free_transfers)
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => { if (seq === requestSeq.current) setLoading(false) })
  }, [])

  // the params load() was last called with - setBreakdownRound reuses these so switching rounds
  // doesn't require pressing Recalculate first, and doesn't jump the ahead to pending edits either
  const lastParams = useRef({ budget: null as number | null, squadMode: 'model' as 'model' | 'custom', drivers: [] as string[], constructors: [] as string[], freeTransfers: 2 })

  // initial load
  useEffect(() => {
    load(lastParams.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const recalculate = useCallback(() => {
    const params = { budget, squadMode: 'custom' as const, drivers: squadDrivers, constructors: squadConstructors, freeTransfers }
    lastParams.current = params
    load(params)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [budget, squadDrivers, squadConstructors, freeTransfers, load])

  const setBudget = useCallback((value: number) => {
    setBudgetState(value)
    setDirty(true)
  }, [])

  const setDriver = useCallback((slot: number, id: string) => {
    setSquadDrivers((prev) => {
      const next = [...prev]
      next[slot] = id
      return next
    })
    setDirty(true)
  }, [])

  const setConstructor = useCallback((slot: number, id: string) => {
    setSquadConstructors((prev) => {
      const next = [...prev]
      next[slot] = id
      return next
    })
    setDirty(true)
  }, [])

  const setFreeTransfers = useCallback((value: number) => {
    setFreeTransfersState(value)
    setDirty(true)
  }, [])

  const setBreakdownRound = useCallback((round: number) => {
    breakdownRound.current = round
    load(lastParams.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load])

  return {
    team, ladder, breakdown, value, trackRecord, error, loading, dirty, budget, squadDrivers, squadConstructors, freeTransfers,
    setBudget, setDriver, setConstructor, setFreeTransfers, recalculate, setBreakdownRound,
  }
}
