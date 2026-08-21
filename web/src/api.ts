export interface Distribution {
  p10: number
  p25: number
  median: number
  p75: number
  p90: number
}

export interface LadderRow {
  id: string
  name: string
  fia_code: string | null
  constructor_id: string
  color: string
  points: number
  price: number
  selected: boolean
  captain: boolean
  distribution: Distribution | null
}

export interface AltDriver {
  id: string
  fia_code: string
  color: string
  differs: boolean
  captain: boolean
}

export interface AltConstructor {
  id: string
  code: string
  color: string
  differs: boolean
}

export interface AlternativeTeam {
  rank: number
  total_points: number
  gap_to_best: number
  spend: number
  captain_driver_id: string
  drivers: AltDriver[]
  constructors: AltConstructor[]
}

export interface LadderResponse {
  season: number
  round: number
  circuit: string
  drivers: LadderRow[]
  constructors: LadderRow[]
  alternative_teams: AlternativeTeam[]
}

export interface TeamParams {
  budget?: number
  squadMode?: 'model' | 'custom'
  drivers?: string[]
  constructors?: string[]
  freeTransfers?: number
}

function teamQueryString(params: TeamParams = {}): string {
  const qs = new URLSearchParams()
  if (params.budget != null) qs.set('budget', String(params.budget))
  if (params.squadMode) qs.set('squad_mode', params.squadMode)
  if (params.drivers?.length) qs.set('drivers', params.drivers.join(','))
  if (params.constructors?.length) qs.set('constructors', params.constructors.join(','))
  if (params.freeTransfers != null) qs.set('free_transfers', String(params.freeTransfers))
  return qs.toString()
}

// the ladder's selection highlighting and alternative teams depend on the same squad-controls
// state as the team section, so it takes the same params and both are always refetched together
export async function fetchLadder(params: TeamParams = {}): Promise<LadderResponse> {
  const res = await fetch(`/api/ladder?${teamQueryString(params)}`)
  if (!res.ok) throw new Error(`ladder request failed: ${res.status}`)
  return res.json()
}

export interface AssetOption {
  id: string
  name: string
  price: number
  inactive: boolean
}

export interface Controls {
  budget: number
  budget_min: number
  budget_max: number
  budget_step: number
  default_budget: number
  squad_mode: 'model' | 'custom'
  free_transfers: number
  current_drivers: string[]
  current_constructors: string[]
  remaining_budget: number
  team_value: number
}

export interface Hero {
  projected_points: number
  likely_range: Distribution
  spend: number
  net_after_hit: number | null
  transfers_made: number
  captain_id: string
}

export interface LineupRow {
  id: string
  name: string
  is_driver: boolean
  captain: boolean
  in: boolean
  constructor_id: string
  color: string
  points: number
  doubled_points: number | null
  price: number
}

export interface TransferRow {
  out_id: string
  out_name: string
  in_id: string
  in_name: string
  delta: number
}

export interface Transfers {
  rows: TransferRow[]
  net: number
  has_state: boolean
  free: number
  paid: number
}

export interface TeamResponse {
  season: number
  round: number
  circuit: string
  status: string
  controls: Controls
  driver_options: AssetOption[]
  constructor_options: AssetOption[]
  hero: Hero
  lineup: LineupRow[]
  transfers: Transfers
}

export async function fetchTeam(params: TeamParams = {}): Promise<TeamResponse> {
  const res = await fetch(`/api/team?${teamQueryString(params)}`)
  if (!res.ok) throw new Error(`team request failed: ${res.status}`)
  return res.json()
}

export interface BreakdownRow {
  id: string
  name: string
  color: string
  selected: boolean
  quali_position: number
  finish_position: number
  positions_gained: number
  overtakes: number
  prob_fl: number
  prob_dotd: number
  dnf_prob: number
  expected_points: number
}

export interface BreakdownResponse {
  season: number
  round: number
  circuit: string
  available_rounds: number[]
  rows: BreakdownRow[]
}

// the breakdown's row highlighting depends on the same squad-controls state as the team/ladder
// sections, so it takes the same params (plus its own independent round to browse)
export async function fetchBreakdown(season: number | undefined, round: number | undefined, params: TeamParams = {}): Promise<BreakdownResponse> {
  const qs = new URLSearchParams(teamQueryString(params))
  if (season != null) qs.set('season', String(season))
  if (round != null) qs.set('round', String(round))
  const res = await fetch(`/api/breakdown?${qs.toString()}`)
  if (!res.ok) throw new Error(`breakdown request failed: ${res.status}`)
  return res.json()
}

export interface Waterfall {
  team_value: number
  cash: number
  expected_rises: number
  budget_next: number
}

export interface PriceMoveRow {
  id: string
  name: string
  color: string
  is_driver: boolean
  price: number
  ppm: number
  move: number | null
}

export interface ValueMapPoint {
  id: string
  name: string
  color: string
  ppm: number
  move: number | null
  points: number
  selected: boolean
}

export interface ValueResponse {
  season: number
  round: number
  circuit: string
  next_round: number
  price_lambda: number
  have_price_data: boolean
  waterfall: Waterfall
  price_moves: PriceMoveRow[]
  value_map: ValueMapPoint[]
}

// the waterfall/price-moves/value-map all depend on the same squad-controls state as the other
// sections, so it takes the same params and is refetched alongside them
export async function fetchValue(params: TeamParams = {}): Promise<ValueResponse> {
  const res = await fetch(`/api/value?${teamQueryString(params)}`)
  if (!res.ok) throw new Error(`value request failed: ${res.status}`)
  return res.json()
}

export interface TrackRecordRound {
  round: number
  pct: number
  is_worst: boolean
}

export interface TrackRecordResponse {
  available: boolean
  season: number
  last_round: number
  model_total: number
  oracle_total: number
  average_pct: number
  best_round: number
  best_pct: number
  worst_round: number
  worst_pct: number
  rounds: TrackRecordRound[]
}

// season-level backtest stats - independent of the current squad selection, so no team params
export async function fetchTrackRecord(): Promise<TrackRecordResponse> {
  const res = await fetch('/api/track-record')
  if (!res.ok) throw new Error(`track record request failed: ${res.status}`)
  return res.json()
}
