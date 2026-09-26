// signed formatters for the suggested-transfers table, where both numeric columns are deltas rather
// than absolute values - a bare "16.9" reads as a points total, "-16.9" as the swing it actually is.
// the value column exists because the optimiser maximises expected points PLUS PRICE_LAMBDA x
// expected price movement: a suggestion can be net-negative on points alone and still win the
// objective by picking up buying power, which looks like a bug when only the points are on screen
import { GREEN, MUTED2, REDNEG } from './theme'

export function signedPoints(value: number) {
  return `${value > 0 ? '+' : value < 0 ? '-' : ''}${Math.abs(value).toFixed(1)}`
}

export function signedValue(value: number) {
  return `${value > 0 ? '+' : value < 0 ? '-' : ''}£${Math.abs(value).toFixed(1)}`
}

// a flat zero is neither good nor bad news - colouring it green would overstate a swap that moved
// the team's value not at all
export function deltaTone(value: number) {
  return value > 0 ? GREEN : value < 0 ? REDNEG : MUTED2
}
