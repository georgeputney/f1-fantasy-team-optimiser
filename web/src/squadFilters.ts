import type { AssetOption } from './api'

// options for one slot = not already picked in another slot of the same category. The slot's own
// current value is always kept even if picked elsewhere (shouldn't happen, but never hide the
// current selection). Duplicates are hard-excluded - there's no legitimate reason to show a driver
// already in another slot - but budget is handled separately (see unaffordableIds) since silently
// hiding options a user might expect to see is confusing; those are shown greyed-out instead.
// inactive options (a held asset that's gone out mid-week) are excluded the same way everywhere
// except the one slot that actually holds them - they're not a valid new pick for any other slot.
export function slotOptions(options: AssetOption[], currentValue: string, otherSelectedIds: string[]): AssetOption[] {
  const excluded = new Set(otherSelectedIds.filter((id) => id && id !== currentValue))
  return options.filter((o) => o.id === currentValue || (!excluded.has(o.id) && !o.inactive))
}

// ids that would push the squad over budget if picked in this slot - shown but disabled, not
// hidden, so the option is still visible and its price explains why it can't be picked right now
export function unaffordableIds(options: AssetOption[], currentValue: string, otherSpend: number, budget: number): Set<string> {
  return new Set(
    options.filter((o) => o.id !== currentValue && otherSpend + o.price > budget).map((o) => o.id),
  )
}

// total price of every filled slot across both categories, keyed by asset id -> price
export function totalSpend(ids: string[], priceOf: Map<string, number>): number {
  return ids.reduce((sum, id) => sum + (id ? priceOf.get(id) ?? 0 : 0), 0)
}

// native <input type="range"> can't actually reach `max` by dragging when (max - min) isn't a
// clean multiple of `step` - min/max here come from a retrospective solve and rarely divide
// evenly, and float error (e.g. 88.5 % 0.1 = 0.09999999999999509, not 0) makes the browser cap
// the reachable value just short of the real max. Driving the <input> off an integer step count
// instead of the raw float range sidesteps the problem entirely - both ends are always reachable.
export function budgetStepCount(min: number, max: number, step: number): number {
  return Math.round((max - min) / step)
}

export function budgetToStep(value: number, min: number, step: number): number {
  return Math.round((value - min) / step)
}

export function stepToBudget(stepValue: number, min: number, step: number): number {
  return Math.round((min + stepValue * step) * 10) / 10
}
