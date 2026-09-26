import { useRef, useState } from 'react'
import { budgetToStep, stepToBudget } from './squadFilters'
import { INK, LINE_STR } from './theme'

interface Props {
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}

// editable twin of the budget slider - type an exact figure, commit on enter or blur. the value
// is clamped to the slider's range and snapped to its step so the two controls never disagree
export function BudgetInput({ value, min, max, step, onChange }: Props) {
  // null while not editing, so the box just mirrors the slider
  const [draft, setDraft] = useState<string | null>(null)
  // escape blurs without committing - blur fires before the reset draft re-renders, so flag it
  const cancelled = useRef(false)

  const shown = draft ?? value.toFixed(1)

  const commit = () => {
    const parsed = parseFloat(shown)
    if (Number.isFinite(parsed)) {
      const clamped = Math.min(max, Math.max(min, parsed))
      const snapped = stepToBudget(budgetToStep(clamped, min, step), min, step)
      if (snapped !== value) onChange(snapped)
    }
  }

  // 16px keeps ios safari from zooming the page when the field takes focus
  return (
    <label style={{ display: 'inline-flex', alignItems: 'baseline', font: '500 16px/1 Archivo,sans-serif', color: INK, cursor: 'text' }}>
      £
      <input
        type="text" inputMode="decimal" enterKeyHint="done" aria-label="Budget in £M"
        value={shown}
        onChange={(e) => {
          // strip anything that isn't a digit or point (so a pasted "£123.9M" still lands), then
          // reject the keystroke unless it's at most 3 digits and 1 decimal place
          const next = e.target.value.replace(/[^\d.]/g, '')
          if (/^\d{0,3}(\.\d?)?$/.test(next)) setDraft(next)
        }}
        onFocus={(e) => { setDraft(value.toFixed(1)); e.target.select() }}
        onBlur={() => {
          if (cancelled.current) cancelled.current = false
          else commit()
          setDraft(null)
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
          if (e.key === 'Escape') { cancelled.current = true; e.currentTarget.blur() }
        }}
        style={{
          width: `${Math.min(Math.max(shown.length, 3), 5) + 0.5}ch`, padding: '0 0 2px', margin: 0,
          font: 'inherit', color: 'inherit', textAlign: 'right', background: 'transparent',
          border: 'none', borderBottom: `1px dashed ${LINE_STR}`, borderRadius: 0, outline: 'none',
        }}
      />
      M
    </label>
  )
}
