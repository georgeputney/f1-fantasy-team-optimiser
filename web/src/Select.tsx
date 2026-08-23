import { useEffect, useRef, useState } from 'react'
import { CARD, FAINT, GREEN_TINT, INK, LINE_MED, MUTED, MUTED2 } from './theme'

interface Option {
  id: string
  name: string
  color?: string
}

interface Props {
  value: string
  options: Option[]
  placeholder: string
  onChange: (id: string) => void
  fontSize?: number
  disabledIds?: Set<string>
  // hugs its content (chevron sits right next to the label) instead of stretching to fill the
  // parent with space-between - use for an inline label like "Round 12 ⌄", not a bordered slot box
  fitContent?: boolean
  // overrides the label's default (INK when a value is selected) - for an inline trigger sitting
  // next to muted plain-text siblings, the default reads noticeably darker/bolder than them
  labelColor?: string
}

const MENU_WIDTH = 200

// same driver id can carry a different team/colour across rounds after a seat swap - a small dot
// is enough to tell two same-named entries apart without needing a full team label in a slot this
// narrow
function ColorDot({ color }: { color: string }) {
  return <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
}

// native <select> options popups can't be restyled (they render as the OS's own UI), so this is a
// button + absolutely-positioned list standing in for one, styled to match the rest of the page.
// disabledIds are shown greyed-out rather than removed from the list - hiding an option a user
// might expect to see (e.g. one that's currently unaffordable) reads as a bug, not a rule.
export function Select({ value, options, placeholder, onChange, fontSize = 14, disabledIds, fitContent, labelColor }: Props) {
  const [open, setOpen] = useState(false)
  const [alignRight, setAlignRight] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  const selected = options.find((o) => o.id === value)

  // the menu is wider than most trigger slots (see MENU_WIDTH comment below), so anchoring it to
  // the trigger's left edge can push it past the right side of a narrow mobile viewport - flip to
  // right-anchored whenever there isn't room, checked fresh each time the menu opens
  const toggleOpen = () => {
    if (!open && ref.current) {
      const rect = ref.current.getBoundingClientRect()
      setAlignRight(rect.left + MENU_WIDTH > window.innerWidth - 8)
    }
    setOpen((v) => !v)
  }

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={toggleOpen}
        style={{
          width: fitContent ? 'auto' : '100%', textAlign: 'left', border: 'none', background: 'transparent', padding: 0,
          font: `400 ${fontSize}px/1 Archivo,sans-serif`, color: labelColor ?? (selected ? INK : FAINT), cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: fitContent ? 'flex-start' : 'space-between', gap: 6,
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
          {selected?.color && <ColorDot color={selected.color} />}
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {selected ? selected.name : placeholder}
          </span>
        </span>
        <span style={{ color: MUTED2, font: '400 10px/1 Archivo,sans-serif', flexShrink: 0 }}>⌄</span>
      </button>

      {open && (
        <div
          style={{
            // wider than the trigger, not just left:0/right:0 matching it - a driver slot can be as
            // narrow as ~86px on desktop, nowhere near enough room for a name plus "over budget"
            position: 'absolute', top: 'calc(100% + 4px)', zIndex: 20,
            ...(alignRight ? { right: 0 } : { left: 0 }),
            minWidth: '100%', width: MENU_WIDTH,
            background: CARD, border: `1px solid ${LINE_MED}`, maxHeight: 220, overflowY: 'auto',
            boxShadow: '0 6px 16px rgba(22,21,15,.16)',
          }}
        >
          {options.map((o) => {
            const isDisabled = disabledIds?.has(o.id) ?? false
            return (
              <div
                key={o.id}
                onClick={() => { if (isDisabled) return; onChange(o.id); setOpen(false) }}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
                  padding: '8px 11px', font: `400 13px/1 Archivo,sans-serif`,
                  cursor: isDisabled ? 'not-allowed' : 'pointer',
                  color: isDisabled ? FAINT : o.id === value ? INK : MUTED,
                  background: o.id === value ? GREEN_TINT : 'transparent',
                }}
                onMouseEnter={(e) => { if (!isDisabled) e.currentTarget.style.background = o.id === value ? GREEN_TINT : 'rgba(22,21,15,.05)' }}
                onMouseLeave={(e) => { e.currentTarget.style.background = o.id === value ? GREEN_TINT : 'transparent' }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                  {o.color && <ColorDot color={o.color} />}
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{o.name}</span>
                </span>
                {isDisabled && <span style={{ font: '400 10.5px/1 Archivo,sans-serif', color: FAINT, flexShrink: 0 }}>over budget</span>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
