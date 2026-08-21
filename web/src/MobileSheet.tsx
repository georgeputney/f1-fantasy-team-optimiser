import type { ReactNode } from 'react'
import { INK, MUTED2 } from './theme'

interface Props {
  title: string
  count: string
  defaultOpen?: boolean
  children: ReactNode
}

// native <details>/<summary> collapsible sheet - no JS state needed, matches the design's
// mobile section pattern (chevron flips via the [open] CSS attribute selector below)
export function MobileSheet({ title, count, defaultOpen, children }: Props) {
  return (
    <details open={defaultOpen} className="pw-sheet">
      <summary
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12,
          padding: '16px 0', minHeight: 48, cursor: 'pointer', listStyle: 'none',
          borderBottom: '1px solid rgba(22,21,15,.12)',
        }}
      >
        <span style={{ font: '500 16px/1 Archivo,sans-serif', color: INK }}>{title}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2 }}>{count}</span>
          <span className="pw-chevron" style={{ font: '400 12px/1 Archivo,sans-serif', color: MUTED2 }}>⌄</span>
        </span>
      </summary>
      <div style={{ padding: '18px 0 6px' }}>{children}</div>
    </details>
  )
}
