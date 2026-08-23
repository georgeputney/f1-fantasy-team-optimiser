import { CARD, LINE, MUTED2 } from './theme'

interface Props {
  mobile?: boolean
}

const WRITEUP_URL = 'http://georgeputney.com/projects/f1-fantasy-team-optimiser/'

// not from the design source (nothing follows Track record there) - kept deliberately minimal so
// it doesn't read as a real section of its own
export function Footer({ mobile }: Props) {
  return (
    <div style={{ padding: mobile ? '20px 22px' : '20px 44px', borderTop: `1px solid ${LINE}`, background: CARD }}>
      <a
        href={WRITEUP_URL}
        target="_blank"
        rel="noopener noreferrer"
        style={{ font: '400 13px/1 Archivo,sans-serif', color: MUTED2, textDecoration: 'none' }}
      >
        Full write-up ↗︎
      </a>
    </div>
  )
}
