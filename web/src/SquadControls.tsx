import type { CSSProperties } from 'react'
import { Select } from './Select'
import { slotOptions, unaffordableIds, totalSpend, budgetStepCount, budgetToStep, stepToBudget } from './squadFilters'
import { GREEN, INK, LINE, MUTED2, PANEL } from './theme'
import type { AssetOption, Controls } from './api'

interface Props {
  controls: Controls
  budget: number
  driverOptions: AssetOption[]
  constructorOptions: AssetOption[]
  squadDrivers: string[]
  squadConstructors: string[]
  freeTransfers: number
  dirty: boolean
  loading: boolean
  onBudgetChange: (value: number) => void
  onDriverChange: (slot: number, id: string) => void
  onConstructorChange: (slot: number, id: string) => void
  onFreeTransfersChange: (value: number) => void
  onRecalculate: () => void
}

const SLOT_STYLE: CSSProperties = {
  padding: '9px 11px', background: '#f1efe8', border: '1px solid rgba(22,21,15,.16)',
}

export function SquadControls({
  controls, budget, driverOptions, constructorOptions, squadDrivers, squadConstructors, freeTransfers,
  dirty, loading, onBudgetChange, onDriverChange, onConstructorChange, onFreeTransfersChange, onRecalculate,
}: Props) {
  const priceOf = new Map<string, number>([
    ...driverOptions.map((o): [string, number] => [o.id, o.price]),
    ...constructorOptions.map((o): [string, number] => [o.id, o.price]),
  ])
  const squadSpend = totalSpend([...squadDrivers, ...squadConstructors], priceOf)

  return (
    <div style={{ padding: '26px 44px', display: 'grid', gridTemplateColumns: '260px 1fr 140px', gap: 44, alignItems: 'stretch', borderBottom: `1px solid ${LINE}`, background: PANEL }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Budget</span>
          <span style={{ font: '500 16px/1 Archivo,sans-serif', color: INK }}>£{budget.toFixed(1)}M</span>
        </div>
        <input
          type="range" min={0} max={budgetStepCount(controls.budget_min, controls.budget_max, controls.budget_step)} step={1}
          value={budgetToStep(budget, controls.budget_min, controls.budget_step)}
          onChange={(e) => onBudgetChange(stepToBudget(Number(e.target.value), controls.budget_min, controls.budget_step))}
          style={{ width: '100%', accentColor: GREEN, height: 20 }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
          <span style={{ font: '400 11px/1 Archivo,sans-serif', color: '#b0aa9c' }}>£{controls.budget_min.toFixed(1)}M</span>
          <span style={{ font: '400 11px/1 Archivo,sans-serif', color: '#b0aa9c' }}>£{controls.budget_max.toFixed(1)}M</span>
        </div>
      </div>

      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
          <span style={{ font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Your squad</span>
          <span style={{ font: '400 12px/1 Archivo,sans-serif', color: '#b0aa9c' }}>change any slot</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10, marginBottom: 12 }}>
          {squadDrivers.map((did, i) => {
            const otherIds = squadDrivers.filter((_, idx) => idx !== i)
            const otherSpend = squadSpend - (priceOf.get(did) ?? 0)
            const options = slotOptions(driverOptions, did, otherIds)
            const disabledIds = unaffordableIds(driverOptions, did, otherSpend, budget)
            return (
              <div key={i} style={SLOT_STYLE}>
                <p style={{ margin: '0 0 5px', font: '400 10.5px/1 Archivo,sans-serif', color: '#b0aa9c' }}>Driver {i + 1}</p>
                <Select value={did} options={options} disabledIds={disabledIds} placeholder="-" onChange={(id) => onDriverChange(i, id)} />
              </div>
            )
          })}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10 }}>
          {squadConstructors.map((cid, i) => {
            const otherIds = squadConstructors.filter((_, idx) => idx !== i)
            const otherSpend = squadSpend - (priceOf.get(cid) ?? 0)
            const options = slotOptions(constructorOptions, cid, otherIds)
            const disabledIds = unaffordableIds(constructorOptions, cid, otherSpend, budget)
            return (
              <div key={i} style={SLOT_STYLE}>
                <p style={{ margin: '0 0 5px', font: '400 10.5px/1 Archivo,sans-serif', color: '#b0aa9c' }}>Constructor {i + 1}</p>
                <Select value={cid} options={options} disabledIds={disabledIds} placeholder="-" onChange={(id) => onConstructorChange(i, id)} />
              </div>
            )
          })}
          <div style={{ gridColumn: 'span 3', ...SLOT_STYLE, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <p style={{ margin: 0, font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Free transfers</p>
            <div style={{ display: 'flex', border: '1px solid rgba(22,21,15,.16)' }}>
              {[1, 2, 3].map((n) => (
                <span
                  key={n}
                  onClick={() => onFreeTransfersChange(n)}
                  style={{
                    padding: '5px 12px', cursor: 'pointer',
                    font: `${n === freeTransfers ? 500 : 400} 13px/1 Archivo,sans-serif`,
                    background: n === freeTransfers ? GREEN : 'transparent',
                    color: n === freeTransfers ? '#f1efe8' : INK,
                  }}
                >
                  {n}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', height: '100%' }}>
        <div>
          <p style={{ margin: '0 0 6px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Remaining budget</p>
          <p style={{ margin: '0 0 16px', font: '500 18px/1 Archivo,sans-serif', color: INK }}>£{(budget - squadSpend).toFixed(1)}M</p>
          <p style={{ margin: '0 0 6px', font: '400 12.5px/1 Archivo,sans-serif', color: MUTED2 }}>Team value</p>
          <p style={{ margin: 0, font: '500 18px/1 Archivo,sans-serif', color: INK }}>£{squadSpend.toFixed(1)}M</p>
        </div>
        <button
          onClick={onRecalculate}
          disabled={!dirty || loading}
          style={{
            marginTop: 'auto', marginBottom: 1, width: 132, padding: '9px 0', border: 'none',
            cursor: dirty && !loading ? 'pointer' : 'default',
            font: '500 13px/1 Archivo,sans-serif', letterSpacing: '.02em',
            background: dirty ? GREEN : 'rgba(22,21,15,.12)',
            color: dirty ? '#f1efe8' : MUTED2,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          }}
        >
          {loading ? (
            <>
              <span>Recalculating</span>
              <span style={{ display: 'flex', gap: 3 }}>
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="pw-loading-dot"
                    style={{ width: 3.5, height: 3.5, borderRadius: '50%', background: '#f1efe8', animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
            </>
          ) : dirty ? 'Recalculate' : 'Up to date'}
        </button>
      </div>
    </div>
  )
}
