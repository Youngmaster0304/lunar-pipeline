const STAGES = [
  { idx: 0, label: 'DFSAR\nPolarimetry', icon: '🛰️', desc: 'CPR / DOP / CRIM' },
  { idx: 1, label: 'OHRC\nSafety', icon: '📸', desc: 'Roughness / Boulders' },
  { idx: 2, label: 'MCDA\nSeeding', icon: '🎯', desc: 'Slope / Exposure' },
  { idx: 3, label: 'PSR\nMapping', icon: '🌑', desc: 'Shadow / DSC' },
  { idx: 4, label: 'Rover\nTraverse', icon: '🗺️', desc: 'A* Path Planning' },
  { idx: 5, label: 'SLAM\nAvoidance', icon: '🤖', desc: 'Bug-2 MAE' },
  { idx: 6, label: 'Ice Vol\nEstimate', icon: '📊', desc: 'Dielectric / EFPI' },
]

interface Props { activeStage: number | null; stageStatuses: Record<number, string> }

export default function FlowDiagram({ activeStage, stageStatuses }: Props) {
  return (
    <div className="card p-5">
      <h3 className="text-xs font-display text-lunar-cyan/70 uppercase tracking-widest mb-4">
        Pipeline Stages
      </h3>
      <div className="flex items-center gap-1 overflow-x-auto pb-2">
        {STAGES.map((s, i) => {
          const status = stageStatuses[s.idx] || 'pending'
          const isActive = activeStage === s.idx
          const colors = {
            success: 'border-lunar-emerald/40 bg-lunar-emerald/10 text-lunar-emerald',
            failed: 'border-lunar-rose/40 bg-lunar-rose/10 text-lunar-rose',
            running: 'border-lunar-cyan/60 bg-lunar-cyan/10 text-lunar-cyan animate-pulse-glow',
            pending: 'border-space-600 bg-space-800 text-slate-500',
          }[status]
          return (
            <div key={s.idx} className="flex items-center gap-1 shrink-0">
              <div
                className={`
                  flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border
                  transition-all duration-300 min-w-[80px]
                  ${colors}
                  ${isActive && status === 'running' ? 'scale-105' : ''}
                `}
              >
                <span className="text-lg">{s.icon}</span>
                <span className="text-[9px] font-display leading-tight text-center whitespace-pre">
                  {s.label}
                </span>
                <span className="text-[8px] opacity-60">{s.desc}</span>
              </div>
              {i < STAGES.length - 1 && (
                <div className={`w-3 h-px ${status === 'success' ? 'bg-lunar-emerald/40' : 'bg-space-600'}`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
