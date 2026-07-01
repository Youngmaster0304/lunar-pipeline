import { useState, useEffect, useRef } from 'react'
import type { PipelineRun, WsMessage } from '../types'
import { getRun, connectWs, downloadUrl } from '../api'

interface Props {
  runId: number
  onClose: () => void
  onDelete: () => void
}

type Tab = 'overview' | 'figures' | 'logs' | 'gis'

export default function RunModal({ runId, onClose, onDelete }: Props) {
  const [run, setRun] = useState<PipelineRun | null>(null)
  const [tab, setTab] = useState<Tab>('overview')
  const [logs, setLogs] = useState<string[]>([])
  const [stageStatuses, setStageStatuses] = useState<Record<number, string>>({})
  const logsEnd = useRef<HTMLDivElement>(null)

  // Fetch run data
  useEffect(() => {
    getRun(runId).then(r => {
      setRun(r)
      // restore stage statuses
      const ss: Record<number, string> = {}
      r.stages.forEach(s => { ss[s.stage_index] = s.status })
      setStageStatuses(ss)
    })
  }, [runId])

  // WebSocket live updates
  useEffect(() => {
    if (!run || run.status !== 'running') return
    const disconnect = connectWs(runId, (msg: WsMessage) => {
      if (msg.type === 'log' && msg.line) {
        setLogs(prev => [...prev, msg.line!])
      }
      if (msg.type === 'stage_status' && msg.stage_index !== undefined && msg.status) {
        setStageStatuses(prev => ({ ...prev, [msg.stage_index!]: msg.status! }))
      }
      if (msg.type === 'run_status') {
        setRun(prev => prev ? { ...prev, status: msg.status as any, duration_s: msg.duration_s || null } : prev)
      }
    })
    return disconnect
  }, [run?.status === 'running'])

  // Scroll logs
  useEffect(() => {
    logsEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  if (!run) return null

  const tabs: { key: Tab; label: string; icon: string }[] = [
    { key: 'overview', label: 'Overview', icon: '📋' },
    { key: 'figures', label: 'Figures', icon: '🖼' },
    { key: 'logs', label: 'Logs', icon: '📄' },
    { key: 'gis', label: 'GIS Data', icon: '🗺' },
  ]

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content max-w-4xl animate-slide-up"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-space-700/50">
          <h2 className="font-display text-sm text-slate-200">
            Mission Report <span className="text-lunar-cyan">#{run.id}</span>
            {run.dsc_name && <span className="text-slate-500 font-body font-normal ml-2">— {run.dsc_name}</span>}
          </h2>
          <div className="flex items-center gap-3">
            <span className={`badge badge-${run.status === 'success' ? 'success' : run.status === 'failed' ? 'failed' : 'running'}`}>
              {run.status}
              {run.duration_s != null ? ` • ${run.duration_s.toFixed(1)}s` : ''}
            </span>
            <button className="btn btn-ghost text-xs px-2 py-1" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-space-700/50 px-4">
          {tabs.map(t => (
            <button
              key={t.key}
              className={`px-4 py-2.5 text-xs font-medium transition-all border-b-2 -mb-px
                ${tab === t.key
                  ? 'text-lunar-cyan border-lunar-cyan'
                  : 'text-slate-500 border-transparent hover:text-slate-300'
                }`}
              onClick={() => setTab(t.key)}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {tab === 'overview' && <OverviewTab run={run} onDelete={onDelete} />}
          {tab === 'figures' && <FiguresTab figures={run.figures} />}
          {tab === 'logs' && (
            <LogsTab logs={logs} run={run} stageStatuses={stageStatuses} logsEnd={logsEnd} />
          )}
          {tab === 'gis' && <GisTab gisFiles={run.gis_files} />}
        </div>
      </div>
    </div>
  )
}

function OverviewTab({ run, onDelete }: { run: PipelineRun; onDelete: () => void }) {
  const metrics = [
    { label: 'Ice Volume', value: run.ice_volume_m3 ? `${run.ice_volume_m3.toExponential(2)} m³` : '—', icon: '🧊' },
    { label: 'Candidates', value: run.n_candidates.toString(), icon: '🎯' },
    { label: 'DSC Craters', value: run.n_dsc_craters.toString(), icon: '🌑' },
    { label: 'Dash Feasible', value: run.dash_feasible ? '✅ Yes' : '❌ No', icon: '🗺️' },
    { label: 'SLAM MAE', value: run.slam_mae != null ? run.slam_mae.toFixed(4) : '—', icon: '🤖' },
    { label: 'EFPI Ice %', value: run.efpi_ice_pct != null ? `${run.efpi_ice_pct.toFixed(2)}%` : '—', icon: '📊' },
  ]

  return (
    <div>
      {run.error_message && (
        <div className="mb-4 p-3 rounded-lg bg-lunar-rose/10 border border-lunar-rose/20 text-lunar-rose text-xs sl">
          {run.error_message}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6">
        {metrics.map(m => (
          <div key={m.label} className="bg-space-800/50 rounded-lg p-3 border border-space-700/30">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{m.icon} {m.label}</div>
            <div className="font-mono text-sm text-lunar-cyan">{m.value}</div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <a href={downloadUrl(run.id, 'json')} download className="btn btn-ghost text-xs">
          ⬇ JSON
        </a>
        <a href={downloadUrl(run.id, 'csv')} download className="btn btn-ghost text-xs">
          ⬇ CSV
        </a>
        <button
          className="btn btn-danger text-xs ml-auto"
          onClick={onDelete}
        >
          🗑 Delete Run
        </button>
      </div>
    </div>
  )
}

function FiguresTab({ figures }: { figures: PipelineRun['figures'] }) {
  const [lightbox, setLightbox] = useState<string | null>(null)

  if (figures.length === 0) {
    return <div className="text-center py-12 text-slate-500 text-sm">No figures generated</div>
  }

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {figures.map(f => (
          <button
            key={f.filename}
            className="group relative rounded-lg overflow-hidden border border-space-700/50
                       hover:border-lunar-cyan/30 transition-all"
            onClick={() => setLightbox(f.url)}
          >
            <img src={f.url} alt={f.title} loading="lazy" className="w-full h-32 object-cover" />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-space-950/80 to-transparent p-2">
              <div className="text-[10px] text-slate-300">{f.title}</div>
            </div>
          </button>
        ))}
      </div>

      {lightbox && (
        <div className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-8" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="" className="max-w-full max-h-full object-contain rounded-lg" />
        </div>
      )}
    </>
  )
}

function LogsTab({
  logs, run, stageStatuses, logsEnd,
}: {
  logs: string[]
  run: PipelineRun
  stageStatuses: Record<number, string>
  logsEnd: React.RefObject<HTMLDivElement | null>
}) {
  const initialLogs = logs.length === 0 && run.stages
    ? run.stages.reduce<string[]>((acc, s) => {
        if (s.log_output) acc.push(...s.log_output.split('\n'))
        return acc
      }, [])
    : logs

  return (
    <div>
      {/* Stage statuses */}
      <div className="flex flex-wrap gap-2 mb-4">
        {run.stages.map(s => (
          <div
            key={s.stage_index}
            className={`text-[10px] px-2 py-1 rounded-md border flex items-center gap-1.5
              ${(stageStatuses[s.stage_index] || s.status) === 'success'
                ? 'bg-lunar-emerald/10 border-lunar-emerald/30 text-lunar-emerald'
                : (stageStatuses[s.stage_index] || s.status) === 'failed'
                ? 'bg-lunar-rose/10 border-lunar-rose/30 text-lunar-rose'
                : 'bg-lunar-cyan/10 border-lunar-cyan/30 text-lunar-cyan animate-pulse'
              }`}
          >
            <span>{s.stage_name.split(' ')[0]}</span>
            <span className="opacity-60">{stageStatuses[s.stage_index] || s.status}</span>
          </div>
        ))}
      </div>

      {/* Logs */}
      <div className="bg-space-950 rounded-lg p-4 border border-space-700/50 max-h-80 overflow-y-auto">
        {initialLogs.length === 0 ? (
          <div className="text-slate-500 text-xs text-center py-8">No log output</div>
        ) : (
          initialLogs.map((line, i) => (
            <div key={i} className="sl text-slate-400 hover:text-slate-200">
              {line}
            </div>
          ))
        )}
        <div ref={logsEnd} />
      </div>
    </div>
  )
}

function GisTab({ gisFiles }: { gisFiles: PipelineRun['gis_files'] }) {
  if (gisFiles.length === 0) {
    return <div className="text-center py-12 text-slate-500 text-sm">No GIS exports for this run</div>
  }

  const geotiffs = gisFiles.filter(g => g.format === 'GeoTIFF')
  const geojsons = gisFiles.filter(g => g.format === 'GeoJSON')

  return (
    <div>
      <div className="text-xs text-slate-500 mb-4">
        Georeferenced exports ready for QGIS, ArcGIS, ENVI.
      </div>

      {[{ label: 'GeoTIFF Rasters', files: geotiffs, icon: '🗺️' },
        { label: 'GeoJSON Vectors', files: geojsons, icon: '📍' },
      ].map(section => section.files.length > 0 ? (
        <div key={section.label} className="mb-5">
          <h4 className="text-xs font-display text-slate-400 uppercase tracking-wider mb-2">
            {section.icon} {section.label}
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {section.files.map(f => (
              <a
                key={f.filename}
                href={f.url}
                download
                className="flex items-center gap-2.5 p-3 rounded-lg bg-space-800/50 border border-space-700/30
                           hover:border-lunar-cyan/30 hover:bg-space-800 transition-all text-slate-300 no-underline"
              >
                <span className="text-lg">{section.icon}</span>
                <div>
                  <div className="text-xs font-medium">{f.filename}</div>
                  <div className="text-[10px] text-slate-500">{f.format}</div>
                </div>
              </a>
            ))}
          </div>
        </div>
      ) : null)}
    </div>
  )
}
