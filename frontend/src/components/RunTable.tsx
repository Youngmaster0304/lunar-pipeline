import { useState, useMemo } from 'react'
import type { RunSummary } from '../types'

interface Props {
  runs: RunSummary[]
  onSelect: (id: number) => void
  onDelete: (id: number) => void
}

type SortKey = 'id' | 'created_at' | 'duration_s' | 'status'
const PAGE = 15

const statusIcon = (s: string) =>
  s === 'success' ? '●' : s === 'failed' ? '●' : '●'

export default function RunTable({ runs, onSelect, onDelete }: Props) {
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<SortKey>('id')
  const [desc, setDesc] = useState(true)
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return runs
      .filter(r => !q || r.id.toString().includes(q) || (r.dsc_name && r.dsc_name.toLowerCase().includes(q)) || r.status.includes(q))
      .sort((a, b) => {
        const av = a[sort], bv = b[sort]
        if (av == null) return 1
        if (bv == null) return -1
        return desc ? (av < bv ? 1 : -1) : (av > bv ? 1 : -1)
      })
  }, [runs, search, sort, desc])

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE))
  const safePage = Math.min(page, pages - 1)
  const pageItems = filtered.slice(safePage * PAGE, (safePage + 1) * PAGE)

  function toggleSort(key: SortKey) {
    if (sort === key) setDesc(d => !d)
    else { setSort(key); setDesc(true) }
    setPage(0)
  }

  const sortArrow = (key: SortKey) =>
    sort === key ? (desc ? ' ↓' : ' ↑') : ''

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center gap-3 p-4 border-b border-space-700/50">
        <input
          className="input max-w-xs"
          placeholder="Search runs…"
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0) }}
        />
        <span className="text-xs text-slate-500 ml-auto">
          {filtered.length} / {runs.length} runs
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-space-700/50 text-xs text-slate-500 uppercase tracking-wider">
              {(['id', 'created_at', 'status', 'duration_s'] as SortKey[]).map(k => (
                <th
                  key={k}
                  className="text-left px-4 py-3 cursor-pointer hover:text-slate-300 font-medium select-none"
                  onClick={() => toggleSort(k)}
                >
                  {k === 'id' ? 'Run' : k === 'duration_s' ? 'Time' : k}{sortArrow(k)}
                </th>
              ))}
              <th className="text-left px-4 py-3 font-medium">Name</th>
              <th className="text-left px-4 py-3 font-medium">Ice Vol</th>
              <th className="text-right px-4 py-3 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {pageItems.map(r => (
              <tr
                key={r.id}
                className="border-b border-space-800/50 hover:bg-space-800/40 cursor-pointer transition-colors"
                onClick={() => onSelect(r.id)}
              >
                <td className="px-4 py-3 font-mono text-xs">#{r.id}</td>
                <td className="px-4 py-3 text-xs text-slate-400">
                  {r.created_at?.slice(0, 19).replace('T', ' ')}
                </td>
                <td className="px-4 py-3">
                  <span className={`badge badge-${r.status === 'success' ? 'success' : r.status === 'failed' ? 'failed' : 'running'}`}>
                    {statusIcon(r.status)} {r.status}
                  </span>
                </td>
                <td className="px-4 py-3 font-mono text-xs text-slate-400">
                  {r.duration_s ? `${r.duration_s.toFixed(1)}s` : '—'}
                </td>
                <td className="px-4 py-3 text-xs text-slate-300">{r.dsc_name || '—'}</td>
                <td className="px-4 py-3 font-mono text-xs text-lunar-cyan">
                  {r.ice_volume_m3 ? `${r.ice_volume_m3.toExponential(2)} m³` : '—'}
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    className="btn btn-ghost text-[10px] px-2 py-1"
                    onClick={e => { e.stopPropagation(); onDelete(r.id) }}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
            {pageItems.length === 0 && (
              <tr><td colSpan={7} className="px-4 py-12 text-center text-slate-500 text-sm">
                {runs.length === 0 ? 'No runs yet. Click Launch to start.' : 'No matches.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {pages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-space-700/50">
          <span className="text-xs text-slate-500">
            Page {safePage + 1} of {pages}
          </span>
          <div className="flex gap-1">
            <button className="btn btn-ghost text-xs px-2 py-1" disabled={safePage === 0} onClick={() => setPage(p => p - 1)}>←</button>
            <button className="btn btn-ghost text-xs px-2 py-1" disabled={safePage >= pages - 1} onClick={() => setPage(p => p + 1)}>→</button>
          </div>
        </div>
      )}
    </div>
  )
}
