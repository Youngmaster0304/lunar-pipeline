import { useEffect, useRef } from 'react'
import type { RunSummary } from '../types'

function css(key: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(key).trim()
}

interface Props { runs: RunSummary[] }

export default function TrendChart({ runs }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const W = c.clientWidth * dpr
    const H = c.clientHeight * dpr
    c.width = W
    c.height = H
    ctx.scale(dpr, dpr)
    const w = c.clientWidth
    const h = c.clientHeight

    const completed = runs
      .filter(r => r.duration_s && r.duration_s > 0 && r.status === 'success')
      .slice(-20)

    if (completed.length < 2) {
      ctx.fillStyle = '#475569'
      ctx.font = '12px "Plus Jakarta Sans", sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Need 2+ completed runs for trend', w / 2, h / 2)
      return
    }

    const vals = completed.map(r => r.duration_s!)
    const min = Math.min(...vals) * 0.9
    const max = Math.max(...vals) * 1.1
    const pad = { top: 16, bottom: 24, left: 40, right: 16 }
    const cw = w - pad.left - pad.right
    const ch = h - pad.top - pad.bottom

    // Grid lines
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
      ctx.fillStyle = '#475569'
      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.textAlign = 'right'
      ctx.fillText((max - (max - min) * (i / 4)).toFixed(1), pad.left - 6, y + 3)
    }

    // Line
    ctx.beginPath()
    ctx.strokeStyle = '#06b6d4'
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    for (let i = 0; i < vals.length; i++) {
      const x = pad.left + (i / (vals.length - 1)) * cw
      const y = pad.top + ch - ((vals[i] - min) / (max - min)) * ch
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Dots
    for (let i = 0; i < vals.length; i++) {
      const x = pad.left + (i / (vals.length - 1)) * cw
      const y = pad.top + ch - ((vals[i] - min) / (max - min)) * ch
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fillStyle = '#06b6d4'
      ctx.fill()
    }
  }, [runs])

  return (
    <div className="card p-4">
      <h3 className="text-xs font-display text-lunar-cyan/70 uppercase tracking-widest mb-3">
        Performance Trend
      </h3>
      <canvas ref={canvas} className="w-full h-32" />
    </div>
  )
}
