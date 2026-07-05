import type { PipelineRun, RunSummary, WsMessage } from './types'
console.log("CACHE BREAKER: V3 Active");

const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
const API = isLocal 
  ? (import.meta.env.VITE_API_URL || 'http://localhost:8000')
  : 'https://abhinavjha0304-lunar-pipeline.hf.space'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 200)}`)
  }
  return res.json()
}

export function createRun(): Promise<{ id: number; status: string }> {
  return req('/api/runs', { method: 'POST' })
}

export function listRuns(): Promise<RunSummary[]> {
  return req('/api/runs')
}

export function getRun(id: number): Promise<PipelineRun> {
  return req(`/api/runs/${id}`)
}

export function deleteRun(id: number): Promise<void> {
  return req(`/api/runs/${id}`, { method: 'DELETE' })
}

export function getHealth(): Promise<{ status: string; version: string }> {
  return req('/api/health')
}

export function downloadUrl(id: number, format: 'json' | 'csv'): string {
  return `${API}/api/runs/${id}/download/${format}`
}

export function connectWs(
  runId: number,
  onMessage: (msg: WsMessage) => void,
  onClose?: () => void,
): () => void {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsBase = API ? API.replace(/^http/, 'ws') : `${protocol}//${location.host}`
  const ws = new WebSocket(`${wsBase}/ws/runs/${runId}`)

  ws.onmessage = (e) => {
    try {
      const msg: WsMessage = JSON.parse(e.data)
      onMessage(msg)
    } catch { /* ignore malformed */ }
  }
  ws.onclose = () => onClose?.()
  ws.onerror = () => ws.close()

  return () => ws.close()
}
