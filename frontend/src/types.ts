export interface StageResult {
  stage_index: number
  stage_name: string
  status: 'running' | 'success' | 'failed'
  log_output: string | null
}

export interface FigureRecord {
  filename: string
  title: string
  url: string
}

export interface GisFile {
  filename: string
  url: string
  format: 'GeoTIFF' | 'GeoJSON'
}

export interface PipelineRun {
  id: number
  created_at: string
  status: 'running' | 'success' | 'failed'
  dsc_name: string | null
  ice_volume_m3: number | null
  n_candidates: number
  n_dsc_craters: number
  dash_feasible: boolean | null
  slam_mae: number | null
  efpi_ice_pct: number | null
  error_message: string | null
  duration_s: number | null
  stages: StageResult[]
  figures: FigureRecord[]
  gis_files: GisFile[]
}

export interface RunSummary {
  id: number
  created_at: string
  status: 'running' | 'success' | 'failed'
  dsc_name: string | null
  ice_volume_m3: number | null
  dash_feasible: boolean | null
  duration_s: number | null
}

export interface WsMessage {
  type: 'log' | 'stage_status' | 'run_status'
  line?: string
  stage_index?: number
  status?: string
  duration_s?: number
  error_message?: string
}
