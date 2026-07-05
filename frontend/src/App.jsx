import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import { createRun, listRuns, getRun, getHealth } from './api';
import './index.css';

// Mock Data Generators for Dashboard Visuals
const mockRadarData = Array.from({ length: 20 }, (_, i) => ({
  index: i,
  volume: (1.5 + Math.random() * 0.5).toFixed(2)
}));

const mockPathData = [
  { x: 2.5, y: 2.0 },
  { x: 5.0, y: 3.0 },
  { x: 11.0, y: 7.0 },
  { x: 14.0, y: 10.0 },
  { x: 15.0, y: 15.0 },
  { x: 17.5, y: 17.0 }
];

const mockSafestPathData = [
  { x: 2.5, y: 2.0 },
  { x: 5.0, y: 0.0 },
  { x: 8.0, y: 0.0 },
  { x: 10.0, y: 4.0 },
  { x: 14.0, y: 6.0 },
  { x: 15.0, y: 17.0 },
  { x: 17.5, y: 17.0 }
];

function App() {
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState([]);
  const [activeRun, setActiveRun] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('OFFLINE');
  const pollInterval = useRef(null);

  const fetchRuns = async () => {
    try {
      const data = await listRuns();
      setRuns(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError(err.message);
    }
  };

  const checkHealth = async () => {
    try {
      const h = await getHealth();
      setStatus(h.status === 'ok' ? 'ONLINE' : 'ERROR');
    } catch {
      setStatus('OFFLINE');
    }
  };

  useEffect(() => {
    fetchRuns();
    checkHealth();
    const interval = setInterval(fetchRuns, 5000);
    const hInterval = setInterval(checkHealth, 10000);
    return () => {
      clearInterval(interval);
      clearInterval(hInterval);
    };
  }, []);

  const runPipeline = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await createRun();
      
      // Start polling for this specific run
      if (pollInterval.current) clearInterval(pollInterval.current);
      pollInterval.current = setInterval(async () => {
        try {
          const runData = await getRun(data.id);
          setActiveRun(runData);
          window.scrollTo({ top: 0, behavior: 'smooth' });
          if (runData.status !== 'running') {
            clearInterval(pollInterval.current);
            setLoading(false);
            fetchRuns(); // update table
          }
        } catch (e) {
          console.error('Polling error', e);
        }
      }, 1000);

    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const viewRun = async (id) => {
    try {
      const data = await getRun(id);
      setActiveRun(data);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="mission-control">
      <header className="mc-header">
        <div className="mc-header-left">
          <h1 className="mc-title">LSP EXPLORATION PIPELINE</h1>
          <div className="mc-subtitle">MISSION CONTROL</div>
        </div>
        
        <button className="mc-launch-btn" onClick={runPipeline} disabled={loading} style={{ marginRight: '1rem' }}>
          {loading ? <><span className="mc-loader"></span> EXECUTING...</> : 'INITIATE PIPELINE'}
        </button>
        <button className="mc-launch-btn" onClick={() => window.print()} style={{ background: 'transparent', borderColor: 'var(--mc-text-dim)', color: 'var(--mc-text-main)' }}>
          EXPORT PDF
        </button>
      </header>

      {error && (
        <div style={{ color: 'var(--mc-red)', fontFamily: 'var(--font-mono)', padding: '1rem', border: '1px solid var(--mc-red)', margin: '1rem' }}>
          SYSTEM ERROR: {error}
        </div>
      )}

      {activeRun && activeRun.status === 'success' && (
        <div className="mc-grid" id="report-content">
          {/* PANEL A */}
          <div className="mc-panel">
            <div className="mc-panel-header">
              <h2 className="mc-panel-title">A. DFSAR Radar Polarimetry</h2>
              <span className="mc-panel-stage">MODULE 01</span>
            </div>
            <div className="mc-panel-content">
              <div className="mc-data-section">
                <div className="mc-objective" style={{ marginBottom: '1rem' }}>
                  <strong>OBJECTIVE:</strong> Process Chandrayaan-2 DFSAR data to compute Circular Polarization Ratio (CPR) and evaluate top 5m volumetric ice.
                </div>
                <div className="mc-data-row"><span className="mc-data-label">Target Anomaly</span><span className="mc-data-value cyan">CPR &gt; 1.0</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Depolarization</span><span className="mc-data-value cyan">DOP &lt; 0.13</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Ice Vol (Top 5m)</span><span className="mc-data-value amber">{activeRun.ice_volume_m3 ? activeRun.ice_volume_m3.toExponential(2) : 0} m³</span></div>
              </div>
              <div className="mc-viz-section">
                {activeRun.image_ice_mask ? <img src={activeRun.image_ice_mask} className="mc-viz-image" alt="Ice Mask" /> : <div style={{padding:'2rem', color:'var(--mc-text-dim)'}}>NO IMAGE DATA</div>}
                <div className="mc-viz-overlay"></div>
              </div>
            </div>
          </div>

          {/* PANEL B */}
          <div className="mc-panel">
            <div className="mc-panel-header">
              <h2 className="mc-panel-title">B. Landing Site Seeding</h2>
              <span className="mc-panel-stage">MODULE 02</span>
            </div>
            <div className="mc-panel-content">
              <div className="mc-data-section">
                <div className="mc-objective" style={{ marginBottom: '1rem' }}>
                  <strong>OBJECTIVE:</strong> Perform Terrain MCDA on DEM topography to locate safe landing zones on the crater rim.
                </div>
                <div className="mc-data-row"><span className="mc-data-label">Sites Found</span><span className="mc-data-value cyan">{activeRun.n_candidates}</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Visibility</span><span className="mc-data-value green">High DTE</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Max Slope</span><span className="mc-data-value green">&lt; 10°</span></div>
              </div>
              <div className="mc-viz-section">
                {activeRun.image_slope_map ? <img src={activeRun.image_slope_map} className="mc-viz-image" alt="Slope Map" /> : <div style={{padding:'2rem', color:'var(--mc-text-dim)'}}>NO IMAGE DATA</div>}
                <div className="mc-viz-overlay"></div>
              </div>
            </div>
          </div>

          {/* PANEL C (TRAVERSE WITH RECHARTS) */}
          <div className="mc-panel" style={{ gridColumn: '1 / -1' }}>
            <div className="mc-panel-header">
              <h2 className="mc-panel-title">C. Rover Traverse Path Planner</h2>
              <span className="mc-panel-stage">MODULE 03 & 04</span>
            </div>
            <div className="mc-panel-content" style={{ flexDirection: 'row', minHeight: '300px' }}>
              <div className="mc-data-section" style={{ flex: '0 0 350px' }}>
                <div className="mc-objective" style={{ marginBottom: '1rem' }}>
                  <strong>OBJECTIVE:</strong> Compute risk-aware A* path enforcing 15% Battery SoC constraint into shadowed crater with Bug-2 SLAM.
                </div>
                <div className="mc-data-row"><span className="mc-data-label">A* Dash Feasible</span><span className={`mc-data-value ${activeRun.dash_feasible ? 'green' : 'red'}`}>{activeRun.dash_feasible ? 'CONFIRMED' : 'REJECTED'}</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Bug-2 Nav Steps</span><span className="mc-data-value amber">{activeRun.bug2_steps} ops</span></div>
              </div>
              <div style={{ flex: 1, position: 'relative' }}>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#14355e" />
                    <XAxis type="number" dataKey="x" domain={[0, 20]} stroke="#7090b0" />
                    <YAxis type="number" dataKey="y" domain={[0, 20]} reversed stroke="#7090b0" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#030a16', borderColor: '#14355e' }} />
                    <Legend />
                    <Scatter name="Efficient" data={mockPathData} fill="#ffaa00" line={{ stroke: '#ffaa00', strokeWidth: 3 }} />
                    <Scatter name="Safest" data={mockSafestPathData} fill="#00ff88" line={{ stroke: '#00ff88', strokeWidth: 3 }} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* PANEL D (BAR CHART WITH RECHARTS) */}
          <div className="mc-panel" style={{ gridColumn: '1 / -1' }}>
            <div className="mc-panel-header">
              <h2 className="mc-panel-title">D. CPR / DOP Radar Signature Panel — Ice Volume Estimate (0–5m)</h2>
              <span className="mc-panel-stage">MODULE 05 & 06</span>
            </div>
            <div className="mc-panel-content" style={{ flexDirection: 'row', minHeight: '250px' }}>
              <div className="mc-data-section" style={{ flex: '0 0 350px' }}>
                <div className="mc-objective" style={{ marginBottom: '1rem' }}>
                  <strong>OBJECTIVE:</strong> Estimate subsurface ice concentration within the top ~5 meters using radar backscatter and dielectric assumptions.
                </div>
                <div className="mc-data-row"><span className="mc-data-label">Radar Model</span><span className="mc-data-value cyan">CRIM Mixing</span></div>
                <div className="mc-data-row"><span className="mc-data-label">Avg Concentration</span><span className="mc-data-value green">{activeRun.ice_volume_pct !== undefined && activeRun.ice_volume_pct !== null ? (activeRun.ice_volume_pct * 100).toFixed(2) : 0} %</span></div>
              </div>
              <div style={{ flex: 1, position: 'relative' }}>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={mockRadarData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#14355e" vertical={false} />
                    <XAxis dataKey="index" stroke="#7090b0" tickLine={false} axisLine={false} />
                    <YAxis domain={[0, 4.5]} stroke="#7090b0" tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: 'rgba(0, 240, 255, 0.1)' }} contentStyle={{ backgroundColor: '#030a16', borderColor: '#14355e' }} />
                    <Bar dataKey="volume" fill="#00f0ff" radius={[4, 4, 0, 0]} barSize={20} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeRun && activeRun.status === 'running' && (
        <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--mc-cyan)', fontFamily: 'var(--font-mono)', fontSize: '1.2rem', letterSpacing: '0.1em' }}>
          <span className="mc-loader" style={{ display: 'inline-block', marginRight: '1rem' }}></span>
          PIPELINE EXECUTING (RUN #{activeRun.id}) ...
        </div>
      )}

      {/* Historical Runs Table */}
      <div className="historical-runs-table" style={{ margin: '2rem 0', border: '1px solid var(--mc-border)', background: 'var(--mc-bg-card)', padding: '1.5rem' }}>
        <h3 style={{ color: 'var(--mc-text-main)', fontFamily: 'var(--font-mono)', marginBottom: '1rem', fontWeight: 600 }}>HISTORICAL RUNS</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--mc-border)', color: 'var(--mc-text-muted)' }}>
              <th style={{ padding: '12px 8px' }}>RUN ID</th>
              <th style={{ padding: '12px 8px' }}>TIMESTAMP</th>
              <th style={{ padding: '12px 8px' }}>STATUS</th>
              <th style={{ padding: '12px 8px' }}>ICE VOL (m³)</th>
              <th style={{ padding: '12px 8px' }}>TRAVERSE</th>
              <th style={{ padding: '12px 8px' }}>DURATION</th>
              <th style={{ padding: '12px 8px' }}>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 ? (
              <tr><td colSpan="7" style={{ padding: '20px', textAlign: 'center', color: 'var(--mc-text-muted)' }}>No historical runs found.</td></tr>
            ) : (
              runs.map(r => (
                <tr key={r.id} style={{ borderBottom: '1px solid rgba(0, 195, 255, 0.1)', cursor: 'pointer', transition: 'background 0.2s' }} onClick={() => viewRun(r.id)}>
                  <td style={{ padding: '12px 8px', color: 'var(--mc-cyan)' }}>#{r.id}</td>
                  <td style={{ padding: '12px 8px', color: 'var(--mc-text-main)' }}>{r.created_at}</td>
                  <td style={{ padding: '12px 8px' }}>
                    <span style={{ color: r.status === 'success' ? 'var(--mc-green)' : (r.status === 'running' ? 'var(--mc-amber)' : 'var(--mc-red)') }}>
                      {r.status.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: '12px 8px', color: 'var(--mc-text-main)' }}>{r.ice_volume_m3 ? r.ice_volume_m3.toExponential(2) : '—'}</td>
                  <td style={{ padding: '12px 8px' }}>{r.dash_feasible ? <span style={{color:'var(--mc-green)'}}>FEASIBLE</span> : (r.dash_feasible === false ? <span style={{color:'var(--mc-red)'}}>INFEASIBLE</span> : '—')}</td>
                  <td style={{ padding: '12px 8px', color: 'var(--mc-text-muted)' }}>{r.duration_s ? r.duration_s.toFixed(1) + 's' : '—'}</td>
                  <td style={{ padding: '12px 8px' }}>
                    <button style={{ background: 'none', border: '1px solid var(--mc-cyan)', color: 'var(--mc-cyan)', padding: '4px 12px', fontSize: '0.7rem', cursor: 'pointer' }} onClick={(e) => {e.stopPropagation(); viewRun(r.id);}}>VIEW</button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <footer className="mc-status-bar">
        <div className="mc-status-item">
          <div className={`mc-status-indicator ${status === 'ONLINE' ? 'active' : 'inactive'}`}></div>
          SYSTEM STATUS: {status}
        </div>
        <div>
          LUNAR SOUTH POLE
        </div>
      </footer>
    </div>
  );
}

export default App;
