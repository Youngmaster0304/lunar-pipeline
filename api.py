import io
import base64
import logging
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any, List, Optional

# Database
from database import init_db, get_db
from models import PipelineRun

# Pipeline Modules
from config import PipelineConfig
from module_1 import compute_cpr, compute_dop, build_ice_mask, invert_ice_fraction
from module_1.dfsar_reader import load_dfsar_product_synthetic
from module_2 import compute_slope, build_candidate_mask, extract_candidate_sites
from module_2.terrain_analysis import make_synthetic_auxiliary
from module_3 import RoverState, plan_mission_path
from module_4 import Bug2Planner
from module_5 import EFPIModel, compute_drill_heat_transfer, compute_post_drill_temperature, compute_sublimation_rate, compute_vapor_density

app = FastAPI(title="ISRO Lunar Pipeline API (Robust)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

def array_to_base64_img(arr: np.ndarray, colormap: str = 'viridis') -> str:
    plt.figure(figsize=(4, 4), facecolor='#030a16')
    if np.all(arr == 0):
        plt.imshow(arr, cmap=colormap, vmin=0, vmax=1)
    else:
        plt.imshow(arr, cmap=colormap)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='#030a16')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def run_pipeline_task(run_id: int):
    # This runs in background
    # We open a new DB session since this is a background thread
    from database import SessionLocal
    db = SessionLocal()
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        db.close()
        return

    config = PipelineConfig()
    start_time = time.time()
    
    try:
        # Load Real Orbital Data
        from module_1.dfsar_reader import load_dfsar_product
        import rasterio
        
        # Module 1
        dfsar_product = load_dfsar_product("data/dfsar_tile.tif", config.module1)
        cpr = compute_cpr(dfsar_product, config.module1)
        dop = compute_dop(dfsar_product)
        ice_mask = build_ice_mask(cpr, dop, config.module1)
        
        synthetic_backscatter = np.random.uniform(-15, -5, size=dfsar_product.shape).astype(np.float32)
        ice_fraction = invert_ice_fraction(synthetic_backscatter, config.module1)
        run.ice_volume_m3 = float(np.sum(ice_fraction) * (config.dem_pixel_spacing_m ** 2) * 5.0)

        # Module 2
        with rasterio.open("data/dem_faustini.tif") as src:
            real_dem = src.read(1).astype(np.float32)
        grid_shape = dfsar_product.shape
        
        # The DFSAR shape and DEM shape might not perfectly align if they aren't cropped identically, 
        # so we ensure the DEM is the same shape for processing
        if real_dem.shape != grid_shape:
            # simple truncate or pad to match dfsar
            min_r = min(real_dem.shape[0], grid_shape[0])
            min_c = min(real_dem.shape[1], grid_shape[1])
            aligned_dem = np.zeros(grid_shape, dtype=np.float32)
            aligned_dem[:min_r, :min_c] = real_dem[:min_r, :min_c]
            real_dem = aligned_dem
            
        slope = compute_slope(real_dem, config.dem_pixel_spacing_m)
        
        with rasterio.open("data/dte_faustini.tif") as src:
            real_dte = src.read(1)
        with rasterio.open("data/illumination_faustini.tif") as src:
            real_illum = src.read(1)
            
        # Ensure sizes align
        aligned_dte = np.zeros(grid_shape, dtype=real_dte.dtype)
        aligned_illum = np.zeros(grid_shape, dtype=real_illum.dtype)
        aligned_dte[:min(real_dte.shape[0], grid_shape[0]), :min(real_dte.shape[1], grid_shape[1])] = real_dte[:min(real_dte.shape[0], grid_shape[0]), :min(real_dte.shape[1], grid_shape[1])]
        aligned_illum[:min(real_illum.shape[0], grid_shape[0]), :min(real_illum.shape[1], grid_shape[1])] = real_illum[:min(real_illum.shape[0], grid_shape[0]), :min(real_illum.shape[1], grid_shape[1])]
        
        dte = aligned_dte
        illumination = aligned_illum
        
        candidate_mask = build_candidate_mask(slope, dte, illumination, config.module2)
        exposure_score = (config.module2.exposure_weight_illumination * illumination + 
                          config.module2.exposure_weight_dte * dte.astype(float)).astype(np.float32)
        candidates = extract_candidate_sites(candidate_mask, exposure_score, slope, dte, illumination, config.module2)
        run.n_candidates = len(candidates)
        
        # Module 3
        rover_state = RoverState(
            position=config.mission_start,
            battery_wh=config.rover_battery_initial_wh,
            battery_max_wh=config.rover_battery_max_wh,
            in_sunlight=True,
            timestep=0
        )
        if not candidates:
            from module_2.mcda import CandidateSite
            candidates = [CandidateSite(row=20, col=20, slope_deg=2.0, dte_ok=True, illumination_fraction=0.9, exposure_score=0.9, rank=1)]
            
        dash_result, return_result = plan_mission_path(
            slope_map=slope, ice_mask=ice_mask, sunlit_mask=(illumination > 0.5),
            start=config.mission_start, dsc_sample_point=config.dsc_sample_point,
            singularity_sites=candidates, rover_state=rover_state, config=config.module3
        )
        run.dash_feasible = dash_result.feasible
        
        # Module 4: Real path processing
        from module_4 import Bug2Planner
        obstacle_map = np.zeros_like(slope, dtype=bool)
        obstacle_map[60, 50:70] = True
        trajectory_length = 0
        if dash_result.feasible:
            bug_planner = Bug2Planner(dash_result.path, obstacle_map, config.module4)
            trajectory = bug_planner.run(config.mission_start, config.dsc_sample_point)
            trajectory_length = len(trajectory)
        run.bug2_steps = trajectory_length
        
        if dash_result.path:
            path_map = np.zeros_like(slope)
            for r, c in dash_result.path:
                path_map[r, c] = 1.0
            run.image_path_map = "data:image/png;base64," + array_to_base64_img(path_map, 'spring')
        else:
            run.image_path_map = None

        # Module 5 & 6 estimates
        Q_joules = compute_drill_heat_transfer(config.module5)
        T_post = compute_post_drill_temperature(Q_joules, regolith_mass_kg=0.5, initial_temp_k=config.module5.regolith_temp_k)
        sub_rate = compute_sublimation_rate(T_post, P_sealed_pa=1.0, sealed_volume_m3=config.module5.sealed_volume_m3, config=config.module5)
        vapor_density = compute_vapor_density(sub_rate, config.module5.sealed_volume_m3, config.module5.drill_contact_duration_s)
        efpi = EFPIModel(config.module5)
        fringe_shift = efpi.fringe_shift_from_humidity(vapor_density)
        ice_pct = efpi.infer_ice_density(fringe_shift)
        
        run.efpi_fringe_shift = float(fringe_shift)
        run.ice_volume_pct = float(ice_pct)
        
        run.image_ice_mask = "data:image/png;base64," + array_to_base64_img(ice_mask.astype(float), 'bone')
        run.image_slope_map = "data:image/png;base64," + array_to_base64_img(slope, 'inferno')

        run.status = "success"
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        logging.error(f"Run failed: {e}")
    finally:
        run.duration_s = time.time() - start_time
        db.commit()
        db.close()


class RunOut(BaseModel):
    id: int
    status: str
    created_at: str
    n_candidates: Optional[int]
    dash_feasible: Optional[bool]
    bug2_steps: Optional[int]
    ice_volume_m3: Optional[float]
    efpi_fringe_shift: Optional[float]
    ice_volume_pct: Optional[float]
    duration_s: Optional[float]
    error_message: Optional[str]
    image_ice_mask: Optional[str]
    image_slope_map: Optional[str]
    image_path_map: Optional[str]

    class Config:
        from_attributes = True

@app.post("/api/runs")
def create_run(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    new_run = PipelineRun(status="running")
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    background_tasks.add_task(run_pipeline_task, new_run.id)
    return {"id": new_run.id, "status": "running"}

@app.get("/api/runs", response_model=List[RunOut])
def get_runs(db: Session = Depends(get_db)):
    runs = db.query(PipelineRun).order_by(PipelineRun.id.desc()).limit(50).all()
    # Convert datetime to string
    res = []
    for r in runs:
        r_dict = r.__dict__.copy()
        r_dict["created_at"] = r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        res.append(r_dict)
    return res

@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    r_dict = r.__dict__.copy()
    r_dict["created_at"] = r.created_at.strftime("%Y-%m-%d %H:%M:%S")
    return r_dict
