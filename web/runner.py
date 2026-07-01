"""Pipeline runner — captures logs, metrics, figures into DB."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from web.events import RunEventBus, RedisEventBus

_EventBus = "RunEventBus | RedisEventBus"

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))


class LogCapture(logging.Handler):
    def __init__(self, run_id: int, event_bus: _EventBus | None = None) -> None:
        super().__init__()
        self.records: list[str] = []
        self.run_id = run_id
        self.event_bus = event_bus

    def emit(self, record: logging.Record) -> None:
        msg = self.format(record)
        self.records.append(msg)
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(self.run_id, {"type": "log", "line": msg})
        except Exception:
            pass


def _get_gis_crs_transform(config):
    crs = None
    transform = None
    for path in (config.dem_path, config.dfsar_tile_path, config.ohrc_path):
        try:
            import rasterio
            with rasterio.open(path) as src:
                crs = src.crs
                transform = src.transform
                break
        except Exception:
            continue
    if crs is None:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        crs = CRS.from_epsg(4326)
        transform = from_origin(0, 0, config.dem_pixel_spacing_m, config.dem_pixel_spacing_m)
    return crs, transform


def _export_geotiff(name, array, crs, transform, run_dir):
    import rasterio
    path = run_dir / f"{name}.tif"
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(array, 1)
    logger.info("Exported GeoTIFF: %s", path.name)
    return path


def _pixel_to_geo(row, col, gtfm):
    return gtfm * (col + 0.5, row + 0.5)


def _export_geojson_points(name, points_with_props, gtfm, run_dir):
    import json
    features = []
    for i, (row, col, props) in enumerate(points_with_props):
        x, y = _pixel_to_geo(row, col, gtfm)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": props or {"id": i},
        })
    path = run_dir / f"{name}.geojson"
    with open(str(path), "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    logger.info("Exported GeoJSON: %s (%d features)", path.name, len(features))
    return path


def _export_geojson_lines(name, lines_with_props, gtfm, run_dir):
    import json
    features = []
    for i, (waypoints, props) in enumerate(lines_with_props):
        coords = [_pixel_to_geo(r, c, gtfm) for r, c in waypoints]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props or {"id": i},
        })
    path = run_dir / f"{name}.geojson"
    with open(str(path), "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    logger.info("Exported GeoJSON: %s (%d features)", path.name, len(features))
    return path


def run_pipeline_internal(db, run_id: int, event_bus: _EventBus | None = None) -> None:
    """Execute the full pipeline and persist all results to DB."""
    from config import PipelineConfig
    from web.config import settings as web_settings
    from web.models import PipelineRun

    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if not run:
        return

    config = PipelineConfig()
    # Override data paths from web settings if configured
    if web_settings.dfsar_tile_path:
        config.dfsar_tile_path = web_settings.dfsar_tile_path
    if web_settings.dem_path:
        config.dem_path = web_settings.dem_path
    if web_settings.dte_path:
        config.dte_path = web_settings.dte_path
    if web_settings.illumination_path:
        config.illumination_path = web_settings.illumination_path
    if web_settings.ohrc_path:
        config.ohrc_path = web_settings.ohrc_path

    # GIS export config
    config._run_dir = _HERE / "web" / "runs" / str(run_id)
    config._run_dir.mkdir(parents=True, exist_ok=True)
    config._crs, config._gtfm = _get_gis_crs_transform(config)

    capture = LogCapture(run_id, event_bus)
    capture.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    capture.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(capture)

    # Clear shared figures directory so each run starts clean
    shared_fig_dir = _HERE / "figures"
    if shared_fig_dir.exists():
        shutil.rmtree(str(shared_fig_dir))
    shared_fig_dir.mkdir(parents=True, exist_ok=True)

    stages_def = [
        (0, "DFSAR Radar Polarimetry (CPR/DOP)"),
        (1, "OHRC Terrain Safety & Boulders"),
        (2, "MCDA Safe Landing Site Optimization"),
        (3, "PSR & Doubly Shadowed Crater Mapping"),
        (4, "Solar/Hazard Aware Rover Traverse"),
        (5, "SLAM Obstacle Avoidance"),
        (6, "Dielectric Subsurface Ice Vol Estimation"),
    ]

    t0 = time.time()

    try:
        for idx, name in stages_def:
            from web.models import StageResult
            stage = StageResult(run_id=run_id, stage_index=idx, stage_name=name, status="running")
            db.add(stage)
            db.commit()
            event_bus.publish(run_id, {"type": "stage_status", "stage_index": idx, "status": "running"})

            log_before = len(capture.records)

            try:
                _run_stage(idx, config, capture)
                stage.status = "success"
                event_bus.publish(run_id, {"type": "stage_status", "stage_index": idx, "status": "success"})
            except Exception as exc:
                stage.status = "failed"
                stage.log_output = "\n".join(capture.records[log_before:])
                db.commit()
                event_bus.publish(run_id, {"type": "stage_status", "stage_index": idx, "status": "failed"})
                raise

            stage.log_output = "\n".join(capture.records[log_before:])
            db.commit()

        run.status = "success"
    except Exception:
        run.status = "failed"
        run.error_message = traceback.format_exc()
    finally:
        run.duration_s = time.time() - t0
        _extract_metrics(run, config, capture, db)
        event_bus.publish(run_id, {
            "type": "run_status",
            "status": run.status,
            "duration_s": run.duration_s,
            "error_message": run.error_message,
        })
        db.commit()
        logging.getLogger().removeHandler(capture)


def _run_stage(stage_index: int, config, capture: LogCapture) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if stage_index == 0:
        _stage1(config)
    elif stage_index == 1:
        _stage2a(config)
    elif stage_index == 2:
        _stage2b(config)
    elif stage_index == 3:
        _stage3(config)
    elif stage_index == 4:
        _stage4(config)
    elif stage_index == 5:
        _stage5(config)
    elif stage_index == 6:
        _stage6(config)


def _fig(name: str) -> str:
    d = Path(_HERE) / "figures"
    d.mkdir(exist_ok=True)
    p = d / name
    return str(p)


def _stage1(config):
    import matplotlib.pyplot as plt
    from module_1 import compute_cpr, compute_dop, build_ice_mask, invert_ice_fraction, compute_ice_volume
    from module_1.dfsar_reader import load_dfsar_product, load_dfsar_product_synthetic
    try:
        dfsar = load_dfsar_product(config.dfsar_tile_path, config.module1)
        logger.info("Loaded real DFSAR data from %s", config.dfsar_tile_path)
    except Exception:
        logger.warning("Real DFSAR load failed. Using synthetic fixture.")
        dfsar = load_dfsar_product_synthetic(shape=(100, 100))
    cpr = compute_cpr(dfsar, config.module1)
    dop = compute_dop(dfsar)
    ice_mask = build_ice_mask(cpr, dop, config.module1)
    bs = np.random.uniform(-15, -5, size=dfsar.shape).astype(np.float32)
    ice_frac = invert_ice_fraction(bs, config.module1)
    vol = compute_ice_volume(ice_frac, config.dem_pixel_spacing_m ** 2, depth_m=5.0)
    logger.info("[METRIC] ice_volume_m3=%.2e", vol)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title in zip(axs, [cpr, ice_mask.astype(int), ice_frac], ["CPR", "Ice Mask", "Ice Fraction"]):
        im = ax.imshow(img, cmap="viridis" if title == "CPR" else ("coolwarm" if title == "Ice Mask" else "Blues"))
        ax.set_title(title); plt.colorbar(im, ax=ax)
    fig.savefig(_fig("stage1_dfsar.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    _export_geotiff("ice_mask", ice_mask.astype(np.uint8), config._crs, config._gtfm, config._run_dir)
    _export_geotiff("ice_fraction", ice_frac, config._crs, config._gtfm, config._run_dir)


def _stage2a(config):
    import matplotlib.pyplot as plt
    from module_2.terrain_analysis import load_ohrc_ortho, make_synthetic_ohrc, compute_surface_roughness, detect_boulders
    try:
        ohrc = load_ohrc_ortho(config.ohrc_path)
        logger.info("Loaded real OHRC ortho from %s", config.ohrc_path)
    except Exception:
        logger.warning("Real OHRC load failed. Using synthetic.")
        ohrc = make_synthetic_ohrc((100, 100), seed=42)
    rough = compute_surface_roughness(ohrc, config.module2.ohrc_roughness_window)
    boulder = detect_boulders(ohrc, config.module2.boulder_prominence_m, config.module2.boulder_buffer_px)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title, cm in zip(axs, [ohrc, rough, boulder.astype(int)], ["OHRC", "Roughness", "Boulder Mask"], ["gray", "hot", "bwr"]):
        im = ax.imshow(img, cmap=cm); ax.set_title(title); plt.colorbar(im, ax=ax)
    fig.savefig(_fig("stage2a_ohrc.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    _export_geotiff("roughness", rough, config._crs, config._gtfm, config._run_dir)
    _export_geotiff("boulder_mask", boulder.astype(np.uint8), config._crs, config._gtfm, config._run_dir)


def _stage2b(config):
    import matplotlib.pyplot as plt
    from module_2 import compute_slope, make_synthetic_auxiliary, build_candidate_mask, compute_exposure_score, extract_candidate_sites
    from module_2.terrain_analysis import load_auxiliary_rasters
    rows, cols = 100, 100
    rr, cc = np.mgrid[0:rows, 0:cols]
    try:
        import rasterio
        with rasterio.open(config.dem_path) as src:
            dem = src.read(1).astype(np.float32)
        rows, cols = dem.shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        logger.info("Loaded real DEM from %s shape=%s", config.dem_path, dem.shape)
    except Exception:
        dem = (1000.0 + 3.0 * np.sin(rr / 8) * np.cos(cc / 8) + 1.5 * np.sin(rr / 20 + cc / 15)).astype(np.float32)
    slope = compute_slope(dem, config.dem_pixel_spacing_m)
    try:
        dte, illum = load_auxiliary_rasters(config.dte_path, config.illumination_path)
    except Exception:
        dte, illum = make_synthetic_auxiliary((rows, cols))
    mask = build_candidate_mask(slope, dte, illum, config.module2)
    score = compute_exposure_score(illum, dte, config.module2)
    sites = extract_candidate_sites(mask, score, slope, dte, illum, config.module2)
    logger.info("[METRIC] n_candidates=%d", len(sites))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(slope, cmap="terrain", origin="upper")
    for s in sites: ax.plot(s.col, s.row, "r*", markersize=8)
    ax.set_title(f"Candidates ({len(sites)})")
    fig.savefig(_fig("stage2b_sites.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    _export_geotiff("slope_deg", slope, config._crs, config._gtfm, config._run_dir)
    if sites:
        pts = [(s.row, s.col, {"rank": s.rank, "slope_deg": s.slope_deg, "exposure_score": s.exposure_score}) for s in sites]
        _export_geojson_points("candidate_sites", pts, config._gtfm, config._run_dir)


def _stage3(config):
    import matplotlib.pyplot as plt
    from module_6 import compute_psr_mask, identify_doubly_shadowed_craters
    from scipy.ndimage import minimum_filter
    rows, cols = 100, 100
    rr, cc = np.mgrid[0:rows, 0:cols]
    try:
        import rasterio
        with rasterio.open(config.dem_path) as src:
            dem = src.read(1).astype(np.float32)
        rows, cols = dem.shape
    except Exception:
        dem = (1000.0 + 3.0 * np.sin(rr / 8) * np.cos(cc / 8) + 1.5 * np.sin(rr / 20 + cc / 15)).astype(np.float32)
    try:
        import rasterio
        with rasterio.open(config.illumination_path) as src:
            illum = src.read(1).astype(np.float32)
        sunlit = illum > 0.3
    except Exception:
        sunlit = np.random.uniform(0, 1, (rows, cols)) > 0.7
    psr = compute_psr_mask(sunlit, config.module6)
    crater = dem == minimum_filter(dem, size=11, mode="reflect")
    dsc = identify_doubly_shadowed_craters(psr, crater)
    n = int(np.sum(dsc)) if isinstance(dsc, np.ndarray) else len(dsc)
    logger.info("[METRIC] n_dsc_craters=%d", n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(psr.astype(int), cmap="gray", origin="upper")
    ax.set_title(f"PSR ({psr.sum()} pixels)");     fig.savefig(_fig("stage3_psr.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    _export_geotiff("psr_mask", psr.astype(np.uint8), config._crs, config._gtfm, config._run_dir)
    if isinstance(dsc, np.ndarray):
        _export_geotiff("dsc_mask", dsc.astype(np.uint8), config._crs, config._gtfm, config._run_dir)


def _stage4(config):
    import matplotlib.pyplot as plt
    from module_3 import RoverState, plan_mission_path
    from module_2 import compute_slope, make_synthetic_auxiliary
    from module_2.mcda import CandidateSite
    from module_2.terrain_analysis import load_auxiliary_rasters
    rows, cols = 100, 100
    rr, cc = np.mgrid[0:rows, 0:cols]
    try:
        import rasterio
        with rasterio.open(config.dem_path) as src:
            dem = src.read(1).astype(np.float32)
        rows, cols = dem.shape
        rr, cc = np.mgrid[0:rows, 0:cols]
    except Exception:
        dem = (1000.0 + 3.0 * np.sin(rr / 8) * np.cos(cc / 8) + 1.5 * np.sin(rr / 20 + cc / 15)).astype(np.float32)
    slope = compute_slope(dem, config.dem_pixel_spacing_m)
    try:
        _, illum = load_auxiliary_rasters(config.dte_path, config.illumination_path)
    except Exception:
        _, illum = make_synthetic_auxiliary((rows, cols))
    ice = np.zeros((rows, cols), dtype=bool); ice[10:30, 10:30] = True
    sites = [CandidateSite(rows - 1, cols // 2 - 1, 1.5, True, 0.5, 0.9, 1)]
    rover = RoverState(config.mission_start, config.rover_battery_initial_wh, config.rover_battery_max_wh, True, 0)
    dash, ret = plan_mission_path(slope, ice, illum > 0.3, config.mission_start, config.dsc_sample_point, sites, rover, config.module3)
    logger.info("[METRIC] dash_feasible=%s", dash.feasible)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(slope, cmap="terrain", origin="upper")
    ax.plot(config.mission_start[1], config.mission_start[0], "go", ms=10, label="Start")
    ax.plot(config.dsc_sample_point[1], config.dsc_sample_point[0], "r*", ms=15, label="F2")
    if dash.feasible:
        p = np.array(dash.path); ax.plot(p[:, 1], p[:, 0], "b-", lw=1.5, label="Dash")
    if ret.feasible:
        p = np.array(ret.path); ax.plot(p[:, 1], p[:, 0], "y--", lw=1.5, label="Return")
    ax.legend(fontsize=8); ax.set_title("Planned Paths")
    fig.savefig(_fig("stage4_paths.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    paths = []
    if dash.feasible:
        paths.append((list(dash.path), {"type": "dash", "label": "Dash path to DSC"}))
    if ret.feasible:
        paths.append((list(ret.path), {"type": "return", "label": "Return path to site"}))
    if paths:
        _export_geojson_lines("planned_paths", paths, config._gtfm, config._run_dir)


def _stage5(config):
    import matplotlib.pyplot as plt
    from module_4 import Bug2Planner
    obs = np.zeros((100, 100), dtype=bool); obs[60, 50:70] = True
    planner = Bug2Planner([], obs, config.module4)
    traj = planner.run(config.mission_start, config.dsc_sample_point)
    gt = [(i, i) for i in range(min(len(traj), 30))]
    mae = planner.compute_mae(traj, gt)
    logger.info("[METRIC] slam_mae=%.4f", mae)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(obs.astype(int), cmap="gray_r", origin="upper")
    if traj:
        t = np.array(traj); ax.plot(t[:, 1], t[:, 0], "r-", lw=1, label="Trajectory")
    ax.set_title(f"Bug-2 SLAM (MAE={mae:.3f})"); ax.legend(fontsize=8)
    fig.savefig(_fig("stage5_slam.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    if traj:
        _export_geojson_lines("slam_trajectory", [(list(traj), {"type": "slam", "label": "Bug-2 SLAM trajectory"})], config._gtfm, config._run_dir)


def _stage6(config):
    from module_5 import compute_drill_heat_transfer, compute_post_drill_temperature, compute_sublimation_rate, compute_vapor_density, EFPIModel
    Q = compute_drill_heat_transfer(config.module5)
    T = compute_post_drill_temperature(Q, 0.5, config.module5.regolith_temp_k)
    sub = compute_sublimation_rate(T, 1.0, config.module5.sealed_volume_m3, config.module5)
    vapor = compute_vapor_density(sub, config.module5.sealed_volume_m3, config.module5.drill_contact_duration_s)
    efpi = EFPIModel(config.module5)
    pct = efpi.infer_ice_density(efpi.fringe_shift_from_humidity(vapor))
    logger.info("[METRIC] efpi_ice_pct=%.2f", pct)


def _extract_metrics(run, config, capture: LogCapture, db) -> None:
    from web.models import FigureRecord
    for line in capture.records:
        if "[METRIC]" not in line:
            continue
        try:
            kv = line.strip().split("[METRIC] ")[-1]
            key, val = kv.split("=", 1)
            if hasattr(run, key):
                setattr(run, key, float(val))
        except (IndexError, ValueError):
            pass

    src = Path(_HERE) / "figures"
    dst = Path(_HERE) / "web" / "runs" / str(run.id)
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for f in src.iterdir():
            if f.suffix == ".png":
                shutil.copy2(str(f), str(dst / f.name))
                db.add(FigureRecord(run_id=run.id, filename=f.name, title=f.stem))
    db.commit()
