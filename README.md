# Lunar South Pole Autonomous Exploration Pipeline

**ISRO Space Tech Hackathon — Team Thunderbolts**

A modular, production-grade pipeline integrating Chandrayaan-2 remote sensing processing with a risk-aware, non-myopic hybrid path planner and in-situ hydrodynamics modeling for lunar ice exploration.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the full end-to-end orchestration script:
```bash
python main.py
```
*Note: If real DFSAR and DEM data files are not found in the `data/` directory, the pipeline will gracefully fall back to generating and using synthetic fixtures for demonstration purposes.*

## Running Tests

The project is fully covered by unit tests using `pytest`. To run the test suite:
```bash
pytest
```

---

## Module Descriptions

### Module 1: Radar Polarimetric Decomposition (DFSAR)
Processes Chandrayaan-2 DFSAR compact polarimetry products to compute Circular Polarization Ratio (CPR) and Degree of Polarization (DOP), isolating potential subsurface ice deposits. It also inverts the complex refractive index using a CRIM mixing rule to estimate volumetric ice fraction.

### Module 2: Landing Site Seeding (Terrain MCDA)
Ingests DEM topography and illumination raster data. Computes local slope via Sobel filters and performs Multi-Criteria Decision Analysis (MCDA) to isolate optimal landing and communication singularity sites on crater rims (e.g., Faustini Crater).

### Module 3: Hybrid Risk-Aware Path Planner (Gorilla Traversal)
A state-aware A* hybrid planner. It enforces strict Battery State-of-Charge (SoC) hard constraints and dynamically penalizes slopes and path revisits. The rover plans a "dash" into a Doubly Shadowed Crater (DSC) and plans a viable return path to the nearest safe singularity.

### Module 4: Reactive Local Obstacle Avoidance & SLAM
Simulates a real-time reactive Bug-2 boundary-following algorithm for immediate obstacle avoidance. Features an integrated simplified EKF-SLAM pose-graph tracker and a Strategic Autonomy mode that throttles rover speed when Direct-to-Earth (DTE) communication is lost.

### Module 5: In-Situ Fiber-Optic Hydrodynamics Model (EFPI)
Models the thermal and optical physics of an Extrinsic Fabry-Perot Interferometer (EFPI) embedded in a drill. Computes drill heat transfer into the regolith, Clausius-Clapeyron ice sublimation into a sealed chamber, and the resultant optical interference fringe shift to infer water-ice concentration.

---

## Technical Appendices

### CPR Channel Convention (DFSAR)
Chandrayaan-2 DFSAR operates in compact polarimetry mode:
- **L-band:** Transmit Left-Hand Circular (LHC), receive H and V linear components (LH, LV).
- **S-band:** Transmit Right-Hand Circular (RHC), receive H and V linear components (RH, RV).

We map these to same-sense and opposite-sense circular basis power:
$$ \text{Same-Sense L-band } (|LL|^2) = \left| \frac{LH - iLV}{\sqrt{2}} \right|^2 $$
$$ \text{Opposite-Sense L-band } (|LR|^2) = \left| \frac{LH + iLV}{\sqrt{2}} \right|^2 $$

*Reference: Rao et al. 2022, Planetary and Space Science.*

### CRIM End-Member Values
The Complex Refractive Index Model (CRIM) requires reference dielectric constants for the end-members:
- **Water Ice:** $\epsilon = 3.15 + 0.001j$ *(Cumming 1952)*
- **Dry Lunar Regolith:** $\epsilon = 2.7 + 0.002j$ *(Olhoeft & Strangway 1975)*
- **Vacuum (Porosity):** $\epsilon = 1.0 + 0j$ (Assumed 40% volume).

*Limitation: These are bulk microwave permittivity values. Actual localized calibration depends heavily on physical regolith compaction state.*

### Known Limitations
1. CRIM inversion currently uses a simplified empirical mapping between radar backscatter cross-section ($\sigma^0$) and effective mixture permittivity.
2. The SLAM module tracks pose covariance analytically via EKF but currently lacks loop-closure graph optimization over the point clouds.
3. The rover battery model assumes a constant nominal power draw during motion, ignoring complex multi-body wheel-slip mechanics.
4. The Clausius-Clapeyron sublimation model assumes pure bulk water ice dynamics, rather than ice intimately bound in regolith micropores.
