# User Guide: Chandrayaan-2 Optical Image Correspondence Pipeline (SIH 26166)

Welcome to the **Chandrayaan-2 Optical Image Correspondence & Registration Pipeline**. This guide explains how to use the web dashboard, CLI tools, configuration parameters, and evaluation suite.

---

## 1. Quick Start

### Starting the Web Dashboard (Recommended)
Open a terminal in the project directory and run:
```powershell
python -m streamlit run lunar_correspondence/ui/app.py
```
Open your browser at: **`http://localhost:8501`**

---

## 2. Using the Interactive Web Dashboard

The web interface is organized into 4 modes available in the left sidebar:

### Mode 1: Interactive Registration Pipeline
1. **Choose Input Data:**
   - **Curated Benchmark Pairs:** Select from pre-generated difficulty buckets:
     - **Easy:** Sun-angle difference $< 15^\circ$, GSD ratio $\approx 1\times$
     - **Medium:** Sun-angle difference $15\text{--}45^\circ$, GSD ratio $\approx 1.2\times$
     - **Hard:** Sun-angle difference $> 45^\circ$, high crater relief
     - **Extreme:** Near-terminator illumination (sun elevation $< 5^\circ$), large shadows
   - **Custom Upload:** Upload custom PDS4 `.img` / GeoTIFF / PNG files along with optional XML labels.
2. **Inspect Metadata:**
   - Review parsed sensor tags (`OHRC`, `TMC-2`, `IIRS`, `LRO_NAC`), GSD ($m/\text{px}$), and sun azimuth/elevation angles.
3. **Execute Pipeline:**
   - Click **"🚀 Run Multi-Stage Registration Pipeline"**.
4. **Inspect the Explainability Trace:**
   - The trace shows each stage executed and explains which matcher was chosen:
     - **Tier 1 (LightGlue):** Fast, sparse learned matching.
     - **Fallback Tier 1 (LoFTR):** Dense detector-free matching if candidates $< 20$ or coverage $< 0.4$.
     - **Fallback Tier 2 (RoMa):** Heavy dense matching for extreme illumination changes ($>45^\circ$).
     - **FAILED:** Graceful failure banner if geometric consensus cannot be found (never outputs a bogus transform).
5. **View Confidence & Diagnostics:**
   - **Confidence Badge:** `HIGH`, `MEDIUM`, `LOW`, or `FAILED`.
   - **Metrics Panel:** Number of candidates, verified inliers, inlier ratio, reprojection RMSE, grid coverage, and sub-pixel accuracy percentages.
   - **Tabs:**
     - **Feature Correspondences:** Matching tie points colored by confidence.
     - **MAGSAC++ Verification:** Inliers in green, rejected outliers in red.
     - **Checkerboard Alignment Blend:** Visual check that crater boundaries align seamlessly across alternating tiles.
     - **Residual Error Heatmap:** Pixel-wise difference map highlighting alignment accuracy.

---

### Mode 2: 8-Stage Ablation Matrix (Blueprint Section 20)
1. Select a difficulty bucket (*Easy*, *Medium*, *Hard*).
2. Click **"🚀 Run 8-Stage Ablation Matrix"**.
3. View the generated 8-row table comparing:
   - `A.` SIFT + RANSAC Baseline
   - `B.` A + Illumination Preprocessing (CLAHE + Gradient)
   - `C.` B + GSD-aware Pyramid
   - `D.` C + Learned Matcher (LightGlue)
   - `E.` D + MAGSAC++
   - `F.` E + Spatial Uniform Grid
   - `G.` F + ECC Sub-Pixel Refinement
   - `H.` G + Adaptive Controller (LoFTR/RoMa)
4. Compare metric improvements on the interactive bar chart.

---

### Mode 3: Dataset & Sensor Specs
Provides a reference table of Chandrayaan-2 and reference sensors:
- **OHRC:** $0.25\text{--}0.32\text{ m/px}$, PDS4 `.img + .xml`
- **TMC-2:** $5.0\text{ m/px}$, $20\text{ km}$ swath, stereo DEMs
- **IIRS:** $80.0\text{ m/px}$, 256-band hyperspectral `.qub`
- **LRO NAC:** $0.5\text{--}2.0\text{ m/px}$
- **LRO WAC:** $100\text{ m/px}$ (intermediate anchor for IIRS multi-hop)

---

### Mode 4: Experiment DB Audit
Inspects runs recorded in the SQLite database (`experiments/experiments.db`). Every run logs timestamp, git commit, matcher tier, hardware, metrics, and individual point correspondences conforming to Section 22 of the blueprint.

---

## 3. Command Line Interface (CLI)

Use `run_pipeline.py` for headless runs, batch evaluations, and tests:

### 1. Run Single Registration Demo
```powershell
python run_pipeline.py --mode demo --difficulty medium
```
Outputs:
- Explainability trace
- Confidence badge and metric summary
- Saves registered image and diagnostic PNGs to `./outputs/`

### 2. Run 8-Stage Ablation Matrix
```powershell
python run_pipeline.py --mode ablation --difficulty easy
```

### 3. Run Automated Regression Tests
```powershell
python run_pipeline.py --mode test
```
Executes all 5 unit tests validating synthetic ground truth, homographies, sub-pixel displacements, and pipeline stages.

---

## 4. Configuration Tuning (`default_config.yaml`)

Custom thresholds are controlled in `lunar_correspondence/configs/default_config.yaml`:

```yaml
matching:
  controller:
    min_candidate_matches: 20             # Below this triggers LoFTR fallback
    min_inlier_ratio_lightglue: 0.15      # Inlier ratio trigger
    min_grid_coverage: 0.40               # Spatial coverage trigger
    escalate_loftr_inlier_ratio: 0.10     # Below this triggers RoMa
    escalate_roma_illum_diff_deg: 45.0    # Sun-angle threshold for RoMa

geometry:
  magsac:
    threshold_coarse_px: 8.0
    threshold_fine_px: 1.5
    confidence: 0.999
  spatial_selection:
    grid_rows: 8
    grid_cols: 8
    max_per_cell: 5                       # Limits cluster density

subpixel:
  method: "ecc"                           # ECC affine with peak-fit fallback
  patch_size: 48
  ecc_max_iterations: 50
```

---

## 5. Output Data Contract

Every registration run produces a structured dictionary complying with Blueprint Section 4:
```json
{
  "registered_image_path": "./outputs/registered_warped_source.png",
  "transform_model": "homography",
  "transform_params": [[...], [...], [...]],
  "correspondences": [
    {
      "x": 142.5,
      "y": 204.8,
      "x_ref": 140.2,
      "y_ref": 201.3,
      "subpixel_dx": -0.23,
      "subpixel_dy": -0.15,
      "confidence": 0.94,
      "residual_error_px": 0.27,
      "matcher_source": "LightGlue"
    }
  ],
  "metrics": {
    "inlier_ratio": 0.42,
    "rmse_px": 0.68,
    "grid_coverage": 0.65,
    "registration_success": true
  },
  "confidence_level": "HIGH",
  "confidence_reasons": ["Inlier ratio >= 0.35", "Grid coverage >= 0.50", "RMSE <= 2.0 px"]
}
```

