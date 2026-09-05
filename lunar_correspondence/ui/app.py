"""Streamlit Interactive Application for Lunar Optical Image Correspondence.
Implements the full wireframe from Section 25 of the Blueprint:
1. Data Input & Curated Pair Selector
2. Metadata & Solar Geometry Panel
3. Stage-by-stage Explainability Pipeline Trace
4. Interactive Matching & MAGSAC++ Verification View
5. Registration Alignment & Difference Heatmap
6. Quantitative Metrics Dashboard & Three-Tier Confidence Badge
7. 8-Stage Ablation Matrix Runner
"""

import streamlit as st
import numpy as np
import cv2
import time
import os
import json

from lunar_correspondence.pipeline import LunarRegistrationPipeline
from lunar_correspondence.data_loader.synthetic_generator import LunarSyntheticGenerator
from lunar_correspondence.data_loader.download_helpers import LunarDataDownloader
from lunar_correspondence.evaluation.ablation import AblationStudyEngine
from lunar_correspondence.evaluation.report_gen import ReportGenerator
from lunar_correspondence.experiments.db import ExperimentDatabase

st.set_page_config(
    page_title="ISRO Chandrayaan-2 Image Correspondence Pipeline",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
    }
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-failed {
        background-color: #991B1B;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return LunarRegistrationPipeline()

@st.cache_resource
def load_generator():
    return LunarSyntheticGenerator(random_seed=42)

@st.cache_resource
def load_ablation_engine():
    return AblationStudyEngine()

@st.cache_resource
def load_db():
    return ExperimentDatabase()

pipeline = load_pipeline()
generator = load_generator()
ablation_engine = load_ablation_engine()
db = load_db()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/full-moon.png", width=64)
st.sidebar.title("SIH 26166 Navigation")
app_mode = st.sidebar.radio(
    "Select Workflow Mode:",
    ["1. Interactive Registration Pipeline", "2. 8-Stage Ablation Matrix", "3. Dataset & Sensor Specs", "4. Experiment DB Audit"]
)

# -----------------------------------------------------------------------------
# MODE 1: Interactive Registration Pipeline
# -----------------------------------------------------------------------------
if app_mode == "1. Interactive Registration Pipeline":
    st.markdown('<div class="main-title">🌕 Chandrayaan-2 Optical Image Correspondence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-modal, Sun-Angle, and Scale-Invariant Lunar Terrain Registration (SIH 26166)</div>', unsafe_allow_html=True)

    # Input Section
    col_input, col_meta = st.columns([1.2, 1.0])

    with col_input:
        st.subheader("1. Input Data Selection")
        data_source = st.selectbox(
            "Choose Input Data Source:",
            [
                "Curated Benchmark Pair: Easy (Sun diff < 15°, GSD ~1x)",
                "Curated Benchmark Pair: Medium (Sun diff 15-45°, GSD ~1.2x)",
                "Curated Benchmark Pair: Hard (Sun diff > 45°, heavy crater relief)",
                "Curated Benchmark Pair: Extreme (Near terminator, Sun diff > 75°)",
                "Custom Upload (PDS4 / GeoTIFF / PNG)"
            ]
        )

        test_pair = None
        if "Easy" in data_source:
            test_pair = generator.generate_pair(difficulty="easy", source_sensor="OHRC", ref_sensor="LRO_NAC")
        elif "Medium" in data_source:
            test_pair = generator.generate_pair(difficulty="medium", source_sensor="OHRC", ref_sensor="LRO_NAC")
        elif "Hard" in data_source:
            test_pair = generator.generate_pair(difficulty="hard", source_sensor="OHRC", ref_sensor="LRO_NAC")
        elif "Extreme" in data_source:
            test_pair = generator.generate_pair(difficulty="extreme", source_sensor="OHRC", ref_sensor="LRO_NAC")
        else:
            file_src = st.file_uploader("Upload Source Image (OHRC / TMC-2 / IIRS)", type=["png", "jpg", "tif", "img"])
            file_ref = st.file_uploader("Upload Reference Image (LRO NAC / WAC)", type=["png", "jpg", "tif", "img"])
            if file_src and file_ref:
                bytes_s = np.asarray(bytearray(file_src.read()), dtype=np.uint8)
                bytes_r = np.asarray(bytearray(file_ref.read()), dtype=np.uint8)
                img_s = cv2.imdecode(bytes_s, cv2.IMREAD_GRAYSCALE)
                img_r = cv2.imdecode(bytes_r, cv2.IMREAD_GRAYSCALE)
                test_pair = {
                    "source_image": img_s,
                    "ref_image": img_r,
                    "metadata_src": {"sensor": "OHRC", "sun_azimuth_deg": 45.0, "sun_elevation_deg": 35.0, "gsd_m_per_px": 0.25},
                    "metadata_ref": {"sensor": "LRO_NAC", "sun_azimuth_deg": 85.0, "sun_elevation_deg": 50.0, "gsd_m_per_px": 0.5},
                    "difficulty": "custom"
                }

    with col_meta:
        st.subheader("2. Mission Metadata & Geometry")
        if test_pair:
            m_s = test_pair["metadata_src"]
            m_r = test_pair["metadata_ref"]
            st.markdown(f"""
            - **Source Sensor:** `{m_s['sensor']}` (GSD: `{m_s['gsd_m_per_px']}` m/px)
            - **Source Sun Geometry:** Azimuth = `{m_s['sun_azimuth_deg']:.1f}°`, Elevation = `{m_s['sun_elevation_deg']:.1f}°`
            - **Reference Sensor:** `{m_r['sensor']}` (GSD: `{m_r['gsd_m_per_px']}` m/px)
            - **Reference Sun Geometry:** Azimuth = `{m_r['sun_azimuth_deg']:.1f}°`, Elevation = `{m_r['sun_elevation_deg']:.1f}°`
            - **Solar Angular Difference:** `|Δθ| = {abs(m_s['sun_azimuth_deg'] - m_r['sun_azimuth_deg']):.1f}°`
            """)

    if test_pair:
        c1, c2 = st.columns(2)
        with c1:
            st.image(test_pair["source_image"], caption="Source Frame (Chandrayaan-2)", use_container_width=True)
        with c2:
            st.image(test_pair["ref_image"], caption="Reference Frame (LRO NAC)", use_container_width=True)

        if st.button("🚀 Run Multi-Stage Registration Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing pipeline stages [0] to [16]..."):
                result = pipeline.process_pair(
                    src_raster=test_pair["source_image"],
                    ref_raster=test_pair["ref_image"],
                    meta_src=test_pair["metadata_src"],
                    meta_ref=test_pair["metadata_ref"],
                    H_ground_truth=test_pair.get("H_gt"),
                    gt_tie_points=test_pair.get("gt_tie_points")
                )

            # Stage-by-stage explainability trace
            st.subheader("3. Pipeline Progress & Controller Explainability Trace")
            with st.expander("🔍 Click to view Stage-by-Stage Trace & Decisions", expanded=True):
                for step in result["pipeline_trace"]:
                    if "Escalation triggered" in step or "Executing Fallback" in step:
                        st.warning(f"⚡ {step}")
                    elif "Success with" in step or "Passed" in step:
                        st.success(f"✅ {step}")
                    elif "FAILED" in step:
                        st.error(f"❌ {step}")
                    else:
                        st.info(f"ℹ️ {step}")

            # Confidence Banner
            conf = result["confidence_level"]
            st.subheader("4. Registration Confidence & Audit Status")
            if conf == "HIGH":
                st.markdown('<span class="badge-high">HIGH CONFIDENCE (Sub-pixel & Spatially Distributed)</span>', unsafe_allow_html=True)
            elif conf == "MEDIUM":
                st.markdown('<span class="badge-med">MEDIUM CONFIDENCE (Reliable registration)</span>', unsafe_allow_html=True)
            elif conf == "LOW":
                st.markdown('<span class="badge-low">LOW CONFIDENCE (Transform estimated, flagged unreliable)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-failed">REGISTRATION FAILED (Ambiguity / Low Overlap)</span>', unsafe_allow_html=True)

            for r in result["confidence_reasons"]:
                st.caption(f"• Reason: {r}")

            # Metrics Dashboard
            st.subheader("5. Quantitative Metrics Dashboard (Section 20)")
            m = result["metrics"]
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Candidates / Inliers", f"{m.get('n_candidates', 0)} / {m.get('n_inliers', 0)}")
            col_m2.metric("Inlier Ratio", f"{m.get('inlier_ratio', 0.0):.1%}")
            col_m3.metric("Reprojection RMSE", f"{m.get('rmse_px', 'N/A')} px")
            col_m4.metric("Grid Coverage", f"{m.get('grid_coverage', 0.0):.1%}")

            if m.get("gt_rmse_px") is not None:
                col_g1, col_g2, col_g3 = st.columns(3)
                col_g1.metric("Ground Truth Sub-Pixel RMSE", f"{m.get('gt_rmse_px'):.2f} px")
                col_g2.metric("Sub-Pixel Error < 0.5 px", f"{m.get('subpixel_pct_under_05px'):.1f}%")
                col_g3.metric("Sub-Pixel Error < 1.0 px", f"{m.get('subpixel_pct_under_10px'):.1f}%")

            # Visualizations
            st.subheader("6. Visual Diagnostics & Registration Alignment")
            tab_match, tab_verify, tab_blend, tab_heatmap = st.tabs([
                "Feature Correspondences",
                "MAGSAC++ Inliers/Outliers",
                "Checkerboard Alignment Blend",
                "Residual Error Heatmap"
            ])

            with tab_match:
                st.image(result["diagnostic_visualizations"][0], caption="Candidate Correspondences (Color-coded by confidence)", use_container_width=True)

            with tab_verify:
                st.image(result["diagnostic_visualizations"][1], caption="MAGSAC++ Geometric Verification (Green: Verified Inliers | Red: Rejected Outliers)", use_container_width=True)

            with tab_blend:
                st.image(result["diagnostic_visualizations"][2], caption="Checkerboard Overlay (Aligned crater edges seamlessly continue across tiles)", use_container_width=True)

            with tab_heatmap:
                st.image(result["diagnostic_visualizations"][3], caption="Pixel-wise Difference Heatmap (Low residual across aligned terrain)", use_container_width=True)

            # Record to database
            exp_id = f"EXP_{int(time.time())}"
            db.record_experiment_run(
                experiment_id=exp_id,
                pair_id=test_pair.get("difficulty", "custom"),
                matcher_tier_used=result["transform_model"] or "None",
                metrics=result["metrics"],
                correspondences=result["correspondences"],
                confidence_level=conf
            )
            st.success(f"Audit log saved to database with Experiment ID: `{exp_id}`")

# -----------------------------------------------------------------------------
# MODE 2: 8-Stage Ablation Matrix Runner
# -----------------------------------------------------------------------------
elif app_mode == "2. 8-Stage Ablation Matrix":
    st.markdown('<div class="main-title">🔬 8-Stage Ablation Matrix (Section 20)</div>', unsafe_allow_html=True)
    st.markdown("""
    This ablation study methodically runs all 8 configurations on identical difficulty-bucketed lunar imagery 
    to provide unambiguous scientific proof of which component improves which metric:
    - **A:** Classical baseline (SIFT + RANSAC)
    - **B:** A + Illumination preprocessing (CLAHE + Gradient orientation)
    - **C:** B + GSD-aware physical scale pyramid
    - **D:** C + Learned Matcher (LightGlue / SuperPoint)
    - **E:** D + MAGSAC++ (replacing RANSAC)
    - **F:** E + Spatial uniform grid selection
    - **G:** F + ECC sub-pixel refinement
    - **H:** G + Adaptive Matcher Controller (LoFTR / RoMa escalation)
    """)

    abl_diff = st.selectbox("Select Difficulty Bucket for Ablation:", ["Medium (Sun diff 15-45°)", "Hard (Sun diff > 45°)", "Easy (Sun diff < 15°)"])
    bucket_key = "medium" if "Medium" in abl_diff else ("hard" if "Hard" in abl_diff else "easy")

    if st.button("🚀 Run 8-Stage Ablation Matrix", type="primary"):
        with st.spinner("Generating controlled test pair and evaluating configurations A through H..."):
            pair = generator.generate_pair(difficulty=bucket_key)
            results = ablation_engine.run_full_ablation(pair)

        st.subheader("Ablation Results Table")
        md_table = ReportGenerator.generate_ablation_markdown_table(results)
        st.markdown(md_table)

        # Bar chart comparison
        import pandas as pd
        df = pd.DataFrame(results)
        st.subheader("Metric Progression: Inlier Ratio & Grid Coverage")
        st.bar_chart(df.set_index("configuration")[["inlier_ratio", "grid_coverage"]])

# -----------------------------------------------------------------------------
# MODE 3: Dataset & Sensor Specs
# -----------------------------------------------------------------------------
elif app_mode == "3. Dataset & Sensor Specs":
    st.markdown('<div class="main-title">🛰️ Chandrayaan-2 Sensor Specifications & Portals</div>', unsafe_allow_html=True)
    downloader = LunarDataDownloader()
    sensors = downloader.get_supported_sensors()
    st.table(sensors)

# -----------------------------------------------------------------------------
# MODE 4: Experiment DB Audit
# -----------------------------------------------------------------------------
elif app_mode == "4. Experiment DB Audit":
    st.markdown('<div class="main-title">🗄️ Experiment Audit & Reproducibility Database</div>', unsafe_allow_html=True)
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT experiment_id, timestamp, matcher_tier_used, hardware FROM Experiment ORDER BY timestamp DESC LIMIT 20")
    exp_rows = cursor.fetchall()

    if exp_rows:
        st.table(exp_rows)
    else:
        st.info("No experiments recorded yet. Run a registration in Mode 1 to log auditable results.")

