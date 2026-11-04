"""
AeroFuse-Net v5.6 — Canonical Stage-1 Frozen Streamlit Dashboard
File: app/dashboard_v5_6.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AeroFuse-Net v5.6 | Stage-1 Frozen Audit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parents[1]
V56_DIR = ROOT_DIR / "results" / "v5_6_final_manifest"
FIG_PATH = ROOT_DIR / "results" / "figures" / "v5_5_canonical" / "CANONICAL_V5_5_ALL_FIGURES.png"

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = V56_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

st.sidebar.title("AeroFuse-Net (Stage-1 Frozen)")
st.sidebar.markdown(
    """
    **Release ID:** `AeroFuse-Net-v5.6-Stage1-Frozen`  
    **Study Site:** Patiala, Punjab (`30.34°N, 76.39°E`)  
    **Reference Data:** ERA5 Hourly (`2019–2025`, `61,368` hrs)  
    **Isolated Windows:** `61,253` (`42,909` Train / `9,172` Val / `9,172` Test)  
    **Backbone Scope:** Frozen Pretrained Backbones (`Seed 42`) + 3-Seed Router Training (`42, 100, 2026`)
    """
)

st.title("AeroFuse-Net: Prior-Anchored Sparse Multimodal Routing for Multihorizon Temperature Forecasting")
st.caption("Reanalysis-Based Stage-1 Benchmark (`2019–2025` ERA5 Reference Series at Patiala, India)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("24h Test RMSE (3-Seed)", "1.4033 ± 0.0002 °C", "-0.0444 °C vs iTransformer")
c2.metric("24h Test MAE (3-Seed)", "1.0002 ± 0.0002 °C", "-0.0215 °C vs iTransformer")
c3.metric("Physical Grad RMSE", "1.2172 °C/h", "-11.26% vs Local Only")
c4.metric("Rolling 30d CQR Coverage", "80.08% PICP", "Nominal 80.0% (MPIW 3.077°C)")
c5.metric("Calm-Sky Gate Sparsity", "76.3%", "Prior-Anchored Dead-Zone")

tabs = st.tabs([
    "1. Timestamp & Split Audit",
    "2. 3-Seed Router Ablation",
    "3. HLN-DM & Holm Statistics",
    "4. 24-Horizon CQR Calibration",
    "5. Spatial Attribution & Figures",
    "6. Raw Window Timestamp Manifest"
])

with tabs[0]:
    st.subheader("Table 1: Exact Timestamp-Level Partition Reconciliation (`0` Discrepancy Hours)")
    st.markdown(
        "Every partition's `Raw_Start_Timestamp` and `Raw_End_Timestamp` matches the exact hourly slice count "
        "(`42,956 + 9,206 + 9,206 = 61,368` hours) and target-isolated window count (`42,909 + 9,172 + 9,172 = 61,253`)."
    )
    st.dataframe(load_csv("table1_v5_6_exact_timestamp_reconciliation.csv"), use_container_width=True)

with tabs[1]:
    st.subheader("Table 3: Individual 3-Seed Router Training Results (`Seeds 42, 100, 2026`)")
    st.markdown(
        "Following the negative control of Uniform 50/50 Fusion (`1.4886°C` RMSE), progressive additions from "
        "Variant 3 through Variant 7 (`1.4033°C` RMSE) yield strictly monotonic error reductions. Backbones are "
        "frozen after deterministic `Seed 42` pretraining; seeds `42, 100, 2026` train the router/gating parameters."
    )
    st.dataframe(load_csv("table3_v5_6_three_seed_individual_router_runs.csv"), use_container_width=True)

with tabs[2]:
    st.subheader("Table 2: Harvey-Leybourne-Newbold Diebold-Mariano & Holm-Bonferroni ($M=20$ Family)")
    st.markdown(
        "Evaluated on per-window squared-error (`MSE` in $\\text{°C}^2$) and absolute-error (`MAE` in $\\text{°C}$) "
        "loss differentials $d_t = L_{\\text{base},t} - L_{\\text{cand},t}$ with HAC bandwidth $\\text{lag} = h-1$ "
        "and sensitivity verification at Newey-West plug-in $\\text{lag} = 11$."
    )
    st.dataframe(load_csv("table2_v5_6_dm_hln_holm_with_bandwidth_sensitivity.csv"), use_container_width=True)

with tabs[3]:
    st.subheader("Table 4: Complete 24-Horizon Rolling Conformalized Quantile Regression (`t+1h` to `t+24h`)")
    st.markdown(
        "Empirical adaptive calibration using a 720-hour (30-day) rolling calibration window with a strict 24-hour "
        "chronological embargo across $N_{\\text{test}} = 9,172$ out-of-sample test windows."
    )
    st.dataframe(load_csv("table4_v5_6_full_24horizon_cqr_with_methodology.csv"), use_container_width=True)

with tabs[4]:
    st.subheader("Table 5: Spatial Grad-CAM Sensitivity vs. Block-Bootstrapped Gaussian Occlusion")
    st.dataframe(load_csv("table5_v5_6_spatial_block_bootstrap_attribution.csv"), use_container_width=True)
    st.divider()
    st.subheader("Composite 6-Panel Benchmark & Diagnostic Figure (`v5.5/v5.6 Canonical`)")
    if FIG_PATH.exists():
        st.image(str(FIG_PATH), use_container_width=True)
    else:
        st.warning(f"Figure not found at {FIG_PATH}. Run src/evaluation/run_v5_5_canonical_lock.py first.")

with tabs[5]:
    st.subheader("First 50 & Last 50 Window Timestamps per Partition (`300` Audit Records)")
    df_man = load_csv("partition_window_manifest_first_last_50_per_split.csv")
    part_filter = st.selectbox("Filter by Partition:", ["All", "Train", "Validation", "Test"])
    if part_filter != "All" and not df_man.empty:
        df_man = df_man[df_man["partition"] == part_filter]
    st.dataframe(df_man, use_container_width=True)