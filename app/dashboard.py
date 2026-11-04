import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure root directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.run_aerofuse_v4_router import AeroFuseNetV4Router

st.set_page_config(
    page_title="AeroFuse-Net | Hyper-Local Weather AI",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CACHED DATA & MODEL LOADERS
# =====================================================================
@st.cache_data
def load_all_metrics_and_data():
    bench_df = pd.read_csv(os.path.join("results", "metrics", "benchmark_comparison.csv"))
    dm_df = pd.read_csv(os.path.join("results", "metrics", "statistical_significance_dm_test_v4.csv"))
    ref_df = pd.read_csv(os.path.join("results", "metrics", "persistence_and_climatology_baselines.csv"))
    regime_df = pd.read_csv(os.path.join("results", "metrics", "seasonal_and_extreme_regime_evaluation.csv"))
    prob_df = pd.read_csv(os.path.join("results", "metrics", "probabilistic_calibration_metrics.csv"))

    local_df = pd.read_parquet(os.path.join("data", "processed", "processed_local_station.parquet"))
    v4_preds = dict(np.load(os.path.join("results", "metrics", "AeroFuse_Net_v4_Router_test_predictions.npz")))
    itrans_preds = dict(np.load(os.path.join("results", "metrics", "iTransformer_2026_test_predictions.npz")))
    cnn1d_preds = dict(np.load(os.path.join("results", "metrics", "1D_CNN_ResNet_test_predictions.npz")))

    return bench_df, dm_df, ref_df, regime_df, prob_df, local_df, v4_preds, itrans_preds, cnn1d_preds


@st.cache_resource
def load_trained_v4_model(num_features: int = 19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroFuseNetV4Router(local_features=num_features, grid_channels=6, seq_len=24, horizon=24)
    ckpt_path = os.path.join("data", "processed", "AeroFuse_Net_v4_Router.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def main():
    bench_df, dm_df, ref_df, regime_df, prob_df, local_df, v4_npz, itrans_npz, cnn1d_npz = load_all_metrics_and_data()
    model, device = load_trained_v4_model(local_df.shape[1])

    # Sidebar Navigation & Controls
    st.sidebar.title("⛈️ AeroFuse-Net v4")
    st.sidebar.caption("Physics-Guided Dual-Modal Spatiotemporal Weather Forecasting System")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Target Region:** Patiala, Punjab (`30.34°N, 76.39°E`)")
    st.sidebar.markdown(f"**Synoptic Grid:** `5×5` ERA5 Mesh (`220km × 195km`)")
    st.sidebar.markdown(f"**Inference Device:** `{str(device).upper()}`")

    selected_window = st.sidebar.slider(
        "Select Test Set Forecast Window Index",
        min_value=0,
        max_value=len(v4_npz["trues"]) - 1,
        value=1250,
        step=1
    )
    horizon_view = st.sidebar.selectbox(
        "Rolling Trajectory Lead Time",
        options=[1, 3, 6, 12, 24],
        index=2,
        format_func=lambda x: f"t + {x} Hours Ahead"
    )

    # Header Banner
    st.title("⛈️ AeroFuse-Net: Dual-Modal Spatiotemporal Weather Forecasting")
    st.markdown(
        "**Fusing 2D-CNN Synoptic Atmospheric Grids (`5×5`) with Local Inverted Transformer & 1D-CNN-BiLSTM Station Dynamics**"
    )

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall 24h RMSE (#1)", "1.4378 °C", "-0.296°C vs Diurnal Cycle")
    c2.metric("Overall 24h MAE", "1.0366 °C", "-75.5% vs Persistence")
    c3.metric("1-Hour Lead MAE", "0.5033 °C", "R² = 0.9674 (#1)")
    c4.metric("Monsoon Regime RMSE (#1)", "1.0754 °C", "Beats iTransformer & 1D-CNN")
    c5.metric("DM Statistical Significance", "p < 0.0001", "Beats 7/8 Models at 99% CI")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Interactive Probabilistic Forecast",
        "🏆 9-Model Benchmark & Ablation Study",
        "🌍 2D-CNN Spatial Saliency & Weather Regimes",
        "🔬 Diebold-Mariano & Architecture Specs"
    ])

    # =====================================================================
    # TAB 1: Interactive 24-Hour Fan Chart & Multi-Day Rolling Trajectory
    # =====================================================================
    with tab1:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader(f"24-Hour Probabilistic Fan Forecast (Window #{selected_window})")
            hours_ahead = np.arange(1, 25)
            y_true_24 = v4_npz["trues"][selected_window]
            y_aero_24 = v4_npz["preds"][selected_window]
            q10_24 = v4_npz["q10"][selected_window]
            q90_24 = v4_npz["q90"][selected_window]
            y_itrans_24 = itrans_npz["preds"][selected_window]
            y_cnn_24 = cnn1d_npz["preds"][selected_window]

            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([hours_ahead, hours_ahead[::-1]]),
                y=np.concatenate([q90_24, q10_24[::-1]]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                name="10th–90th Percentile Uncertainty Band"
            ))
            fig_fan.add_trace(go.Scatter(
                x=hours_ahead, y=y_true_24, mode="lines+markers",
                name="Ground Truth (°C)", line=dict(color="#2c3e50", width=3)
            ))
            fig_fan.add_trace(go.Scatter(
                x=hours_ahead, y=y_aero_24, mode="lines+markers",
                name="AeroFuse-Net v4 Router", line=dict(color="#c0392b", width=3, dash="dash")
            ))
            fig_fan.add_trace(go.Scatter(
                x=hours_ahead, y=y_itrans_24, mode="lines",
                name="iTransformer (2026)", line=dict(color="#27ae60", width=2, dash="dot")
            ))
            fig_fan.add_trace(go.Scatter(
                x=hours_ahead, y=y_cnn_24, mode="lines",
                name="1D-CNN ResNet", line=dict(color="#2980b9", width=2, dash="dot")
            ))
            fig_fan.update_layout(
                xaxis_title="Forecast Horizon (Hours Ahead)",
                yaxis_title="2m Air Temperature (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                height=440
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        with col_right:
            st.subheader(f"7-Day (168h) Rolling Forecast Trajectory (t+{horizon_view}h Lead)")
            h_idx = horizon_view - 1
            end_w = min(selected_window + 168, len(v4_npz["trues"]))
            t_range = np.arange(end_w - selected_window)

            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=np.concatenate([t_range, t_range[::-1]]),
                y=np.concatenate([
                    v4_npz["q90"][selected_window:end_w, h_idx],
                    v4_npz["q10"][selected_window:end_w, h_idx][::-1]
                ]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Uncertainty Interval (q10–q90)"
            ))
            fig_roll.add_trace(go.Scatter(
                x=t_range, y=v4_npz["trues"][selected_window:end_w, h_idx],
                mode="lines", name="Actual Station Temp (°C)", line=dict(color="#2c3e50", width=2.5)
            ))
            fig_roll.add_trace(go.Scatter(
                x=t_range, y=v4_npz["preds"][selected_window:end_w, h_idx],
                mode="lines", name=f"AeroFuse-Net v4 (t+{horizon_view}h)", line=dict(color="#c0392b", width=2, dash="dash")
            ))
            fig_roll.update_layout(
                xaxis_title="Hours from Selected Origin",
                yaxis_title="2m Temperature (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                height=440
            )
            st.plotly_chart(fig_roll, use_container_width=True)

    # =====================================================================
    # TAB 2: 9-Model Benchmark & Ablation Study
    # =====================================================================
    with tab2:
        st.subheader("Master Multi-Horizon Benchmark & Architectural Ablation Table")
        st.dataframe(bench_df.style.highlight_min(subset=["MAE_1h", "MAE_6h", "MAE_12h", "MAE_24h", "Overall_MAE", "Overall_RMSE"], color="#d4edda"), use_container_width=True)

        c_left, c_right = st.columns(2)
        with c_left:
            f1_img = os.path.join("results", "figures", "v4_router", "fig1_horizon_degradation_v4.png")
            if os.path.exists(f1_img):
                st.image(f1_img, caption="Multi-Horizon Error Degradation (t+1h to t+24h)", use_container_width=True)
        with c_right:
            f4_img = os.path.join("results", "figures", "v4_router", "fig4_ablation_progression_v4.png")
            if os.path.exists(f4_img):
                st.image(f4_img, caption="Ablation Study Progression (v1 -> v2 -> v3 -> v4 Router)", use_container_width=True)

        st.subheader("Comparison Against Meteorological Reference Standards (Persistence & Diurnal Climatology)")
        st.dataframe(ref_df, use_container_width=True)

    # =====================================================================
    # TAB 3: 2D-CNN Spatial Saliency & Patiala Seasonal Regimes
    # =====================================================================
    with tab3:
        col_s1, col_s2 = st.columns([1, 1.3])
        with col_s1:
            st.subheader("2D-CNN Spatial Attention Saliency Map")
            f3_img = os.path.join("results", "figures", "v4_router", "fig3_spatial_cnn_gradcam_v4.png")
            if os.path.exists(f3_img):
                st.image(f3_img, use_container_width=True)
            st.info(
                "**Physical Insight:** Peak spatial gradient sensitivity (`1.00` at `30.84°N, 77.39°E`) "
                "aligns with the **Shivalik Foothills / Himachal boundary** North-East of Patiala, capturing "
                "katabatic mountain-valley wind outflows and convective systems propagating into the Indo-Gangetic Plains."
            )

        with col_s2:
            st.subheader("Seasonal, Diurnal & Extreme Weather Regime Evaluation")
            st.dataframe(regime_df, use_container_width=True)
            f5_img = os.path.join("results", "figures", "reviewer_suite", "fig5_seasonal_regime_breakdown.png")
            if os.path.exists(f5_img):
                st.image(f5_img, use_container_width=True)

    # =====================================================================
    # TAB 4: Statistical Significance (Diebold-Mariano) & Uncertainty
    # =====================================================================
    with tab4:
        st.subheader("Diebold-Mariano (DM) Statistical Significance Test Results")
        st.dataframe(dm_df, use_container_width=True)

        st.subheader("Probabilistic Uncertainty & Conformal Prediction Evaluation")
        st.dataframe(prob_df, use_container_width=True)

        st.subheader("Mathematical Formulation of AeroFuse-Net v4 Router")
        st.latex(r"\hat{\mathbf{Y}}_{t+1:t+24} = \boldsymbol{\alpha}_h \odot \sum_{m=1}^{4} \text{softmax}\Big(\mathbf{P}_{m,h} + \gamma \tanh\big(\mathcal{R}_\phi(\text{Conv2D}(\mathcal{X}_{\text{grid}}, \Delta\mathcal{X}_{\text{adv}}), \mathbf{x}_{\text{local}, t})\big)\Big) \cdot \mathcal{F}_m(\mathcal{X}) + \boldsymbol{\beta}_h")


if __name__ == "__main__":
    main()