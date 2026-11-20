"""
AeroFuse-Weather-AI — Live 7-Day Patiala Forecast & CNN Feature Extraction Dashboard
File: app/dashboard_final_patiala.py
"""

from pathlib import Path
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="AeroFuse-Net | Live 7-Day Patiala Forecast & CNN Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
V6_DIR = ROOT / "results" / "v6_real_validation"
V6_FIG = ROOT / "results" / "figures" / "v6_real_validation" / "fig_v6_real_cnn_image_feature_extraction.png"

USER_AGENT = "AeroFuse-Weather-AI/1.0 (https://github.com/prakhart056-student/AeroFuse-Weather-AI)"

# -------------------------------------------------------------------------
# 1. Resilient Live Weather API Fetchers (Open-Meteo + MET Norway Fallback)
# -------------------------------------------------------------------------
def _http_get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_live_patiala_7day():
    """
    Fetches live 7-day (168h) hourly Patiala forecast.
    Primary: Open-Meteo (1 point). Automatic Backup: MET Norway Locationforecast 2.0 (ECMWF).
    """
    # Primary: Open-Meteo (single coordinate)
    try:
        base_url = "https://api.open-meteo.com/v1/forecast"
        params_pt = urllib.parse.urlencode({
            "latitude": 30.34,
            "longitude": 76.39,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": 7,
            "timezone": "Asia/Kolkata"
        })
        pt_json = _http_get_json(f"{base_url}?{params_pt}", timeout=12)
        df_h = pd.DataFrame(pt_json["hourly"])
        df_h["time"] = pd.to_datetime(df_h["time"])
        df_daily = pd.DataFrame(pt_json["daily"]).iloc[:7].reset_index(drop=True)
        return df_h.iloc[:168].reset_index(drop=True), df_daily, "Open-Meteo (ECMWF/GFS Seamless)"
    except Exception:
        pass

    # Automatic Fallback: MET Norway Locationforecast 2.0 (Free, No API Key, ECMWF-backed)
    met_url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=30.34&lon=76.39"
    met_json = _http_get_json(met_url, timeout=15)
    rows = []
    for entry in met_json["properties"]["timeseries"]:
        dt_ist = pd.to_datetime(entry["time"], utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
        inst = entry.get("data", {}).get("instant", {}).get("details", {})
        rain = (
            entry.get("data", {}).get("next_1_hours", {}).get("details", {}).get("precipitation_amount")
            or entry.get("data", {}).get("next_6_hours", {}).get("details", {}).get("precipitation_amount")
            or 0.0
        )
        rows.append({
            "time": dt_ist,
            "temperature_2m": float(inst.get("air_temperature", np.nan)),
            "relative_humidity_2m": float(inst.get("relative_humidity", np.nan)),
            "surface_pressure": float(inst.get("air_pressure_at_sea_level", np.nan)),
            "cloud_cover": float(inst.get("cloud_area_fraction", np.nan)),
            "wind_speed_10m": float(inst.get("wind_speed", np.nan)) * 3.6,
            "wind_direction_10m": float(inst.get("wind_from_direction", np.nan)),
            "precipitation": float(rain),
        })

    df_h = (
        pd.DataFrame(rows)
        .set_index("time")
        .resample("1h")
        .interpolate(method="time")
        .reset_index()
    )
    df_h["precipitation"] = df_h["precipitation"].fillna(0.0)
    df_h = df_h.iloc[:168].reset_index(drop=True)

    df_daily = (
        df_h.assign(day=df_h["time"].dt.floor("D"))
        .groupby("day", as_index=False)
        .agg(
            temperature_2m_max=("temperature_2m", "max"),
            temperature_2m_min=("temperature_2m", "min"),
            precipitation_sum=("precipitation", "sum"),
            wind_speed_10m_max=("wind_speed_10m", "max"),
        )
        .rename(columns={"day": "time"})
        .iloc[:7]
        .reset_index(drop=True)
    )
    return df_h, df_daily, "MET Norway Locationforecast 2.0 (ECMWF Fallback)"

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_live_5x5_grid():
    """Fetches live 5x5 regional weather image independently so a 429 never blocks Tab 1."""
    lats = [31.34, 30.84, 30.34, 29.84, 29.34]
    lons = [75.39, 75.89, 76.39, 76.89, 77.39]
    lat_list = [str(la) for la in lats for _ in lons]
    lon_list = [str(lo) for _ in lats for lo in lons]
    params_grid = urllib.parse.urlencode({
        "latitude": ",".join(lat_list),
        "longitude": ",".join(lon_list),
        "current": "temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "Asia/Kolkata"
    })
    time.sleep(1.2)
    grid_json = _http_get_json(f"https://api.open-meteo.com/v1/forecast?{params_grid}", timeout=18)
    ch_names = [
        "temperature_2m", "surface_pressure", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m", "cloud_cover"
    ]
    live_img = np.zeros((6, 5, 5), dtype=np.float32)
    for idx, cell in enumerate(grid_json):
        r, c = divmod(idx, 5)
        cur = cell.get("current", {})
        for ch_i, ch_k in enumerate(ch_names):
            live_img[ch_i, r, c] = float(cur.get(ch_k, np.nan))
    return live_img, lats, lons

@st.cache_data
def load_csv_safe(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

# -------------------------------------------------------------------------
# 2. Sidebar
# -------------------------------------------------------------------------
st.sidebar.title("AeroFuse-Weather-AI")
st.sidebar.caption("Dual-Modal CNN + Transformer Spatiotemporal System")
st.sidebar.markdown(
    """
    **Target Site:** Patiala, Punjab (`30.34°N, 76.39°E`)  
    **Dataset:** `61,368` hourly ERA5 records (`2019–2025`)  
    **Test Evaluation:** `9,172` isolated test windows  
    **Spatial Modality:** `6 × 5 × 5` Weather Image (`0.5°` mesh)  
    **Evaluation:** Chronological Out-of-Sample (2025)
    """
)
st.sidebar.divider()
show_cqr_band = st.sidebar.checkbox("Overlay Calibrated 80% Rolling CQR Band", value=True)
selected_day_view = st.sidebar.slider("Inspect Forecast Window (Hours Ahead)", 24, 168, 168, step=24)

# -------------------------------------------------------------------------
# 3. Top Banner & Verified KPI Metrics
# -------------------------------------------------------------------------
st.title("AeroFuse-Net: CNN Atmospheric Image Feature Extraction & Dual-Modal Temperature Forecasting")
st.caption("Live 7-Day Patiala Forecast & 13-Model Spatiotemporal Benchmark (v1–v4 & v6 Sparse Router)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("AeroFuse v4 RMSE (#1)", "1.4378 °C", "R² = 0.9674 (#1 Overall)")
k2.metric("iTransformer MAE (#1)", "1.0217 °C", "v4 = 1.0366 °C (Best Hybrid)")
k3.metric("Monsoon RMSE Advantage", "1.0754 °C", "-4.4% vs iTransformer")
k4.metric("Extreme Tail Safety (P99)", "4.79 °C", "-0.24 °C vs iTransformer")
k5.metric("Rolling 30d CQR Coverage", "79.4% PICP", "Restored from 73.05% Raw")

# -------------------------------------------------------------------------
# 4. Main Tabs
# -------------------------------------------------------------------------
tabs = st.tabs([
    "🌤️ Live 7-Day Patiala Forecast (168h)",
    "🖼️ CNN Image Feature Extraction Lab",
    "📊 13-Model & Regime Benchmarks",
    "⚖️ Statistical Tests & Findings"
])

live_img, lats, lons = None, None, None

# ==================== TAB 1: LIVE 7-DAY PATIALA FORECAST ====================
with tabs[0]:
    st.subheader("Live Next-7-Day (168-Hour) Temperature Forecast for Patiala, Punjab (IST)")
    st.info(
        "**Operational Transparency:** Live 7-day NWP trajectories are fetched from Open-Meteo (with automatic MET Norway "
        "ECMWF fallback). AeroFuse-Net's calibrated **80% Rolling CQR Uncertainty Envelope** (empirical width `1.62°C` at `t+1h` "
        "scaling to `3.78°C` at `t+24h` from our 2025 held-out test set, extended heuristically across Days 2–7) is overlaid directly."
    )

    # OUTER TRY: Fetches 1-point Patiala 7-day forecast (with automatic backup API)
    try:
        df_h, df_daily, api_source = fetch_live_patiala_7day()
        df_next = df_h.iloc[:selected_day_view].reset_index(drop=True)
        st.caption(f"Live NWP Feed Source: **{api_source}** (Cached 3h)")

        # Daily 7-Day Summary Strip
        if len(df_daily) > 0:
            day_cols = st.columns(len(df_daily))
            for i, row in df_daily.iterrows():
                dt_lbl = pd.to_datetime(row["time"]).strftime("%a %d %b")
                day_cols[i].metric(
                    dt_lbl,
                    f"{row['temperature_2m_max']:.0f}°/{row['temperature_2m_min']:.0f}°C",
                    f"Rain: {row['precipitation_sum']:.1f} mm"
                )

        # INNER TRY: Fetches 25-point 5x5 regional grid independently so a 429 never blocks Tab 1
        try:
            live_img, lats, lons = fetch_live_5x5_grid()
            spatial_temp_spread = float(np.nanstd(live_img[0]))
            is_volatile_now = spatial_temp_spread > 1.35
            if is_volatile_now:
                st.warning(
                    f"**Live Spatial Gradient Indicator: ELEVATED** — Regional 5×5 Temp Spread = `{spatial_temp_spread:.2f}°C` "
                    f"(rule-based indicator; AeroFuse-Net historically cuts monsoon RMSE by 4.4% & volatile RMSE by 2.3%)."
                )
            else:
                st.success(
                    f"**Live Spatial Gradient Indicator: CALM** — Regional 5×5 Temp Spread = `{spatial_temp_spread:.2f}°C` "
                    f"(local temporal dynamics typically dominate in calm radiative conditions)."
                )
        except Exception:
            st.caption("Live 25-cell regional grid is temporarily rate-limited on the shared cloud IP; 7-day forecast and trained CNN diagnostics remain active.")

        # Plot 7-Day Interactive Temperature Trajectory + Calibrated CQR Band
        hrs = np.arange(len(df_next))
        temp_fc = df_next["temperature_2m"].values
        cqr_hw = np.minimum(1.89 + 0.15 * np.floor(hrs / 24.0), 2.65)
        n_first_day = min(24, len(cqr_hw))
        cqr_hw[:n_first_day] = np.linspace(0.81, 1.89, n_first_day)

        fig, ax = plt.subplots(figsize=(14, 4.6), dpi=160)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        if show_cqr_band:
            ax.fill_between(
                df_next["time"], temp_fc - cqr_hw, temp_fc + cqr_hw,
                color="#38b2ac", alpha=0.22,
                label="80% Rolling CQR Band (0–24h calibrated; Days 2–7 heuristic)"
            )
        ax.plot(df_next["time"], temp_fc, color="#4fd1c5", linewidth=2.2, label=f"Live NWP Forecast ({api_source})")
        for d_idx in range(24, len(df_next), 24):
            ax.axvline(df_next["time"].iloc[d_idx], color="#4a5568", linestyle=":", linewidth=0.9)
        ax.set_ylabel("2m Air Temperature (°C)", color="white")
        ax.set_xlabel("Date & Time (IST — Patiala, Punjab)", color="white")
        ax.set_title(
            f"Patiala Next-{selected_day_view}h ({selected_day_view // 24}-Day) Forecast & Uncertainty Envelope",
            color="white", fontweight="bold"
        )
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right", facecolor="#1a202c", labelcolor="white")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("View Live Hourly Patiala Forecast Table (Next 7 Days)"):
            st.dataframe(df_next, use_container_width=True)

    except Exception as e:
        st.error(f"Both primary (Open-Meteo) and fallback (MET Norway) live weather APIs are temporarily unavailable ({e}).")

# ==================== TAB 2: CNN IMAGE FEATURE EXTRACTION LAB ====================
with tabs[1]:
    st.subheader("2D-CNN Feature Extraction from Multi-Channel Atmospheric Grids")
    st.markdown(
        """
        Every hourly synoptic snapshot around Patiala (`±110 km`, `5 × 5` grid at `0.5°` spacing) is processed as a
        **6-channel meteorological image** alongside its **spatial anomaly image** ($\\Delta I = I - I_{\\text{Patiala}}$).
        Passing this tensor through learned $3 \\times 3$ 2D-CNN convolutional filters extracts horizontal thermal gradients
        ($\\nabla T$), barometric pressure troughs ($\\nabla P$), and upwind advective structures.
        """
    )

    if live_img is not None:
        st.markdown("#### A. Live Real-Time `5 × 5` Atmospheric Input Images Right Now Over Patiala")
        fig_g, axes_g = plt.subplots(1, 4, figsize=(16, 3.6), dpi=160)
        panels = [
            (live_img[0], "Ch 0: 2m Temperature (°C)", "coolwarm"),
            (live_img[0] - live_img[0, 2, 2], "Advection Image ΔT (°C)", "RdBu_r"),
            (live_img[1], "Ch 1: Surface Pressure (hPa)", "viridis"),
            (live_img[2], "Ch 2: Relative Humidity (%)", "YlGnBu"),
        ]
        for ax_i, (mat, ttl, cm) in zip(axes_g, panels):
            im = ax_i.imshow(mat, cmap=cm)
            ax_i.set_title(ttl, fontsize=9, fontweight="bold")
            ax_i.set_xticks(range(5))
            ax_i.set_xticklabels([f"{lo:.1f}°E" for lo in lons], fontsize=7)
            ax_i.set_yticks(range(5))
            ax_i.set_yticklabels([f"{la:.1f}°N" for la in lats], fontsize=7)
            ax_i.text(
                2, 2, "PAT", ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8),
                fontsize=7, fontweight="bold"
            )
            plt.colorbar(im, ax=ax_i, fraction=0.046)
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    st.divider()
    st.markdown("#### B. 2D-CNN Convolutional Feature Maps & Spatial Occlusion Analysis")
    if V6_FIG.exists():
        st.image(
            str(V6_FIG),
            caption="Trained 2D-CNN Conv2 Feature Activation Maps (Volatile Event vs. Calm Radiative Event)",
            use_container_width=True
        )
    else:
        st.warning("Run `src/evaluation/run_v6_real_cnn_image_and_deadzone_validation.py` to generate the 2D-CNN feature-map figure.")

    phys_json = V6_DIR / "real_cnn_feature_physics_validation.json"
    if phys_json.exists():
        with open(phys_json) as f:
            c_stats = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("r(CNN Features, |∇T|)", f"{c_stats['CNN_Feature_Energy_vs_Horizontal_Temp_Gradient_r']:.3f}", "Horizontal Thermal Gradient")
        c2.metric("r(CNN Features, |∇P|)", f"{c_stats['CNN_Feature_Energy_vs_Horizontal_Pressure_Gradient_r']:.3f}", "Barometric Trough Gradient")
        c3.metric("Gate α (Calm Hours)", f"{c_stats['Calm_Hours_Mean_Alpha']:.4f}", "Calm Regime Mean Gate")
        c4.metric("Gate α (Volatile Hours)", f"{c_stats['Volatile_Hours_Mean_Alpha']:.4f}", "Volatile Regime Mean Gate")

    df_occ = load_csv_safe(V6_DIR / "real_cnn_image_occlusion_validation.csv")
    if not df_occ.empty:
        st.markdown("**Spatial Occlusion & Grid-Permutation Sensitivity (Held-Out Evaluation):**")
        st.dataframe(df_occ, use_container_width=True)

# ==================== TAB 3: 13-MODEL & REGIME BENCHMARKS ====================
with tabs[2]:
    st.subheader("13-Model Benchmark (`9,172` Isolated Held-Out Test Windows, 2025)")
    df_bench = load_csv_safe(METRICS_DIR / "benchmark_comparison.csv")
    if not df_bench.empty:
        st.dataframe(df_bench, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Seasonal & Extreme Weather Regimes (RMSE in °C)")
        df_reg = load_csv_safe(METRICS_DIR / "seasonal_and_extreme_regime_evaluation.csv")
        if not df_reg.empty:
            st.dataframe(df_reg, use_container_width=True)
    with col_b:
        st.markdown("#### Calm (`77.3%`) vs. Volatile (`22.7%`) Regime Breakdown")
        df_cv = pd.DataFrame([
            {
                "Atmospheric Regime": "Calm Hours (77.3% of year)",
                "iTransformer MAE": 0.992, "AeroFuse v4 MAE": 1.017,
                "iTransformer RMSE": 1.397, "AeroFuse v4 RMSE": 1.397,
                "Diagnostic Insight": "Soft gate adds +2.5% MAE jitter on calm skies"
            },
            {
                "Atmospheric Regime": "Volatile Hours (22.7% of year)",
                "iTransformer MAE": 1.122, "AeroFuse v4 MAE": 1.104,
                "iTransformer RMSE": 1.608, "AeroFuse v4 RMSE": 1.570,
                "Diagnostic Insight": "CNN spatial features cut RMSE by -2.3% & MAE by -1.6%"
            },
        ])
        st.dataframe(df_cv, use_container_width=True)

        df_v6_comp = load_csv_safe(V6_DIR / "real_v6_oos_comparison.csv")
        if not df_v6_comp.empty:
            st.markdown("#### Sparse Dead-Zone Router (v6) — Held-Out Evaluation")
            st.dataframe(df_v6_comp, use_container_width=True)

# ==================== TAB 4: STATISTICAL TESTS & FINDINGS ====================
with tabs[3]:
    st.subheader("Statistical Significance (Bartlett HAC HLN-DM) & Calibrated Uncertainty (Rolling CQR)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### `v4` vs `iTransformer` HLN-DM (Full Test Year, `9,172` Windows)")
        df_dm_v4 = pd.DataFrame([
            {"Horizon (HAC Lag)": "All 24h (lag=23)", "Loss": "Squared Error (MSE)", "Mean Diff (v4 better > 0)": "+0.0286 °C²", "HLN Stat": 0.73, "p-value": 0.47, "Reading": "Statistically equivalent overall"},
            {"Horizon (HAC Lag)": "All 24h (lag=23)", "Loss": "Absolute Error (MAE)", "Mean Diff (v4 better > 0)": "-0.0149 °C", "HLN Stat": -1.69, "p-value": 0.09, "Reading": "No difference at α=0.05"},
            {"Horizon (HAC Lag)": "t+1h (lag=0)", "Loss": "Absolute Error (MAE)", "Mean Diff (v4 better > 0)": "-0.0243 °C", "HLN Stat": -9.54, "p-value": "<0.0001", "Reading": "iTransformer better at 1h lead"},
            {"Horizon (HAC Lag)": "Monsoon (Jul–Sep)", "Loss": "Regime RMSE", "Mean Diff (v4 better > 0)": "-0.0500 °C (-4.4%)", "HLN Stat": "Regime #1", "p-value": "—", "Reading": "AeroFuse v4 clearest advantage"},
        ])
        st.dataframe(df_dm_v4, use_container_width=True)
    with col_s2:
        st.markdown("#### Rolling 30-Day CQR Calibration (`v4` Quantiles, Nominal 80%)")
        df_cqr = pd.DataFrame([
            {"Method": "Raw 10th–90th Quantiles", "All-24h PICP": "73.05%", "Mean Width (MPIW)": "2.82 °C", "t+1h / t+6h / t+24h PICP": "83.9% / 71.3% / 74.3%"},
            {"Method": "Static Split-CQR (First 20%)", "All-24h PICP": "82.90%", "Mean Width (MPIW)": "3.41 °C", "t+1h / t+6h / t+24h PICP": "84.6% / — / 77.8%"},
            {"Method": "Rolling 720h CQR (Per-Horizon)", "All-24h PICP": "79.40%", "Mean Width (MPIW)": "3.24 °C", "t+1h / t+6h / t+24h PICP": "80.4% / 79.5% / 79.0%"},
        ])
        st.dataframe(df_cqr, use_container_width=True)

    df_v6_dm = load_csv_safe(V6_DIR / "real_v6_hln_dm_tests.csv")
    if not df_v6_dm.empty:
        st.markdown("#### Sparse Dead-Zone Router (v6) HLN-DM Statistical Tests")
        st.dataframe(df_v6_dm, use_container_width=True)

    st.divider()
    st.success(
        """
        ### Key Research Findings
        1. **CNN Image Feature Extraction Works Selectively, Not Uniformly:** Treating the 5×5 regional atmospheric field as a 6-channel weather image allows 2D-CNNs to extract horizontal thermal and pressure gradients. Naive joint fusion (`v1`: 1.2139°C, `v3`: 1.2468°C MAE) degrades accuracy because 77.3% of hours are calm radiative cycles.
        2. **Decoupled Routing Solves the Multimodal Trade-off:** Freezing specialized experts and routing via CNN-extracted image features (`AeroFuse v4`) achieves the **best overall RMSE (1.4378°C)** and **$R^2$ (0.9674)**, reduces **Monsoon RMSE by 4.4%**, cuts **Volatile-hour RMSE by 2.3%**, and trims **99th-percentile extreme tail errors** from `5.03°C` to `4.79°C`.
        3. **Sparse Dead-Zone Routing + Rolling CQR Complete the System:** Conditioning a sparse dead-zone gate on CNN image features achieves statistically significant out-of-sample HLN-DM improvements over both `iTransformer` and `v4`, while horizon-specific Rolling 30-Day CQR restores **79.4%** empirical coverage across all 24 lead hours.
        """
    )