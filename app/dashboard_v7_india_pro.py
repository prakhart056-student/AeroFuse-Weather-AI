"""
AeroFuse-Net v7 — All-India Cross-Dataset Generalization & Live Forecast Dashboard
File: app/dashboard_v7_india_pro.py
"""

from pathlib import Path
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AeroFuse-Net | All-India Cross-City Weather AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# 1. Custom Dark Glassmorphism UI Theme (CSS)
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean dark gradient canvas */
    .stApp {
        background: radial-gradient(circle at 12% 10%, #111c2d 0%, #090d16 55%, #06080f 100%);
        color: #e2e8f0;
    }
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.14) 0%, rgba(56, 189, 248, 0.08) 50%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        background: linear-gradient(90deg, #5eead4, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.96rem;
        color: #94a3b8;
        margin: 0;
    }
    /* Glassmorphic KPI Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(45, 212, 191, 0.55);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }
    /* Modern Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.55);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 16px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.25), rgba(56, 189, 248, 0.22)) !important;
        color: #5eead4 !important;
        border: 1px solid rgba(45, 212, 191, 0.4) !important;
    }
    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# 2. Paths & City Registry (Patiala + 10 Unseen Validation Cities)
# -------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
V6_DIR      = ROOT / "results" / "v6_real_validation"
V6_FIG      = ROOT / "results" / "figures" / "v6_real_validation" / "fig_v6_real_cnn_image_feature_extraction.png"
V7_DIR      = ROOT / "results" / "v7_cross_city"
USER_AGENT  = "AeroFuse-Weather-AI/7.0 (https://github.com/prakhart056-student/AeroFuse-Weather-AI)"

CITY_CATALOG = {
    "Patiala (Train Origin)": (30.34, 76.39, "Indo-Gangetic Plains"),
    "Srinagar":               (34.08, 74.80, "Himalayan Valley"),
    "Jodhpur":                (26.24, 73.02, "Thar Arid Desert"),
    "Lucknow":                (26.85, 80.95, "Central Gangetic Plains"),
    "Bhopal":                 (23.26, 77.41, "Central Plateau"),
    "Ahmedabad":              (23.02, 72.57, "Western Semi-Arid"),
    "Mumbai":                 (19.08, 72.88, "West Coast Monsoon"),
    "Kolkata":                (22.57, 88.36, "Eastern Delta"),
    "Guwahati":               (26.14, 91.74, "Northeast Humid"),
    "Bengaluru":              (12.97, 77.59, "Deccan Plateau"),
    "Chennai":                (13.08, 80.27, "East Coast Tropical"),
}

# -------------------------------------------------------------------------
# 3. Resilient Live API Fetchers (Open-Meteo + MET Norway Automatic Fallback)
# -------------------------------------------------------------------------
def _http_get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_live_city_7day(lat: float, lon: float):
    # Primary: Open-Meteo (1 point)
    try:
        params = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": 7, "timezone": "Asia/Kolkata"
        })
        js = _http_get_json(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=12)
        df_h = pd.DataFrame(js["hourly"])
        df_h["time"] = pd.to_datetime(df_h["time"])
        df_d = pd.DataFrame(js["daily"]).iloc[:7].reset_index(drop=True)
        return df_h.iloc[:168].reset_index(drop=True), df_d, "Open-Meteo (ECMWF/GFS)"
    except Exception:
        pass

    # Fallback: MET Norway Locationforecast 2.0 (ECMWF-backed, Free, No API Key)
    met_url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    js = _http_get_json(met_url, timeout=15)
    rows = []
    for entry in js["properties"]["timeseries"]:
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
            "precipitation": float(rain),
        })
    df_h = pd.DataFrame(rows).set_index("time").resample("1h").interpolate(method="time").reset_index()
    df_h["precipitation"] = df_h["precipitation"].fillna(0.0)
    df_h = df_h.iloc[:168].reset_index(drop=True)
    df_d = (
        df_h.assign(day=df_h["time"].dt.floor("D"))
        .groupby("day", as_index=False)
        .agg(
            temperature_2m_max=("temperature_2m", "max"),
            temperature_2m_min=("temperature_2m", "min"),
            precipitation_sum=("precipitation", "sum"),
            wind_speed_10m_max=("wind_speed_10m", "max"),
        )
        .rename(columns={"day": "time"}).iloc[:7].reset_index(drop=True)
    )
    return df_h, df_d, "MET Norway Locationforecast 2.0 (ECMWF)"

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_live_5x5_grid(lat: float, lon: float):
    lats = [round(lat + d, 2) for d in [1.0, 0.5, 0.0, -0.5, -1.0]]
    lons = [round(lon + d, 2) for d in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    lat_l = [str(la) for la in lats for _ in lons]
    lon_l = [str(lo) for _ in lats for lo in lons]
    params = urllib.parse.urlencode({
        "latitude": ",".join(lat_l), "longitude": ",".join(lon_l),
        "current": "temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Kolkata"
    })
    time.sleep(1.2)
    js = _http_get_json(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=18)
    img = np.zeros((4, 5, 5), dtype=np.float32)
    keys = ["temperature_2m", "surface_pressure", "relative_humidity_2m", "wind_speed_10m"]
    for idx, cell in enumerate(js):
        r, c = divmod(idx, 5)
        cur = cell.get("current", {})
        for k_i, k_n in enumerate(keys):
            img[k_i, r, c] = float(cur.get(k_n, np.nan))
    return img, lats, lons

@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

# -------------------------------------------------------------------------
# 4. Sidebar Controls
# -------------------------------------------------------------------------
st.sidebar.markdown("## 🛰️ AeroFuse-Net v7")
st.sidebar.caption("Dual-Modal CNN + Transformer Weather AI")
st.sidebar.divider()

selected_city = st.sidebar.selectbox("📍 Select Indian City (Live 7-Day Feed)", list(CITY_CATALOG.keys()), index=0)
c_lat, c_lon, c_zone = CITY_CATALOG[selected_city]
st.sidebar.markdown(
    f"""
    **Coordinates:** `{c_lat:.2f}°N, {c_lon:.2f}°E`  
    **Climate Zone:** `{c_zone}`  
    **Training Origin:** `Patiala (2019–2023)`  
    **Zero-Shot Scope:** `10 Unseen Indian Cities (2025)`
    """
)
st.sidebar.divider()
show_cqr = st.sidebar.toggle("Overlay 80% Rolling CQR Envelope", value=True)
view_hrs = st.sidebar.slider("Forecast Horizon Window (Hours)", 24, 168, 168, step=24)

# -------------------------------------------------------------------------
# 5. Hero Banner & Top KPI Strip
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">AeroFuse-Net: Dual-Modal CNN + Transformer Spatiotemporal Weather AI</div>
        <p class="hero-sub">
            2D-CNN Atmospheric Image Feature Extraction · Horizon-Wise Expert Routing · Rolling Conformal Uncertainty ·
            Zero-Shot Cross-Dataset Validation Across 10 Unseen Indian Cities
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

v7_summary = load_json(V7_DIR / "v7_cross_city_summary.json")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Patiala 24h RMSE (#1)", "1.4378 °C", "R² = 0.9674 (AeroFuse v4)")
k2.metric("Patiala 24h MAE (#1)", "1.0217 °C", "iTransformer (v4: 1.0366°C)")
k3.metric("Monsoon RMSE Gain", "-4.44%", "1.075°C vs 1.125°C")
k4.metric("Rolling 30d CQR (80%)", "79.4% PICP", "Restored from 73.05% Raw")
if v7_summary:
    k5.metric(
        "10-City Zero-Shot RMSE",
        f"{v7_summary['Unseen_10Cities_Mean_AeroFuse_RMSE_C']:.3f} °C",
        f"Skill: +{v7_summary['Unseen_10Cities_Mean_AeroFuse_Skill_Pct']:.1f}% vs Diurnal"
    )
else:
    k5.metric("Zero-Shot Validation", "10 Cities", "Run v7 script to populate")

# -------------------------------------------------------------------------
# 6. Five Interactive Tabs
# -------------------------------------------------------------------------
tabs = st.tabs([
    "🌤️ Live 7-Day Forecast",
    "🗺️ Cross-Dataset 10-City India Validation",
    "🖼️ 2D-CNN Atmospheric Image Lab",
    "📊 13-Model & Regime Benchmarks",
    "⚖️ Statistical Rigor & Findings"
])

live_img, g_lats, g_lons = None, None, None

# ==================== TAB 1: LIVE 7-DAY INTERACTIVE PLOTLY FORECAST ====================
with tabs[0]:
    st.subheader(f"Live 7-Day (168h) Forecast — {selected_city}")
    try:
        df_h, df_d, src_lbl = fetch_live_city_7day(c_lat, c_lon)
        df_sub = df_h.iloc[:view_hrs].reset_index(drop=True)
        st.caption(
            f"Live NWP Boundary Feed: **{src_lbl}** (3h cache) · "
            f"Shaded band overlays AeroFuse-Net's calibrated 0–24h Rolling CQR half-widths (`±0.81°C → ±1.89°C`) "
            f"extended heuristically across Days 2–7."
        )

        # Compact 7-day strip
        d_cols = st.columns(len(df_d))
        for i, r in df_d.iterrows():
            d_cols[i].metric(
                pd.to_datetime(r["time"]).strftime("%a %d %b"),
                f"{r['temperature_2m_max']:.0f}° / {r['temperature_2m_min']:.0f}°C",
                f"Rain: {r['precipitation_sum']:.1f} mm"
            )

        # Independent 5x5 regional grid check (inner try so 429 never breaks forecast)
        try:
            live_img, g_lats, g_lons = fetch_live_5x5_grid(c_lat, c_lon)
            spread_t = float(np.nanstd(live_img[0]))
            if spread_t > 1.35:
                st.warning(f"**Regional Spatial Gradient: ELEVATED (`σ_T = {spread_t:.2f}°C`)** — Stronger horizontal temperature contrast detected across the 5×5 regional grid.")
            else:
                st.success(f"**Regional Spatial Gradient: CALM (`σ_T = {spread_t:.2f}°C`)** — Locally driven diurnal radiative regime across the 5×5 grid.")
        except Exception:
            st.caption("Live 25-cell spatial grid temporarily rate-limited on shared cloud IP; 7-day city forecast remains active.")

        # Interactive Plotly Forecast Chart
        hrs = np.arange(len(df_sub))
        t_vals = df_sub["temperature_2m"].values
        hw = np.minimum(1.89 + 0.15 * np.floor(hrs / 24.0), 2.65)
        n1 = min(24, len(hw))
        hw[:n1] = np.linspace(0.81, 1.89, n1)

        fig_fc = go.Figure()
        if show_cqr:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([df_sub["time"], df_sub["time"][::-1]]),
                y=np.concatenate([t_vals + hw, (t_vals - hw)[::-1]]),
                fill="toself",
                fillcolor="rgba(45, 212, 191, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="80% Rolling CQR Band (0–24h Calibrated; Days 2–7 Heuristic)"
            ))
        fig_fc.add_trace(go.Scatter(
            x=df_sub["time"], y=t_vals, mode="lines",
            line=dict(color="#2dd4bf", width=3),
            name=f"Live Temperature Forecast ({src_lbl})",
            hovertemplate="<b>%{x|%a %d %b, %H:%M IST}</b><br>Temp: %{y:.1f} °C<extra></extra>"
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.45)",
            height=430,
            margin=dict(l=20, r=20, t=35, b=20),
            xaxis_title="Date & Time (IST)",
            yaxis_title="2m Air Temperature (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        with st.expander("Inspect Live Hourly Forecast Table"):
            st.dataframe(df_sub, use_container_width=True)

    except Exception as exc:
        st.error(f"Live weather APIs temporarily unreachable ({exc}).")

# ==================== TAB 2: CROSS-DATASET 10-CITY INDIA VALIDATION ====================
with tabs[1]:
    st.subheader("Zero-Shot Cross-Dataset Validation Across 10 Unseen Indian Cities (2025)")
    st.markdown(
        "All models were trained **strictly on Patiala (2019–2023)** with **zero fine-tuning** and evaluated "
        "zero-shot across the entire 2025 year (`8,713` windows/city) on **10 geographically non-overlapping Indian cities** "
        "(`> 2.0°` footprint separation across every pair)."
    )

    df_v7 = load_csv(V7_DIR / "v7_cross_city_benchmark.csv")
    if df_v7.empty:
        st.info(
            "Run `src/evaluation/run_v7_cross_city_validation.py` once on your GPU to populate real zero-shot results for all 10 unseen cities."
        )
    else:
        df_unseen = df_v7[df_v7["Evaluation_Type"] != "In-Distribution (2025)"]
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("10-City Mean AeroFuse RMSE", f"{df_unseen['AeroFuse_v7_RMSE_C'].mean():.4f} °C")
        c_m2.metric("10-City Mean iTransformer RMSE", f"{df_unseen['iTransformer_RMSE_C'].mean():.4f} °C")
        c_m3.metric("10-City Mean Diurnal RMSE", f"{df_unseen['Diurnal_24h_RMSE_C'].mean():.4f} °C")
        c_m4.metric("AeroFuse Best-RMSE Cities", f"{int((df_unseen['Best_Model_RMSE'] == 'AeroFuse_v7').sum())} / 10")

        col_map, col_bar = st.columns([1.05, 1.25])

        with col_map:
            st.markdown("#### Interactive India Geographic Transfer Map")
            fig_map = px.scatter_geo(
                df_v7, lat="Latitude", lon="Longitude",
                color="AeroFuse_v7_RMSE_C",
                size=np.clip(df_v7["AeroFuse_Diurnal_Skill_Pct"], 10, 80),
                hover_name="City",
                hover_data={
                    "Climate_Zone": True,
                    "Distance_from_Patiala_km": True,
                    "AeroFuse_v7_RMSE_C": ":.4f",
                    "iTransformer_RMSE_C": ":.4f",
                    "AeroFuse_Diurnal_Skill_Pct": ":.1f",
                    "Best_Model_RMSE": True,
                    "Latitude": False, "Longitude": False
                },
                color_continuous_scale="Tealgrn_r",
                labels={"AeroFuse_v7_RMSE_C": "RMSE (°C)"},
                projection="mercator"
            )
            fig_map.update_geos(
                lataxis_range=[6.5, 36.5], lonaxis_range=[67.5, 97.5],
                showcountries=True, countrycolor="rgba(148,163,184,0.45)",
                showcoastlines=True, coastlinecolor="rgba(148,163,184,0.45)",
                showland=True, landcolor="#111827",
                showocean=True, oceancolor="#070b14",
                bgcolor="rgba(0,0,0,0)"
            )
            fig_map.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=460,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_bar:
            st.markdown("#### Model RMSE Comparison Across Climate Regimes")
            df_melt = df_v7.melt(
                id_vars=["City", "Climate_Zone"],
                value_vars=["AeroFuse_v7_RMSE_C", "iTransformer_RMSE_C", "CNN1D_RMSE_C", "Diurnal_24h_RMSE_C"],
                var_name="Model", value_name="RMSE (°C)"
            )
            df_melt["Model"] = df_melt["Model"].replace({
                "AeroFuse_v7_RMSE_C": "AeroFuse-v7 Router",
                "iTransformer_RMSE_C": "iTransformer (Local)",
                "CNN1D_RMSE_C": "1D-CNN ResNet",
                "Diurnal_24h_RMSE_C": "Diurnal 24h Baseline",
            })
            fig_bar = px.bar(
                df_melt, x="City", y="RMSE (°C)", color="Model", barmode="group",
                color_discrete_sequence=["#2dd4bf", "#f59e0b", "#38bdf8", "#64748b"]
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.45)",
                height=460,
                margin=dict(l=15, r=15, t=15, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Complete Zero-Shot Cross-City Validation Table (2025)")
        st.dataframe(df_v7, use_container_width=True)

# ==================== TAB 3: 2D-CNN ATMOSPHERIC IMAGE LAB ====================
with tabs[2]:
    st.subheader("2D-CNN Feature Extraction from Multi-Channel Atmospheric Grids")
    if live_img is not None:
        st.markdown(f"#### A. Live `5 × 5` Multi-Channel Weather Images Around {selected_city}")
        g1, g2, g3, g4 = st.columns(4)
        maps = [
            (g1, live_img[0], "2m Temp (°C)", "RdBu_r"),
            (g2, live_img[0] - live_img[0, 2, 2], "Advection ΔT (°C)", "Picnic"),
            (g3, live_img[1], "Pressure (hPa)", "Viridis"),
            (g4, live_img[2], "Humidity (%)", "YlGnBu"),
        ]
        for col_w, mat, title, cs in maps:
            fig_h = px.imshow(
                mat, x=[f"{lo:.1f}°E" for lo in g_lons], y=[f"{la:.1f}°N" for la in g_lats],
                color_continuous_scale=cs, text_auto=".1f", aspect="equal"
            )
            fig_h.update_layout(
                title=dict(text=title, font=dict(size=13)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=290, margin=dict(l=10, r=10, t=35, b=10),
                coloraxis_showscale=False
            )
            col_w.plotly_chart(fig_h, use_container_width=True)

    st.divider()
    st.markdown("#### B. Trained 2D-CNN Feature Activation Maps & Spatial Occlusion Sensitivity (`v6_real`)")
    if V6_FIG.exists():
        st.image(str(V6_FIG), use_container_width=True)
    phys = load_json(V6_DIR / "real_cnn_feature_physics_validation.json")
    if phys:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("r(CNN Energy, |∇T|)", f"{phys['CNN_Feature_Energy_vs_Horizontal_Temp_Gradient_r']:.3f}")
        p2.metric("r(CNN Energy, |∇P|)", f"{phys['CNN_Feature_Energy_vs_Horizontal_Pressure_Gradient_r']:.3f}")
        p3.metric("Mean Gate α (Calm)", f"{phys['Calm_Hours_Mean_Alpha']:.4f}")
        p4.metric("Mean Gate α (Volatile)", f"{phys['Volatile_Hours_Mean_Alpha']:.4f}")
    df_occ = load_csv(V6_DIR / "real_cnn_image_occlusion_validation.csv")
    if not df_occ.empty:
        st.dataframe(df_occ, use_container_width=True)

# ==================== TAB 4: 13-MODEL & REGIME BENCHMARKS ====================
with tabs[3]:
    st.subheader("Patiala 13-Model Benchmark (`9,172` Isolated Held-Out Windows, 2025)")
    df_b = load_csv(METRICS_DIR / "benchmark_comparison.csv")
    if not df_b.empty:
        if "Model" in df_b.columns and "Overall_RMSE" in df_b.columns:
            fig_b = px.bar(
                df_b.sort_values("Overall_RMSE", ascending=False),
                y="Model", x="Overall_RMSE", orientation="h",
                color="Overall_RMSE", color_continuous_scale="Tealgrn_r", text_auto=".4f"
            )
            fig_b.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.45)", height=380,
                margin=dict(l=15, r=20, t=15, b=15), coloraxis_showscale=False
            )
            st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(df_b, use_container_width=True)

    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("#### Seasonal & Extreme Regimes (RMSE °C)")
        df_reg = load_csv(METRICS_DIR / "seasonal_and_extreme_regime_evaluation.csv")
        if not df_reg.empty:
            st.dataframe(df_reg, use_container_width=True)
    with c_r:
        st.markdown("#### Sparse Dead-Zone Router (`v6_real`) OOS Split")
        df_v6 = load_csv(V6_DIR / "real_v6_oos_comparison.csv")
        if not df_v6.empty:
            st.dataframe(df_v6, use_container_width=True)

# ==================== TAB 5: STATISTICAL RIGOR & FINDINGS ====================
with tabs[4]:
    st.subheader("Autocorrelation-Aware Statistics (Bartlett-HAC HLN-DM) & Rolling CQR")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### `v4` vs `iTransformer` (Full Test Year, `9,172` Windows)")
        st.dataframe(pd.DataFrame([
            {"Horizon (HAC Lag)": "All 24h (lag=23)", "Loss": "MSE", "Mean Diff": "+0.0286 °C²", "HLN": 0.73, "p-val": 0.47, "Reading": "Statistically tied"},
            {"Horizon (HAC Lag)": "All 24h (lag=23)", "Loss": "MAE", "Mean Diff": "-0.0149 °C", "HLN": -1.69, "p-val": 0.09, "Reading": "No diff at 5%"},
            {"Horizon (HAC Lag)": "t+1h (lag=0)", "Loss": "MAE", "Mean Diff": "-0.0243 °C", "HLN": -9.54, "p-val": "<1e-4", "Reading": "iTransformer better"},
            {"Horizon (HAC Lag)": "Monsoon (Jul–Sep)", "Loss": "RMSE", "Mean Diff": "-4.44%", "HLN": "Regime #1", "p-val": "—", "Reading": "AeroFuse v4 leads"},
        ]), use_container_width=True)
    with col_s2:
        st.markdown("#### Rolling 30-Day CQR Calibration (`v4` Quantiles)")
        st.dataframe(pd.DataFrame([
            {"Method": "Raw 10–90% Quantiles", "All-24h PICP": "73.05%", "MPIW": "2.82 °C", "1h / 6h / 24h": "83.9% / 71.3% / 74.3%"},
            {"Method": "Static Split-CQR (20%)", "All-24h PICP": "82.90%", "MPIW": "3.41 °C", "1h / 6h / 24h": "84.6% / — / 77.8%"},
            {"Method": "Rolling 720h CQR", "All-24h PICP": "79.40%", "MPIW": "3.24 °C", "1h / 6h / 24h": "80.4% / 79.5% / 79.0%"},
        ]), use_container_width=True)

    df_v6_dm = load_csv(V6_DIR / "real_v6_hln_dm_tests.csv")
    if not df_v6_dm.empty:
        st.markdown("#### `v6_real` Sparse Dead-Zone Router HLN-DM Tests")
        st.dataframe(df_v6_dm, use_container_width=True)