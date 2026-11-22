"""
AeroFuse-Weather-AI — v7 Real Zero-Shot Cross-Dataset (10-City India) Validation
File: src/evaluation/run_v7_cross_city_validation.py

100% REAL EMPIRICAL EXECUTION (Zero synthetic numbers, strictly non-destructive):
1. Trains iTransformer, 1D-CNN ResNet, 2D Spatial ResNet, and AeroFuse-v7 Router
   STRICTLY on Patiala Training data (2019-2023, indices 0..42,955) using residual
   forecasting (T_{t+h} = T_t + dT) & spatial anomaly images (I - I_center) for zero-shot transfer.
2. Fetches unseen 2025 hourly data (8,760 hours x 19 local features + 6x5x5 grid)
   from Open-Meteo Archive API for 10 geographically non-overlapping Indian cities.
3. Evaluates the FROZEN Patiala-trained models zero-shot (zero fine-tuning) on all
   10 unseen Indian cities + Patiala 2025 reference, with Bartlett-HAC HLN-DM tests.
"""

from pathlib import Path
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Paths, Config & 10 Non-Overlapping Unseen Indian Cities
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_LOCAL_PATH = ROOT / "data" / "raw" / "local_station_timeseries.parquet"
RAW_GRID_PATH  = ROOT / "data" / "raw" / "spatial_grid_tensor.npz"

CACHE_DIR  = ROOT / "data" / "raw" / "cross_city_2025"
MODEL_DIR  = ROOT / "models" / "v7_cross_city"
OUT_DIR    = ROOT / "results" / "v7_cross_city"
FIG_DIR    = ROOT / "results" / "figures" / "v7_cross_city"

for d in [CACHE_DIR, MODEL_DIR, OUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USER_AGENT = "AeroFuse-Weather-AI/7.0 (https://github.com/prakhart056-student/AeroFuse-Weather-AI)"
SEQ_LEN = 24
HORIZON = 24

UNSEEN_CITIES = [
    {"city": "Srinagar",  "lat": 34.08, "lon": 74.80, "zone": "Himalayan Valley",       "dist_km": 441},
    {"city": "Jodhpur",   "lat": 26.24, "lon": 73.02, "zone": "Thar Arid Desert",       "dist_km": 562},
    {"city": "Lucknow",   "lat": 26.85, "lon": 80.95, "zone": "Central Gangetic Plains","dist_km": 590},
    {"city": "Bhopal",    "lat": 23.26, "lon": 77.41, "zone": "Central Plateau",        "dist_km": 792},
    {"city": "Ahmedabad", "lat": 23.02, "lon": 72.57, "zone": "Western Semi-Arid",      "dist_km": 897},
    {"city": "Mumbai",    "lat": 19.08, "lon": 72.88, "zone": "West Coast Monsoon",     "dist_km": 1299},
    {"city": "Kolkata",   "lat": 22.57, "lon": 88.36, "zone": "Eastern Delta",          "dist_km": 1469},
    {"city": "Guwahati",  "lat": 26.14, "lon": 91.74, "zone": "Northeast Humid",        "dist_km": 1572},
    {"city": "Bengaluru", "lat": 12.97, "lon": 77.59, "zone": "Deccan Plateau",         "dist_km": 1932},
    {"city": "Chennai",   "lat": 13.08, "lon": 80.27, "zone": "East Coast Tropical",    "dist_km": 1957},
]

LOCAL_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation"
]
GRID_VARS = [
    "temperature_2m", "surface_pressure", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover"
]

# -------------------------------------------------------------------------
# 2. Unified 19-Feature Engineering (Identical for Patiala & All 10 Cities)
# -------------------------------------------------------------------------
def build_19_features(df_raw: pd.DataFrame) -> np.ndarray:
    """Converts 12 raw meteorological variables + timestamp into exact (T, 19) float32 array."""
    df = df_raw.copy()
    if "dewpoint_2m" in df.columns and "dew_point_2m" not in df.columns:
        df["dew_point_2m"] = df["dewpoint_2m"]

    # Determine datetime index
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"])
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    else:
        dt = pd.date_range("2019-01-01 00:00:00", periods=len(df), freq="h")

    for col in LOCAL_VARS:
        if col not in df.columns:
            df[col] = 0.0
    df[LOCAL_VARS] = df[LOCAL_VARS].ffill().bfill().fillna(0.0)

    # 7 Engineered Features
    hour = dt.hour.values if hasattr(dt, "hour") else dt.dt.hour.values
    doy  = dt.dayofyear.values if hasattr(dt, "dayofyear") else dt.dt.dayofyear.values
    temp = df["temperature_2m"].values.astype(np.float32)
    rh   = np.clip(df["relative_humidity_2m"].values.astype(np.float32), 0.0, 100.0)
    pres = df["surface_pressure"].values.astype(np.float32)

    # Magnus-Tetens Vapor Pressure Deficit (kPa)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    vpd = es * (1.0 - rh / 100.0)

    t_roll24 = pd.Series(temp).rolling(24, min_periods=1).mean().values.astype(np.float32)
    p_std12  = pd.Series(pres).rolling(12, min_periods=1).std().fillna(0.0).values.astype(np.float32)

    feats = np.column_stack([
        df[LOCAL_VARS].values.astype(np.float32),
        np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
        np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
        np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
        np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
        vpd.astype(np.float32),
        t_roll24,
        p_std12,
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

def create_windows(local_norm: np.ndarray, grid_anom: np.ndarray, raw_temp: np.ndarray, stride: int = 1):
    """Builds sliding (X_local, X_grid_latest, base_temp, Y_target) arrays."""
    T = len(raw_temp)
    idxs = np.arange(0, T - SEQ_LEN - HORIZON + 1, stride)
    n = len(idxs)
    X_loc  = np.empty((n, SEQ_LEN, local_norm.shape[1]), dtype=np.float32)
    X_grid = np.empty((n, 12, 5, 5), dtype=np.float32)
    T_base = np.empty((n, 1), dtype=np.float32)
    Y_true = np.empty((n, HORIZON), dtype=np.float32)

    for k, i in enumerate(idxs):
        X_loc[k]  = local_norm[i : i + SEQ_LEN]
        X_grid[k] = grid_anom[i + SEQ_LEN - 1]
        T_base[k] = raw_temp[i + SEQ_LEN - 1]
        Y_true[k] = raw_temp[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
    return X_loc, X_grid, T_base, Y_true

def build_spatial_12ch(grid_raw: np.ndarray, ch_mean: np.ndarray, ch_std: np.ndarray) -> np.ndarray:
    """
    Creates 12-channel spatial image:
    Ch 0..5  : Per-channel anomaly relative to center cell (terrain-invariant for transfer!)
    Ch 6..11 : Global Patiala-standardized channel values (centered by city mean on Ch 0 & 1)
    """
    g = np.nan_to_num(grid_raw.astype(np.float32), nan=0.0)
    # Center-relative spatial contrast (pure upwind gradient, immune to city elevation & climate mean!)
    delta_center = (g - g[:, :, 2:3, 2:3]) / ch_std
    # Standardized local spatial variation (removes static city elevation/mean offset per channel)
    city_cell_mean = g.mean(axis=0, keepdims=True)
    g_anom = (g - city_cell_mean) / ch_std
    return np.concatenate([delta_center, g_anom], axis=1).astype(np.float32)

# -------------------------------------------------------------------------
# 3. Climate-Transferable PyTorch Models (Residual Forecasting: T_t + dT)
# -------------------------------------------------------------------------
class ResidualITransformer(nn.Module):
    def __init__(self, n_vars=19, seq_len=24, horizon=24, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.var_embed = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_vars * d_model, 128), nn.GELU(), nn.Linear(128, horizon))

    def forward(self, x_loc, t_base):
        # x_loc: (B, 24, 19) -> transpose to (B, 19, 24)
        tokens = self.var_embed(x_loc.transpose(1, 2))
        h = self.encoder(tokens)
        return t_base + self.head(h)

class Residual1DCNN(nn.Module):
    def __init__(self, n_vars=19, horizon=24):
        super().__init__()
        self.conv1 = nn.Conv1d(n_vars, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(256, horizon))

    def forward(self, x_loc, t_base):
        x = x_loc.transpose(1, 2)
        h1 = F.gelu(self.bn1(self.conv1(x)))
        h2 = F.gelu(self.bn2(self.conv2(h1))) + h1
        return t_base + self.head(h2)

class Residual2DSpatialCNN(nn.Module):
    def __init__(self, in_ch=12, horizon=24):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, horizon))

    def forward(self, x_grid, t_base):
        h = F.gelu(self.bn1(self.conv1(x_grid)))
        h = F.gelu(self.bn2(self.conv2(h)))
        return t_base + self.head(h)

class AeroFuseV7CrossCityRouter(nn.Module):
    """2D-CNN Weather-Image Encoder driving a horizon-wise router over 3 frozen experts."""
    def __init__(self, in_ch=12, n_vars=19, horizon=24):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, padding=1), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.BatchNorm2d(24), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(24 + n_vars + horizon, 48), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(48, 3 * horizon)
        )
        # Local-heavy horizon prior: [iTransformer, 1D-CNN, 2D-Spatial]
        prior = torch.tensor([1.25, 0.85, -0.40], dtype=torch.float32).view(1, 3, 1).repeat(1, 1, horizon)
        self.horizon_prior = nn.Parameter(prior)

    def forward(self, x_grid, x_loc_last, p_itran, p_cnn1d, p_cnn2d):
        cnn_vec = self.cnn(x_grid)
        disagreement = torch.abs(p_itran - p_cnn2d)
        logits = self.mlp(torch.cat([cnn_vec, x_loc_last, disagreement], dim=1)).view(-1, 3, 24)
        w = F.softmax(self.horizon_prior + 0.20 * torch.tanh(logits), dim=1)
        stacked = torch.stack([p_itran, p_cnn1d, p_cnn2d], dim=1)  # (B, 3, 24)
        return (w * stacked).sum(dim=1), w

# -------------------------------------------------------------------------
# 4. Polite Open-Meteo Archive Fetcher for Unseen 2025 Indian Cities
# -------------------------------------------------------------------------
def _get_json_with_retry(url: str, retries: int = 4, base_wait: float = 3.0):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            wait = base_wait * (2 ** attempt)
            print(f"      [Retry {attempt+1}/{retries}] {exc} — sleeping {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch URL after {retries} retries.")

def fetch_or_load_city_2025(city_info: dict):
    """Fetches (or loads cached) full 2025 hourly local + 5x5 grid data for an unseen Indian city."""
    cname = city_info["city"]
    cache_file = CACHE_DIR / f"{cname.lower()}_2025.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        return d["local_19"], d["grid_raw"], d["raw_temp"]

    lat, lon = city_info["lat"], city_info["lon"]
    print(f"[v7_fetch] Downloading unseen 2025 data for {cname} ({lat}N, {lon}E)...")

    # A. Local center point (12 variables, full 2025 year: 8,760 hours)
    params_loc = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "hourly": ",".join(LOCAL_VARS),
        "timezone": "UTC"
    })
    loc_json = _get_json_with_retry(f"https://archive-api.open-meteo.com/v1/archive?{params_loc}")
    df_loc = pd.DataFrame(loc_json["hourly"])
    local_19 = build_19_features(df_loc)
    raw_temp = df_loc["temperature_2m"].ffill().bfill().values.astype(np.float32)
    time.sleep(2.0)

    # B. 5x5 spatial grid (25 coordinates batched in 1 multi-location archive call)
    lats = [round(lat + d, 2) for d in [1.0, 0.5, 0.0, -0.5, -1.0]]
    lons = [round(lon + d, 2) for d in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    lat_list = [str(la) for la in lats for _ in lons]
    lon_list = [str(lo) for _ in lats for lo in lons]

    params_grid = urllib.parse.urlencode({
        "latitude": ",".join(lat_list),
        "longitude": ",".join(lon_list),
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "hourly": ",".join(GRID_VARS),
        "timezone": "UTC"
    })
    grid_json = _get_json_with_retry(f"https://archive-api.open-meteo.com/v1/archive?{params_grid}")
    T_hrs = len(raw_temp)
    grid_raw = np.zeros((T_hrs, 6, 5, 5), dtype=np.float32)
    for idx, cell in enumerate(grid_json):
        r, c = divmod(idx, 5)
        h_dict = cell["hourly"]
        for ch_i, vname in enumerate(GRID_VARS):
            arr = pd.Series(h_dict[vname]).ffill().bfill().fillna(0.0).values[:T_hrs]
            grid_raw[:, ch_i, r, c] = arr.astype(np.float32)

    np.savez_compressed(cache_file, local_19=local_19, grid_raw=grid_raw, raw_temp=raw_temp)
    print(f"      -> Cached {cname}: local {local_19.shape}, grid {grid_raw.shape}")
    time.sleep(3.0)
    return local_19, grid_raw, raw_temp

# -------------------------------------------------------------------------
# 5. Statistical Helper (Bartlett-HAC HLN-DM)
# -------------------------------------------------------------------------
def hln_dm_test(loss_base: np.ndarray, loss_cand: np.ndarray, lag: int = 23):
    d = loss_base - loss_cand
    T = len(d)
    d_mean = float(np.mean(d))
    dc = d - d_mean
    var_hac = float(np.dot(dc, dc) / T)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        var_hac += 2.0 * w * float(np.dot(dc[k:], dc[:-k]) / T)
    var_hac = max(var_hac, 1e-12)
    dm = d_mean / np.sqrt(var_hac / T)
    h = lag + 1
    hln = dm * np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    pval = float(2.0 * stats.t.sf(abs(hln), df=T - 1))
    return d_mean, float(hln), pval

# -------------------------------------------------------------------------
# 6. Train Strictly on Patiala (2019–2023) & Save Checkpoints
# -------------------------------------------------------------------------
print(f"[v7] Loading Patiala training archive on {DEVICE}...")
df_pat = pd.read_parquet(RAW_LOCAL_PATH)
pat_local_19 = build_19_features(df_pat)
pat_raw_temp = df_pat["temperature_2m"].ffill().bfill().values.astype(np.float32)

grid_npz = np.load(RAW_GRID_PATH)
pat_grid_raw = grid_npz[grid_npz.files[0]].astype(np.float32)

# Exact chronological splits (Train: 0..42,955 | Val: 42,956..52,161 | Test: 52,162..61,367)
TR_END, VAL_END = 42956, 52162

# Scalers fitted STRICTLY on Patiala Train
loc_mean = pat_local_19[:TR_END].mean(axis=0, keepdims=True)
loc_std  = pat_local_19[:TR_END].std(axis=0, keepdims=True) + 1e-6
grid_ch_mean = pat_grid_raw[:TR_END].mean(axis=(0, 2, 3), keepdims=True)
grid_ch_std  = pat_grid_raw[:TR_END].std(axis=(0, 2, 3), keepdims=True) + 1e-6

pat_loc_norm = (pat_local_19 - loc_mean) / loc_std
pat_grid_12  = build_spatial_12ch(pat_grid_raw, grid_ch_mean, grid_ch_std)

Xtr_l, Xtr_g, Ttr_b, Ytr = create_windows(pat_loc_norm[:TR_END], pat_grid_12[:TR_END], pat_raw_temp[:TR_END])
Xva_l, Xva_g, Tva_b, Yva = create_windows(pat_loc_norm[TR_END:VAL_END], pat_grid_12[TR_END:VAL_END], pat_raw_temp[TR_END:VAL_END])
Xte_l, Xte_g, Tte_b, Yte = create_windows(pat_loc_norm[VAL_END:], pat_grid_12[VAL_END:], pat_raw_temp[VAL_END:])

def train_expert(model, name, use_grid=False, epochs=12, lr=1e-3):
    ckpt = MODEL_DIR / f"{name}_patiala_only.pt"
    model = model.to(DEVICE)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        print(f"   [Loaded cached Patiala checkpoint] {ckpt.name}")
        return model

    torch.manual_seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_in = Xtr_g if use_grid else Xtr_l
    X_va_in = Xva_g if use_grid else Xva_l

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr_in), torch.from_numpy(Ttr_b), torch.from_numpy(Ytr)),
                       batch_size=256, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va_in), torch.from_numpy(Tva_b), torch.from_numpy(Yva)),
                       batch_size=512, shuffle=False)

    best_mse, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        for xb, tb, yb in dl_tr:
            xb, tb, yb = xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, tb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        va_sq, n_va = 0.0, 0
        with torch.no_grad():
            for xb, tb, yb in dl_va:
                p = model(xb.to(DEVICE), tb.to(DEVICE))
                va_sq += F.mse_loss(p, yb.to(DEVICE), reduction="sum").item()
                n_va += yb.numel()
        mse = va_sq / n_va
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, ckpt)
    print(f"   [Trained Patiala-only] {name:18s} | Best Val RMSE = {np.sqrt(best_mse):.4f} C")
    return model

@torch.no_grad()
def predict_expert(model, X_in, T_b):
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_in), torch.from_numpy(T_b)), batch_size=512, shuffle=False)
    outs = [model(xb.to(DEVICE), tb.to(DEVICE)).cpu().numpy() for xb, tb in dl]
    return np.concatenate(outs, axis=0)

print("[v7] Training / Loading Base Experts Strictly on Patiala (2019–2023)...")
m_itran = train_expert(ResidualITransformer(), "iTransformer_v7", use_grid=False, epochs=12)
m_cnn1d = train_expert(Residual1DCNN(),        "CNN1D_v7",        use_grid=False, epochs=12)
m_cnn2d = train_expert(Residual2DSpatialCNN(), "CNN2D_Spatial_v7",use_grid=True,  epochs=12)

# Train AeroFuse-v7 Router strictly on Patiala Validation split (frozen experts!)
print("[v7] Training / Loading AeroFuse-v7 Zero-Shot Router on Patiala Val (Frozen Experts)...")
router_ckpt = MODEL_DIR / "AeroFuse_v7_router_patiala_only.pt"
m_router = AeroFuseV7CrossCityRouter().to(DEVICE)

if router_ckpt.exists():
    m_router.load_state_dict(torch.load(router_ckpt, map_location=DEVICE, weights_only=True))
    print(f"   [Loaded cached Patiala checkpoint] {router_ckpt.name}")
else:
    p_va_it = predict_expert(m_itran, Xva_l, Tva_b)
    p_va_1d = predict_expert(m_cnn1d, Xva_l, Tva_b)
    p_va_2d = predict_expert(m_cnn2d, Xva_g, Tva_b)
    torch.manual_seed(42)
    opt_r = torch.optim.AdamW(m_router.parameters(), lr=2e-3, weight_decay=1e-4)
    dl_r = DataLoader(TensorDataset(
        torch.from_numpy(Xva_g), torch.from_numpy(Xva_l[:, -1, :]),
        torch.from_numpy(p_va_it), torch.from_numpy(p_va_1d), torch.from_numpy(p_va_2d),
        torch.from_numpy(Yva)
    ), batch_size=256, shuffle=True)
    for ep in range(15):
        m_router.train()
        for xg, xl, p1, p2, p3, yb in dl_r:
            xg, xl, p1, p2, p3, yb = [z.to(DEVICE) for z in (xg, xl, p1, p2, p3, yb)]
            opt_r.zero_grad()
            pred, _ = m_router(xg, xl, p1, p2, p3)
            loss = F.huber_loss(pred, yb, delta=1.0) + 0.05 * F.l1_loss(pred[:, 1:] - pred[:, :-1], yb[:, 1:] - yb[:, :-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_router.parameters(), 1.0)
            opt_r.step()
    torch.save(m_router.state_dict(), router_ckpt)
    print("   [Trained Patiala-only] AeroFuse_v7_Router saved.")

@torch.no_grad()
def predict_router(X_g, X_l, p_it, p_1d, p_2d):
    m_router.eval()
    dl = DataLoader(TensorDataset(
        torch.from_numpy(X_g), torch.from_numpy(X_l[:, -1, :]),
        torch.from_numpy(p_it), torch.from_numpy(p_1d), torch.from_numpy(p_2d)
    ), batch_size=512, shuffle=False)
    preds, wts = [], []
    for xg, xl, p1, p2, p3 in dl:
        p, w = m_router(xg.to(DEVICE), xl.to(DEVICE), p1.to(DEVICE), p2.to(DEVICE), p3.to(DEVICE))
        preds.append(p.cpu().numpy())
        wts.append(w.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(wts, axis=0)

# -------------------------------------------------------------------------
# 7. Run Strict Zero-Shot Cross-Dataset Evaluation on All 10 Unseen Cities + Patiala
# -------------------------------------------------------------------------
def evaluate_city_split(city_name, lat, lon, zone, dist_km, split_type, X_l, X_g, T_b, Y_t, raw_temp_series):
    p_it = predict_expert(m_itran, X_l, T_b)
    p_1d = predict_expert(m_cnn1d, X_l, T_b)
    p_2d = predict_expert(m_cnn2d, X_g, T_b)
    p_af, w_af = predict_router(X_g, X_l, p_it, p_1d, p_2d)

    # Reference baselines on exact same windows
    p_pers = np.repeat(T_b, HORIZON, axis=1)
    # Diurnal 24h baseline: same hour yesterday = input window's 24h raw temperatures
    n_w = len(Y_t)
    p_diurnal = np.array([raw_temp_series[i : i + HORIZON] for i in range(n_w)], dtype=np.float32)

    def metrics_of(pred):
        err = Y_t - pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((Y_t - Y_t.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        return mae, rmse, r2

    mae_it, rmse_it, r2_it = metrics_of(p_it)
    mae_1d, rmse_1d, r2_1d = metrics_of(p_1d)
    mae_2d, rmse_2d, r2_2d = metrics_of(p_2d)
    mae_af, rmse_af, r2_af = metrics_of(p_af)
    mae_di, rmse_di, _     = metrics_of(p_diurnal)
    mae_pe, rmse_pe, _     = metrics_of(p_pers)

    # Diurnal MSE Skill Score (%)
    skill_af = (1.0 - (rmse_af ** 2) / (rmse_di ** 2)) * 100.0
    skill_it = (1.0 - (rmse_it ** 2) / (rmse_di ** 2)) * 100.0

    # Bartlett-HAC HLN-DM (AeroFuse_v7 vs iTransformer on 24h window MSE)
    mse_it_w = np.mean((Y_t - p_it) ** 2, axis=1)
    mse_af_w = np.mean((Y_t - p_af) ** 2, axis=1)
    d_mse, hln_mse, p_mse = hln_dm_test(mse_it_w, mse_af_w, lag=23)

    best_rmse_model = min(
        [("AeroFuse_v7", rmse_af), ("iTransformer", rmse_it), ("1D-CNN", rmse_1d), ("2D-CNN", rmse_2d)],
        key=lambda x: x[1]
    )[0]

    return {
        "City": city_name, "Latitude": lat, "Longitude": lon, "Climate_Zone": zone,
        "Distance_from_Patiala_km": dist_km, "Evaluation_Type": split_type, "Test_Windows_N": n_w,
        "AeroFuse_v7_MAE_C": round(mae_af, 4), "AeroFuse_v7_RMSE_C": round(rmse_af, 4), "AeroFuse_v7_R2": round(r2_af, 4),
        "iTransformer_MAE_C": round(mae_it, 4), "iTransformer_RMSE_C": round(rmse_it, 4),
        "CNN1D_MAE_C": round(mae_1d, 4), "CNN1D_RMSE_C": round(rmse_1d, 4),
        "CNN2D_Spatial_MAE_C": round(mae_2d, 4), "CNN2D_Spatial_RMSE_C": round(rmse_2d, 4),
        "Diurnal_24h_RMSE_C": round(rmse_di, 4), "Persistence_RMSE_C": round(rmse_pe, 4),
        "AeroFuse_Diurnal_Skill_Pct": round(skill_af, 2),
        "iTransformer_Diurnal_Skill_Pct": round(skill_it, 2),
        "Mean_2D_CNN_Router_Weight": round(float(w_af[:, 2, :].mean()), 4),
        "HLN_DM_v7_vs_iTrans_MSE_Stat": round(hln_mse, 3),
        "HLN_DM_v7_vs_iTrans_p_value": round(p_mse, 5),
        "Best_Model_RMSE": best_rmse_model,
    }

results_rows = []

# 1. Patiala Held-Out 2025 (In-Distribution Reference)
print("\n[v7] Evaluating In-Distribution Reference: Patiala (2025 Held-Out Test)...")
row_pat = evaluate_city_split(
    "Patiala", 30.34, 76.39, "Indo-Gangetic Plains (Train Origin)", 0, "In-Distribution (2025)",
    Xte_l, Xte_g, Tte_b, Yte, pat_raw_temp[VAL_END:]
)
results_rows.append(row_pat)
print(f"   Patiala (In-Dist) -> AeroFuse RMSE={row_pat['AeroFuse_v7_RMSE_C']:.4f} C | iTrans={row_pat['iTransformer_RMSE_C']:.4f} C | Skill={row_pat['AeroFuse_Diurnal_Skill_Pct']:.1f}%")

# 2. All 10 Unseen Indian Cities (Zero-Shot Cross-Dataset Evaluation!)
print("\n[v7] Running Strict ZERO-SHOT Evaluation Across 10 Unseen Indian Cities (Full 2025 Year)...")
for cinfo in UNSEEN_CITIES:
    loc19, g_raw, r_temp = fetch_or_load_city_2025(cinfo)
    # Apply STRICTLY PATIALA's training scaler (zero leakage, zero fine-tuning!)
    loc_norm = (loc19 - loc_mean) / loc_std
    g_12 = build_spatial_12ch(g_raw, grid_ch_mean, grid_ch_std)
    Xl, Xg, Tb, Yt = create_windows(loc_norm, g_12, r_temp)
    row_c = evaluate_city_split(
        cinfo["city"], cinfo["lat"], cinfo["lon"], cinfo["zone"], cinfo["dist_km"],
        "Zero-Shot Unseen City (2025)", Xl, Xg, Tb, Yt, r_temp
    )
    results_rows.append(row_c)
    print(
        f"   {cinfo['city']:10s} ({cinfo['dist_km']:4d} km) -> "
        f"AeroFuse RMSE={row_c['AeroFuse_v7_RMSE_C']:.4f} C | iTrans={row_c['iTransformer_RMSE_C']:.4f} C | "
        f"1D-CNN={row_c['CNN1D_RMSE_C']:.4f} C | Diurnal Skill={row_c['AeroFuse_Diurnal_Skill_Pct']:5.1f}% | "
        f"Winner={row_c['Best_Model_RMSE']}"
    )

df_res = pd.DataFrame(results_rows)
df_res.to_csv(OUT_DIR / "v7_cross_city_benchmark.csv", index=False)

# Aggregate Zero-Shot Summary (strictly across the 10 unseen cities)
df_unseen = df_res[df_res["Evaluation_Type"] == "Zero-Shot Unseen City (2025)"]
summary = {
    "Patiala_InDist_RMSE_C": float(row_pat["AeroFuse_v7_RMSE_C"]),
    "Unseen_10Cities_Mean_AeroFuse_RMSE_C": round(float(df_unseen["AeroFuse_v7_RMSE_C"].mean()), 4),
    "Unseen_10Cities_Mean_iTransformer_RMSE_C": round(float(df_unseen["iTransformer_RMSE_C"].mean()), 4),
    "Unseen_10Cities_Mean_CNN1D_RMSE_C": round(float(df_unseen["CNN1D_RMSE_C"].mean()), 4),
    "Unseen_10Cities_Mean_CNN2D_RMSE_C": round(float(df_unseen["CNN2D_Spatial_RMSE_C"].mean()), 4),
    "Unseen_10Cities_Mean_Diurnal_RMSE_C": round(float(df_unseen["Diurnal_24h_RMSE_C"].mean()), 4),
    "Unseen_10Cities_Mean_AeroFuse_Skill_Pct": round(float(df_unseen["AeroFuse_Diurnal_Skill_Pct"].mean()), 2),
    "AeroFuse_Wins_Out_of_10_Unseen_Cities": int((df_unseen["Best_Model_RMSE"] == "AeroFuse_v7").sum()),
    "Best_Transfer_City": str(df_unseen.loc[df_unseen["AeroFuse_v7_RMSE_C"].idxmin(), "City"]),
    "Hardest_Transfer_City": str(df_unseen.loc[df_unseen["AeroFuse_v7_RMSE_C"].idxmax(), "City"]),
}
with open(OUT_DIR / "v7_cross_city_summary.json", "w") as fj:
    json.dump(summary, fj, indent=2)

# Save Static Publication Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=180)
x = np.arange(len(df_res))
w = 0.22
ax1.bar(x - 1.5*w, df_res["Diurnal_24h_RMSE_C"], w, label="Diurnal 24h Baseline", color="#718096")
ax1.bar(x - 0.5*w, df_res["CNN1D_RMSE_C"],       w, label="1D-CNN ResNet",        color="#4299e1")
ax1.bar(x + 0.5*w, df_res["iTransformer_RMSE_C"],w, label="iTransformer (Local)", color="#ed8936")
ax1.bar(x + 1.5*w, df_res["AeroFuse_v7_RMSE_C"], w, label="AeroFuse-v7 Router",   color="#1b9e77")
ax1.set_xticks(x)
ax1.set_xticklabels(df_res["City"], rotation=35, ha="right", fontsize=8.5)
ax1.set_ylabel("24-Hour Test RMSE (°C — Lower is Better)")
ax1.set_title("Zero-Shot Cross-City Generalization Across 10 Unseen Indian Cities (2025)", fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

ax2.scatter(df_unseen["Distance_from_Patiala_km"], df_unseen["AeroFuse_Diurnal_Skill_Pct"],
            s=90, c="#1b9e77", edgecolor="black", zorder=3)
for _, r in df_unseen.iterrows():
    ax2.annotate(r["City"], (r["Distance_from_Patiala_km"] + 25, r["AeroFuse_Diurnal_Skill_Pct"]), fontsize=8)
ax2.axhline(row_pat["AeroFuse_Diurnal_Skill_Pct"], color="#e53e3e", linestyle="--", label=f"Patiala In-Dist Skill ({row_pat['AeroFuse_Diurnal_Skill_Pct']:.1f}%)")
ax2.set_xlabel("Geographic Distance from Patiala Training Site (km)")
ax2.set_ylabel("AeroFuse-v7 Skill vs. Diurnal 24h Baseline (%)")
ax2.set_title("Zero-Shot Geographic Transfer Skill vs. Distance from Patiala", fontweight="bold")
ax2.legend(fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_v7_india_zero_shot_generalization.png", bbox_inches="tight")
plt.close(fig)

print("\n[v7] ===== ZERO-SHOT 10-CITY VALIDATION COMPLETE =====")
print(json.dumps(summary, indent=2))