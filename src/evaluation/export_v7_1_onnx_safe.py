"""
AeroFuse-Weather-AI — Export Frozen Patiala-Trained v7 Models to Verified CPU ONNX
File: src/evaluation/export_v7_1_onnx_safe.py
"""

from pathlib import Path
import inspect
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
RAW_LOCAL_PATH = ROOT / "data" / "raw" / "local_station_timeseries.parquet"
RAW_GRID_PATH  = ROOT / "data" / "raw" / "spatial_grid_tensor.npz"
MODEL_DIR      = ROOT / "models" / "v7_cross_city"
ONNX_DIR       = ROOT / "models" / "v7_onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation"
]

# -------------------------------------------------------------------------
# 1. Exact Trained v7 Model Definitions (For Loading Saved .pt Checkpoints)
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
        prior = torch.tensor([1.25, 0.85, -0.40], dtype=torch.float32).view(1, 3, 1).repeat(1, 1, horizon)
        self.horizon_prior = nn.Parameter(prior)

    def forward(self, x_grid, x_loc_last, p_itran, p_cnn1d, p_cnn2d):
        cnn_vec = self.cnn(x_grid)
        disagreement = torch.abs(p_itran - p_cnn2d)
        logits = self.mlp(torch.cat([cnn_vec, x_loc_last, disagreement], dim=1)).view(-1, 3, 24)
        w = F.softmax(self.horizon_prior + 0.20 * torch.tanh(logits), dim=1)
        stacked = torch.stack([p_itran, p_cnn1d, p_cnn2d], dim=1)
        return (w * stacked).sum(dim=1), w

# -------------------------------------------------------------------------
# 2. ONNX-Native Exact Wrappers (No aten::_transformer_encoder_layer_fwd)
# -------------------------------------------------------------------------
class OnnxSafeEncoderLayer(nn.Module):
    def __init__(self, src_layer: nn.TransformerEncoderLayer):
        super().__init__()
        self.d_model = src_layer.self_attn.embed_dim
        self.nhead = src_layer.self_attn.num_heads
        self.head_dim = self.d_model // self.nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.in_proj_weight = nn.Parameter(src_layer.self_attn.in_proj_weight.detach().clone())
        self.in_proj_bias   = nn.Parameter(src_layer.self_attn.in_proj_bias.detach().clone())
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj.weight.data.copy_(src_layer.self_attn.out_proj.weight.detach())
        self.out_proj.bias.data.copy_(src_layer.self_attn.out_proj.bias.detach())

        self.linear1 = src_layer.linear1
        self.linear2 = src_layer.linear2
        self.norm1   = src_layer.norm1
        self.norm2   = src_layer.norm2

    def forward(self, x):
        B, L, D = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs  = F.softmax(attn_scores, dim=-1)
        attn_out    = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, L, D)
        attn_out    = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.linear2(F.gelu(self.linear1(x))))
        return x

class OnnxSafeITransformer(nn.Module):
    def __init__(self, src: ResidualITransformer):
        super().__init__()
        self.var_embed = src.var_embed
        self.layers = nn.ModuleList([OnnxSafeEncoderLayer(l) for l in src.encoder.layers])
        self.head = src.head

    def forward(self, x_loc, t_base):
        h = self.var_embed(x_loc.transpose(1, 2))
        for layer in self.layers:
            h = layer(h)
        return t_base + self.head(h)

class OnnxSafe1DCNN(nn.Module):
    def __init__(self, src: Residual1DCNN):
        super().__init__()
        self.conv1 = src.conv1
        self.bn1 = src.bn1
        self.conv2 = src.conv2
        self.bn2 = src.bn2
        self.pool = nn.AvgPool1d(kernel_size=6, stride=6)
        self.flat = src.head[1]
        self.fc = src.head[2]

    def forward(self, x_loc, t_base):
        x = x_loc.transpose(1, 2)
        h1 = F.gelu(self.bn1(self.conv1(x)))
        h2 = F.gelu(self.bn2(self.conv2(h1))) + h1
        return t_base + self.fc(self.flat(self.pool(h2)))

class OnnxSafe2DSpatialCNN(nn.Module):
    def __init__(self, src: Residual2DSpatialCNN):
        super().__init__()
        self.conv1 = src.conv1
        self.bn1 = src.bn1
        self.conv2 = src.conv2
        self.bn2 = src.bn2
        self.fc1 = src.head[2]
        self.fc2 = src.head[4]

    def forward(self, x_grid, t_base):
        h = F.gelu(self.bn1(self.conv1(x_grid)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = h.mean(dim=(2, 3))
        return t_base + self.fc2(F.gelu(self.fc1(h)))

class OnnxSafeRouter(nn.Module):
    def __init__(self, src: AeroFuseV7CrossCityRouter):
        super().__init__()
        self.conv1, self.bn1 = src.cnn[0], src.cnn[1]
        self.conv2, self.bn2 = src.cnn[3], src.cnn[4]
        self.mlp = src.mlp
        self.horizon_prior = src.horizon_prior

    def forward(self, x_grid, x_loc_last, p_itran, p_cnn1d, p_cnn2d):
        h = F.gelu(self.bn1(self.conv1(x_grid)))
        h = F.gelu(self.bn2(self.conv2(h)))
        cnn_vec = h.mean(dim=(2, 3))
        disagreement = torch.abs(p_itran - p_cnn2d)
        logits = self.mlp(torch.cat([cnn_vec, x_loc_last, disagreement], dim=1)).reshape(-1, 3, 24)
        w = F.softmax(self.horizon_prior + 0.20 * torch.tanh(logits), dim=1)
        stacked = torch.stack([p_itran, p_cnn1d, p_cnn2d], dim=1)
        return (w * stacked).sum(dim=1), w

class AeroFuseV7EndToEnd(nn.Module):
    def __init__(self, m_itran, m_cnn1d, m_cnn2d, m_router):
        super().__init__()
        self.m_itran = m_itran
        self.m_cnn1d = m_cnn1d
        self.m_cnn2d = m_cnn2d
        self.m_router = m_router

    def forward(self, x_loc, x_grid, t_base):
        p_it = self.m_itran(x_loc, t_base)
        p_1d = self.m_cnn1d(x_loc, t_base)
        p_2d = self.m_cnn2d(x_grid, t_base)
        p_af, weights = self.m_router(x_grid, x_loc[:, -1, :], p_it, p_1d, p_2d)
        return p_af, p_it, p_1d, p_2d, weights

# -------------------------------------------------------------------------
# 3. Extract & Save Exact Patiala Training Scalers (Indices 0..42,955)
# -------------------------------------------------------------------------
def build_19_features(df_raw: pd.DataFrame) -> np.ndarray:
    df = df_raw.copy()
    if "dewpoint_2m" in df.columns and "dew_point_2m" not in df.columns:
        df["dew_point_2m"] = df["dewpoint_2m"]
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

    hour = dt.hour.values if hasattr(dt, "hour") else dt.dt.hour.values
    doy  = dt.dayofyear.values if hasattr(dt, "dayofyear") else dt.dt.dayofyear.values
    temp = df["temperature_2m"].values.astype(np.float32)
    rh   = np.clip(df["relative_humidity_2m"].values.astype(np.float32), 0.0, 100.0)
    pres = df["surface_pressure"].values.astype(np.float32)

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

print("[ONNX v7.1] Saving exact Patiala training scalers (2019-2023, indices 0..42,955)...")
TR_END = 42956
df_pat = pd.read_parquet(RAW_LOCAL_PATH)
pat_local_19 = build_19_features(df_pat)
loc_mean = pat_local_19[:TR_END].mean(axis=0, keepdims=True).astype(np.float32)
loc_std  = (pat_local_19[:TR_END].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

grid_npz = np.load(RAW_GRID_PATH)
pat_grid_raw = grid_npz[grid_npz.files[0]].astype(np.float32)
grid_ch_mean = pat_grid_raw[:TR_END].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
grid_ch_std  = (pat_grid_raw[:TR_END].std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)

scaler_path = ONNX_DIR / "v7_patiala_scalers.npz"
np.savez_compressed(
    scaler_path,
    loc_mean=loc_mean,
    loc_std=loc_std,
    grid_ch_mean=grid_ch_mean,
    grid_ch_std=grid_ch_std,
)
print(f"   -> Saved scalers: {scaler_path} ({scaler_path.stat().st_size / 1024:.1f} KB)")

# -------------------------------------------------------------------------
# 4. Load Checkpoints & Export ONNX-Native Wrapper (e2e_safe)
# -------------------------------------------------------------------------
required = {
    "iTransformer": MODEL_DIR / "iTransformer_v7_patiala_only.pt",
    "CNN1D":        MODEL_DIR / "CNN1D_v7_patiala_only.pt",
    "CNN2D":        MODEL_DIR / "CNN2D_Spatial_v7_patiala_only.pt",
    "Router":       MODEL_DIR / "AeroFuse_v7_router_patiala_only.pt",
}
for name, p in required.items():
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint {p}. Run run_v7_cross_city_validation.py first!")

m_it = ResidualITransformer()
m_1d = Residual1DCNN()
m_2d = Residual2DSpatialCNN()
m_rt = AeroFuseV7CrossCityRouter()

m_it.load_state_dict(torch.load(required["iTransformer"], map_location="cpu", weights_only=True))
m_1d.load_state_dict(torch.load(required["CNN1D"],        map_location="cpu", weights_only=True))
m_2d.load_state_dict(torch.load(required["CNN2D"],        map_location="cpu", weights_only=True))
m_rt.load_state_dict(torch.load(required["Router"],       map_location="cpu", weights_only=True))

e2e_orig = AeroFuseV7EndToEnd(m_it, m_1d, m_2d, m_rt).eval()
e2e_safe = AeroFuseV7EndToEnd(
    OnnxSafeITransformer(m_it),
    OnnxSafe1DCNN(m_1d),
    OnnxSafe2DSpatialCNN(m_2d),
    OnnxSafeRouter(m_rt),
).eval()

torch.manual_seed(42)
dummy_loc  = torch.randn(1, 24, 19, dtype=torch.float32)
dummy_grid = torch.randn(1, 12, 5, 5, dtype=torch.float32)
dummy_base = torch.tensor([[21.5]], dtype=torch.float32)

onnx_path = ONNX_DIR / "aerofuse_v7_e2e.onnx"
export_kwargs = dict(
    input_names=["x_loc", "x_grid", "t_base"],
    output_names=["pred_aerofuse", "pred_itran", "pred_cnn1d", "pred_cnn2d", "router_weights"],
    opset_version=17,
    do_constant_folding=True,
)
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    export_kwargs["dynamo"] = False

print("[ONNX v7.1] Exporting ONNX-safe AeroFuse-v7 end-to-end graph (static B=1)...")
with torch.no_grad():
    torch.onnx.export(e2e_safe, (dummy_loc, dummy_grid, dummy_base), str(onnx_path), **export_kwargs)

size_mb = onnx_path.stat().st_size / (1024 ** 2)
print(f"   -> Exported ONNX model: {onnx_path} ({size_mb:.2f} MB)")

# -------------------------------------------------------------------------
# 5. Strict Numerical Parity Test (ORIGINAL PyTorch vs. ONNX Runtime CPU)
# -------------------------------------------------------------------------
print("[ONNX v7.1] Running Original PyTorch vs. ONNX Runtime CPU verification...")
with torch.no_grad():
    pt_outs = [t.numpy() for t in e2e_orig(dummy_loc, dummy_grid, dummy_base)]

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ort_outs = sess.run(
    None,
    {
        "x_loc": dummy_loc.numpy(),
        "x_grid": dummy_grid.numpy(),
        "t_base": dummy_base.numpy(),
    },
)

names = ["pred_aerofuse", "pred_itran", "pred_cnn1d", "pred_cnn2d", "router_weights"]
for nm, pt_a, ort_a in zip(names, pt_outs, ort_outs):
    max_diff = float(np.max(np.abs(pt_a - ort_a)))
    print(f"   [Parity Check] {nm:15s} | shape={ort_a.shape} | max_abs_diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Parity check failed for {nm}: {max_diff}"

print("\n[ONNX v7.1] SUCCESS! Verified ONNX model and scalers are ready for Streamlit Cloud.")