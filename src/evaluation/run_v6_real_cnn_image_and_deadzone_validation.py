"""
AeroFuse-Weather-AI — Real v6 CNN Image Feature Extraction & Sparse Dead-Zone Validation
File: src/evaluation/run_v6_real_cnn_image_and_deadzone_validation.py

100% REAL EMPIRICAL EXECUTION (Zero synthetic or hardcoded metrics):
- Strictly Non-Destructive: writes exclusively to results/v6_real_validation/
- Validates Professor's Input: 2D-CNN feature extraction from 6-channel 5x5 meteorological images
  (convolutional feature maps, correlation with physical spatial gradients |grad T| & |grad P|,
  and real spatial occlusion / permutation impact).
- Trains a REAL 3-Seed Sparse Dead-Zone Router (Seeds 42, 100, 2026) conditioned on CNN
  image features over the real saved test/val predictions, evaluated with Bartlett HAC HLN-DM.
"""

from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Paths & Device Setup
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "results" / "metrics"
RAW_GRID_PATH = ROOT / "data" / "raw" / "spatial_grid_tensor.npz"
PROC_LOCAL_PATH = ROOT / "data" / "processed" / "processed_local_station.parquet"

OUT_DIR = ROOT / "results" / "v6_real_validation"
FIG_DIR = ROOT / "results" / "figures" / "v6_real_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[v6_real] Running on device: {DEVICE}")

# -------------------------------------------------------------------------
# 2. Auto-Discover & Load Real Saved Test Predictions (.npz)
# -------------------------------------------------------------------------
def load_saved_predictions(metrics_dir: Path):
    """Auto-discovers saved test prediction arrays in results/metrics/."""
    npz_files = list(metrics_dir.glob("*.npz"))
    found = {}
    y_true = None
    print(f"[v6_real] Scanning {metrics_dir} ({len(npz_files)} .npz files found)...")
    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        for k in data.files:
            arr = data[k]
            if arr.ndim == 2 and arr.shape[1] == 24 and arr.shape[0] >= 8000:
                kl = k.lower()
                fl = fpath.stem.lower()
                if any(t in kl for t in ["true", "target", "y_test", "actual"]):
                    y_true = arr.astype(np.float32)
                else:
                    found[f"{fpath.stem}::{k}"] = arr.astype(np.float32)
    return y_true, found

y_true, pred_dict = load_saved_predictions(METRICS_DIR)
if y_true is None or len(pred_dict) == 0:
    raise FileNotFoundError(
        f"Could not find 2D (N, 24) test prediction .npz files in {METRICS_DIR}. "
        f"Check existing .npz keys with np.load(...).files."
    )

def pick_model(pred_map, keywords):
    for key, arr in pred_map.items():
        kl = key.lower()
        if all(kw in kl for kw in keywords):
            return key, arr
    return None, None

k_itran, pred_itran = pick_model(pred_dict, ["itransformer"])
k_v4, pred_v4 = pick_model(pred_dict, ["v4"])
if k_v4 is None:
    k_v4, pred_v4 = pick_model(pred_dict, ["router"])
k_cnn1d, pred_cnn1d = pick_model(pred_dict, ["1d"])
k_grid2d, pred_grid2d = pick_model(pred_dict, ["2d"])

if pred_itran is None or pred_v4 is None:
    print("Available prediction keys:", list(pred_dict.keys()))
    raise KeyError("Need at least iTransformer and AeroFuse v4 saved predictions in results/metrics/.")

N_TEST = min(len(y_true), len(pred_itran), len(pred_v4))
y_true = y_true[:N_TEST]
pred_itran = pred_itran[:N_TEST]
pred_v4 = pred_v4[:N_TEST]
print(f"[v6_real] Loaded REAL test predictions (N={N_TEST}):")
print(f"   iTransformer ({k_itran}): MAE={np.mean(np.abs(y_true-pred_itran)):.4f} C | RMSE={np.sqrt(np.mean((y_true-pred_itran)**2)):.4f} C")
print(f"   AeroFuse v4  ({k_v4})   : MAE={np.mean(np.abs(y_true-pred_v4)):.4f} C | RMSE={np.sqrt(np.mean((y_true-pred_v4)**2)):.4f} C")

# -------------------------------------------------------------------------
# 3. Load Real 6-Channel 5x5 Atmospheric Images for the Test Split
# -------------------------------------------------------------------------
grid_npz = np.load(RAW_GRID_PATH)
grid_key = grid_npz.files[0]
grid_all = grid_npz[grid_key].astype(np.float32)  # Expected (61368, 6, 5, 5)
print(f"[v6_real] Loaded real spatial grid tensor: {grid_all.shape} (key='{grid_key}')")

# Align test-origin timestamps: test target_start indices run from 52173..61344 (9,172 windows)
# Origin hour (latest observed atmospheric image at t=0 before predicting t+1..t+24) is idx - 1
TEST_ORIGIN_START = 52172
grid_test_raw = grid_all[TEST_ORIGIN_START : TEST_ORIGIN_START + N_TEST]  # (N_TEST, 6, 5, 5)

# Standardize per channel using strictly Train partition (indices 0..42955)
train_grid = grid_all[:42956]
ch_mean = train_grid.mean(axis=(0, 2, 3), keepdims=True)
ch_std  = train_grid.std(axis=(0, 2, 3), keepdims=True) + 1e-6
grid_test_norm = (grid_test_raw - ch_mean) / ch_std

# Compute spatial anomaly image: Delta_I = I - I_center (isolates upwind advection contrast)
center_vals = grid_test_norm[:, :, 2:3, 2:3]
grid_adv = grid_test_norm - center_vals  # (N_TEST, 6, 5, 5)
# Stack raw image (6 ch) + advection image (6 ch) -> 12-channel atmospheric image (N_TEST, 12, 5, 5)
X_img_all = np.concatenate([grid_test_norm, grid_adv], axis=1)

# Physical horizontal spatial gradient magnitudes (for CNN feature validation)
# Channel 0 = Temperature, Channel 1 = Pressure (in ERA5 raw grid)
temp_img = grid_test_raw[:, 0, :, :]
pres_img = grid_test_raw[:, 1, :, :]
grad_T_mag = np.sqrt(np.gradient(temp_img, axis=1)**2 + np.gradient(temp_img, axis=2)**2).mean(axis=(1, 2))
grad_P_mag = np.sqrt(np.gradient(pres_img, axis=1)**2 + np.gradient(pres_img, axis=2)**2).mean(axis=(1, 2))
true_6h_vol = np.abs(y_true[:, 5] - y_true[:, 0])

# -------------------------------------------------------------------------
# 4. Define REAL 2D-CNN Atmospheric Image Feature Extractor + Sparse Dead-Zone Router
# -------------------------------------------------------------------------
class AtmosphericImageCNNSparseRouter(nn.Module):
    """
    2D-CNN Feature Extractor over 12-channel (6 raw + 6 advection) 5x5 Weather Images,
    feeding a Sparse Dead-Zone Gate: alpha = ReLU(sigmoid(z) - tau) / (1 - tau).
    Outputs horizon-wise spatial routing weights alpha in [0, 1]^(B, 24) that blend
    local expert (iTransformer) and spatial-aware expert (AeroFuse v4 / 2D-CNN).
    """
    def __init__(self, in_channels=12, hidden_ch=24, tau_init=0.45):
        super().__init__()
        # Block 1: Learnable 3x3 spatial differential filters over the 5x5 weather image
        self.conv1 = nn.Conv2d(in_channels, hidden_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        # Block 2: Higher-order mesoscale synoptic pattern extraction
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Gate head conditioned on CNN spatial features + expert disagreement signal
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_ch + 24, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24)
        )
        # Learnable dead-zone threshold tau bounded in [0.15, 0.85]
        self.tau_logit = nn.Parameter(torch.tensor(np.log(tau_init / (1.0 - tau_init)), dtype=torch.float32))

    def extract_cnn_feature_maps(self, x_img):
        f1 = F.gelu(self.bn1(self.conv1(x_img)))   # (B, hidden_ch, 5, 5)
        f2 = F.gelu(self.bn2(self.conv2(f1)))      # (B, hidden_ch, 5, 5)
        return f1, f2

    def forward(self, x_img, p_local, p_spatial):
        f1, f2 = self.extract_cnn_feature_maps(x_img)
        cnn_vec = self.pool(f2).flatten(1)  # (B, hidden_ch)
        expert_diff = torch.abs(p_spatial - p_local)  # (B, 24)
        logits = self.gate_head(torch.cat([cnn_vec, expert_diff], dim=1))
        z = torch.sigmoid(logits)
        tau = 0.15 + 0.70 * torch.sigmoid(self.tau_logit)
        # Exact Sparse Dead-Zone Activation: zero when z <= tau
        alpha = F.relu(z - tau) / (1.0 - tau + 1e-6)
        pred = (1.0 - alpha) * p_local + alpha * p_spatial
        return pred, alpha, f2, tau

# -------------------------------------------------------------------------
# 5. Strictly Causal Evaluation Split (First 30% of Test = Router Dev, Last 70% = Strict OOS Eval)
#    Note: Because saved .npz contains Test split only, we use a strict chronological
#    30% / 70% forward split with a 48h embargo gap so the router NEVER sees the OOS evaluation period!
# -------------------------------------------------------------------------
DEV_END = int(N_TEST * 0.30)
EMBARGO = 48
EVAL_START = DEV_END + EMBARGO

X_dev,  pl_dev,  ps_dev,  y_dev  = X_img_all[:DEV_END],    pred_itran[:DEV_END],    pred_v4[:DEV_END],    y_true[:DEV_END]
X_eval, pl_eval, ps_eval, y_eval = X_img_all[EVAL_START:], pred_itran[EVAL_START:], pred_v4[EVAL_START:], y_true[EVAL_START:]
N_EVAL = len(y_eval)
print(f"[v6_real] Strict Causal Router Split: Dev={DEV_END}h | Embargo={EMBARGO}h | Untouched OOS Eval={N_EVAL}h")

def hln_dm_test(loss_base, loss_cand, lag=23):
    """Harvey-Leybourne-Newbold Diebold-Mariano test with Bartlett HAC variance."""
    d = loss_base - loss_cand
    T = len(d)
    d_mean = float(np.mean(d))
    d_c = d - d_mean
    gamma0 = np.dot(d_c, d_c) / T
    var_hac = gamma0
    for k in range(1, lag + 1):
        w_k = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(d_c[k:], d_c[:-k]) / T
        var_hac += 2.0 * w_k * gamma_k
    var_hac = max(var_hac, 1e-12)
    dm_stat = d_mean / np.sqrt(var_hac / T)
    # HLN small-sample adjustment
    h = lag + 1
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    hln_stat = dm_stat * hln_factor
    p_val = float(2.0 * stats.t.sf(abs(hln_stat), df=T - 1))
    return d_mean, float(hln_stat), p_val

# -------------------------------------------------------------------------
# 6. Train REAL 3-Seed Sparse Dead-Zone Router (Seeds 42, 100, 2026)
# -------------------------------------------------------------------------
SEEDS = [42, 100, 2026]
seed_rows = []
best_model = None
best_eval_pred = None
best_eval_alpha = None
best_eval_f2 = None

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AtmosphericImageCNNSparseRouter(in_channels=12, hidden_ch=24, tau_init=0.45).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    ds_dev = TensorDataset(
        torch.from_numpy(X_dev), torch.from_numpy(pl_dev),
        torch.from_numpy(ps_dev), torch.from_numpy(y_dev)
    )
    dl_dev = DataLoader(ds_dev, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(25):
        for xb, plb, psb, yb in dl_dev:
            xb, plb, psb, yb = xb.to(DEVICE), plb.to(DEVICE), psb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred, alpha, _, _ = model(xb, plb, psb)
            # Combined L1 + L2 + small L1 sparsity regularizer + temporal derivative smoothness
            loss_mse = F.mse_loss(pred, yb)
            loss_mae = F.l1_loss(pred, yb)
            loss_grad = F.l1_loss(pred[:, 1:] - pred[:, :-1], yb[:, 1:] - yb[:, :-1])
            loss_sparse = 0.015 * alpha.mean()
            loss = 0.5 * loss_mse + 0.5 * loss_mae + 0.05 * loss_grad + loss_sparse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        xb_e = torch.from_numpy(X_eval).to(DEVICE)
        pl_e = torch.from_numpy(pl_eval).to(DEVICE)
        ps_e = torch.from_numpy(ps_eval).to(DEVICE)
        pred_e, alpha_e, f2_e, tau_val = model(xb_e, pl_e, ps_e)
        pred_np = pred_e.cpu().numpy()
        alpha_np = alpha_e.cpu().numpy()
        f2_np = f2_e.cpu().numpy()

    mae_e = float(np.mean(np.abs(y_eval - pred_np)))
    rmse_e = float(np.sqrt(np.mean((y_eval - pred_np)**2)))
    p95_e = float(np.percentile(np.abs(y_eval - pred_np), 95))
    grad_e = float(np.sqrt(np.mean(((pred_np[:, 1:] - pred_np[:, :-1]) - (y_eval[:, 1:] - y_eval[:, :-1]))**2)))
    sparsity_e = float(np.mean(alpha_np == 0.0) * 100.0)
    mean_alpha_e = float(np.mean(alpha_np))

    seed_rows.append({
        "Seed": seed,
        "Learned_Tau_Threshold": round(float(tau_val.item()), 4),
        "OOS_MAE_C": round(mae_e, 4),
        "OOS_RMSE_C": round(rmse_e, 4),
        "OOS_P95_AbsErr_C": round(p95_e, 4),
        "OOS_Grad_RMSE_C_per_h": round(grad_e, 4),
        "Exact_Zero_Gate_Sparsity_Pct": round(sparsity_e, 2),
        "Mean_Spatial_Gate_Alpha": round(mean_alpha_e, 4),
    })
    print(f"   [Seed {seed}] REAL OOS MAE={mae_e:.4f} C | RMSE={rmse_e:.4f} C | Sparsity={sparsity_e:.1f}% | tau={tau_val.item():.3f}")

    if seed == 42:
        best_model = model
        best_eval_pred = pred_np
        best_eval_alpha = alpha_np
        best_eval_f2 = f2_np

# Baseline metrics on the EXACT same untouched OOS evaluation period
def eval_metrics(name, preds):
    err = y_eval - preds
    return {
        "Model_Name": name,
        "OOS_Windows_N": N_EVAL,
        "OOS_MAE_C": round(float(np.mean(np.abs(err))), 4),
        "OOS_RMSE_C": round(float(np.sqrt(np.mean(err**2))), 4),
        "OOS_P95_AbsErr_C": round(float(np.percentile(np.abs(err), 95)), 4),
        "OOS_P99_AbsErr_C": round(float(np.percentile(np.abs(err), 99)), 4),
        "OOS_Grad_RMSE_C_per_h": round(float(np.sqrt(np.mean(((preds[:, 1:] - preds[:, :-1]) - (y_eval[:, 1:] - y_eval[:, :-1]))**2))), 4),
    }

df_comp = pd.DataFrame([
    eval_metrics("iTransformer_2026 (Local Only)", pl_eval),
    eval_metrics("AeroFuse_Net_v4 (Soft Spatial Router)", ps_eval),
    eval_metrics("AeroFuse_Net_v6_Real (CNN-Image Sparse Dead-Zone Router, Seed 42)", best_eval_pred),
])
df_seeds = pd.DataFrame(seed_rows)
df_comp.to_csv(OUT_DIR / "real_v6_oos_comparison.csv", index=False)
df_seeds.to_csv(OUT_DIR / "real_v6_3seed_runs.csv", index=False)

# Real HLN-DM tests on untouched OOS evaluation period
mse_itran = np.mean((y_eval - pl_eval)**2, axis=1)
mse_v4    = np.mean((y_eval - ps_eval)**2, axis=1)
mse_v6    = np.mean((y_eval - best_eval_pred)**2, axis=1)
mae_itran = np.mean(np.abs(y_eval - pl_eval), axis=1)
mae_v4    = np.mean(np.abs(y_eval - ps_eval), axis=1)
mae_v6    = np.mean(np.abs(y_eval - best_eval_pred), axis=1)

dm_rows = []
for label, lb, lc in [
    ("v6_Real vs iTransformer (24h Mean MSE)", mse_itran, mse_v6),
    ("v6_Real vs iTransformer (24h Mean MAE)", mae_itran, mae_v6),
    ("v6_Real vs v4_Router (24h Mean MSE)",    mse_v4,    mse_v6),
    ("v6_Real vs v4_Router (24h Mean MAE)",    mae_v4,    mae_v6),
]:
    dbar, hln, pval = hln_dm_test(lb, lc, lag=23)
    dm_rows.append({"Comparison": label, "HAC_Lag": 23, "Mean_Loss_Diff": round(dbar, 5), "HLN_DM_Stat": round(hln, 3), "p_value": round(pval, 6)})
df_dm = pd.DataFrame(dm_rows)
df_dm.to_csv(OUT_DIR / "real_v6_hln_dm_tests.csv", index=False)

# -------------------------------------------------------------------------
# 7. REAL CNN IMAGE FEATURE EXTRACTION VALIDATION (Professor's Requirement!)
#    A) Correlation between 2D-CNN feature activation norms & physical weather gradients
#    B) Real Spatial Occlusion & Grid-Shuffle Perturbation on Untouched OOS Data
# -------------------------------------------------------------------------
f2_spatial_energy = np.mean(np.abs(best_eval_f2), axis=(1, 2, 3))  # (N_EVAL,)
grad_T_eval = grad_T_mag[EVAL_START:]
grad_P_eval = grad_P_mag[EVAL_START:]
vol_eval    = true_6h_vol[EVAL_START:]
gate_mean_eval = best_eval_alpha.mean(axis=1)

corr_T, p_T = stats.pearsonr(f2_spatial_energy, grad_T_eval)
corr_P, p_P = stats.pearsonr(f2_spatial_energy, grad_P_eval)
corr_V, p_V = stats.pearsonr(gate_mean_eval, vol_eval)

# Real Occlusion & Spatial Shuffle Test
base_rmse = float(np.sqrt(np.mean((y_eval - best_eval_pred)**2)))
vol_mask = vol_eval >= np.percentile(vol_eval, 77.3)
base_vol_rmse = float(np.sqrt(np.mean((y_eval[vol_mask] - best_eval_pred[vol_mask])**2)))

occlusion_tests = [
    ("Unperturbed 5x5 Atmospheric Image (Reference)", None),
    ("Center Cell Occluded (Patiala 30.34N, 76.39E)", (2, 2)),
    ("NE Shivalik Foothills Occluded (30.84N, 77.39E)", (1, 4)),
    ("NW Amritsar Plains Occluded (31.34N, 75.39E)", (0, 0)),
    ("SW Rajasthan Border Occluded (29.34N, 75.39E)", (4, 0)),
    ("Full 5x5 Image Shuffled Across Days (Spatial Destruction)", "shuffle"),
]
occ_rows = []
with torch.no_grad():
    pl_e = torch.from_numpy(pl_eval).to(DEVICE)
    ps_e = torch.from_numpy(ps_eval).to(DEVICE)
    for desc, mode in occlusion_tests:
        X_pert = X_eval.copy()
        if mode == "shuffle":
            rng_s = np.random.default_rng(42)
            X_pert = X_pert[rng_s.permutation(len(X_pert))]
        elif isinstance(mode, tuple):
            r, c = mode
            X_pert[:, :, r, c] = 0.0
        p_pert, a_pert, _, _ = best_model(torch.from_numpy(X_pert).to(DEVICE), pl_e, ps_e)
        p_pert_np = p_pert.cpu().numpy()
        rmse_all = float(np.sqrt(np.mean((y_eval - p_pert_np)**2)))
        rmse_vol = float(np.sqrt(np.mean((y_eval[vol_mask] - p_pert_np[vol_mask])**2)))
        occ_rows.append({
            "Perturbation_Test": desc,
            "All_OOS_RMSE_C": round(rmse_all, 5),
            "Delta_All_RMSE_mC": round((rmse_all - base_rmse) * 1000.0, 3),
            "Volatile_Regime_RMSE_C": round(rmse_vol, 5),
            "Delta_Volatile_RMSE_mC": round((rmse_vol - base_vol_rmse) * 1000.0, 3),
            "Mean_Gate_Alpha": round(float(a_pert.mean().item()), 4),
        })

df_occ = pd.DataFrame(occ_rows)
df_occ.to_csv(OUT_DIR / "real_cnn_image_occlusion_validation.csv", index=False)

# Save real CNN physics-correlation summary & real predictions
cnn_summary = {
    "N_OOS_Windows": int(N_EVAL),
    "CNN_Feature_Energy_vs_Horizontal_Temp_Gradient_r": round(float(corr_T), 4),
    "CNN_Feature_Energy_vs_Horizontal_Temp_Gradient_p": float(p_T),
    "CNN_Feature_Energy_vs_Horizontal_Pressure_Gradient_r": round(float(corr_P), 4),
    "CNN_Feature_Energy_vs_Horizontal_Pressure_Gradient_p": float(p_P),
    "Sparse_Gate_Alpha_vs_6h_Temp_Volatility_r": round(float(corr_V), 4),
    "Calm_Hours_Mean_Alpha": round(float(gate_mean_eval[~vol_mask].mean()), 4),
    "Volatile_Hours_Mean_Alpha": round(float(gate_mean_eval[vol_mask].mean()), 4),
}
with open(OUT_DIR / "real_cnn_feature_physics_validation.json", "w") as fj:
    json.dump(cnn_summary, fj, indent=2)

np.savez_compressed(
    OUT_DIR / "real_v6_oos_predictions.npz",
    y_eval=y_eval,
    pred_itran=pl_eval,
    pred_v4=ps_eval,
    pred_v6_real=best_eval_pred,
    gate_alpha=best_eval_alpha,
)

# -------------------------------------------------------------------------
# 8. Plot REAL 2D-CNN Meteorological Image Feature Extraction Diagnostic Figure
# -------------------------------------------------------------------------
idx_storm = int(np.argmax(vol_eval))
idx_calm  = int(np.argmin(vol_eval))

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=180)
titles_row = [
    f"Volatile/Storm Event (|dT/6h|={vol_eval[idx_storm]:.1f}°C)",
    f"Calm Radiative Event (|dT/6h|={vol_eval[idx_calm]:.1f}°C)"
]
for row_i, ev_idx in enumerate([idx_storm, idx_calm]):
    # Col 0: Raw 5x5 Temperature Weather Image
    im0 = axes[row_i, 0].imshow(grid_test_raw[EVAL_START + ev_idx, 0], cmap="coolwarm")
    axes[row_i, 0].set_title(f"{titles_row[row_i]}\nRaw 5x5 Temp Image (°C)", fontsize=8.5, fontweight="bold")
    plt.colorbar(im0, ax=axes[row_i, 0], fraction=0.046)
    # Col 1: Spatial Advection Anomaly Image (Delta T = T - T_center)
    im1 = axes[row_i, 1].imshow(X_eval[ev_idx, 6], cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    axes[row_i, 1].set_title("Spatial Anomaly Image\n(ΔT = T_grid - T_Patiala)", fontsize=8.5)
    plt.colorbar(im1, ax=axes[row_i, 1], fraction=0.046)
    # Col 2: Learned 2D-CNN Conv2 Feature Activation Map (Mean across 24 filters)
    fmap = best_eval_f2[ev_idx].mean(axis=0)
    im2 = axes[row_i, 2].imshow(fmap, cmap="viridis")
    axes[row_i, 2].set_title("2D-CNN Extracted Feature Map\n(Mean of 24 Learned 3x3 Filters)", fontsize=8.5)
    plt.colorbar(im2, ax=axes[row_i, 2], fraction=0.046)
    # Col 3: Horizon-wise Sparse Dead-Zone Gate Alpha(h)
    axes[row_i, 3].bar(range(1, 25), best_eval_alpha[ev_idx], color="#1b9e77" if row_i == 0 else "#718096")
    axes[row_i, 3].set_ylim(0, 1.05)
    axes[row_i, 3].set_title(f"Sparse Gate α(h) Output\n(Mean α = {best_eval_alpha[ev_idx].mean():.3f})", fontsize=8.5)
    axes[row_i, 3].set_xlabel("Forecast Horizon (h)")
    axes[row_i, 3].set_ylabel("Spatial Gate Weight α")
    axes[row_i, 3].grid(axis="y", linestyle="--", alpha=0.4)

plt.suptitle(
    f"REAL 2D-CNN Meteorological Image Feature Extraction Proof | "
    f"r(CNN Energy, |∇T|) = {corr_T:.3f}, r(CNN Energy, |∇P|) = {corr_P:.3f}",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_v6_real_cnn_image_feature_extraction.png", bbox_inches="tight")
plt.close(fig)

print("\n[v6_real] ===== REAL VALIDATION COMPLETE =====")
print(df_comp.to_string(index=False))
print("\n[v6_real] Real HLN-DM Statistical Tests:")
print(df_dm.to_string(index=False))
print("\n[v6_real] Real CNN Feature Correlation Summary:", json.dumps(cnn_summary, indent=2))