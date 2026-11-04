import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.upgrade_aerofuse import (
    SEQ_LEN,
    HORIZON,
    prepare_dual_modal_splits
)
from src.models.architectures_v2 import (
    TemporalCNN1D,
    SpatialResNet2D,
    iTransformer2026,
    PatchTST2026,
    PhysicsInformedWeatherLoss
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# AEROFUSE-NET v4 ROUTER:
# Decoupled Multi-Modal Backbones + 2D-CNN Synoptic Horizon Router
# =====================================================================
class AeroFuseNetV4Router(nn.Module):
    """
    Version 4: Eliminates Multi-Modal Gradient Competition (observed in v3)
    by routing across decoupled temporal/variate/spatial backbones using a
    2D-CNN Synoptic Context Gate conditioned on horizontal advection gradients.
    """
    def __init__(self, local_features: int, grid_channels: int = 6, seq_len: int = 24, horizon: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon

        # 1. Expert Modal Backbones
        self.branch_itrans = iTransformer2026(local_features, seq_len, horizon)
        self.branch_cnn1d = TemporalCNN1D(local_features, seq_len, horizon)
        self.branch_patch = PatchTST2026(local_features, seq_len, horizon)
        self.branch_spatial2d = SpatialResNet2D(grid_channels, seq_len, horizon)

        # Load pre-trained single-modal weights so branches do not fight during warmup
        models_dir = os.path.join("data", "processed")
        ckpt_map = [
            (self.branch_itrans, "iTransformer_2026.pt"),
            (self.branch_cnn1d, "1D_CNN_ResNet.pt"),
            (self.branch_patch, "PatchTST_2026.pt"),
            (self.branch_spatial2d, "2D_Spatial_ResNet.pt"),
        ]
        for module, fname in ckpt_map:
            fpath = os.path.join(models_dir, fname)
            if os.path.exists(fpath):
                module.load_state_dict(torch.load(fpath, map_location="cpu", weights_only=True))

        for module, _ in ckpt_map:
            for param in module.parameters():
                param.requires_grad = False

        # 2. 2D-CNN Synoptic Advection Router (Extracts spatial front gradients)
        self.synoptic_conv = nn.Sequential(
            nn.Conv2d(grid_channels * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.context_router = nn.Sequential(
            nn.Linear(32 + local_features, 64),
            nn.GELU(),
            nn.Linear(64, 4 * horizon)
        )

        # Horizon-specific prior logits (4 branches x 24 lead hours)
        init_priors = torch.tensor([1.35, 0.95, 0.75, -0.35]).view(1, 4, 1).repeat(1, 1, horizon)
        self.horizon_priors = nn.Parameter(init_priors)

        # Fine-scale residual bias & scale calibration per horizon
        self.horizon_scale = nn.Parameter(torch.ones(1, horizon))
        self.horizon_bias = nn.Parameter(torch.zeros(1, horizon))

        # Calibrated Non-Crossing Quantile Uncertainty Heads (10th & 90th Percentiles)
        self.lower_offset = nn.Sequential(
            nn.Linear(32 + local_features, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )
        self.upper_offset = nn.Sequential(
            nn.Linear(32 + local_features, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x_local, x_grid, return_uncertainty: bool = False):
        B, T, C_g, H, W = x_grid.shape

        p_itrans = self.branch_itrans(x_local, x_grid)
        p_cnn1d = self.branch_cnn1d(x_local, x_grid)
        p_patch = self.branch_patch(x_local, x_grid)
        p_spat2d = self.branch_spatial2d(x_local, x_grid)

        expert_stack = torch.stack([p_itrans, p_cnn1d, p_patch, p_spat2d], dim=1)

        # Extract 2D Synoptic Advection Context from the latest timestep grid
        last_grid = x_grid[:, -1, :, :, :]
        center = last_grid[:, :, H // 2 : H // 2 + 1, W // 2 : W // 2 + 1]
        advection_grid = torch.cat([last_grid, last_grid - center], dim=1)

        sp_context = self.synoptic_conv(advection_grid)
        loc_context = x_local[:, -1, :]
        joint_context = torch.cat([sp_context, loc_context], dim=-1)

        dynamic_logits = self.context_router(joint_context).view(B, 4, self.horizon)
        gate_weights = F.softmax(self.horizon_priors + 0.15 * torch.tanh(dynamic_logits), dim=1)

        fused = torch.sum(gate_weights * expert_stack, dim=1)
        pred_mean = self.horizon_scale * fused + self.horizon_bias

        if return_uncertainty:
            ctx_det = joint_context.detach()
            base_det = pred_mean.detach()
            epistemic_std = expert_stack.detach().std(dim=1)
            q10 = base_det - (0.65 * epistemic_std + F.softplus(self.lower_offset(ctx_det)))
            q90 = base_det + (0.65 * epistemic_std + F.softplus(self.upper_offset(ctx_det)))
            return pred_mean, q10, q90

        return pred_mean


def train_aerofuse_v4(train_loader, val_loader, test_loader, num_features, device):
    print("=" * 75)
    print(f"1. Training AeroFuse_Net_v4_Router (Decoupled Horizon-Gated Fusion) on {device}...")
    print("=" * 75)

    epochs = 10
    model = AeroFuseNetV4Router(num_features, 6, SEQ_LEN, HORIZON).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PhysicsInformedWeatherLoss(lambda_phys=0.05, lambda_quant=0.25)

    best_val_mse = float("inf")
    best_path = os.path.join("data", "processed", "AeroFuse_Net_v4_Router.pt")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        if epoch == 8:
            for p in model.branch_itrans.proj.parameters():
                p.requires_grad = True
            for p in model.branch_cnn1d.head.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.branch_itrans.proj.parameters(), "lr": 1e-4})
            optimizer.add_param_group({"params": model.branch_cnn1d.head.parameters(), "lr": 1e-4})

        model.train()
        train_loss = 0.0
        for x_loc, x_grd, y in train_loader:
            x_loc, x_grd, y = x_loc.to(device), x_grd.to(device), y.to(device)
            optimizer.zero_grad()
            pred, q10, q90 = model(x_loc, x_grd, return_uncertainty=True)
            loss = criterion(pred, y, q10, q90)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for x_loc, x_grd, y in val_loader:
                x_loc, x_grd, y = x_loc.to(device), x_grd.to(device), y.to(device)
                pred = model(x_loc, x_grd)
                val_mse += F.mse_loss(pred, y).item() * len(y)
        val_mse /= len(val_loader.dataset)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), best_path)

        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            w_avg = F.softmax(model.horizon_priors, dim=1).mean(dim=-1).squeeze().detach().cpu().numpy()
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Modal Weights: {np.round(w_avg, 3)}")

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    preds_list, trues_list, q10_list, q90_list = [], [], [], []
    with torch.no_grad():
        for x_loc, x_grd, y in test_loader:
            x_loc, x_grd = x_loc.to(device), x_grd.to(device)
            p, q10, q90 = model(x_loc, x_grd, return_uncertainty=True)
            preds_list.append(p.cpu().numpy())
            trues_list.append(y.numpy())
            q10_list.append(q10.cpu().numpy())
            q90_list.append(q90.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    q10s = np.concatenate(q10_list, axis=0)
    q90s = np.concatenate(q90_list, axis=0)

    # Save as a separate v4 file so v1, v2, and v3 predictions remain intact!
    np.savez_compressed(
        os.path.join("results", "metrics", "AeroFuse_Net_v4_Router_test_predictions.npz"),
        preds=preds, trues=trues, q10=q10s, q90=q90s
    )

    mae_1h = mean_absolute_error(trues[:, 0], preds[:, 0])
    mae_6h = mean_absolute_error(trues[:, 5], preds[:, 5])
    mae_12h = mean_absolute_error(trues[:, 11], preds[:, 11])
    mae_24h = mean_absolute_error(trues[:, 23], preds[:, 23])
    overall_mae = mean_absolute_error(trues.flatten(), preds.flatten())
    overall_rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    overall_r2 = r2_score(trues.flatten(), preds.flatten())

    picp = np.mean((trues >= q10s) & (trues <= q90s)) * 100.0
    print(f"--> [AeroFuse_Net_v4_Router] 1h MAE: {mae_1h:.4f} | 6h MAE: {mae_6h:.4f} | 24h Avg MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} | 80% Quantile Coverage: {picp:.1f}%")

    metrics = {
        "Model": "AeroFuse_Net_v4_Router",
        "MAE_1h": round(mae_1h, 4),
        "MAE_6h": round(mae_6h, 4),
        "MAE_12h": round(mae_12h, 4),
        "MAE_24h": round(mae_24h, 4),
        "Overall_MAE": round(overall_mae, 4),
        "Overall_RMSE": round(overall_rmse, 4),
        "R2_Score": round(overall_r2, 4),
        "Train_Time_Sec": round(elapsed, 1)
    }
    return model, metrics, preds, trues, q10s, q90s


def run_diebold_mariano_tests_v4(aero_preds, trues):
    print("\n" + "=" * 75)
    print("2. Running Diebold-Mariano (DM) Tests (v4 Router vs. All 8 Previous Models)...")
    print("=" * 75)

    baseline_files = [
        ("iTransformer_2026", "iTransformer_2026_test_predictions.npz"),
        ("1D_CNN_ResNet", "1D_CNN_ResNet_test_predictions.npz"),
        ("PatchTST_2026", "PatchTST_2026_test_predictions.npz"),
        ("AeroFuse_Net_v2_Proposed", "aerofuse_v2_test_predictions.npz"),
        ("AeroFuse_Net_v1_Base", "aerofuse_test_predictions.npz"),
        ("AeroFuse_Net_v3_Final", "AeroFuse_Net_v3_Final_test_predictions.npz"),
        ("BiLSTM_Sequential", "BiLSTM_Sequential_test_predictions.npz"),
        ("2D_Spatial_ResNet", "2D_Spatial_ResNet_test_predictions.npz"),
    ]

    e_aero = np.abs(trues - aero_preds).mean(axis=1)
    dm_rows = []

    for name, fname in baseline_files:
        fpath = os.path.join("results", "metrics", fname)
        if not os.path.exists(fpath):
            continue
        b_preds = np.load(fpath)["preds"]
        e_base = np.abs(trues - b_preds).mean(axis=1)

        d = e_base - e_aero
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        dm_stat = mean_d / np.sqrt(var_d / len(d) + 1e-12)
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

        dm_rows.append({
            "Comparison": f"AeroFuse_Net_v4_Router vs. {name}",
            "MAE_Gain_DegC": f"+{mean_d:.4f}" if mean_d > 0 else f"{mean_d:.4f}",
            "DM_Statistic": round(float(dm_stat), 3),
            "p_value": "< 0.0001" if p_val < 1e-4 else f"{p_val:.5f}",
            "Significant_99pct": "YES (p < 0.01)" if (p_val < 0.01 and mean_d > 0) else "NO"
        })

    dm_df = pd.DataFrame(dm_rows)
    # Saved as a separate _v4 CSV so your earlier DM table is also preserved!
    dm_path = os.path.join("results", "metrics", "statistical_significance_dm_test_v4.csv")
    dm_df.to_csv(dm_path, index=False)
    print(dm_df.to_string(index=False))
    return dm_df


def generate_v4_figures(model, test_loader, aero_preds, trues, q10s, q90s, df_bench, device):
    print("\n" + "=" * 75)
    print("3. Saving v4 Publication Figures to results/figures/v4_router/ ...")
    print("=" * 75)

    # Dedicated subfolder so v3 figures in results/figures/ are never overwritten!
    fig_dir = os.path.join("results", "figures", "v4_router")
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # --- FIGURE 1: Multi-Horizon Forecast Degradation Curve ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    horizons = np.arange(1, HORIZON + 1)

    models_to_plot = [
        ("2D_Spatial_ResNet", "2D_Spatial_ResNet_test_predictions.npz", "#7f8c8d", "--"),
        ("BiLSTM_Sequential", "BiLSTM_Sequential_test_predictions.npz", "#e67e22", "--"),
        ("AeroFuse_Net_v3_Final", "AeroFuse_Net_v3_Final_test_predictions.npz", "#d35400", ":"),
        ("AeroFuse_Net_v2_Proposed", "aerofuse_v2_test_predictions.npz", "#f39c12", "-."),
        ("PatchTST_2026", "PatchTST_2026_test_predictions.npz", "#9b59b6", "-."),
        ("1D_CNN_ResNet", "1D_CNN_ResNet_test_predictions.npz", "#2980b9", "-."),
        ("iTransformer_2026", "iTransformer_2026_test_predictions.npz", "#27ae60", "-"),
    ]

    for label, fname, color, ls in models_to_plot:
        fpath = os.path.join("results", "metrics", fname)
        if os.path.exists(fpath):
            p = np.load(fpath)["preds"]
            h_mae = np.mean(np.abs(trues - p), axis=0)
            ax.plot(horizons, h_mae, label=label, color=color, linestyle=ls, linewidth=1.8, alpha=0.85)

    aero_h_mae = np.mean(np.abs(trues - aero_preds), axis=0)
    ax.plot(horizons, aero_h_mae, label="AeroFuse-Net v4 Router (Proposed)", color="#c0392b", linewidth=3.0, marker="o", markersize=4)

    ax.set_title("Multi-Horizon Forecast Error Degradation Across 24-Hour Lead Times (All 9 Models)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Lead Time Horizon (Hours Ahead)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (MAE in °C)", fontsize=11)
    ax.set_xticks(range(1, 25))
    ax.legend(frameon=True, fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    f1_path = os.path.join(fig_dir, "fig1_horizon_degradation_v4.png")
    fig.savefig(f1_path)
    plt.close(fig)

    # --- FIGURE 2: 7-Day Probabilistic Forecast Trajectory ---
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    start_idx = 1200
    window_len = 168
    t_axis = np.arange(window_len)

    h_idx = 5
    y_true_slice = trues[start_idx : start_idx + window_len, h_idx]
    y_pred_slice = aero_preds[start_idx : start_idx + window_len, h_idx]
    q10_slice = q10s[start_idx : start_idx + window_len, h_idx]
    q90_slice = q90s[start_idx : start_idx + window_len, h_idx]

    ax.fill_between(t_axis, q10_slice, q90_slice, color="#e74c3c", alpha=0.22, label="Calibrated 10th–90th Percentile Uncertainty Band")
    ax.plot(t_axis, y_true_slice, color="#2c3e50", linewidth=2.0, label="Ground Truth Station Temperature (°C)")
    ax.plot(t_axis, y_pred_slice, color="#c0392b", linewidth=1.8, linestyle="--", label="AeroFuse-Net v4 Forecast (t+6h)")

    ax.set_title("AeroFuse-Net v4 Probabilistic Temperature Forecasting & Uncertainty Quantification (168h)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Steps (Hours)", fontsize=11)
    ax.set_ylabel("2m Air Temperature (°C)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    fig.tight_layout()
    f2_path = os.path.join(fig_dir, "fig2_probabilistic_fan_chart_v4.png")
    fig.savefig(f2_path)
    plt.close(fig)

    # --- FIGURE 3: 2D-CNN Spatial Gradient Saliency Map ---
    model.eval()
    x_loc_batch, x_grd_batch, _ = next(iter(test_loader))
    x_loc_batch = x_loc_batch.to(device)
    x_grd_batch = x_grd_batch.to(device).requires_grad_(True)

    out = model(x_loc_batch, x_grd_batch)
    out.mean().backward()

    saliency_grid = x_grd_batch.grad.abs().mean(dim=(0, 1, 2)).detach().cpu().numpy()
    saliency_grid = saliency_grid / (saliency_grid.max() + 1e-8)

    grid_npz = np.load(os.path.join("data", "raw", "spatial_grid_tensor.npz"))
    lats = grid_npz["lats"]
    lons = grid_npz["lons"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(saliency_grid, cmap="YlOrRd", interpolation="bicubic")
    ax.set_xticks(range(len(lons)))
    ax.set_yticks(range(len(lats)))
    ax.set_xticklabels([f"{lon:.2f}°E" for lon in lons], fontsize=9.5)
    ax.set_yticklabels([f"{lat:.2f}°N" for lat in lats], fontsize=9.5)

    for i in range(len(lats)):
        for j in range(len(lons)):
            marker = "[Patiala]\n" if (i == 2 and j == 2) else ""
            ax.text(j, i, f"{marker}{saliency_grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if saliency_grid[i, j] > 0.6 else "black", fontweight="bold", fontsize=9)

    plt.colorbar(im, ax=ax, label="Normalized Spatial Gradient Attribution")
    ax.set_title("2D-CNN Synoptic Spatial Attention Saliency Map (v4 Router)\n(5x5 ERA5 Grid Centered on Patiala, Punjab)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=10.5)
    ax.set_ylabel("Latitude", fontsize=10.5)
    fig.tight_layout()
    f3_path = os.path.join(fig_dir, "fig3_spatial_cnn_gradcam_v4.png")
    fig.savefig(f3_path)
    plt.close(fig)

    # --- FIGURE 4: Complete 9-Model Benchmark & Ablation Bar Chart (v1 vs v2 vs v3 vs v4) ---
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    df_sorted = df_bench.sort_values("Overall_MAE", ascending=False)
    colors = [
        "#c0392b" if "v4_Router" in m else ("#e67e22" if "AeroFuse" in m else "#34495e")
        for m in df_sorted["Model"]
    ]

    bars = ax.barh(df_sorted["Model"], df_sorted["Overall_MAE"], color=colors, height=0.62)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}°C", va="center", fontsize=9.5, fontweight="bold")

    ax.set_xlim(0, df_sorted["Overall_MAE"].max() * 1.15)
    ax.set_title("Complete 9-Model Benchmark & Ablation Study (v1 -> v2 -> v3 -> v4 Router)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Overall 24-Hour Mean Absolute Error (°C — Lower is Better)", fontsize=11)
    fig.tight_layout()
    f4_path = os.path.join(fig_dir, "fig4_ablation_progression_v4.png")
    fig.savefig(f4_path)
    plt.close(fig)

    print("[SUCCESS] Saved all 4 v4 figures to results/figures/v4_router/ (leaving v3 figures untouched)!")
    try:
        os.startfile(os.path.abspath(f1_path))
        os.startfile(os.path.abspath(f4_path))
    except Exception:
        pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_features = prepare_dual_modal_splits()

    model, v4_metrics, aero_preds, trues, q10s, q90s = train_aerofuse_v4(
        train_loader, val_loader, test_loader, num_features, device
    )

    # Append v4 to benchmark_comparison.csv while keeping v1, v2, and v3!
    csv_path = os.path.join("results", "metrics", "benchmark_comparison.csv")
    df_bench = pd.read_csv(csv_path)
    df_bench = df_bench[df_bench["Model"] != "AeroFuse_Net_v4_Router"]
    df_bench = pd.concat([df_bench, pd.DataFrame([v4_metrics])], ignore_index=True)
    df_bench = df_bench.sort_values("Overall_MAE", ascending=True).reset_index(drop=True)
    df_bench.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("COMPLETE 9-MODEL MASTER BENCHMARK & ABLATION TABLE (v1, v2, v3, AND v4 PRESERVED):")
    print("=" * 85)
    print(df_bench.to_string(index=False))
    print("=" * 85)

    run_diebold_mariano_tests_v4(aero_preds, trues)
    generate_v4_figures(model, test_loader, aero_preds, trues, q10s, q90s, df_bench, device)


if __name__ == "__main__":
    main()