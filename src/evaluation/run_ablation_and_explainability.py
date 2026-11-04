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
matplotlib.use("Agg")  # Non-interactive backend so plots save cleanly without popup windows
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.upgrade_aerofuse import (
    SEQ_LEN,
    HORIZON,
    prepare_dual_modal_splits
)
from src.models.architectures_v2 import PhysicsInformedWeatherLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# AEROFUSE-NET v3 (FINAL PROPOSED ARCHITECTURE):
# Dual-Stream (Inverted Variate + 1D-CNN) + Physics Advection 2D-CNN
# with Zero-Initialized Cross-Modal Gating
# =====================================================================
class AeroFuseNetV3Final(nn.Module):
    def __init__(
        self,
        local_features: int,
        grid_channels: int = 6,
        seq_len: int = 24,
        horizon: int = 24,
        d_model: int = 64
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model

        # --- Stream 1: Inverted Multivariate Physics Attention (Variate Tokens) ---
        self.variate_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=0.15, activation="gelu", batch_first=True
        )
        self.variate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.variate_head = nn.Linear(d_model * local_features, horizon)

        # --- Stream 2: Multi-Scale Residual 1D-CNN (Local Temporal Slopes) ---
        self.conv1 = nn.Conv1d(local_features, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.cnn_drop = nn.Dropout(0.15)
        self.temporal_head = nn.Linear(d_model * seq_len, horizon)

        # --- Stream 3: Synoptic 2D-CNN Spatial Advection Encoder (Raw + Center Gradient) ---
        # Input channels = grid_channels * 2 (Raw field + Horizontal difference from Patiala center cell)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(grid_channels * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Cross-Modal Attention: Local Variate Tokens (Q) query Synoptic 2D Grid Trajectory (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.15, batch_first=True
        )
        self.spatial_head = nn.Linear(d_model * local_features, horizon)

        # Learnable Mixing Weights & Zero-Initialized Cross-Modal Gate
        self.mix_logit = nn.Parameter(torch.tensor(0.35))   # Balances Variate vs 1D-CNN streams
        self.spatial_gate = nn.Parameter(torch.zeros(1))    # Zero-init prevents early spatial noise

        # Monotonic Non-Crossing Quantile Uncertainty Heads (10th & 90th Percentiles)
        self.lower_offset = nn.Sequential(
            nn.Linear(d_model * local_features, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )
        self.upper_offset = nn.Sequential(
            nn.Linear(d_model * local_features, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x_local, x_grid, return_uncertainty: bool = False):
        B, T, C_g, H, W = x_grid.shape

        # 1. Inverted Variate Stream
        x_inv = x_local.transpose(1, 2)                    # (B, N_vars, T)
        var_tokens = self.variate_embedding(x_inv)         # (B, N_vars, d_model)
        var_encoded = self.variate_encoder(var_tokens)     # (B, N_vars, d_model)
        var_flat = var_encoded.flatten(start_dim=1)
        out_variate = self.variate_head(var_flat)

        # 2. Residual 1D-CNN Temporal Stream
        h_cnn = F.gelu(self.bn1(self.conv1(x_inv)))
        h_cnn = self.cnn_drop(F.gelu(self.bn2(self.conv2(h_cnn))) + h_cnn)
        out_temporal = self.temporal_head(h_cnn.flatten(start_dim=1))

        # 3. Physics-Guided 2D-CNN Spatial Advection Stream
        center_cell = x_grid[:, :, :, H // 2 : H // 2 + 1, W // 2 : W // 2 + 1]
        spatial_advection = x_grid - center_cell           # Horizontal gradient relative to target station
        grid_augmented = torch.cat([x_grid, spatial_advection], dim=2)  # (B, T, 12, 5, 5)

        sp_feat = self.spatial_conv(grid_augmented.view(B * T, C_g * 2, H, W))
        sp_seq = self.spatial_pool(sp_feat).view(B, T, self.d_model)    # (B, T, d_model)

        # Local variate representations attend to the 2D-CNN synoptic sequence
        cross_out, _ = self.cross_attn(query=var_encoded, key=sp_seq, value=sp_seq)
        out_spatial = self.spatial_head(cross_out.flatten(start_dim=1))

        # 4. Adaptive Gated Multi-Modal Fusion
        w_temp = torch.sigmoid(self.mix_logit)
        pred_mean = (1.0 - w_temp) * out_variate + w_temp * out_temporal + torch.tanh(self.spatial_gate) * out_spatial

        if return_uncertainty:
            feat_detached = var_flat.detach()
            base_detached = pred_mean.detach()
            q10 = base_detached - F.softplus(self.lower_offset(feat_detached))
            q90 = base_detached + F.softplus(self.upper_offset(feat_detached))
            return pred_mean, q10, q90

        return pred_mean


def train_aerofuse_v3(train_loader, val_loader, test_loader, num_features, device):
    print("=" * 75)
    print(f"1. Training AeroFuse_Net_v3_Final (Zero-Gated Advection Fusion) on {device}...")
    print("=" * 75)

    epochs = 15
    model = AeroFuseNetV3Final(num_features, 6, SEQ_LEN, HORIZON).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PhysicsInformedWeatherLoss(lambda_phys=0.08, lambda_quant=0.15)

    best_val_mse = float("inf")
    best_path = os.path.join("data", "processed", "AeroFuse_Net_v3_Final.pt")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
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

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Spatial Gate: {torch.tanh(model.spatial_gate).item():.4f}")

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

    np.savez_compressed(
        os.path.join("results", "metrics", "AeroFuse_Net_v3_Final_test_predictions.npz"),
        preds=preds, trues=trues, q10=q10s, q90=q90s
    )

    mae_1h = mean_absolute_error(trues[:, 0], preds[:, 0])
    mae_6h = mean_absolute_error(trues[:, 5], preds[:, 5])
    mae_12h = mean_absolute_error(trues[:, 11], preds[:, 11])
    mae_24h = mean_absolute_error(trues[:, 23], preds[:, 23])
    overall_mae = mean_absolute_error(trues.flatten(), preds.flatten())
    overall_rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    overall_r2 = r2_score(trues.flatten(), preds.flatten())

    # Calculate 80% Confidence Interval Coverage Probability (PICP)
    picp = np.mean((trues >= q10s) & (trues <= q90s)) * 100.0
    print(f"--> [AeroFuse_Net_v3_Final] 1h MAE: {mae_1h:.4f} | 6h MAE: {mae_6h:.4f} | 24h Avg MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} | Quantile Coverage: {picp:.1f}%")

    metrics = {
        "Model": "AeroFuse_Net_v3_Final",
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


def run_diebold_mariano_tests(aero_preds, trues):
    """Computes formal Diebold-Mariano statistical significance tests against all baselines."""
    print("\n" + "=" * 75)
    print("2. Running Diebold-Mariano (DM) Statistical Significance Tests...")
    print("=" * 75)

    baseline_files = [
        ("1D_CNN_ResNet", "1D_CNN_ResNet_test_predictions.npz"),
        ("2D_Spatial_ResNet", "2D_Spatial_ResNet_test_predictions.npz"),
        ("BiLSTM_Sequential", "BiLSTM_Sequential_test_predictions.npz"),
        ("PatchTST_2026", "PatchTST_2026_test_predictions.npz"),
        ("iTransformer_2026", "iTransformer_2026_test_predictions.npz"),
        ("AeroFuse_Net_v1_Base", "aerofuse_test_predictions.npz"),
        ("AeroFuse_Net_v2_Proposed", "aerofuse_v2_test_predictions.npz"),
    ]

    e_aero = np.abs(trues - aero_preds).mean(axis=1)  # Mean absolute error per forecast window
    dm_rows = []

    for name, fname in baseline_files:
        fpath = os.path.join("results", "metrics", fname)
        if not os.path.exists(fpath):
            continue
        b_preds = np.load(fpath)["preds"]
        e_base = np.abs(trues - b_preds).mean(axis=1)

        # Loss differential d_t = e_baseline - e_aerofuse (positive means AeroFuse is more accurate)
        d = e_base - e_aero
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        dm_stat = mean_d / np.sqrt(var_d / len(d) + 1e-12)
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

        dm_rows.append({
            "Comparison": f"AeroFuse_Net_v3 vs. {name}",
            "Mean_MAE_Reduction_DegC": round(float(mean_d), 4),
            "DM_Statistic": round(float(dm_stat), 3),
            "p_value": "< 0.0001" if p_val < 1e-4 else f"{p_val:.5f}",
            "Statistically_Significant_99pct": "YES (p < 0.01)" if p_val < 0.01 else "NO"
        })

    dm_df = pd.DataFrame(dm_rows)
    dm_path = os.path.join("results", "metrics", "statistical_significance_dm_test.csv")
    dm_df.to_csv(dm_path, index=False)
    print(dm_df.to_string(index=False))
    return dm_df


def generate_publication_figures(model, test_loader, aero_preds, trues, q10s, q90s, df_bench, device):
    print("\n" + "=" * 75)
    print("3. Generating 300-DPI Publication Figures in results/figures/ ...")
    print("=" * 75)

    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # --- FIGURE 1: Multi-Horizon Forecast Degradation Curve (t+1h to t+24h) ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    horizons = np.arange(1, HORIZON + 1)

    models_to_plot = [
        ("2D_Spatial_ResNet", "2D_Spatial_ResNet_test_predictions.npz", "#7f8c8d", "--"),
        ("BiLSTM_Sequential", "BiLSTM_Sequential_test_predictions.npz", "#e67e22", "--"),
        ("PatchTST_2026", "PatchTST_2026_test_predictions.npz", "#9b59b6", "-."),
        ("1D_CNN_ResNet", "1D_CNN_ResNet_test_predictions.npz", "#2980b9", "-."),
        ("iTransformer_2026", "iTransformer_2026_test_predictions.npz", "#27ae60", "-"),
    ]

    for label, fname, color, ls in models_to_plot:
        fpath = os.path.join("results", "metrics", fname)
        if os.path.exists(fpath):
            p = np.load(fpath)["preds"]
            h_mae = np.mean(np.abs(trues - p), axis=0)
            ax.plot(horizons, h_mae, label=label, color=color, linestyle=ls, linewidth=2.0, alpha=0.85)

    aero_h_mae = np.mean(np.abs(trues - aero_preds), axis=0)
    ax.plot(horizons, aero_h_mae, label="AeroFuse-Net v3 (Proposed)", color="#c0392b", linewidth=3.0, marker="o", markersize=4)

    ax.set_title("Multi-Horizon Forecast Error Degradation Across 24-Hour Lead Times (Patiala Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Lead Time Horizon (Hours Ahead)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (MAE in °C)", fontsize=11)
    ax.set_xticks(range(1, 25))
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_horizon_degradation.png"))
    plt.close(fig)

    # --- FIGURE 2: 7-Day (168h) Probabilistic Forecast Trajectory with Uncertainty Band ---
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    start_idx = 1200  # Representative 7-day seasonal transition window
    window_len = 168
    t_axis = np.arange(window_len)

    # Plot 6-hour ahead rolling forecast + 10th-90th percentile uncertainty fan
    h_idx = 5  # 6-hour ahead horizon
    y_true_slice = trues[start_idx : start_idx + window_len, h_idx]
    y_pred_slice = aero_preds[start_idx : start_idx + window_len, h_idx]
    q10_slice = q10s[start_idx : start_idx + window_len, h_idx]
    q90_slice = q90s[start_idx : start_idx + window_len, h_idx]

    ax.fill_between(t_axis, q10_slice, q90_slice, color="#e74c3c", alpha=0.22, label="Calibrated 10th–90th Percentile Uncertainty Band")
    ax.plot(t_axis, y_true_slice, color="#2c3e50", linewidth=2.0, label="Ground Truth Station Temperature (°C)")
    ax.plot(t_axis, y_pred_slice, color="#c0392b", linewidth=1.8, linestyle="--", label="AeroFuse-Net v3 Forecast (t+6h)")

    ax.set_title("AeroFuse-Net Probabilistic Temperature Forecasting & Uncertainty Quantification (168-Hour Window)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Steps (Hours)", fontsize=11)
    ax.set_ylabel("2m Air Temperature (°C)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_probabilistic_fan_chart.png"))
    plt.close(fig)

    # --- FIGURE 3: 2D-CNN Spatial Gradient Saliency Map Over the 5x5 Patiala Region Grid ---
    model.eval()
    x_loc_batch, x_grd_batch, _ = next(iter(test_loader))
    x_loc_batch = x_loc_batch.to(device)
    x_grd_batch = x_grd_batch.to(device).requires_grad_(True)

    out = model(x_loc_batch, x_grd_batch)
    out.mean().backward()

    # Aggregate absolute gradient sensitivity across Batch, Time, and Channels -> (5, 5)
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
            marker = "★ Patiala\n" if (i == 2 and j == 2) else ""
            ax.text(j, i, f"{marker}{saliency_grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if saliency_grid[i, j] > 0.6 else "black", fontweight="bold", fontsize=9)

    plt.colorbar(im, ax=ax, label="Normalized Spatial Gradient Attribution")
    ax.set_title("2D-CNN Synoptic Spatial Attention Saliency Map\n(5×5 ERA5 Grid Centered on Patiala, Punjab)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=10.5)
    ax.set_ylabel("Latitude", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_spatial_cnn_gradcam.png"))
    plt.close(fig)

    # --- FIGURE 4: Architectural Benchmark & Ablation Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df_sorted = df_bench.sort_values("Overall_MAE", ascending=False)
    colors = ["#c0392b" if "v3_Final" in m else ("#e67e22" if "AeroFuse" in m else "#34495e") for m in df_sorted["Model"]]

    bars = ax.barh(df_sorted["Model"], df_sorted["Overall_MAE"], color=colors, height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}°C", va="center", fontsize=9.5, fontweight="bold")

    ax.set_xlim(0, df_sorted["Overall_MAE"].max() * 1.15)
    ax.set_title("Comprehensive Benchmark & Ablation Study (24-Hour Overall MAE — Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Overall 24-Hour Mean Absolute Error (°C)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_ablation_progression.png"))
    plt.close(fig)

    print("[SUCCESS] Generated 4 high-resolution figures in results/figures/:")
    print("  1. fig1_horizon_degradation.png")
    print("  2. fig2_probabilistic_fan_chart.png")
    print("  3. fig3_spatial_cnn_gradcam.png")
    print("  4. fig4_ablation_progression.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_features = prepare_dual_modal_splits()

    # 1. Train Final Zero-Gated AeroFuse-Net v3
    model, v3_metrics, aero_preds, trues, q10s, q90s = train_aerofuse_v3(
        train_loader, val_loader, test_loader, num_features, device
    )

    # 2. Update Master Benchmark CSV
    csv_path = os.path.join("results", "metrics", "benchmark_comparison.csv")
    df_bench = pd.read_csv(csv_path)
    df_bench = df_bench[df_bench["Model"] != "AeroFuse_Net_v3_Final"]
    df_bench = pd.concat([df_bench, pd.DataFrame([v3_metrics])], ignore_index=True)
    df_bench = df_bench.sort_values("Overall_MAE", ascending=True).reset_index(drop=True)
    df_bench.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("FINAL 8-MODEL MASTER BENCHMARK & ABLATION TABLE (SORTED BY OVERALL MAE):")
    print("=" * 85)
    print(df_bench.to_string(index=False))
    print("=" * 85)

    # 3. Run Diebold-Mariano Statistical Significance Tests
    run_diebold_mariano_tests(aero_preds, trues)

    # 4. Generate 300-DPI Publication Figures
    generate_publication_figures(model, test_loader, aero_preds, trues, q10s, q90s, df_bench, device)


if __name__ == "__main__":
    main()