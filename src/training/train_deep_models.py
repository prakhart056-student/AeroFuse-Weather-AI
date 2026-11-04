import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root directory is in path so src.models imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.architectures_v1 import (
    TemporalCNN1D,
    SpatialResNet2D,
    BiLSTMForecaster,
    iTransformer2026,
    AeroFuseNet,
    PhysicsInformedWeatherLoss
)

# =====================================================================
# HYPERPARAMETERS (Optimized for RTX 4050 6GB + 16GB RAM)
# =====================================================================
SEQ_LEN = 24          # 24 hours of historical lookback
HORIZON = 24          # Predict next 24 hours simultaneously (t+1 ... t+24)
BATCH_SIZE = 256      # Fast GPU batching
EPOCHS = 12           # Converges rapidly on 42,000+ hourly windows
LR = 1e-3
TARGET_COL = "temperature_2m"


class DualModalWeatherDataset(Dataset):
    """Memory-efficient zero-copy sliding window dataset for Dual-Modal tensors."""
    def __init__(self, local_data: np.ndarray, grid_data: np.ndarray, targets: np.ndarray, seq_len: int, horizon: int):
        self.local_data = torch.tensor(local_data, dtype=torch.float32)
        self.grid_data = torch.tensor(grid_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.length = len(local_data) - seq_len - horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_loc = self.local_data[idx : idx + self.seq_len]
        x_grd = self.grid_data[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x_loc, x_grd, y


def prepare_dual_modal_splits():
    local_path = os.path.join("data", "processed", "processed_local_station.parquet")
    grid_path = os.path.join("data", "raw", "spatial_grid_tensor.npz")

    df = pd.read_parquet(local_path)
    grid_npz = np.load(grid_path)
    spatial_tensor = grid_npz["spatial_tensor"]  # (Time, 6, 5, 5)

    raw_targets = df[TARGET_COL].values.astype(np.float32)
    local_values = df.values.astype(np.float32)

    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    # Fit StandardScaler strictly on Training Split (Zero Data Leakage!)
    local_scaler = StandardScaler()
    local_train = local_scaler.fit_transform(local_values[:train_end])
    local_val = local_scaler.transform(local_values[train_end:val_end])
    local_test = local_scaler.transform(local_values[val_end:])

    # Normalize Spatial Grid Channels using training statistics
    grid_mean = spatial_tensor[:train_end].mean(axis=(0, 2, 3), keepdims=True)
    grid_std = spatial_tensor[:train_end].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    grid_norm = (spatial_tensor - grid_mean) / grid_std

    # Save scalers for deployment dashboard
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    joblib.dump({
        "local_scaler": local_scaler,
        "grid_mean": grid_mean,
        "grid_std": grid_std,
        "feature_names": list(df.columns)
    }, os.path.join("data", "processed", "scalers.pkl"))

    train_ds = DualModalWeatherDataset(local_train, grid_norm[:train_end], raw_targets[:train_end], SEQ_LEN, HORIZON)
    val_ds = DualModalWeatherDataset(local_val, grid_norm[train_end:val_end], raw_targets[train_end:val_end], SEQ_LEN, HORIZON)
    test_ds = DualModalWeatherDataset(local_test, grid_norm[val_end:], raw_targets[val_end:], SEQ_LEN, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, local_values.shape[1]


def train_and_evaluate_model(model, model_name, train_loader, val_loader, test_loader, device, is_aerofuse=False):
    print(f"\n" + "-" * 65)
    print(f"Training {model_name} on {device}...")
    print("-" * 65)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = PhysicsInformedWeatherLoss() if is_aerofuse else nn.HuberLoss()

    best_val_loss = float("inf")
    best_weights_path = os.path.join("data", "processed", f"{model_name}.pt")

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_loc, x_grd, y in train_loader:
            x_loc, x_grd, y = x_loc.to(device), x_grd.to(device), y.to(device)
            optimizer.zero_grad()

            if is_aerofuse:
                pred, q10, q90 = model(x_loc, x_grd, return_uncertainty=True)
                loss = criterion(pred, y, q10, q90)
            else:
                pred = model(x_loc, x_grd)
                loss = criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_loc, x_grd, y in val_loader:
                x_loc, x_grd, y = x_loc.to(device), x_grd.to(device), y.to(device)
                pred = model(x_loc, x_grd)
                val_loss += F.mse_loss(pred, y).item() * len(y)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 3 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    elapsed = time.time() - start_time

    # Load Best Checkpoint & Evaluate on Unseen Test Split
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    model.eval()

    preds_list, trues_list = [], []
    q10_list, q90_list = [], []
    with torch.no_grad():
        for x_loc, x_grd, y in test_loader:
            x_loc, x_grd = x_loc.to(device), x_grd.to(device)
            if is_aerofuse:
                p, q10, q90 = model(x_loc, x_grd, return_uncertainty=True)
                q10_list.append(q10.cpu().numpy())
                q90_list.append(q90.cpu().numpy())
            else:
                p = model(x_loc, x_grd)
            preds_list.append(p.cpu().numpy())
            trues_list.append(y.numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # Save test predictions for statistical significance & plots
    if is_aerofuse:
        np.savez_compressed(
            os.path.join("results", "metrics", "aerofuse_test_predictions.npz"),
            preds=preds,
            trues=trues,
            q10=np.concatenate(q10_list, axis=0),
            q90=np.concatenate(q90_list, axis=0)
        )
    else:
        np.savez_compressed(
            os.path.join("results", "metrics", f"{model_name}_test_predictions.npz"),
            preds=preds,
            trues=trues
        )

    # Compute Multi-Horizon Metrics (t+1h, t+6h, t+12h, t+24h, and Overall 24h Average)
    mae_1h = mean_absolute_error(trues[:, 0], preds[:, 0])
    mae_6h = mean_absolute_error(trues[:, 5], preds[:, 5])
    mae_12h = mean_absolute_error(trues[:, 11], preds[:, 11])
    mae_24h = mean_absolute_error(trues[:, 23], preds[:, 23])

    overall_mae = mean_absolute_error(trues.flatten(), preds.flatten())
    overall_rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    overall_r2 = r2_score(trues.flatten(), preds.flatten())

    print(f"--> [{model_name}] 1h MAE: {mae_1h:.4f} | 6h MAE: {mae_6h:.4f} | 24h Avg MAE: {overall_mae:.4f} | 24h RMSE: {overall_rmse:.4f} | R2: {overall_r2:.4f} ({elapsed:.1f}s)")

    return {
        "Model": model_name,
        "MAE_1h": round(mae_1h, 4),
        "MAE_6h": round(mae_6h, 4),
        "MAE_12h": round(mae_12h, 4),
        "MAE_24h": round(mae_24h, 4),
        "Overall_MAE": round(overall_mae, 4),
        "Overall_RMSE": round(overall_rmse, 4),
        "R2_Score": round(overall_r2, 4),
        "Train_Time_Sec": round(elapsed, 1)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Launching Deep Learning & 2026 SOTA Benchmark | Device: {device}")
    if device.type == "cuda":
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    train_loader, val_loader, test_loader, num_local_features = prepare_dual_modal_splits()

    models_to_run = [
        ("1D_CNN_ResNet", TemporalCNN1D(num_local_features, SEQ_LEN, HORIZON), False),
        ("2D_Spatial_ResNet", SpatialResNet2D(6, SEQ_LEN, HORIZON), False),
        ("BiLSTM_Sequential", BiLSTMForecaster(num_local_features, 64, HORIZON), False),
        ("iTransformer_2026", iTransformer2026(num_local_features, SEQ_LEN, HORIZON), False),
        ("AeroFuse_Net_Proposed", AeroFuseNet(num_local_features, 6, SEQ_LEN, HORIZON), True),
    ]

    all_metrics = []
    for name, model_inst, is_aero in models_to_run:
        metrics = train_and_evaluate_model(
            model_inst, name, train_loader, val_loader, test_loader, device, is_aerofuse=is_aero
        )
        all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    csv_path = os.path.join("results", "metrics", "benchmark_comparison.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL MULTI-HORIZON BENCHMARK TABLE (SAVED TO CSV FOR PAPER):")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()