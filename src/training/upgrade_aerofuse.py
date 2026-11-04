import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import directly from your new architectures_v2.py file!
from src.models.architectures_v2 import (
    PatchTST2026,
    AeroFuseNet,
    PhysicsInformedWeatherLoss
)

SEQ_LEN = 24
HORIZON = 24
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
TARGET_COL = "temperature_2m"


class DualModalWeatherDataset(Dataset):
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
    spatial_tensor = grid_npz["spatial_tensor"]

    raw_targets = df[TARGET_COL].values.astype(np.float32)
    local_values = df.values.astype(np.float32)

    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    local_scaler = StandardScaler()
    local_train = local_scaler.fit_transform(local_values[:train_end])
    local_val = local_scaler.transform(local_values[train_end:val_end])
    local_test = local_scaler.transform(local_values[val_end:])

    grid_mean = spatial_tensor[:train_end].mean(axis=(0, 2, 3), keepdims=True)
    grid_std = spatial_tensor[:train_end].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    grid_norm = (spatial_tensor - grid_mean) / grid_std

    train_ds = DualModalWeatherDataset(local_train, grid_norm[:train_end], raw_targets[:train_end], SEQ_LEN, HORIZON)
    val_ds = DualModalWeatherDataset(local_val, grid_norm[train_end:val_end], raw_targets[train_end:val_end], SEQ_LEN, HORIZON)
    test_ds = DualModalWeatherDataset(local_test, grid_norm[val_end:], raw_targets[val_end:], SEQ_LEN, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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

    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    model.eval()

    preds_list, trues_list, q10_list, q90_list = [], [], [], []
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

    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    if is_aerofuse:
        np.savez_compressed(
            os.path.join("results", "metrics", "aerofuse_v2_test_predictions.npz"),
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


def run_upgrade():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Running PatchTST_2026 & Upgraded AeroFuse-Net v2 on {device}...")
    print("=" * 70)

    train_loader, val_loader, test_loader, num_local_features = prepare_dual_modal_splits()

    patch_metrics = train_and_evaluate_model(
        PatchTST2026(num_local_features, SEQ_LEN, HORIZON),
        "PatchTST_2026",
        train_loader, val_loader, test_loader, device, is_aerofuse=False
    )

    aero_v2_metrics = train_and_evaluate_model(
        AeroFuseNet(num_local_features, 6, SEQ_LEN, HORIZON),
        "AeroFuse_Net_v2_Proposed",
        train_loader, val_loader, test_loader, device, is_aerofuse=True
    )

    csv_path = os.path.join("results", "metrics", "benchmark_comparison.csv")
    df = pd.read_csv(csv_path)

    # Rename v1 in table so you can compare v1 vs v2 directly in your paper's Ablation Study!
    df["Model"] = df["Model"].replace({"AeroFuse_Net_Proposed": "AeroFuse_Net_v1_Base"})
    df = df[~df["Model"].isin(["PatchTST_2026", "AeroFuse_Net_v2_Proposed"])]
    df = pd.concat([df, pd.DataFrame([patch_metrics, aero_v2_metrics])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("UPDATED 7-MODEL BENCHMARK & ABLATION TABLE (SAVED FOR PAPER):")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_upgrade()