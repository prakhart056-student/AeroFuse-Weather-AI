import os
import numpy as np
import pandas as pd

def process_weather_data():
    print("=" * 70)
    print("Starting Physics-Informed Preprocessing & Feature Engineering...")
    print("=" * 70)

    # 1. Load Raw Local Station Data
    local_path = os.path.join("data", "raw", "local_station_timeseries.parquet")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Run fetch_dual_modal_data.py first! Missing: {local_path}")
    
    df = pd.read_parquet(local_path)
    print(f"Loaded Local Station Data Shape: {df.shape}")

    # 2. Impute missing values (forward fill then backward fill for meteorological continuity)
    df = df.ffill().bfill()

    # 3. Cyclical Time Encoding (This step will be Crucial for our Neural Networks & Transformers)
    timestamps = df.index
    df["hour_sin"] = np.sin(2 * np.pi * timestamps.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * timestamps.hour / 24.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * timestamps.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * timestamps.dayofyear / 365.25)

    # 4. Physics-Informed Feature: Vapor Pressure Deficit & Magnus-Tetens Check
    # T = temperature_2m, RH = relative_humidity_2m
    T = df["temperature_2m"]
    RH = df["relative_humidity_2m"]
    # Saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    # Actual vapor pressure
    ea = es * (RH / 100.0)
    df["vapor_pressure_deficit"] = es - ea

    # 5. Rolling Statistical Features (12-hour and 24-hour trends)
    df["temp_rolling_mean_24h"] = T.rolling(window=24, min_periods=1).mean()
    df["pressure_rolling_std_12h"] = df["surface_pressure"].rolling(window=12, min_periods=1).std().fillna(0)

    # Save processed tabular data
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_local_path = os.path.join(processed_dir, "processed_local_station.parquet")
    df.to_parquet(processed_local_path)
    print(f"[SUCCESS] Processed Tabular Data Saved: {df.shape} -> {processed_local_path}")

    # 6. Verify Spatial Grid Tensor Exists
    grid_path = os.path.join("data", "raw", "spatial_grid_tensor.npz")
    if os.path.exists(grid_path):
        data = np.load(grid_path)
        spatial_tensor = data["spatial_tensor"]
        print(f"[SUCCESS] Spatial Grid Tensor verified: {spatial_tensor.shape}")
    else:
        print("[WARNING] Spatial grid tensor not found. Ensure fetch script completed fully.")

    print("=" * 70)
    print("Preprocessing & Feature Engineering Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    process_weather_data()