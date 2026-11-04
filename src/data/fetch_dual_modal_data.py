import os
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================================
# CONFIGURATION: Dual-Modal Spatiotemporal Grid around Patiala, India
# =====================================================================
CENTER_LAT = 30.3398   # Patiala / TIET Region Latitude
CENTER_LON = 76.3869   # Patiala / TIET Region Longitude
GRID_SIZE = 5          # 5x5 Spatial Grid for 2D-CNN Feature Extraction
GRID_STEP = 0.5        # 0.5 degree spacing (~55 km between grid nodes)

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"  # 7 Full Years of Hourly Data (~61,368 hours)

# 6 Core Synoptic Variables for the 5x5 2D-CNN Spatial Grid
GRID_VARIABLES = [
    "temperature_2m",
    "surface_pressure",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover"
]

# 12 High-Detail Variables for the Target Central Station (1D-CNN / BiLSTM / Transformer)
LOCAL_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation"
]

RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_point_data(lat: float, lon: float, variables: list, retries: int = 3) -> pd.DataFrame:
    """Fetches hourly historical ERA5/station reanalysis data for a single (lat, lon) coordinate."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(variables),
        "timezone": "UTC"
    }
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                return df
            elif response.status_code == 429:
                print("Rate limit hit, sleeping for 15 seconds...")
                time.sleep(15)
            else:
                print(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for ({lat}, {lon}): {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch data for ({lat}, {lon}) after {retries} retries.")


def build_dual_modal_dataset():
    print("=" * 70)
    print("1. Fetching Modal 1: High-Resolution Local Station Time-Series...")
    print("=" * 70)
    local_df = fetch_point_data(CENTER_LAT, CENTER_LON, LOCAL_VARIABLES)
    local_path = os.path.join(RAW_DATA_DIR, "local_station_timeseries.parquet")
    local_df.to_parquet(local_path)
    print(f"[SUCCESS] Saved Local Station Data: {local_df.shape} -> {local_path}\n")

    print("=" * 70)
    print(f"2. Fetching Modal 2: {GRID_SIZE}x{GRID_SIZE} Synoptic Spatial Grid for 2D-CNN...")
    print("=" * 70)
    half = GRID_SIZE // 2
    lats = [CENTER_LAT + (i * GRID_STEP) for i in range(half, -half - 1, -1)]  # North to South
    lons = [CENTER_LON + (j * GRID_STEP) for j in range(-half, half + 1)]      # West to East

    num_timesteps = len(local_df)
    num_channels = len(GRID_VARIABLES)

    # Shape: (Time, Channels, Height=5, Width=5)
    spatial_tensor = np.zeros((num_timesteps, num_channels, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    pbar = tqdm(total=GRID_SIZE * GRID_SIZE, desc="Downloading 5x5 Spatial Grid")
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            grid_df = fetch_point_data(lat, lon, GRID_VARIABLES)
            # Align index with local_df just in case
            grid_df = grid_df.reindex(local_df.index).interpolate(method="linear")
            for c, var in enumerate(GRID_VARIABLES):
                spatial_tensor[:, c, i, j] = grid_df[var].values.astype(np.float32)
            pbar.update(1)
            time.sleep(1.2)  # Polite delay to respect free API rate limits
    pbar.close()

    grid_path = os.path.join(RAW_DATA_DIR, "spatial_grid_tensor.npz")
    np.savez_compressed(
        grid_path,
        spatial_tensor=spatial_tensor,
        timestamps=local_df.index.astype(str).values,
        channels=np.array(GRID_VARIABLES),
        lats=np.array(lats),
        lons=np.array(lons)
    )
    print(f"\n[SUCCESS] Saved 4D Spatial Grid Tensor: {spatial_tensor.shape} -> {grid_path}")
    print("Tensor Format: (TimeSteps, Channels=6, Height=5, Width=5)")


if __name__ == "__main__":
    build_dual_modal_dataset()