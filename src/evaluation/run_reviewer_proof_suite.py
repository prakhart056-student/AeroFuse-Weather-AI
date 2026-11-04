import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQ_LEN = 24
HORIZON = 24


def run_reviewer_suite():
    print("=" * 80)
    print("RUNNING REVIEWER-PROOF EVALUATION SUITE (Persistence, Seasons, Calibration)")
    print("=" * 80)

    # 1. Load Processed Dataset & Reconstruct Exact Test Timestamps
    df = pd.read_parquet(os.path.join("data", "processed", "processed_local_station.parquet"))
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    raw_temps = df["temperature_2m"].values.astype(np.float32)
    test_temps = raw_temps[val_end:]
    test_times = df.index[val_end:]

    # Load saved predictions
    v4_npz = np.load(os.path.join("results", "metrics", "AeroFuse_Net_v4_Router_test_predictions.npz"))
    aero_preds = v4_npz["preds"]
    trues = v4_npz["trues"]
    q10_raw = v4_npz["q10"]
    q90_raw = v4_npz["q90"]

    itrans_preds = np.load(os.path.join("results", "metrics", "iTransformer_2026_test_predictions.npz"))["preds"]
    cnn1d_preds = np.load(os.path.join("results", "metrics", "1D_CNN_ResNet_test_predictions.npz"))["preds"]
    spat2d_preds = np.load(os.path.join("results", "metrics", "2D_Spatial_ResNet_test_predictions.npz"))["preds"]

    num_windows = len(trues)

    # =====================================================================
    # EXPERIMENT A: Persistence & Diurnal Climatology Reference Baselines
    # =====================================================================
    # 1. Naive Persistence: Forecast for all next 24h = last observed temperature at t
    persistence_preds = np.zeros_like(trues)
    # 2. Diurnal 24h Lag Persistence (Seasonal/Daily cycle baseline): y_{t+h} = y_{t+h-24}
    diurnal_preds = np.zeros_like(trues)

    for i in range(num_windows):
        persistence_preds[i, :] = test_temps[i + SEQ_LEN - 1]
        diurnal_preds[i, :] = test_temps[i : i + SEQ_LEN]

    ref_rows = []
    for name, p in [
        ("Naive_Persistence_Baseline", persistence_preds),
        ("Diurnal_24h_Cycle_Baseline", diurnal_preds),
        ("AeroFuse_Net_v4_Router", aero_preds),
    ]:
        ref_rows.append({
            "Model": name,
            "MAE_1h": round(mean_absolute_error(trues[:, 0], p[:, 0]), 4),
            "MAE_6h": round(mean_absolute_error(trues[:, 5], p[:, 5]), 4),
            "MAE_12h": round(mean_absolute_error(trues[:, 11], p[:, 11]), 4),
            "MAE_24h": round(mean_absolute_error(trues[:, 23], p[:, 23]), 4),
            "Overall_MAE": round(mean_absolute_error(trues.flatten(), p.flatten()), 4),
            "Overall_RMSE": round(np.sqrt(mean_squared_error(trues.flatten(), p.flatten())), 4),
        })

    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(os.path.join("results", "metrics", "persistence_and_climatology_baselines.csv"), index=False)
    print("\n[1] Meteorological Reference Baselines vs. AeroFuse-Net v4:")
    print(ref_df.to_string(index=False))

    # =====================================================================
    # EXPERIMENT B: Patiala Seasonal, Diurnal & Extreme Weather Regimes
    # =====================================================================
    # Timestamp of the forecast origin for each window
    origin_times = test_times[SEQ_LEN - 1 : SEQ_LEN - 1 + num_windows]
    months = origin_times.month
    hours = origin_times.hour

    # Define meteorological regimes for Patiala (Indo-Gangetic Plains)
    regimes = {
        "Winter_Fog_Regime (Dec-Feb)": np.isin(months, [12, 1, 2]),
        "Summer_Heatwave_Regime (Apr-Jun)": np.isin(months, [4, 5, 6]),
        "Monsoon_Convective_Regime (Jul-Sep)": np.isin(months, [7, 8, 9]),
        "Post_Monsoon_Transition (Oct-Nov)": np.isin(months, [10, 11]),
        "Daytime_Solar_Hours (08:00-18:00 UTC)": (hours >= 8) & (hours <= 18),
        "Nighttime_Radiative_Cooling": (hours < 8) | (hours > 18),
        "Extreme_Heat_Events (>38°C)": (trues.max(axis=1) > 38.0),
        "Rapid_Front_Transitions (|dT/6h| > 6°C)": (np.abs(trues[:, 5] - test_temps[SEQ_LEN - 1 : SEQ_LEN - 1 + num_windows]) > 6.0),
    }

    regime_rows = []
    for reg_name, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        t_sub = trues[mask]
        a_sub = aero_preds[mask]
        i_sub = itrans_preds[mask]
        c_sub = cnn1d_preds[mask]

        aero_rmse = np.sqrt(mean_squared_error(t_sub.flatten(), a_sub.flatten()))
        itrans_rmse = np.sqrt(mean_squared_error(t_sub.flatten(), i_sub.flatten()))
        cnn_rmse = np.sqrt(mean_squared_error(t_sub.flatten(), c_sub.flatten()))

        regime_rows.append({
            "Weather_Regime_Patiala": reg_name,
            "Sample_Windows": int(np.sum(mask)),
            "AeroFuse_v4_RMSE": round(float(aero_rmse), 4),
            "iTransformer_RMSE": round(float(itrans_rmse), 4),
            "1D_CNN_RMSE": round(float(cnn_rmse), 4),
            "Best_Model_in_Regime": "AeroFuse_Net_v4" if aero_rmse <= min(itrans_rmse, cnn_rmse) else "iTransformer_2026"
        })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(os.path.join("results", "metrics", "seasonal_and_extreme_regime_evaluation.csv"), index=False)
    print("\n[2] Seasonal, Diurnal & Extreme Weather Regime Breakdown (RMSE in °C):")
    print(regime_df.to_string(index=False))

    # =====================================================================
    # EXPERIMENT C: Conformal Uncertainty Calibration & Probabilistic Metrics
    # =====================================================================
    def pinball(y, q_pred, tau):
        err = y - q_pred
        return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))

    # Split test into calibration half and evaluation half to demonstrate Conformal Calibration
    cal_split = num_windows // 2
    nonconformity_low = q10_raw[:cal_split] - trues[:cal_split]
    nonconformity_high = trues[:cal_split] - q90_raw[:cal_split]

    # Compute 80% conformal adjustment offset
    delta_low = np.quantile(nonconformity_low, 0.82, axis=0, keepdims=True)
    delta_high = np.quantile(nonconformity_high, 0.82, axis=0, keepdims=True)

    q10_cal = q10_raw[cal_split:] - delta_low
    q90_cal = q90_raw[cal_split:] + delta_high
    trues_eval = trues[cal_split:]

    raw_picp = float(np.mean((trues_eval >= q10_raw[cal_split:]) & (trues_eval <= q90_raw[cal_split:])) * 100.0)
    raw_mpiw = float(np.mean(q90_raw[cal_split:] - q10_raw[cal_split:]))
    raw_pinball = (pinball(trues_eval, q10_raw[cal_split:], 0.10) + pinball(trues_eval, q90_raw[cal_split:], 0.90)) / 2.0

    cal_picp = float(np.mean((trues_eval >= q10_cal) & (trues_eval <= q90_cal)) * 100.0)
    cal_mpiw = float(np.mean(q90_cal - q10_cal))
    cal_pinball = (pinball(trues_eval, q10_cal, 0.10) + pinball(trues_eval, q90_cal, 0.90)) / 2.0

    prob_df = pd.DataFrame([
        {
            "Uncertainty_Method": "AeroFuse-Net v4 Raw Quantile Head (10th-90th)",
            "Target_Coverage_Pct": 80.0,
            "Empirical_Coverage_PICP_Pct": round(raw_picp, 2),
            "Mean_Interval_Width_DegC": round(raw_mpiw, 4),
            "Mean_Pinball_Loss": round(raw_pinball, 4),
        },
        {
            "Uncertainty_Method": "AeroFuse-Net v4 + Conformal Calibration (CQR)",
            "Target_Coverage_Pct": 80.0,
            "Empirical_Coverage_PICP_Pct": round(cal_picp, 2),
            "Mean_Interval_Width_DegC": round(cal_mpiw, 4),
            "Mean_Pinball_Loss": round(cal_pinball, 4),
        }
    ])
    prob_df.to_csv(os.path.join("results", "metrics", "probabilistic_calibration_metrics.csv"), index=False)
    print("\n[3] Probabilistic Uncertainty & Conformal Calibration Evaluation:")
    print(prob_df.to_string(index=False))

    # Plot Regime Comparison Bar Chart
    fig_dir = os.path.join("results", "figures", "reviewer_suite")
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    y_pos = np.arange(len(regime_df))
    width = 0.28

    ax.barh(y_pos + width, regime_df["1D_CNN_RMSE"], width, label="1D_CNN_ResNet", color="#2980b9")
    ax.barh(y_pos, regime_df["iTransformer_RMSE"], width, label="iTransformer_2026", color="#27ae60")
    ax.barh(y_pos - width, regime_df["AeroFuse_v4_RMSE"], width, label="AeroFuse-Net v4 Router (Proposed)", color="#c0392b")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regime_df["Weather_Regime_Patiala"], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Root Mean Squared Error (RMSE in °C — Lower is Better)", fontsize=11)
    ax.set_title("Model Performance Across Patiala Seasonal & Extreme Weather Regimes", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, "fig5_seasonal_regime_breakdown.png")
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"\n[SUCCESS] Saved Reviewer Suite tables and Figure 5 -> {fig_path}")
    try:
        os.startfile(os.path.abspath(fig_path))
    except Exception:
        pass


if __name__ == "__main__":
    run_reviewer_suite()