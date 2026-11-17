# AeroFuse-Weather-AI

Multi-horizon (1 to 24 hour) 2 m temperature forecasting for Patiala, Punjab, India, from hourly ERA5-based reanalysis (2019 to 2025). The project benchmarks 13 forecasting models and develops **AeroFuse-Net**, a dual-modal model that combines a local time series with a 5 x 5 regional grid through a horizon-wise router over frozen expert forecasters.

> **Status of results.** Everything reported below comes from trained models and the saved test predictions in `results/metrics/`, or was recomputed from those files. The earlier "v5" sparse-router experiments are **not reported as results**; see [Status of the v5 sparse dead-zone router](#status-of-the-v5-sparse-dead-zone-router).

---

## 1. Summary

- **Task.** Predict `temperature_2m` for the next 24 hours from the previous 24 hours of 19 local features and a 24 x 6 x 5 x 5 regional grid.
- **Data.** 61,368 hourly records, 2019-01-01 00:00 to 2025-12-31 23:00 UTC, chronological 70 / 15 / 15 split, 61,253 windows (42,909 train / 9,172 validation / 9,172 test).
- **Main finding.** A tuned local-only inverted transformer (iTransformer) is the strongest single model by MAE (1.0217 C). AeroFuse-Net v4 reaches the lowest RMSE (1.4378 C) and R2 (0.9674) but is **statistically indistinguishable** from iTransformer overall once correlated forecast errors are accounted for. It is worse at 1 hour ahead and better in the monsoon regime.
- **Negative result.** Naive joint fusion of local and grid inputs (AeroFuse v1, v3) is worse than a good local-only model. Freezing the experts and routing between them (v4) removes that penalty.
- **Uncertainty.** The raw 10% to 90% quantile head covers about 73% of outcomes (target 80%). Rolling conformalized quantile regression (CQR) restores about 79.4% coverage.
- **Scope.** One location, one test year, reanalysis (not station observations), no operational NWP baseline.

---

## 2. Data

| Item | Detail |
| --- | --- |
| Source | Open-Meteo historical archive API (ERA5-based reanalysis), no API key |
| Period | 2019-01-01 to 2025-12-31, hourly, UTC, 61,368 rows, no missing timestamps |
| Target site | Patiala, 30.34 N, 76.39 E |
| Local variables (12) | temperature, relative humidity, dew point, apparent temperature, precipitation, rain, surface pressure, cloud cover, wind speed, wind direction, wind gusts, shortwave radiation |
| Engineered features (7) | sin/cos of hour of day, sin/cos of day of year, vapour pressure deficit (Magnus-Tetens saturation formula), 24 h rolling mean of temperature, 12 h rolling standard deviation of pressure |
| Local input | 24 hours x 19 features |
| Spatial grid | 5 x 5 points at 0.5 degree spacing (about 55 km), 6 channels (temperature, pressure, humidity, wind speed, wind direction, cloud cover); tensor `(61368, 6, 5, 5)` |
| Missing values | forward fill, then backward fill (the raw download had none) |
| Scaling | `StandardScaler` for local features and per-channel mean/std for the grid, **fitted on training data only**; targets are kept in degrees C |

### Splits and windows

Chronological, no shuffling. Windows (24 h input, 24 h target) are built inside each split so no target crosses a split boundary.

| Partition | Raw hours | Period (UTC) | Windows |
| --- | --- | --- | --- |
| Train | 42,956 | 2019-01-01 to 2023-11-25 19:00 | 42,909 |
| Validation | 9,206 | 2023-11-25 20:00 to 2024-12-13 09:00 | 9,172 |
| Test | 9,206 | 2024-12-13 10:00 to 2025-12-31 23:00 | 9,172 |
| **Total** | **61,368** | | **61,253** |

The 115 steps removed are 47 for train (24 warm-up + 23 end-of-split target trim) and 34 each for validation and test (11 boundary + 23 end trim). The date arithmetic is in `results/v5_6_final_manifest/table1_v5_6_exact_timestamp_reconciliation.csv`.

### Important caveats about the data

- The "local station" series is the ERA5-based value at the centre grid point, **not a physical weather station**. The target and the grid inputs come from the same product, and the centre grid cell holds the same temperature that is being predicted.
- The grid reaches only about 110 km from Patiala. At a 10 m/s wind, air moves about 36 km per hour, so the grid shows upwind conditions roughly 3 hours ahead. Larger grids are needed for synoptic systems that start several hundred km away.
- The test set covers a single calendar year, and neighbouring windows overlap by 47 of 48 hours.
- Timestamps are UTC. Local time in Patiala is UTC + 5:30.

---

## 3. Models

| Tier | Model | Notes |
| --- | --- | --- |
| Reference | Naive persistence | repeats the last observed value |
| Reference | Diurnal 24 h cycle | each hour equals the same hour yesterday |
| Classical ML | XGBoost, LightGBM | gradient-boosted trees on flattened 24 h x 19 windows (1 h ahead) |
| Deep, single-modal | 1D-CNN ResNet | 2 conv layers, 64 channels, kernel 3, batch norm, GELU, skip connection |
| Deep, single-modal | 2D Spatial ResNet | 144 stacked grid channels (24 x 6) through a 2D CNN on the 5 x 5 grid |
| Deep, single-modal | BiLSTM | 2 layers, 64 hidden units |
| Transformer | iTransformer | each of the 19 variables' 24 h history is one 64-d token; 2 layers, 4 heads |
| Transformer | PatchTST | patch length 4, stride 2, 2 layers |
| Proposed | AeroFuse-Net v1 to v4 | see below |

**Baseline training recipe.** AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, 12 epochs, batch size 256, Huber loss, gradient clipping at 1.0, best-validation-MSE checkpoint. Hardware: HP Omen 16, RTX 4050 6 GB, PyTorch. Five deep models train in about 3.5 minutes in total.

### AeroFuse-Net versions

| Version | Design | Test MAE / RMSE (C) | Lesson |
| --- | --- | --- | --- |
| v1 | 2D-CNN + SE block on the grid, 1D-CNN + BiLSTM on local data, cross-modal attention (local state as query, grid as key/value) | 1.2139 / 1.5827 | Fusion beats grid-only and BiLSTM, loses to iTransformer |
| v2 | residual skip around the BiLSTM, linear autoregressive highway, non-crossing detached quantile heads | 1.1128 / 1.4996 | fixes tanh squashing, dropout before the output layer and quantile-gradient interference |
| v3 | joint training from scratch, zero-initialised spatial gate, advection features | 1.2468 / 1.6440 | worse; hypothesised branch gradient competition (gradient norms were not logged) |
| **v4 Router** | four frozen experts (iTransformer, 1D-CNN, PatchTST, 2D ResNet) combined by a horizon-wise router | **1.0366 / 1.4378** | best RMSE and R2; accuracy equivalent to iTransformer |

**How the v4 router works.**

1. Four frozen experts each produce a 24 h forecast.
2. A small CNN encodes the latest 5 x 5 grid and the grid minus its centre cell (a crude advection signal) into 32 numbers.
3. These plus the latest local feature vector feed an MLP that outputs 4 x 24 logits.
4. Weights = softmax(horizon prior + 0.15 x tanh(logits)). Priors start at 1.35, 0.95, 0.75, -0.35 (iTransformer, 1D-CNN, PatchTST, 2D grid), so the model starts as a local-heavy mixture and can only move modestly.
5. Forecast = weighted sum of experts, then a per-horizon scale and bias.
6. The 10% and 90% quantile heads use expert disagreement (0.65 x standard deviation across experts) plus a learned softplus offset on detached inputs.

**Training.** 10 epochs, AdamW lr 2e-3, cosine schedule, about 45 s. Loss = Huber + 0.05 x L1 on hour-to-hour differences (derivative matching) + 0.25 x (pinball loss at 10% + pinball loss at 90%).

"Physics-informed" here means the vapour pressure deficit input feature and the derivative-matching loss term. There is no dew-point consistency loss.

---

## 4. Results

All values are on the 9,172 held-out test windows, in degrees C. Source: `results/metrics/benchmark_comparison.csv` and `persistence_and_climatology_baselines.csv`. I re-derived MAE and RMSE for iTransformer, v4 and the 1D-CNN from the saved prediction files and they match.

| Model | MAE 1h | MAE 6h | MAE 12h | MAE 24h | Overall MAE | Overall RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| iTransformer | 0.479 | 0.940 | 1.085 | 1.190 | **1.0217** | 1.4477 | 0.9670 |
| AeroFuse v4 Router | 0.503 | 0.958 | 1.094 | 1.201 | 1.0366 | **1.4378** | **0.9674** |
| 1D-CNN ResNet | 0.636 | 0.994 | 1.112 | 1.211 | 1.0626 | 1.4441 | 0.9672 |
| PatchTST | 0.656 | 0.992 | 1.118 | 1.242 | 1.0741 | 1.5136 | 0.9639 |
| AeroFuse v2 | 0.661 | 1.058 | 1.180 | 1.266 | 1.1128 | 1.4996 | 0.9646 |
| AeroFuse v1 | 1.057 | 1.176 | 1.219 | 1.316 | 1.2139 | 1.5827 | 0.9606 |
| AeroFuse v3 | 0.686 | 1.140 | 1.306 | 1.436 | 1.2468 | 1.6440 | 0.9574 |
| BiLSTM | 0.877 | 1.170 | 1.320 | 1.270 | 1.2594 | 1.6655 | 0.9563 |
| 2D Spatial ResNet | 1.049 | 1.200 | 1.327 | 1.410 | 1.2819 | 1.6526 | 0.9570 |
| Diurnal 24 h baseline | 1.187 | 1.187 | 1.187 | 1.189 | 1.1872 | 1.7341 | n/a |
| Naive persistence | 0.926 | 4.605 | 6.318 | 1.189 | 4.2263 | 5.4690 | n/a |

LightGBM, trained for 1 hour ahead only, reached MAE 0.4044 and RMSE 0.6280 (from the training log; not stored in `results/`).

### Observations

1. **The winner depends on the metric.** iTransformer is best on MAE and at short lead times (1.5% ahead of v4 on MAE). v4 is best on RMSE (0.7% ahead) and R2.
2. **Gains over a simple rule are modest.** Against the diurnal baseline, iTransformer cuts MAE by 14% and RMSE by 16.5%. Squared-error skill against the diurnal baseline is 30.3% (iTransformer) and 31.3% (v4).
3. **At 24 hours ahead no model beats "same hour yesterday"** (diurnal MAE 1.1886; iTransformer 1.1901; v4 1.2006).
4. **Error grows with lead time**, from 0.48 C at 1 h to 1.19 C at 24 h for iTransformer.
5. **Space alone is weak.** The grid-only model is the worst model, and naive fusion (v1, v3) hurts.
6. **Tails.** v4 has slightly lighter tails: 95th percentile of absolute error 2.72 C vs 2.79 C and 99th percentile 4.79 C vs 5.03 C.

### Seasonal and extreme regimes (RMSE, C)

Source: `results/metrics/seasonal_and_extreme_regime_evaluation.csv`.

| Regime | Windows | AeroFuse v4 | iTransformer | 1D-CNN | Lowest |
| --- | --- | --- | --- | --- | --- |
| Winter fog (Dec to Feb) | 2,559 | 1.379 | 1.413 | 1.365 | 1D-CNN |
| Summer heat (Apr to Jun) | 2,184 | 1.871 | 1.877 | 1.861 | 1D-CNN |
| Monsoon (Jul to Sep) | 2,208 | 1.075 | 1.125 | 1.120 | AeroFuse v4 |
| Post-monsoon (Oct to Nov) | 1,464 | 1.192 | 1.121 | 1.198 | iTransformer |
| 08 to 18 UTC (13:30 to 23:30 local) | 4,201 | 1.441 | 1.462 | 1.449 | AeroFuse v4 |
| Night (radiative cooling) | 4,958 | 1.436 | 1.436 | 1.440 | tie |
| Extreme heat (above 38 C) | 912 | 1.974 | 1.898 | 1.913 | iTransformer |
| Rapid fronts (6 h change above 6 C) | 2,547 | 1.504 | 1.488 | 1.504 | iTransformer |

v4's clearest advantage is the monsoon (4.4% lower RMSE than iTransformer). It is worse in extreme heat (about 4%) and rapid fronts (about 1%). March is not assigned to any season, so the four seasons cover 8,415 of 9,172 windows. The `Best_Model_in_Regime` column in the CSV is not reliable for winter fog and summer heat, where the 1D-CNN has the lowest RMSE.

---

## 5. Statistical comparison

Forecast errors at the same lead time in neighbouring hourly windows are strongly correlated (lag-1 autocorrelation of the 6 h error is 0.84), so a plain Diebold-Mariano test overstates significance. The table below recomputes the v4 vs iTransformer comparison from the saved predictions with the Harvey-Leybourne-Newbold small-sample correction and a Bartlett-kernel HAC variance (positive difference means v4 is better; p-values unadjusted).

| Horizon (HAC lag) | Loss | Mean difference | HLN statistic | p-value | Reading |
| --- | --- | --- | --- | --- | --- |
| All 24 h (23) | squared error | +0.0286 C squared | 0.73 | 0.47 | no difference |
| All 24 h (23) | absolute error | -0.0149 C | -1.69 | 0.09 | no difference at 5% |
| 1 h (0) | squared error | -0.0265 C squared | -5.01 | 6e-7 | iTransformer better |
| 1 h (0) | absolute error | -0.0243 C | -9.54 | below 1e-4 | iTransformer better |
| 6 h (5) | squared error | +0.0003 C squared | 0.01 | 0.99 | identical |
| 6 h (5) | absolute error | -0.0175 C | -2.17 | 0.03 | iTransformer better, weak |
| 12 h (11) | squared error | +0.0604 C squared | 1.20 | 0.23 | no difference |
| 24 h (23) | squared error | +0.0252 C squared | 0.64 | 0.52 | no difference |

A moving-block bootstrap (168 h blocks, 1,000 resamples) gives a 95% interval of [-0.062, +0.113] C squared for the squared-error gain and [-0.039, +0.008] C for the absolute-error gain. Both include zero.

**Conclusion:** v4 matches the best single model overall; its advantages are lighter tails and the monsoon regime, and it is worse at 1 hour ahead. The pairwise tests in `results/metrics/statistical_significance_dm_test*.csv` use an uncorrected DM test and should be treated as indicative only.

### Where a spatial gate could help

Splitting hours by how much the true temperature moves (top 23% = volatile; a diagnostic, since the true change is unknown at forecast time):

| Hours | Share | MAE iTransformer | MAE v4 | RMSE iTransformer | RMSE v4 |
| --- | --- | --- | --- | --- | --- |
| Calm | 77.3% | 0.992 | 1.017 | 1.397 | 1.397 |
| Volatile | 22.7% | 1.122 | 1.104 | 1.608 | 1.570 |

On calm hours v4's soft gate costs about 2.5% in MAE; on volatile hours it gains about 1.6% in MAE and 2.3% in RMSE. This motivates a gate that switches the grid off in calm conditions (below).

---

## 6. Uncertainty quantification

The v4 quantile head predicts the 10th and 90th percentiles (nominal 80% interval). Metrics: PICP (share of truths inside), MPIW (mean interval width), pinball loss.

**CQR.** For each past forecast the score is `s = max(q10 - y, y - q90)`; the interval is widened on both sides by the `ceil((n+1)(1-alpha))/n` empirical quantile of the scores, computed per horizon.

**Earlier bug (fixed in the method described here).** The first CQR in `run_reviewer_proof_suite.py` computed the lower and upper offsets separately as signed distances and took their 82% quantile. Most truths lie inside the band, so these quantiles are negative and the "calibrated" band shrank, dropping coverage from 73.81% to 58.15%. The correct score is the combined maximum above.

| Check (v4 saved quantiles, full test year) | Result |
| --- | --- |
| Raw coverage, all horizons | 73.05%, mean width 2.82 C |
| Raw coverage at 1 / 6 / 12 / 24 h | 83.9% / 71.3% / 71.4% / 74.3% |
| Rolling CQR, 720 h trailing window, gap of one horizon, 8,415 evaluated windows | **79.4%**, mean width 3.24 C |
| Rolling CQR coverage at 1 / 6 / 12 / 24 h | 80.4% / 79.5% / 79.5% / 79.0% |
| Rolling CQR width at 1 h / 24 h | 1.62 C / 3.78 C |
| One-off CQR calibrated on the first 20% of the test year | 82.9% (84.6% at 1 h, 77.8% at 24 h), width 3.41 C |

The raw band is too wide at 1 hour and too narrow elsewhere, so a single global correction cannot fix it; per-horizon rolling calibration can. Coverage is marginal (averaged over all conditions); coverage within individual regimes has not been evaluated. The older CSV `probabilistic_calibration_metrics.csv` scores only the second half of the test year, which is why its raw coverage (73.81%) differs from the full-year figure.

---

## 7. Interpretability

`results/figures/fig3_spatial_cnn_gradcam.png` and the v4 variant show **input-gradient saliency**: the mean absolute gradient of the forecast with respect to the grid input, averaged over time and channels, for one batch. It is not Grad-CAM (there is no convolutional feature-map weighting) and shows sensitivity, not causation. No validated occlusion experiment has been run, so no claim is made about the importance of individual grid cells (for example the north-eastern Shivalik foothills).

---

## 8. Status of the v5 sparse dead-zone router

The v5 idea is a router whose spatial weight is exactly zero when the grid signal is small (a dead-zone), to avoid injecting noise on calm hours. The calm/volatile analysis in Section 5 supports the motivation.

**However, no v5 model was trained or evaluated.** The scripts `src/evaluation/run_v5_reviewer_audit_and_fixes.py` and `run_v5_1` to `run_v5_6`, and the tables under `results/v5_audit/`, `results/v5_4_Stage1_Frozen_Backup/`, `results/v5_5_canonical/` and `results/v5_6_final_manifest/`, were generated with synthetic errors fitted to target RMSE/MAE values or with typed-in numbers. They contain **no real model results** and must not be cited. Specifically, these quantities are *not* results: the sparse-router RMSE 1.4033 / MAE 1.0002, 76.3% calm-sky sparsity, the three-seed ablation table, the v5 Diebold-Mariano and Holm p-values, the 24-horizon CQR table (80.08% coverage, 3.077 C), and the occlusion table. The dashboards `app/dashboard_v5.py`, `dashboard_v5_6.py` and `dashboard_v5_7_complete.py` draw synthetic forecast curves.

The one correct artefact among them is the window reconciliation (`table1_*`), which is plain date arithmetic.

**Planned.** Train the dead-zone router for real (spatial weight `alpha = max(0, z - tau) / (1 - tau)`, tau chosen on validation), over several seeds, save predictions, and evaluate with the same pipeline as Sections 4 to 6.

---

## 9. Limitations

- One site, one test year, reanalysis rather than station observations; target and inputs share a data product.
- No operational NWP baseline (for example IMD or ECMWF forecasts).
- Baselines and v4 were trained once; differences of about 0.01 C between models are within plausible seed-to-seed variation and no multi-seed estimates are reported.
- Baselines share one hyperparameter recipe and were not tuned. The classical tier contains only XGBoost and LightGBM; TimesNet, SARIMAX, Prophet, CatBoost, ConvLSTM and foundation models are not included.
- The v1 to v4 progression changes several things at once (backbone, training regime, features); a clean router/physics ablation on real models has not been run.
- The v4 router is a gated ensemble of already-trained models, so its novelty is in the horizon-wise, grid-conditioned routing, not a new backbone.
- The grid is small (about +/- 110 km) and covers six surface variables only.

---

## 10. Repository layout

```
app/
  dashboard.py                   Streamlit dashboard reading the saved v4 predictions
  dashboard_v5*.py               audit dashboards built on the withdrawn v5 tables (synthetic curves)
data/
  raw/                           (git-ignored) local_station_timeseries.parquet, spatial_grid_tensor.npz
  processed/                     trained checkpoints (.pt, .pkl) and scalers
results/
  metrics/                       saved test predictions (.npz) and benchmark, regime, DM and calibration CSVs
  figures/                       benchmark, fan chart, saliency and ablation figures (v1 to v4)
  v5_*                           withdrawn v5 tables (see Section 8)
src/
  data/fetch_dual_modal_data.py  download ERA5-based data from Open-Meteo
  features/preprocess.py         imputation, cyclical features, vapour pressure deficit, rolling features
  models/architectures_v1.py, architectures_v2.py
  training/                      train_ml_baselines.py, train_deep_models.py, upgrade_aerofuse.py
  evaluation/                    run_ablation_and_explainability.py (v3), run_aerofuse_v4_router.py,
                                 run_reviewer_proof_suite.py, and the withdrawn run_v5*.py scripts
```

---

## 11. Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements.txt scikit-learn xgboost lightgbm pyarrow

python src/data/fetch_dual_modal_data.py        # writes data/raw/*
python src/features/preprocess.py               # writes data/processed/processed_local_station.parquet
python src/training/train_ml_baselines.py       # XGBoost, LightGBM (1 h)
python src/training/train_deep_models.py        # 1D-CNN, 2D ResNet, BiLSTM, iTransformer, AeroFuse v1
python src/training/upgrade_aerofuse.py         # PatchTST, AeroFuse v2
python src/evaluation/run_ablation_and_explainability.py   # AeroFuse v3, DM tests, figures
python src/evaluation/run_aerofuse_v4_router.py            # AeroFuse v4 router
python src/evaluation/run_reviewer_proof_suite.py          # baselines, regimes, quantile calibration
streamlit run app/dashboard.py
```

The dashboard and the saved checkpoints under `data/processed/` and predictions under `results/metrics/` let you inspect v1 to v4 results without retraining. Exact numbers vary slightly with hardware and random seeds.

---

## 12. Citation and acknowledgements

Data: ERA5 reanalysis (Copernicus / ECMWF) via the Open-Meteo historical API. Model families: iTransformer, PatchTST, ResNet, BiLSTM; uncertainty: conformalized quantile regression (Romano et al., 2019); forecast comparison: Diebold-Mariano with the Harvey-Leybourne-Newbold correction.

If you use this repository, cite the repository and state that the data are reanalysis for a single site.
