import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib

def load_and_prepare_data(target_col="temperature_2m", input_window=24, horizon=1):
    """Loads processed data and creates supervised lagging features for ML models."""
    processed_path = os.path.join("data", "processed", "processed_local_station.parquet")
    df = pd.read_parquet(processed_path)
    
    # Create lagged features for autoregressive forecasting
    features = []
    target = []
    
    values = df[target_col].values
    # Use multiple features including our engineered physics features
    feature_matrix = df.values
    
    X, y = [], []
    for i in range(input_window, len(df) - horizon):
        X.append(feature_matrix[i - input_window:i].flatten())
        y.append(feature_matrix[i + horizon - 1, df.columns.get_loc(target_col)])
        
    X = np.array(X)
    y = np.array(y)
    
    # Chronological Train-Val-Test Split (70% Train, 15% Val, 15% Test)
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{model_name}] Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse}

def run_ml_training():
    print("=" * 70)
    print("Training Tier 1 & 2 ML Baselines (XGBoost & LightGBM)...")
    print("=" * 70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    results_dir = os.path.join("results", "metrics")
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join("data", "processed")
    
    # 1. XGBoost Baseline
    print("\nTraining XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist" # Uses fast CPU/GPU histogram binning
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    xgb_preds = xgb_model.predict(X_test)
    res_xgb = evaluate_model(y_test, xgb_preds, "XGBoost")
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_baseline.pkl"))
    
    # 2. LightGBM Baseline
    print("\nTraining LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    lgb_preds = lgb_model.predict(X_test)
    res_lgb = evaluate_model(y_test, lgb_preds, "LightGBM")
    joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm_baseline.pkl"))
    
    print("=" * 70)
    print("[SUCCESS] ML Baselines trained, evaluated, and saved successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_ml_training()