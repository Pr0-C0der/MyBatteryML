from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from batteryml.data.battery_data import BatteryData
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.data_analysis.correlation_mod import ModularCorrelationAnalyzer, CycleScalarFeature
from batteryml.data_analysis.cycle_features import (
    charge_cycle_length,
    discharge_cycle_length,
    peak_cc_length,
    peak_cv_length,
    power_during_charge_cycle,
    power_during_discharge_cycle,
)
from tqdm import tqdm


def _build_voltage_current_analyzer(data_path: str, output_dir: str) -> ModularCorrelationAnalyzer:
    analyzer = ModularCorrelationAnalyzer(data_path, output_dir, verbose=False)
    # Only voltage/current based features
    analyzer.register_attr_mean_feature('voltage', 'voltage_in_V', description='Mean voltage per cycle')
    analyzer.register_attr_mean_feature('current', 'current_in_A', description='Mean current per cycle')
    analyzer.register_feature(CycleScalarFeature('charge_cycle_length', charge_cycle_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('discharge_cycle_length', discharge_cycle_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('peak_cc_length', peak_cc_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('peak_cv_length', peak_cv_length, depends_on=['voltage_in_V', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('power_during_charge_cycle', power_during_charge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('power_during_discharge_cycle', power_during_discharge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']))
    return analyzer


def _load_feature_matrix(analyzer: ModularCorrelationAnalyzer, file_path: Path) -> pd.DataFrame:
    cell = BatteryData.load(file_path)
    df = analyzer.build_cycle_feature_matrix(cell)
    df['battery_id'] = cell.cell_id
    return df


def _combine(files: List[Path], analyzer: ModularCorrelationAnalyzer, desc: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in tqdm(files, desc=desc):
        try:
            frames.append(_load_feature_matrix(analyzer, f))
        except Exception as e:
            print(f"[warn] skipping {f}: {e}")
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {'rul', 'cycle_number'}
    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    X = df[feat_cols].to_numpy()
    y = df['rul'].to_numpy()
    return X, y, feat_cols


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {'MAE': float(mae), 'RMSE': float(rmse)}


def _build_models(use_gpu: bool = False) -> Dict[str, Pipeline]:
    steps_base = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
    models: Dict[str, Pipeline] = {}
    models['linear_regression'] = Pipeline(steps_base + [('model', LinearRegression())])
    models['ridge'] = Pipeline(steps_base + [('model', Ridge(alpha=1.0))])
    models['elastic_net'] = Pipeline(steps_base + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
    models['svr_rbf'] = Pipeline(steps_base + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
    models['random_forest'] = Pipeline(steps_base + [('model', RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))])
    models['gbr'] = Pipeline(steps_base + [('model', GradientBoostingRegressor(random_state=42))])
    models['knn'] = Pipeline(steps_base + [('model', KNeighborsRegressor(n_neighbors=7))])
    # Gaussian Process can be heavy; include but expect slower runtime
    models['gaussian_process'] = Pipeline(steps_base + [('model', GaussianProcessRegressor())])
    # Shallow MLP
    models['mlp'] = Pipeline(steps_base + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
    if _HAS_XGB:
        models['xgb'] = Pipeline(steps_base + [('model', XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42, tree_method='hist', device=('cuda' if use_gpu else 'cpu')))])
    return models


def run(dataset: str, data_path: str, output_dir: str, use_gpu: bool = False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = build_train_test_lists(dataset, data_path)
    print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}.")
    try:
        print(f"Train batteries ({len(train_files)}): {', '.join([p.stem for p in train_files])}")
        print(f"Test batteries ({len(test_files)}): {', '.join([p.stem for p in test_files])}")
    except Exception:
        pass

    analyzer = _build_voltage_current_analyzer(data_path, str(out_dir / f"{dataset.upper()}_tmp"))
    df_train = _combine(train_files, analyzer, desc='Building train matrices')
    df_test = _combine(test_files, analyzer, desc='Building test matrices')
    if df_train.empty or df_test.empty:
        print("No data after feature extraction.")
        return

    X_train, y_train, feature_cols = _prepare_xy(df_train)
    X_test, y_test, _ = _prepare_xy(df_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(feature_cols)}")

    models = _build_models(use_gpu=use_gpu)
    results = []
    preds_out: Dict[str, List[float]] = {}
    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _evaluate(y_test, y_pred)
        results.append({'model': name, **metrics})
        preds_out[name] = list(map(float, y_pred))
        print(f"  {name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")

    pd.DataFrame(results).to_csv(out_dir / f"{dataset.upper()}_all_models_metrics.csv", index=False)
    # Save predictions per model (optional)
    with open(out_dir / f"{dataset.upper()}_all_models_predictions.json", 'w') as f:
        json.dump(preds_out, f)


def main():
    parser = argparse.ArgumentParser(description='Train/test many ML models for RUL using only voltage/current-derived features (per-cycle).')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'CALCE'], help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='rul_all_models', help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU for XGBoost if available')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, use_gpu=args.gpu)


if __name__ == '__main__':
    main()


