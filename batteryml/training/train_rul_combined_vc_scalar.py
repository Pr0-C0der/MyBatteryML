from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.feature.combined_vc_scalar import CombinedVCScalarFeatureExtractor


def _compute_total_rul(battery: BatteryData) -> int:
    annot = RULLabelAnnotator()
    try:
        v = int(annot.process_cell(battery).item())
        return v if np.isfinite(v) else 0
    except Exception:
        return 0


def _make_battery_vector_combined(
    battery: BatteryData,
    cycle_limit: int,
    diff_base: int,
    vc_cycle_average: Optional[int],
    verbose: bool,
) -> np.ndarray:
    ext = CombinedVCScalarFeatureExtractor(
        interp_dim=1000,
        diff_base=diff_base,
        min_cycle_index=0,
        max_cycle_index=int(cycle_limit) - 1,
        use_precalculated_qdlin=False,
        smooth_enabled=True,
        cycle_average=vc_cycle_average,
        verbose=verbose,
    )
    try:
        mat = ext.process_cell(battery).numpy()  # [cycles, dim]
    except Exception:
        return np.zeros((0,), dtype=float)
    if mat.size == 0:
        return np.zeros((0,), dtype=float)
    # Flatten row-major (by cycle then feature)
    vec = mat.reshape(-1).astype(float)
    # Zero-fill any non-finite to match author behavior
    vec[~np.isfinite(vec)] = 0.0
    return vec


def _fit_label_transform(y: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=float)
    yt = np.log1p(np.clip(y, a_min=0.0, a_max=None))
    mu = float(np.mean(yt))
    sigma = float(np.std(yt)) if np.std(yt) > 1e-6 else 1e-6
    yt = (yt - mu) / sigma
    return yt, {'mu': mu, 'sigma': sigma}


def _inverse_label_transform(yt: np.ndarray, stats: dict) -> np.ndarray:
    mu = stats['mu']; sigma = stats['sigma']
    y_log = yt * sigma + mu
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


def _prepare_dataset(
    files: List[Path],
    cycle_limit: int,
    diff_base: int,
    vc_cycle_average: Optional[int],
    progress_desc: str,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    gs: List[str] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            b = BatteryData.load(f)
        except Exception:
            continue
        vec = _make_battery_vector_combined(b, cycle_limit, diff_base, vc_cycle_average, verbose)
        if vec.size == 0:
            continue
        Xs.append(vec)
        ys.append(float(_compute_total_rul(b)))
        gs.append(b.cell_id)
    if not Xs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    # Pad to max length so all vectors equal-sized
    max_len = max(x.size for x in Xs)
    X = np.vstack([np.pad(x, (0, max_len - x.size), mode='constant', constant_values=0.0).reshape(1, -1) for x in Xs])
    y = np.array(ys, dtype=float)
    g = np.array(gs, dtype=object)
    return X, y, g


def run(
    dataset: str,
    data_path: List[str],
    output_dir: str,
    cycle_limit: int,
    diff_base: int = 9,
    vc_cycle_average: Optional[int] = None,
    tune: bool = False,
    cv_splits: int = 5,
    verbose: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = build_train_test_lists(dataset, data_path)
    print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}.")

    X_train, y_train, g_train = _prepare_dataset(train_files, cycle_limit, diff_base, vc_cycle_average, 'Building train combined vectors', verbose)
    X_test, y_test, _ = _prepare_dataset(test_files, cycle_limit, diff_base, vc_cycle_average, 'Building test combined vectors', verbose)

    if X_train.size == 0 or X_test.size == 0:
        print("No data available after feature building. Aborting.")
        return

    print(f"Train batteries: {X_train.shape}, Test batteries: {X_test.shape}, Feature dim: {X_train.shape[1] if X_train.size else 0}")

    # Label transforms (author-style)
    y_train_t, train_label_stats = _fit_label_transform(y_train)

    # Linear regression pipeline (author baseline-like): imputer(0) + scaler + LR
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])

    best_params = None
    est = pipe
    if tune:
        # Small grid for LR
        param_grid = {'model__fit_intercept': [True, False]}
        unique_groups = np.unique(g_train)
        n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
        cv = GroupKFold(n_splits=n_splits)
        grid = list(ParameterGrid(param_grid))
        results = []
        best_score = np.inf
        pbar = tqdm(grid, desc=f"Tuning linear_regression ({len(grid)} cfgs x {n_splits} folds)")
        for params in pbar:
            fold_rmses = []
            for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                y_tr_t, fold_stats = _fit_label_transform(y_tr)
                model = clone(pipe).set_params(**params)
                model.fit(X_tr, y_tr_t)
                pred_t = model.predict(X_va)
                pred = _inverse_label_transform(np.asarray(pred_t, dtype=float), fold_stats)
                fold_rmses.append(mean_squared_error(y_va, pred) ** 0.5)
            mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
            results.append({**{f"param:{k}": v for k, v in params.items()}, 'mean_RMSE': mean_rmse})
            pbar.set_postfix({"RMSE": f"{mean_rmse:.3f}"})
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params
                est = clone(pipe).set_params(**params)
        cv_path = out_dir / f"{dataset.upper()}_linear_regression_cv_results_combined.csv"
        pd.DataFrame(results).to_csv(cv_path, index=False)

    # Fit full
    est.fit(X_train, y_train_t)
    y_pred_t = est.predict(X_test)
    y_pred = _inverse_label_transform(np.asarray(y_pred_t, dtype=float), train_label_stats)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"linear_regression -> MAE={mae:.3f}, RMSE={rmse:.3f}")

    rows = [{'model': 'linear_regression', 'MAE': float(mae), 'RMSE': float(rmse)}]
    if best_params is not None:
        rows[0].update({f"param:{k}": v for k, v in best_params.items()})

    suffix = f"_combined_cl{cycle_limit}"
    pd.DataFrame(rows).to_csv(out_dir / f"rul_metrics{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Author-style training: Combined VC + scalar features (Linear Regression)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'MATR1', 'MATR2', 'CLO', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'], help='Dataset name')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='One or more paths: directories with .pkl or a file listing .pkl paths')
    parser.add_argument('--output_dir', type=str, default='rul_combined', help='Output directory')
    parser.add_argument('--cycle_limit', type=int, default=100, help='Use cycles <= this index (early cycles)')
    parser.add_argument('--diff_base', type=int, default=9, help='Cycle index used as diff base for VC-diff')
    parser.add_argument('--vc_cycle_average', type=int, default=None, help='Optional stride to downsample VC curve (e.g., 5)')
    parser.add_argument('--tune', action='store_true', help='Enable GroupKFold grid search for LinearRegression')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of GroupKFold splits')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging from feature extractor')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, args.cycle_limit, diff_base=args.diff_base, vc_cycle_average=args.vc_cycle_average, tune=args.tune, cv_splits=args.cv_splits, verbose=args.verbose)


if __name__ == '__main__':
    main()


