from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Optional RAPIDS cuML (GPU) support
try:
    import cuml  # noqa: F401
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

# Optional GPyTorch (GPU Gaussian Processes)
try:
    import torch  # noqa: F401
    import gpytorch  # noqa: F401
    _HAS_GPYTORCH = True
except Exception:
    _HAS_GPYTORCH = False

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.data_analysis import cycle_features as cf


def _available_feature_fns() -> Dict[str, callable]:
    # Expose a curated set of feature functions from cycle_features
    names = [
        'avg_c_rate', 'max_temperature', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length', 'peak_cv_length',
        'cycle_length', 'power_during_charge_cycle', 'power_during_discharge_cycle',
        'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio'
    ]
    feats: Dict[str, callable] = {}
    for n in names:
        fn = getattr(cf, n, None)
        if callable(fn):
            feats[n] = fn
    return feats


def _default_feature_names() -> List[str]:
    # Reasonable default subset
    return [
        'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length',
        'cycle_length'
    ]


def _compute_total_rul(battery: BatteryData) -> int:
    annot = RULLabelAnnotator()
    try:
        v = int(annot.process_cell(battery).item())
        return v if np.isfinite(v) else 0
    except Exception:
        return 0


def _build_cycle_feature_table(battery: BatteryData, feature_fns: Dict[str, callable]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for c in battery.cycle_data:
        row: Dict[str, float] = {'cycle_number': c.cycle_number}
        for name, fn in feature_fns.items():
            try:
                val = fn(battery, c)
                row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_df_by_cycle_limit(df: pd.DataFrame, cycle_limit: Optional[int], verbose: bool = False) -> pd.DataFrame:
    dfn = df.sort_values('cycle_number').reset_index(drop=True)
    if cycle_limit is not None:
        dfn = dfn[dfn['cycle_number'] <= int(cycle_limit)].reset_index(drop=True)
        if verbose:
            print(f"[filter] cycle_limit={cycle_limit} -> {len(dfn)} rows remaining")
    return dfn


def _make_windows(df: pd.DataFrame, feature_names: List[str], total_rul: int, window_size: int, cycle_limit: Optional[int] = None, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # df must be sorted by cycle_number ascending
    dfn = _filter_df_by_cycle_limit(df, cycle_limit, verbose=verbose)
    # Ensure features present
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    num_cycles = len(dfn)
    # Window targets the next cycle's RUL: cycles [k, k+N-1] -> target at (k+N)
    last_start = num_cycles - (window_size + 1)
    if last_start < 0:
        if verbose:
            print(f"[skip] not enough cycles for windows: cycles={num_cycles}, window_size={window_size}")
        return np.zeros((0, len(cols) * window_size), dtype=float), np.zeros((0,), dtype=float)

    feat_mat = dfn[cols].to_numpy()  # shape [num_cycles, num_feats]
    for k in range(0, last_start + 1):
        w = feat_mat[k:k + window_size, :]  # [N, F]
        # If any row is all NaN, imputer will handle later; leave NaN as np.nan
        x = w.reshape(-1)  # flatten row-major: time progresses first
        # target RUL at cycle (k+window_size)
        rul = max(0, total_rul - (k + window_size))
        X_list.append(x)
        y_list.append(float(rul))

    X = np.vstack(X_list) if X_list else np.zeros((0, len(cols) * window_size), dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def _make_battery_vector(df: pd.DataFrame, feature_names: List[str], cycle_limit: int, verbose: bool = False) -> np.ndarray:
    dfn = _filter_df_by_cycle_limit(df, cycle_limit, verbose=verbose)
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return np.zeros((0,), dtype=float)
    # Ensure exactly cycle_limit rows: pad with NaN or truncate
    K = int(cycle_limit)
    mat = dfn[cols].to_numpy()  # [num_cycles, F]
    F = len(cols)
    if mat.shape[0] >= K:
        matK = mat[:K, :]
    else:
        pad = np.full((K - mat.shape[0], F), np.nan, dtype=float)
        matK = np.vstack([mat, pad])
    vec = matK.reshape(-1)
    if verbose:
        n_nans = int(np.isnan(vec).sum())
        print(f"[battery_vector] len={vec.size}, features={F}, cycles_used={min(K, mat.shape[0])}, pad={(K - min(K, mat.shape[0]))}, NaNs={n_nans}")
    return vec


def _report_missing_for_battery(cell_id: str, df: pd.DataFrame, feature_names: List[str], cycle_limit: Optional[int], verbose: bool = False) -> None:
    dfn = _filter_df_by_cycle_limit(df, cycle_limit, verbose=False)
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return
    # Identify rows with any NaN among selected features
    sub = dfn[['cycle_number'] + cols].copy()
    mask_any = sub[cols].isna().any(axis=1)
    if not mask_any.any():
        if verbose:
            print(f"[missing] {cell_id}: no NaNs in selected features")
        return
    for _, row in sub[mask_any].iterrows():
        missing_feats = [c for c in cols if pd.isna(row[c])]
        print(f"[missing] battery={cell_id} cycle={int(row['cycle_number'])} features={missing_feats}")


def _prepare_dataset_windows(files: List[Path], feature_fns: Dict[str, callable], feature_names: List[str], window_size: int, cycle_limit: Optional[int], verbose: bool, report_missing: bool, progress_desc: str) -> Tuple[np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            battery = BatteryData.load(f)
        except Exception:
            continue
        total_rul = _compute_total_rul(battery)
        df = _build_cycle_feature_table(battery, feature_fns)
        if report_missing:
            _report_missing_for_battery(battery.cell_id, df, feature_names, cycle_limit, verbose=verbose)
        Xw, yw = _make_windows(df, feature_names, total_rul, window_size, cycle_limit=cycle_limit, verbose=verbose)
        if Xw.size and yw.size:
            Xs.append(Xw)
            ys.append(yw)
        elif verbose:
            print(f"[skip] {f.name}: produced no windows")
    if not Xs:
        return np.zeros((0, len(feature_names) * window_size), dtype=float), np.zeros((0,), dtype=float)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y


def _prepare_dataset_battery_level(files: List[Path], feature_fns: Dict[str, callable], feature_names: List[str], cycle_limit: int, verbose: bool, report_missing: bool, progress_desc: str) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            battery = BatteryData.load(f)
        except Exception:
            continue
        total_rul = _compute_total_rul(battery)
        df = _build_cycle_feature_table(battery, feature_fns)
        if report_missing:
            _report_missing_for_battery(battery.cell_id, df, feature_names, cycle_limit, verbose=verbose)
        vec = _make_battery_vector(df, feature_names, cycle_limit=cycle_limit, verbose=verbose)
        if vec.size == 0:
            if verbose:
                print(f"[skip] {f.name}: no features present after filtering")
            continue
        X_list.append(vec)
        y_list.append(float(total_rul))
    if not X_list:
        return np.zeros((0, len(feature_names) * int(cycle_limit)), dtype=float), np.zeros((0,), dtype=float)
    X = np.vstack([x.reshape(1, -1) for x in X_list])
    y = np.array(y_list, dtype=float)
    if verbose:
        print(f"[battery_level] X={X.shape}, y={y.shape}")
    return X, y


def _build_models(use_gpu: bool = False) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    base_steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
    # Prefer cuML implementations when available and GPU requested
    if use_gpu and _HAS_CUML:
        from cuml.linear_model import LinearRegression as cuLinearRegression, Ridge as cuRidge, ElasticNet as cuElasticNet
        from cuml.svm import SVR as cuSVR
        from cuml.ensemble import RandomForestRegressor as cuRF
        # Linear models (GPU)
        models['linear_regression'] = Pipeline(base_steps + [('model', cuLinearRegression())])
        models['ridge'] = Pipeline(base_steps + [('model', cuRidge())])
        models['elastic_net'] = Pipeline(base_steps + [('model', cuElasticNet())])
        # Kernel
        models['svr_rbf'] = Pipeline(base_steps + [('model', cuSVR(kernel='rbf', C=10.0))])
        # Trees
        models['random_forest'] = Pipeline(base_steps + [('model', cuRF(n_estimators=40, random_state=42))])
    else:
        # Linear models (CPU)
        models['linear_regression'] = Pipeline(base_steps + [('model', LinearRegression())])
        models['ridge'] = Pipeline(base_steps + [('model', Ridge(alpha=1.0))])
        models['elastic_net'] = Pipeline(base_steps + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
        # Kernel (CPU)
        models['svr_rbf'] = Pipeline(base_steps + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
        # Trees (CPU)
        models['random_forest'] = Pipeline(base_steps + [('model', RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1))])
    # (GradientBoosting removed)
    # Shallow MLP
    models['mlp'] = Pipeline(base_steps + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
    # XGBoost
    if _HAS_XGB:
        models['xgboost'] = Pipeline(base_steps + [('model', XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method='hist', device=('cuda' if use_gpu else 'cpu')))])
    # PLSR
    models['plsr'] = Pipeline(base_steps + [('model', PLSRegression(n_components=10))])
    # PCR (PCA + Linear)
    models['pcr'] = Pipeline(base_steps + [('model', Pipeline([('pca', PCA(n_components=20)), ('lr', LinearRegression())]))])

    # Optional: GPyTorch SVGP (GPU scalable GP)
    if use_gpu and _HAS_GPYTORCH:
        class _SVGPRegressor:
            def __init__(self, inducing_points: int = 1024, batch_size: int = 2048, iters: int = 1500, lr: float = 1e-2):
                self.m = int(inducing_points)
                self.batch = int(batch_size)
                self.iters = int(iters)
                self.lr = float(lr)
                self._predict_fn = None

            def fit(self, X, y):
                import torch
                import gpytorch
                from torch.utils.data import TensorDataset, DataLoader

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                X_t = torch.tensor(X, dtype=torch.float32, device=device)
                y_t = torch.tensor(y, dtype=torch.float32, device=device)
                y_mean = y_t.mean(); y_std = y_t.std().clamp_min(1e-6)
                y_n = (y_t - y_mean) / y_std

                class GPModel(gpytorch.models.ApproximateGP):
                    def __init__(self, inducing, d):
                        var = gpytorch.variational.VariationalStrategy(
                            self,
                            inducing,
                            gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0)),
                            learn_inducing_locations=True,
                        )
                        super().__init__(var)
                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel(ard_num_dims=d)
                        )

                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

                m = min(self.m, X_t.shape[0])
                inducing = X_t[torch.randperm(X_t.shape[0])[:m]].contiguous()
                model = GPModel(inducing, X_t.shape[1]).to(device)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                model.train(); likelihood.train()
                opt = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()}
                ], lr=self.lr)
                mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_t.shape[0])

                loader = DataLoader(TensorDataset(X_t, y_n), batch_size=self.batch, shuffle=True, drop_last=False)
                for _ in tqdm(range(self.iters), desc='SVGP epochs', leave=False):
                    for xb, yb in tqdm(loader, desc='SVGP batches', leave=False):
                        opt.zero_grad(set_to_none=True)
                        out = model(xb)
                        loss = -mll(out, yb)
                        loss.backward()
                        opt.step()

                model.eval(); likelihood.eval()

                def _predict(Xtest):
                    Xt = torch.tensor(Xtest, dtype=torch.float32, device=device)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        mean = likelihood(model(Xt)).mean
                    return (mean * y_std + y_mean).detach().cpu().numpy()

                self._predict_fn = _predict
                return self

            def predict(self, X):
                if self._predict_fn is None:
                    raise RuntimeError('SVGP model not fitted')
                return self._predict_fn(X)

        models['gpytorch_svgp'] = Pipeline(base_steps + [('model', _SVGPRegressor())])

    return models


def run(dataset: str, data_path: str, output_dir: str, window_size: int, features: Optional[List[str]] = None, use_gpu: bool = False, cycle_limit: Optional[int] = None, battery_level: bool = False, verbose: bool = False, report_missing: bool = False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = build_train_test_lists(dataset, data_path)
    print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}.")
    try:
        train_names = [p.stem for p in train_files]
        test_names = [p.stem for p in test_files]
        if train_names:
            print(f"Train batteries ({len(train_names)}): {', '.join(train_names)}")
        if test_names:
            print(f"Test batteries ({len(test_names)}): {', '.join(test_names)}")
    except Exception:
        pass

    # Feature selection
    all_fns = _available_feature_fns()
    if not features or features == ['default']:
        feature_names = _default_feature_names()
    elif features == ['all']:
        feature_names = list(all_fns.keys())
    else:
        feature_names = []
        for n in features:
            if n not in all_fns:
                print(f"[warn] unknown feature '{n}' — skipping")
                continue
            feature_names.append(n)
        if not feature_names:
            feature_names = _default_feature_names()

    feature_fns = {n: all_fns[n] for n in feature_names if n in all_fns}

    # Build datasets
    if battery_level:
        if cycle_limit is None:
            raise ValueError("--battery_level requires --cycle_limit to define vector length")
        X_train, y_train = _prepare_dataset_battery_level(train_files, feature_fns, feature_names, cycle_limit=int(cycle_limit), verbose=verbose, report_missing=report_missing, progress_desc='Building train battery vectors')
        X_test, y_test = _prepare_dataset_battery_level(test_files, feature_fns, feature_names, cycle_limit=int(cycle_limit), verbose=verbose, report_missing=report_missing, progress_desc='Building test battery vectors')
    else:
        X_train, y_train = _prepare_dataset_windows(train_files, feature_fns, feature_names, window_size, cycle_limit=cycle_limit, verbose=verbose, report_missing=report_missing, progress_desc='Building train windows')
        X_test, y_test = _prepare_dataset_windows(test_files, feature_fns, feature_names, window_size, cycle_limit=cycle_limit, verbose=verbose, report_missing=report_missing, progress_desc='Building test windows')

    if X_train.size == 0 or X_test.size == 0:
        print("No data available after windowing. Aborting.")
        return

    if battery_level:
        print(f"Train batteries: {X_train.shape}, Test batteries: {X_test.shape}, Feature dim: {X_train.shape[1] if X_train.size else 0}")
    else:
        print(f"Train windows: {X_train.shape}, Test windows: {X_test.shape}, Features: {len(feature_names)} × window {window_size}")

    models = _build_models(use_gpu=use_gpu)
    rows = []
    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        rows.append({'model': name, 'MAE': float(mae), 'RMSE': float(rmse)})
        print(f"  {name}: MAE={mae:.3f} RMSE={rmse:.3f}")

    suffix = (f"_bl_cl{cycle_limit}" if battery_level else f"_ws{window_size}") + ("_cl" if cycle_limit and not battery_level else "")
    pd.DataFrame(rows).to_csv(out_dir / f"rul_metrics{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='RUL prediction with per-cycle features (windowed or battery-level)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'MATR1', 'MATR2', 'CLO', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'], help='Dataset name')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='One or more paths: directories with .pkl or a file listing .pkl paths')
    parser.add_argument('--output_dir', type=str, default='rul_windows', help='Output directory')
    parser.add_argument('--window_size', type=int, default=100, help='Number of past cycles in each window')
    parser.add_argument('--features', type=str, nargs='*', default=['default'],
                        help="Which features to use: 'default', 'all', or list like avg_voltage avg_current ...")
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration for XGBoost if available')
    parser.add_argument('--cycle_limit', type=int, default=None, help='Use only cycles <= this index for features (early-cycle cap)')
    parser.add_argument('--battery_level', action='store_true', help='Enable battery-level RUL: one vector per battery, one scalar label')
    parser.add_argument('--verbose', action='store_true', help='Verbose diagnostics (shapes, NaNs, skips)')
    parser.add_argument('--report_missing', action='store_true', help='Report per-battery, per-cycle missing features (NaNs)')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, args.window_size, features=args.features, use_gpu=args.gpu, cycle_limit=args.cycle_limit, battery_level=args.battery_level, verbose=args.verbose, report_missing=args.report_missing)


if __name__ == '__main__':
    main()


