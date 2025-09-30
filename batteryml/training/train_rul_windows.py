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
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.base import clone

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
# Optional Torch for NN wrappers
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from batteryml.models.rul_predictors.cnn import CNNRULPredictor
    from batteryml.models.rul_predictors.lstm import LSTMRULPredictor
    from batteryml.models.rul_predictors.transformer import TransformerRULPredictor
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

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

    default_features = [
        'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length', 'peak_cv_length',
        'cycle_length', 'power_during_charge_cycle', 'power_during_discharge_cycle',
        'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio'
        'avg_voltage', 'avg_current'
    ]

    print(f'Selected features: {default_features}')
    
    return default_features

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


def _apply_feature_processing(dfn: pd.DataFrame,
                              feature_names: List[str],
                              kind: str = 'none',
                              mm_window: int = 5,
                              verbose: bool = False) -> pd.DataFrame:
    """Apply modular feature processing along cycle axis (sorted by cycle_number).

    - kind: 'none' | 'moving_mean'
    - mm_window: window for moving mean (>=1)
    """
    if kind is None or kind.lower() == 'none':
        return dfn
    dfn = dfn.sort_values('cycle_number').reset_index(drop=True)
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return dfn
    if kind == 'moving_mean':
        w = max(1, int(mm_window or 1))
        try:
            # rolling mean per column along cycles
            dfn[cols] = dfn[cols].rolling(window=w, min_periods=1).mean()
        except Exception as e:
            if verbose:
                print(f"[warn] moving_mean failed (w={w}): {e}")
            return dfn
        return dfn
    # Unknown kind -> no-op
    if verbose:
        print(f"[warn] unknown feature_processing '{kind}', skipping")
    return dfn


def _filter_df_by_cycle_limit(df: pd.DataFrame, cycle_limit: Optional[int], verbose: bool = False) -> pd.DataFrame:
    dfn = df.sort_values('cycle_number').reset_index(drop=True)
    if cycle_limit is not None:
        dfn = dfn[dfn['cycle_number'] <= int(cycle_limit)].reset_index(drop=True)
        if verbose:
            print(f"[filter] cycle_limit={cycle_limit} -> {len(dfn)} rows remaining")
    return dfn


def _make_windows(df: pd.DataFrame, feature_names: List[str], total_rul: int, window_size: int,
                  cycle_limit: Optional[int] = None,
                  verbose: bool = False,
                  feature_processing: str = 'none',
                  fp_mm_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # df must be sorted by cycle_number ascending
    dfn = _filter_df_by_cycle_limit(df, cycle_limit, verbose=verbose)
    # Apply feature processing after cycle filter
    dfn = _apply_feature_processing(dfn, feature_names, kind=feature_processing, mm_window=fp_mm_window, verbose=verbose)
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


def _make_battery_vector(df: pd.DataFrame, feature_names: List[str], cycle_limit: int,
                         verbose: bool = False,
                         feature_processing: str = 'none',
                         fp_mm_window: int = 5) -> np.ndarray:
    dfn = _filter_df_by_cycle_limit(df, cycle_limit, verbose=verbose)
    dfn = _apply_feature_processing(dfn, feature_names, kind=feature_processing, mm_window=fp_mm_window, verbose=verbose)
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


def _prepare_dataset_windows(files: List[Path], feature_fns: Dict[str, callable], feature_names: List[str], window_size: int,
                             cycle_limit: Optional[int], verbose: bool, report_missing: bool, progress_desc: str,
                             feature_processing: str, fp_mm_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[str] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            battery = BatteryData.load(f)
        except Exception:
            continue
        total_rul = _compute_total_rul(battery)
        df = _build_cycle_feature_table(battery, feature_fns)
        if report_missing:
            _report_missing_for_battery(battery.cell_id, df, feature_names, cycle_limit, verbose=verbose)
        Xw, yw = _make_windows(df, feature_names, total_rul, window_size,
                               cycle_limit=cycle_limit, verbose=verbose,
                               feature_processing=feature_processing, fp_mm_window=fp_mm_window)
        if Xw.size and yw.size:
            Xs.append(Xw)
            ys.append(yw)
            groups.extend([battery.cell_id] * Xw.shape[0])
        elif verbose:
            print(f"[skip] {f.name}: produced no windows")
    if not Xs:
        return np.zeros((0, len(feature_names) * window_size), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.array(groups, dtype=object)
    return X, y, g


def _prepare_dataset_battery_level(files: List[Path], feature_fns: Dict[str, callable], feature_names: List[str], cycle_limit: int,
                                   verbose: bool, report_missing: bool, progress_desc: str,
                                   feature_processing: str, fp_mm_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    groups: List[str] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            battery = BatteryData.load(f)
        except Exception:
            continue
        total_rul = _compute_total_rul(battery)
        df = _build_cycle_feature_table(battery, feature_fns)
        if report_missing:
            _report_missing_for_battery(battery.cell_id, df, feature_names, cycle_limit, verbose=verbose)
        vec = _make_battery_vector(df, feature_names, cycle_limit=cycle_limit, verbose=verbose,
                                   feature_processing=feature_processing, fp_mm_window=fp_mm_window)
        if vec.size == 0:
            if verbose:
                print(f"[skip] {f.name}: no features present after filtering")
            continue
        X_list.append(vec)
        y_list.append(float(total_rul))
        groups.append(battery.cell_id)
    if not X_list:
        return np.zeros((0, len(feature_names) * int(cycle_limit)), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    X = np.vstack([x.reshape(1, -1) for x in X_list])
    y = np.array(y_list, dtype=float)
    g = np.array(groups, dtype=object)
    if verbose:
        print(f"[battery_level] X={X.shape}, y={y.shape}")
    return X, y, g


def _build_models(use_gpu: bool = False) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    base_steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler())]

    # Determine if GPU is truly available (even if --gpu passed)
    gpu_available = False
    if use_gpu:
        try:
            import torch  # type: ignore
            gpu_available = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
        except Exception:
            gpu_available = False

    # Prefer cuML implementations only when GPU is available
    if gpu_available and _HAS_CUML:
        from cuml.linear_model import LinearRegression as cuLinearRegression, Ridge as cuRidge, ElasticNet as cuElasticNet
        from cuml.svm import SVR as cuSVR
        from cuml.ensemble import RandomForestRegressor as cuRF
        models['linear_regression'] = Pipeline(base_steps + [('model', cuLinearRegression())])
        models['ridge'] = Pipeline(base_steps + [('model', cuRidge())])
        models['elastic_net'] = Pipeline(base_steps + [('model', cuElasticNet())])
        models['svr'] = Pipeline(base_steps + [('model', cuSVR(kernel='rbf', C=10.0))])
        models['random_forest'] = Pipeline(base_steps + [('model', cuRF(n_estimators=40, random_state=42))])
    else:
        # Linear models (CPU)
        models['linear_regression'] = Pipeline(base_steps + [('model', LinearRegression())])
        models['ridge'] = Pipeline(base_steps + [('model', Ridge(alpha=1.0))])
        models['elastic_net'] = Pipeline(base_steps + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
        # Kernel (CPU)
        models['svr'] = Pipeline(base_steps + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
        # Trees (CPU)
        models['random_forest'] = Pipeline(base_steps + [('model', RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1))])
    # (GradientBoosting removed)
    # Shallow MLP
    models['mlp'] = Pipeline(base_steps + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
    # XGBoost
    if _HAS_XGB:
        # Use CUDA only if GPU is truly available
        xgb_device = 'cuda' if gpu_available else 'cpu'
        models['xgboost'] = Pipeline(base_steps + [('model', XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            device=xgb_device
        ))])
    # PLSR
    models['plsr'] = Pipeline(base_steps + [('model', PLSRegression(n_components=10))])
    # PCR (PCA + Linear)
    models['pcr'] = Pipeline(base_steps + [('model', Pipeline([('pca', PCA(n_components=20)), ('lr', LinearRegression())]))])

    # Optional: GPyTorch SVGP (GPU scalable GP)
    if gpu_available and _HAS_GPYTORCH:
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


class _TorchNNRegressor:
    """Sklearn-like wrapper around Torch NN models (CNN/LSTM/Transformer).

    Assumes input X is [B, F * window_size] flattened in cycle-major order.
    """
    def __init__(self, model_type: str, window_size: int, channels: int = 64, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, use_gpu: bool = False):
        self.model_type = model_type
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.use_gpu = bool(use_gpu)
        self._model = None
        self._device = 'cpu'

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        B, D = X.shape
        if self.window_size <= 0 or D % self.window_size != 0:
            # Fallback: infer window_size as best guess (avoid crash)
            w = self.window_size if self.window_size > 0 else max(1, D)
            F = max(1, D // w)
        else:
            F = D // self.window_size
        # reshape to [B, H, F]
        X3 = X.reshape(B, self.window_size, F)
        # models accept [B, H, F] or [B, 1, H, F]
        return X3

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not _HAS_TORCH:
            # No torch available; behave as a no-op linear fit fallback to avoid runtime error
            self._model = ('noop', np.mean(y))
            self.is_fitted_ = True
            return self
        X3 = self._reshape(np.asarray(X, dtype=np.float32))
        yv = np.asarray(y, dtype=np.float32)
        B, H, F = X3.shape
        in_channels = 1
        input_height = H
        input_width = F
        # Select device
        self._device = 'cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
        # Build model
        if self.model_type == 'cnn':
            # Use kernel_size=1 to avoid aggressive downsampling on narrow feature width
            model = CNNRULPredictor(in_channels=in_channels, channels=self.channels, input_height=input_height, input_width=input_width, kernel_size=1)
        elif self.model_type == 'lstm':
            model = LSTMRULPredictor(in_channels=in_channels, channels=self.channels, input_height=input_height, input_width=input_width)
        else:
            # transformer
            model = TransformerRULPredictor(in_channels=in_channels, channels=self.channels, input_height=input_height, input_width=input_width)
        model = model.to(self._device)
        # Prepare tensors
        X_t = torch.tensor(X3, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(yv, dtype=torch.float32, device=self._device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(max(1, self.epochs)):
            for xb, yb in loader:
                pred = model(xb, yb, return_loss=False)
                loss = torch.mean((pred - yb.view(-1)) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model = model.eval()
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, 'is_fitted_') and not getattr(self, 'is_fitted_', False):
            # Defensive: if somehow predict called before fit
            return np.zeros((X.shape[0],), dtype=float)
        if isinstance(self._model, tuple) and self._model[0] == 'noop':
            return np.full((X.shape[0],), float(self._model[1]), dtype=float)
        if not _HAS_TORCH or self._model is None:
            return np.zeros((X.shape[0],), dtype=float)
        X3 = self._reshape(np.asarray(X, dtype=np.float32))
        Xt = torch.tensor(X3, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            # Dummy labels not used when return_loss=False
            yt = torch.zeros((Xt.shape[0],), dtype=torch.float32, device=self._device)
            out = self._model(Xt, yt, return_loss=False)
        return out.detach().cpu().numpy().astype(float)


def run(dataset: str, data_path: str, output_dir: str, window_size: int, features: Optional[List[str]] = None, use_gpu: bool = False,
        cycle_limit: Optional[int] = None, battery_level: bool = False, verbose: bool = False, report_missing: bool = False,
        tune: bool = False, cv_splits: int = 5,
        feature_processing: str = 'none', fp_mm_window: int = 5):
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
        X_train, y_train, g_train = _prepare_dataset_battery_level(train_files, feature_fns, feature_names, cycle_limit=int(cycle_limit),
                                                                   verbose=verbose, report_missing=report_missing,
                                                                   progress_desc='Building train battery vectors',
                                                                   feature_processing=feature_processing, fp_mm_window=fp_mm_window)
        X_test, y_test, _ = _prepare_dataset_battery_level(test_files, feature_fns, feature_names, cycle_limit=int(cycle_limit),
                                                           verbose=verbose, report_missing=report_missing,
                                                           progress_desc='Building test battery vectors',
                                                           feature_processing=feature_processing, fp_mm_window=fp_mm_window)
    else:
        X_train, y_train, g_train = _prepare_dataset_windows(train_files, feature_fns, feature_names, window_size,
                                                             cycle_limit=cycle_limit, verbose=verbose, report_missing=report_missing,
                                                             progress_desc='Building train windows', feature_processing=feature_processing,
                                                             fp_mm_window=fp_mm_window)
        X_test, y_test, _ = _prepare_dataset_windows(test_files, feature_fns, feature_names, window_size,
                                                     cycle_limit=cycle_limit, verbose=verbose, report_missing=report_missing,
                                                     progress_desc='Building test windows', feature_processing=feature_processing,
                                                     fp_mm_window=fp_mm_window)

    if X_train.size == 0 or X_test.size == 0:
        print("No data available after windowing. Aborting.")
        return

    if battery_level:
        print(f"Train batteries: {X_train.shape}, Test batteries: {X_test.shape}, Feature dim: {X_train.shape[1] if X_train.size else 0}")
    else:
        print(f"Train windows: {X_train.shape}, Test windows: {X_test.shape}, Features: {len(feature_names)} × window {window_size}")

    # ----------------------
    # Label transformations
    # ----------------------
    def _fit_label_transform(y: np.ndarray) -> tuple[np.ndarray, dict]:
        y = np.asarray(y, dtype=float)
        yt = np.log1p(np.clip(y, a_min=0.0, a_max=None))  # log1p
        mu = float(np.mean(yt))
        sigma = float(np.std(yt)) if np.std(yt) > 1e-6 else 1e-6
        yt = (yt - mu) / sigma
        return yt, {'mu': mu, 'sigma': sigma}

    def _inverse_label_transform(yt: np.ndarray, stats: dict) -> np.ndarray:
        mu = stats['mu']; sigma = stats['sigma']
        y_log = yt * sigma + mu
        y = np.expm1(y_log)
        return np.clip(y, a_min=0.0, a_max=None)

    # Fit transform on full training labels for final model
    y_train_t, train_label_stats = _fit_label_transform(y_train)

    models = _build_models(use_gpu=use_gpu)
    # Append NN wrappers at the end if torch is available
    if _HAS_TORCH:
        time_len = window_size if not battery_level else (cycle_limit or window_size)
        models['cnn'] = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler()), ('model', _TorchNNRegressor('cnn', window_size=time_len, channels=32, epochs=10, batch_size=64, lr=1e-3, use_gpu=use_gpu))])
        models['lstm'] = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler()), ('model', _TorchNNRegressor('lstm', window_size=time_len, channels=64, epochs=10, batch_size=64, lr=1e-3, use_gpu=use_gpu))])
        models['transformer'] = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler()), ('model', _TorchNNRegressor('transformer', window_size=time_len, channels=64, epochs=10, batch_size=64, lr=5e-4, use_gpu=use_gpu))])
    rows = []
    for name, pipe in models.items():
        print(f"Training {name}...")
        est = pipe
        best_params = None
        if tune:
            if name == 'linear_regression':
                param_grid = {
                    'model__fit_intercept': [True, False]
                }
            elif name == 'ridge':
                param_grid = {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__fit_intercept': [True, False],
                }
            elif name == 'elastic_net':
                param_grid = {
                    'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'model__max_iter': [5000, 10000, 50000],
                }
            elif name == 'svr':
                param_grid = {
                    'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'model__C': [0.001, 0.01, 0.1, 1.0],
                    'model__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                    'model__epsilon': [0.05, 0.1, 0.2],
                    'model__degree': [2, 3, 4],
                }
            elif name == 'random_forest':
                param_grid = {
                    'model__n_estimators': [200, 400, 800],
                    'model__max_depth': [None, 10, 20, 40, 80],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 'log2', None],
                }
            elif name == 'xgboost':
                param_grid = {
                    'model__n_estimators': [300, 600],
                    'model__max_depth': [4, 6, 8, 10, 20],
                    'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'model__reg_alpha': [0.0, 0.001, 0.01],
                    'model__reg_lambda': [1.0, 5.0, 10.0]
                }
            elif name == 'mlp':
                param_grid = {
                    'model__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
                    'model__alpha': [1e-6, 1e-5, 1e-4],
                    'model__learning_rate_init': [0.0005, 0.001, 0.01],
                    'model__activation': ['relu', 'tanh'],
                    'model__batch_size': [8, 32, 64, 128],
                    'model__max_iter': [300, 900, 2000],
                }
            elif name == 'plsr':
                param_grid = {
                    'model__n_components': [5, 10, 15, 20, 30],
                }
            elif name == 'pcr':
                param_grid = {
                    'model__pca__n_components': [10, 20, 40, 60],
                    'model__lr__fit_intercept': [True, False],
                }
            elif name == 'gpytorch_svgp':
                param_grid = {
                    'model__inducing_points': [256, 512, 1024],
                    'model__batch_size': [512, 1024, 2048],
                    'model__iters': [100, 200],
                    'model__lr': [1e-2, 5e-3, 1e-3],
                }
            elif name == 'cnn':
                param_grid = {
                    'model__channels': [16, 32, 64],
                    'model__epochs': [10, 20],
                    'model__batch_size': [32, 64],
                    'model__lr': [1e-3, 5e-4],
                }
            elif name == 'lstm':
                param_grid = {
                    'model__channels': [32, 64, 128],
                    'model__epochs': [10, 20],
                    'model__batch_size': [32, 64],
                    'model__lr': [1e-3, 5e-4],
                }
            elif name == 'transformer':
                param_grid = {
                    'model__channels': [32, 64, 128],
                    'model__epochs': [10, 20],
                    'model__batch_size': [32, 64],
                    'model__lr': [1e-3, 5e-4],
                }
            else:
                param_grid = {}

            if param_grid:
                unique_groups = np.unique(g_train)
                n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
                cv = GroupKFold(n_splits=n_splits)
                grid = list(ParameterGrid(param_grid))
                from tqdm import tqdm as _tqdm
                results = []
                best_score = np.inf
                best_params = None
                best_estimator = None
                pbar = _tqdm(grid, desc=f"Tuning {name} ({len(grid)} cfgs x {n_splits} folds)")
                for params in pbar:
                    fold_rmses = []
                    fold_maes = []
                    for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                        # Fit label transform on training fold
                        y_tr_t, fold_stats = _fit_label_transform(y_tr)
                        model = clone(pipe)
                        model.set_params(**params)
                        model.fit(X_tr, y_tr_t)
                        pred_t = model.predict(X_va)
                        pred = _inverse_label_transform(np.asarray(pred_t, dtype=float), fold_stats)
                        y_va_arr = np.asarray(y_va, dtype=float)
                        fold_maes.append(mean_absolute_error(y_va_arr, np.asarray(pred, dtype=float)))
                        fold_rmses.append(mean_squared_error(y_va_arr, np.asarray(pred, dtype=float)) ** 0.5)
                    mean_mae = float(np.mean(fold_maes)) if fold_maes else np.inf
                    mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
                    results.append({**{f"param:{k}": v for k, v in params.items()}, 'mean_MAE': mean_mae, 'mean_RMSE': mean_rmse})
                    pbar.set_postfix({"RMSE": f"{mean_rmse:.3f}"})
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_params = params
                        best_estimator = clone(pipe).set_params(**params)
                if best_estimator is not None:
                    est = best_estimator
                # Save CV results
                cv_path = Path(output_dir) / f"{dataset.upper()}_{name}_cv_results.csv"
                pd.DataFrame(results).to_csv(cv_path, index=False)

        # Rebuild estimator with best params (if any) and ALWAYS fit on full train set (transformed labels)
        if best_params is not None:
            est = clone(pipe).set_params(**best_params)
        est.fit(X_train, y_train_t)
        y_pred_t = est.predict(X_test)
        y_pred = _inverse_label_transform(np.asarray(y_pred_t, dtype=float), train_label_stats)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        row = {'model': name, 'MAE': float(mae), 'RMSE': float(rmse)}
        if best_params is not None:
            row.update({f"param:{k}": v for k, v in best_params.items()})
        rows.append(row)
        print(f"  {name}: MAE={mae:.3f} RMSE={rmse:.3f}")

    suffix = (f"_bl_cl{cycle_limit}" if battery_level else f"_ws{window_size}") + ("_cl" if cycle_limit and not battery_level else "")
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(out_dir / f"rul_metrics{suffix}.csv", index=False)
    # Save test errors (MAE, RMSE) under test_results/
    test_results_dir = out_dir / 'test_results'
    test_results_dir.mkdir(exist_ok=True)
    try:
        df_metrics[['model', 'MAE', 'RMSE']].to_csv(test_results_dir / f"test_results{suffix}.csv", index=False)
    except Exception:
        # Fallback: write full metrics if column subset fails
        df_metrics.to_csv(test_results_dir / f"test_results{suffix}.csv", index=False)

    # Update global matrices (one CSV per metric) with models as rows and datasets as columns
    def _update_matrix(csv_path: Path, metric_col: str, dataset_name: str, df: pd.DataFrame):
        series = df.set_index('model')[metric_col]
        if csv_path.exists():
            try:
                mat = pd.read_csv(csv_path, index_col=0)
            except Exception:
                mat = pd.DataFrame()
        else:
            mat = pd.DataFrame()
        # Ensure index union and set/update column
        if mat.empty:
            mat = pd.DataFrame(series)
            mat.columns = [dataset_name]
        else:
            # align by model index
            mat = mat.reindex(mat.index.union(series.index))
            mat[dataset_name] = series.reindex(mat.index)
        mat.to_csv(csv_path)

    # Place matrices at the parent of output_dir so they aggregate across datasets
    aggregate_root = out_dir.parent
    aggregate_root.mkdir(parents=True, exist_ok=True)
    _update_matrix(aggregate_root / 'MAE.csv', 'MAE', dataset, df_metrics)
    _update_matrix(aggregate_root / 'RMSE.csv', 'RMSE', dataset, df_metrics)


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
    parser.add_argument('--tune', action='store_true', help='Enable GridSearchCV with GroupKFold on training set')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of GroupKFold splits for tuning (default 5)')
    parser.add_argument('--feature_processing', type=str, default='none', choices=['none', 'moving_mean'], help='Feature processing to apply before flattening')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, args.window_size, features=args.features, use_gpu=args.gpu,
        cycle_limit=args.cycle_limit, battery_level=args.battery_level, verbose=args.verbose, report_missing=args.report_missing,
        tune=args.tune, cv_splits=args.cv_splits, feature_processing=args.feature_processing, fp_mm_window=5)


if __name__ == '__main__':
    main()


