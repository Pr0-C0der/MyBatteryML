from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Dict
from scipy.signal import medfilt, savgol_filter
from tqdm import tqdm

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import (
    get_extractor_class,
    DatasetSpecificCycleFeatures,
)


# Per-cycle scalar feature computed via extractor method
@dataclass
class CycleScalarFeature:
    name: str
    method_name: str
    depends_on: List[str] = field(default_factory=list)
    description: Optional[str] = None


class PerBatteryCorrelationAnalyzer:
    """Correlation analyzer for battery-level RUL prediction using early-cycle features."""

    def __init__(self, data_path: str, output_dir: str = 'per_battery_correlation_analysis', verbose: bool = False, dataset_hint: Optional[str] = None, cycle_limit: Optional[int] = None, smoothing: str = 'none', ma_window: int = 5):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.cycle_limit = cycle_limit
        self.smoothing = str(smoothing or 'none').lower()
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 5
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # subdirs
        # Chemistry subfolder name inferred from data_path
        self._chemistry_name = Path(data_path).name
        self.feature_box_dir = self.output_dir / self._chemistry_name / 'feature_rul_boxplots'
        self.feature_box_dir.mkdir(exist_ok=True)

        self.features: Dict[str, CycleScalarFeature] = {}
        self.rul_annotator = RULLabelAnnotator()

    @staticmethod
    def _safe_filename(name: str) -> str:
        invalid = '<>:"/\\|?*'
        s = ''.join(('_' if ch in invalid else ch) for ch in str(name))
        s = s.strip().replace(' ', '_')
        s = ''.join(ch for ch in s if ch.isprintable())
        return s or 'unknown'

    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            kernel = np.ones(w, dtype=float)
            mask = np.isfinite(arr).astype(float)
            arr0 = np.nan_to_num(arr, nan=0.0)
            num = np.convolve(arr0, kernel, mode='same')
            den = np.convolve(mask, kernel, mode='same')
            out = num / np.maximum(den, 1e-8)
            out[den < 1e-8] = np.nan
            return out
        except Exception:
            return y

    @staticmethod
    def _moving_median(y: np.ndarray, window: int) -> np.ndarray:
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            if w < 1:
                return arr
            pad = w // 2
            padded = np.pad(arr, (pad, pad), mode='edge')
            out = np.empty_like(arr)
            for i in range(arr.size):
                seg = padded[i:i + w]
                m = np.isfinite(seg)
                out[i] = np.nanmedian(seg[m]) if np.any(m) else np.nan
            return out
        except Exception:
            return y

    @staticmethod
    def _hampel_filter(y: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        n = arr.size
        if n == 0 or window_size < 3:
            return arr
        w = int(window_size)
        half = w // 2
        out = arr.copy()
        for i in range(n):
            l = max(0, i - half)
            r = min(n, i + half + 1)
            seg = arr[l:r]
            m = np.isfinite(seg)
            if not np.any(m):
                continue
            med = float(np.nanmedian(seg[m]))
            mad = float(np.nanmedian(np.abs(seg[m] - med)))
            if mad <= 0:
                continue
            thr = n_sigmas * 1.4826 * mad
            if np.isfinite(arr[i]) and abs(arr[i] - med) > thr:
                out[i] = med
        return out

    @staticmethod
    def _hms_filter(y: np.ndarray) -> np.ndarray:
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size == 0:
                return arr
            # 1) Hampel
            h = PerBatteryCorrelationAnalyzer._hampel_filter(arr, window_size=11, n_sigmas=3.0)
            # 2) Median filter (size=5)
            try:
                m = medfilt(h, kernel_size=5)
            except Exception:
                m = h
            # 3) Savitzky–Golay (window_length=101, polyorder=3), adjusted to length
            wl = 101
            if m.size < wl:
                wl = m.size if m.size % 2 == 1 else max(1, m.size - 1)
            if wl >= 5 and wl > 3:
                try:
                    s = savgol_filter(m, window_length=wl, polyorder=3, mode='interp')
                except Exception:
                    s = m
            else:
                s = m
            return s
        except Exception:
            return y

    @staticmethod
    def _compute_derivative(y: np.ndarray) -> np.ndarray:
        """Compute first derivative using finite differences"""
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size < 2:
                return np.zeros_like(arr)
            # Use numpy gradient for better numerical stability
            return np.gradient(arr)
        except Exception:
            return np.zeros_like(y)

    @staticmethod
    def _compute_double_derivative(y: np.ndarray) -> np.ndarray:
        """Compute second derivative using finite differences"""
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size < 3:
                return np.zeros_like(arr)
            # Compute second derivative as gradient of gradient
            first_deriv = np.gradient(arr)
            return np.gradient(first_deriv)
        except Exception:
            return np.zeros_like(y)

    # -------------
    # Dataset logic
    # -------------
    def _infer_dataset_from_path(self, p: Path) -> Optional[str]:
        name = p.as_posix().upper()
        candidates = [
            'MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX'
        ]
        for key in candidates:
            if f"/{key}/" in name or name.endswith(f"/{key}.PKL") or (f"_{key}_" in name) or (f"-{key}-" in name) or (f"/{key}_" in name):
                return key
        return None

    def _infer_dataset(self, src: Path, battery: BatteryData) -> Optional[str]:
        if self.dataset_hint:
            return str(self.dataset_hint).upper()
        key = self._infer_dataset_from_path(src)
        if key:
            return key
        cid = str(getattr(battery, 'cell_id', '')).upper()
        for key in ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']:
            if key in cid:
                return key
        for attr in ['reference', 'description']:
            txt = str(getattr(battery, attr, '')).upper()
            for key in ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']:
                if key in txt:
                    return key
        return None

    def _get_extractor(self, src: Path, battery: BatteryData) -> Optional[DatasetSpecificCycleFeatures]:
        ds = self._infer_dataset(src, battery)
        if ds is None:
            if self.verbose:
                print(f"[warn] unable to infer dataset for {src.name}; skipping chemistry-specific features")
            return None
        cls = get_extractor_class(ds)
        if cls is None:
            if self.verbose:
                print(f"[warn] no extractor class registered for dataset '{ds}'")
            return None
        return cls()

    # -----------
    # Registration
    # -----------
    def register_feature(self, spec: CycleScalarFeature):
        self.features[spec.name] = spec
        if self.verbose:
            print(f"[register] feature: {spec.name}")

    # --------------
    # Data utilities
    # --------------
    def _battery_files(self) -> List[Path]:
        return list(self.data_path.glob('*.pkl'))

    def _compute_total_rul(self, battery: BatteryData) -> int:
        try:
            rul_tensor = self.rul_annotator.process_cell(battery)
            v = int(rul_tensor.item())
            return v if np.isfinite(v) else 0
        except Exception:
            return 0

    # ----------------------
    # Matrix / plot builders
    # ----------------------
    def build_battery_feature_matrix(self, src: Path, battery: BatteryData, extractor: Optional[DatasetSpecificCycleFeatures]) -> pd.DataFrame:
        """Build feature matrix for battery-level analysis using early cycles"""
        data: List[Dict[str, float]] = []
        total_rul = self._compute_total_rul(battery)
        
        # Apply cycle limit if specified
        cycles_to_process = battery.cycle_data
        if self.cycle_limit is not None and self.cycle_limit > 0:
            cycles_to_process = cycles_to_process[:self.cycle_limit]
        
        # Extract feature values for each cycle
        feature_series = {}
        for name, spec in self.features.items():
            feature_series[name] = []
            if extractor is not None:
                for c in cycles_to_process:
                    try:
                        fn = getattr(extractor, spec.method_name, None)
                        val = fn(battery, c) if fn is not None else None
                        if val is None:
                            feature_series[name].append(np.nan)
                        else:
                            f = float(val)
                            feature_series[name].append(f if np.isfinite(f) else np.nan)
                    except Exception:
                        feature_series[name].append(np.nan)
            else:
                feature_series[name] = [np.nan] * len(cycles_to_process)
        
        # Convert to numpy arrays and apply smoothing
        for name in feature_series:
            series = np.array(feature_series[name])
            if self.smoothing == 'ma':
                smoothed = self._moving_average(series, self.ma_window)
            elif self.smoothing == 'median':
                smoothed = self._moving_median(series, self.ma_window)
            elif self.smoothing == 'hms':
                smoothed = self._hms_filter(series)
            else:
                smoothed = series
            feature_series[name] = smoothed
        
        # Create DataFrame with features, derivatives, and double derivatives
        df_data = {'total_rul': total_rul}
        
        for name, series in feature_series.items():
            # Original feature
            df_data[name] = np.nanmean(series) if len(series) > 0 else np.nan
            
            # First derivative
            deriv = self._compute_derivative(series)
            df_data[f'{name}_derivative'] = np.nanmean(deriv) if len(deriv) > 0 else np.nan
            
            # Second derivative
            double_deriv = self._compute_double_derivative(series)
            df_data[f'{name}_double_derivative'] = np.nanmean(double_deriv) if len(double_deriv) > 0 else np.nan
        
        return pd.DataFrame([df_data])

    def plot_feature_rul_correlation_boxplot(self, all_battery_data: List[tuple]):
        """Generate boxplot for battery-level feature-RUL correlations"""
        rows = []
        
        for cell_id, df in all_battery_data:
            if df.empty or 'total_rul' not in df.columns:
                continue
                
            total_rul = df['total_rul'].iloc[0]
            if not np.isfinite(total_rul) or total_rul <= 0:
                continue
            
            # Calculate correlations for each feature type (original, derivative, double derivative)
            for feature_name in self.features.keys():
                # Original feature
                if feature_name in df.columns:
                    val = df[feature_name].iloc[0]
                    if np.isfinite(val):
                        rows.append({'feature': feature_name, 'corr': float(val), 'type': 'original'})
                
                # First derivative
                deriv_name = f'{feature_name}_derivative'
                if deriv_name in df.columns:
                    val = df[deriv_name].iloc[0]
                    if np.isfinite(val):
                        rows.append({'feature': feature_name, 'corr': float(val), 'type': 'derivative'})
                
                # Second derivative
                double_deriv_name = f'{feature_name}_double_derivative'
                if double_deriv_name in df.columns:
                    val = df[double_deriv_name].iloc[0]
                    if np.isfinite(val):
                        rows.append({'feature': feature_name, 'corr': float(val), 'type': 'double_derivative'})
        
        if not rows:
            if self.verbose:
                print("[warn] No correlation data found for boxplot")
            return
            
        df = pd.DataFrame(rows)
        
        # Create separate boxplots for each type
        for plot_type in ['original', 'derivative', 'double_derivative']:
            type_data = df[df['type'] == plot_type]
            if type_data.empty:
                continue
                
            num_feats = type_data['feature'].nunique()
            fig_w = max(12, min(1.0 * num_feats, 36))
            plt.figure(figsize=(fig_w, 6))
            
            sns.boxplot(data=type_data, x='feature', y='corr')
            plt.axhline(0.0, color='0.5', linewidth=1)
            plt.ylabel(f'Feature Value ({plot_type})')
            plt.xlabel('Feature')
            plt.title(f'Battery-Level Feature Values ({plot_type.title()}) vs Total RUL')
            plt.xticks(rotation=45, ha='right')
            
            out_path = self.feature_box_dir / f'battery_level_feature_{plot_type}_boxplot.png'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f"[ok] {plot_type.title()} boxplot saved to {out_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[warn] Failed to save {plot_type} boxplot: {e}")
            finally:
                plt.close()

    # ---------------------
    # High-level operations
    # ---------------------
    def analyze_dataset(self):
        files = self._battery_files()
        if self.verbose:
            print(f"Found {len(files)} battery files under {self.data_path}")
        
        # Store data for boxplot generation
        all_battery_data = []
        
        # Process each battery with progress bar
        for f in tqdm(files, desc="Processing batteries", unit="battery"):
            try:
                b = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            
            # Store data for boxplot generation
            extractor = self._get_extractor(f, b)
            df = self.build_battery_feature_matrix(f, b, extractor)
            all_battery_data.append((b.cell_id, df))

        # Generate boxplot
        try:
            if self.verbose:
                print("Generating battery-level feature correlation boxplot...")
            self.plot_feature_rul_correlation_boxplot(all_battery_data)
        except Exception as e:
            if self.verbose:
                print(f"[warn] failed feature boxplot: {e}")


def register_default_features(analyzer: PerBatteryCorrelationAnalyzer):
    specs = [
        CycleScalarFeature('avg_voltage', 'avg_voltage', description='Mean voltage per cycle'),
        CycleScalarFeature('avg_current', 'avg_current', description='Mean current per cycle'),
        CycleScalarFeature('avg_c_rate', 'avg_c_rate', description='Average |I|/C per cycle'),
        CycleScalarFeature('max_discharge_capacity', 'max_discharge_capacity'),
        CycleScalarFeature('max_charge_capacity', 'max_charge_capacity'),
        CycleScalarFeature('charge_cycle_length', 'charge_cycle_length'),
        CycleScalarFeature('discharge_cycle_length', 'discharge_cycle_length'),
        CycleScalarFeature('cycle_length', 'cycle_length'),
        CycleScalarFeature('energy_during_charge_cycle', 'power_during_charge_cycle'),
        CycleScalarFeature('energy_during_discharge_cycle', 'power_during_discharge_cycle'),
        CycleScalarFeature('avg_charge_c_rate', 'avg_charge_c_rate'),
        CycleScalarFeature('max_charge_c_rate', 'max_charge_c_rate'),
        CycleScalarFeature('avg_discharge_c_rate', 'avg_discharge_c_rate'),
        CycleScalarFeature('charge_to_discharge_time_ratio', 'charge_to_discharge_time_ratio'),
    ]
    for spec in specs:
        analyzer.register_feature(spec)


def build_default_analyzer(data_path: str, output_dir: str = 'per_battery_correlation_analysis', verbose: bool = False, dataset_hint: Optional[str] = None, cycle_limit: Optional[int] = None, smoothing: str = 'none', ma_window: int = 5) -> PerBatteryCorrelationAnalyzer:
    analyzer = PerBatteryCorrelationAnalyzer(data_path, output_dir, verbose=verbose, dataset_hint=dataset_hint, cycle_limit=cycle_limit, smoothing=smoothing, ma_window=ma_window)
    register_default_features(analyzer)
    return analyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Battery-level correlation analysis for RUL prediction')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing *.pkl (e.g., a chemistry subfolder)')
    parser.add_argument('--output_dir', type=str, default='per_battery_correlation_analysis', help='Output directory for correlation outputs')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--cycle_limit', type=int, default=None, help='Limit analysis to first N cycles (None for all cycles)')
    parser.add_argument('--smoothing', type=str, default='none', choices=['none', 'ma', 'median', 'hms'], help='Smoothing method for feature data')
    parser.add_argument('--ma_window', type=int, default=5, help='Window size for moving average/median smoothing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    analyzer = build_default_analyzer(args.data_path, args.output_dir, verbose=args.verbose, dataset_hint=args.dataset_hint, cycle_limit=args.cycle_limit, smoothing=args.smoothing, ma_window=args.ma_window)
    analyzer.analyze_dataset()


if __name__ == '__main__':
    main()
