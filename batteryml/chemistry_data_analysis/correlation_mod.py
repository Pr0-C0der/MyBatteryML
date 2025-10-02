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


class ChemistryCorrelationAnalyzer:
    """Correlation analyzer using chemistry-aware cycle feature extractors."""

    def __init__(self, data_path: str, output_dir: str = 'chemistry_correlation_analysis_mod', verbose: bool = False, dataset_hint: Optional[str] = None, cycle_limit: Optional[int] = None, smoothing: str = 'none', ma_window: int = 5, boxplot_only: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.cycle_limit = cycle_limit
        self.smoothing = str(smoothing or 'none').lower()
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 5
        self.boxplot_only = bool(boxplot_only)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # subdirs
        # Chemistry subfolder name inferred from data_path
        self._chemistry_name = Path(data_path).name
        self.heatmaps_dir = self.output_dir / self._chemistry_name / 'heatmaps'
        self.matrices_dir = self.output_dir / self._chemistry_name / 'matrices'
        self.rul_bars_dir = self.output_dir / self._chemistry_name / 'rul_barh'
        self.feature_vs_batt_dir = self.output_dir / self._chemistry_name / 'feature_vs_batteries'
        self.feature_box_dir = self.output_dir / self._chemistry_name / 'feature_rul_boxplots'
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.matrices_dir.mkdir(exist_ok=True)
        self.rul_bars_dir.mkdir(exist_ok=True)
        self.feature_vs_batt_dir.mkdir(exist_ok=True)
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
            h = ChemistryCorrelationAnalyzer._hampel_filter(arr, window_size=11, n_sigmas=3.0)
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
    def build_cycle_feature_matrix(self, src: Path, battery: BatteryData, extractor: Optional[DatasetSpecificCycleFeatures]) -> pd.DataFrame:
        data: List[Dict[str, float]] = []
        total_rul = self._compute_total_rul(battery)
        
        # Apply cycle limit if specified
        cycles_to_process = battery.cycle_data
        if self.cycle_limit is not None and self.cycle_limit > 0:
            cycles_to_process = cycles_to_process[:self.cycle_limit]
        
        for idx, c in enumerate(cycles_to_process):
            row: Dict[str, float] = {
                'cycle_number': c.cycle_number,
                'rul': max(0, total_rul - idx)
            }
            if extractor is not None:
                for name, spec in self.features.items():
                    try:
                        fn = getattr(extractor, spec.method_name, None)
                        val = fn(battery, c) if fn is not None else None
                        if val is None:
                            row[name] = np.nan
                        else:
                            f = float(val)
                            row[name] = f if np.isfinite(f) else np.nan
                    except Exception:
                        row[name] = np.nan
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Apply smoothing to feature columns (not cycle_number or rul)
        if self.smoothing != 'none' and len(df) > 1:
            feature_cols = [col for col in df.columns if col not in ['cycle_number', 'rul']]
            for col in feature_cols:
                if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    series = df[col].values
                    if self.smoothing == 'ma':
                        smoothed = self._moving_average(series, self.ma_window)
                    elif self.smoothing == 'median':
                        smoothed = self._moving_median(series, self.ma_window)
                    elif self.smoothing == 'hms':
                        smoothed = self._hms_filter(series)
                    else:
                        smoothed = series
                    df[col] = smoothed
        
        return df

    def _save_matrix(self, battery: BatteryData, df: pd.DataFrame):
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        df.to_csv(self.matrices_dir / f"{safe_id}_cycle_feature_matrix.csv", index=False)

    def plot_correlation_heatmap(self, battery: BatteryData, df: pd.DataFrame):
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return
        corr = numeric.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8}, annot_kws={'size': 8})
        plt.title(f'Feature Correlation Matrix - {battery.cell_id}', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        safe_id = self._safe_filename(battery.cell_id)
        out_path = self.heatmaps_dir / f"{safe_id}_correlation_heatmap.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()

    def plot_rul_barh(self, battery: BatteryData, df: pd.DataFrame):
        if 'rul' not in df.columns:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        if 'rul' not in corr.columns:
            return
        series = corr['rul'].drop('rul')
        series = series.sort_values(key=lambda x: np.abs(x), ascending=False)
        colors = ['red' if v < 0 else 'blue' for v in series.values]
        plt.figure(figsize=(10, max(6, len(series) * 0.35)))
        y_pos = np.arange(len(series))
        plt.barh(y_pos, series.values, color=colors, alpha=0.85)
        plt.yticks(y_pos, series.index)
        plt.xlabel('Correlation with RUL')
        plt.title(f'Feature correlations with RUL - {battery.cell_id}')
        plt.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(series.values):
            plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right')
        safe_id = self._safe_filename(battery.cell_id)
        out_path = self.rul_bars_dir / f"{safe_id}_rul_correlations.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()

    # ---------------------
    # High-level operations
    # ---------------------
    def analyze_battery(self, src: Path, battery: BatteryData):
        extractor = self._get_extractor(src, battery)
        df = self.build_cycle_feature_matrix(src, battery, extractor)
        
        if not self.boxplot_only:
            self._save_matrix(battery, df)
            self.plot_correlation_heatmap(battery, df)
            self.plot_rul_barh(battery, df)
        
        if self.verbose:
            print(f"[ok] analyzed {battery.cell_id} -> features: {list(self.features.keys())}")

    def analyze_dataset(self):
        files = self._battery_files()
        if self.verbose:
            print(f"Found {len(files)} battery files under {self.data_path}")
        
        # Process each battery with progress bar
        for f in tqdm(files, desc="Processing batteries", unit="battery"):
            try:
                b = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            self.analyze_battery(f, b)

        if not self.boxplot_only:
            # After per-battery analysis, generate feature-vs-battery correlation plots
            for feature in tqdm(self.features.keys(), desc="Generating feature-vs-battery plots", unit="feature"):
                try:
                    self.plot_feature_rul_correlation_across_batteries(feature)
                except Exception as e:
                    if self.verbose:
                        print(f"[warn] failed feature_vs_batteries for {feature}: {e}")

        # Boxplot across features (correlation with RUL distributions)
        try:
            if self.verbose:
                print("Generating feature correlation boxplot...")
            self.plot_feature_rul_correlation_boxplot()
        except Exception as e:
            if self.verbose:
                print(f"[warn] failed feature boxplot: {e}")

    # ------------------------------
    # Across-battery correlation plot
    # ------------------------------
    def _compute_feature_rul_corr(self, df: pd.DataFrame, feature: str) -> Optional[float]:
        if 'rul' not in df.columns or feature not in df.columns:
            return None
        cols = df[["rul", feature]].copy()
        sub = cols.replace([np.inf, -np.inf], np.nan).dropna()
        if sub.shape[0] < 2 or sub['rul'].nunique() < 2 or sub[feature].nunique() < 2:
            return None
        try:
            corr = sub.corr().loc['rul', feature]
            return float(corr) if np.isfinite(corr) else None
        except Exception:
            return None

    def _collect_feature_corrs(self, feature: str) -> (List[str], List[float]):
        names: List[str] = []
        corrs: List[float] = []
        csv_files = list(self.matrices_dir.glob("*_cycle_feature_matrix.csv"))
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            val = self._compute_feature_rul_corr(df, feature)
            if val is None:
                continue
            stem = csv.stem
            if stem.endswith('_cycle_feature_matrix'):
                name = stem[:-len('_cycle_feature_matrix')]
            else:
                name = stem
            names.append(name)
            corrs.append(val)
        return names, corrs

    def plot_feature_rul_correlation_across_batteries(self, feature: str, chunk_size: int = 40):
        names, corrs = self._collect_feature_corrs(feature)
        if not names:
            return
        order = np.argsort(np.array(corrs))
        names = [names[i] for i in order]
        corrs = [corrs[i] for i in order]

        mean_val = float(np.mean(corrs))
        std_val = float(np.std(corrs))
        median_val = float(np.median(corrs))

        total = len(names)
        pages = int(np.ceil(total / float(chunk_size)))
        for p in range(pages):
            start = p * chunk_size
            end = min(total, (p + 1) * chunk_size)
            n_chunk = end - start
            x = np.arange(n_chunk)
            n_chunk_names = names[start:end]
            n_chunk_corrs = corrs[start:end]
            colors = ['red' if v < 0 else ('blue' if v > 0 else 'gray') for v in n_chunk_corrs]

            plt.figure(figsize=(max(12, min(0.6 * n_chunk, 28)), 6))
            plt.plot(range(n_chunk), n_chunk_corrs, color='0.4', linewidth=1.5, alpha=0.9)
            plt.scatter(range(n_chunk), n_chunk_corrs, c=colors, s=30, zorder=3)
            plt.axhline(mean_val, color='green', linestyle='--', linewidth=1.1, label=f'Mean {mean_val:.3f}')
            plt.axhline(median_val, color='purple', linestyle=':', linewidth=1.1, label=f'Median {median_val:.3f}')
            plt.fill_between(range(n_chunk), mean_val - std_val, mean_val + std_val, color='green', alpha=0.08, label='±1σ')
            plt.xticks(range(n_chunk), n_chunk_names, rotation=45, ha='right')
            plt.ylabel(f'Correlation of {feature} with RUL')
            plt.xlabel('Battery')
            title_suffix = f" (part {p+1}/{pages})" if pages > 1 else ""
            plt.title(f'{feature}–RUL Correlation Across Batteries (sorted){title_suffix}')
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(loc='best', fontsize=9)
            fname = f'{feature}_corr_vs_batteries' + (f'_part{p+1}' if pages > 1 else '') + '.png'
            (self.feature_vs_batt_dir).mkdir(parents=True, exist_ok=True)
            try:
                plt.tight_layout()
                plt.savefig(self.feature_vs_batt_dir / fname, dpi=300, bbox_inches='tight')
            except Exception:
                pass
            finally:
                plt.close()

    def plot_feature_rul_correlation_boxplot(self):
        rows = []
        for feature in self.features.keys():
            _, corrs = self._collect_feature_corrs(feature)
            for v in corrs:
                if np.isfinite(v):
                    rows.append({'feature': feature, 'corr': float(v)})
        if not rows:
            return
        df = pd.DataFrame(rows)
        num_feats = df['feature'].nunique()
        fig_w = max(12, min(1.0 * num_feats, 36))
        plt.figure(figsize=(max(12, min(1.0 * num_feats, 36)), 6))
        sns.boxplot(data=df, x='feature', y='corr')
        plt.axhline(0.0, color='0.5', linewidth=1)
        plt.ylabel('Correlation with RUL')
        plt.xlabel('Feature')
        plt.title('Feature–RUL Correlation Distributions Across Batteries')
        plt.xticks(rotation=45, ha='right')
        out_path = self.feature_box_dir / 'feature_rul_correlation_boxplot.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()


def register_default_features(analyzer: ChemistryCorrelationAnalyzer):
    specs = [
        CycleScalarFeature('avg_voltage', 'avg_voltage', description='Mean voltage per cycle'),
        CycleScalarFeature('avg_current', 'avg_current', description='Mean current per cycle'),
        CycleScalarFeature('avg_c_rate', 'avg_c_rate', description='Average |I|/C per cycle'),
        CycleScalarFeature('max_discharge_capacity', 'max_discharge_capacity'),
        CycleScalarFeature('max_charge_capacity', 'max_charge_capacity'),
        CycleScalarFeature('charge_cycle_length', 'charge_cycle_length'),
        CycleScalarFeature('discharge_cycle_length', 'discharge_cycle_length'),
        # peak_cv_length intentionally omitted for now
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


def build_default_analyzer(data_path: str, output_dir: str = 'chemistry_correlation_analysis_mod', verbose: bool = False, dataset_hint: Optional[str] = None, cycle_limit: Optional[int] = None, smoothing: str = 'none', ma_window: int = 5, boxplot_only: bool = False) -> ChemistryCorrelationAnalyzer:
    analyzer = ChemistryCorrelationAnalyzer(data_path, output_dir, verbose=verbose, dataset_hint=dataset_hint, cycle_limit=cycle_limit, smoothing=smoothing, ma_window=ma_window, boxplot_only=boxplot_only)
    register_default_features(analyzer)
    return analyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chemistry-aware correlation analysis for battery data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing *.pkl (e.g., a chemistry subfolder)')
    parser.add_argument('--output_dir', type=str, default='chemistry_correlation_analysis_mod', help='Output directory for correlation outputs')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--cycle_limit', type=int, default=None, help='Limit analysis to first N cycles (None for all cycles)')
    parser.add_argument('--smoothing', type=str, default='none', choices=['none', 'ma', 'median', 'hms'], help='Smoothing method for feature data')
    parser.add_argument('--ma_window', type=int, default=5, help='Window size for moving average/median smoothing')
    parser.add_argument('--boxplot_only', action='store_true', help='Only generate feature correlation boxplot (skip individual battery analysis)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    analyzer = build_default_analyzer(args.data_path, args.output_dir, verbose=args.verbose, dataset_hint=args.dataset_hint, cycle_limit=args.cycle_limit, smoothing=args.smoothing, ma_window=args.ma_window, boxplot_only=args.boxplot_only)
    analyzer.analyze_dataset()


if __name__ == '__main__':
    main()


