# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

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

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.data_analysis.cycle_features import (
    avg_c_rate,
    max_temperature,
    max_discharge_capacity,
    max_charge_capacity,
    charge_cycle_length,
    discharge_cycle_length,
    peak_cc_length,
    peak_cv_length,
    cycle_length,
    power_during_charge_cycle,
    power_during_discharge_cycle,
    avg_charge_c_rate,
    avg_discharge_c_rate,
    charge_to_discharge_time_ratio,
)


# -----------------------------
# Feature specification objects
# -----------------------------

# Per-cycle scalar feature: returns Optional[float] for each cycle
ComputeCycleFn = Callable[[BatteryData, object], Optional[float]]


@dataclass
class CycleScalarFeature:
    name: str
    compute: ComputeCycleFn
    depends_on: List[str] = field(default_factory=list)
    description: Optional[str] = None


class ModularCorrelationAnalyzer:
    """Modular correlation analyzer with pluggable per-cycle scalar features.

    - Register features (existing or derived) via CycleScalarFeature
    - Builds per-battery cycle-feature matrices (always includes 'rul' and 'cycle_number')
    - Produces correlation heatmaps and barh plots of correlation with RUL
    """

    def __init__(self, data_path: str, output_dir: str = 'correlation_analysis_mod', verbose: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # subdirs
        self.heatmaps_dir = self.output_dir / 'heatmaps'
        self.matrices_dir = self.output_dir / 'matrices'
        self.rul_bars_dir = self.output_dir / 'rul_barh'
        self.feature_vs_batt_dir = self.output_dir / 'feature_vs_batteries'
        self.feature_box_dir = self.output_dir / 'feature_rul_boxplots'
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.matrices_dir.mkdir(exist_ok=True)
        self.rul_bars_dir.mkdir(exist_ok=True)
        self.feature_vs_batt_dir.mkdir(exist_ok=True)
        self.feature_box_dir.mkdir(exist_ok=True)

        self.features: Dict[str, CycleScalarFeature] = {}
        self.rul_annotator = RULLabelAnnotator()

    # -----------
    # Registration
    # -----------
    def register_feature(self, spec: CycleScalarFeature):
        self.features[spec.name] = spec
        if self.verbose:
            print(f"[register] feature: {spec.name}")

    def register_attr_mean_feature(self, name: str, attr: str, description: Optional[str] = None):
        """Convenience: feature is the per-cycle mean of a CycleData attribute array or scalar."""
        def compute(b: BatteryData, c) -> Optional[float]:
            val = getattr(c, attr, None)
            if val is None:
                return None
            try:
                if np.isscalar(val):
                    f = float(val)
                    return f if np.isfinite(f) else None
                arr = np.array(val)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return None
                return float(np.mean(arr))
            except Exception:
                return None
        self.register_feature(CycleScalarFeature(name=name, compute=compute, depends_on=[attr], description=description))

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
    def build_cycle_feature_matrix(self, battery: BatteryData) -> pd.DataFrame:
        data: List[Dict[str, float]] = []
        total_rul = self._compute_total_rul(battery)
        for idx, c in enumerate(battery.cycle_data):
            row: Dict[str, float] = {
                'cycle_number': c.cycle_number,
                'rul': max(0, total_rul - idx)
            }
            # Registered features
            for name, spec in self.features.items():
                try:
                    val = spec.compute(battery, c)
                    if val is None:
                        row[name] = np.nan
                        if self.verbose:
                            print(f"[nan] {battery.cell_id} cycle {c.cycle_number}: feature '{name}' returned None")
                    else:
                        f = float(val)
                        if np.isfinite(f):
                            row[name] = f
                        else:
                            row[name] = np.nan
                            if self.verbose:
                                print(f"[inf/nan] {battery.cell_id} cycle {c.cycle_number}: feature '{name}' value not finite ({val})")
                except Exception:
                    row[name] = np.nan
                    if self.verbose:
                        print(f"[error] {battery.cell_id} cycle {c.cycle_number}: feature '{name}' compute raised exception")
            data.append(row)
        return pd.DataFrame(data)

    def _save_matrix(self, battery: BatteryData, df: pd.DataFrame):
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        df.to_csv(self.matrices_dir / f"{safe_id}_cycle_feature_matrix.csv", index=False)

    def plot_correlation_heatmap(self, battery: BatteryData, df: pd.DataFrame):
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            if self.verbose:
                print(f"[skip] {battery.cell_id}: not enough numeric columns for heatmap")
            return
        corr = numeric.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                    cbar_kws={"shrink": .8}, annot_kws={'size': 8})
        plt.title(f'Feature Correlation Matrix - {battery.cell_id}', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        plt.savefig(self.heatmaps_dir / f"{safe_id}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_rul_barh(self, battery: BatteryData, df: pd.DataFrame):
        if 'rul' not in df.columns:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        if 'rul' not in corr.columns:
            return
        series = corr['rul'].drop('rul')
        # sort by absolute magnitude desc
        series = series.sort_values(key=lambda x: np.abs(x), ascending=False)
        plt.figure(figsize=(10, max(6, len(series) * 0.35)))
        colors = ['red' if v < 0 else 'blue' for v in series.values]
        bars = plt.barh(range(len(series)), series.values, color=colors, alpha=0.8)
        plt.yticks(range(len(series)), series.index)
        plt.xlabel('Correlation with RUL')
        plt.title(f'Feature correlations with RUL - {battery.cell_id}')
        plt.grid(True, alpha=0.3, axis='x')
        for i, (bar, v) in enumerate(zip(bars, series.values)):
            plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right')
        plt.tight_layout()
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        plt.savefig(self.rul_bars_dir / f"{safe_id}_rul_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------
    # High-level operations
    # ---------------------
    def analyze_battery(self, battery: BatteryData):
        df = self.build_cycle_feature_matrix(battery)
        self._save_matrix(battery, df)
        self.plot_correlation_heatmap(battery, df)
        self.plot_rul_barh(battery, df)
        if self.verbose:
            print(f"[ok] analyzed {battery.cell_id} -> features: {list(self.features.keys())}")

    def analyze_dataset(self):
        files = self._battery_files()
        if self.verbose:
            print(f"Found {len(files)} battery files under {self.data_path}")
        for f in files:
            try:
                b = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            self.analyze_battery(b)

        # After per-battery analysis, generate feature-vs-battery correlation plots
        for feature in self.features.keys():
            try:
                self.plot_feature_rul_correlation_across_batteries(feature)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] failed feature_vs_batteries for {feature}: {e}")

        # Boxplot across features (correlation with RUL distributions)
        try:
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
        # Report non-finite values if verbose
        if self.verbose:
            n_inf = int((~np.isfinite(cols.values)).sum())
            n_nan = int(np.isnan(cols.values).sum())
            if n_inf or n_nan:
                print(f"[warn] non-finite values in columns [rul,{feature}]: inf={n_inf}, nan={n_nan}")
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
            if self.verbose:
                print(f"[skip] no correlations for feature '{feature}'")
            return
        order = np.argsort(np.array(corrs))
        names = [names[i] for i in order]
        corrs = [corrs[i] for i in order]

        # Global stats for lines/band
        mean_val = float(np.mean(corrs))
        std_val = float(np.std(corrs))
        median_val = float(np.median(corrs))

        # Chunk to avoid overly wide plots
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

            # Dynamic width but capped
            fig_w = max(12, min(0.6 * n_chunk, 28))
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            ax.plot(x, n_chunk_corrs, color='0.4', linewidth=1.5, alpha=0.9)
            ax.scatter(x, n_chunk_corrs, c=colors, s=30, zorder=3)
            # Mean/median lines and ±1σ band (global)
            ax.axhline(mean_val, color='green', linestyle='--', linewidth=1.1, label=f'Mean {mean_val:.3f}')
            ax.axhline(median_val, color='purple', linestyle=':', linewidth=1.1, label=f'Median {median_val:.3f}')
            ax.axhspan(mean_val - std_val, mean_val + std_val, color='green', alpha=0.08, label='±1σ')
            ax.set_xticks(x)
            ax.set_xticklabels(n_chunk_names, rotation=45, ha='right')
            ax.set_ylabel(f'Correlation of {feature} with RUL')
            ax.set_xlabel('Battery')
            title_suffix = f" (part {p+1}/{pages})" if pages > 1 else ""
            ax.set_title(f'{feature}–RUL Correlation Across Batteries (sorted){title_suffix}')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
            fig.tight_layout()
            fname = f'{feature}_corr_vs_batteries' + (f'_part{p+1}' if pages > 1 else '') + '.png'
            fig.savefig(self.feature_vs_batt_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_feature_rul_correlation_boxplot(self):
        rows = []
        for feature in self.features.keys():
            _, corrs = self._collect_feature_corrs(feature)
            for v in corrs:
                if np.isfinite(v):
                    rows.append({'feature': feature, 'corr': float(v)})
        if not rows:
            if self.verbose:
                print("[skip] no correlations available for boxplot")
            return
        df = pd.DataFrame(rows)
        # Dynamic width based on number of features
        num_feats = df['feature'].nunique()
        fig_w = max(12, min(1.0 * num_feats, 36))
        plt.figure(figsize=(fig_w, 6))
        sns.boxplot(data=df, x='feature', y='corr')
        plt.axhline(0.0, color='0.5', linewidth=1)
        plt.ylabel('Correlation with RUL')
        plt.xlabel('Feature')
        plt.title('Feature–RUL Correlation Distributions Across Batteries')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.feature_box_dir / 'feature_rul_correlation_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()


# ---------------------------------
# Default feature registration utils
# ---------------------------------
def register_default_features(analyzer: ModularCorrelationAnalyzer):
    # Means of common attributes
    analyzer.register_attr_mean_feature('voltage', 'voltage_in_V', description='Mean voltage per cycle')
    analyzer.register_attr_mean_feature('current', 'current_in_A', description='Mean current per cycle')
    analyzer.register_attr_mean_feature('discharge_capacity', 'discharge_capacity_in_Ah', description='Mean discharge capacity per cycle')
    analyzer.register_attr_mean_feature('charge_capacity', 'charge_capacity_in_Ah', description='Mean charge capacity per cycle')
    analyzer.register_attr_mean_feature('temperature', 'temperature_in_C', description='Mean temperature per cycle')
    analyzer.register_attr_mean_feature('internal_resistance', 'internal_resistance_in_ohm', description='Mean internal resistance per cycle')

    # Derived features imported from shared module

    analyzer.register_feature(CycleScalarFeature('avg_c_rate', avg_c_rate, depends_on=['current_in_A'], description='Average |I|/C per cycle'))
    analyzer.register_feature(CycleScalarFeature('max_temperature', max_temperature, depends_on=['temperature_in_C']))
    analyzer.register_feature(CycleScalarFeature('max_discharge_capacity', max_discharge_capacity, depends_on=['discharge_capacity_in_Ah']))
    analyzer.register_feature(CycleScalarFeature('max_charge_capacity', max_charge_capacity, depends_on=['charge_capacity_in_Ah']))
    analyzer.register_feature(CycleScalarFeature('charge_cycle_length', charge_cycle_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('discharge_cycle_length', discharge_cycle_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('peak_cc_length', peak_cc_length, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('peak_cv_length', peak_cv_length, depends_on=['voltage_in_V', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('cycle_length', cycle_length, depends_on=['time_in_s']))
    analyzer.register_feature(CycleScalarFeature('power_during_charge_cycle', power_during_charge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('power_during_discharge_cycle', power_during_discharge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('avg_charge_c_rate', avg_charge_c_rate, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('avg_discharge_c_rate', avg_discharge_c_rate, depends_on=['current_in_A', 'time_in_s']))
    analyzer.register_feature(CycleScalarFeature('charge_to_discharge_time_ratio', charge_to_discharge_time_ratio, depends_on=['current_in_A', 'time_in_s']))


def build_default_analyzer(data_path: str, output_dir: str = 'correlation_analysis_mod', verbose: bool = False) -> ModularCorrelationAnalyzer:
    analyzer = ModularCorrelationAnalyzer(data_path, output_dir, verbose=verbose)
    register_default_features(analyzer)
    return analyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Modular correlation analysis for battery data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data directory containing *.pkl')
    parser.add_argument('--output_dir', type=str, default='correlation_analysis_mod', help='Output directory for correlation outputs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    analyzer = build_default_analyzer(args.data_path, args.output_dir, verbose=args.verbose)
    analyzer.analyze_dataset()


if __name__ == '__main__':
    main()


