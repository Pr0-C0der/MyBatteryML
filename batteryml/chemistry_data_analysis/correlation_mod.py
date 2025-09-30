from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Dict

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

    def __init__(self, data_path: str, output_dir: str = 'chemistry_correlation_analysis_mod', verbose: bool = False, dataset_hint: Optional[str] = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
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
        for idx, c in enumerate(battery.cycle_data):
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
        return pd.DataFrame(data)

    def _save_matrix(self, battery: BatteryData, df: pd.DataFrame):
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        df.to_csv(self.matrices_dir / f"{safe_id}_cycle_feature_matrix.csv", index=False)

    def plot_correlation_heatmap(self, battery: BatteryData, df: pd.DataFrame):
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return
        corr = numeric.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmid=0))
        fig.update_layout(title=f'Feature Correlation Matrix - {battery.cell_id}', template='plotly_white')
        safe_id = self._safe_filename(battery.cell_id)
        out_path = self.heatmaps_dir / f"{safe_id}_correlation_heatmap.png"
        try:
            fig.write_image(str(out_path), scale=2)
        except Exception:
            pass

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
        fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation='h', marker_color=colors))
        fig.update_layout(title=f'Feature correlations with RUL - {battery.cell_id}', xaxis_title='Correlation with RUL', template='plotly_white')
        safe_id = self._safe_filename(battery.cell_id)
        out_path = self.rul_bars_dir / f"{safe_id}_rul_correlations.png"
        try:
            fig.write_image(str(out_path), scale=2)
        except Exception:
            pass

    # ---------------------
    # High-level operations
    # ---------------------
    def analyze_battery(self, src: Path, battery: BatteryData):
        extractor = self._get_extractor(src, battery)
        df = self.build_cycle_feature_matrix(src, battery, extractor)
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
            self.analyze_battery(f, b)

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

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_chunk)), y=n_chunk_corrs, mode='lines+markers', name='corr'))
            fig.add_hline(y=mean_val, line_dash='dash', line_color='green', annotation_text=f'Mean {mean_val:.3f}', annotation_position='top left')
            fig.add_hline(y=median_val, line_dash='dot', line_color='purple', annotation_text=f'Median {median_val:.3f}', annotation_position='bottom left')
            fig.add_shape(type='rect', x0=-0.5, x1=n_chunk-0.5, y0=mean_val - std_val, y1=mean_val + std_val, line=dict(color='green', width=0), fillcolor='rgba(0,128,0,0.08)')
            title_suffix = f" (part {p+1}/{pages})" if pages > 1 else ""
            fig.update_layout(title=f'{feature}–RUL Correlation Across Batteries (sorted){title_suffix}', xaxis=dict(tickmode='array', tickvals=list(range(n_chunk)), ticktext=n_chunk_names), yaxis_title=f'Correlation of {feature} with RUL', xaxis_title='Battery', template='plotly_white')
            fname = f'{feature}_corr_vs_batteries' + (f'_part{p+1}' if pages > 1 else '') + '.png'
            try:
                fig.write_image(str(self.feature_vs_batt_dir / fname), scale=2)
            except Exception:
                pass

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
        fig = px.box(df, x='feature', y='corr', points='outliers', title='Feature–RUL Correlation Distributions Across Batteries')
        fig.add_hline(y=0.0, line_color='gray')
        fig.update_layout(template='plotly_white')
        try:
            fig.write_image(str(self.feature_box_dir / 'feature_rul_correlation_boxplot.png'), scale=2)
        except Exception:
            pass


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
        CycleScalarFeature('power_during_charge_cycle', 'power_during_charge_cycle'),
        CycleScalarFeature('power_during_discharge_cycle', 'power_during_discharge_cycle'),
        CycleScalarFeature('avg_charge_c_rate', 'avg_charge_c_rate'),
        CycleScalarFeature('avg_discharge_c_rate', 'avg_discharge_c_rate'),
        CycleScalarFeature('charge_to_discharge_time_ratio', 'charge_to_discharge_time_ratio'),
    ]
    for spec in specs:
        analyzer.register_feature(spec)


def build_default_analyzer(data_path: str, output_dir: str = 'chemistry_correlation_analysis_mod', verbose: bool = False, dataset_hint: Optional[str] = None) -> ChemistryCorrelationAnalyzer:
    analyzer = ChemistryCorrelationAnalyzer(data_path, output_dir, verbose=verbose, dataset_hint=dataset_hint)
    register_default_features(analyzer)
    return analyzer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chemistry-aware correlation analysis for battery data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing *.pkl (e.g., a chemistry subfolder)')
    parser.add_argument('--output_dir', type=str, default='chemistry_correlation_analysis_mod', help='Output directory for correlation outputs')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    analyzer = build_default_analyzer(args.data_path, args.output_dir, verbose=args.verbose, dataset_hint=args.dataset_hint)
    analyzer.analyze_dataset()


if __name__ == '__main__':
    main()


