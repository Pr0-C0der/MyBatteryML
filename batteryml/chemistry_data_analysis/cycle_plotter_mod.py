from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import (
    get_extractor_class,
    BaseCycleFeatures,
    DatasetSpecificCycleFeatures,
)


ComputeTimeFn = Callable[[BatteryData, object], Tuple[np.ndarray, np.ndarray, str, str]]


@dataclass
class TimeFeatureSpec:
    name: str
    compute: ComputeTimeFn
    depends_on: List[str] = field(default_factory=list)


@dataclass
class CycleFeatureSpec:
    name: str
    ylabel: Optional[str]
    # Will be bound per-battery to an extractor instance
    method_name: str
    depends_on: List[str] = field(default_factory=list)


class ChemistryCyclePlotter:
    """Cycle plotter that uses chemistry-aware feature extractors.

    For each battery, we infer/select a dataset-specific extractor and then
    compute cycle-level scalar features via that extractor.
    """

    def __init__(self, data_path: str, output_dir: str = "chemistry_cycle_plots_mod", cycle_gap: int = 100, verbose: bool = False, dataset_hint: Optional[str] = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.cycle_gap = int(cycle_gap)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.time_features: Dict[str, TimeFeatureSpec] = {}
        self.cycle_features: Dict[str, CycleFeatureSpec] = {}
        # Chemistry subfolder name inferred from data_path
        self._chemistry_name = self.data_path.name

    # -------------
    # Smoothing (moving average for feature vs cycle plots)
    # -------------
    _MA_WINDOW: int = 5

    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            kernel = np.ones(w, dtype=float)
            # handle NaNs robustly
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
    def _safe_filename(name: str) -> str:
        # Replace characters invalid on Windows filesystems and collapse whitespace
        invalid = '<>:"/\\|?*'
        s = ''.join(('_' if ch in invalid else ch) for ch in str(name))
        s = s.strip().replace(' ', '_')
        # Remove any residual control characters
        s = ''.join(ch for ch in s if ch.isprintable())
        return s or 'unknown'

    # -------------
    # Save helpers
    # -------------
    def _save_png(self, out_base: Path):
        out_base.parent.mkdir(parents=True, exist_ok=True)
        png_path = out_base.with_suffix('.png')
        try:
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()

    # -----------------
    # Dataset resolution
    # -----------------
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
        # Try path first
        key = self._infer_dataset_from_path(src)
        if key:
            return key
        # Try cell_id text
        cid = str(getattr(battery, 'cell_id', '')).upper()
        for key in ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']:
            if key in cid:
                return key
        # Try reference/description
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

    # --------------
    # Data utilities
    # --------------
    def _battery_files(self) -> List[Path]:
        return list(self.data_path.glob('*.pkl'))

    def _select_cycles(self, total_cycles: int) -> List[int]:
        if total_cycles <= 1:
            return [0]
        sel = [0]
        cur = self.cycle_gap
        while cur < total_cycles:
            sel.append(cur)
            cur += self.cycle_gap
        if sel[-1] != total_cycles - 1:
            sel.append(total_cycles - 1)
        return sel

    # -------------
    # Registrations
    # -------------
    def register_time_feature(self, spec: TimeFeatureSpec):
        self.time_features[spec.name] = spec

    def register_cycle_feature(self, spec: CycleFeatureSpec):
        self.cycle_features[spec.name] = spec
        if self.verbose:
            print(f"[register] cycle feature: {spec.name}")

    # -----------------
    # Plotting routines
    # -----------------
    def plot_time_feature(self, battery: BatteryData, spec: TimeFeatureSpec) -> bool:
        if len(battery.cycle_data) == 0:
            if self.verbose:
                print(f"[skip] {battery.cell_id}: no cycle_data for time feature '{spec.name}'")
            return False
        sel = self._select_cycles(len(battery.cycle_data))

        any_plotted = False
        plt.figure(figsize=(12, 8))
        x_label = 'Relative Time (s)'
        y_label = spec.name.replace('_', ' ').title()
        for i, idx in enumerate(sel):
            c = battery.cycle_data[idx]
            try:
                x, y, xl, yl = spec.compute(battery, c)
                if x is None or y is None:
                    continue
                x = np.asarray(x); y = np.asarray(y)
                n = min(len(x), len(y))
                if n == 0:
                    continue
                x = x[:n]; y = y[:n]
                m = np.isfinite(x) & np.isfinite(y)
                if not np.any(m):
                    continue
                xr = x[m] - x[m][0]
                plt.plot(xr, y[m], linewidth=1.5, alpha=0.9, label=f'Cycle {c.cycle_number}')
                any_plotted = True
                x_label, y_label = xl, yl
            except Exception as e:
                if self.verbose:
                    print(f"[warn] {battery.cell_id}: time compute failed for '{spec.name}' on cycle {c.cycle_number}: {e}")
                continue

        if not any_plotted:
            return False

        plt.title(f'{spec.name.title()} vs Time - {battery.cell_id}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=8)
        out_dir = self.output_dir / self._chemistry_name / 'feature_vs_time_graphs' / f"{spec.name}"
        safe_id = self._safe_filename(battery.cell_id)
        self._save_png(out_dir / f"{safe_id}_{spec.name}_time")
        return True

    def plot_cycle_feature(self, battery: BatteryData, src: Path, spec: CycleFeatureSpec, extractor: DatasetSpecificCycleFeatures) -> bool:
        xs: List[float] = []
        ys: List[float] = []
        method = getattr(extractor, spec.method_name, None)
        if method is None:
            return False
        for c in battery.cycle_data:
            try:
                val = method(battery, c)
            except Exception:
                val = None
            if val is None:
                continue
            try:
                fval = float(val)
                if np.isfinite(fval):
                    xs.append(c.cycle_number)
                    ys.append(fval)
            except Exception:
                continue
        if not xs:
            return False
        order = np.argsort(np.array(xs, dtype=float))
        xs_s = np.array(xs, dtype=float)[order]
        ys_s = np.array(ys, dtype=float)[order]
        m = np.isfinite(xs_s) & np.isfinite(ys_s)
        if not np.any(m):
            return False
        xs_s = xs_s[m]; ys_s = ys_s[m]
        plt.figure(figsize=(10, 6))
        ys_sm = self._moving_average(ys_s, self._MA_WINDOW)
        plt.plot(xs_s, ys_sm, marker='o', linewidth=1.6, alpha=0.9)
        plt.title(f'{spec.name.title()} vs Cycle Number - {battery.cell_id}')
        plt.xlabel('Cycle Number')
        plt.ylabel(spec.ylabel or spec.name.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        out_dir = self.output_dir / self._chemistry_name / 'feature_vs_cycle_graphs' / f"{spec.name}"
        safe_id = self._safe_filename(battery.cell_id)
        self._save_png(out_dir / f"{safe_id}_{spec.name}_cycle")
        return True

    # ---------------------
    # High-level operations
    # ---------------------
    def plot_battery(self, src: Path, battery: BatteryData):
        extractor = self._get_extractor(src, battery)
        # Time features
        for spec in self.time_features.values():
            self.plot_time_feature(battery, spec)
        # Cycle features (only if extractor available)
        if extractor is not None:
            for spec in self.cycle_features.values():
                self.plot_cycle_feature(battery, src, spec, extractor)

    def plot_dataset(self, include_time_batteries: Optional[List[str]] = None):
        include_set = None
        if include_time_batteries:
            include_set = set([s.replace('\\\\', '\\').replace('///', '/').strip() for s in include_time_batteries])
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
            if include_set is None or b.cell_id in include_set:
                for spec in self.time_features.values():
                    self.plot_time_feature(b, spec)
            self.plot_battery(f, b)



# ---------------------------------
# Default feature registration utils
# ---------------------------------
def register_default_time_features(plotter: ChemistryCyclePlotter):
    def attr_time(name: str, attr: str, ylabel: str):
        def compute(b: BatteryData, c) -> Tuple[np.ndarray, np.ndarray, str, str]:
            y = getattr(c, attr, None)
            t = getattr(c, 'time_in_s', None)
            if y is None or t is None:
                return None, None, 'Relative Time (s)', ylabel
            y = np.array(y); t = np.array(t)
            n = min(len(y), len(t))
            y = y[:n]; t = t[:n]
            m = np.isfinite(y) & np.isfinite(t)
            if not np.any(m):
                return None, None, 'Relative Time (s)', ylabel
            return t[m], y[m], 'Relative Time (s)', ylabel
        plotter.register_time_feature(TimeFeatureSpec(name=name, compute=compute, depends_on=[attr, 'time_in_s']))

    attr_time('voltage', 'voltage_in_V', 'Voltage (V)')
    attr_time('current', 'current_in_A', 'Current (A)')
    attr_time('discharge_capacity', 'discharge_capacity_in_Ah', 'Discharge Capacity (Ah)')
    attr_time('charge_capacity', 'charge_capacity_in_Ah', 'Charge Capacity (Ah)')
    attr_time('temperature', 'temperature_in_C', 'Temperature (°C)')


def register_default_cycle_features(plotter: ChemistryCyclePlotter):
    # Names correspond to methods on DatasetSpecificCycleFeatures/BaseCycleFeatures
    specs = [
        CycleFeatureSpec('avg_voltage', 'Average Voltage (V)', 'avg_voltage', depends_on=['voltage_in_V']),
        CycleFeatureSpec('avg_current', 'Average Current (A)', 'avg_current', depends_on=['current_in_A']),
        CycleFeatureSpec('avg_c_rate', 'Average C-rate (|I|/C_nom)', 'avg_c_rate', depends_on=['current_in_A']),
        CycleFeatureSpec('cycle_length', 'Cycle Length (s)', 'cycle_length', depends_on=['time_in_s']),
        CycleFeatureSpec('max_charge_capacity', 'Max Charge Capacity (Ah)', 'max_charge_capacity', depends_on=['charge_capacity_in_Ah']),
        CycleFeatureSpec('max_discharge_capacity', 'Max Discharge Capacity (Ah)', 'max_discharge_capacity', depends_on=['discharge_capacity_in_Ah']),
        CycleFeatureSpec('charge_cycle_length', 'Charge Cycle Length (s)', 'charge_cycle_length', depends_on=['current_in_A', 'time_in_s']),
        CycleFeatureSpec('discharge_cycle_length', 'Discharge Cycle Length (s)', 'discharge_cycle_length', depends_on=['current_in_A', 'time_in_s']),
        # peak_cv_length intentionally omitted for now
        CycleFeatureSpec('energy_during_charge_cycle', 'Energy during Charge (W·s)', 'power_during_charge_cycle', depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']),
        CycleFeatureSpec('energy_during_discharge_cycle', 'Energy during Discharge (W·s)', 'power_during_discharge_cycle', depends_on=['voltage_in_V', 'current_in_A', 'time_in_s']),
        CycleFeatureSpec('avg_charge_c_rate', 'Avg C-rate (charge)', 'avg_charge_c_rate', depends_on=['current_in_A', 'time_in_s']),
        CycleFeatureSpec('avg_discharge_c_rate', 'Avg C-rate (discharge)', 'avg_discharge_c_rate', depends_on=['current_in_A', 'time_in_s']),
        CycleFeatureSpec('max_charge_c_rate', 'Max C-rate (charge)', 'max_charge_c_rate', depends_on=['current_in_A']),
        CycleFeatureSpec('charge_to_discharge_time_ratio', 'Charge/Discharge Time Ratio', 'charge_to_discharge_time_ratio', depends_on=['current_in_A', 'time_in_s']),
    ]
    for spec in specs:
        plotter.register_cycle_feature(spec)


def build_default_plotter(data_path: str, output_dir: str = "chemistry_cycle_plots_mod", cycle_gap: int = 100, verbose: bool = False, dataset_hint: Optional[str] = None) -> ChemistryCyclePlotter:
    plotter = ChemistryCyclePlotter(data_path, output_dir, cycle_gap, verbose=verbose, dataset_hint=dataset_hint)
    register_default_time_features(plotter)
    register_default_cycle_features(plotter)
    return plotter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chemistry-aware modular cycle plotting for battery data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing *.pkl (e.g., a chemistry subfolder)')
    parser.add_argument('--output_dir', type=str, default='chemistry_cycle_plots_mod', help='Output directory')
    parser.add_argument('--cycle_gap', type=int, default=100, help='Gap between cycles for time plots')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--include_time_batteries', type=str, nargs='*', default=None, help='Optional list of battery IDs for time features')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    plotter = build_default_plotter(args.data_path, args.output_dir, args.cycle_gap, verbose=args.verbose, dataset_hint=args.dataset_hint)
    plotter.plot_dataset(include_time_batteries=args.include_time_batteries)


if __name__ == '__main__':
    main()


