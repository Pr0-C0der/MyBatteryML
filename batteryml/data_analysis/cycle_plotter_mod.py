# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

from batteryml.data.battery_data import BatteryData
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

ComputeTimeFn = Callable[[BatteryData, object], Tuple[np.ndarray, np.ndarray, str, str]]
ComputeCycleFn = Callable[[BatteryData, object], Optional[float]]


@dataclass
class TimeFeatureSpec:
    name: str
    compute: ComputeTimeFn
    depends_on: List[str] = field(default_factory=list)


@dataclass
class CycleFeatureSpec:
    name: str
    compute: ComputeCycleFn
    depends_on: List[str] = field(default_factory=list)
    ylabel: Optional[str] = None


class ModularCyclePlotter:
    """Modular, extensible cycle plotter.

    - Register time-series features via TimeFeatureSpec
    - Register per-cycle scalar features via CycleFeatureSpec
    - Easy to add new features by providing `depends_on` and a compute function
    """

    def __init__(self, data_path: str, output_dir: str = "cycle_plots_mod", cycle_gap: int = 100, verbose: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.cycle_gap = int(cycle_gap)
        self.verbose = bool(verbose)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.time_features: Dict[str, TimeFeatureSpec] = {}
        self.cycle_features: Dict[str, CycleFeatureSpec] = {}

    # -----------
    # Registration
    # -----------
    def register_time_feature(self, spec: TimeFeatureSpec):
        self.time_features[spec.name] = spec

    def register_cycle_feature(self, spec: CycleFeatureSpec):
        self.cycle_features[spec.name] = spec
        print(f"Registered cycle feature: {spec.name}")

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
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(sel)))
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
                # Report non-finite counts if verbose
                if self.verbose:
                    cnt_x_inf = int((~np.isfinite(x)).sum())
                    cnt_y_inf = int((~np.isfinite(y)).sum())
                    if cnt_x_inf or cnt_y_inf:
                        print(f"[warn] {battery.cell_id} cycle {c.cycle_number}: time feature '{spec.name}' has non-finite values: x_inf={cnt_x_inf}, y_inf={cnt_y_inf}")
                m = np.isfinite(x) & np.isfinite(y)
                if not np.any(m):
                    if self.verbose:
                        print(f"[skip] {battery.cell_id} cycle {c.cycle_number}: no finite samples for '{spec.name}'")
                    continue
                xr = x[m] - x[m][0]
                plt.plot(xr, y[m], color=colors[i], linewidth=1.5, alpha=0.85, label=f'Cycle {c.cycle_number}')
                any_plotted = True
                x_label, y_label = xl, yl
            except Exception as e:
                if self.verbose:
                    print(f"[warn] {battery.cell_id}: time compute failed for '{spec.name}' on cycle {c.cycle_number}: {e}")
                continue

        if not any_plotted:
            if self.verbose:
                print(f"[skip] {battery.cell_id}: no valid data to plot for time feature '{spec.name}'")
            plt.close()
            return False

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{spec.name.title()} vs Time - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=len(sel)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Cycle Index')
        h, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        plt.tight_layout()
        out_dir = self.output_dir / 'feature_vs_time_graphs' / f"{spec.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        plt.savefig(out_dir / f"{safe_id}_{spec.name}_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        if self.verbose:
            print(f"[ok] {battery.cell_id}: saved time plot '{spec.name}'")
        return True

    def plot_cycle_feature(self, battery: BatteryData, spec: CycleFeatureSpec) -> bool:
        xs: List[float] = []
        ys: List[float] = []
        none_count = 0
        nonfinite_count = 0
        for c in battery.cycle_data:
            try:
                val = spec.compute(battery, c)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] {battery.cell_id}: cycle compute failed for '{spec.name}' on cycle {getattr(c,'cycle_number',-1)}: {e}")
                val = None
            if val is None:
                none_count += 1
                continue
            try:
                fval = float(val)
                if np.isfinite(fval):
                    xs.append(c.cycle_number)
                    ys.append(fval)
                else:
                    nonfinite_count += 1
            except Exception:
                continue

        if self.verbose and (none_count or nonfinite_count):
            print(f"[info] {battery.cell_id}: feature '{spec.name}' skipped values — None={none_count}, non-finite={nonfinite_count}")

        if not xs:
            if self.verbose:
                print(f"[skip] {battery.cell_id}: no values for cycle feature '{spec.name}'")
            return False

        order = np.argsort(np.array(xs, dtype=float))
        xs_s = np.array(xs, dtype=float)[order]
        ys_s = np.array(ys, dtype=float)[order]
        m = np.isfinite(xs_s) & np.isfinite(ys_s)
        if not np.any(m):
            if self.verbose:
                print(f"[skip] {battery.cell_id}: non-finite values for cycle feature '{spec.name}'")
            return False
        xs_s = xs_s[m]; ys_s = ys_s[m]

        plt.figure(figsize=(10, 6))
        plt.plot(xs_s, ys_s, marker='o', linewidth=1.6, alpha=0.9)
        plt.xlabel('Cycle Number')
        plt.ylabel(spec.ylabel or spec.name.replace('_', ' ').title())
        plt.title(f'{spec.name.title()} vs Cycle Number - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_dir = self.output_dir / 'feature_vs_cycle_graphs' / f"{spec.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
        plt.savefig(out_dir / f"{safe_id}_{spec.name}_cycle.png", dpi=300, bbox_inches='tight')
        plt.close()
        if self.verbose:
            print(f"[ok] {battery.cell_id}: saved cycle plot '{spec.name}'")
        return True

    # ---------------------
    # High-level operations
    # ---------------------
    def plot_battery(self, battery: BatteryData):
        # Time features
        for spec in self.time_features.values():
            self.plot_time_feature(battery, spec)
        # Cycle features
        for spec in self.cycle_features.values():
            self.plot_cycle_feature(battery, spec)

    def plot_dataset(self, include_time_batteries: Optional[List[str]] = None):
        """Plot the dataset.

        - Time features: plot for all batteries by default; if include_time_batteries is provided and non-empty,
          only plot time features for those battery IDs.
        - Cycle features: always plotted for all batteries.
        """
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
            # Time features with optional filtering
            if include_set is None or b.cell_id in include_set:
                for spec in self.time_features.values():
                    self.plot_time_feature(b, spec)
            # Cycle features always
            for spec in self.cycle_features.values():
                self.plot_cycle_feature(b, spec)


# ---------------------------------
# Default feature registration utils
# ---------------------------------
def register_default_time_features(plotter: ModularCyclePlotter):
    # Helper: attribute-based time series
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


def register_default_cycle_features(plotter: ModularCyclePlotter):
    # Generic mean-of-attribute helper
    def mean_attr(attr: str) -> Callable[[BatteryData, object], Optional[float]]:
        def _fn(b: BatteryData, c) -> Optional[float]:
            vals = getattr(c, attr, None)
            if vals is None:
                return None
            try:
                arr = np.array(vals)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return None
                return float(np.mean(arr))
            except Exception:
                return None
        return _fn

    # Averages of key time-series as per-cycle scalars
    plotter.register_cycle_feature(CycleFeatureSpec(
        'avg_voltage', mean_attr('voltage_in_V'), depends_on=['voltage_in_V'], ylabel='Average Voltage (V)'
    ))
    plotter.register_cycle_feature(CycleFeatureSpec(
        'avg_current', mean_attr('current_in_A'), depends_on=['current_in_A'], ylabel='Average Current (A)'
    ))
    plotter.register_cycle_feature(CycleFeatureSpec(
        'avg_charge_capacity', mean_attr('charge_capacity_in_Ah'), depends_on=['charge_capacity_in_Ah'], ylabel='Average Charge Capacity (Ah)'
    ))
    plotter.register_cycle_feature(CycleFeatureSpec(
        'avg_discharge_capacity', mean_attr('discharge_capacity_in_Ah'), depends_on=['discharge_capacity_in_Ah'], ylabel='Average Discharge Capacity (Ah)'
    ))
    plotter.register_cycle_feature(CycleFeatureSpec(
        'avg_temperature', mean_attr('temperature_in_C'), depends_on=['temperature_in_C'], ylabel='Average Temperature (°C)'
    ))

    # Derived features imported from shared module

    plotter.register_cycle_feature(CycleFeatureSpec('avg_c_rate', avg_c_rate, depends_on=['current_in_A'], ylabel='Average C-rate (|I|/C_nom)'))
    plotter.register_cycle_feature(CycleFeatureSpec('max_temperature', max_temperature, depends_on=['temperature_in_C'], ylabel='Max Temperature (°C)'))
    plotter.register_cycle_feature(CycleFeatureSpec('max_discharge_capacity', max_discharge_capacity, depends_on=['discharge_capacity_in_Ah'], ylabel='Max Discharge Capacity (Ah)'))
    plotter.register_cycle_feature(CycleFeatureSpec('max_charge_capacity', max_charge_capacity, depends_on=['charge_capacity_in_Ah'], ylabel='Max Charge Capacity (Ah)'))
    plotter.register_cycle_feature(CycleFeatureSpec('charge_cycle_length', charge_cycle_length, depends_on=['current_in_A', 'time_in_s'], ylabel='Charge Cycle Length (s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('discharge_cycle_length', discharge_cycle_length, depends_on=['current_in_A', 'time_in_s'], ylabel='Discharge Cycle Length (s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('peak_cc_length', peak_cc_length, depends_on=['current_in_A', 'time_in_s'], ylabel='Peak Constant Current Length (s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('peak_cv_length', peak_cv_length, depends_on=['voltage_in_V', 'time_in_s'], ylabel='Peak Constant Voltage Length (s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('cycle_length', cycle_length, depends_on=['time_in_s'], ylabel='Cycle Length (s)'))

    # New power and C-rate features
    plotter.register_cycle_feature(CycleFeatureSpec('power_during_charge_cycle', power_during_charge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s'], ylabel='Energy during Charge (W·s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('power_during_discharge_cycle', power_during_discharge_cycle, depends_on=['voltage_in_V', 'current_in_A', 'time_in_s'], ylabel='Energy during Discharge (W·s)'))
    plotter.register_cycle_feature(CycleFeatureSpec('avg_charge_c_rate', avg_charge_c_rate, depends_on=['current_in_A', 'time_in_s'], ylabel='Avg C-rate (charge)'))
    plotter.register_cycle_feature(CycleFeatureSpec('avg_discharge_c_rate', avg_discharge_c_rate, depends_on=['current_in_A', 'time_in_s'], ylabel='Avg C-rate (discharge)'))
    plotter.register_cycle_feature(CycleFeatureSpec('charge_to_discharge_time_ratio', charge_to_discharge_time_ratio, depends_on=['current_in_A', 'time_in_s'], ylabel='Charge/Discharge Time Ratio'))


def build_default_plotter(data_path: str, output_dir: str = "cycle_plots_mod", cycle_gap: int = 100, verbose: bool = False) -> ModularCyclePlotter:
    plotter = ModularCyclePlotter(data_path, output_dir, cycle_gap, verbose=verbose)
    register_default_time_features(plotter)
    register_default_cycle_features(plotter)
    return plotter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Modular cycle plotting for battery data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data directory containing *.pkl')
    parser.add_argument('--output_dir', type=str, default='cycle_plots_mod', help='Output directory')
    parser.add_argument('--cycle_gap', type=int, default=100, help='Gap between cycles for time plots')
    parser.add_argument('--include_time_batteries', type=str, nargs='*', default=None,
                        help='Optional list of battery IDs to include for time features; default all')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    plotter = build_default_plotter(args.data_path, args.output_dir, args.cycle_gap, verbose=args.verbose)
    plotter.plot_dataset(include_time_batteries=args.include_time_batteries)


if __name__ == '__main__':
    main()