# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
matplotlib.use('Agg')

from batteryml.data.battery_data import BatteryData


class CyclePlotter:
    """Plotter for generating time-series plots across different cycles."""
    
    def __init__(self, data_path: str, output_dir: str = "cycle_plots", cycle_gap: int = 100):
        """
        Initialize the cycle plotter.
        
        Args:
            data_path: Path to the processed battery data directory
            output_dir: Directory to save cycle plots
            cycle_gap: Gap between cycles to plot (default: 100)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.cycle_gap = cycle_gap
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect available features dynamically
        self.features = self._detect_available_features()
        
        # Create subdirectories for different features
        for feature in self.features:
            feature_dir = self.output_dir / f"{feature}_vs_time"
            feature_dir.mkdir(exist_ok=True)
    
    def load_battery_data(self, file_path: Path) -> Optional[BatteryData]:
        """Load a single battery data file."""
        try:
            with open(file_path, 'rb') as f:
                return BatteryData.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_battery_files(self) -> List[Path]:
        """Get list of battery pickle files."""
        return list(self.data_path.glob("*.pkl"))
    
    def _detect_available_features(self) -> List[str]:
        """Detect available features by examining the first battery file."""
        battery_files = self.get_battery_files()
        if not battery_files:
            # Default features if no files found
            return ['voltage', 'current', 'capacity', 'temperature']
        
        # Load the first battery to detect available features
        sample_battery = self.load_battery_data(battery_files[0])
        if not sample_battery or not sample_battery.cycle_data:
            return ['voltage', 'current', 'capacity', 'temperature']
        
        # Check the first cycle for available features
        first_cycle = sample_battery.cycle_data[0]
        available_features = []
        
        # Map CycleData attributes to feature names
        feature_mapping = {
            'voltage_in_V': 'voltage',
            'current_in_A': 'current', 
            'discharge_capacity_in_Ah': 'capacity',
            'charge_capacity_in_Ah': 'charge_capacity',
            'temperature_in_C': 'temperature',
            'internal_resistance_in_ohm': 'internal_resistance',
            'energy_charge': 'energy_charge',
            'energy_discharge': 'energy_discharge',
            'Qdlin': 'qdlin',
            'Tdlin': 'tdlin'
        }
        
        for attr, feature_name in feature_mapping.items():
            attr_value = None
            if hasattr(first_cycle, attr):
                attr_value = getattr(first_cycle, attr)
            else:
                # Fallback: additional_data may store MATR extras (e.g., Qdlin, Tdlin)
                if hasattr(first_cycle, 'additional_data') and attr in getattr(first_cycle, 'additional_data', {}):
                    attr_value = first_cycle.additional_data.get(attr)
            if attr_value is None:
                continue
            # Consider scalars as available; for sequences require non-empty
            if isinstance(attr_value, (int, float)):
                available_features.append(feature_name)
            else:
                try:
                    if len(attr_value) > 0:
                        available_features.append(feature_name)
                except TypeError:
                    available_features.append(feature_name)
        
        # Always include basic features if they exist
        basic_features = ['voltage', 'current', 'capacity', 'temperature']
        for feature in basic_features:
            if feature in available_features and feature not in available_features:
                available_features.insert(0, feature)
        
        return available_features
    
    def select_cycles(self, total_cycles: int) -> List[int]:
        """Select cycles to plot based on cycle gap."""
        if total_cycles <= 1:
            return [0]
        
        # Always include cycle 1 (index 0)
        selected_cycles = [0]
        
        # Add cycles at regular intervals
        current_cycle = self.cycle_gap
        while current_cycle < total_cycles:
            selected_cycles.append(current_cycle)
            current_cycle += self.cycle_gap
        
        # Always include the last cycle if it's not already included
        if selected_cycles[-1] != total_cycles - 1:
            selected_cycles.append(total_cycles - 1)
        
        return selected_cycles
    
    def plot_feature_vs_time(self, battery: BatteryData, feature_name: str, save_path: Path):
        """Generic method to plot any feature vs time for selected cycles."""
        selected_cycles = self.select_cycles(len(battery.cycle_data))
        
        plt.figure(figsize=(12, 8))
        # Use a color gradient from blue (early cycles) to red (late cycles)
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(selected_cycles)))
        
        # Map feature names to CycleData attributes
        feature_mapping = {
            'voltage': ('voltage_in_V', 'Voltage (V)'),
            'current': ('current_in_A', 'Current (A)'),
            'capacity': ('discharge_capacity_in_Ah', 'Discharge Capacity (Ah)'),
            'charge_capacity': ('charge_capacity_in_Ah', 'Charge Capacity (Ah)'),
            'temperature': ('temperature_in_C', 'Temperature (°C)'),
            'internal_resistance': ('internal_resistance_in_ohm', 'Internal Resistance (Ω)'),
            'energy_charge': ('energy_charge', 'Charge Energy (Wh)'),
            'energy_discharge': ('energy_discharge', 'Discharge Energy (Wh)'),
            'qdlin': ('Qdlin', 'Qdlin (Ah)'),
            'tdlin': ('Tdlin', 'Tdlin (°C)')
        }
        
        if feature_name not in feature_mapping:
            print(f"Warning: Unknown feature '{feature_name}'")
            return
        
        attr_name, ylabel = feature_mapping[feature_name]
        
        for i, cycle_idx in enumerate(selected_cycles):
            if cycle_idx < len(battery.cycle_data):
                cycle_data = battery.cycle_data[cycle_idx]
                # Always get time first
                time_data = cycle_data.time_in_s
                # Resolve feature data from attribute or additional_data
                feature_data = None
                if hasattr(cycle_data, attr_name):
                    feature_data = getattr(cycle_data, attr_name)
                elif hasattr(cycle_data, 'additional_data') and attr_name in getattr(cycle_data, 'additional_data', {}):
                    feature_data = cycle_data.additional_data.get(attr_name)
                # Only plot if both series data exist and are non-empty arrays
                if feature_data is not None and time_data is not None:
                    try:
                        feature_values = np.array(feature_data)
                        time = np.array(time_data)
                        if feature_values.size > 0 and time.size > 0:
                            # Filter valid data
                            valid_mask = (~np.isnan(feature_values)) & (~np.isnan(time))
                            if attr_name in ['voltage_in_V', 'discharge_capacity_in_Ah', 'charge_capacity_in_Ah']:
                                valid_mask = valid_mask & (feature_values > 0)
                            if np.any(valid_mask):
                                relative_time = time[valid_mask] - time[valid_mask][0]
                                plt.plot(relative_time, feature_values[valid_mask],
                                         color=colors[i], linewidth=1.5, alpha=0.8,
                                         label=f'Cycle {cycle_data.cycle_number}')
                    except Exception:
                        # Skip plotting this cycle/feature on conversion issues
                        pass
        
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{feature_name.title()} vs Relative Time - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add colorbar to show cycle progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=len(selected_cycles)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Cycle Index', fontsize=10)
        
        plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _is_scalar_feature(self, battery: BatteryData, feature_name: str) -> bool:
        """Heuristically determine if a feature is scalar per cycle (no time dimension)."""
        feature_mapping = {
            'voltage': 'voltage_in_V',
            'current': 'current_in_A',
            'capacity': 'discharge_capacity_in_Ah',
            'charge_capacity': 'charge_capacity_in_Ah',
            'temperature': 'temperature_in_C',
            'internal_resistance': 'internal_resistance_in_ohm',
            'energy_charge': 'energy_charge',
            'energy_discharge': 'energy_discharge',
            'qdlin': 'Qdlin',
            'tdlin': 'Tdlin'
        }
        attr_name = feature_mapping.get(feature_name)
        if attr_name is None:
            return False
        for cycle_data in battery.cycle_data:
            if hasattr(cycle_data, attr_name):
                val = getattr(cycle_data, attr_name)
                if val is None:
                    continue
                # numpy scalar or python number indicates scalar feature
                if np.isscalar(val):
                    return True
                # If it's a sequence-like, it's not scalar
                try:
                    _ = len(val)
                    return False
                except TypeError:
                    return True
        return False

    def plot_feature_vs_cycle(self, battery: BatteryData, feature_name: str, save_path: Path):
        """Plot scalar feature values against cycle number."""
        feature_mapping = {
            'voltage': 'voltage_in_V',
            'current': 'current_in_A',
            'capacity': 'discharge_capacity_in_Ah',
            'charge_capacity': 'charge_capacity_in_Ah',
            'temperature': 'temperature_in_C',
            'internal_resistance': 'internal_resistance_in_ohm',
            'energy_charge': 'energy_charge',
            'energy_discharge': 'energy_discharge',
            'qdlin': 'Qdlin',
            'tdlin': 'Tdlin'
        }
        attr_name = feature_mapping.get(feature_name)
        if attr_name is None:
            return

        xs, ys = [], []
        for c in battery.cycle_data:
            if hasattr(c, attr_name):
                val = getattr(c, attr_name)
                if val is None:
                    continue
                try:
                    if np.isscalar(val):
                        f = float(val)
                        if not np.isnan(f):
                            xs.append(c.cycle_number)
                            ys.append(f)
                except Exception:
                    continue

        if len(xs) == 0:
            return

        plt.figure(figsize=(10, 6))
        order = np.argsort(np.array(xs))
        xs_sorted = np.array(xs)[order]
        ys_sorted = np.array(ys)[order]
        plt.plot(xs_sorted, ys_sorted, marker='o', linewidth=1.5, alpha=0.9)
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel(feature_name.replace('_', ' ').title(), fontsize=12)
        plt.title(f'{feature_name.title()} vs Cycle Number - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def plot_battery_features(self, battery: BatteryData):
        """Plot all available features for a single battery."""
        cell_id = battery.cell_id.replace('/', '_').replace('\\', '_')  # Safe filename
        
        try:
            # Plot all detected features
            for feature in self.features:
                if self._is_scalar_feature(battery, feature):
                    feature_dir = self.output_dir / f"{feature}_vs_cycle"
                    feature_dir.mkdir(exist_ok=True)
                    save_path = feature_dir / f"{cell_id}_{feature}_cycle.png"
                    self.plot_feature_vs_cycle(battery, feature, save_path)
                else:
                    feature_dir = self.output_dir / f"{feature}_vs_time"
                    feature_dir.mkdir(exist_ok=True)
                    save_path = feature_dir / f"{cell_id}_{feature}_time.png"
                    self.plot_feature_vs_time(battery, feature, save_path)

            # Derived analysis: Average C-rate per cycle
            c_rate_dir = self.output_dir / "avg_c_rate_vs_cycle"
            c_rate_dir.mkdir(exist_ok=True)
            self.plot_avg_c_rate_vs_cycle(battery, c_rate_dir / f"{cell_id}_avg_c_rate_cycle.png")

            # Voltage-Current twin-axis plots for cycle 0 and last cycle
            vi_base = self.output_dir / "voltage_current"
            (vi_base / "cycle_0").mkdir(parents=True, exist_ok=True)
            (vi_base / "cycle_last").mkdir(parents=True, exist_ok=True)
            if len(battery.cycle_data) > 0:
                self.plot_voltage_current_twin(
                    battery,
                    cycle_index=0,
                    save_path=vi_base / "cycle_0" / f"{cell_id}_v_i.png"
                )
                self.plot_voltage_current_twin(
                    battery,
                    cycle_index=len(battery.cycle_data) - 1,
                    save_path=vi_base / "cycle_last" / f"{cell_id}_v_i.png"
                )

            # Overlay plots: first vs last cycle for Current and Voltage (blue=first, red=last)
            ov_base_cur = self.output_dir / "current_first_last"
            ov_base_vol = self.output_dir / "voltage_first_last"
            ov_base_cur.mkdir(parents=True, exist_ok=True)
            ov_base_vol.mkdir(parents=True, exist_ok=True)
            if len(battery.cycle_data) > 0:
                self.plot_first_last_overlay(
                    battery,
                    feature='current',
                    attr='current_in_A',
                    ylabel='Current (A)',
                    save_path=ov_base_cur / f"{cell_id}_current_first_last.png"
                )
                self.plot_first_last_overlay(
                    battery,
                    feature='voltage',
                    attr='voltage_in_V',
                    ylabel='Voltage (V)',
                    save_path=ov_base_vol / f"{cell_id}_voltage_first_last.png"
                )

            # New features vs cycle: peak constant current/voltage lengths and total cycle length
            pcc_dir = self.output_dir / "peak_constant_current_length_vs_cycle"
            pcv_dir = self.output_dir / "peak_constant_voltage_length_vs_cycle"
            cyc_dir = self.output_dir / "cycle_length_vs_cycle"
            pcc_dir.mkdir(parents=True, exist_ok=True)
            pcv_dir.mkdir(parents=True, exist_ok=True)
            cyc_dir.mkdir(parents=True, exist_ok=True)
            self.plot_peak_lengths_and_cycle_length(
                battery,
                pcc_dir / f"{cell_id}_peak_cc_length_cycle.png",
                pcv_dir / f"{cell_id}_peak_cv_length_cycle.png",
                cyc_dir / f"{cell_id}_cycle_length_cycle.png"
            )
            
        except Exception as e:
            print(f"Error plotting features for battery {battery.cell_id}: {e}")
    
    def plot_dataset_features(self):
        """Plot features for all batteries in the dataset."""
        print(f"Starting cycle plotting for dataset in {self.data_path}")
        print(f"Cycle gap: {self.cycle_gap}")
        print(f"Detected features: {', '.join(self.features)}")
        
        battery_files = self.get_battery_files()
        if not battery_files:
            print(f"No battery files found in {self.data_path}")
            return
        
        print(f"Found {len(battery_files)} battery files")
        
        # Generate plots for each battery
        for file_path in tqdm(battery_files, desc="Processing batteries"):
            battery = self.load_battery_data(file_path)
            if battery:
                self.plot_battery_features(battery)
            else:
                print(f"Warning: Could not load battery from {file_path}")
        
        print(f"Cycle plotting complete! Results saved to {self.output_dir}")
        print(f"Feature plots saved in subdirectories:")
        for feature in self.features:
            print(f"  - {feature} vs time: {self.output_dir / f'{feature}_vs_time'}")

    def plot_voltage_current_twin(self, battery: BatteryData, cycle_index: int, save_path: Path):
        """Plot voltage (left y, blue) and current (right y, red) vs relative time for a cycle."""
        if cycle_index < 0 or cycle_index >= len(battery.cycle_data):
            return
        c = battery.cycle_data[cycle_index]
        if c.voltage_in_V is None or c.current_in_A is None or c.time_in_s is None:
            return
        try:
            V = np.array(c.voltage_in_V)
            I = np.array(c.current_in_A)
            t = np.array(c.time_in_s)
            n = min(len(V), len(I), len(t))
            V, I, t = V[:n], I[:n], t[:n]
            mask = (~np.isnan(V)) & (~np.isnan(I)) & (~np.isnan(t))
            if not np.any(mask):
                return
            V, I, t = V[mask], I[mask], t[mask]
            tr = t - t[0]

            fig, ax1 = plt.subplots(figsize=(12, 6))
            color_v = 'tab:blue'
            color_i = 'tab:red'
            ax1.set_xlabel('Relative Time (s)')
            ax1.set_ylabel('Voltage (V)', color=color_v)
            ax1.plot(tr, V, color=color_v, linewidth=1.6)
            ax1.tick_params(axis='y', labelcolor=color_v)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Current (A)', color=color_i)
            ax2.plot(tr, I, color=color_i, linewidth=1.2, alpha=0.9)
            ax2.tick_params(axis='y', labelcolor=color_i)

            plt.title(f'Voltage & Current vs Relative Time - {battery.cell_id} (Cycle {c.cycle_number})', fontsize=13)
            fig.tight_layout()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            return

    def plot_first_last_overlay(self, battery: BatteryData, feature: str, attr: str, ylabel: str, save_path: Path):
        """Overlay first (blue) and last (red) cycle for a single feature vs relative time."""
        if len(battery.cycle_data) == 0:
            return
        idx_first = 0
        idx_last = len(battery.cycle_data) - 1

        def get_xy(c):
            vals = getattr(c, attr, None)
            t = getattr(c, 'time_in_s', None)
            if vals is None or t is None:
                return None, None
            try:
                y = np.array(vals); x = np.array(t)
                n = min(len(y), len(x))
                y, x = y[:n], x[:n]
                m = (~np.isnan(y)) & (~np.isnan(x))
                if not np.any(m):
                    return None, None
                return x[m] - x[m][0], y[m]
            except Exception:
                return None, None

        c_first = battery.cycle_data[idx_first]
        c_last = battery.cycle_data[idx_last]
        x1, y1 = get_xy(c_first)
        x2, y2 = get_xy(c_last)
        if x1 is None or x2 is None:
            return
        plt.figure(figsize=(12, 6))
        plt.plot(x1, y1, color='tab:blue', linewidth=1.6, label=f'Cycle {c_first.cycle_number}')
        plt.plot(x2, y2, color='tab:red', linewidth=1.6, label=f'Cycle {c_last.cycle_number}')
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{feature.title()} (First vs Last Cycle) - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _last_peak_length(self, arr: np.ndarray, t: np.ndarray, rtol: float = 1e-3, atol: float = 1e-6) -> float:
        """Length from start to the last index where value equals its maximum (plateau end), using isclose.
        Returns time in seconds (t[last_peak] - t[0]). If invalid, returns NaN.
        """
        if arr.size == 0 or t.size == 0:
            return np.nan
        vmax = np.nanmax(arr)
        if not np.isfinite(vmax):
            return np.nan
        close_mask = np.isclose(arr, vmax, rtol=rtol, atol=atol)
        if not np.any(close_mask):
            return np.nan
        last_idx = np.where(close_mask)[0][-1]
        try:
            return float(t[last_idx] - t[0])
        except Exception:
            return np.nan

    def plot_peak_lengths_and_cycle_length(self, battery: BatteryData, save_pcc: Path, save_pcv: Path, save_cyc: Path):
        xs, pcc, pcv, cyc = [], [], [], []
        for c in battery.cycle_data:
            t = np.array(c.time_in_s or [])
            I = np.array(c.current_in_A or [])
            V = np.array(c.voltage_in_V or [])
            n = min(len(t), len(I), len(V)) if len(I) and len(V) else len(t)
            t = t[:n]
            I = I[:n] if len(I) else np.array([])
            V = V[:n] if len(V) else np.array([])
            if t.size == 0:
                continue
            m_t = ~np.isnan(t)
            t = t[m_t]
            if t.size == 0:
                continue
            # Peak constant current length
            if I.size:
                I = I[m_t[:I.size]] if I.size == m_t[:I.size].size else I
                pcc_len = self._last_peak_length(I, t)
            else:
                pcc_len = np.nan
            # Peak constant voltage length
            if V.size:
                V = V[m_t[:V.size]] if V.size == m_t[:V.size].size else V
                pcv_len = self._last_peak_length(V, t)
            else:
                pcv_len = np.nan
            # Complete cycle length
            cyc_len = float(t[-1] - t[0]) if t.size > 0 else np.nan
            xs.append(c.cycle_number)
            pcc.append(pcc_len)
            pcv.append(pcv_len)
            cyc.append(cyc_len)
        if len(xs) == 0:
            return
        order = np.argsort(np.array(xs))
        xs_sorted = np.array(xs)[order]
        pcc_sorted = np.array(pcc)[order]
        pcv_sorted = np.array(pcv)[order]
        cyc_sorted = np.array(cyc)[order]
        # Plot PCC
        plt.figure(figsize=(10, 6))
        plt.plot(xs_sorted, pcc_sorted, marker='o', linewidth=1.5)
        plt.xlabel('Cycle Number'); plt.ylabel('Peak Constant Current Length (s)')
        plt.title(f'Peak Constant Current Length per Cycle - {battery.cell_id}')
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_pcc, dpi=300, bbox_inches='tight'); plt.close()
        # Plot PCV
        plt.figure(figsize=(10, 6))
        plt.plot(xs_sorted, pcv_sorted, marker='o', linewidth=1.5)
        plt.xlabel('Cycle Number'); plt.ylabel('Peak Constant Voltage Length (s)')
        plt.title(f'Peak Constant Voltage Length per Cycle - {battery.cell_id}')
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_pcv, dpi=300, bbox_inches='tight'); plt.close()
        # Plot cycle length
        plt.figure(figsize=(10, 6))
        plt.plot(xs_sorted, cyc_sorted, marker='o', linewidth=1.5)
        plt.xlabel('Cycle Number'); plt.ylabel('Cycle Length (s)')
        plt.title(f'Cycle Length per Cycle - {battery.cell_id}')
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_cyc, dpi=300, bbox_inches='tight'); plt.close()

    def plot_avg_c_rate_vs_cycle(self, battery: BatteryData, save_path: Path):
        """Plot average |I|/C over each cycle (C-rate)."""
        C = battery.nominal_capacity_in_Ah or 0.0
        xs, ys = [], []
        for c in battery.cycle_data:
            if c.current_in_A is None:
                continue
            try:
                I = np.array(c.current_in_A)
                I = I[~np.isnan(I)]
                if I.size == 0:
                    continue
                if C and C > 0:
                    avg_c = float(np.mean(np.abs(I)) / C)
                else:
                    avg_c = np.nan
                if not np.isnan(avg_c):
                    xs.append(c.cycle_number)
                    ys.append(avg_c)
            except Exception:
                continue
        if len(xs) == 0:
            return
        plt.figure(figsize=(10, 6))
        order = np.argsort(np.array(xs))
        xs_sorted = np.array(xs)[order]
        ys_sorted = np.array(ys)[order]
        plt.plot(xs_sorted, ys_sorted, marker='o', linewidth=1.5, alpha=0.9)
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Average C-rate (|I|/C_nom)', fontsize=12)
        plt.title(f'Average C-rate per Cycle - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # (Removed) power/energy/coulombic efficiency plotting functions per user request


def main():
    """Main function to run cycle plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cycle plots for battery data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='cycle_plots',
                       help='Output directory for cycle plots')
    parser.add_argument('--cycle_gap', type=int, default=100,
                       help='Gap between cycles to plot (default: 100)')
    
    args = parser.parse_args()
    
    plotter = CyclePlotter(args.data_path, args.output_dir, args.cycle_gap)
    plotter.plot_dataset_features()


if __name__ == "__main__":
    main()
