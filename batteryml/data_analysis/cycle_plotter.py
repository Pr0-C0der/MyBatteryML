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
            if hasattr(first_cycle, attr):
                attr_value = getattr(first_cycle, attr)
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
                if hasattr(cycle_data, attr_name):
                    feature_data = getattr(cycle_data, attr_name)
                    time_data = cycle_data.time_in_s
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

            # Derived analysis: Power vs time
            power_dir = self.output_dir / "power_vs_time"
            power_dir.mkdir(exist_ok=True)
            self.plot_power_vs_time(battery, power_dir / f"{cell_id}_power_time.png")

            # Derived analysis: Average C-rate per cycle
            c_rate_dir = self.output_dir / "avg_c_rate_vs_cycle"
            c_rate_dir.mkdir(exist_ok=True)
            self.plot_avg_c_rate_vs_cycle(battery, c_rate_dir / f"{cell_id}_avg_c_rate_cycle.png")

            # Derived analysis: Energy per cycle (charge/discharge)
            energy_dir = self.output_dir / "energy_vs_cycle"
            energy_dir.mkdir(exist_ok=True)
            self.plot_energy_vs_cycle(battery, energy_dir / f"{cell_id}_energy_cycle.png")

            # Derived analysis: Coulombic efficiency per cycle
            eff_dir = self.output_dir / "coulombic_efficiency_vs_cycle"
            eff_dir.mkdir(exist_ok=True)
            self.plot_coulombic_efficiency_vs_cycle(battery, eff_dir / f"{cell_id}_ceff_cycle.png")
            
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

    def plot_power_vs_time(self, battery: BatteryData, save_path: Path):
        """Plot power P=V*I vs relative time for selected cycles."""
        selected_cycles = self.select_cycles(len(battery.cycle_data))
        plt.figure(figsize=(12, 8))
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(selected_cycles)))
        for i, idx in enumerate(selected_cycles):
            if idx >= len(battery.cycle_data):
                continue
            c = battery.cycle_data[idx]
            if c.voltage_in_V is None or c.current_in_A is None or c.time_in_s is None:
                continue
            try:
                V = np.array(c.voltage_in_V)
                I = np.array(c.current_in_A)
                t = np.array(c.time_in_s)
                if V.size == 0 or I.size == 0 or t.size == 0:
                    continue
                n = min(len(V), len(I), len(t))
                V, I, t = V[:n], I[:n], t[:n]
                mask = (~np.isnan(V)) & (~np.isnan(I)) & (~np.isnan(t))
                if not np.any(mask):
                    continue
                P = V[mask] * I[mask]
                tr = t[mask] - t[mask][0]
                plt.plot(tr, P, color=colors[i], linewidth=1.5, alpha=0.8, label=f'Cycle {c.cycle_number}')
            except Exception:
                continue
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.title(f'Power vs Relative Time - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=len(selected_cycles)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Cycle Index', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

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

    def plot_energy_vs_cycle(self, battery: BatteryData, save_path: Path):
        """Plot aggregate charge/discharge energy (Wh) per cycle."""
        xs, e_ch, e_dis = [], [], []
        for c in battery.cycle_data:
            if c.voltage_in_V is None or c.current_in_A is None or c.time_in_s is None:
                continue
            try:
                V = np.array(c.voltage_in_V)
                I = np.array(c.current_in_A)
                t = np.array(c.time_in_s)
                n = min(len(V), len(I), len(t))
                V, I, t = V[:n], I[:n], t[:n]
                mask = (~np.isnan(V)) & (~np.isnan(I)) & (~np.isnan(t))
                if not np.any(mask):
                    continue
                V, I, t = V[mask], I[mask], t[mask]
                # Charge energy: integrate where I>0
                ch_mask = I > 0
                dis_mask = I < 0
                e_c = np.trapz(V[ch_mask] * I[ch_mask], t[ch_mask]) / 3600.0 if np.any(ch_mask) else 0.0
                # Discharge energy: make positive
                e_d = -np.trapz(V[dis_mask] * I[dis_mask], t[dis_mask]) / 3600.0 if np.any(dis_mask) else 0.0
                xs.append(c.cycle_number)
                e_ch.append(float(e_c))
                e_dis.append(float(e_d))
            except Exception:
                continue
        if len(xs) == 0:
            return
        plt.figure(figsize=(10, 6))
        order = np.argsort(np.array(xs))
        xs_sorted = np.array(xs)[order]
        e_ch_sorted = np.array(e_ch)[order]
        e_dis_sorted = np.array(e_dis)[order]
        plt.plot(xs_sorted, e_ch_sorted, marker='o', linewidth=1.5, alpha=0.9, label='Charge Energy (Wh)')
        plt.plot(xs_sorted, e_dis_sorted, marker='s', linewidth=1.5, alpha=0.9, label='Discharge Energy (Wh)')
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Energy (Wh)', fontsize=12)
        plt.title(f'Energy per Cycle - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_coulombic_efficiency_vs_cycle(self, battery: BatteryData, save_path: Path):
        """Plot Coulombic efficiency η = Qd_out / Qc_in per cycle."""
        xs, etas = [], []
        for c in battery.cycle_data:
            Qc = c.charge_capacity_in_Ah
            Qd = c.discharge_capacity_in_Ah
            if Qc is None or Qd is None:
                continue
            try:
                Qc_arr = np.array(Qc)
                Qd_arr = np.array(Qd)
                Qc_max = np.nanmax(Qc_arr) if Qc_arr.size > 0 else np.nan
                Qd_max = np.nanmax(Qd_arr) if Qd_arr.size > 0 else np.nan
                if np.isfinite(Qc_max) and Qc_max > 0 and np.isfinite(Qd_max):
                    eta = float(Qd_max / Qc_max)
                    xs.append(c.cycle_number)
                    etas.append(eta)
            except Exception:
                continue
        if len(xs) == 0:
            return
        plt.figure(figsize=(10, 6))
        order = np.argsort(np.array(xs))
        xs_sorted = np.array(xs)[order]
        etas_sorted = np.array(etas)[order]
        plt.plot(xs_sorted, etas_sorted, marker='o', linewidth=1.5, alpha=0.9)
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Coulombic Efficiency (Qd/Qc)', fontsize=12)
        plt.title(f'Coulombic Efficiency per Cycle - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


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
