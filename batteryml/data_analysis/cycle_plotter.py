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
                if attr_value is not None and len(attr_value) > 0:
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
                    if (feature_data and cycle_data.time_in_s and 
                        len(feature_data) > 0 and len(cycle_data.time_in_s) > 0):
                        
                        feature_values = np.array(feature_data)
                        time = np.array(cycle_data.time_in_s)
                        
                        # Filter valid data
                        valid_mask = (~np.isnan(feature_values)) & (~np.isnan(time))
                        if attr_name in ['voltage_in_V', 'discharge_capacity_in_Ah', 'charge_capacity_in_Ah']:
                            valid_mask = valid_mask & (feature_values > 0)
                        
                        if np.any(valid_mask):
                            # Convert to relative time (start from 0)
                            relative_time = time[valid_mask] - time[valid_mask][0]
                            plt.plot(relative_time, feature_values[valid_mask], 
                                    color=colors[i], linewidth=1.5, alpha=0.8,
                                    label=f'Cycle {cycle_data.cycle_number}')
        
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
    
    
    def plot_battery_features(self, battery: BatteryData):
        """Plot all available features for a single battery."""
        cell_id = battery.cell_id.replace('/', '_').replace('\\', '_')  # Safe filename
        
        try:
            # Plot all detected features
            for feature in self.features:
                feature_dir = self.output_dir / f"{feature}_vs_time"
                save_path = feature_dir / f"{cell_id}_{feature}_time.png"
                self.plot_feature_vs_time(battery, feature, save_path)
            
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
