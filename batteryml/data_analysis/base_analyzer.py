# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from batteryml.data.battery_data import BatteryData


class BaseDataAnalyzer:
    """Base class for battery data analysis."""
    
    def __init__(self, data_path: str, output_dir: str = "analysis_output"):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the processed battery data directory
            output_dir: Directory to save analysis results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different plot types
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.capacity_fade_dir = self.plots_dir / "capacity_fade"
        self.voltage_capacity_dir = self.plots_dir / "voltage_capacity"
        self.qc_qd_dir = self.plots_dir / "qc_qd"
        self.current_time_dir = self.plots_dir / "current_time"
        self.voltage_time_dir = self.plots_dir / "voltage_time"
        
        for dir_path in [self.capacity_fade_dir, self.voltage_capacity_dir, 
                        self.qc_qd_dir, self.current_time_dir, self.voltage_time_dir]:
            dir_path.mkdir(exist_ok=True)
    
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
    
    def calculate_capacity_fade(self, battery: BatteryData) -> Tuple[List[int], List[float]]:
        """Calculate capacity fade over cycles."""
        cycles = []
        capacities = []
        
        for cycle_data in battery.cycle_data:
            if cycle_data.discharge_capacity_in_Ah:
                max_discharge_cap = max(cycle_data.discharge_capacity_in_Ah)
                if max_discharge_cap > 0:  # Filter out invalid data
                    cycles.append(cycle_data.cycle_number)
                    capacities.append(max_discharge_cap)
        
        return cycles, capacities
    
    def plot_capacity_fade(self, battery: BatteryData, save_path: Path):
        """Plot capacity fade curve for a single battery."""
        cycles, capacities = self.calculate_capacity_fade(battery)
        
        if not cycles:
            print(f"No valid capacity data for {battery.cell_id}")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(cycles, capacities, 'b-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Cycle Number')
        plt.ylabel('Discharge Capacity (Ah)')
        plt.title(f'Capacity Fade - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        
        # Add nominal capacity line if available
        if battery.nominal_capacity_in_Ah:
            plt.axhline(y=battery.nominal_capacity_in_Ah, color='r', 
                       linestyle='--', alpha=0.7, label=f'Nominal Capacity ({battery.nominal_capacity_in_Ah:.2f} Ah)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_voltage_capacity_curves(self, battery: BatteryData, save_path: Path, 
                                   max_cycles: int = 10):
        """Plot voltage vs capacity curves for selected cycles."""
        plt.figure(figsize=(12, 8))
        
        # Select cycles to plot (every nth cycle)
        total_cycles = len(battery.cycle_data)
        if total_cycles > max_cycles:
            step = total_cycles // max_cycles
            selected_cycles = list(range(0, total_cycles, step))[:max_cycles]
        else:
            selected_cycles = list(range(total_cycles))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_cycles)))
        
        for i, cycle_idx in enumerate(selected_cycles):
            cycle_data = battery.cycle_data[cycle_idx]
            if (cycle_data.voltage_in_V and cycle_data.discharge_capacity_in_Ah and 
                len(cycle_data.voltage_in_V) > 0 and len(cycle_data.discharge_capacity_in_Ah) > 0):
                
                # Convert to numpy arrays for easier handling
                voltage = np.array(cycle_data.voltage_in_V)
                capacity = np.array(cycle_data.discharge_capacity_in_Ah)
                
                # Filter out invalid data points
                valid_mask = (voltage > 0) & (capacity > 0) & (~np.isnan(voltage)) & (~np.isnan(capacity))
                if np.any(valid_mask):
                    plt.plot(capacity[valid_mask], voltage[valid_mask], 
                            color=colors[i], linewidth=1.5, 
                            label=f'Cycle {cycle_data.cycle_number}')
        
        plt.xlabel('Discharge Capacity (Ah)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Voltage vs Capacity Curves - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_qc_vs_qd(self, battery: BatteryData, save_path: Path):
        """Plot charge capacity vs discharge capacity."""
        qc_values = []
        qd_values = []
        
        for cycle_data in battery.cycle_data:
            if (cycle_data.charge_capacity_in_Ah and cycle_data.discharge_capacity_in_Ah and
                len(cycle_data.charge_capacity_in_Ah) > 0 and len(cycle_data.discharge_capacity_in_Ah) > 0):
                
                max_qc = max(cycle_data.charge_capacity_in_Ah)
                max_qd = max(cycle_data.discharge_capacity_in_Ah)
                
                if max_qc > 0 and max_qd > 0:
                    qc_values.append(max_qc)
                    qd_values.append(max_qd)
        
        if not qc_values:
            print(f"No valid QC/QD data for {battery.cell_id}")
            return
        
        plt.figure(figsize=(10, 8))
        plt.scatter(qc_values, qd_values, alpha=0.6, s=20)
        plt.plot([0, max(max(qc_values), max(qd_values))], 
                [0, max(max(qc_values), max(qd_values))], 
                'r--', alpha=0.7, label='y=x line')
        
        plt.xlabel('Charge Capacity (Ah)')
        plt.ylabel('Discharge Capacity (Ah)')
        plt.title(f'Charge vs Discharge Capacity - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_current_time(self, battery: BatteryData, save_path: Path, max_cycles: int = 5):
        """Plot current vs time for selected cycles."""
        plt.figure(figsize=(12, 8))
        
        # Select cycles to plot
        total_cycles = len(battery.cycle_data)
        if total_cycles > max_cycles:
            step = total_cycles // max_cycles
            selected_cycles = list(range(0, total_cycles, step))[:max_cycles]
        else:
            selected_cycles = list(range(total_cycles))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cycles)))
        
        for i, cycle_idx in enumerate(selected_cycles):
            cycle_data = battery.cycle_data[cycle_idx]
            if (cycle_data.current_in_A and cycle_data.time_in_s and 
                len(cycle_data.current_in_A) > 0 and len(cycle_data.time_in_s) > 0):
                
                current = np.array(cycle_data.current_in_A)
                time = np.array(cycle_data.time_in_s)
                
                # Filter out invalid data
                valid_mask = (~np.isnan(current)) & (~np.isnan(time))
                if np.any(valid_mask):
                    plt.plot(time[valid_mask], current[valid_mask], 
                            color=colors[i], linewidth=1.5, 
                            label=f'Cycle {cycle_data.cycle_number}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title(f'Current vs Time - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_voltage_time(self, battery: BatteryData, save_path: Path, max_cycles: int = 5):
        """Plot voltage vs time for selected cycles."""
        plt.figure(figsize=(12, 8))
        
        # Select cycles to plot
        total_cycles = len(battery.cycle_data)
        if total_cycles > max_cycles:
            step = total_cycles // max_cycles
            selected_cycles = list(range(0, total_cycles, step))[:max_cycles]
        else:
            selected_cycles = list(range(total_cycles))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cycles)))
        
        for i, cycle_idx in enumerate(selected_cycles):
            cycle_data = battery.cycle_data[cycle_idx]
            if (cycle_data.voltage_in_V and cycle_data.time_in_s and 
                len(cycle_data.voltage_in_V) > 0 and len(cycle_data.time_in_s) > 0):
                
                voltage = np.array(cycle_data.voltage_in_V)
                time = np.array(cycle_data.time_in_s)
                
                # Filter out invalid data
                valid_mask = (~np.isnan(voltage)) & (~np.isnan(time))
                if np.any(valid_mask):
                    plt.plot(time[valid_mask], voltage[valid_mask], 
                            color=colors[i], linewidth=1.5, 
                            label=f'Cycle {cycle_data.cycle_number}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Voltage vs Time - {battery.cell_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def create_summary_table(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary table of dataset statistics."""
        summary_data = []
        
        # Basic counts
        summary_data.append(['Total Batteries', stats['total_batteries']])
        summary_data.append(['Average Cycles per Battery', 
                           np.mean(stats['cycle_counts']) if stats['cycle_counts'] else 0])
        summary_data.append(['Max Cycles', 
                           max(stats['cycle_counts']) if stats['cycle_counts'] else 0])
        summary_data.append(['Min Cycles', 
                           min(stats['cycle_counts']) if stats['cycle_counts'] else 0])
        
        # Nominal capacity statistics
        if stats['nominal_capacities']:
            summary_data.append(['Nominal Capacity (Ah) - Mean', np.mean(stats['nominal_capacities'])])
            summary_data.append(['Nominal Capacity (Ah) - Std', np.std(stats['nominal_capacities'])])
            summary_data.append(['Nominal Capacity (Ah) - Min', np.min(stats['nominal_capacities'])])
            summary_data.append(['Nominal Capacity (Ah) - Max', np.max(stats['nominal_capacities'])])
        
        # Material composition
        if stats['cathode_materials']:
            cathode_counts = pd.Series(stats['cathode_materials']).value_counts()
            for material, count in cathode_counts.items():
                summary_data.append([f'Cathode Material - {material}', count])
        
        if stats['anode_materials']:
            anode_counts = pd.Series(stats['anode_materials']).value_counts()
            for material, count in anode_counts.items():
                summary_data.append([f'Anode Material - {material}', count])
        
        # Form factors
        if stats['form_factors']:
            form_counts = pd.Series(stats['form_factors']).value_counts()
            for form, count in form_counts.items():
                summary_data.append([f'Form Factor - {form}', count])
        
        # Voltage ranges
        if stats['voltage_ranges']:
            min_voltages = [vr[0] for vr in stats['voltage_ranges']]
            max_voltages = [vr[1] for vr in stats['voltage_ranges']]
            summary_data.append(['Min Voltage (V) - Mean', np.mean(min_voltages)])
            summary_data.append(['Max Voltage (V) - Mean', np.mean(max_voltages)])
            summary_data.append(['Voltage Range (V) - Mean', np.mean([vr[1] - vr[0] for vr in stats['voltage_ranges']])])
        
        # Capacity fade statistics
        if stats['capacity_fade_rates']:
            summary_data.append(['Capacity Fade Rate (Ah/cycle) - Mean', np.mean(stats['capacity_fade_rates'])])
            summary_data.append(['Capacity Fade Rate (Ah/cycle) - Std', np.std(stats['capacity_fade_rates'])])
        
        if stats['final_capacities']:
            summary_data.append(['Final Capacity (Ah) - Mean', np.mean(stats['final_capacities'])])
            summary_data.append(['Final Capacity (Ah) - Std', np.std(stats['final_capacities'])])
        
        if stats['cycle_lives']:
            summary_data.append(['Cycle Life - Mean', np.mean(stats['cycle_lives'])])
            summary_data.append(['Cycle Life - Std', np.std(stats['cycle_lives'])])
            summary_data.append(['Cycle Life - Min', np.min(stats['cycle_lives'])])
            summary_data.append(['Cycle Life - Max', np.max(stats['cycle_lives'])])
        
        return pd.DataFrame(summary_data, columns=['Feature', 'Value'])
    
    def analyze_dataset(self):
        """Main method to analyze the entire dataset."""
        print(f"Starting analysis of dataset in {self.data_path}")
        
        battery_files = self.get_battery_files()
        if not battery_files:
            print(f"No battery files found in {self.data_path}")
            return
        
        print(f"Found {len(battery_files)} battery files")
        
        # Initialize statistics collection
        print("Collecting statistics from individual batteries...")
        stats = self._collect_statistics_incrementally(battery_files)
        
        # Create summary table
        summary_table = self.create_summary_table(stats)
        summary_table.to_csv(self.output_dir / "dataset_summary.csv", index=False)
        print(f"Summary table saved to {self.output_dir / 'dataset_summary.csv'}")
        
        # Generate plots for each battery (one at a time)
        print("Generating plots for individual batteries...")
        for file_path in tqdm(battery_files, desc="Processing batteries"):
            battery = self.load_battery_data(file_path)
            if battery:
                self._analyze_single_battery(battery)
            else:
                print(f"Warning: Could not load battery from {file_path}")
        
        # Generate combined plots for random selection of batteries
        print("\nGenerating combined plots for random selection of batteries...")
        self._generate_combined_plots(battery_files)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
        print(f"Individual plots saved in subdirectories:")
        print(f"  - Capacity fade: {self.capacity_fade_dir}")
        print(f"  - Voltage vs Capacity: {self.voltage_capacity_dir}")
        print(f"  - QC vs QD: {self.qc_qd_dir}")
        print(f"  - Current vs Time: {self.current_time_dir}")
        print(f"  - Voltage vs Time: {self.voltage_time_dir}")
        print(f"Combined plots saved in: {self.output_dir}/combined_plots/")
    
    def _collect_statistics_incrementally(self, battery_files):
        """Collect statistics by processing batteries one at a time."""
        stats = {
            'total_batteries': 0,
            'battery_ids': [],
            'cycle_counts': [],
            'nominal_capacities': [],
            'cathode_materials': [],
            'anode_materials': [],
            'form_factors': [],
            'voltage_ranges': [],
            'capacity_fade_rates': [],
            'final_capacities': [],
            'cycle_lives': []
        }
        
        for file_path in tqdm(battery_files, desc="Collecting statistics"):
            battery = self.load_battery_data(file_path)
            if battery:
                stats['total_batteries'] += 1
                stats['battery_ids'].append(battery.cell_id)
                stats['cycle_counts'].append(len(battery.cycle_data))
                
                if battery.nominal_capacity_in_Ah:
                    stats['nominal_capacities'].append(battery.nominal_capacity_in_Ah)
                
                if battery.cathode_material:
                    stats['cathode_materials'].append(battery.cathode_material)
                
                if battery.anode_material:
                    stats['anode_materials'].append(battery.anode_material)
                
                if battery.form_factor:
                    stats['form_factors'].append(battery.form_factor)
                
                # Calculate voltage range
                all_voltages = []
                for cycle_data in battery.cycle_data:
                    if cycle_data.voltage_in_V:
                        all_voltages.extend(cycle_data.voltage_in_V)
                
                if all_voltages:
                    stats['voltage_ranges'].append((min(all_voltages), max(all_voltages)))
                
                # Calculate capacity fade
                cycles, capacities = self.calculate_capacity_fade(battery)
                if len(cycles) > 1:
                    # Calculate fade rate (capacity loss per cycle)
                    if len(capacities) > 10:  # Only for batteries with enough data
                        initial_cap = np.mean(capacities[:5])  # Average of first 5 cycles
                        final_cap = np.mean(capacities[-5:])   # Average of last 5 cycles
                        fade_rate = (initial_cap - final_cap) / len(cycles)
                        stats['capacity_fade_rates'].append(fade_rate)
                        stats['final_capacities'].append(final_cap)
                        stats['cycle_lives'].append(len(cycles))
        
        return stats
    
    def _analyze_single_battery(self, battery):
        """Analyze a single battery and generate all plots."""
        cell_id = battery.cell_id.replace('/', '_').replace('\\', '_')  # Safe filename
        
        try:
            # Capacity fade plot
            self.plot_capacity_fade(battery, 
                                  self.capacity_fade_dir / f"{cell_id}_capacity_fade.png")
            
            # Voltage vs capacity plot
            self.plot_voltage_capacity_curves(battery, 
                                            self.voltage_capacity_dir / f"{cell_id}_voltage_capacity.png")
            
            # QC vs QD plot
            self.plot_qc_vs_qd(battery, 
                             self.qc_qd_dir / f"{cell_id}_qc_qd.png")
            
            # Current vs time plot
            self.plot_current_time(battery, 
                                 self.current_time_dir / f"{cell_id}_current_time.png")
            
            # Voltage vs time plot
            self.plot_voltage_time(battery, 
                                 self.voltage_time_dir / f"{cell_id}_voltage_time.png")
            
        except Exception as e:
            print(f"Error analyzing battery {battery.cell_id}: {e}")
            # Continue with next battery instead of failing completely
    
    def _generate_combined_plots(self, battery_files):
        """Generate combined plots for randomly selected batteries."""
        try:
            from .combined_plots import CombinedPlotGenerator
            
            # Create combined plot generator
            plot_generator = CombinedPlotGenerator(self.output_dir, num_batteries=20)
            
            # Generate combined plots
            plot_generator.generate_combined_plots(battery_files, self)
            
        except ImportError as e:
            print(f"Warning: Could not import combined plots module: {e}")
            print("Combined plots will be skipped.")
        except Exception as e:
            print(f"Warning: Error generating combined plots: {e}")
            print("Combined plots will be skipped.")
