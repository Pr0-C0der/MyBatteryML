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
        
        # Create subdirectory for capacity fade plots only
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.capacity_fade_dir = self.plots_dir / "capacity_fade"
        self.capacity_fade_dir.mkdir(exist_ok=True)
    
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
        """Analyze a single battery and generate capacity fade plot only."""
        cell_id = battery.cell_id.replace('/', '_').replace('\\', '_')  # Safe filename
        
        try:
            # Capacity fade plot only
            self.plot_capacity_fade(battery, 
                                  self.capacity_fade_dir / f"{cell_id}_capacity_fade.png")
            
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
