# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from batteryml.data.battery_data import BatteryData
from .base_analyzer import BaseDataAnalyzer


class CombinedPlotGenerator:
    """Generate combined plots for multiple batteries."""
    
    def __init__(self, output_dir: Path, num_batteries: int = 20):
        """
        Initialize combined plot generator.
        
        Args:
            output_dir: Base output directory
            num_batteries: Number of random batteries to select
        """
        self.output_dir = output_dir
        self.num_batteries = num_batteries
        self.combined_plots_dir = output_dir / "combined_plots"
        self.combined_plots_dir.mkdir(exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_combined_plots(self, battery_files: List[Path], analyzer: BaseDataAnalyzer):
        """Generate combined plots for randomly selected batteries."""
        print(f"Generating combined plots for {self.num_batteries} randomly selected batteries...")
        
        # Select random batteries
        if len(battery_files) <= self.num_batteries:
            selected_files = battery_files
            print(f"Using all {len(battery_files)} available batteries")
        else:
            selected_files = random.sample(battery_files, self.num_batteries)
            print(f"Randomly selected {self.num_batteries} batteries from {len(battery_files)} available")
        
        # Load selected batteries
        selected_batteries = []
        for file_path in tqdm(selected_files, desc="Loading selected batteries"):
            battery = analyzer.load_battery_data(file_path)
            if battery:
                selected_batteries.append(battery)
        
        if not selected_batteries:
            print("No valid batteries found for combined plots")
            return
        
        print(f"Successfully loaded {len(selected_batteries)} batteries for combined plots")
        
        # Generate combined plots
        self._plot_combined_capacity_fade(selected_batteries, analyzer)
        self._plot_combined_voltage_capacity(selected_batteries, analyzer)
        self._plot_combined_qc_qd(selected_batteries, analyzer)
        self._plot_combined_current_time(selected_batteries, analyzer)
        self._plot_combined_voltage_time(selected_batteries, analyzer)
        self._plot_capacity_distribution(selected_batteries, analyzer)
        
        print(f"Combined plots saved to {self.combined_plots_dir}")
    
    def _plot_combined_capacity_fade(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot combined capacity fade curves."""
        plt.figure(figsize=(14, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(batteries)))
        
        for i, battery in enumerate(batteries):
            cycles, capacities = analyzer.calculate_capacity_fade(battery)
            if cycles and capacities:
                plt.plot(cycles, capacities, 
                        color=colors[i], linewidth=1.5, alpha=0.7,
                        label=battery.cell_id)
        
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Discharge Capacity (Ah)', fontsize=12)
        plt.title(f'Combined Capacity Fade Curves ({len(batteries)} batteries)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "combined_capacity_fade.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_voltage_capacity(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot combined voltage vs capacity curves."""
        plt.figure(figsize=(14, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(batteries)))
        
        for i, battery in enumerate(batteries):
            # Select a few cycles from each battery
            total_cycles = len(battery.cycle_data)
            if total_cycles > 0:
                # Select cycles: first, middle, and last
                selected_cycles = [0, total_cycles//2, total_cycles-1] if total_cycles > 2 else [0]
                
                for cycle_idx in selected_cycles:
                    cycle_data = battery.cycle_data[cycle_idx]
                    if (cycle_data.voltage_in_V and cycle_data.discharge_capacity_in_Ah and 
                        len(cycle_data.voltage_in_V) > 0 and len(cycle_data.discharge_capacity_in_Ah) > 0):
                        
                        voltage = np.array(cycle_data.voltage_in_V)
                        capacity = np.array(cycle_data.discharge_capacity_in_Ah)
                        
                        # Filter valid data
                        valid_mask = (voltage > 0) & (capacity > 0) & (~np.isnan(voltage)) & (~np.isnan(capacity))
                        if np.any(valid_mask):
                            label = f"{battery.cell_id}_C{cycle_data.cycle_number}" if len(selected_cycles) > 1 else battery.cell_id
                            plt.plot(capacity[valid_mask], voltage[valid_mask], 
                                   color=colors[i], linewidth=1, alpha=0.7,
                                   label=label)
        
        plt.xlabel('Discharge Capacity (Ah)', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.title(f'Combined Voltage vs Capacity Curves ({len(batteries)} batteries)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "combined_voltage_capacity.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_qc_qd(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot combined charge vs discharge capacity."""
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(batteries)))
        
        for i, battery in enumerate(batteries):
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
            
            if qc_values and qd_values:
                plt.scatter(qc_values, qd_values, 
                           color=colors[i], alpha=0.6, s=20,
                           label=battery.cell_id)
        
        # Add y=x line
        if qc_values and qd_values:
            max_val = max(max(qc_values), max(qd_values))
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, linewidth=2, label='y=x line')
        
        plt.xlabel('Charge Capacity (Ah)', fontsize=12)
        plt.ylabel('Discharge Capacity (Ah)', fontsize=12)
        plt.title(f'Combined Charge vs Discharge Capacity ({len(batteries)} batteries)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "combined_qc_qd.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_current_time(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot combined current vs time curves."""
        plt.figure(figsize=(14, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(batteries)))
        
        for i, battery in enumerate(batteries):
            # Select a few cycles from each battery
            total_cycles = len(battery.cycle_data)
            if total_cycles > 0:
                # Select cycles: first, middle, and last
                selected_cycles = [0, total_cycles//2, total_cycles-1] if total_cycles > 2 else [0]
                
                for cycle_idx in selected_cycles:
                    cycle_data = battery.cycle_data[cycle_idx]
                    if (cycle_data.current_in_A and cycle_data.time_in_s and 
                        len(cycle_data.current_in_A) > 0 and len(cycle_data.time_in_s) > 0):
                        
                        current = np.array(cycle_data.current_in_A)
                        time = np.array(cycle_data.time_in_s)
                        
                        # Filter valid data
                        valid_mask = (~np.isnan(current)) & (~np.isnan(time))
                        if np.any(valid_mask):
                            label = f"{battery.cell_id}_C{cycle_data.cycle_number}" if len(selected_cycles) > 1 else battery.cell_id
                            plt.plot(time[valid_mask], current[valid_mask], 
                                   color=colors[i], linewidth=1, alpha=0.7,
                                   label=label)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Current (A)', fontsize=12)
        plt.title(f'Combined Current vs Time ({len(batteries)} batteries)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "combined_current_time.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_voltage_time(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot combined voltage vs time curves."""
        plt.figure(figsize=(14, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(batteries)))
        
        for i, battery in enumerate(batteries):
            # Select a few cycles from each battery
            total_cycles = len(battery.cycle_data)
            if total_cycles > 0:
                # Select cycles: first, middle, and last
                selected_cycles = [0, total_cycles//2, total_cycles-1] if total_cycles > 2 else [0]
                
                for cycle_idx in selected_cycles:
                    cycle_data = battery.cycle_data[cycle_idx]
                    if (cycle_data.voltage_in_V and cycle_data.time_in_s and 
                        len(cycle_data.voltage_in_V) > 0 and len(cycle_data.time_in_s) > 0):
                        
                        voltage = np.array(cycle_data.voltage_in_V)
                        time = np.array(cycle_data.time_in_s)
                        
                        # Filter valid data
                        valid_mask = (~np.isnan(voltage)) & (~np.isnan(time))
                        if np.any(valid_mask):
                            label = f"{battery.cell_id}_C{cycle_data.cycle_number}" if len(selected_cycles) > 1 else battery.cell_id
                            plt.plot(time[valid_mask], voltage[valid_mask], 
                                   color=colors[i], linewidth=1, alpha=0.7,
                                   label=label)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.title(f'Combined Voltage vs Time ({len(batteries)} batteries)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "combined_voltage_time.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_capacity_distribution(self, batteries: List[BatteryData], analyzer: BaseDataAnalyzer):
        """Plot capacity distribution statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect capacity data
        all_initial_capacities = []
        all_final_capacities = []
        all_cycle_lives = []
        all_fade_rates = []
        
        for battery in batteries:
            cycles, capacities = analyzer.calculate_capacity_fade(battery)
            if cycles and capacities and len(capacities) > 10:
                # Initial capacity (average of first 5 cycles)
                initial_cap = np.mean(capacities[:5])
                all_initial_capacities.append(initial_cap)
                
                # Final capacity (average of last 5 cycles)
                final_cap = np.mean(capacities[-5:])
                all_final_capacities.append(final_cap)
                
                # Cycle life
                cycle_life = len(cycles)
                all_cycle_lives.append(cycle_life)
                
                # Fade rate
                fade_rate = (initial_cap - final_cap) / cycle_life
                all_fade_rates.append(fade_rate)
        
        # Plot 1: Initial vs Final Capacity
        if all_initial_capacities and all_final_capacities:
            axes[0, 0].scatter(all_initial_capacities, all_final_capacities, alpha=0.7)
            axes[0, 0].plot([min(all_initial_capacities), max(all_initial_capacities)], 
                          [min(all_initial_capacities), max(all_initial_capacities)], 
                          'r--', alpha=0.7, label='y=x line')
            axes[0, 0].set_xlabel('Initial Capacity (Ah)')
            axes[0, 0].set_ylabel('Final Capacity (Ah)')
            axes[0, 0].set_title('Initial vs Final Capacity')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Plot 2: Cycle Life Distribution
        if all_cycle_lives:
            axes[0, 1].hist(all_cycle_lives, bins=10, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Cycle Life')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Cycle Life Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fade Rate Distribution
        if all_fade_rates:
            axes[1, 0].hist(all_fade_rates, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Fade Rate (Ah/cycle)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Capacity Fade Rate Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Capacity vs Cycle Life
        if all_initial_capacities and all_cycle_lives:
            axes[1, 1].scatter(all_initial_capacities, all_cycle_lives, alpha=0.7)
            axes[1, 1].set_xlabel('Initial Capacity (Ah)')
            axes[1, 1].set_ylabel('Cycle Life')
            axes[1, 1].set_title('Initial Capacity vs Cycle Life')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Capacity Distribution Analysis ({len(batteries)} batteries)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.combined_plots_dir / "capacity_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for combined plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate combined plots for battery data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='combined_analysis',
                       help='Output directory for combined plots')
    parser.add_argument('--num_batteries', type=int, default=20,
                       help='Number of random batteries to select')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BaseDataAnalyzer(args.data_path, args.output_dir)
    
    # Get battery files
    battery_files = analyzer.get_battery_files()
    if not battery_files:
        print(f"No battery files found in {args.data_path}")
        return
    
    # Create combined plot generator
    plot_generator = CombinedPlotGenerator(Path(args.output_dir), args.num_batteries)
    
    # Generate combined plots
    plot_generator.generate_combined_plots(battery_files, analyzer)
    
    print(f"Combined plots generated successfully!")
    print(f"Check the results in: {args.output_dir}/combined_plots/")


if __name__ == "__main__":
    main()
