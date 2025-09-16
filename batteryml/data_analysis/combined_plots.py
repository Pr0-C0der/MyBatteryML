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
        
        # Generate combined plots - only capacity fade
        self._plot_combined_capacity_fade(selected_batteries, analyzer)
        
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
