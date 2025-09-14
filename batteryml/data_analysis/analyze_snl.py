# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
import numpy as np
from .base_analyzer import BaseDataAnalyzer


class SNLAnalyzer(BaseDataAnalyzer):
    """Analyzer for SNL battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "snl_analysis"):
        """
        Initialize SNL analyzer.
        
        Args:
            data_path: Path to SNL processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "SNL"
    
    def analyze_snl_specific_features(self):
        """Analyze SNL-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        snl_stats = {
            'cathode_materials': [],
            'temperature_groups': [],
            'soc_ranges': [],
            'discharge_rates': [],
            'capacity_ranges': [],
            'cycle_life_distribution': []
        }
        
        for file_path in battery_files:
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # Extract features from cell ID format: SNL_18650_NMC_25C_0-100_0.5-1C_a
            cell_id = battery.cell_id
            if 'SNL_' in cell_id:
                parts = cell_id.split('_')
                if len(parts) >= 6:
                    cathode = parts[2]  # NMC, NCA, LFP
                    temp = parts[3]    # 15C, 25C, 35C
                    soc_range = parts[4]  # 0-100, 20-80, 40-60
                    rates = parts[5]   # 0.5-1C, 0.5-2C, etc.
                    
                    snl_stats['cathode_materials'].append(cathode)
                    snl_stats['temperature_groups'].append(temp)
                    snl_stats['soc_ranges'].append(soc_range)
                    snl_stats['discharge_rates'].append(rates)
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                snl_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                snl_stats['cycle_life_distribution'].append(len(cycles))
        
        # Save SNL-specific statistics
        self._save_snl_summary(snl_stats)
    
    def _save_snl_summary(self, stats):
        """Save SNL-specific summary."""
        summary_lines = [
            f"SNL Dataset Analysis Summary",
            f"============================",
            f"",
            f"Cathode Material Distribution:",
        ]
        
        from collections import Counter
        cathode_counts = Counter(stats['cathode_materials'])
        for cathode, count in cathode_counts.items():
            summary_lines.append(f"  {cathode}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Temperature Group Distribution:",
        ])
        
        temp_counts = Counter(stats['temperature_groups'])
        for temp, count in temp_counts.items():
            summary_lines.append(f"  {temp}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"SOC Range Distribution:",
        ])
        
        soc_counts = Counter(stats['soc_ranges'])
        for soc, count in soc_counts.items():
            summary_lines.append(f"  {soc}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Discharge Rate Distribution:",
        ])
        
        rate_counts = Counter(stats['discharge_rates'])
        for rate, count in rate_counts.items():
            summary_lines.append(f"  {rate}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ])
        
        with open(self.output_dir / "snl_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"SNL-specific summary saved to {self.output_dir / 'snl_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include SNL-specific analysis."""
        super().analyze_dataset()
        self.analyze_snl_specific_features()


def main():
    """Main function to run SNL analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SNL battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to SNL processed data directory')
    parser.add_argument('--output_dir', type=str, default='snl_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = SNLAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
