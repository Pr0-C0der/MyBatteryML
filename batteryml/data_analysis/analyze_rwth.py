# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class RWTHAnalyzer(BaseDataAnalyzer):
    """Analyzer for RWTH battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "rwth_analysis"):
        """
        Initialize RWTH analyzer.
        
        Args:
            data_path: Path to RWTH processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "RWTH"
    
    def analyze_rwth_specific_features(self):
        """Analyze RWTH-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        rwth_stats = {
            'cell_ids': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'voltage_ranges': []
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing RWTH features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # RWTH specific analysis
            cell_id = battery.cell_id
            rwth_stats['cell_ids'].append(cell_id)
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                rwth_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                rwth_stats['cycle_life_distribution'].append(len(cycles))
            
            # Voltage range analysis
            all_voltages = []
            for cycle_data in battery.cycle_data:
                if cycle_data.voltage_in_V:
                    all_voltages.extend(cycle_data.voltage_in_V)
            
            if all_voltages:
                rwth_stats['voltage_ranges'].append((min(all_voltages), max(all_voltages)))
            
            # Clear battery from memory
            del battery
        
        # Save RWTH-specific statistics
        self._save_rwth_summary(rwth_stats)
    
    def _save_rwth_summary(self, stats):
        """Save RWTH-specific summary."""
        summary_lines = [
            f"RWTH Dataset Analysis Summary",
            f"=============================",
            f"",
            f"Total Cells: {len(stats['cell_ids'])}",
            f"Cell IDs: {', '.join(stats['cell_ids'])}",
            f"",
            f"Voltage Range:",
            f"  Min Voltage: {min([vr[0] for vr in stats['voltage_ranges']]) if stats['voltage_ranges'] else 'N/A'} V",
            f"  Max Voltage: {max([vr[1] for vr in stats['voltage_ranges']]) if stats['voltage_ranges'] else 'N/A'} V",
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ]
        
        with open(self.output_dir / "rwth_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"RWTH-specific summary saved to {self.output_dir / 'rwth_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include RWTH-specific analysis."""
        super().analyze_dataset()
        self.analyze_rwth_specific_features()


def main():
    """Main function to run RWTH analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze RWTH battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to RWTH processed data directory')
    parser.add_argument('--output_dir', type=str, default='rwth_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = RWTHAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
