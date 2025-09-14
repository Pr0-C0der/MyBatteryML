# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class OXAnalyzer(BaseDataAnalyzer):
    """Analyzer for OX battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "ox_analysis"):
        """
        Initialize OX analyzer.
        
        Args:
            data_path: Path to OX processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "OX"
    
    def analyze_ox_specific_features(self):
        """Analyze OX-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        ox_stats = {
            'cell_ids': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'voltage_ranges': []
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing OX features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # OX specific analysis
            cell_id = battery.cell_id
            ox_stats['cell_ids'].append(cell_id)
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                ox_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                ox_stats['cycle_life_distribution'].append(len(cycles))
            
            # Voltage range analysis
            all_voltages = []
            for cycle_data in battery.cycle_data:
                if cycle_data.voltage_in_V:
                    all_voltages.extend(cycle_data.voltage_in_V)
            
            if all_voltages:
                ox_stats['voltage_ranges'].append((min(all_voltages), max(all_voltages)))
            
            # Clear battery from memory
            del battery
        
        # Save OX-specific statistics
        self._save_ox_summary(ox_stats)
    
    def _save_ox_summary(self, stats):
        """Save OX-specific summary."""
        summary_lines = [
            f"OX Dataset Analysis Summary",
            f"===========================",
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
        
        with open(self.output_dir / "ox_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"OX-specific summary saved to {self.output_dir / 'ox_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include OX-specific analysis."""
        super().analyze_dataset()
        self.analyze_ox_specific_features()


def main():
    """Main function to run OX analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze OX battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to OX processed data directory')
    parser.add_argument('--output_dir', type=str, default='ox_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = OXAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
