# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class HUSTAnalyzer(BaseDataAnalyzer):
    """Analyzer for HUST battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "hust_analysis"):
        """
        Initialize HUST analyzer.
        
        Args:
            data_path: Path to HUST processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "HUST"
    
    def analyze_hust_specific_features(self):
        """Analyze HUST-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        hust_stats = {
            'discharge_rate_groups': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'temperature_ranges': []
        }
        
        # HUST discharge rate mapping (from preprocessing code)
        discharge_rates = {
            '1-1': [5, 1, 1], '1-2': [5, 1, 2], '1-3': [5, 1, 3], '1-4': [5, 1, 4], '1-5': [5, 1, 5],
            '1-6': [5, 2, 1], '1-7': [5, 2, 2], '1-8': [5, 2, 3], '2-2': [5, 2, 5], '2-3': [5, 3, 1],
            '2-4': [5, 3, 2], '2-5': [5, 3, 3], '2-6': [5, 3, 4], '2-7': [5, 3, 5], '2-8': [5, 4, 1],
            '3-1': [5, 4, 2], '3-2': [5, 4, 3], '3-3': [5, 4, 4], '3-4': [5, 4, 5], '3-5': [5, 5, 1],
            '3-6': [5, 5, 2], '3-7': [5, 5, 3], '3-8': [5, 5, 4], '4-1': [5, 5, 5], '4-2': [4, 1, 1],
            '4-3': [4, 1, 2], '4-4': [4, 1, 3], '4-5': [4, 1, 4], '4-6': [4, 1, 5], '4-7': [4, 2, 1],
            '4-8': [4, 2, 2], '5-1': [4, 2, 3], '5-2': [4, 2, 4], '5-3': [4, 2, 5], '5-4': [4, 3, 1],
            '5-5': [4, 3, 2], '5-6': [4, 3, 3], '5-7': [4, 3, 4], '6-1': [4, 4, 1], '6-2': [4, 4, 2],
            '6-3': [4, 4, 3], '6-4': [4, 4, 4], '6-5': [4, 4, 5], '6-6': [4, 5, 1], '6-8': [4, 5, 3],
            '7-1': [4, 5, 4], '7-2': [4, 5, 5], '7-3': [3, 1, 1], '7-4': [3, 1, 2], '7-5': [3, 1, 3],
            '7-6': [3, 1, 4], '7-7': [3, 1, 5], '7-8': [3, 2, 1], '8-1': [3, 2, 2], '8-2': [3, 2, 3],
            '8-3': [3, 2, 4], '8-4': [3, 2, 5], '8-5': [3, 3, 1], '8-6': [3, 3, 2], '8-7': [3, 3, 3],
            '8-8': [3, 3, 4], '9-1': [3, 3, 5], '9-2': [3, 4, 1], '9-3': [3, 4, 2], '9-4': [3, 4, 3],
            '9-5': [3, 4, 4], '9-6': [3, 4, 5], '9-7': [3, 5, 1], '9-8': [3, 5, 2], '10-1': [3, 5, 3],
            '10-2': [3, 5, 4], '10-3': [3, 5, 5], '10-4': [2, 4, 1], '10-5': [2, 5, 2], '10-6': [2, 3, 3],
            '10-7': [2, 2, 4], '10-8': [2, 1, 5]
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing HUST features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # Extract cell ID from HUST format (e.g., "HUST_1-1" -> "1-1")
            cell_id = battery.cell_id.replace('HUST_', '')
            
            # Get discharge rate group
            if cell_id in discharge_rates:
                rates = discharge_rates[cell_id]
                hust_stats['discharge_rate_groups'].append(f"{rates[0]}-{rates[1]}-{rates[2]}C")
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                hust_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                hust_stats['cycle_life_distribution'].append(len(cycles))
            
            # Temperature analysis
            all_temps = []
            for cycle_data in battery.cycle_data:
                if cycle_data.temperature_in_C:
                    all_temps.extend(cycle_data.temperature_in_C)
            
            if all_temps:
                hust_stats['temperature_ranges'].append((min(all_temps), max(all_temps)))
            
            # Clear battery from memory
            del battery
        
        # Save HUST-specific statistics
        self._save_hust_summary(hust_stats)
    
    def _save_hust_summary(self, stats):
        """Save HUST-specific summary."""
        summary_lines = [
            f"HUST Dataset Analysis Summary",
            f"=============================",
            f"",
            f"Discharge Rate Groups:",
        ]
        
        from collections import Counter
        rate_counts = Counter(stats['discharge_rate_groups'])
        for rate_group, count in rate_counts.items():
            summary_lines.append(f"  {rate_group}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Temperature Range:",
            f"  Min Temperature: {min([tr[0] for tr in stats['temperature_ranges']]) if stats['temperature_ranges'] else 'N/A'} °C",
            f"  Max Temperature: {max([tr[1] for tr in stats['temperature_ranges']]) if stats['temperature_ranges'] else 'N/A'} °C",
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ])
        
        with open(self.output_dir / "hust_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"HUST-specific summary saved to {self.output_dir / 'hust_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include HUST-specific analysis."""
        super().analyze_dataset()
        self.analyze_hust_specific_features()


def main():
    """Main function to run HUST analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze HUST battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to HUST processed data directory')
    parser.add_argument('--output_dir', type=str, default='hust_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = HUSTAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
