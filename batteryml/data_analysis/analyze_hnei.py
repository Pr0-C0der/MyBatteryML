# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class HNEIAnalyzer(BaseDataAnalyzer):
    """Analyzer for HNEI battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "hnei_analysis"):
        """
        Initialize HNEI analyzer.
        
        Args:
            data_path: Path to HNEI processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "HNEI"
    
    def analyze_hnei_specific_features(self):
        """Analyze HNEI-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        hnei_stats = {
            'cell_types': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'voltage_ranges': []
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing HNEI features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # HNEI specific analysis
            cell_id = battery.cell_id
            
            # Determine cell type from ID
            if 'NMC_LCO' in cell_id:
                hnei_stats['cell_types'].append('NMC_LCO')
            else:
                hnei_stats['cell_types'].append('Unknown')
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                hnei_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                hnei_stats['cycle_life_distribution'].append(len(cycles))
            
            # Voltage range analysis
            all_voltages = []
            for cycle_data in battery.cycle_data:
                if cycle_data.voltage_in_V:
                    all_voltages.extend(cycle_data.voltage_in_V)
            
            if all_voltages:
                hnei_stats['voltage_ranges'].append((min(all_voltages), max(all_voltages)))
            
            # Clear battery from memory
            del battery
        
        # Save HNEI-specific statistics
        self._save_hnei_summary(hnei_stats)
    
    def _save_hnei_summary(self, stats):
        """Save HNEI-specific summary."""
        summary_lines = [
            f"HNEI Dataset Analysis Summary",
            f"=============================",
            f"",
            f"Cell Type Distribution:",
        ]
        
        from collections import Counter
        cell_type_counts = Counter(stats['cell_types'])
        for cell_type, count in cell_type_counts.items():
            summary_lines.append(f"  {cell_type}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Voltage Range:",
            f"  Min Voltage: {min([vr[0] for vr in stats['voltage_ranges']]) if stats['voltage_ranges'] else 'N/A'} V",
            f"  Max Voltage: {max([vr[1] for vr in stats['voltage_ranges']]) if stats['voltage_ranges'] else 'N/A'} V",
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ])
        
        with open(self.output_dir / "hnei_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"HNEI-specific summary saved to {self.output_dir / 'hnei_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include HNEI-specific analysis."""
        super().analyze_dataset()
        self.analyze_hnei_specific_features()


def main():
    """Main function to run HNEI analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze HNEI battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to HNEI processed data directory')
    parser.add_argument('--output_dir', type=str, default='hnei_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = HNEIAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
