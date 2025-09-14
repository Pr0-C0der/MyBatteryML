# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from .base_analyzer import BaseDataAnalyzer


class UL_PURAnalyzer(BaseDataAnalyzer):
    """Analyzer for UL_PUR battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "ul_pur_analysis"):
        """
        Initialize UL_PUR analyzer.
        
        Args:
            data_path: Path to UL_PUR processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "UL_PUR"
    
    def analyze_ul_pur_specific_features(self):
        """Analyze UL_PUR-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        ul_pur_stats = {
            'cell_types': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'voltage_ranges': []
        }
        
        for file_path in battery_files:
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # UL_PUR specific analysis
            cell_id = battery.cell_id
            
            # Determine cell type from ID
            if 'N15' in cell_id:
                ul_pur_stats['cell_types'].append('N15')
            elif 'N20' in cell_id:
                ul_pur_stats['cell_types'].append('N20')
            else:
                ul_pur_stats['cell_types'].append('Unknown')
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                ul_pur_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                ul_pur_stats['cycle_life_distribution'].append(len(cycles))
            
            # Voltage range analysis
            all_voltages = []
            for cycle_data in battery.cycle_data:
                if cycle_data.voltage_in_V:
                    all_voltages.extend(cycle_data.voltage_in_V)
            
            if all_voltages:
                ul_pur_stats['voltage_ranges'].append((min(all_voltages), max(all_voltages)))
        
        # Save UL_PUR-specific statistics
        self._save_ul_pur_summary(ul_pur_stats)
    
    def _save_ul_pur_summary(self, stats):
        """Save UL_PUR-specific summary."""
        summary_lines = [
            f"UL_PUR Dataset Analysis Summary",
            f"===============================",
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
        
        with open(self.output_dir / "ul_pur_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"UL_PUR-specific summary saved to {self.output_dir / 'ul_pur_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include UL_PUR-specific analysis."""
        super().analyze_dataset()
        self.analyze_ul_pur_specific_features()


def main():
    """Main function to run UL_PUR analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze UL_PUR battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to UL_PUR processed data directory')
    parser.add_argument('--output_dir', type=str, default='ul_pur_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = UL_PURAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
