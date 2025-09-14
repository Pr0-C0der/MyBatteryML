# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class CALCEAnalyzer(BaseDataAnalyzer):
    """Analyzer for CALCE battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "calce_analysis"):
        """
        Initialize CALCE analyzer.
        
        Args:
            data_path: Path to CALCE processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "CALCE"
    
    def analyze_calce_specific_features(self):
        """Analyze CALCE-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        calce_stats = {
            'cell_types': [],
            'capacity_ranges': [],
            'voltage_limits': [],
            'cycle_life_distribution': []
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing CALCE features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # CALCE specific analysis
            cell_id = battery.cell_id
            
            # Determine cell type from ID (CS vs CX)
            if 'CS' in cell_id.upper():
                calce_stats['cell_types'].append('CS (1.1 Ah)')
            elif 'CX' in cell_id.upper():
                calce_stats['cell_types'].append('CX (1.35 Ah)')
            else:
                calce_stats['cell_types'].append('Unknown')
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                calce_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                calce_stats['cycle_life_distribution'].append(len(cycles))
            
            # Voltage limits
            if battery.max_voltage_limit_in_V and battery.min_voltage_limit_in_V:
                calce_stats['voltage_limits'].append((
                    battery.min_voltage_limit_in_V, 
                    battery.max_voltage_limit_in_V
                ))
            
            # Clear battery from memory
            del battery
        
        # Save CALCE-specific statistics
        self._save_calce_summary(calce_stats)
    
    def _save_calce_summary(self, stats):
        """Save CALCE-specific summary."""
        summary_lines = [
            f"CALCE Dataset Analysis Summary",
            f"==============================",
            f"",
            f"Cell Type Distribution:",
        ]
        
        from collections import Counter
        cell_type_counts = Counter(stats['cell_types'])
        for cell_type, count in cell_type_counts.items():
            summary_lines.append(f"  {cell_type}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Voltage Limits:",
            f"  Min Voltage: {min([vl[0] for vl in stats['voltage_limits']]) if stats['voltage_limits'] else 'N/A'} V",
            f"  Max Voltage: {max([vl[1] for vl in stats['voltage_limits']]) if stats['voltage_limits'] else 'N/A'} V",
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ])
        
        with open(self.output_dir / "calce_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"CALCE-specific summary saved to {self.output_dir / 'calce_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include CALCE-specific analysis."""
        super().analyze_dataset()
        self.analyze_calce_specific_features()


def main():
    """Main function to run CALCE analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CALCE battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CALCE processed data directory')
    parser.add_argument('--output_dir', type=str, default='calce_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = CALCEAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
