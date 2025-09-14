# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from pathlib import Path
import numpy as np
from tqdm import tqdm
from .base_analyzer import BaseDataAnalyzer


class MATRAnalyzer(BaseDataAnalyzer):
    """Analyzer for MATR battery dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "matr_analysis"):
        """
        Initialize MATR analyzer.
        
        Args:
            data_path: Path to MATR processed data directory
            output_dir: Directory to save analysis results
        """
        super().__init__(data_path, output_dir)
        self.dataset_name = "MATR"
    
    def analyze_matr_specific_features(self):
        """Analyze MATR-specific features."""
        print(f"Analyzing {self.dataset_name} specific features...")
        
        battery_files = self.get_battery_files()
        matr_stats = {
            'batch_distribution': [],
            'charge_policies': [],
            'internal_resistance': [],
            'capacity_ranges': [],
            'cycle_life_distribution': [],
            'qdlin_features': []
        }
        
        # Process batteries one at a time to avoid memory issues
        for file_path in tqdm(battery_files, desc="Analyzing MATR features"):
            battery = self.load_battery_data(file_path)
            if not battery:
                continue
            
            # Extract batch information from cell ID (e.g., "MATR_b1c0" -> "b1")
            cell_id = battery.cell_id
            if 'MATR_' in cell_id:
                batch_id = cell_id.split('_')[1][:2]  # Extract batch part
                matr_stats['batch_distribution'].append(batch_id)
            
            # Analyze charge protocols
            if battery.charge_protocol:
                protocol_str = str(battery.charge_protocol)
                matr_stats['charge_policies'].append(protocol_str)
            
            # Analyze internal resistance
            resistance_values = []
            for cycle_data in battery.cycle_data:
                if hasattr(cycle_data, 'internal_resistance_in_ohm') and cycle_data.internal_resistance_in_ohm:
                    resistance_values.append(cycle_data.internal_resistance_in_ohm)
            
            if resistance_values:
                matr_stats['internal_resistance'].append({
                    'min': min(resistance_values),
                    'max': max(resistance_values),
                    'mean': np.mean(resistance_values),
                    'std': np.std(resistance_values)
                })
            
            # Capacity analysis
            cycles, capacities = self.calculate_capacity_fade(battery)
            if capacities:
                matr_stats['capacity_ranges'].append((min(capacities), max(capacities)))
                matr_stats['cycle_life_distribution'].append(len(cycles))
            
            # Qdlin analysis (MATR specific feature)
            qdlin_stats = self._analyze_qdlin_features(battery)
            if qdlin_stats:
                matr_stats['qdlin_features'].append(qdlin_stats)
            
            # Clear battery from memory
            del battery
        
        # Save MATR-specific statistics
        self._save_matr_summary(matr_stats)
    
    def _analyze_qdlin_features(self, battery):
        """Analyze Qdlin features specific to MATR dataset."""
        qdlin_stats = {
            'cycle_count': 0,
            'qdlin_lengths': [],
            'qdlin_ranges': []
        }
        
        for cycle_data in battery.cycle_data:
            if hasattr(cycle_data, 'Qdlin') and cycle_data.Qdlin:
                qdlin_values = cycle_data.Qdlin
                if isinstance(qdlin_values, list) and len(qdlin_values) > 0:
                    qdlin_stats['cycle_count'] += 1
                    qdlin_stats['qdlin_lengths'].append(len(qdlin_values))
                    qdlin_stats['qdlin_ranges'].append((min(qdlin_values), max(qdlin_values)))
        
        return qdlin_stats if qdlin_stats['cycle_count'] > 0 else None
    
    def _save_matr_summary(self, stats):
        """Save MATR-specific summary."""
        summary_lines = [
            f"MATR Dataset Analysis Summary",
            f"=============================",
            f"",
            f"Batch Distribution:",
        ]
        
        from collections import Counter
        batch_counts = Counter(stats['batch_distribution'])
        for batch, count in batch_counts.items():
            summary_lines.append(f"  {batch}: {count} cells")
        
        summary_lines.extend([
            f"",
            f"Charge Policy Distribution:",
        ])
        
        policy_counts = Counter(stats['charge_policies'])
        for policy, count in policy_counts.items():
            summary_lines.append(f"  {policy}: {count} cells")
        
        # Internal resistance statistics
        if stats['internal_resistance']:
            all_resistances = []
            for res_data in stats['internal_resistance']:
                all_resistances.extend([res_data['min'], res_data['max']])
            
            summary_lines.extend([
                f"",
                f"Internal Resistance (Ohm):",
                f"  Min: {min(all_resistances):.4f}",
                f"  Max: {max(all_resistances):.4f}",
                f"  Mean: {np.mean(all_resistances):.4f}",
                f"  Std: {np.std(all_resistances):.4f}",
            ])
        
        # Qdlin statistics
        if stats['qdlin_features']:
            all_lengths = []
            all_ranges = []
            for qdlin_data in stats['qdlin_features']:
                all_lengths.extend(qdlin_data['qdlin_lengths'])
                all_ranges.extend([r[1] - r[0] for r in qdlin_data['qdlin_ranges']])
            
            summary_lines.extend([
                f"",
                f"Qdlin Features:",
                f"  Cycles with Qdlin: {len(stats['qdlin_features'])}",
                f"  Qdlin Length - Mean: {np.mean(all_lengths):.1f}",
                f"  Qdlin Length - Std: {np.std(all_lengths):.1f}",
                f"  Qdlin Range - Mean: {np.mean(all_ranges):.4f}",
                f"  Qdlin Range - Std: {np.std(all_ranges):.4f}",
            ])
        
        summary_lines.extend([
            f"",
            f"Cycle Life Statistics:",
            f"  Mean: {sum(stats['cycle_life_distribution']) / len(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Min: {min(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
            f"  Max: {max(stats['cycle_life_distribution']) if stats['cycle_life_distribution'] else 'N/A'}",
        ])
        
        with open(self.output_dir / "matr_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"MATR-specific summary saved to {self.output_dir / 'matr_summary.txt'}")
    
    def analyze_dataset(self):
        """Override to include MATR-specific analysis."""
        super().analyze_dataset()
        self.analyze_matr_specific_features()


def main():
    """Main function to run MATR analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze MATR battery dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to MATR processed data directory')
    parser.add_argument('--output_dir', type=str, default='matr_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = MATRAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
