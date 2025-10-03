from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from scipy import stats

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, DatasetSpecificCycleFeatures


class RULDistributionAnalyzer:
    """Analyzer for RUL distribution across different chemistries."""
    
    def __init__(self, data_path: str, output_dir: str = 'chemistry_rul_distributions', 
                 verbose: bool = False, dataset_hint: Optional[str] = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rul_annotator = RULLabelAnnotator()
        
        # Chemistry name from data path
        self.chemistry_name = self.data_path.name
        
        # Create output subdirectories
        self.distributions_dir = self.output_dir / self.chemistry_name
        self.distributions_dir.mkdir(parents=True, exist_ok=True)

    def _battery_files(self) -> List[Path]:
        """Get all battery files in the chemistry folder."""
        return sorted(self.data_path.glob('*.pkl'))

    def _infer_dataset_for_battery(self, battery: BatteryData) -> Optional[str]:
        """Infer dataset for a battery."""
        if self.dataset_hint:
            return str(self.dataset_hint).upper()
        
        # Try tokens in metadata - handle both UL_PUR and UL-PUR patterns
        tokens = ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'UL-PUR', 'HUST', 'OX']
        
        def _txt(x):
            try:
                return str(x).upper()
            except Exception:
                return ''
        
        # Check cell_id first
        cell_id = _txt(battery.cell_id)
        for t in tokens:
            if t in cell_id:
                # Normalize UL-PUR to UL_PUR for consistency
                return 'UL_PUR' if t in ['UL_PUR', 'UL-PUR'] else t
        
        # Check reference and description
        for source in [getattr(battery, 'reference', ''), getattr(battery, 'description', '')]:
            s = _txt(source)
            for t in tokens:
                if t in s:
                    # Normalize UL-PUR to UL_PUR for consistency
                    return 'UL_PUR' if t in ['UL_PUR', 'UL-PUR'] else t
        
        # If no dataset found, return None
        if self.verbose:
            print(f"[warn] Could not infer dataset for battery: {battery.cell_id}")
        return None

    def analyze_chemistry(self) -> Dict[str, float]:
        """Analyze RUL distribution for this chemistry."""
        files = self._battery_files()
        if not files:
            if self.verbose:
                print(f"No battery files found in {self.data_path}")
            return {}
        
        if self.verbose:
            print(f"Analyzing RUL distribution for {self.chemistry_name}...")
            print(f"Found {len(files)} battery files")
        
        # Collect RUL values
        rul_values = []
        battery_info = []
        
        for f in tqdm(files, desc="Processing batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            
            # Debug battery information
            if self.verbose:
                print(f"Processing: {f.name}")
                print(f"  Cell ID: {battery.cell_id}")
                print(f"  Cycles: {len(battery.cycle_data)}")
                print(f"  Inferred dataset: {self._infer_dataset_for_battery(battery)}")
            
            # Calculate RUL
            try:
                rul_tensor = self.rul_annotator.process_cell(battery)
                rul_value = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
                if self.verbose:
                    print(f"  RUL: {rul_value}")
            except Exception as e:
                if self.verbose:
                    print(f"[warn] RUL calculation failed for {f}: {e}")
                rul_value = 0
            
            if rul_value > 0:
                rul_values.append(rul_value)
                battery_info.append({
                    'cell_id': battery.cell_id,
                    'rul': rul_value,
                    'dataset': self._infer_dataset_for_battery(battery),
                    'file': f.name
                })
            elif self.verbose:
                print(f"  Skipped (RUL <= 0)")
        
        if not rul_values:
            if self.verbose:
                print("No valid RUL values found")
            return {}
        
        # Calculate statistics
        rul_array = np.array(rul_values)
        stats = {
            'count': len(rul_values),
            'mean': float(np.mean(rul_array)),
            'std': float(np.std(rul_array)),
            'min': float(np.min(rul_array)),
            'max': float(np.max(rul_array)),
            'median': float(np.median(rul_array)),
            'q25': float(np.percentile(rul_array, 25)),
            'q75': float(np.percentile(rul_array, 75))
        }
        
        if self.verbose:
            print(f"RUL Statistics for {self.chemistry_name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']}")
            print(f"  Max: {stats['max']}")
            print(f"  Median: {stats['median']:.2f}")
            print(f"  Q25: {stats['q25']:.2f}")
            print(f"  Q75: {stats['q75']:.2f}")
        
        # Save detailed data
        df = pd.DataFrame(battery_info)
        csv_path = self.distributions_dir / f"{self.chemistry_name}_rul_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Plot distribution curve
        self.plot_distribution(rul_values, self.chemistry_name)
        
        return stats

    def plot_distribution(self, rul_values: List[float], chemistry_name: str) -> None:
        """Plot RUL distribution curve for a single chemistry."""
        if not rul_values:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create density curve using KDE
        rul_array = np.array(rul_values)
        
        # Create KDE
        kde = stats.gaussian_kde(rul_array)
        x_range = np.linspace(rul_array.min(), rul_array.max(), 200)
        density = kde(x_range)
        
        # Plot the curve
        plt.plot(x_range, density, linewidth=2, alpha=0.8, label=f'{chemistry_name} Distribution')
        plt.fill_between(x_range, density, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(rul_values)
        median_val = np.median(rul_values)
        std_val = np.std(rul_values)
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        
        plt.xlabel('RUL (cycles)')
        plt.ylabel('Density')
        plt.title(f'RUL Distribution Curve - {chemistry_name}\n(n={len(rul_values)}, μ={mean_val:.1f}, σ={std_val:.1f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.distributions_dir / f"{chemistry_name}_rul_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Saved distribution plot: {plot_path}")


def analyze_all_chemistries(chemistry_dirs: List[str], output_dir: str = 'chemistry_rul_distributions', 
                           verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """Analyze RUL distributions for all chemistries and create combined plot."""
    all_stats = {}
    all_rul_data = {}
    
    # Analyze each chemistry
    for chem_dir in chemistry_dirs:
        analyzer = RULDistributionAnalyzer(chem_dir, output_dir, verbose)
        stats = analyzer.analyze_chemistry()
        if stats:
            all_stats[analyzer.chemistry_name] = stats
            
            # Load the RUL data for combined plotting
            csv_path = analyzer.distributions_dir / f"{analyzer.chemistry_name}_rul_data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_rul_data[analyzer.chemistry_name] = df['rul'].tolist()
    
    # Create combined distribution plot
    if all_rul_data:
        plot_combined_distributions(all_rul_data, output_dir, verbose)
    
    return all_stats


def plot_combined_distributions(all_rul_data: Dict[str, List[float]], output_dir: str, verbose: bool = False) -> None:
    """Create a combined plot showing RUL distribution curves for all chemistries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create KDE curves for each chemistry
    # Calculate global min/max for consistent x-axis
    all_values = []
    for rul_values in all_rul_data.values():
        all_values.extend(rul_values)
    x_min, x_max = min(all_values), max(all_values)
    x_range = np.linspace(x_min, x_max, 300)
    
    # Plot curves for each chemistry
    for i, (chemistry, rul_values) in enumerate(all_rul_data.items()):
        rul_array = np.array(rul_values)
        
        # Create KDE
        kde = stats.gaussian_kde(rul_array)
        density = kde(x_range)
        
        # Normalize density to make curves comparable
        density = density / np.max(density)
        
        # Plot the curve
        color = colors[i % len(colors)]
        plt.plot(x_range, density, linewidth=2.5, alpha=0.8, 
                label=f'{chemistry}', color=color)
        plt.fill_between(x_range, density, alpha=0.2, color=color)
    
    # Add statistics text box
    stats_text = []
    for chemistry, rul_values in all_rul_data.items():
        mean_val = np.mean(rul_values)
        std_val = np.std(rul_values)
        n_val = len(rul_values)
        stats_text.append(f'{chemistry}: n={n_val}, μ={mean_val:.1f}, σ={std_val:.1f}')
    
    # Add text box with statistics
    stats_str = '\n'.join(stats_text)
    plt.text(0.02, 0.98, stats_str, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontsize=10, family='monospace')
    
    plt.xlabel('RUL (cycles)', fontsize=12)
    plt.ylabel('Normalized Density', fontsize=12)
    plt.title('RUL Distribution Curves Across All Chemistries', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save combined plot
    combined_path = output_path / 'all_chemistries_rul_distributions.png'
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved combined distribution plot: {combined_path}")
    
    # Create box plot comparison
    plot_boxplot_comparison(all_rul_data, output_path, verbose)


def plot_boxplot_comparison(all_rul_data: Dict[str, List[float]], output_path: Path, verbose: bool = False) -> None:
    """Create a box plot comparing RUL distributions across chemistries."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    data_for_box = []
    labels = []
    
    for chemistry, rul_values in all_rul_data.items():
        data_for_box.append(rul_values)
        labels.append(f'{chemistry}\n(n={len(rul_values)})')
    
    # Create box plot
    box_plot = plt.boxplot(data_for_box, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Chemistry')
    plt.ylabel('RUL (cycles)')
    plt.title('RUL Distribution Comparison Across Chemistries', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = []
    for chemistry, rul_values in all_rul_data.items():
        mean_val = np.mean(rul_values)
        std_val = np.std(rul_values)
        stats_text.append(f'{chemistry}: μ={mean_val:.1f}, σ={std_val:.1f}')
    
    # Add text box with statistics
    stats_str = '\n'.join(stats_text)
    plt.text(0.02, 0.98, stats_str, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save box plot
    boxplot_path = output_path / 'all_chemistries_rul_boxplot.png'
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved box plot comparison: {boxplot_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze RUL distributions across chemistries')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to chemistry folder containing *.pkl files')
    parser.add_argument('--output_dir', type=str, default='chemistry_rul_distributions',
                       help='Output directory for results')
    parser.add_argument('--dataset_hint', type=str, default=None,
                       help='Optional dataset name hint')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    
    analyzer = RULDistributionAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dataset_hint=args.dataset_hint
    )
    
    stats = analyzer.analyze_chemistry()
    if stats:
        # Load RUL values for plotting
        csv_path = analyzer.distributions_dir / f"{analyzer.chemistry_name}_rul_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rul_values = df['rul'].tolist()
            analyzer.plot_distribution(rul_values, analyzer.chemistry_name)
    
    if args.verbose:
        print(f"Analysis completed for {analyzer.chemistry_name}")


if __name__ == '__main__':
    main()
