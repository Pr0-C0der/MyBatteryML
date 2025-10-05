#!/usr/bin/env python3
"""
Find Common Statistical Correlations Across Datasets

This script processes all datasets (MATR, CALCE, HNEI, OX, RWTH, SNL, HUST, UL_PUR)
and finds statistical features that have high correlation (> 0.5) with log(RUL)
that are common across most datasets (more than 2).

The script:
1. Loads battery data from each dataset
2. Calculates statistical features (mean, std, min, max, median, q25, q75)
3. Computes correlations with log(RUL) for each dataset
4. Identifies features with high correlation (> 0.5) in multiple datasets
5. Saves results and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import argparse
from collections import defaultdict

# Import the working correlation analyzer
from .feature_rul_correlation import FeatureRULCorrelationAnalyzer
from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')


class CommonStatisticalCorrelationFinder:
    """Find common statistical correlations across multiple datasets."""

    def __init__(self, data_path: str = 'data/preprocessed', output_dir: str = 'common_correlations_results',
                 correlation_threshold: float = 0.5, min_datasets: int = 3, verbose: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.correlation_threshold = correlation_threshold
        self.min_datasets = min_datasets
        self.verbose = bool(verbose)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rul_annotator = RULLabelAnnotator()
        
        # Statistical measures to calculate
        self.statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis']
        
        # Dataset names to process
        self.dataset_names = ['MATR', 'CALCE', 'HNEI', 'OX', 'RWTH', 'SNL', 'HUST', 'UL_PUR']
        
        # Feature names to analyze
        self.feature_names = [
            'avg_c_rate', 'max_temperature', 'max_discharge_capacity', 'max_charge_capacity',
            'avg_discharge_capacity', 'avg_charge_capacity', 'charge_cycle_length', 
            'discharge_cycle_length', 'peak_cv_length', 'cycle_length', 
            'power_during_charge_cycle', 'power_during_discharge_cycle',
            'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio',
            'avg_voltage', 'avg_current'
        ]
        
        # Results storage
        self.dataset_results = {}
        self.common_features = {}
        self.correlation_summary = pd.DataFrame()

    def _infer_dataset_for_battery(self, battery: BatteryData, dataset_name: str) -> bool:
        """Check if battery belongs to the specified dataset."""
        # Try tokens in metadata - handle both UL_PUR and UL-PUR patterns
        tokens = ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'UL-PUR', 'HUST', 'OX']
        
        def _txt(x):
            try:
                return str(x).upper()
            except Exception:
                return ''
        
        for source in [battery.cell_id, getattr(battery, 'reference', ''), getattr(battery, 'description', '')]:
            s = _txt(source)
            for t in tokens:
                if t in s:
                    # Normalize UL-PUR to UL_PUR for consistency
                    normalized_token = 'UL_PUR' if t in ['UL_PUR', 'UL-PUR'] else t
                    return normalized_token == dataset_name
        return False

    def _build_cycle_feature_table_extractor(self, battery: BatteryData, feature_names: List[str]) -> pd.DataFrame:
        """Build cycle feature table using dataset-specific extractor."""
        # Try to infer dataset
        dataset = None
        for ds_name in self.dataset_names:
            if self._infer_dataset_for_battery(battery, ds_name):
                dataset = ds_name
                break
        
        if not dataset:
            return pd.DataFrame()
        
        cls = get_extractor_class(dataset)
        if cls is None:
            return pd.DataFrame()
        
        extractor = cls()
        rows: List[Dict[str, float]] = []
        
        # Add progress bar for cycle processing if there are many cycles
        cycle_data = battery.cycle_data
        if len(cycle_data) > 100:  # Only show progress bar for batteries with many cycles
            cycle_data = tqdm(cycle_data, desc=f"Extracting features for {battery.cell_id}", 
                            unit="cycle", leave=False)
        
        for c in cycle_data:
            row: Dict[str, float] = {'cycle_number': c.cycle_number}
            for name in feature_names:
                fn = getattr(extractor, name, None)
                if not callable(fn):
                    continue
                try:
                    val = fn(battery, c)
                    row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
                except Exception:
                    row[name] = np.nan
            rows.append(row)
        
        return pd.DataFrame(rows)

    def _calculate_statistical_measure(self, values: np.ndarray, measure: str) -> float:
        """Calculate a statistical measure for given values."""
        try:
            if measure == 'mean':
                return np.mean(values)
            elif measure == 'std':
                return np.std(values)
            elif measure == 'min':
                return np.min(values)
            elif measure == 'max':
                return np.max(values)
            elif measure == 'median':
                return np.median(values)
            elif measure == 'q25':
                return np.percentile(values, 25)
            elif measure == 'q75':
                return np.percentile(values, 75)
            elif measure == 'skewness':
                from scipy.stats import skew
                return skew(values)
            elif measure == 'kurtosis':
                from scipy.stats import kurtosis
                return kurtosis(values)
            else:
                return np.nan
        except Exception:
            return np.nan

    def process_dataset(self, dataset_name: str) -> Dict:
        """Process a single dataset and calculate statistical correlations."""
        dataset_path = self.data_path / dataset_name
        
        if not dataset_path.exists():
            if self.verbose:
                print(f"Dataset path {dataset_path} does not exist, skipping...")
            return {}
        
        # Get all battery files
        battery_files = list(dataset_path.glob('*.pkl'))
        
        if not battery_files:
            if self.verbose:
                print(f"No PKL files found in {dataset_path}")
            return {}
        
        if self.verbose:
            print(f"Processing {dataset_name}: {len(battery_files)} batteries")
        
        # Calculate statistical features for all batteries
        all_battery_data = []
        
        for battery_file in tqdm(battery_files, desc=f"Processing {dataset_name}", unit="battery"):
            try:
                battery = BatteryData.load(battery_file)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load {battery_file}: {e}")
                continue
            
            # Build cycle feature table
            try:
                df = self._build_cycle_feature_table_extractor(battery, self.feature_names)
            except Exception:
                df = pd.DataFrame()
            
            if df.empty:
                continue
            
            # Calculate statistical measures for each feature
            battery_row = {'battery_id': battery.cell_id}
            
            # Add progress bar for feature processing if there are many features
            features_to_process = self.feature_names
            if len(features_to_process) > 10:  # Only show progress bar for many features
                features_to_process = tqdm(features_to_process, desc=f"Calculating stats for {battery.cell_id}", 
                                        unit="feature", leave=False)
            
            for feature in features_to_process:
                if feature in df.columns:
                    feature_values = df[feature].dropna().values
                    
                    if len(feature_values) > 0:
                        for measure in self.statistical_measures:
                            stat_value = self._calculate_statistical_measure(feature_values, measure)
                            battery_row[f"{feature}_{measure}"] = stat_value
                    else:
                        # Fill with NaN if no valid values
                        for measure in self.statistical_measures:
                            battery_row[f"{feature}_{measure}"] = np.nan
                else:
                    # Fill with NaN if feature not available
                    for measure in self.statistical_measures:
                        battery_row[f"{feature}_{measure}"] = np.nan
            
            # Calculate RUL
            try:
                rul_tensor = self.rul_annotator.process_cell(battery)
                total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
                battery_row['rul'] = total_rul
                battery_row['log_rul'] = np.log(total_rul + 1) if total_rul > 0 else np.nan
            except Exception:
                battery_row['rul'] = 0
                battery_row['log_rul'] = np.nan
            
            if battery_row['rul'] > 0:  # Only include batteries with valid RUL
                all_battery_data.append(battery_row)
        
        if not all_battery_data:
            if self.verbose:
                print(f"No valid data for {dataset_name}")
            return {}
        
        # Convert to DataFrame
        data_df = pd.DataFrame(all_battery_data)
        
        if self.verbose:
            print(f"Valid data for {dataset_name}: {len(data_df)} batteries")
        
        # Calculate correlations with log_rul
        feature_cols = [col for col in data_df.columns 
                       if col not in ['battery_id', 'rul', 'log_rul']]
        
        correlations = {}
        for feature in feature_cols:
            if feature in data_df.columns:
                # Remove NaN values for correlation calculation
                valid_data = data_df[['log_rul', feature]].dropna()
                if len(valid_data) >= 2:
                    try:
                        from scipy.stats import spearmanr
                        corr, p_value = spearmanr(valid_data['log_rul'], valid_data[feature])
                        if not np.isnan(corr):
                            correlations[feature] = {
                                'correlation': corr,
                                'p_value': p_value,
                                'n_samples': len(valid_data)
                            }
                    except Exception:
                        continue
        
        return {
            'dataset': dataset_name,
            'n_batteries': len(data_df),
            'correlations': correlations
        }

    def find_common_correlations(self) -> Dict:
        """Find statistical features with high correlations common across datasets."""
        print("Finding common statistical correlations across datasets...")
        print(f"Correlation threshold: {self.correlation_threshold}")
        print(f"Minimum datasets: {self.min_datasets}")
        print("=" * 60)
        
        # Process each dataset with progress bar
        for dataset_name in tqdm(self.dataset_names, desc="Processing datasets", unit="dataset"):
            print(f"\nProcessing {dataset_name}...")
            result = self.process_dataset(dataset_name)
            if result:
                self.dataset_results[dataset_name] = result
        
        print(f"\nProcessed {len(self.dataset_results)} datasets")
        
        # Find common high correlations with progress bar
        feature_correlation_counts = defaultdict(list)
        
        print("\nAnalyzing correlations...")
        for dataset_name, result in tqdm(self.dataset_results.items(), desc="Analyzing correlations", unit="dataset"):
            correlations = result['correlations']
            for feature, corr_data in correlations.items():
                if abs(corr_data['correlation']) > self.correlation_threshold:
                    feature_correlation_counts[feature].append({
                        'dataset': dataset_name,
                        'correlation': corr_data['correlation'],
                        'p_value': corr_data['p_value'],
                        'n_samples': corr_data['n_samples']
                    })
        
        # Filter features that appear in multiple datasets
        print("\nFiltering common features...")
        common_features = {}
        for feature, corr_list in tqdm(feature_correlation_counts.items(), desc="Filtering features", unit="feature"):
            if len(corr_list) >= self.min_datasets:
                common_features[feature] = corr_list
        
        self.common_features = common_features
        
        print(f"\nFound {len(common_features)} features with high correlations in {self.min_datasets}+ datasets")
        
        return common_features

    def create_correlation_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of correlations across datasets."""
        print("\nCreating correlation summary...")
        summary_data = []
        
        for feature, corr_list in tqdm(self.common_features.items(), desc="Building summary", unit="feature"):
            for corr_data in corr_list:
                summary_data.append({
                    'feature': feature,
                    'dataset': corr_data['dataset'],
                    'correlation': corr_data['correlation'],
                    'abs_correlation': abs(corr_data['correlation']),
                    'p_value': corr_data['p_value'],
                    'n_samples': corr_data['n_samples']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            # Sort by absolute correlation (descending)
            summary_df = summary_df.sort_values('abs_correlation', ascending=False)
            
            # Add feature counts
            feature_counts = summary_df.groupby('feature').size().reset_index(name='dataset_count')
            summary_df = summary_df.merge(feature_counts, on='feature')
        
        self.correlation_summary = summary_df
        return summary_df

    def save_results(self):
        """Save results to files."""
        print("\nSaving results...")
        
        # Save correlation summary
        if not self.correlation_summary.empty:
            summary_file = self.output_dir / 'correlation_summary.csv'
            self.correlation_summary.to_csv(summary_file, index=False)
            print(f"Correlation summary saved to {summary_file}")
        
        # Save detailed results for each dataset
        print("Saving individual dataset results...")
        for dataset_name, result in tqdm(self.dataset_results.items(), desc="Saving datasets", unit="dataset"):
            dataset_file = self.output_dir / f'{dataset_name}_correlations.csv'
            corr_data = []
            for feature, corr_info in result['correlations'].items():
                corr_data.append({
                    'feature': feature,
                    'correlation': corr_info['correlation'],
                    'abs_correlation': abs(corr_info['correlation']),
                    'p_value': corr_info['p_value'],
                    'n_samples': corr_info['n_samples']
                })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data).sort_values('abs_correlation', ascending=False)
                corr_df.to_csv(dataset_file, index=False)
                print(f"{dataset_name} correlations saved to {dataset_file}")
        
        # Save common features summary
        print("Saving common features summary...")
        common_file = self.output_dir / 'common_features_summary.csv'
        common_data = []
        for feature, corr_list in tqdm(self.common_features.items(), desc="Building common summary", unit="feature"):
            avg_corr = np.mean([abs(c['correlation']) for c in corr_list])
            max_corr = max([abs(c['correlation']) for c in corr_list])
            min_corr = min([abs(c['correlation']) for c in corr_list])
            datasets = [c['dataset'] for c in corr_list]
            
            common_data.append({
                'feature': feature,
                'dataset_count': len(corr_list),
                'datasets': ', '.join(datasets),
                'avg_abs_correlation': avg_corr,
                'max_abs_correlation': max_corr,
                'min_abs_correlation': min_corr
            })
        
        if common_data:
            common_df = pd.DataFrame(common_data).sort_values('avg_abs_correlation', ascending=False)
            common_df.to_csv(common_file, index=False)
            print(f"Common features summary saved to {common_file}")

    def plot_correlation_heatmap(self):
        """Create a heatmap of correlations across datasets."""
        if self.correlation_summary.empty:
            print("No correlation data to plot")
            return
        
        print("\nCreating correlation heatmap...")
        
        # Pivot the data for heatmap
        heatmap_data = self.correlation_summary.pivot_table(
            index='feature', 
            columns='dataset', 
            values='correlation', 
            fill_value=0
        )
        
        # Sort by average absolute correlation
        avg_abs_corr = heatmap_data.abs().mean(axis=1).sort_values(ascending=False)
        heatmap_data = heatmap_data.loc[avg_abs_corr.index]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation with log(RUL)'})
        
        plt.title(f'Statistical Feature Correlations with log(RUL)\n(Threshold: {self.correlation_threshold}, Min Datasets: {self.min_datasets})')
        plt.xlabel('Dataset')
        plt.ylabel('Statistical Feature')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        heatmap_file = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {heatmap_file}")

    def plot_feature_frequency(self):
        """Plot the frequency of high-correlation features across datasets."""
        if not self.common_features:
            print("No common features to plot")
            return
        
        print("\nCreating feature frequency plot...")
        
        # Count how many datasets each feature appears in
        feature_counts = {feature: len(corr_list) for feature, corr_list in self.common_features.items()}
        
        # Sort by count
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(features)), counts)
        
        # Color bars by count
        for i, bar in enumerate(bars):
            if counts[i] >= len(self.dataset_results) * 0.8:  # 80% of datasets
                bar.set_color('green')
            elif counts[i] >= len(self.dataset_results) * 0.5:  # 50% of datasets
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title(f'Frequency of High-Correlation Features Across Datasets\n(Threshold: {self.correlation_threshold})')
        plt.xlabel('Statistical Feature')
        plt.ylabel('Number of Datasets')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        freq_file = self.output_dir / 'feature_frequency.png'
        plt.savefig(freq_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature frequency plot saved to {freq_file}")

    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "=" * 80)
        print("COMMON STATISTICAL CORRELATIONS SUMMARY")
        print("=" * 80)
        
        print(f"Datasets processed: {len(self.dataset_results)}")
        print(f"Correlation threshold: {self.correlation_threshold}")
        print(f"Minimum datasets required: {self.min_datasets}")
        print(f"Common features found: {len(self.common_features)}")
        
        if self.common_features:
            print("\nTop 10 most common high-correlation features:")
            print("-" * 50)
            
            # Sort by number of datasets and average correlation
            feature_stats = []
            for feature, corr_list in self.common_features.items():
                avg_corr = np.mean([abs(c['correlation']) for c in corr_list])
                feature_stats.append((feature, len(corr_list), avg_corr))
            
            feature_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            for i, (feature, count, avg_corr) in enumerate(feature_stats[:10]):
                print(f"{i+1:2d}. {feature:<40} ({count} datasets, avg |corr| = {avg_corr:.3f})")
        
        print("\n" + "=" * 80)


def main():
    """Main function to run the common correlation analysis."""
    parser = argparse.ArgumentParser(description='Find common statistical correlations across datasets')
    parser.add_argument('--data_path', type=str, default='data/preprocessed',
                       help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='common_correlations_results',
                       help='Output directory for results')
    parser.add_argument('--correlation_threshold', type=float, default=0.5,
                       help='Minimum absolute correlation threshold')
    parser.add_argument('--min_datasets', type=int, default=3,
                       help='Minimum number of datasets required for common features')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMMON STATISTICAL CORRELATIONS ANALYSIS")
    print("=" * 80)
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Correlation Threshold: {args.correlation_threshold}")
    print(f"Minimum Datasets: {args.min_datasets}")
    print("=" * 80)
    
    # Create the finder
    finder = CommonStatisticalCorrelationFinder(
        data_path=args.data_path,
        output_dir=args.output_dir,
        correlation_threshold=args.correlation_threshold,
        min_datasets=args.min_datasets,
        verbose=args.verbose
    )
    
    # Define main steps for progress tracking
    main_steps = [
        "Finding common correlations",
        "Creating correlation summary", 
        "Saving results",
        "Creating visualizations",
        "Printing summary"
    ]
    
    # Execute main steps with progress bar
    with tqdm(total=len(main_steps), desc="Overall Progress", unit="step") as pbar:
        # Find common correlations
        pbar.set_description("Finding common correlations")
        common_features = finder.find_common_correlations()
        pbar.update(1)
        
        # Create summary
        pbar.set_description("Creating correlation summary")
        summary_df = finder.create_correlation_summary()
        pbar.update(1)
        
        # Save results
        pbar.set_description("Saving results")
        finder.save_results()
        pbar.update(1)
        
        # Create visualizations
        pbar.set_description("Creating visualizations")
        finder.plot_correlation_heatmap()
        finder.plot_feature_frequency()
        pbar.update(1)
        
        # Print summary
        pbar.set_description("Printing summary")
        finder.print_summary()
        pbar.update(1)
    
    print(f"\nAnalysis complete! Results saved to {finder.output_dir}")


if __name__ == '__main__':
    main()
