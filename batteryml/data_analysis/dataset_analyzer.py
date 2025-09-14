# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from batteryml.data.battery_data import BatteryData
from .battery_analyzer import BatteryAnalyzer
from .utils import AnalysisUtils


class DatasetAnalyzer:
    """Analyzer for entire battery datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset analyzer.
        
        Args:
            dataset_path: Path to the dataset directory containing pickle files
        """
        self.dataset_path = Path(dataset_path)
        self.battery_files = AnalysisUtils.get_battery_files(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.analysis_results = []
        self.summary_stats = {}
        
    def analyze_dataset(self, max_batteries: Optional[int] = None, 
                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Analyze all batteries in the dataset.
        
        Args:
            max_batteries: Maximum number of batteries to analyze (None for all)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing dataset analysis results
        """
        print(f"\nAnalyzing dataset: {self.dataset_name}")
        print(f"Found {len(self.battery_files)} battery files")
        
        if max_batteries:
            self.battery_files = self.battery_files[:max_batteries]
            print(f"Analyzing first {len(self.battery_files)} batteries")
        
        # Analyze each battery
        iterator = tqdm(self.battery_files, desc="Analyzing batteries") if show_progress else self.battery_files
        
        for file_path in iterator:
            try:
                battery_data = AnalysisUtils.safe_load_battery(file_path)
                if battery_data is not None:
                    analyzer = BatteryAnalyzer(battery_data)
                    analysis = analyzer.get_comprehensive_analysis()
                    analysis['file_path'] = file_path
                    self.analysis_results.append(analysis)
                else:
                    print(f"Warning: Could not load battery from {file_path}")
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
                continue
        
        # Generate summary statistics
        self.summary_stats = self._generate_summary_statistics()
        
        print(f"Successfully analyzed {len(self.analysis_results)} batteries")
        return self.summary_stats
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.analysis_results:
            return {}
        
        summary = {
            'dataset_name': self.dataset_name,
            'total_batteries': len(self.analysis_results),
            'successful_analyses': len(self.analysis_results)
        }
        
        # Extract all unique features and their statistics
        all_features = self._extract_all_features()
        summary['features'] = all_features
        
        # Dataset-level statistics
        summary['dataset_stats'] = self._calculate_dataset_statistics()
        
        # Chemistry distribution
        summary['chemistry_distribution'] = self._get_chemistry_distribution()
        
        # Cycle life distribution
        summary['cycle_life_distribution'] = self._get_cycle_life_distribution()
        
        return summary
    
    def _extract_all_features(self) -> Dict[str, Dict[str, float]]:
        """
        Extract all features and their statistics from the dataset.
        
        Returns:
            Dictionary containing feature statistics
        """
        features = {}
        
        # Collect all feature values
        feature_values = {}
        
        for analysis in self.analysis_results:
            # Extract all numeric values from the analysis
            self._extract_numeric_values(analysis, feature_values)
        
        # Calculate statistics for each feature
        for feature_name, values in feature_values.items():
            if values:  # Only process if we have values
                values = np.array(values)
                values = values[~np.isnan(values)]  # Remove NaN values
                
                if len(values) > 0:
                    features[feature_name] = {
                        'count': len(values),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
                else:
                    features[feature_name] = {
                        'count': 0,
                        'min': np.nan,
                        'max': np.nan,
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'q25': np.nan,
                        'q75': np.nan
                    }
        
        return features
    
    def _extract_numeric_values(self, analysis: Dict[str, Any], 
                               feature_values: Dict[str, List[float]]):
        """
        Recursively extract numeric values from analysis dictionary.
        
        Args:
            analysis: Analysis dictionary
            feature_values: Dictionary to store feature values
        """
        for key, value in analysis.items():
            if isinstance(value, dict):
                self._extract_numeric_values(value, feature_values)
            elif isinstance(value, (int, float)) and not np.isnan(value):
                if key not in feature_values:
                    feature_values[key] = []
                feature_values[key].append(value)
    
    def _calculate_dataset_statistics(self) -> Dict[str, Any]:
        """
        Calculate dataset-level statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}
        
        # Total cycles across all batteries
        total_cycles = sum(analysis['basic_info']['total_cycles'] 
                          for analysis in self.analysis_results)
        stats['total_cycles'] = total_cycles
        
        # Average cycle life
        cycle_lives = [analysis['cycle_life_analysis']['cycle_life'] 
                      for analysis in self.analysis_results]
        cycle_lives = [cl for cl in cycle_lives if not np.isnan(cl)]
        
        if cycle_lives:
            stats['avg_cycle_life'] = float(np.mean(cycle_lives))
            stats['median_cycle_life'] = float(np.median(cycle_lives))
            stats['std_cycle_life'] = float(np.std(cycle_lives))
            stats['min_cycle_life'] = float(np.min(cycle_lives))
            stats['max_cycle_life'] = float(np.max(cycle_lives))
        else:
            stats['avg_cycle_life'] = np.nan
            stats['median_cycle_life'] = np.nan
            stats['std_cycle_life'] = np.nan
            stats['min_cycle_life'] = np.nan
            stats['max_cycle_life'] = np.nan
        
        # Capacity statistics
        nominal_capacities = [analysis['basic_info']['nominal_capacity'] 
                             for analysis in self.analysis_results]
        nominal_capacities = [nc for nc in nominal_capacities if nc is not None and not np.isnan(nc)]
        
        if nominal_capacities:
            stats['avg_nominal_capacity'] = float(np.mean(nominal_capacities))
            stats['median_nominal_capacity'] = float(np.median(nominal_capacities))
            stats['std_nominal_capacity'] = float(np.std(nominal_capacities))
        else:
            stats['avg_nominal_capacity'] = np.nan
            stats['median_nominal_capacity'] = np.nan
            stats['std_nominal_capacity'] = np.nan
        
        return stats
    
    def _get_chemistry_distribution(self) -> Dict[str, int]:
        """
        Get distribution of battery chemistries.
        
        Returns:
            Dictionary containing chemistry counts
        """
        chemistry_counts = {}
        
        for analysis in self.analysis_results:
            cathode = analysis['basic_info']['cathode_material']
            anode = analysis['basic_info']['anode_material']
            
            if cathode and anode:
                chemistry = f"{cathode}/{anode}"
                chemistry_counts[chemistry] = chemistry_counts.get(chemistry, 0) + 1
            elif cathode:
                chemistry_counts[cathode] = chemistry_counts.get(cathode, 0) + 1
            elif anode:
                chemistry_counts[anode] = chemistry_counts.get(anode, 0) + 1
            else:
                chemistry_counts['Unknown'] = chemistry_counts.get('Unknown', 0) + 1
        
        return chemistry_counts
    
    def _get_cycle_life_distribution(self) -> Dict[str, float]:
        """
        Get cycle life distribution statistics.
        
        Returns:
            Dictionary containing cycle life distribution
        """
        cycle_lives = [analysis['cycle_life_analysis']['cycle_life'] 
                      for analysis in self.analysis_results]
        cycle_lives = [cl for cl in cycle_lives if not np.isnan(cl)]
        
        if not cycle_lives:
            return {}
        
        distribution = {
            'mean': float(np.mean(cycle_lives)),
            'median': float(np.median(cycle_lives)),
            'std': float(np.std(cycle_lives)),
            'min': float(np.min(cycle_lives)),
            'max': float(np.max(cycle_lives)),
            'q25': float(np.percentile(cycle_lives, 25)),
            'q75': float(np.percentile(cycle_lives, 75))
        }
        
        return distribution
    
    def get_feature_summary_table(self) -> pd.DataFrame:
        """
        Get a summary table of all features in the dataset.
        
        Returns:
            DataFrame containing feature summary
        """
        if not self.summary_stats or 'features' not in self.summary_stats:
            return pd.DataFrame()
        
        features_data = []
        for feature_name, stats in self.summary_stats['features'].items():
            row = {
                'Feature': feature_name,
                'Count': stats['count'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std'],
                'Q25': stats['q25'],
                'Q75': stats['q75']
            }
            features_data.append(row)
        
        return pd.DataFrame(features_data)
    
    def print_dataset_summary(self):
        """Print a comprehensive summary of the dataset analysis."""
        if not self.summary_stats:
            print("No analysis results available. Run analyze_dataset() first.")
            return
        
        print(f"\n{'='*80}")
        print(f"DATASET ANALYSIS SUMMARY: {self.summary_stats['dataset_name']}")
        print(f"{'='*80}")
        
        # Basic dataset info
        print(f"Total Batteries: {self.summary_stats['total_batteries']}")
        print(f"Successfully Analyzed: {self.summary_stats['successful_analyses']}")
        
        # Dataset statistics
        if 'dataset_stats' in self.summary_stats:
            stats = self.summary_stats['dataset_stats']
            print(f"\n--- DATASET STATISTICS ---")
            print(f"Total Cycles: {stats.get('total_cycles', 'N/A')}")
            print(f"Average Cycle Life: {stats.get('avg_cycle_life', 'N/A'):.2f}")
            print(f"Median Cycle Life: {stats.get('median_cycle_life', 'N/A'):.2f}")
            print(f"Cycle Life Std: {stats.get('std_cycle_life', 'N/A'):.2f}")
            print(f"Min Cycle Life: {stats.get('min_cycle_life', 'N/A'):.2f}")
            print(f"Max Cycle Life: {stats.get('max_cycle_life', 'N/A'):.2f}")
            print(f"Average Nominal Capacity: {stats.get('avg_nominal_capacity', 'N/A'):.4f} Ah")
        
        # Chemistry distribution
        if 'chemistry_distribution' in self.summary_stats:
            print(f"\n--- CHEMISTRY DISTRIBUTION ---")
            for chemistry, count in self.summary_stats['chemistry_distribution'].items():
                print(f"{chemistry}: {count} batteries")
        
        # Cycle life distribution
        if 'cycle_life_distribution' in self.summary_stats:
            print(f"\n--- CYCLE LIFE DISTRIBUTION ---")
            dist = self.summary_stats['cycle_life_distribution']
            print(f"Mean: {dist.get('mean', 'N/A'):.2f}")
            print(f"Median: {dist.get('median', 'N/A'):.2f}")
            print(f"Std: {dist.get('std', 'N/A'):.2f}")
            print(f"Min: {dist.get('min', 'N/A'):.2f}")
            print(f"Max: {dist.get('max', 'N/A'):.2f}")
            print(f"Q25: {dist.get('q25', 'N/A'):.2f}")
            print(f"Q75: {dist.get('q75', 'N/A'):.2f}")
        
        print(f"{'='*80}\n")
    
    def print_feature_summary(self, top_n: int = 20):
        """
        Print a summary of the top features in the dataset.
        
        Args:
            top_n: Number of top features to display
        """
        if not self.summary_stats or 'features' not in self.summary_stats:
            print("No feature analysis available. Run analyze_dataset() first.")
            return
        
        features_df = self.get_feature_summary_table()
        
        if features_df.empty:
            print("No features found in the dataset.")
            return
        
        # Sort by count (descending) and take top N
        features_df = features_df.sort_values('Count', ascending=False).head(top_n)
        
        print(f"\n{'='*100}")
        print(f"TOP {top_n} FEATURES IN DATASET: {self.summary_stats['dataset_name']}")
        print(f"{'='*100}")
        
        # Print formatted table
        print(f"{'Feature':<40} {'Count':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Median':<12} {'Std':<12}")
        print("-" * 100)
        
        for _, row in features_df.iterrows():
            print(f"{row['Feature']:<40} {row['Count']:<8} {row['Min']:<12.4f} {row['Max']:<12.4f} "
                  f"{row['Mean']:<12.4f} {row['Median']:<12.4f} {row['Std']:<12.4f}")
        
        print(f"{'='*100}\n")
    
    def save_analysis_results(self, output_path: str):
        """
        Save analysis results to files.
        
        Args:
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics
        summary_file = output_path / f"{self.dataset_name}_summary.json"
        import json
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            summary_serializable = self._make_serializable(self.summary_stats)
            json.dump(summary_serializable, f, indent=2)
        
        # Save feature summary table
        features_df = self.get_feature_summary_table()
        if not features_df.empty:
            features_file = output_path / f"{self.dataset_name}_features.csv"
            features_df.to_csv(features_file, index=False)
        
        print(f"Analysis results saved to {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
