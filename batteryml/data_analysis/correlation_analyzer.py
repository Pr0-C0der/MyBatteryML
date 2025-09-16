# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
matplotlib.use('Agg')

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator


class CorrelationAnalyzer:
    """Analyzer for creating correlation matrices between cycle features and RUL."""
    
    def __init__(self, data_path: str, output_dir: str = "correlation_analysis"):
        """
        Initialize the correlation analyzer.
        
        Args:
            data_path: Path to the processed battery data directory
            output_dir: Directory to save correlation analysis results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.matrices_dir = self.output_dir / "matrices"
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.matrices_dir.mkdir(exist_ok=True)
        
        # Initialize RUL label annotator
        self.rul_annotator = RULLabelAnnotator()
        
        # Detect available features
        self.features = self._detect_available_features()
    
    def load_battery_data(self, file_path: Path) -> Optional[BatteryData]:
        """Load a single battery data file."""
        try:
            with open(file_path, 'rb') as f:
                return BatteryData.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_battery_files(self) -> List[Path]:
        """Get list of battery pickle files."""
        return list(self.data_path.glob("*.pkl"))
    
    def _detect_available_features(self) -> List[str]:
        """Detect available features by examining the first battery file."""
        battery_files = self.get_battery_files()
        if not battery_files:
            return ['voltage', 'current', 'capacity', 'temperature']
        
        # Load the first battery to detect available features
        sample_battery = self.load_battery_data(battery_files[0])
        if not sample_battery or not sample_battery.cycle_data:
            return ['voltage', 'current', 'capacity', 'temperature']
        
        # Check the first cycle for available features
        first_cycle = sample_battery.cycle_data[0]
        available_features = []
        
        # Map CycleData attributes to feature names
        feature_mapping = {
            'voltage_in_V': 'voltage',
            'current_in_A': 'current', 
            'discharge_capacity_in_Ah': 'capacity',
            'charge_capacity_in_Ah': 'charge_capacity',
            'temperature_in_C': 'temperature',
            'internal_resistance_in_ohm': 'internal_resistance',
            'energy_charge': 'energy_charge',
            'energy_discharge': 'energy_discharge',
            'Qdlin': 'qdlin',
            'Tdlin': 'tdlin'
        }
        
        for attr, feature_name in feature_mapping.items():
            if hasattr(first_cycle, attr):
                attr_value = getattr(first_cycle, attr)
                if attr_value is not None and len(attr_value) > 0:
                    available_features.append(feature_name)
        
        return available_features
    
    def calculate_rul(self, battery: BatteryData) -> int:
        """Calculate RUL for a battery using the label annotator."""
        try:
            rul_tensor = self.rul_annotator.process_cell(battery)
            rul_value = rul_tensor.item()
            return int(rul_value) if not np.isnan(rul_value) else 0
        except Exception as e:
            print(f"Error calculating RUL for {battery.cell_id}: {e}")
            return 0
    
    def create_cycle_feature_matrix(self, battery: BatteryData) -> pd.DataFrame:
        """Create a matrix with cycles as rows and features as columns."""
        matrix_data = []
        
        # Calculate RUL for this battery
        total_rul = self.calculate_rul(battery)
        
        for cycle_idx, cycle_data in enumerate(battery.cycle_data):
            cycle_number = cycle_data.cycle_number
            row_data = {'cycle_number': cycle_number}
            
            # Calculate RUL for this cycle (remaining cycles until EOL)
            cycle_rul = max(0, total_rul - cycle_idx)
            row_data['rul'] = cycle_rul
            
            # Calculate mean values for each feature
            feature_mapping = {
                'voltage': 'voltage_in_V',
                'current': 'current_in_A',
                'capacity': 'discharge_capacity_in_Ah',
                'charge_capacity': 'charge_capacity_in_Ah',
                'temperature': 'temperature_in_C',
                'internal_resistance': 'internal_resistance_in_ohm',
                'energy_charge': 'energy_charge',
                'energy_discharge': 'energy_discharge',
                'qdlin': 'Qdlin',
                'tdlin': 'Tdlin'
            }
            
            for feature_name in self.features:
                if feature_name in feature_mapping:
                    attr_name = feature_mapping[feature_name]
                    if hasattr(cycle_data, attr_name):
                        feature_data = getattr(cycle_data, attr_name)
                        if feature_data is not None and len(feature_data) > 0:
                            # Convert to numpy array and filter valid data
                            feature_array = np.array(feature_data)
                            valid_data = feature_array[~np.isnan(feature_array)]
                            if len(valid_data) > 0:
                                row_data[feature_name] = np.mean(valid_data)
                            else:
                                row_data[feature_name] = np.nan
                        else:
                            row_data[feature_name] = np.nan
                    else:
                        row_data[feature_name] = np.nan
                else:
                    row_data[feature_name] = np.nan
            
            matrix_data.append(row_data)
        
        return pd.DataFrame(matrix_data)
    
    def plot_correlation_heatmap(self, battery: BatteryData, matrix: pd.DataFrame, save_path: Path):
        """Create a correlation heatmap for a single battery."""
        # Select only numeric columns for correlation
        numeric_cols = matrix.select_dtypes(include=[np.number]).columns
        correlation_matrix = matrix[numeric_cols].corr()
        
        # Create the heatmap
        plt.figure(figsize=(12, 10))
        
        # Create full heatmap without mask
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={'size': 8})
        
        plt.title(f'Feature Correlation Matrix - {battery.cell_id}', fontsize=14, pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rul_correlation(self, battery: BatteryData, matrix: pd.DataFrame, save_path: Path):
        """Create a specific heatmap focusing on RUL correlations."""
        # Select only numeric columns
        numeric_cols = matrix.select_dtypes(include=[np.number]).columns
        
        # Calculate correlations with RUL
        rul_correlations = matrix[numeric_cols].corr()['rul'].drop('rul')
        rul_correlations = rul_correlations.sort_values(key=abs, ascending=False)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        
        colors = ['red' if x < 0 else 'blue' for x in rul_correlations.values]
        bars = plt.barh(range(len(rul_correlations)), rul_correlations.values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(rul_correlations)), rul_correlations.index)
        plt.xlabel('Correlation with RUL', fontsize=12)
        plt.title(f'Feature Correlations with RUL - {battery.cell_id}', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values on bars
        for i, (bar, value) in enumerate(zip(bars, rul_correlations.values)):
            plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                    va='center', ha='left' if value >= 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_battery(self, battery: BatteryData):
        """Analyze a single battery and create correlation matrices."""
        cell_id = battery.cell_id.replace('/', '_').replace('\\', '_')  # Safe filename
        
        try:
            # Create cycle-feature matrix
            matrix = self.create_cycle_feature_matrix(battery)
            
            # Save matrix as CSV
            matrix_path = self.matrices_dir / f"{cell_id}_cycle_feature_matrix.csv"
            matrix.to_csv(matrix_path, index=False)
            
            # Create correlation heatmap
            heatmap_path = self.heatmaps_dir / f"{cell_id}_correlation_heatmap.png"
            self.plot_correlation_heatmap(battery, matrix, heatmap_path)
            
            # Create RUL-specific correlation plot
            rul_plot_path = self.heatmaps_dir / f"{cell_id}_rul_correlations.png"
            self.plot_rul_correlation(battery, matrix, rul_plot_path)
            
            print(f"Analysis completed for {battery.cell_id}")
            print(f"  - Matrix shape: {matrix.shape}")
            print(f"  - Features: {', '.join(self.features)}")
            print(f"  - RUL range: {matrix['rul'].min()} to {matrix['rul'].max()}")
            
        except Exception as e:
            print(f"Error analyzing battery {battery.cell_id}: {e}")
    
    def analyze_dataset(self):
        """Analyze all batteries in the dataset."""
        print(f"Starting correlation analysis for dataset in {self.data_path}")
        print(f"Detected features: {', '.join(self.features)}")
        
        battery_files = self.get_battery_files()
        if not battery_files:
            print(f"No battery files found in {self.data_path}")
            return
        
        print(f"Found {len(battery_files)} battery files")
        
        # Analyze each battery
        for file_path in tqdm(battery_files, desc="Analyzing batteries"):
            battery = self.load_battery_data(file_path)
            if battery:
                self.analyze_battery(battery)
            else:
                print(f"Warning: Could not load battery from {file_path}")
        
        print(f"Correlation analysis complete! Results saved to {self.output_dir}")
        print(f"  - Heatmaps: {self.heatmaps_dir}")
        print(f"  - Matrices: {self.matrices_dir}")


def main():
    """Main function to run correlation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate correlation analysis for battery data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='correlation_analysis',
                       help='Output directory for correlation analysis')
    
    args = parser.parse_args()
    
    analyzer = CorrelationAnalyzer(args.data_path, args.output_dir)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
