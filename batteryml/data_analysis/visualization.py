# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class BatteryDataVisualizer:
    """
    Visualization utilities for battery data analysis.
    """
    
    def __init__(self, output_dir: str = "analysis_plots", dataset_name: str = "unknown"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            dataset_name: Name of the dataset for folder organization
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_dataset_overview(self, dataset_stats: Dict[str, Any], save: bool = True) -> None:
        """
        Create dataset overview visualizations.
        
        Args:
            dataset_stats: Dataset statistics from analyzer
            save: Whether to save plots to disk
        """
        if not dataset_stats:
            print("No dataset statistics provided")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Batteries per dataset
        if 'datasets' in dataset_stats and dataset_stats['datasets']:
            datasets = list(dataset_stats['datasets'].keys())
            counts = list(dataset_stats['datasets'].values())
            
            axes[0, 0].bar(datasets, counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Batteries per Dataset')
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('Number of Batteries')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(counts):
                axes[0, 0].text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # 2. Batteries per chemistry
        if 'chemistries' in dataset_stats and dataset_stats['chemistries']:
            chemistries = list(dataset_stats['chemistries'].keys())
            counts = list(dataset_stats['chemistries'].values())
            
            axes[0, 1].pie(counts, labels=chemistries, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Batteries per Chemistry')
        
        # 3. Capacity distribution
        if 'capacities' in dataset_stats and len(dataset_stats['capacities']) > 0:
            capacities = dataset_stats['capacities']
            axes[1, 0].hist(capacities, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Nominal Capacity Distribution')
            axes[1, 0].set_xlabel('Capacity (Ah)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Cycle life distribution
        if 'cycle_lives' in dataset_stats and len(dataset_stats['cycle_lives']) > 0:
            cycle_lives = dataset_stats['cycle_lives']
            axes[1, 1].hist(cycle_lives, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Cycle Life Distribution')
            axes[1, 1].set_xlabel('Number of Cycles')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
            print(f"Dataset overview plot saved to {self.output_dir / 'dataset_overview.png'}")
        
        plt.close()  # Close the figure instead of showing it
    
    def plot_feature_statistics(self, feature_stats: Dict[str, Any], save: bool = True) -> None:
        """
        Create feature statistics visualizations.
        
        Args:
            feature_stats: Feature statistics from analyzer
            save: Whether to save plots to disk
        """
        if not feature_stats:
            print("No feature statistics provided")
            return
        
        # Create a simple summary table instead of complex plots
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create summary data
            summary_data = []
            for feature_name, stats in feature_stats.items():
                key_prefix = list(stats.keys())[0].split('_')[0]
                count = stats.get(f'{key_prefix}_count', 0)
                mean_val = stats.get(f'{key_prefix}_mean', np.nan)
                std_val = stats.get(f'{key_prefix}_std', np.nan)
                min_val = stats.get(f'{key_prefix}_min', np.nan)
                max_val = stats.get(f'{key_prefix}_max', np.nan)
                
                if not np.isnan(mean_val) and count > 0:
                    summary_data.append({
                        'Feature': feature_name,
                        'Count': count,
                        'Mean': f"{mean_val:.3f}",
                        'Std': f"{std_val:.3f}",
                        'Min': f"{min_val:.3f}",
                        'Max': f"{max_val:.3f}"
                    })
            
            if summary_data:
                # Create a simple bar chart of feature counts
                features = [item['Feature'] for item in summary_data]
                counts = [item['Count'] for item in summary_data]
                
                bars = ax.bar(range(len(features)), counts, color='skyblue', alpha=0.7)
                ax.set_xlabel('Features')
                ax.set_ylabel('Data Points Count')
                ax.set_title('Feature Data Points Count', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(features)))
                ax.set_xticklabels(features, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid feature data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Feature Statistics', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save:
                plt.savefig(self.output_dir / 'feature_statistics.png', dpi=150, bbox_inches='tight')
                print(f"Feature statistics plot saved to {self.output_dir / 'feature_statistics.png'}")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create feature statistics plot: {e}")
            # Create a simple text-based summary instead
            if save:
                with open(self.output_dir / 'feature_statistics.txt', 'w') as f:
                    f.write("Feature Statistics Summary\n")
                    f.write("=" * 50 + "\n\n")
                    for feature_name, stats in feature_stats.items():
                        key_prefix = list(stats.keys())[0].split('_')[0]
                        count = stats.get(f'{key_prefix}_count', 0)
                        mean_val = stats.get(f'{key_prefix}_mean', np.nan)
                        f.write(f"{feature_name}: {count} data points, mean={mean_val:.3f}\n")
                print(f"Feature statistics summary saved to {self.output_dir / 'feature_statistics.txt'}")
    
    def plot_correlation_heatmap(self, feature_stats: Dict[str, Any], save: bool = True) -> None:
        """
        Create a correlation heatmap for features (simulated).
        
        Args:
            feature_stats: Feature statistics from analyzer
            save: Whether to save plots to disk
        """
        if not feature_stats:
            print("No feature statistics provided")
            return
        
        # Extract numeric features for correlation
        numeric_features = {}
        for feature_name, stats in feature_stats.items():
            key_prefix = list(stats.keys())[0].split('_')[0]
            mean_val = stats.get(f'{key_prefix}_mean', np.nan)
            if not np.isnan(mean_val):
                numeric_features[feature_name] = mean_val
        
        if len(numeric_features) < 2:
            print("Not enough numeric features for correlation analysis")
            return
        
        # Create correlation matrix (simulated)
        feature_names = list(numeric_features.keys())
        n_features = len(feature_names)
        
        # Generate random correlation matrix
        np.random.seed(42)
        correlation_matrix = np.random.rand(n_features, n_features)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Feature Correlation Matrix (Simulated)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
            print(f"Feature correlation plot saved to {self.output_dir / 'feature_correlation.png'}")
        
        plt.close()  # Close the figure instead of showing it
    
    def create_summary_dashboard(self, dataset_stats: Dict[str, Any], 
                               feature_stats: Dict[str, Any], save: bool = True) -> None:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            dataset_stats: Dataset statistics
            feature_stats: Feature statistics
            save: Whether to save plots to disk
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Battery Data Analysis Dashboard - {self.dataset_name}', fontsize=16, fontweight='bold')
            
            # 1. Dataset overview (top left)
            ax1 = axes[0, 0]
            if 'datasets' in dataset_stats and dataset_stats['datasets']:
                datasets = list(dataset_stats['datasets'].keys())
                counts = list(dataset_stats['datasets'].values())
                bars = ax1.bar(datasets, counts, color='skyblue', alpha=0.7)
                ax1.set_title('Batteries per Dataset', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Dataset')
                ax1.set_ylabel('Number of Batteries')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
            
            # 2. Chemistry distribution (top right)
            ax2 = axes[0, 1]
            if 'chemistries' in dataset_stats and dataset_stats['chemistries']:
                chemistries = list(dataset_stats['chemistries'].keys())
                counts = list(dataset_stats['chemistries'].values())
                ax2.pie(counts, labels=chemistries, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Batteries per Chemistry', fontsize=12, fontweight='bold')
            
            # 3. Capacity distribution (bottom left)
            ax3 = axes[1, 0]
            if 'capacities' in dataset_stats and len(dataset_stats['capacities']) > 0:
                capacities = dataset_stats['capacities']
                ax3.hist(capacities, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
                ax3.set_title('Nominal Capacity Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Capacity (Ah)')
                ax3.set_ylabel('Frequency')
            
            # 4. Cycle life distribution (bottom right)
            ax4 = axes[1, 1]
            if 'cycle_lives' in dataset_stats and len(dataset_stats['cycle_lives']) > 0:
                cycle_lives = dataset_stats['cycle_lives']
                ax4.hist(cycle_lives, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
                ax4.set_title('Cycle Life Distribution', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Number of Cycles')
                ax4.set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save:
                plt.savefig(self.output_dir / 'analysis_dashboard.png', dpi=150, bbox_inches='tight')
                print(f"Analysis dashboard saved to {self.output_dir / 'analysis_dashboard.png'}")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create dashboard plot: {e}")
            # Create a simple text summary instead
            if save:
                with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
                    f.write(f"Battery Data Analysis Summary - {self.dataset_name}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Total batteries: {dataset_stats.get('total_batteries', 0)}\n")
                    f.write(f"Datasets: {list(dataset_stats.get('datasets', {}).keys())}\n")
                    f.write(f"Chemistries: {list(dataset_stats.get('chemistries', {}).keys())}\n")
                    f.write(f"Features analyzed: {len(feature_stats)}\n")
                print(f"Analysis summary saved to {self.output_dir / 'analysis_summary.txt'}")
    
    def save_all_plots(self, dataset_stats: Dict[str, Any], feature_stats: Dict[str, Any]) -> None:
        """
        Generate and save all visualization plots.
        
        Args:
            dataset_stats: Dataset statistics
            feature_stats: Feature statistics
        """
        print("Generating all visualization plots...")
        
        self.plot_dataset_overview(dataset_stats, save=True)
        self.plot_feature_statistics(feature_stats, save=True)
        self.plot_correlation_heatmap(feature_stats, save=True)
        self.create_summary_dashboard(dataset_stats, feature_stats, save=True)
        
        print(f"All plots saved to: {self.output_dir}")
