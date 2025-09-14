# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class AnalysisVisualizer:
    """Visualization utilities for battery data analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
    
    def plot_dataset_overview(self, summary_stats: Dict[str, Any], 
                            save_path: Optional[str] = None):
        """
        Create an overview plot of the dataset.
        
        Args:
            summary_stats: Dataset summary statistics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Dataset Overview: {summary_stats.get('dataset_name', 'Unknown')}", 
                    fontsize=16, fontweight='bold')
        
        # 1. Total batteries
        axes[0, 0].bar(['Total Batteries', 'Analyzed'], 
                      [summary_stats.get('total_batteries', 0), 
                       summary_stats.get('successful_analyses', 0)],
                      color=self.colors[0:2])
        axes[0, 0].set_title('Dataset Size')
        axes[0, 0].set_ylabel('Number of Batteries')
        
        # 2. Chemistry distribution
        if 'chemistry_distribution' in summary_stats:
            chemistry_data = summary_stats['chemistry_distribution']
            if chemistry_data:
                axes[0, 1].pie(chemistry_data.values(), 
                             labels=chemistry_data.keys(),
                             autopct='%1.1f%%',
                             startangle=90)
                axes[0, 1].set_title('Chemistry Distribution')
        
        # 3. Cycle life distribution
        if 'cycle_life_distribution' in summary_stats:
            cycle_life_data = summary_stats['cycle_life_distribution']
            if cycle_life_data and not np.isnan(cycle_life_data.get('mean', np.nan)):
                stats = ['Mean', 'Median', 'Min', 'Max']
                values = [cycle_life_data.get('mean', 0),
                         cycle_life_data.get('median', 0),
                         cycle_life_data.get('min', 0),
                         cycle_life_data.get('max', 0)]
                
                axes[1, 0].bar(stats, values, color=self.colors[2:6])
                axes[1, 0].set_title('Cycle Life Statistics')
                axes[1, 0].set_ylabel('Cycles')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Dataset statistics
        if 'dataset_stats' in summary_stats:
            dataset_stats = summary_stats['dataset_stats']
            stats_text = f"Total Cycles: {dataset_stats.get('total_cycles', 'N/A')}\n"
            stats_text += f"Avg Cycle Life: {dataset_stats.get('avg_cycle_life', 'N/A'):.1f}\n"
            stats_text += f"Avg Capacity: {dataset_stats.get('avg_nominal_capacity', 'N/A'):.3f} Ah"
            
            axes[1, 1].text(0.1, 0.5, stats_text, 
                           transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Key Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_distribution(self, features_df: pd.DataFrame, 
                                top_n: int = 15, 
                                save_path: Optional[str] = None):
        """
        Plot distribution of top features.
        
        Args:
            features_df: DataFrame containing feature statistics
            top_n: Number of top features to plot
            save_path: Optional path to save the plot
        """
        if features_df.empty:
            print("No features to plot")
            return
        
        # Get top N features by count
        top_features = features_df.nlargest(top_n, 'Count')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {top_n} Features Distribution', fontsize=16, fontweight='bold')
        
        # 1. Feature counts
        axes[0, 0].barh(range(len(top_features)), top_features['Count'], 
                       color=self.colors[0])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['Feature'], fontsize=8)
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_title('Feature Availability')
        axes[0, 0].invert_yaxis()
        
        # 2. Mean values
        axes[0, 1].barh(range(len(top_features)), top_features['Mean'], 
                       color=self.colors[1])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['Feature'], fontsize=8)
        axes[0, 1].set_xlabel('Mean Value')
        axes[0, 1].set_title('Mean Values')
        axes[0, 1].invert_yaxis()
        
        # 3. Standard deviation
        axes[1, 0].barh(range(len(top_features)), top_features['Std'], 
                       color=self.colors[2])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['Feature'], fontsize=8)
        axes[1, 0].set_xlabel('Standard Deviation')
        axes[1, 0].set_title('Feature Variability')
        axes[1, 0].invert_yaxis()
        
        # 4. Range (Max - Min)
        range_values = top_features['Max'] - top_features['Min']
        axes[1, 1].barh(range(len(top_features)), range_values, 
                       color=self.colors[3])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=8)
        axes[1, 1].set_xlabel('Range (Max - Min)')
        axes[1, 1].set_title('Feature Range')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_cycle_life_distribution(self, analysis_results: List[Dict[str, Any]], 
                                   save_path: Optional[str] = None):
        """
        Plot cycle life distribution across batteries.
        
        Args:
            analysis_results: List of battery analysis results
            save_path: Optional path to save the plot
        """
        cycle_lives = []
        chemistries = []
        
        for analysis in analysis_results:
            cycle_life = analysis.get('cycle_life_analysis', {}).get('cycle_life', np.nan)
            if not np.isnan(cycle_life):
                cycle_lives.append(cycle_life)
                chemistry = analysis.get('basic_info', {}).get('cathode_material', 'Unknown')
                chemistries.append(chemistry)
        
        if not cycle_lives:
            print("No cycle life data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cycle Life Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(cycle_lives, bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].set_xlabel('Cycle Life')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cycle Life Distribution')
        axes[0, 0].axvline(np.mean(cycle_lives), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(cycle_lives):.1f}')
        axes[0, 0].axvline(np.median(cycle_lives), color='orange', linestyle='--', 
                          label=f'Median: {np.median(cycle_lives):.1f}')
        axes[0, 0].legend()
        
        # 2. Box plot
        axes[0, 1].boxplot(cycle_lives, patch_artist=True, 
                          boxprops=dict(facecolor=self.colors[1], alpha=0.7))
        axes[0, 1].set_ylabel('Cycle Life')
        axes[0, 1].set_title('Cycle Life Box Plot')
        axes[0, 1].set_xticklabels(['All Batteries'])
        
        # 3. Chemistry comparison (if we have chemistry data)
        if len(set(chemistries)) > 1:
            chemistry_data = {}
            for cycle_life, chemistry in zip(cycle_lives, chemistries):
                if chemistry not in chemistry_data:
                    chemistry_data[chemistry] = []
                chemistry_data[chemistry].append(cycle_life)
            
            chemistry_names = list(chemistry_data.keys())
            chemistry_values = [chemistry_data[chem] for chem in chemistry_names]
            
            axes[1, 0].boxplot(chemistry_values, labels=chemistry_names, patch_artist=True)
            axes[1, 0].set_ylabel('Cycle Life')
            axes[1, 0].set_title('Cycle Life by Chemistry')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient chemistry data\nfor comparison', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Chemistry Comparison')
        
        # 4. Statistics summary
        stats_text = f"Total Batteries: {len(cycle_lives)}\n"
        stats_text += f"Mean: {np.mean(cycle_lives):.1f}\n"
        stats_text += f"Median: {np.median(cycle_lives):.1f}\n"
        stats_text += f"Std: {np.std(cycle_lives):.1f}\n"
        stats_text += f"Min: {np.min(cycle_lives):.1f}\n"
        stats_text += f"Max: {np.max(cycle_lives):.1f}\n"
        stats_text += f"Q25: {np.percentile(cycle_lives, 25):.1f}\n"
        stats_text += f"Q75: {np.percentile(cycle_lives, 75):.1f}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, 
                       transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Statistics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_capacity_analysis(self, analysis_results: List[Dict[str, Any]], 
                             save_path: Optional[str] = None):
        """
        Plot capacity-related analysis.
        
        Args:
            analysis_results: List of battery analysis results
            save_path: Optional path to save the plot
        """
        # Extract capacity data
        discharge_capacities = []
        charge_capacities = []
        capacity_retentions = []
        cycle_lives = []
        
        for analysis in analysis_results:
            # Discharge capacity
            discharge_mean = analysis.get('capacity_stats', {}).get('discharge_capacity', {}).get('mean', np.nan)
            if not np.isnan(discharge_mean):
                discharge_capacities.append(discharge_mean)
            
            # Charge capacity
            charge_mean = analysis.get('capacity_stats', {}).get('charge_capacity', {}).get('mean', np.nan)
            if not np.isnan(charge_mean):
                charge_capacities.append(charge_mean)
            
            # Capacity retention
            retention = analysis.get('capacity_stats', {}).get('capacity_retention', np.nan)
            if not np.isnan(retention):
                capacity_retentions.append(retention)
            
            # Cycle life
            cycle_life = analysis.get('cycle_life_analysis', {}).get('cycle_life', np.nan)
            if not np.isnan(cycle_life):
                cycle_lives.append(cycle_life)
        
        if not discharge_capacities and not charge_capacities:
            print("No capacity data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Capacity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Discharge capacity distribution
        if discharge_capacities:
            axes[0, 0].hist(discharge_capacities, bins=20, alpha=0.7, 
                           color=self.colors[0], edgecolor='black')
            axes[0, 0].set_xlabel('Discharge Capacity (Ah)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Discharge Capacity Distribution')
            axes[0, 0].axvline(np.mean(discharge_capacities), color='red', linestyle='--',
                              label=f'Mean: {np.mean(discharge_capacities):.3f}')
            axes[0, 0].legend()
        
        # 2. Charge vs Discharge capacity
        if discharge_capacities and charge_capacities:
            min_len = min(len(discharge_capacities), len(charge_capacities))
            axes[0, 1].scatter(discharge_capacities[:min_len], charge_capacities[:min_len], 
                              alpha=0.6, color=self.colors[1])
            axes[0, 1].set_xlabel('Discharge Capacity (Ah)')
            axes[0, 1].set_ylabel('Charge Capacity (Ah)')
            axes[0, 1].set_title('Charge vs Discharge Capacity')
            
            # Add diagonal line
            max_val = max(max(discharge_capacities), max(charge_capacities))
            axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # 3. Capacity retention
        if capacity_retentions:
            axes[1, 0].hist(capacity_retentions, bins=20, alpha=0.7, 
                           color=self.colors[2], edgecolor='black')
            axes[1, 0].set_xlabel('Capacity Retention')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Capacity Retention Distribution')
            axes[1, 0].axvline(np.mean(capacity_retentions), color='red', linestyle='--',
                              label=f'Mean: {np.mean(capacity_retentions):.3f}')
            axes[1, 0].legend()
        
        # 4. Capacity vs Cycle Life
        if discharge_capacities and cycle_lives:
            min_len = min(len(discharge_capacities), len(cycle_lives))
            axes[1, 1].scatter(discharge_capacities[:min_len], cycle_lives[:min_len], 
                              alpha=0.6, color=self.colors[3])
            axes[1, 1].set_xlabel('Discharge Capacity (Ah)')
            axes[1, 1].set_ylabel('Cycle Life')
            axes[1, 1].set_title('Capacity vs Cycle Life')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, dataset_analyzer, output_dir: str):
        """
        Create a comprehensive analysis report with all visualizations.
        
        Args:
            dataset_analyzer: DatasetAnalyzer object with analysis results
            output_dir: Directory to save the report
        """
        import os
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = dataset_analyzer.dataset_name
        
        # 1. Dataset overview
        self.plot_dataset_overview(
            dataset_analyzer.summary_stats,
            save_path=str(output_dir / f"{dataset_name}_overview.png")
        )
        
        # 2. Feature distribution
        features_df = dataset_analyzer.get_feature_summary_table()
        if not features_df.empty:
            self.plot_feature_distribution(
                features_df,
                save_path=str(output_dir / f"{dataset_name}_features.png")
            )
        
        # 3. Cycle life analysis
        if dataset_analyzer.analysis_results:
            self.plot_cycle_life_distribution(
                dataset_analyzer.analysis_results,
                save_path=str(output_dir / f"{dataset_name}_cycle_life.png")
            )
            
            # 4. Capacity analysis
            self.plot_capacity_analysis(
                dataset_analyzer.analysis_results,
                save_path=str(output_dir / f"{dataset_name}_capacity.png")
            )
        
        print(f"Comprehensive report saved to {output_dir}")
    
    def plot_battery_comparison(self, battery_analyzers: List, 
                              save_path: Optional[str] = None):
        """
        Plot comparison between multiple batteries.
        
        Args:
            battery_analyzers: List of BatteryAnalyzer objects
            save_path: Optional path to save the plot
        """
        if not battery_analyzers:
            print("No battery analyzers provided")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Battery Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        cell_ids = []
        cycle_lives = []
        total_cycles = []
        capacities = []
        chemistries = []
        
        for analyzer in battery_analyzers:
            basic_info = analyzer.get_basic_info()
            cell_ids.append(basic_info['cell_id'])
            cycle_lives.append(basic_info['cycle_life'])
            total_cycles.append(basic_info['total_cycles'])
            chemistries.append(basic_info['cathode_material'])
            
            capacity_stats = analyzer.get_capacity_statistics()
            if 'discharge_capacity' in capacity_stats:
                capacities.append(capacity_stats['discharge_capacity']['mean'])
            else:
                capacities.append(np.nan)
        
        # 1. Cycle life comparison
        axes[0, 0].bar(range(len(cell_ids)), cycle_lives, color=self.colors[0])
        axes[0, 0].set_xlabel('Battery Index')
        axes[0, 0].set_ylabel('Cycle Life')
        axes[0, 0].set_title('Cycle Life Comparison')
        axes[0, 0].set_xticks(range(len(cell_ids)))
        axes[0, 0].set_xticklabels([f"B{i+1}" for i in range(len(cell_ids))], rotation=45)
        
        # 2. Total cycles vs cycle life
        axes[0, 1].scatter(total_cycles, cycle_lives, alpha=0.7, color=self.colors[1])
        axes[0, 1].set_xlabel('Total Cycles')
        axes[0, 1].set_ylabel('Cycle Life')
        axes[0, 1].set_title('Total Cycles vs Cycle Life')
        
        # 3. Capacity comparison
        valid_capacities = [(i, cap) for i, cap in enumerate(capacities) if not np.isnan(cap)]
        if valid_capacities:
            indices, caps = zip(*valid_capacities)
            axes[1, 0].bar(indices, caps, color=self.colors[2])
            axes[1, 0].set_xlabel('Battery Index')
            axes[1, 0].set_ylabel('Mean Discharge Capacity (Ah)')
            axes[1, 0].set_title('Capacity Comparison')
            axes[1, 0].set_xticks(indices)
            axes[1, 0].set_xticklabels([f"B{i+1}" for i in indices], rotation=45)
        
        # 4. Chemistry distribution
        chemistry_counts = {}
        for chem in chemistries:
            chemistry_counts[chem] = chemistry_counts.get(chem, 0) + 1
        
        if chemistry_counts:
            axes[1, 1].pie(chemistry_counts.values(), 
                          labels=chemistry_counts.keys(),
                          autopct='%1.1f%%',
                          startangle=90)
            axes[1, 1].set_title('Chemistry Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
