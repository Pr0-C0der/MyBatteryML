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
            'Qdlin': 'qdlin',
            'Tdlin': 'tdlin'
        }
        
        for attr, feature_name in feature_mapping.items():
            if hasattr(first_cycle, attr):
                attr_value = getattr(first_cycle, attr)
                if attr_value is None:
                    continue
                # Consider scalars as available; for sequences require non-empty
                if np.isscalar(attr_value):
                    available_features.append(feature_name)
                else:
                    try:
                        if len(attr_value) > 0:
                            available_features.append(feature_name)
                    except TypeError:
                        # Fallback: treat as available if not None
                        available_features.append(feature_name)

        # Add new derived features to advertised list
        for extra in ['avg_c_rate', 'peak_cc_length', 'peak_cv_length', 'cycle_length', 'max_temperature', 'max_discharge_capacity', 'max_charge_capacity', 'charge_cycle_length', 'discharge_cycle_length']:
            if extra not in available_features:
                available_features.append(extra)
        
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
                'qdlin': 'Qdlin',
                'tdlin': 'Tdlin',
                # New derived features (computed below)
                'avg_c_rate': 'avg_c_rate',
                'peak_cc_length': 'peak_cc_length',
                'peak_cv_length': 'peak_cv_length',
                'cycle_length': 'cycle_length',
                'max_temperature': 'max_temperature',
                'max_discharge_capacity': 'max_discharge_capacity',
                'max_charge_capacity': 'max_charge_capacity',
                'charge_cycle_length': 'charge_cycle_length',
                'discharge_cycle_length': 'discharge_cycle_length',
            }
            
            for feature_name in self.features:
                if feature_name in feature_mapping:
                    attr_name = feature_mapping[feature_name]
                    
                    # Derived computations
                    if feature_name == 'avg_c_rate':
                        try:
                            I = np.array(cycle_data.current_in_A or [])
                            I = I[~np.isnan(I)]
                            C = battery.nominal_capacity_in_Ah or 0.0
                            row_data['avg_c_rate'] = float(np.mean(np.abs(I))/C) if (I.size>0 and C>0) else np.nan
                        except Exception:
                            row_data['avg_c_rate'] = np.nan
                    elif feature_name == 'max_temperature':
                        try:
                            temp_data = np.array(cycle_data.temperature_in_C or [])
                            if temp_data.size > 0:
                                max_temp = np.nanmax(temp_data)
                                row_data['max_temperature'] = float(max_temp) if not np.isnan(max_temp) else np.nan
                            else:
                                row_data['max_temperature'] = np.nan
                        except Exception:
                            row_data['max_temperature'] = np.nan
                    elif feature_name == 'max_discharge_capacity':
                        try:
                            cap_data = np.array(cycle_data.discharge_capacity_in_Ah or [])
                            if cap_data.size > 0:
                                max_cap = np.nanmax(cap_data)
                                row_data['max_discharge_capacity'] = float(max_cap) if not np.isnan(max_cap) else np.nan
                            else:
                                row_data['max_discharge_capacity'] = np.nan
                        except Exception:
                            row_data['max_discharge_capacity'] = np.nan
                    elif feature_name == 'max_charge_capacity':
                        try:
                            cap_data = np.array(cycle_data.charge_capacity_in_Ah or [])
                            if cap_data.size > 0:
                                max_cap = np.nanmax(cap_data)
                                row_data['max_charge_capacity'] = float(max_cap) if not np.isnan(max_cap) else np.nan
                            else:
                                row_data['max_charge_capacity'] = np.nan
                        except Exception:
                            row_data['max_charge_capacity'] = np.nan
                    elif feature_name in ['charge_cycle_length', 'discharge_cycle_length']:
                        try:
                            current_data = np.array(cycle_data.current_in_A or [])
                            time_data = np.array(cycle_data.time_in_s or [])
                            if current_data.size > 0 and time_data.size > 0:
                                # Find current sign changes
                                sign_changes = np.where(np.diff(np.sign(current_data)))[0]
                                
                                if len(sign_changes) > 0:
                                    # Use first significant sign change as transition point
                                    transition_idx = sign_changes[0]
                                    
                                    if feature_name == 'charge_cycle_length':
                                        # Charge phase: start to transition point
                                        charge_length = time_data[transition_idx] - time_data[0]
                                        row_data['charge_cycle_length'] = float(charge_length) if not np.isnan(charge_length) else np.nan
                                    elif feature_name == 'discharge_cycle_length':
                                        # Discharge phase: transition point to end
                                        discharge_length = time_data[-1] - time_data[transition_idx]
                                        row_data['discharge_cycle_length'] = float(discharge_length) if not np.isnan(discharge_length) else np.nan
                                else:
                                    # Fallback: split at midpoint
                                    mid_idx = len(current_data) // 2
                                    if feature_name == 'charge_cycle_length':
                                        charge_length = time_data[mid_idx] - time_data[0]
                                        row_data['charge_cycle_length'] = float(charge_length) if not np.isnan(charge_length) else np.nan
                                    elif feature_name == 'discharge_cycle_length':
                                        discharge_length = time_data[-1] - time_data[mid_idx]
                                        row_data['discharge_cycle_length'] = float(discharge_length) if not np.isnan(discharge_length) else np.nan
                            else:
                                if feature_name == 'charge_cycle_length':
                                    row_data['charge_cycle_length'] = np.nan
                                elif feature_name == 'discharge_cycle_length':
                                    row_data['discharge_cycle_length'] = np.nan
                        except Exception:
                            if feature_name == 'charge_cycle_length':
                                row_data['charge_cycle_length'] = np.nan
                            elif feature_name == 'discharge_cycle_length':
                                row_data['discharge_cycle_length'] = np.nan
                    elif feature_name in ['peak_cc_length', 'peak_cv_length', 'cycle_length']:
                        try:
                            t = np.array(cycle_data.time_in_s or [])
                            V = np.array(cycle_data.voltage_in_V or [])
                            I = np.array(cycle_data.current_in_A or [])
                            if t.size == 0:
                                raise ValueError
                            # align and mask NaNs
                            n = min(len(t), len(V)) if V.size else len(t)
                            n = min(n, len(I)) if I.size else n
                            t = t[:n]
                            V = V[:n] if V.size else np.array([])
                            I = I[:n] if I.size else np.array([])
                            m_t = ~np.isnan(t)
                            t = t[m_t]
                            if V.size:
                                V = V[m_t]
                            if I.size:
                                I = I[m_t]
                            def last_peak_len(arr, tt):
                                if arr.size == 0:
                                    return np.nan
                                vmax = np.nanmax(arr)
                                close = np.isclose(arr, vmax, rtol=1e-3, atol=1e-6)
                                if not np.any(close):
                                    return np.nan
                                last_idx = np.where(close)[0][-1]
                                return float(tt[last_idx] - tt[0])
                            if feature_name == 'peak_cc_length':
                                row_data['peak_cc_length'] = last_peak_len(I, t)
                            elif feature_name == 'peak_cv_length':
                                row_data['peak_cv_length'] = last_peak_len(V, t)
                            elif feature_name == 'cycle_length':
                                row_data['cycle_length'] = float(t[-1] - t[0]) if t.size > 0 else np.nan
                        except Exception:
                            if feature_name == 'peak_cc_length':
                                row_data['peak_cc_length'] = np.nan
                            elif feature_name == 'peak_cv_length':
                                row_data['peak_cv_length'] = np.nan
                            elif feature_name == 'cycle_length':
                                row_data['cycle_length'] = np.nan
                    else:
                        # Original handling for non-derived features
                        if hasattr(cycle_data, attr_name):
                            feature_data = getattr(cycle_data, attr_name, None)
                            if feature_data is None:
                                row_data[feature_name] = np.nan
                            else:
                                if np.isscalar(feature_data):
                                    try:
                                        val = float(feature_data)
                                        row_data[feature_name] = val if not np.isnan(val) else np.nan
                                    except Exception:
                                        row_data[feature_name] = np.nan
                                else:
                                    try:
                                        feature_array = np.array(feature_data)
                                        valid_data = feature_array[~np.isnan(feature_array)]
                                        row_data[feature_name] = np.mean(valid_data) if valid_data.size > 0 else np.nan
                                    except Exception:
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

        # After individual analyses, generate per-feature correlation vs batteries plots
        try:
            features_to_plot = [f for f in self.features if f not in ['rul', 'cycle_number']]
            for feature in features_to_plot:
                self.plot_feature_rul_correlation_across_batteries(feature)
        except Exception as e:
            print(f"Warning: Failed generating feature-vs-batteries correlation plots: {e}")


    def _compute_feature_rul_corr(self, df: pd.DataFrame, feature: str) -> Optional[float]:
        """Compute Pearson correlation between provided feature and RUL for a single battery matrix.

        Returns NaN or None if insufficient data.
        """
        if 'rul' not in df.columns or feature not in df.columns:
            return None
        sub = df[["rul", feature]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        # Require variability
        try:
            if sub['rul'].nunique() < 2 or sub[feature].nunique() < 2:
                return None
            corr = sub.corr().loc['rul', feature]
            if pd.isna(corr):
                return None
            return float(corr)
        except Exception:
            return None

    def plot_feature_rul_correlation_across_batteries(self, feature: str):
        """Plot correlation(feature, RUL) vs batteries (sorted) across the dataset."""
        out_dir = self.output_dir / 'feature_vs_batteries'
        out_dir.mkdir(exist_ok=True)

        names: List[str] = []
        corrs: List[float] = []

        # Prefer using saved matrices generated during analyze_battery
        csv_files = list(self.matrices_dir.glob("*_cycle_feature_matrix.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            corr_val = self._compute_feature_rul_corr(df, feature)
            if corr_val is None:
                continue
            # Derive battery name from filename
            safe_stem = csv_path.stem
            # Remove trailing marker if present
            if safe_stem.endswith('_cycle_feature_matrix'):
                battery_name = safe_stem[:-len('_cycle_feature_matrix')]
            else:
                battery_name = safe_stem
            names.append(battery_name)
            corrs.append(corr_val)

        if not names:
            return

        # Sort by correlation value (ascending)
        order = np.argsort(np.array(corrs))
        names_sorted = [names[i] for i in order]
        corrs_sorted = [corrs[i] for i in order]

        # Plot as line with colored markers, mean/median lines, ±1σ band, annotations
        fig, ax = plt.subplots(figsize=(max(12, int(0.6 * len(names_sorted))), 6))
        x = np.arange(len(names_sorted))

        # Base line
        ax.plot(x, corrs_sorted, color='0.4', linewidth=1.5, alpha=0.9)

        # Colored markers by sign
        marker_colors = ['red' if v < 0 else ('blue' if v > 0 else 'gray') for v in corrs_sorted]
        ax.scatter(x, corrs_sorted, c=marker_colors, s=36, zorder=3)

        # Mean, median, and std band
        mean_val = float(np.mean(corrs_sorted))
        std_val = float(np.std(corrs_sorted))
        median_val = float(np.median(corrs_sorted))
        ax.axhline(mean_val, color='green', linestyle='--', linewidth=1.2, label=f'Mean = {mean_val:.3f}')
        ax.axhline(median_val, color='purple', linestyle=':', linewidth=1.2, label=f'Median = {median_val:.3f}')
        # ±1σ horizontal band across full width
        band_low = mean_val - std_val
        band_high = mean_val + std_val
        ax.axhspan(band_low, band_high, color='green', alpha=0.08, label='±1σ')

        # Set limits with padding to keep annotations inside
        y_min_raw = min(min(corrs_sorted), band_low)
        y_max_raw = max(max(corrs_sorted), band_high)
        y_pad = max(0.05, 0.06 * (y_max_raw - y_min_raw if y_max_raw > y_min_raw else 1.0))
        y_min = max(-1.05, y_min_raw - y_pad)
        y_max = min(1.05, y_max_raw + y_pad)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, len(names_sorted) - 0.5)

        # Annotate extremes: bottom-2 negatives and top-2 positives (handle fewer gracefully)
        neg_idxs = [i for i, v in enumerate(corrs_sorted) if v < 0]
        pos_idxs = [i for i, v in enumerate(corrs_sorted) if v > 0]
        bottom_idxs = neg_idxs[:min(2, len(neg_idxs))]
        top_idxs = pos_idxs[-min(2, len(pos_idxs)):] if len(pos_idxs) else []

        # Dynamic offset based on y-span and clamp inside limits
        y_span = max(1e-6, y_max - y_min)
        base_off = 0.04 * y_span
        for idx in bottom_idxs + top_idxs:
            if idx < 0 or idx >= len(x):
                continue
            y = corrs_sorted[idx]
            label = names_sorted[idx]
            offset = base_off if y >= mean_val else -base_off
            y_text = y + offset
            # Clamp text inside axis bounds with small margin
            margin = 0.02 * y_span
            y_text = min(max(y_text, y_min + margin), y_max - margin)
            va = 'bottom' if y_text >= y else 'top'
            ax.annotate(
                f"{label}\n{y:.3f}",
                xy=(x[idx], y),
                xytext=(x[idx], y_text),
                textcoords='data',
                ha='center',
                va=va,
                fontsize=8,
                color=marker_colors[idx],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
                arrowprops=dict(arrowstyle='-', color=marker_colors[idx], lw=0.8, alpha=0.7),
                annotation_clip=True
            )

        # Axes and labels
        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=45, ha='right')
        ax.set_ylabel(f'Correlation of {feature} with RUL')
        ax.set_xlabel('Battery')
        ax.set_title(f'{feature}–RUL Correlation Across Batteries (sorted)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f'{feature}_corr_vs_batteries.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


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
