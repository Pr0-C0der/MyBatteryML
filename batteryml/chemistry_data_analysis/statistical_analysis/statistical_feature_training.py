#!/usr/bin/env python3
"""
Statistical Feature Training for RUL Prediction

This script:
1. Loads battery data and extracts cycle features
2. Calculates RUL labels for each cycle
3. Computes statistical features for each battery
4. Finds correlations between statistical features and RUL
5. Selects top features and trains XGBoost model
6. Evaluates performance and generates plots
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, DatasetSpecificCycleFeatures
from batteryml.label.rul import RULLabelAnnotator

# Suppress warnings
warnings.filterwarnings('ignore')


class StatisticalFeatureTrainer:
    """Trainer for RUL prediction using statistical features with highest correlation."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the statistical feature trainer.
        
        Args:
            data_dir: Base directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.rul_annotator = RULLabelAnnotator()
        self.model = None
        self.feature_names = None
        self.correlations = None
        
        # Define cycle features to extract
        self.cycle_feature_names = [
            'avg_voltage', 'avg_current', 'avg_c_rate', 'cycle_length',
            'max_charge_capacity', 'max_discharge_capacity',
            'charge_cycle_length', 'discharge_cycle_length',
            'avg_charge_c_rate', 'avg_discharge_c_rate', 'max_charge_c_rate',
            'avg_charge_capacity', 'avg_discharge_capacity',
            'power_during_charge_cycle', 'power_during_discharge_cycle',
            'charge_to_discharge_time_ratio'
        ]
        
        # Statistical measures to calculate
        self.statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        
    def load_battery_data(self, dataset_name: str, cycle_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load battery data for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use (None for all)
            
        Returns:
            DataFrame containing battery data with cycle features
        """
        print(f"Loading battery data for dataset: {dataset_name}")
        print(f"Data directory: {self.data_dir}")
        
        data_path = self.data_dir / dataset_name
        print(f"Looking for data in: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
        
        # Load all battery files in the dataset
        battery_files = list(data_path.glob("*.pkl"))
        print(f"Found {len(battery_files)} PKL files")
        
        if not battery_files:
            raise FileNotFoundError(f"No PKL files found in {data_path}")
        
        all_cycle_data = []
        successful_loads = 0
        failed_loads = 0
        
        for file_path in tqdm(battery_files, desc="Loading batteries", unit="battery"):
            try:
                # Load battery data
                battery = BatteryData.load(file_path)
                print(f"Loaded battery {battery.cell_id} with {len(battery.cycle_data)} cycles")
                
                # Extract cycle features
                cycle_data = self._extract_cycle_features(battery, dataset_name)
                print(f"Extracted {len(cycle_data)} cycle records with {len(cycle_data.columns)} features")
                
                if not cycle_data.empty:
                    cycle_data['battery_id'] = battery.cell_id
                    all_cycle_data.append(cycle_data)
                    successful_loads += 1
                    print(f"Successfully processed battery {battery.cell_id}")
                else:
                    print(f"Warning: No cycle data extracted for battery {battery.cell_id}")
                    failed_loads += 1
                    
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                failed_loads += 1
                continue
        
        print(f"Successfully loaded {successful_loads} batteries, failed {failed_loads}")
        
        if not all_cycle_data:
            raise ValueError(f"No valid battery data found in {data_path}")
        
        # Combine all data
        data = pd.concat(all_cycle_data, ignore_index=True)
        print(f"Combined data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")
        
        # Apply cycle limit if specified
        if cycle_limit is not None:
            original_count = len(data)
            data = data[data['cycle_number'] <= cycle_limit]
            print(f"Applied cycle limit {cycle_limit}: {original_count} -> {len(data)} records")
        
        print(f"Final data: {len(data)} cycle records from {data['battery_id'].nunique()} batteries")
        return data
    
    def _extract_cycle_features(self, battery: BatteryData, dataset_name: str) -> pd.DataFrame:
        """
        Extract cycle features from battery data.
        
        Args:
            battery: BatteryData object
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with cycle features
        """
        print(f"Extracting features for battery {battery.cell_id} using dataset {dataset_name}")
        
        try:
            # Try chemistry-specific extractor first
            extractor_class = get_extractor_class(dataset_name)
            print(f"Extractor class: {extractor_class}")
            
            if extractor_class is not None:
                extractor = extractor_class()
                rows = []
                valid_cycles = 0
                
                for cycle in battery.cycle_data:
                    row = {'cycle_number': cycle.cycle_number}
                    valid_features = 0
                    
                    for feature_name in self.cycle_feature_names:
                        if hasattr(extractor, feature_name):
                            try:
                                value = getattr(extractor, feature_name)(battery, cycle)
                                if value is not None and np.isfinite(float(value)):
                                    row[feature_name] = float(value)
                                    valid_features += 1
                                else:
                                    row[feature_name] = np.nan
                            except Exception as e:
                                row[feature_name] = np.nan
                        else:
                            row[feature_name] = np.nan
                    
                    rows.append(row)
                    if valid_features > 0:
                        valid_cycles += 1
                
                result = pd.DataFrame(rows)
                print(f"Chemistry extractor: {len(result)} cycles, {valid_cycles} with valid features")
                return result
            else:
                print("No chemistry-specific extractor found, using basic extraction")
                return self._extract_basic_features(battery)
                
        except Exception as e:
            print(f"Warning: Chemistry-specific extractor failed: {e}")
            return self._extract_basic_features(battery)
    
    def _extract_basic_features(self, battery: BatteryData) -> pd.DataFrame:
        """
        Extract basic cycle features as fallback.
        
        Args:
            battery: BatteryData object
            
        Returns:
            DataFrame with basic cycle features
        """
        print(f"Using basic feature extraction for battery {battery.cell_id}")
        rows = []
        valid_cycles = 0
        
        for cycle in battery.cycle_data:
            if not cycle.discharge_capacity_in_Ah or not cycle.charge_capacity_in_Ah:
                continue
                
            row = {
                'cycle_number': cycle.cycle_number,
                'avg_voltage': np.mean(cycle.voltage_in_V) if cycle.voltage_in_V is not None else np.nan,
                'avg_current': np.mean(cycle.current_in_A) if cycle.current_in_A is not None else np.nan,
                'cycle_length': len(cycle.voltage_in_V) if cycle.voltage_in_V is not None else 0,
                'max_charge_capacity': cycle.charge_capacity_in_Ah,
                'max_discharge_capacity': cycle.discharge_capacity_in_Ah,
            }
            
            # Fill other features with NaN
            for feature in self.cycle_feature_names:
                if feature not in row:
                    row[feature] = np.nan
            
            rows.append(row)
            valid_cycles += 1
        
        result = pd.DataFrame(rows)
        print(f"Basic extractor: {len(result)} cycles processed, {valid_cycles} valid")
        return result
    
    def calculate_rul_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RUL labels for the battery data.
        
        Args:
            data: DataFrame with cycle features
            
        Returns:
            DataFrame with RUL labels added
        """
        print("Calculating RUL labels...")
        
        data_with_rul = []
        for battery_id, battery_data in data.groupby('battery_id'):
            try:
                # Calculate total cycles for this battery
                max_cycle = battery_data['cycle_number'].max()
                
                # Calculate RUL for each cycle
                battery_data = battery_data.copy()
                battery_data['rul'] = max_cycle - battery_data['cycle_number']
                
                # Add log(RUL) for correlation analysis
                battery_data['log_rul'] = np.log(battery_data['rul'] + 1)  # +1 to avoid log(0)
                
                data_with_rul.append(battery_data)
                
            except Exception as e:
                print(f"Warning: Could not calculate RUL for battery {battery_id}: {e}")
                continue
        
        if not data_with_rul:
            raise ValueError("No valid RUL labels could be calculated")
        
        result = pd.concat(data_with_rul, ignore_index=True)
        print(f"RUL calculated for {result['battery_id'].nunique()} batteries")
        print(f"log_rul range: {result['log_rul'].min():.3f} to {result['log_rul'].max():.3f}")
        
        return result
    
    def calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical features for each battery.
        
        Args:
            data: DataFrame with cycle features and RUL labels
            
        Returns:
            DataFrame with statistical features
        """
        print("Calculating statistical features...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['battery_id', 'cycle_number', 'log_rul', 'rul']]
        
        print(f"Processing {len(feature_cols)} features with {len(self.statistical_measures)} statistical measures")
        
        battery_stats = []
        
        for battery_id, battery_data in tqdm(data.groupby('battery_id'), desc="Processing batteries", unit="battery"):
            try:
                # Calculate statistical measures for each feature
                battery_row = {'battery_id': battery_id}
                
                for feature in feature_cols:
                    feature_values = battery_data[feature].dropna().values
                    
                    if len(feature_values) > 0:
                        for measure in self.statistical_measures:
                            stat_value = self._calculate_statistical_measure(feature_values, measure)
                            battery_row[f"{feature}_{measure}"] = stat_value
                    else:
                        # Fill with NaN if no valid values
                        for measure in self.statistical_measures:
                            battery_row[f"{feature}_{measure}"] = np.nan
                
                # Add RUL information
                rul_values = battery_data['log_rul'].dropna().values
                if len(rul_values) > 0:
                    battery_row['log_rul'] = np.mean(rul_values)
                    battery_row['rul'] = np.exp(np.mean(rul_values)) - 1  # Convert back to actual RUL
                    battery_stats.append(battery_row)
                
            except Exception as e:
                print(f"Warning: Could not process battery {battery_id}: {e}")
                continue
        
        if not battery_stats:
            raise ValueError("No valid statistical features could be calculated")
        
        result = pd.DataFrame(battery_stats)
        print(f"Statistical features calculated for {len(result)} batteries")
        
        return result
    
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
            else:
                return np.nan
        except Exception:
            return np.nan
    
    def calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between statistical features and RUL.
        
        Args:
            data: DataFrame with statistical features
            
        Returns:
            DataFrame with correlation results
        """
        print("Calculating feature-RUL correlations...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['battery_id', 'log_rul', 'rul']]
        
        print(f"Calculating correlations for {len(feature_cols)} features")
        print(f"First 5 features: {feature_cols[:5]}")
        
        correlations = []
        skipped_features = []
        
        for feature in tqdm(feature_cols, desc="Calculating correlations", unit="feature"):
            try:
                # Get valid data points
                valid_data = data[['log_rul', feature]].dropna()
                
                if len(valid_data) < 2:
                    skipped_features.append(f"{feature} (insufficient data: {len(valid_data)} samples)")
                    continue
                
                # Check if feature has any variation
                if valid_data[feature].std() == 0:
                    skipped_features.append(f"{feature} (no variation)")
                    continue
                
                # Calculate Spearman correlation
                correlation, _ = spearmanr(valid_data['log_rul'], valid_data[feature], nan_policy='omit')
                
                if not np.isnan(correlation):
                    correlations.append({
                        'feature': feature,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'n_samples': len(valid_data)
                    })
                else:
                    skipped_features.append(f"{feature} (NaN correlation)")
                            
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature}: {e}")
                skipped_features.append(f"{feature} (error: {str(e)[:50]})")
                continue

        if not correlations:
            raise ValueError("No valid correlations could be calculated")
        
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False)
        
        print(f"Calculated {len(correlations)} valid correlations")
        return correlation_df
    
    def select_top_features(self, correlation_df: pd.DataFrame, n_features: int = 15) -> List[str]:
        """
        Select top N features with highest absolute correlation to RUL.
        
        Args:
            correlation_df: DataFrame with correlation results
            n_features: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        print(f"Selecting top {n_features} features...")
        
        if correlation_df.empty:
            raise ValueError("No correlations available for feature selection")
        
        top_features = correlation_df.head(n_features)['feature'].tolist()
        
        print("Selected features:")
        for i, (_, row) in enumerate(correlation_df.head(n_features).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:40s} (correlation: {row['correlation']:6.3f}, n={row['n_samples']})")
        
        return top_features
    
    def prepare_training_data(self, data: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with selected features.
        
        Args:
            data: DataFrame with statistical features
            feature_names: List of selected feature names
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Preparing training data...")
        
        # Check if all features exist
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Prepare feature matrix
        X = data[feature_names].values
        y = data['log_rul'].values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Training data shape: {X.shape}")
        print(f"Target range: {y.min():.3f} to {y.max():.3f}")
        
        return X, y
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Train XGBoost model and evaluate performance.
        
        Args:
            X_train: Training features
            y_train: Training targets (log RUL)
            X_test: Test features
            y_test: Test targets (log RUL)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Training XGBoost model...")
        
        # Initialize XGBoost regressor
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Convert to actual RUL for evaluation
        y_train_actual = np.exp(y_train)
        y_test_actual = np.exp(y_test)
        y_train_pred_actual = np.exp(y_train_pred)
        y_test_pred_actual = np.exp(y_test_pred)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
        
        train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
        
        train_mape = np.mean(np.abs((y_train_actual - y_train_pred_actual) / y_train_actual)) * 100
        test_mape = np.mean(np.abs((y_test_actual - y_test_pred_actual) / y_test_actual)) * 100
        
        train_r2 = r2_score(y_train_actual, y_train_pred_actual)
        test_r2 = r2_score(y_test_actual, y_test_pred_actual)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS (on actual RUL)")
        print("="*60)
        print(f"Training RMSE: {train_rmse:.2f} cycles")
        print(f"Test RMSE:     {test_rmse:.2f} cycles")
        print(f"Training MAE:  {train_mae:.2f} cycles")
        print(f"Test MAE:      {test_mae:.2f} cycles")
        print(f"Training MAPE: {train_mape:.2f}%")
        print(f"Test MAPE:     {test_mape:.2f}%")
        print(f"Training R²:   {train_r2:.3f}")
        print(f"Test R²:       {test_r2:.3f}")
        print("="*60)
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def plot_feature_importance(self, output_path: Optional[str] = None):
        """Plot feature importance from the trained model."""
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting feature importance...")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance (Top Correlated Features)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Feature importance plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                        output_path: Optional[str] = None):
        """Plot actual vs predicted RUL values."""
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting predictions...")
        
        y_test_pred = self.model.predict(X_test)
        
        # Convert to actual RUL for plotting
        y_test_actual = np.exp(y_test)
        y_test_pred_actual = np.exp(y_test_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_test_actual.min(), y_test_pred_actual.min())
        max_val = max(y_test_actual.max(), y_test_pred_actual.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual RUL (cycles)')
        plt.ylabel('Predicted RUL (cycles)')
        plt.title('Actual vs Predicted RUL (Test Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_test_actual, y_test_pred_actual)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Predictions plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def train_and_evaluate(self, dataset_name: str, cycle_limit: Optional[int] = None, 
                          n_features: int = 15, test_size: float = 0.3,
                          random_state: int = 42) -> Dict[str, float]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use
            n_features: Number of top features to select
            test_size: Test set size (0.3 for 70/30 split)
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Starting statistical feature training for dataset: {dataset_name}")
        print("=" * 60)
        
        # Load and process data
        data = self.load_battery_data(dataset_name, cycle_limit)
        data_with_rul = self.calculate_rul_labels(data)
        statistical_data = self.calculate_statistical_features(data_with_rul)
        
        # Calculate correlations and select features
        correlation_df = self.calculate_correlations(statistical_data)
        self.correlations = correlation_df
        self.feature_names = self.select_top_features(correlation_df, n_features)
        
        # Prepare training data
        X, y = self.prepare_training_data(statistical_data, self.feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train model
        metrics = self.train_model(X_train, y_train, X_test, y_test)
        
        return metrics


def main():
    """Main function to run statistical feature training."""
    parser = argparse.ArgumentParser(description='Train XGBoost on statistical features with highest RUL correlation')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--cycle_limit', '-c', type=int, default=None, 
                       help='Maximum number of cycles to use (default: all)')
    parser.add_argument('--n_features', '-n', type=int, default=15, 
                       help='Number of top features to select (default: 15)')
    parser.add_argument('--test_size', '-t', type=float, default=0.3, 
                       help='Test set size (default: 0.3 for 70/30 split)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output_dir', '-o', default='statistical_training_results', 
                       help='Output directory for plots (default: statistical_training_results)')
    parser.add_argument('--random_state', '-r', type=int, default=42, 
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize trainer
        trainer = StatisticalFeatureTrainer(args.data_dir)
        
        # Run training and evaluation
        metrics = trainer.train_and_evaluate(
            dataset_name=args.dataset_name,
            cycle_limit=args.cycle_limit,
            n_features=args.n_features,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Generate plots
        if trainer.model is not None:
            print("Generating plots...")
            
            # Feature importance plot
            importance_path = output_dir / f"feature_importance_{args.dataset_name}.png"
            trainer.plot_feature_importance(str(importance_path))
            
            # Predictions plot
            data = trainer.load_battery_data(args.dataset_name, args.cycle_limit)
            data_with_rul = trainer.calculate_rul_labels(data)
            statistical_data = trainer.calculate_statistical_features(data_with_rul)
            X, y = trainer.prepare_training_data(statistical_data, trainer.feature_names)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state
            )
            
            predictions_path = output_dir / f"predictions_{args.dataset_name}.png"
            trainer.plot_predictions(X_test, y_test, str(predictions_path))
        
        print(f"\nResults saved to: {output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())