#!/usr/bin/env python3
"""
Statistical Feature Training for RUL Prediction

This script:
1. Calculates feature-RUL correlations for a dataset
2. Selects the top 15 features with highest absolute correlation to RUL
3. Trains XGBoost on these features to predict RUL
4. Uses 70/30 train/test split
5. Reports RMSE, MAE, and MAPE metrics
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import extract_cycle_features
from feature_rul_correlation import FeatureRULCorrelationAnalyzer

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
        self.data_dir = data_dir
        self.correlation_analyzer = FeatureRULCorrelationAnalyzer(data_dir)
        self.model = None
        self.feature_names = None
        self.correlations = None
        
    def load_and_extract_features(self, dataset_name: str, cycle_limit: int = None) -> pd.DataFrame:
        """
        Load battery data and extract cycle features using the correlation analyzer.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use (None for all)
            
        Returns:
            DataFrame with cycle features
        """
        print(f"Loading and extracting features for dataset: {dataset_name}")
        
        # Use the correlation analyzer to load data
        data = self.correlation_analyzer.load_battery_data(dataset_name)
        
        # Apply cycle limit if specified
        if cycle_limit is not None:
            data = data[data['cycle_number'] <= cycle_limit]
        
        return data
    
    def calculate_rul_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RUL labels for the battery data.
        The data already has log_rul from the correlation analyzer.
        
        Args:
            data: DataFrame with cycle features
            
        Returns:
            DataFrame with RUL labels (already present)
        """
        print("RUL labels already calculated by correlation analyzer...")
        
        # The data already has log_rul from the correlation analyzer
        # We just need to ensure it's present
        if 'log_rul' not in data.columns:
            raise ValueError("log_rul column not found in data. Check correlation analyzer setup.")
        
        return data
    
    def calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between battery-level aggregated features and RUL.
        
        Args:
            data: DataFrame with cycle features and RUL labels
            
        Returns:
            DataFrame with correlation results
        """
        print("Calculating battery-level feature-RUL correlations...")
        
        # Get feature columns (exclude metadata columns)
        feature_cols = [col for col in data.columns 
                       if col not in ['battery_id', 'cycle_number', 'log_rul']]
        
        # Statistical measures to calculate for each feature
        statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        
        correlations = []
        
        print(f"Calculating correlations for {len(feature_cols)} features with {len(statistical_measures)} statistical measures...")
        
        for feature in tqdm(feature_cols, desc="Calculating correlations", unit="feature"):
            # Calculate correlations for each statistical measure
            for measure in statistical_measures:
                try:
                    # Get battery-level aggregated values
                    battery_features = []
                    battery_ruls = []
                    
                    for battery_id, battery_data in data.groupby('battery_id'):
                        feature_values = battery_data[feature].dropna().values
                        rul_values = battery_data['log_rul'].dropna().values
                        
                        if len(feature_values) > 0 and len(rul_values) > 0:
                            # Calculate statistical measure for this battery
                            if measure == 'mean':
                                stat_value = np.mean(feature_values)
                            elif measure == 'std':
                                stat_value = np.std(feature_values)
                            elif measure == 'min':
                                stat_value = np.min(feature_values)
                            elif measure == 'max':
                                stat_value = np.max(feature_values)
                            elif measure == 'median':
                                stat_value = np.median(feature_values)
                            elif measure == 'q25':
                                stat_value = np.percentile(feature_values, 25)
                            elif measure == 'q75':
                                stat_value = np.percentile(feature_values, 75)
                            
                            if not np.isnan(stat_value):
                                battery_features.append(stat_value)
                                # Use mean RUL for this battery
                                battery_ruls.append(np.mean(rul_values))
                    
                    # Calculate correlation if we have enough data points
                    if len(battery_features) > 1:
                        from scipy.stats import spearmanr
                        correlation, _ = spearmanr(battery_features, battery_ruls, nan_policy='omit')
                        
                        if not np.isnan(correlation):
                            correlations.append({
                                'feature': f"{feature}_{measure}",
                                'base_feature': feature,
                                'measure': measure,
                                'correlation': correlation,
                                'abs_correlation': abs(correlation)
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for {feature}_{measure}: {e}")
                    continue
        
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False)
        
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
        print(f"Selecting top {n_features} features with highest correlation to RUL...")
        
        top_features = correlation_df.head(n_features)['feature'].tolist()
        
        print(f"Selected features:")
        for i, (_, row) in enumerate(correlation_df.head(n_features).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:30s} (correlation: {row['correlation']:6.3f})")
        
        return top_features
    
    def prepare_training_data(self, data: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with selected battery-level aggregated features.
        
        Args:
            data: DataFrame with cycle features and RUL labels
            feature_names: List of selected feature names (e.g., 'avg_voltage_mean', 'cycle_length_std')
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Preparing battery-level training data...")
        
        # Create battery-level aggregated features
        battery_features = []
        battery_ruls = []
        
        print(f"Creating battery-level features for {data['battery_id'].nunique()} batteries...")
        
        for battery_id, battery_data in tqdm(data.groupby('battery_id'), desc="Aggregating features", unit="battery"):
            # Calculate aggregated features for this battery
            battery_row = {}
            
            for feature_name in feature_names:
                # Parse feature name (e.g., 'avg_voltage_mean' -> 'avg_voltage', 'mean')
                if '_' in feature_name:
                    base_feature, measure = feature_name.rsplit('_', 1)
                else:
                    base_feature = feature_name
                    measure = 'mean'
                
                if base_feature in battery_data.columns:
                    feature_values = battery_data[base_feature].dropna().values
                    
                    if len(feature_values) > 0:
                        # Calculate the statistical measure
                        if measure == 'mean':
                            stat_value = np.mean(feature_values)
                        elif measure == 'std':
                            stat_value = np.std(feature_values)
                        elif measure == 'min':
                            stat_value = np.min(feature_values)
                        elif measure == 'max':
                            stat_value = np.max(feature_values)
                        elif measure == 'median':
                            stat_value = np.median(feature_values)
                        elif measure == 'q25':
                            stat_value = np.percentile(feature_values, 25)
                        elif measure == 'q75':
                            stat_value = np.percentile(feature_values, 75)
                        else:
                            stat_value = np.nan
                        
                        battery_row[feature_name] = stat_value
                    else:
                        battery_row[feature_name] = np.nan
                else:
                    battery_row[feature_name] = np.nan
            
            # Get RUL for this battery (use mean log_rul)
            rul_values = battery_data['log_rul'].dropna().values
            if len(rul_values) > 0:
                battery_rul = np.mean(rul_values)
                
                # Only include if we have valid features and RUL
                if not np.isnan(battery_rul) and not all(np.isnan(list(battery_row.values()))):
                    battery_features.append(list(battery_row.values()))
                    battery_ruls.append(battery_rul)
        
        # Convert to numpy arrays
        X = np.array(battery_features)
        y = np.array(battery_ruls)
        
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
            Dictionary with evaluation metrics (calculated on actual RUL)
        """
        print("Training XGBoost model...")
        print("Training on log(RUL), evaluating on actual RUL")
        
        # Initialize XGBoost regressor
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=1  # Enable progress output
        )
        
        # Train the model with progress bar
        print("Training XGBoost model (this may take a few minutes)...")
        self.model.fit(X_train, y_train)
        
        # Make predictions (these are log RUL predictions)
        y_train_pred_log = self.model.predict(X_train)
        y_test_pred_log = self.model.predict(X_test)
        
        # Convert predictions and targets to actual RUL for evaluation
        y_train_actual = np.exp(y_train)  # Convert log RUL to actual RUL
        y_test_actual = np.exp(y_test)    # Convert log RUL to actual RUL
        y_train_pred_actual = np.exp(y_train_pred_log)  # Convert log RUL predictions to actual RUL
        y_test_pred_actual = np.exp(y_test_pred_log)    # Convert log RUL predictions to actual RUL
        
        # Calculate metrics on actual RUL
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
        
        train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
        
        # Calculate MAPE on actual RUL
        train_mape = np.mean(np.abs((y_train_actual - y_train_pred_actual) / y_train_actual)) * 100
        test_mape = np.mean(np.abs((y_test_actual - y_test_pred_actual) / y_test_actual)) * 100
        
        # Print detailed results
        print("\n" + "="*60)
        print("EVALUATION RESULTS (on actual RUL)")
        print("="*60)
        print(f"Training RMSE: {train_rmse:.2f} cycles")
        print(f"Test RMSE:     {test_rmse:.2f} cycles")
        print(f"Training MAE:  {train_mae:.2f} cycles")
        print(f"Test MAE:      {test_mae:.2f} cycles")
        print(f"Training MAPE: {train_mape:.2f}%")
        print(f"Test MAPE:     {test_mape:.2f}%")
        print("="*60)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape
        }
        
        return metrics
    
    def plot_feature_importance(self, output_path: str = None):
        """
        Plot feature importance from the trained model.
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = self.feature_names
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance (Top 15 Correlated Features)')
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
                        output_path: str = None):
        """
        Plot actual vs predicted RUL values.
        
        Args:
            X_test: Test features
            y_test: Test targets (log RUL)
            output_path: Path to save the plot (optional)
        """
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting predictions...")
        
        # Make predictions (these are log RUL predictions)
        y_test_pred_log = self.model.predict(X_test)
        
        # Convert to actual RUL for plotting
        y_test_actual = np.exp(y_test)  # Convert log RUL to actual RUL
        y_test_pred_actual = np.exp(y_test_pred_log)  # Convert log RUL predictions to actual RUL
        
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
        from sklearn.metrics import r2_score
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
    
    def train_and_evaluate(self, dataset_name: str, cycle_limit: int = None, 
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
        
        # Create progress bar for main steps
        main_steps = [
            "Loading and extracting features",
            "Calculating RUL labels", 
            "Calculating correlations",
            "Selecting top features",
            "Preparing training data",
            "Training XGBoost model"
        ]
        
        with tqdm(total=len(main_steps), desc="Training Pipeline", unit="step") as pbar:
            # Load and extract features
            pbar.set_description("Loading and extracting features")
            data = self.load_and_extract_features(dataset_name, cycle_limit)
            print(f"Loaded {len(data)} cycle records from {data['battery_id'].nunique()} batteries")
            pbar.update(1)
            
            # Calculate RUL labels
            pbar.set_description("Calculating RUL labels")
            data_with_rul = self.calculate_rul_labels(data)
            pbar.update(1)
            
            # Calculate correlations
            pbar.set_description("Calculating correlations")
            correlation_df = self.calculate_correlations(data_with_rul)
            self.correlations = correlation_df
            pbar.update(1)
            
            # Select top features
            pbar.set_description("Selecting top features")
            self.feature_names = self.select_top_features(correlation_df, n_features)
            pbar.update(1)
            
            # Prepare training data
            pbar.set_description("Preparing training data")
            X, y = self.prepare_training_data(data_with_rul, self.feature_names)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Test set size: {X_test.shape[0]}")
            pbar.update(1)
            
            # Train model
            pbar.set_description("Training XGBoost model")
            metrics = self.train_model(X_train, y_train, X_test, y_test)
            pbar.update(1)
        
        # Print results
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Features used: {len(self.feature_names)}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print()
        print("TRAINING METRICS:")
        print(f"  RMSE: {metrics['train_rmse']:.3f}")
        print(f"  MAE:  {metrics['train_mae']:.3f}")
        print(f"  MAPE: {metrics['train_mape']:.2f}%")
        print()
        print("TEST METRICS:")
        print(f"  RMSE: {metrics['test_rmse']:.3f}")
        print(f"  MAE:  {metrics['test_mae']:.3f}")
        print(f"  MAPE: {metrics['test_mape']:.2f}%")
        print("=" * 60)
        
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
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
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
            print("  - Creating feature importance plot...")
            importance_path = output_dir / f"feature_importance_{args.dataset_name}.png"
            trainer.plot_feature_importance(str(importance_path))
            
            # Predictions plot (need to reload test data)
            print("  - Creating predictions plot...")
            data = trainer.load_and_extract_features(args.dataset_name, args.cycle_limit)
            data_with_rul = trainer.calculate_rul_labels(data)
            X, y = trainer.prepare_training_data(data_with_rul, trainer.feature_names)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state
            )
            
            predictions_path = output_dir / f"predictions_{args.dataset_name}.png"
            trainer.plot_predictions(X_test, y_test, str(predictions_path))
        
        print(f"\nResults saved to: {output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
