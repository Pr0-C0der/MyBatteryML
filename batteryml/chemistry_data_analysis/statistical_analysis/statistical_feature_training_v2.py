#!/usr/bin/env python3
"""
Statistical Feature Training for RUL Prediction - Version 2

This script integrates with the existing working correlation analysis code
and adds machine learning training functionality.

Uses the proven FeatureRULCorrelationAnalyzer for data loading and correlation calculation.
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

# Import the working correlation analyzer
from feature_rul_correlation import FeatureRULCorrelationAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')


class StatisticalFeatureTrainerV2:
    """Trainer for RUL prediction using statistical features with proven correlation analysis."""
    
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
        
        # Statistical measures to calculate (same as working correlation analysis)
        self.statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        
    def load_and_prepare_data(self, dataset_name: str, cycle_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load battery data and prepare it for training using the working correlation analyzer.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use (None for all)
            
        Returns:
            DataFrame with cycle features and RUL labels
        """
        print(f"Loading and preparing data for dataset: {dataset_name}")
        
        # Use the working correlation analyzer to load data
        data = self.correlation_analyzer.load_battery_data(dataset_name)
        print(f"Loaded {len(data)} cycle records from {data['battery_id'].nunique()} batteries")
        
        # Calculate RUL labels using the working method
        data_with_rul = self.correlation_analyzer.calculate_rul_labels(data, dataset_name)
        print(f"RUL calculated for {data_with_rul['battery_id'].nunique()} batteries")
        print(f"log_rul range: {data_with_rul['log_rul'].min():.3f} to {data_with_rul['log_rul'].max():.3f}")
        
        # Apply cycle limit if specified
        if cycle_limit is not None:
            original_count = len(data_with_rul)
            data_with_rul = data_with_rul[data_with_rul['cycle_number'] <= cycle_limit]
            print(f"Applied cycle limit {cycle_limit}: {original_count} -> {len(data_with_rul)} records")
        
        return data_with_rul
    
    def calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical features for each battery using the working correlation method.
        
        Args:
            data: DataFrame with cycle features and RUL labels
            
        Returns:
            DataFrame with statistical features (one row per battery)
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
        print(f"Output shape: {result.shape}")
        
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
        Calculate correlations between statistical features and RUL using the proven method.
        
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
        
        correlations = []
        
        for feature in tqdm(feature_cols, desc="Calculating correlations", unit="feature"):
            try:
                # Get valid data points
                valid_data = data[['log_rul', feature]].dropna()
                
                if len(valid_data) < 2:
                    continue
                
                # Check if feature has any variation
                if valid_data[feature].std() == 0:
                    continue
                
                # Calculate Spearman correlation using the proven method
                from scipy.stats import spearmanr
                correlation, _ = spearmanr(valid_data['log_rul'], valid_data[feature], nan_policy='omit')
                
                if not np.isnan(correlation):
                    correlations.append({
                        'feature': feature,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'n_samples': len(valid_data)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature}: {e}")
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
        
        # Load and prepare data using the working correlation analyzer
        data = self.load_and_prepare_data(dataset_name, cycle_limit)
        
        # Calculate statistical features
        statistical_data = self.calculate_statistical_features(data)
        
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
    parser.add_argument('--cycle_limit', '-c', type=int, default= 100, 
                       help='Maximum number of cycles to use (default: 100)')
    parser.add_argument('--n_features', '-n', type=int, default=15, 
                       help='Number of top features to select (default: 15)')
    parser.add_argument('--test_size', '-t', type=float, default=0.3, 
                       help='Test set size (default: 0.3 for 70/30 split)')
    parser.add_argument('--data_dir', '-d', default='data/preprocessed', 
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
        trainer = StatisticalFeatureTrainerV2(args.data_dir)
        
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
            data = trainer.load_and_prepare_data(args.dataset_name, args.cycle_limit)
            statistical_data = trainer.calculate_statistical_features(data)
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
