#!/usr/bin/env python3
"""
Test script to check data loading logic
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from batteryml.chemistry_data_analysis.statistical_analysis.statistical_feature_training import StatisticalFeatureTrainer

def test_data_loading():
    """Test the data loading functionality"""
    print("Testing data loading logic...")
    
    try:
        # Initialize trainer
        trainer = StatisticalFeatureTrainer("data")
        
        # Test loading a small dataset
        print("\n=== Testing MATR dataset ===")
        data = trainer.load_battery_data("MATR", cycle_limit=5)
        
        print(f"\nData loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Batteries: {data['battery_id'].nunique()}")
        print(f"Cycles per battery: {data.groupby('battery_id').size().describe()}")
        
        # Check for NaN values
        print(f"\nNaN values per column:")
        for col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} ({nan_count/len(data)*100:.1f}%)")
        
        # Test RUL calculation
        print(f"\n=== Testing RUL calculation ===")
        data_with_rul = trainer.calculate_rul_labels(data)
        print(f"RUL calculated successfully!")
        print(f"log_rul range: {data_with_rul['log_rul'].min():.3f} to {data_with_rul['log_rul'].max():.3f}")
        
        # Test statistical feature calculation
        print(f"\n=== Testing statistical features ===")
        statistical_data = trainer.calculate_statistical_features(data_with_rul)
        print(f"Statistical features calculated successfully!")
        print(f"Shape: {statistical_data.shape}")
        print(f"Features with all NaN: {(statistical_data.isna().all()).sum()}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!")
