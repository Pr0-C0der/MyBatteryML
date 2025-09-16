#!/usr/bin/env python3
"""
Test script to demonstrate dynamic feature detection in cycle plotting.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from batteryml.data_analysis.cycle_plotter import CyclePlotter


def test_feature_detection():
    """Test dynamic feature detection for different datasets."""
    
    datasets = ['CALCE', 'MATR', 'OX', 'SNL', 'HUST']
    
    for dataset in datasets:
        data_path = f"data/processed/{dataset}"
        print(f"\n{'='*50}")
        print(f"Testing {dataset} dataset")
        print(f"{'='*50}")
        
        if not Path(data_path).exists():
            print(f"Data path does not exist: {data_path}")
            continue
        
        try:
            # Create plotter instance (this will detect features)
            plotter = CyclePlotter(data_path, f"test_{dataset.lower()}_features", cycle_gap=50)
            
            print(f"Detected features: {', '.join(plotter.features)}")
            print(f"Number of features: {len(plotter.features)}")
            
            # Show what plots would be generated
            print("Plots to be generated:")
            for feature in plotter.features:
                print(f"  - {feature}_vs_time/")
            
        except Exception as e:
            print(f"Error testing {dataset}: {e}")


if __name__ == "__main__":
    test_feature_detection()
