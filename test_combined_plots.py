#!/usr/bin/env python3
"""
Test script for combined plots functionality.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.combined_plots import CombinedPlotGenerator
from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer


def test_combined_plots():
    """Test combined plots functionality."""
    print("Testing Combined Plots Functionality")
    print("=" * 40)
    
    # Check if data exists
    data_dir = Path("data/processed/MATR")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure you have processed MATR data first.")
        return False
    
    try:
        # Create analyzer
        analyzer = BaseDataAnalyzer(str(data_dir), "data_analysis_results/MATR")
        
        # Get battery files
        battery_files = analyzer.get_battery_files()
        if not battery_files:
            print("No battery files found")
            return False
        
        print(f"Found {len(battery_files)} battery files")
        
        # Create combined plot generator
        plot_generator = CombinedPlotGenerator(Path("data_analysis_results/MATR"), num_batteries=5)
        
        # Test with first 5 batteries
        test_files = battery_files[:5]
        print(f"Testing with {len(test_files)} batteries")
        
        # Generate combined plots
        plot_generator.generate_combined_plots(test_files, analyzer)
        
        print("✓ Combined plots generated successfully!")
        print(f"Check results in: data_analysis_results/MATR/combined_plots/")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run combined plots test."""
    success = test_combined_plots()
    
    if success:
        print("\n" + "=" * 40)
        print("Combined plots test completed successfully!")
    else:
        print("\n" + "=" * 40)
        print("Combined plots test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
