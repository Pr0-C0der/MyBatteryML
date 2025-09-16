#!/usr/bin/env python3
"""
Test script for cycle plotting functionality.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from batteryml.data_analysis.cycle_plotter import CyclePlotter


def test_cycle_plotting():
    """Test cycle plotting with CALCE dataset."""
    
    # Test with CALCE dataset
    data_path = "data/processed/CALCE"
    output_dir = "test_cycle_plots"
    cycle_gap = 50  # Plot every 50th cycle
    
    print("Testing cycle plotting functionality...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cycle gap: {cycle_gap}")
    print("-" * 50)
    
    # Check if data path exists
    if not Path(data_path).exists():
        print(f"Error: Data path does not exist: {data_path}")
        print("Please make sure the CALCE dataset is processed and available.")
        return False
    
    try:
        # Create plotter instance
        plotter = CyclePlotter(data_path, output_dir, cycle_gap)
        
        # Run cycle plotting
        plotter.plot_dataset_features()
        
        print(f"\nTest completed successfully!")
        print(f"Check the results in: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cycle_plotting()
    if success:
        print("\n✓ Cycle plotting test passed!")
    else:
        print("\n✗ Cycle plotting test failed!")
        sys.exit(1)
