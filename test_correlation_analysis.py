#!/usr/bin/env python3
"""
Test script for correlation analysis functionality.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from batteryml.data_analysis.correlation_analyzer import CorrelationAnalyzer


def test_correlation_analysis():
    """Test correlation analysis with CALCE dataset."""
    
    # Test with CALCE dataset
    data_path = "data/processed/CALCE"
    output_dir = "data_analysis_results/CALCE"
    
    print("Testing correlation analysis functionality...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Check if data path exists
    if not Path(data_path).exists():
        print(f"Error: Data path does not exist: {data_path}")
        print("Please make sure the CALCE dataset is processed and available.")
        return False
    
    try:
        # Create analyzer instance
        analyzer = CorrelationAnalyzer(data_path, output_dir)
        
        print(f"Detected features: {', '.join(analyzer.features)}")
        
        # Run correlation analysis
        analyzer.analyze_dataset()
        
        print(f"\nTest completed successfully!")
        print(f"Check the results in: {output_dir}")
        print(f"  - Heatmaps: {output_dir}/heatmaps/")
        print(f"  - Matrices: {output_dir}/matrices/")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_correlation_analysis()
    if success:
        print("\n✓ Correlation analysis test passed!")
    else:
        print("\n✗ Correlation analysis test failed!")
        sys.exit(1)
