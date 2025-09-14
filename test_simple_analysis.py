#!/usr/bin/env python3
"""
Simple test script for the data analysis module.
This tests the core functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without heavy imports."""
    print("Testing basic data analysis functionality...")
    
    try:
        # Test that we can import the modules
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test utils module
        from batteryml.data_analysis.utils import AnalysisUtils
        print("✓ AnalysisUtils imported successfully")
        
        # Test safe_divide function
        result = AnalysisUtils.safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        
        result = AnalysisUtils.safe_divide(10, 0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        
        print("✓ AnalysisUtils.safe_divide working correctly")
        
        # Test file path utilities
        test_path = Path("test_directory")
        files = AnalysisUtils.get_battery_files(str(test_path))
        assert files == [], f"Expected empty list, got {files}"
        
        print("✓ AnalysisUtils.get_battery_files working correctly")
        
        print("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_analysis_script():
    """Test that the analysis script can be imported."""
    print("Testing analysis script import...")
    
    try:
        # Test that the main analysis script exists and is readable
        script_path = Path("batteryml/data_analysis/analyze_datasets.py")
        assert script_path.exists(), f"Analysis script not found at {script_path}"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert "def main():" in content, "Main function not found in script"
            assert "DatasetAnalyzer" in content, "DatasetAnalyzer not found in script"
        
        print("✓ Analysis script is valid")
        return True
        
    except Exception as e:
        print(f"✗ Analysis script test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Simple BatteryML Data Analysis Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_analysis_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic tests passed! The analysis module structure is correct.")
        print("\nTo run the full analysis, you'll need to install the required dependencies:")
        print("pip install numpy pandas matplotlib seaborn tqdm")
        print("\nThen you can run:")
        print("python batteryml/data_analysis/analyze_datasets.py --help")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
