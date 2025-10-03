#!/usr/bin/env python3
"""
Test script for RUL distribution analysis
"""

import sys
from pathlib import Path
from batteryml.chemistry_data_analysis.rul_distribution import RULDistributionAnalyzer

def test_single_chemistry():
    """Test RUL distribution analysis for a single chemistry."""
    
    # Test with UL_PUR dataset
    ul_pur_path = "data/datasets_requiring_access/UL_PUR"
    
    if not Path(ul_pur_path).exists():
        print(f"❌ UL_PUR directory not found: {ul_pur_path}")
        return False
    
    print(f"🧪 Testing RUL distribution analysis for UL_PUR...")
    
    try:
        analyzer = RULDistributionAnalyzer(
            data_path=ul_pur_path,
            output_dir="test_rul_distributions",
            verbose=True,
            dataset_hint="UL_PUR"
        )
        
        stats = analyzer.analyze_chemistry()
        
        if stats:
            print(f"✅ UL_PUR analysis successful!")
            print(f"   Found {stats['count']} batteries with valid RUL")
            print(f"   Mean RUL: {stats['mean']:.1f} cycles")
            return True
        else:
            print(f"❌ No valid RUL data found for UL_PUR")
            return False
            
    except Exception as e:
        print(f"❌ Error testing UL_PUR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🧪 Testing RUL Distribution Analysis")
    print("=" * 40)
    
    success = test_single_chemistry()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("You can now run the full analysis with:")
        print("  python analyze_rul_distributions.py")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
