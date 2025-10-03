#!/usr/bin/env python3
"""
RUL Distribution Analysis Script
Analyzes and visualizes RUL distributions across different battery chemistries.
"""

import sys
from pathlib import Path
from batteryml.chemistry_data_analysis.rul_distribution import analyze_all_chemistries

def main():
    """Main function to analyze RUL distributions for all available chemistries."""
    
    # Define chemistry directories
    base_path = Path("data/datasets_requiring_access")
    chemistry_dirs = [
        str(base_path / "HNEI"),    # Mixed NMC-LCO chemistry
        str(base_path / "SNL"),     # Multiple chemistries (LFP, NCA, NMC)
        str(base_path / "UL_PUR"),  # NCA chemistry
        str(base_path / "MATR"),    # MATR dataset
        str(base_path / "CALCE"),   # CALCE dataset
        str(base_path / "HUST"),    # HUST dataset
        str(base_path / "RWTH"),    # RWTH dataset
        str(base_path / "OX")       # OX dataset
    ]
    
    # Check which directories exist
    existing_dirs = []
    for chem_dir in chemistry_dirs:
        if Path(chem_dir).exists():
            existing_dirs.append(chem_dir)
            print(f"Found chemistry directory: {chem_dir}")
        else:
            print(f"Directory not found: {chem_dir}")
    
    if not existing_dirs:
        print("No chemistry directories found. Please check the paths.")
        return 1
    
    print(f"\nAnalyzing {len(existing_dirs)} chemistry directories...")
    
    try:
        # Analyze all chemistries
        stats = analyze_all_chemistries(existing_dirs, 'chemistry_rul_distributions', verbose=True)
        
        # Print summary
        print("\n" + "="*60)
        print("RUL DISTRIBUTION ANALYSIS SUMMARY")
        print("="*60)
        
        for chem, chem_stats in stats.items():
            print(f"\n{chem.upper()}:")
            print(f"  Count: {chem_stats['count']} batteries")
            print(f"  Mean RUL: {chem_stats['mean']:.1f} cycles")
            print(f"  Std RUL: {chem_stats['std']:.1f} cycles")
            print(f"  Median RUL: {chem_stats['median']:.1f} cycles")
            print(f"  Range: {chem_stats['min']} - {chem_stats['max']} cycles")
            print(f"  Q25-Q75: {chem_stats['q25']:.1f} - {chem_stats['q75']:.1f} cycles")
        
        print("\n" + "="*60)
        print("OUTPUT FILES GENERATED:")
        print("="*60)
        print("📊 Individual chemistry distributions:")
        for chem in stats.keys():
            print(f"   - chemistry_rul_distributions/{chem}/{chem}_rul_distribution.png")
            print(f"   - chemistry_rul_distributions/{chem}/{chem}_rul_data.csv")
        
        print("\n📊 Combined analysis:")
        print("   - chemistry_rul_distributions/all_chemistries_rul_distributions.png")
        print("   - chemistry_rul_distributions/all_chemistries_rul_boxplot.png")
        
        print(f"\n✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
