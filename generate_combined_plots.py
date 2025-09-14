#!/usr/bin/env python3
"""
Standalone script to generate combined plots for battery datasets.
This script generates combined plots for randomly selected batteries.
"""

import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.combined_plots import CombinedPlotGenerator
from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer


def main():
    """Main function for generating combined plots."""
    parser = argparse.ArgumentParser(
        description='Generate combined plots for battery datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate combined plots for MATR dataset
  python generate_combined_plots.py --data_path data/processed/MATR
  
  # Generate combined plots with custom number of batteries
  python generate_combined_plots.py --data_path data/processed/CALCE --num_batteries 30
  
  # Generate combined plots with custom output directory
  python generate_combined_plots.py --data_path data/processed/HUST --output_dir my_combined_plots
        """
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='combined_analysis',
                       help='Output directory for combined plots')
    parser.add_argument('--num_batteries', type=int, default=20,
                       help='Number of random batteries to select (default: 20)')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    print("Battery Data Combined Plots Generator")
    print("=" * 50)
    print(f"Data path: {data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of batteries: {args.num_batteries}")
    print()
    
    try:
        # Create analyzer
        analyzer = BaseDataAnalyzer(str(data_path), args.output_dir)
        
        # Get battery files
        battery_files = analyzer.get_battery_files()
        if not battery_files:
            print(f"No battery files found in {data_path}")
            sys.exit(1)
        
        print(f"Found {len(battery_files)} battery files")
        
        # Create combined plot generator
        plot_generator = CombinedPlotGenerator(Path(args.output_dir), args.num_batteries)
        
        # Generate combined plots
        plot_generator.generate_combined_plots(battery_files, analyzer)
        
        print("\n" + "=" * 50)
        print("Combined plots generated successfully!")
        print(f"Check the results in: {args.output_dir}/combined_plots/")
        print("\nGenerated plots:")
        print("  - combined_capacity_fade.png")
        print("  - combined_voltage_capacity.png")
        print("  - combined_qc_qd.png")
        print("  - combined_current_time.png")
        print("  - combined_voltage_time.png")
        print("  - capacity_distribution.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
