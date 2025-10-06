#!/usr/bin/env python3
"""
Combined Chemistry Correlation Analysis

This script creates combined correlation boxplots for multiple chemistries,
showing how different features correlate with RUL across all chemistry types.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from batteryml.chemistry_data_analysis.correlation_mod import plot_combined_chemistry_correlations
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create combined correlation boxplots for multiple chemistries')
    parser.add_argument('--chemistry_dirs', type=str, nargs='+', required=True,
                       help='List of chemistry directory paths (e.g., data_chemistries/lfp data_chemistries/nmc)')
    parser.add_argument('--output_dir', type=str, default='combined_chemistry_correlations',
                       help='Output directory for combined plots')
    parser.add_argument('--dataset_hint', type=str, default=None,
                       help='Optional dataset name hint to override auto detection')
    parser.add_argument('--cycle_limit', type=int, default=None,
                       help='Limit analysis to first N cycles (None for all cycles)')
    parser.add_argument('--smoothing', type=str, default='none', 
                       choices=['none', 'ma', 'median', 'hms'],
                       help='Smoothing method for feature data')
    parser.add_argument('--ma_window', type=int, default=5,
                       help='Window size for moving average/median smoothing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    print("Combined Chemistry Correlation Analysis")
    print("=====================================")
    print(f"Chemistry directories: {args.chemistry_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cycle limit: {args.cycle_limit if args.cycle_limit else 'All cycles'}")
    print(f"Smoothing: {args.smoothing}")
    print()
    
    # Create combined correlation plots
    plot_combined_chemistry_correlations(
        chemistry_dirs=args.chemistry_dirs,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dataset_hint=args.dataset_hint,
        cycle_limit=args.cycle_limit,
        smoothing=args.smoothing,
        ma_window=args.ma_window
    )
    
    print("\nCombined correlation analysis completed!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
