#!/usr/bin/env python3
"""
Script to read RMSE.csv and create a PNG table with lowest values highlighted in green.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import argparse
from pathlib import Path


def create_rmse_table_png(csv_path: str, output_path: str = None, figsize: tuple = (12, 8)):
    """
    Read RMSE.csv and create a PNG table with lowest values highlighted in green.
    
    Args:
        csv_path: Path to the RMSE.csv file
        output_path: Output path for PNG file (default: same as CSV with .png extension)
        figsize: Figure size for the plot
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty at {csv_path}")
        return
    
    if df.empty:
        print("Error: No data in CSV file")
        return
    
    # Set output path if not provided
    if output_path is None:
        output_path = str(Path(csv_path).with_suffix('.png'))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = df.round(2).astype(str)
    
    # Find minimum values in each column (excluding NaN)
    min_values = {}
    for col in df.columns:
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        if not numeric_values.isna().all():
            min_val = numeric_values.min()
            min_values[col] = min_val
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    rowLabels=table_data.index,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header row
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')  # Green header
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the row labels
    for i in range(len(table_data.index)):
        table[(i + 1, -1)].set_facecolor('#E8F5E8')  # Light green for row labels
        table[(i + 1, -1)].set_text_props(weight='bold')
    
    # Highlight minimum values in each column with green background
    for col_idx, col_name in enumerate(table_data.columns):
        if col_name in min_values:
            min_val = min_values[col_name]
            for row_idx, row_name in enumerate(table_data.index):
                cell_value = pd.to_numeric(df.loc[row_name, col_name], errors='coerce')
                if not pd.isna(cell_value) and abs(cell_value - min_val) < 1e-6:  # Check if it's the minimum
                    table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')  # Light green
                    table[(row_idx + 1, col_idx)].set_text_props(weight='bold')
    
    # Add title
    plt.title('RMSE Results by Model and Dataset\n(Lowest values highlighted in green)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"RMSE table saved to: {output_path}")
    print(f"Table dimensions: {len(table_data.index)} models × {len(table_data.columns)} datasets")
    
    # Print summary of minimum values
    print("\nMinimum RMSE values by dataset:")
    for col, min_val in min_values.items():
        print(f"  {col}: {min_val:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Create PNG table from RMSE.csv with highlighted minimum values')
    parser.add_argument('csv_path', help='Path to RMSE.csv file')
    parser.add_argument('--output', '-o', help='Output PNG file path (default: same as CSV with .png extension)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Reading CSV from: {args.csv_path}")
        if args.output:
            print(f"Output will be saved to: {args.output}")
    
    create_rmse_table_png(args.csv_path, args.output, tuple(args.figsize))


if __name__ == '__main__':
    main()
