from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def format_rmse(df: pd.DataFrame, threshold: float = 1000.0) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        # Skip non-numeric columns if any slipped in
        def _fmt(v):
            try:
                f = float(v)
                if not np.isfinite(f):
                    return ''
                return '> 1000' if f > threshold else f
            except Exception:
                return v
        out[col] = out[col].map(_fmt)
    return out


def main():
    p = argparse.ArgumentParser(description='Format and display RMSE table (values > threshold shown as "> 1000")')
    p.add_argument('--rmse_csv', type=str, required=True, help='Path to RMSE.csv produced by train_rul_windows (global aggregate)')
    p.add_argument('--out_csv', type=str, default=None, help='Optional path to save the formatted table (CSV)')
    p.add_argument('--threshold', type=float, default=1000.0, help='Threshold above which cells are shown as "> 1000"')
    args = p.parse_args()

    rmse_path = Path(args.rmse_csv)
    if not rmse_path.exists():
        print(f"RMSE file not found: {rmse_path}")
        sys.exit(1)

    # Expect rows=models, columns=datasets
    df = pd.read_csv(rmse_path, index_col=0)
    # Normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]
    # Reorder columns as requested; append any extras at the end
    desired = ['MATR1', 'MATR2', 'HUST', 'SNL', 'CLO', 'CRUH', 'CRUSH', 'MIX100']
    ordered_cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    if ordered_cols:
        df = df.reindex(columns=ordered_cols)
    df_fmt = format_rmse(df, threshold=args.threshold)

    # Print to stdout as a simple table
    try:
        # Align columns visually
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_fmt.to_string())
    except Exception:
        print(df_fmt)

    # Save formatted CSV if requested; else save next to original
    out_csv = Path(args.out_csv) if args.out_csv else rmse_path.with_name('RMSE_display.csv')
    try:
        df_fmt.to_csv(out_csv)
        print(f"Saved formatted table to {out_csv}")
    except Exception:
        pass

    # Also save a PNG with bold minimum values per column
    try:
        # Build cell text
        col_labels = list(df_fmt.columns)
        row_labels = list(df_fmt.index)
        # Build a 3-decimal formatted view for the PNG (keeping '> 1000' strings)
        def _fmt3(v):
            try:
                f = float(v)
                if not np.isfinite(f):
                    return ''
                return f"> {int(args.threshold)}" if f > args.threshold else f"{f:.3f}"
            except Exception:
                return str(v)
        cell_text = [[_fmt3(df_fmt.loc[row, col]) for col in col_labels] for row in row_labels]

        # Determine minima per column using numeric df aligned to display columns
        numeric_df = df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.reindex(columns=col_labels)
        min_rows_per_col = {}
        for j, col in enumerate(col_labels):
            col_vals = numeric_df[col]
            if col_vals.dropna().empty:
                min_rows_per_col[j] = set()
                continue
            min_val = col_vals.min()
            min_idxs = set(col_vals.index[col_vals == min_val])
            min_rows_per_col[j] = min_idxs

        nrows, ncols = len(row_labels), len(col_labels)
        fig_w = max(8, ncols * 2)
        fig_h = max(2, nrows * 0.5 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')
        the_table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.2)

        # Highlight minima: data cells start at (row+1, col+1)
        row_pos = {idx: i for i, idx in enumerate(row_labels)}
        for j in range(ncols):
            for idx in min_rows_per_col.get(j, set()):
                if idx in row_pos:
                    r = row_pos[idx] + 1
                    c = j + 1
                    try:
                        cell = the_table[(r, c)]
                        cell.get_text().set_weight('bold')
                        cell.set_facecolor('#d4edda')  # light green
                    except Exception:
                        pass

        out_png = rmse_path.with_name('RMSE_display.png')
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved formatted PNG to {out_png}")
    except Exception as e:
        print(f"[warn] failed to save PNG: {e}")


if __name__ == '__main__':
    main()


