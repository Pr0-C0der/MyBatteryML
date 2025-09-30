from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


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


if __name__ == '__main__':
    main()


