#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_data_analysis.sh [BASE_DATA_PATH] [OUTPUT_DIR]
# Defaults:
#   BASE_DATA_PATH = data/processed
#   OUTPUT_DIR     = data_analysis_results

BASE_DATA_PATH="${1:-data/processed}"
OUTPUT_DIR="${2:-data_analysis_results}"

# Prefer python3 if available
PYTHON_BIN="python"
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "[1/4] Running per-dataset analyses into $OUTPUT_DIR ..."
"$PYTHON_BIN" batteryml/data_analysis/run_analysis.py --all --data_path "$BASE_DATA_PATH" --output_dir "$OUTPUT_DIR"

echo "[2/4] Generating combined plots per dataset ..."
DATASETS=(CALCE HUST MATR SNL HNEI RWTH UL_PUR OX)
for ds in "${DATASETS[@]}"; do
  if [ -d "$BASE_DATA_PATH/$ds" ]; then
    echo "  - $ds"
    "$PYTHON_BIN" batteryml/data_analysis/run_analysis.py --combined-plots "$ds" --data_path "$BASE_DATA_PATH/$ds" --output_dir "$OUTPUT_DIR/$ds"
  else
    echo "  - Skipping $ds (no data at $BASE_DATA_PATH/$ds)"
  fi
done

echo "[3/4] Running correlation analysis for all datasets ..."
"$PYTHON_BIN" batteryml/data_analysis/run_correlation_analysis.py --all --data_path "$BASE_DATA_PATH" --output_dir "$OUTPUT_DIR"

echo "[4/4] Generating cycle plots for all datasets ..."
"$PYTHON_BIN" batteryml/data_analysis/run_cycle_plots.py --all --data_path "$BASE_DATA_PATH" --output_dir "$OUTPUT_DIR"

echo "\nAll analyses complete. Results saved under: $OUTPUT_DIR"

