# RUL Distribution Analysis for All Chemistries
# This script analyzes and visualizes RUL distributions across different battery chemistries

Write-Host "Starting RUL distribution analysis for all chemistries..."

# Define chemistry directories (adjust paths as needed)
$chemistry_dirs = @(
    "data/datasets_requiring_access/HNEI",  # Mixed NMC-LCO chemistry
    "data/datasets_requiring_access/SNL",   # Multiple chemistries (LFP, NCA, NMC)
    "data/datasets_requiring_access/UL_PUR", # NCA chemistry
    "data/datasets_requiring_access/MATR",  # MATR dataset
    "data/datasets_requiring_access/CALCE", # CALCE dataset
    "data/datasets_requiring_access/HUST",  # HUST dataset
    "data/datasets_requiring_access/RWTH",  # RWTH dataset
    "data/datasets_requiring_access/OX"     # OX dataset
)

# Check which directories exist
$existing_dirs = @()
foreach ($dir in $chemistry_dirs) {
    if (Test-Path $dir) {
        $existing_dirs += $dir
        Write-Host "Found chemistry directory: $dir"
    }
    else {
        Write-Host "Directory not found: $dir" -ForegroundColor Yellow
    }
}

if ($existing_dirs.Count -eq 0) {
    Write-Host "No chemistry directories found. Please check the paths." -ForegroundColor Red
    exit 1
}

Write-Host "Found $($existing_dirs.Count) chemistry directories to analyze"

# Analyze each chemistry individually
foreach ($chem_dir in $existing_dirs) {
    Write-Host "`nAnalyzing RUL distribution for: $chem_dir"
    python -m batteryml.chemistry_data_analysis.rul_distribution --data_path $chem_dir --output_dir chemistry_rul_distributions --verbose
}

# Create combined analysis
Write-Host "`nCreating combined RUL distribution analysis..."

# Run the combined analysis script
python -c "
from batteryml.chemistry_data_analysis.rul_distribution import analyze_all_chemistries
import sys

chemistry_dirs = [d for d in ['$($existing_dirs -join "', '")'] if d]
print(f'Analyzing chemistries: {chemistry_dirs}')

try:
    stats = analyze_all_chemistries(chemistry_dirs, 'chemistry_rul_distributions', verbose=True)
    print('\\nRUL Distribution Analysis Summary:')
    print('=' * 50)
    for chem, chem_stats in stats.items():
        print(f'{chem}:')
        print(f'  Count: {chem_stats[\"count\"]}')
        print(f'  Mean RUL: {chem_stats[\"mean\"]:.1f} cycles')
        print(f'  Std RUL: {chem_stats[\"std\"]:.1f} cycles')
        print(f'  Range: {chem_stats[\"min\"]} - {chem_stats[\"max\"]} cycles')
        print()
except Exception as e:
    print(f'Error in combined analysis: {e}')
    sys.exit(1)
"

Write-Host "`nRUL distribution analysis completed!"
Write-Host "Check the following output directories:"
Write-Host "- chemistry_rul_distributions/ (individual chemistry distributions)"
Write-Host "- chemistry_rul_distributions/all_chemistries_rul_distributions.png (combined histogram plot)"
Write-Host "- chemistry_rul_distributions/all_chemistries_rul_boxplot.png (box plot comparison)"
Write-Host "- chemistry_rul_distributions/*/rul_data.csv (detailed RUL data for each chemistry)"
