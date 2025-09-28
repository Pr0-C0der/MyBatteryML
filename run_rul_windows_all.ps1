# Runs battery-level RUL windows for all datasets and aggregates results.

$ErrorActionPreference = 'Stop'

# Datasets to run (include MATR1/MATR2/CLO variants)
$datasets = @('CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100', 'MATR1', 'MATR2', 'CLO', 'CALCE')
$cycleLimit = 100
$outputRoot = 'rul_windows'
$aggregatePath = Join-Path $outputRoot "aggregate_metrics_bl_cl$cycleLimit.csv"

if (!(Test-Path $outputRoot)) { New-Item -ItemType Directory -Path $outputRoot | Out-Null }
if (Test-Path $aggregatePath) { Remove-Item $aggregatePath -Force }

function Get-DataPathsForDataset($ds) {
    switch ($ds.ToUpper()) {
        'MATR' { return @('data/preprocessed/MATR') }
        'MATR1' { return @('data/preprocessed/MATR') }
        'MATR2' { return @('data/preprocessed/MATR') }
        'CLO' { return @('data/preprocessed/MATR') }
        'CALCE' { return @('data/preprocessed/CALCE') }
        'HUST' { return @('data/preprocessed/HUST') }
        'SNL' { return @('data/preprocessed/SNL') }
        'CRUH' { return @('data/preprocessed/CALCE', 'data/preprocessed/RWTH', 'data/preprocessed/UL_PUR', 'data/preprocessed/HNEI') }
        'CRUSH' { return @('data/preprocessed/CALCE', 'data/preprocessed/RWTH', 'data/preprocessed/UL_PUR', 'data/preprocessed/HNEI', 'data/preprocessed/SNL') }
        'MIX100' { return @('data/preprocessed/HUST', 'data/preprocessed/MATR', 'data/preprocessed/RWTH', 'data/preprocessed/CALCE', 'data/preprocessed/UL_PUR', 'data/preprocessed/HNEI') }
        default { throw "Unknown dataset $ds" }
    }
}

$first = $true
foreach ($ds in $datasets) {
    try {
        $paths = Get-DataPathsForDataset $ds
        $dsOutDir = Join-Path $outputRoot $ds
        if (!(Test-Path $dsOutDir)) { New-Item -ItemType Directory -Path $dsOutDir | Out-Null }

        $dataArgs = ($paths -join ' ')
        $cmd = "python -m batteryml.training.train_rul_windows --dataset $ds --data_path $dataArgs --output_dir $dsOutDir --battery_level --cycle_limit $cycleLimit --features default --gpu --verbose --report_missing"
        Write-Host "Running: $cmd"
        & $cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Command failed for dataset $ds (exit $LASTEXITCODE). Skipping aggregation."
            continue
        }

        $resultFile = Join-Path $dsOutDir "rul_metrics_bl_cl$cycleLimit.csv"
        if (!(Test-Path $resultFile)) {
            Write-Warning "Result file not found for $ds at $resultFile. Skipping."
            continue
        }

        $rows = Import-Csv $resultFile
        foreach ($r in $rows) { $r | Add-Member -NotePropertyName dataset -NotePropertyValue $ds }

        if ($first) {
            $rows | Export-Csv -Path $aggregatePath -NoTypeInformation
            $first = $false
        }
        else {
            $rows | Export-Csv -Path $aggregatePath -NoTypeInformation -Append
        }
        Write-Host "Aggregated: $ds -> $aggregatePath"
    }
    catch {
        Write-Warning "Error for dataset ${ds}: $($_.Exception.Message)"
    }
}

Write-Host "Done. Aggregate CSV: $aggregatePath"


