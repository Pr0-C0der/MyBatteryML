# Statistical Feature Training Experiments Script
# Runs multiple experiments with different parameters

Write-Host "Statistical Feature Training Experiments" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Experiment configurations
$Experiments = @(
    @{
        Name = "MATR_50cycles_10features"
        Dataset = "MATR"
        CycleLimit = 50
        NFeatures = 10
        Description = "MATR dataset, 50 cycles, 10 features"
    },
    @{
        Name = "MATR_100cycles_15features"
        Dataset = "MATR"
        CycleLimit = 100
        NFeatures = 15
        Description = "MATR dataset, 100 cycles, 15 features"
    },
    @{
        Name = "CALCE_50cycles_10features"
        Dataset = "CALCE"
        CycleLimit = 50
        NFeatures = 10
        Description = "CALCE dataset, 50 cycles, 10 features"
    },
    @{
        Name = "CALCE_100cycles_15features"
        Dataset = "CALCE"
        CycleLimit = 100
        NFeatures = 15
        Description = "CALCE dataset, 100 cycles, 15 features"
    },
    @{
        Name = "UL_PUR_50cycles_10features"
        Dataset = "UL_PUR"
        CycleLimit = 50
        NFeatures = 10
        Description = "UL_PUR dataset, 50 cycles, 10 features"
    }
)

# Common parameters
$DataDir = "data/preprocessed"
$BaseOutputDir = "statistical_experiments"
$TestSize = 0.3
$RandomState = 42

Write-Host "Running $($Experiments.Count) experiments..." -ForegroundColor Cyan
Write-Host ""

$Results = @()
$SuccessCount = 0

foreach ($Exp in $Experiments) {
    Write-Host "=========================================" -ForegroundColor Yellow
    Write-Host "Experiment: $($Exp.Name)" -ForegroundColor Yellow
    Write-Host "Description: $($Exp.Description)" -ForegroundColor Yellow
    Write-Host "=========================================" -ForegroundColor Yellow
    
    # Create experiment-specific output directory
    $OutputDir = Join-Path $BaseOutputDir $Exp.Name
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }
    
    # Check if dataset exists
    $DatasetPath = Join-Path $DataDir $Exp.Dataset
    if (-not (Test-Path $DatasetPath)) {
        Write-Host "⚠️  Dataset not found: $DatasetPath" -ForegroundColor Yellow
        Write-Host "Skipping experiment: $($Exp.Name)" -ForegroundColor Yellow
        $Results += @{
            Name = $Exp.Name
            Success = $false
            Error = "Dataset not found"
        }
        continue
    }
    
    # Run the experiment
    $Command = "python batteryml/chemistry_data_analysis/statistical_analysis/statistical_feature_training_v2.py"
    $Command += " $($Exp.Dataset)"
    $Command += " --cycle_limit $($Exp.CycleLimit)"
    $Command += " --n_features $($Exp.NFeatures)"
    $Command += " --test_size $TestSize"
    $Command += " --data_dir `"$DataDir`""
    $Command += " --output_dir `"$OutputDir`""
    $Command += " --random_state $RandomState"
    
    Write-Host "Command: $Command" -ForegroundColor Cyan
    Write-Host ""
    
    $StartTime = Get-Date
    try {
        Invoke-Expression $Command
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Experiment completed successfully!" -ForegroundColor Green
            Write-Host "Duration: $($Duration.TotalMinutes.ToString('F2')) minutes" -ForegroundColor Green
            $Results += @{
                Name = $Exp.Name
                Success = $true
                Duration = $Duration
                Error = $null
            }
            $SuccessCount++
        } else {
            Write-Host "❌ Experiment failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
            $Results += @{
                Name = $Exp.Name
                Success = $false
                Duration = $Duration
                Error = "Exit code: $LASTEXITCODE"
            }
        }
    } catch {
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        $Results += @{
            Name = $Exp.Name
            Success = $false
            Duration = $Duration
            Error = $_.Exception.Message
        }
    }
    
    Write-Host ""
}

# Summary
Write-Host "=========================================" -ForegroundColor Green
Write-Host "EXPERIMENTS SUMMARY" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Total experiments: $($Experiments.Count)" -ForegroundColor Cyan
Write-Host "Successful: $SuccessCount" -ForegroundColor Green
Write-Host "Failed: $($Experiments.Count - $SuccessCount)" -ForegroundColor Red
Write-Host ""

Write-Host "Detailed Results:" -ForegroundColor Cyan
foreach ($Result in $Results) {
    $Status = if ($Result.Success) { "✅ Success" } else { "❌ Failed" }
    $Color = if ($Result.Success) { "Green" } else { "Red" }
    $DurationStr = if ($Result.Duration) { " ($($Result.Duration.TotalMinutes.ToString('F2')) min)" } else { "" }
    $ErrorStr = if ($Result.Error) { " - $($Result.Error)" } else { "" }
    Write-Host "  $($Result.Name): $Status$DurationStr$ErrorStr" -ForegroundColor $Color
}

Write-Host "`nResults saved in: $BaseOutputDir" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Green
