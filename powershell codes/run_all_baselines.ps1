# Run all baselines and log results (works on Windows PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_all_rul_baseline.ps1

$ConfigRoot = "configs"
$WorkspaceRoot = "workspaces_new"

# Helper: build workspace path under workspaces_new\<relative-config-without-ext>
function Get-WorkspacePath([string]$configPath) {
  $absConfig = (Resolve-Path $configPath).Path
  $absRoot = (Resolve-Path $ConfigRoot).Path
  $rel = $absConfig.Replace("$absRoot\", "")
  $relNoExt = [System.IO.Path]::ChangeExtension($rel, $null)
  return (Join-Path $WorkspaceRoot $relNoExt)
}

# sklearn baselines (single seed)
Get-ChildItem -Recurse "configs\baselines\sklearn" -Filter *.yaml | ForEach-Object {
  $config = $_.FullName
  $workspace = Get-WorkspacePath $config
  New-Item -ItemType Directory -Path $workspace -Force | Out-Null
  $seed = 0
  python batteryml.py run $config --workspace $workspace --train --eval --skip_if_executed false --seed $seed `
  | Tee-Object -FilePath (Join-Path $workspace "log.$seed")
}

# nn_models baselines (single seed, like sklearn)
Get-ChildItem -Recurse "configs\baselines\nn_models" -Filter *.yaml | ForEach-Object {
  $config = $_.FullName
  $workspace = Get-WorkspacePath $config
  New-Item -ItemType Directory -Path $workspace -Force | Out-Null
  $seed = 0
  python batteryml.py run $config --workspace $workspace --train --eval --seed $seed --device cuda --skip_if_executed false `
  | Tee-Object -FilePath (Join-Path $workspace "log.$seed")
}

Write-Host "Done. Global CSVs are in .\results\metrics_RMSE.csv, metrics_MAE.csv, metrics_MAPE.csv"