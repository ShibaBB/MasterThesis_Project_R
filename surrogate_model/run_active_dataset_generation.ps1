param(
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$SurrogateRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $SurrogateRoot
$VenvPython = Join-Path $SurrogateRoot 'segmented_symbolic_regression\.venv_py311\Scripts\python.exe'
$PythonExe = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { 'python' }
$ActiveConfig = Get-Content -Raw -LiteralPath (Join-Path $SurrogateRoot 'dataset_run_config.json') | ConvertFrom-Json
$RunId = [string]$ActiveConfig.default_dataset_run
$RunDir = Join-Path $SurrogateRoot (Join-Path 'datasets' $RunId)
$RunConfig = Get-Content -Raw -LiteralPath (Join-Path $RunDir 'dataset_config.json') | ConvertFrom-Json
$Material = [string]$RunConfig.generation.material
$TeacherFile = Join-Path $RunDir (Join-Path 'MLP' ($Material + '_surrogate_dataset.mat'))
$SplitFile = Join-Path $RunDir 'shared_curve_split.json'
$SegmentedFile = Join-Path $RunDir (Join-Path 'segmented_SR' ($Material + '_symbolic_segmented.mat'))
$GlobalFile = Join-Path $RunDir (Join-Path 'global_SR' ($Material + '_symbolic_global.mat'))

Push-Location $ProjectRoot
try {
    & $PythonExe (Join-Path $SurrogateRoot 'sync_dataset_manifest.py') --check
    if ($LASTEXITCODE -ne 0) { throw 'Dataset-run configuration validation failed.' }
    if ($ValidateOnly) { return }

    if (Test-Path -LiteralPath $TeacherFile) {
        Write-Output "Skipping existing teacher dataset: $TeacherFile"
    }
    else {
        matlab -batch "run('surrogate_model/data_generation/run_teacher_dataset_generation.m')"
        if ($LASTEXITCODE -ne 0) { throw 'Teacher dataset generation failed.' }
    }

    if (Test-Path -LiteralPath $SplitFile) {
        Write-Output "Skipping existing shared split: $SplitFile"
    }
    else {
        & $PythonExe (Join-Path $SurrogateRoot 'generate_shared_curve_split.py')
        if ($LASTEXITCODE -ne 0) { throw 'Shared curve split generation failed.' }
    }

    if (Test-Path -LiteralPath $SegmentedFile) {
        Write-Output "Skipping existing segmented SR dataset: $SegmentedFile"
    }
    else {
        matlab -batch "run('surrogate_model/segmented_symbolic_regression/run_full_segmented_symbolic_dataset_generation.m')"
        if ($LASTEXITCODE -ne 0) { throw 'Segmented SR dataset generation failed.' }
    }

    if (Test-Path -LiteralPath $GlobalFile) {
        Write-Output "Skipping existing global SR dataset: $GlobalFile"
    }
    else {
        matlab -batch "run('surrogate_model/global_symbolic_regression/run_global_symbolic_dataset_generation.m')"
        if ($LASTEXITCODE -ne 0) { throw 'Global SR dataset generation failed.' }
    }

    & $PythonExe (Join-Path $SurrogateRoot 'sync_dataset_manifest.py')
    if ($LASTEXITCODE -ne 0) { throw 'Dataset manifest synchronization failed.' }
}
finally {
    Pop-Location
}
