param(
  [string]$Base = 'C:\repos\Sokosti_Measurement_Archive\application_testing'
)

$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
$Repo = Split-Path -Parent $Here
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$GapStat = Join-Path $Here 'emg_gap_stat.py'
$Collect = Join-Path $Here 'collect_gap_stats.py'
$Out = Join-Path $Here 'gap_statistics_combined.csv'
$Fs = 980
$Base = [System.IO.Path]::GetFullPath($Base)

if (-not (Test-Path -LiteralPath $Base -PathType Container)) {
  [Console]::Error.WriteLine("Base directory not found: $Base")
  exit 1
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if ($null -eq $PythonCommand) {
    [Console]::Error.WriteLine("Python not found. Expected '$Python' or a python command on PATH.")
    exit 1
  }
  $Python = $PythonCommand.Source
}

$Csvs = @(
  Get-ChildItem -LiteralPath $Base -Recurse -File |
    Where-Object { $_.Name -like '*_PC.csv' -or $_.Name -like '*_SD.csv' } |
    Sort-Object -Property FullName
)

if ($Csvs.Count -eq 0) {
  [Console]::Error.WriteLine("No trial captures (*_PC.csv / *_SD.csv) under $Base")
  exit 1
}

Write-Output "Processing $($Csvs.Count) trial CSVs (fs=$Fs) under $Base"
$Failures = [System.Collections.Generic.List[string]]::new()
$BasePrefix = $Base.TrimEnd('\') + '\'
foreach ($Csv in $Csvs) {
  $RelativePath = $Csv.FullName.Substring($BasePrefix.Length)
  Write-Output "  $RelativePath"
  & $Python $GapStat $Csv.FullName --fs $Fs *> $null
  if ($LASTEXITCODE -ne 0) {
    $Failures.Add($Csv.FullName)
  }
}

if ($Failures.Count -gt 0) {
  [Console]::Error.WriteLine("FAILED ($($Failures.Count)):`n  $($Failures -join "`n  ")")
}

& $Python $Collect --base $Base --out $Out --fs $Fs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $Python (Join-Path $Here 'plot_gap_violin.py') --csv $Out --out-dir (Join-Path $Here 'outputs')
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
