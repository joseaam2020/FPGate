# Compile the RTL and run one or more testbenches in a private work library, so it can run while sim.ps1
# is busy (sim.ps1 deletes sim_work). Prints the lines a testbench marks as PASS / FAIL / results.
# Usage: .\tools\sim_tb.ps1 tb_raw32 [tb_other ...]   (run from anywhere; the simulation starts in accel/)
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Tb)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ms = 'C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem'
$work = Join-Path $root 'sim_work_tb'

Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
& "$ms\vlib.exe" $work | Out-Null
$rtl = @('fpgate_pkg.sv') + (Get-ChildItem (Join-Path $root 'rtl') -Filter *.sv | Where-Object Name -ne 'fpgate_pkg.sv' | ForEach-Object Name)
$files = $rtl | ForEach-Object { Join-Path $root "rtl\$_" }
$files += Get-ChildItem (Join-Path $root 'tb') -Filter *.sv | Where-Object { $_.Name -notlike 'tb_*' } | ForEach-Object FullName
$files += $Tb | ForEach-Object { Join-Path $root "tb\$_.sv" }
& "$ms\vlog.exe" -sv -work $work @files
if ($LASTEXITCODE -ne 0) { Write-Host 'compile failed'; exit 1 }

$fail = 0
Push-Location $root
foreach ($t in $Tb) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $out = & "$ms\vsim.exe" -c -work $work -do 'run -all; quit -f' $t
  $out | Where-Object { $_ -match '^# ' -and $_ -notmatch '^# (//|Loading|vsim|Start time|End time|Errors:|Warnings:|run -all|quit)' -and $_ -notmatch '^# \*\* Note' } | ForEach-Object { Write-Host $_ }
  if (-not ($out | Where-Object { $_ -match '^# PASS' }) -or ($out | Where-Object { $_ -match '^# FAIL' })) { $fail = 1 }
  Write-Host ("[{0}] {1:N0} s" -f $t, $sw.Elapsed.TotalSeconds)
}
Pop-Location
exit $fail
