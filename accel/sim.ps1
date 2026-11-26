# Run a testbench with ModelSim (Intel FPGA Starter/ASE).
# Usage: .\sim.ps1 [tb_name ...]   (default: every testbench in tb/)
# Exit code is 0 only if every testbench prints PASS.
param([string[]]$Tb)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ms = 'C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem'
$work = Join-Path $root 'sim_work'

if (-not $Tb) { $Tb = Get-ChildItem (Join-Path $root 'tb') -Filter 'tb_*.sv' | ForEach-Object { $_.BaseName } }

Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
& "$ms\vlib.exe" $work | Out-Null

$rtl = @('fpgate_pkg.sv') + (Get-ChildItem (Join-Path $root 'rtl') -Filter *.sv | Where-Object Name -ne 'fpgate_pkg.sv' | ForEach-Object Name)
$files = $rtl | ForEach-Object { Join-Path $root "rtl\$_" }
$files += Get-ChildItem (Join-Path $root 'tb') -Filter *.sv | Where-Object { $_.Name -notlike 'tb_*' } | ForEach-Object FullName   # helper models first
$files += $Tb | ForEach-Object { Join-Path $root "tb\$_.sv" }
& "$ms\vlog.exe" -sv -work $work @files
if ($LASTEXITCODE -ne 0) { exit 1 }

$fail = 0
Push-Location $root   # testbenches read vectors/ relative to accel/
foreach ($t in $Tb) {
  $out = & "$ms\vsim.exe" -c -work $work -do 'run -all; quit -f' $t
  $out | Where-Object { $_ -match '^# (PASS|FAIL)' -or $_ -match 'Error' } | ForEach-Object { Write-Host $_ }
  if (-not ($out | Where-Object { $_ -match '^# PASS' }) -or ($out | Where-Object { $_ -match '^# FAIL' })) { $fail = 1 }
}
exit $fail
