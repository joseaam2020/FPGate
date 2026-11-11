# Cycle counts from the simulator with the dimensions run on the board (tb/tb_cycles.sv).
# Usage: .\tools\sim_cycles.ps1 [-Cases slices|full|all] [-Mem ideal|noisy|ddr|all]
#   ideal: no waits, 4 cycles of read latency
#   noisy: the default testbench memory (25 % waits, 25 % gaps)
#   ddr  : 10 % waits, 10 % gaps, 30 cycles of read latency (closer to a real DDR3 behind the HPS)
# Uses its own work library so it can run while sim.ps1 is not running (sim.ps1 deletes sim_work).
param(
  [ValidateSet('slices', 'full', 'scaled', 'all')][string]$Cases = 'slices',
  [ValidateSet('ideal', 'noisy', 'ddr', 'all')][string]$Mem = 'ideal'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ms = 'C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem'
$work = Join-Path $root 'sim_work_cycles'

Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
& "$ms\vlib.exe" $work | Out-Null
$rtl = @('fpgate_pkg.sv') + (Get-ChildItem (Join-Path $root 'rtl') -Filter *.sv | Where-Object Name -ne 'fpgate_pkg.sv' | ForEach-Object Name)
$files = $rtl | ForEach-Object { Join-Path $root "rtl\$_" }
$files += Join-Path $root 'tb\avalon_mem_model.sv'
$files += Join-Path $root 'tb\tb_cycles.sv'
& "$ms\vlog.exe" -sv -work $work @files | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host 'compile failed'; exit 1 }

# name, M, N, K, in bits, out bits, sub-tiles per panel  (as printed by accel_test on the board)
$slices = @(
  @('b0_q_w',    32, 32,  768,  8, 16, 2),
  @('b0_v_w',    32, 32,  768,  8,  8, 2),
  @('b0_fc1g_w', 32, 32,  768,  8,  8, 2),
  @('b0_fc2_w',  32, 32, 1232, 16, 16, 2),
  @('b1_fc2_w',  32, 32, 2048, 16, 16, 2)
)
$full = @(
  @('full_b0_q_w',   1025, 512,  768,  8, 16, 16),
  @('full_b0_fc2_w', 1025, 768, 1232, 16, 16,  4)
)
# Full layers are too long for ModelSim ASE (~700 cycles/s). Cycles grow linearly with the
# number of 16-column output tiles, so simulate 1, 2 and 4 tiles with the real M, K and
# sub-tiles per panel and extrapolate (cycles = a + b * tiles).
$scaled = @()
foreach ($n in 16, 32, 64) {
  $scaled += ,@("q_w_N$n",   1025, $n,  768,  8, 16, 16)
  $scaled += ,@("fc2_w_N$n", 1025, $n, 1232, 16, 16,  4)
}
$runList = @()
if ($Cases -in 'slices', 'all') { $runList += $slices }
if ($Cases -in 'full', 'all')   { $runList += $full }
if ($Cases -eq 'scaled')        { $runList += $scaled }

$mems = @{ ideal = @(0, 0, 4); noisy = @(25, 25, 4); ddr = @(10, 10, 30) }
$memNames = if ($Mem -eq 'all') { 'ideal', 'noisy', 'ddr' } else { @($Mem) }

Push-Location $root
foreach ($mn in $memNames) {
  foreach ($c in $runList) {
    $m = $mems[$mn]
    $g = @("-gM=$($c[1])", "-gN=$($c[2])", "-gK=$($c[3])", "-gIB=$($c[4])", "-gOB=$($c[5])", "-gSPP=$($c[6])",
           "-gWAIT_PCT=$($m[0])", "-gGAP_PCT=$($m[1])", "-gLAT=$($m[2])")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $out = & "$ms\vsim.exe" -c -work $work @g -do 'run -all; quit -f' tb_cycles
    $line = ($out | Where-Object { $_ -match 'RESULT' }) -replace '^# ', ''
    Write-Host ("[{0}] {1}: {2}  ({3:N0} s)" -f $mn, $c[0], $line, $sw.Elapsed.TotalSeconds)
  }
}
Pop-Location
