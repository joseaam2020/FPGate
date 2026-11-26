# Hold timing report for the DE10_Standard_GHRD project (run with quartus_sta -t).
# Usage (from accel/quartus_hps): quartus_sta -t report_hold.tcl

project_open DE10_Standard_GHRD

create_timing_netlist -model slow -temperature 0 -voltage 1100
read_sdc
update_timing_netlist

create_timing_netlist -model fast -temperature 0 -voltage 1100
read_sdc
update_timing_netlist

report_timing -hold -npaths 5 -detail full_path \
    -to_clock [get_clocks *pll_0*divclk*] \
    -file hold_paths.txt

project_close
