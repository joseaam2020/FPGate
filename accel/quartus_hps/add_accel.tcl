# Adds the FPGate accelerator to the Terasic GHRD system.
# Run from accel/quartus_hps:
#   qsys-script --script=add_accel.tcl --search-path="ip/**/*,$"
# Result: the accelerator's registers sit at the start of the lightweight bridge
# (0xFF200000, as in the design document); the GHRD peripherals behind mm_bridge_0
# move up to 0xFF240000.
package require -exact qsys 16.1

load_system soc_system.qsys

add_instance fpgate_accel fpgate_accel 1.0
add_connection clk_0.clk fpgate_accel.clock 
add_connection clk_0.clk_reset fpgate_accel.reset

add_connection hps_0.h2f_lw_axi_master fpgate_accel.avs
set_connection_parameter_value hps_0.h2f_lw_axi_master/fpgate_accel.avs baseAddress 0x00000000
set_connection_parameter_value hps_0.h2f_lw_axi_master/mm_bridge_0.s0 baseAddress 0x00040000

save_system soc_system.qsys
