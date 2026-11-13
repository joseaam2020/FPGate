# Builds the H4 system from the untouched Terasic GHRD (ghrd_original.qsys):
#  - a 50 -> 150 MHz PLL for the accelerator and the HPS SDRAM port clock,
#  - fpgate_accel_core: registers at the start of the lightweight bridge (0xFF200000),
#    the other GHRD peripherals (LEDs, switches, keys, JTAG UART, debug masters) are removed,
#  - the core's four masters (engine read/write, softmax unit read2/write2) on hps_0.f2h_sdram0_data.
# Idempotent: it always starts from ghrd_original.qsys and writes soc_system.qsys.
# Run from accel/quartus_hps:
#   qsys-script --script=build_system.tcl --search-path="ip/**/*,$"
package require -exact qsys 16.1

# accelerator clock (the PLL also clocks the HPS SDRAM port): lowered in 5 MHz steps until timing closes
set ACCEL_MHZ 120

load_system ghrd_original.qsys

# unused GHRD peripherals and debug masters (the accelerator is the only lightweight-bridge slave)
foreach inst {hps_only_master fpga_only_master f2sdram_only_master ILC jtag_uart led_pio button_pio dipsw_pio mm_bridge_0} {
    remove_instance $inst
}

# PLL
add_instance pll_0 altera_pll
set_instance_parameter_value pll_0 gui_reference_clock_frequency 50.0
set_instance_parameter_value pll_0 gui_number_of_clocks 1
set_instance_parameter_value pll_0 gui_output_clock_frequency0 [format %.1f $ACCEL_MHZ]
set_instance_parameter_value pll_0 gui_use_locked false
add_connection clk_0.clk pll_0.refclk
add_connection clk_0.clk_reset pll_0.reset

# accelerator core
add_instance fpgate_accel fpgate_accel_core 1.0
add_connection pll_0.outclk0 fpgate_accel.clock
add_connection clk_0.clk_reset fpgate_accel.reset

add_connection hps_0.h2f_lw_axi_master fpgate_accel.avs
set_connection_parameter_value hps_0.h2f_lw_axi_master/fpgate_accel.avs baseAddress 0x00000000

# HPS SDRAM port: clocked by the PLL, driven by the two masters (plus the GHRD debug master)
foreach c [get_connections hps_0.f2h_sdram0_clock] { remove_connection $c }
add_connection pll_0.outclk0 hps_0.f2h_sdram0_clock
set_instance_parameter_value hps_0 F2H_SDRAM0_CLOCK_FREQ [expr {int($ACCEL_MHZ * 1000000)}]

add_connection fpgate_accel.rd hps_0.f2h_sdram0_data
set_connection_parameter_value fpgate_accel.rd/hps_0.f2h_sdram0_data baseAddress 0x00000000
add_connection fpgate_accel.wr hps_0.f2h_sdram0_data
set_connection_parameter_value fpgate_accel.wr/hps_0.f2h_sdram0_data baseAddress 0x00000000
add_connection fpgate_accel.rd2 hps_0.f2h_sdram0_data
set_connection_parameter_value fpgate_accel.rd2/hps_0.f2h_sdram0_data baseAddress 0x00000000
add_connection fpgate_accel.wr2 hps_0.f2h_sdram0_data
set_connection_parameter_value fpgate_accel.wr2/hps_0.f2h_sdram0_data baseAddress 0x00000000

save_system soc_system.qsys
