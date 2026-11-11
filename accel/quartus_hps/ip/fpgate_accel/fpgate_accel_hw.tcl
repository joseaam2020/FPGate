# Platform Designer component for the FPGate accelerator (H0: CSR + placeholder engine).
# The RTL lives in accel/rtl; paths below are relative to this file.
package require -exact qsys 16.1

set_module_property NAME fpgate_accel
set_module_property DISPLAY_NAME "FPGate Accelerator"
set_module_property VERSION 1.0
set_module_property GROUP "FPGate"
set_module_property DESCRIPTION "Control registers and engine of the FPGate NSFW accelerator"
set_module_property INTERNAL false
set_module_property EDITABLE false

add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL fpgate_accel_top
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
add_fileset_file fpgate_accel_top.sv SYSTEM_VERILOG PATH ../../../rtl/fpgate_accel_top.sv TOP_LEVEL_FILE
add_fileset_file csr.sv SYSTEM_VERILOG PATH ../../../rtl/csr.sv
add_fileset_file h0_dummy_engine.sv SYSTEM_VERILOG PATH ../../../rtl/h0_dummy_engine.sv

# clock and reset
add_interface clock clock end
set_interface_property clock clockRate 0
add_interface_port clock clk clk Input 1

add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
add_interface_port reset rst_n reset_n Input 1

# register slave: 64 words x 32 bit, read latency 1, no wait states
add_interface avs avalon end
set_interface_property avs addressUnits WORDS
set_interface_property avs associatedClock clock
set_interface_property avs associatedReset reset
set_interface_property avs bitsPerSymbol 8
set_interface_property avs burstOnBurstBoundariesOnly false
set_interface_property avs explicitAddressSpan 0
set_interface_property avs holdTime 0
set_interface_property avs linewrapBursts false
set_interface_property avs maximumPendingReadTransactions 0
set_interface_property avs readLatency 1
set_interface_property avs readWaitTime 0
set_interface_property avs setupTime 0
set_interface_property avs timingUnits Cycles
set_interface_property avs writeWaitTime 0
add_interface_port avs avs_address     address     Input  6
add_interface_port avs avs_write       write       Input  1
add_interface_port avs avs_writedata   writedata   Input  32
add_interface_port avs avs_byteenable  byteenable  Input  4
add_interface_port avs avs_read        read        Input  1
add_interface_port avs avs_readdata    readdata    Output 32

# interrupt (not connected in H0; software polls STATUS)
add_interface irq interrupt end
set_interface_property irq associatedAddressablePoint avs
set_interface_property irq associatedClock clock
set_interface_property irq associatedReset reset
add_interface_port irq irq irq Output 1
