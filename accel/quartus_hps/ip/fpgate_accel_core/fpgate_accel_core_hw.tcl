# Platform Designer component for the FPGate accelerator core: CSR + GEMM engine + attention softmax unit.
# Register slave (128 words: engine at 0x000..0x0FF, softmax unit at 0x100..0x1FF) plus ONE Avalon-MM master
# (mem: 256-bit, byte addresses, bursts up to 16 beats) towards the HPS SDRAM port; the four internal DMA
# masters are merged by mem_arbiter. RTL lives in accel/rtl; paths are relative to this file.
package require -exact qsys 16.1

set_module_property NAME fpgate_accel_core
set_module_property DISPLAY_NAME "FPGate Accelerator Core"
set_module_property VERSION 1.0
set_module_property GROUP "FPGate"
set_module_property DESCRIPTION "CSR + GEMM engine (DMA, buffers, array, epilogue) + attention softmax unit of the FPGate NSFW accelerator"
set_module_property INTERNAL false
set_module_property EDITABLE false

add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL fpgate_accel_core
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
# the package first: later files import it
foreach f {fpgate_pkg pe gemm_array requant_lane epilogue_row linear_tile tile_buffers
           operand_feeder tile_sequencer dma_rd dma_wr mem_arbiter rope_unit swiglu_lane swiglu_stage gemm_engine csr csr_sm
           softmax_idx_lane softmax_p3_lane softmax_exp_rom softmax_recip_rom softmax_rowram attn_softmax fpgate_accel_core} {
  if {$f eq "fpgate_accel_core"} {
    add_fileset_file $f.sv SYSTEM_VERILOG PATH ../../../rtl/$f.sv TOP_LEVEL_FILE
  } else {
    add_fileset_file $f.sv SYSTEM_VERILOG PATH ../../../rtl/$f.sv
  }
}

# clock and reset
add_interface clock clock end
set_interface_property clock clockRate 0
add_interface_port clock clk clk Input 1

add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
add_interface_port reset rst_n reset_n Input 1

# register slave: 128 words x 32 bit, read latency 1, no wait states
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
add_interface_port avs avs_address     address     Input  7
add_interface_port avs avs_write       write       Input  1
add_interface_port avs avs_writedata   writedata   Input  32
add_interface_port avs avs_byteenable  byteenable  Input  4
add_interface_port avs avs_read        read        Input  1
add_interface_port avs avs_readdata    readdata    Output 32

# memory master: read bursts up to 16 beats (up to 16 outstanding), single-beat writes with byte enables
add_interface mem avalon start
set_interface_property mem addressUnits SYMBOLS
set_interface_property mem associatedClock clock
set_interface_property mem associatedReset reset
set_interface_property mem bitsPerSymbol 8
set_interface_property mem burstOnBurstBoundariesOnly false
set_interface_property mem linewrapBursts false
set_interface_property mem maximumPendingReadTransactions 16
set_interface_property mem readLatency 0
add_interface_port mem mem_address        address        Output 32
add_interface_port mem mem_read           read           Output 1
add_interface_port mem mem_write          write          Output 1
add_interface_port mem mem_writedata      writedata      Output 256
add_interface_port mem mem_byteenable     byteenable     Output 32
add_interface_port mem mem_burstcount     burstcount     Output 8
add_interface_port mem mem_waitrequest    waitrequest    Input  1
add_interface_port mem mem_readdata       readdata       Input  256
add_interface_port mem mem_readdatavalid  readdatavalid  Input  1

# interrupt (software polls STATUS for now)
add_interface irq interrupt end
set_interface_property irq associatedAddressablePoint avs
set_interface_property irq associatedClock clock
set_interface_property irq associatedReset reset
add_interface_port irq irq irq Output 1
