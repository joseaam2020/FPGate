# Isolation test for the FPGA-to-SDRAM port: uses the JTAG master of the system
# (f2sdram_only_master) to read/write the DDR3 region from the PC, without the
# accelerator DMA and without Linux in the loop.
#
# Run (FPGA programmed with h4_DE10_Standard_GHRD.sof, Linux booted):
#   C:\intelFPGA_lite\18.1\quartus\sopc_builder\bin\system-console.exe --script=ddr_probe.tcl
#
# Steps are ordered from safest to most aggressive; the script stops after each one
# prints, so a hang tells us which access caused it.

set paths [get_service_paths master]
puts "masters found:"
foreach p $paths { puts "  $p" }

set m ""
foreach p $paths {
    if {[string match -nocase *f2sdram* $p]} { set m $p }
}
if {$m eq ""} {
    puts "no master with 'f2sdram' in its path: pick one from the list above and edit this script"
    return
}
puts "using: $m"
set mm [claim_service master $m ddr_probe]

set BASE 0x30100000

puts "1) read 8 words from the region (reads only)"
puts [master_read_32 $mm 0x30000000 8]

puts "2) write 8 words at $BASE"
master_write_32 $mm $BASE {0xA5A5A5A5 0x5A5A5A5A 0x11111111 0x22222222 0x33333333 0x44444444 0x55555555 0x66666666}
puts "   written"

puts "3) read them back"
puts [master_read_32 $mm $BASE 8]

puts "4) write a single byte (partial write) at [format 0x%08x [expr {$BASE + 0x40}]]"
master_write_8 $mm [expr {$BASE + 0x40}] 0x7E
puts "   written"
puts [master_read_32 $mm [expr {$BASE + 0x40}] 1]

puts "5) read 256 words (64 beats of 128 bit... as many single reads)"
set d [master_read_32 $mm $BASE 256]
puts "   read [llength $d] words"

puts "done: now check from Linux:  devmem 0x30100000 32   (expect 0xA5A5A5A5)"
close_service master $mm
