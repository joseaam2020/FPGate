# Identify which JTAG master of the system is f2sdram_only_master. The paths do not
# carry the instance names, so we probe each master with read-only accesses:
#   f2sdram_only_master : reads the DDR3 region, cannot reach 0xFF200000
#   hps_only_master     : reads the DDR3 region AND 0xFF200000 (accelerator VERSION
#                         through the HPS lightweight bridge, 0x46500001)
#   fpga_only_master    : reaches the FPGA peripherals only
# Every access is wrapped in catch; nothing is written.

set paths [get_service_paths master]
set i 0
foreach p $paths {
    puts "== master $i: $p"
    set mm [claim_service master $p id_probe]
    foreach addr {0x30000000 0xFF200000} {
        if {[catch {set v [master_read_32 $mm $addr 1]} err]} {
            puts "   read $addr : ERROR ($err)"
        } else {
            puts "   read $addr : $v"
        }
    }
    close_service master $mm
    incr i
}
puts "done"
