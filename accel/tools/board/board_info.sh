#!/bin/sh
# Read-only dump of board information for the DE10-Standard (Linux on the HPS).
# Writes nothing on the board. Usage on the board:  sh board_info.sh > board_info.txt
# Every value is printed between "RESULT <key>: BEGIN" and "RESULT <key>: END".

section() {
    echo "RESULT $1: BEGIN"
}

end_section() {
    echo "RESULT $1: END"
}

# 1. Kernel and build identification
section uname
uname -a
end_section uname

# 2. Kernel config options relevant to /dev/mem access, CMA and modules
section kconfig
if [ -f /proc/config.gz ]; then
    zcat /proc/config.gz 2>/dev/null | grep -E 'STRICT_DEVMEM|CONFIG_CMA|CONFIG_DMA_CMA|CONFIG_MODULES|CONFIG_UIO'
else
    echo "NO_CONFIG_GZ"
fi
end_section kconfig

# 3. Physical memory map
section iomem
cat /proc/iomem 2>&1
end_section iomem

# 4. Installed kernel modules directory
section modules
ls "/lib/modules/$(uname -r)" 2>&1 | head
end_section modules

# 5. Device-tree nodes related to the FPGA bridges (ff2... addresses or "bridge")
section devtree
find /proc/device-tree -type d \( -name '*ff2*' -o -name '*bridge*' \) 2>/dev/null | sort | while IFS= read -r node; do
    echo "NODE $node"
    echo "  name: $(basename "$node")"
    if [ -f "$node/compatible" ]; then
        echo "  compatible: $(tr '\0' ' ' < "$node/compatible")"
    fi
    if [ -f "$node/reg" ]; then
        echo "  reg (hex):"
        if command -v hexdump >/dev/null 2>&1; then
            hexdump -C "$node/reg" | sed 's/^/    /'
        else
            od -A x -t x1z "$node/reg" | sed 's/^/    /'
        fi
    fi
done
end_section devtree

# 6. Kernel command line
section cmdline
cat /proc/cmdline 2>&1
end_section cmdline

# 7. U-Boot environment (only if the tool exists)
section fw_printenv
if command -v fw_printenv >/dev/null 2>&1; then
    fw_printenv 2>/dev/null | head -50
else
    echo "NO_FW_PRINTENV"
fi
end_section fw_printenv
