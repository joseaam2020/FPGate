/* fpgate_mem: shared between the kernel module and the user-space driver (accel/hps/src/accel_drv.c) */
#ifndef FPGATE_MEM_H
#define FPGATE_MEM_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint32_t __u32;
#endif

#define FPGATE_MEM_NAME "fpgate_mem"
#define FPGATE_MEM_PATH "/dev/fpgate_mem"

struct fpgate_mem_range {
	__u32 offset;	/* bytes from the start of the window */
	__u32 len;
};

#define FPGATE_MEM_MAGIC 0xF6
#define FPGATE_MEM_CLEAN _IOW(FPGATE_MEM_MAGIC, 1, struct fpgate_mem_range)
#define FPGATE_MEM_INVAL _IOW(FPGATE_MEM_MAGIC, 2, struct fpgate_mem_range)
#define FPGATE_MEM_FLUSH _IOW(FPGATE_MEM_MAGIC, 3, struct fpgate_mem_range)

#endif
