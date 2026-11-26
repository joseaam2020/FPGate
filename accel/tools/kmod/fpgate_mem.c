// SPDX-License-Identifier: GPL-2.0
/*
 * fpgate_mem: cached user mapping of the DDR3 window reserved for the FPGA accelerator.
 *
 * The window (default 0x30000000, 128 MiB) lies outside the RAM Linux manages, so /dev/mem maps it
 * uncached (~110-155 MB/s from the Cortex-A9). This device maps it write-back cached and gives the
 * user two ioctls for the cache maintenance that keeps the FPGA and the CPU coherent:
 *   FPGATE_MEM_CLEAN  (offset, len): CPU writes -> DDR3, before the accelerator reads them;
 *   FPGATE_MEM_INVAL  (offset, len): drop CPU copies, before the CPU reads what the accelerator wrote.
 * Both work on L1 (by MVA, through a kernel alias) and on the L2 (PL310, by physical address).
 *
 * Load:  insmod fpgate_mem.ko [base=0x30000000] [size=0x8000000]
 * Node:  /dev/fpgate_mem (misc, created by devtmpfs/mdev)
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/io.h>
#include <linux/uaccess.h>
#include <linux/miscdevice.h>
#include <linux/dma-mapping.h>
#include <asm/cacheflush.h>
#include <asm/outercache.h>

#include "fpgate_mem.h"

static unsigned long base = 0x30000000UL;
static unsigned long size = 0x08000000UL;
module_param(base, ulong, 0444);
MODULE_PARM_DESC(base, "physical base of the accelerator window");
module_param(size, ulong, 0444);
MODULE_PARM_DESC(size, "size of the window in bytes");

static void __iomem *kva;	/* cached kernel alias of the whole window, for L1 maintenance by MVA */

static int fm_mmap(struct file *f, struct vm_area_struct *vma)
{
	const unsigned long len = vma->vm_end - vma->vm_start;
	const unsigned long off = vma->vm_pgoff << PAGE_SHIFT;

	if (off >= size || len > size - off)
		return -EINVAL;
	/* vm_page_prot as set by the core for a shared mapping: normal write-back cached memory */
	vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
	return remap_pfn_range(vma, vma->vm_start, (base + off) >> PAGE_SHIFT, len, vma->vm_page_prot);
}

static int fm_range_ok(const struct fpgate_mem_range *r)
{
	return r->len != 0 && r->offset < size && r->len <= size - r->offset;
}

static long fm_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct fpgate_mem_range r;
	void *va;
	phys_addr_t pa;

	if (copy_from_user(&r, (void __user *)arg, sizeof r))
		return -EFAULT;
	if (!fm_range_ok(&r))
		return -EINVAL;
	va = (void *)((char __force *)kva + r.offset);
	pa = base + r.offset;

	switch (cmd) {
	/* Of the L1 range operations, 4.5 exports to modules only __cpuc_flush_dcache_area (clean + invalidate
	 * by kernel address); the outer-cache (L2) operations are all available. Clean + invalidate is correct
	 * for both directions: the CPU never holds dirty lines of the areas the accelerator writes (it only
	 * reads them), so on an invalidate the write-back part has nothing to write. */
	case FPGATE_MEM_CLEAN:
		/* same order as the DMA API for DMA_TO_DEVICE: inner first, then outer */
		__cpuc_flush_dcache_area(va, r.len);
		outer_clean_range(pa, pa + r.len);
		return 0;
	case FPGATE_MEM_INVAL:
		/* DMA_FROM_DEVICE: outer first, then inner */
		outer_inv_range(pa, pa + r.len);
		__cpuc_flush_dcache_area(va, r.len);
		return 0;
	case FPGATE_MEM_FLUSH:
		/* clean + invalidate in both levels, when the CPU wrote and the accelerator will overwrite */
		__cpuc_flush_dcache_area(va, r.len);
		outer_flush_range(pa, pa + r.len);
		return 0;
	default:
		return -ENOTTY;
	}
}

static const struct file_operations fm_fops = {
	.owner = THIS_MODULE,
	.mmap = fm_mmap,
	.unlocked_ioctl = fm_ioctl,
	.compat_ioctl = fm_ioctl,
};

static struct miscdevice fm_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = FPGATE_MEM_NAME,
	.fops = &fm_fops,
	.mode = 0600,
};

static int __init fm_init(void)
{
	int rc;

	if (!size || (base & ~PAGE_MASK) || (size & ~PAGE_MASK))
		return -EINVAL;
	kva = ioremap_cache(base, size);
	if (!kva) {
		pr_err("fpgate_mem: cannot map %#lx+%#lx cached\n", base, size);
		return -ENOMEM;
	}
	rc = misc_register(&fm_dev);
	if (rc) {
		iounmap(kva);
		return rc;
	}
	pr_info("fpgate_mem: window %#lx+%#lx mapped cached, /dev/%s\n", base, size, FPGATE_MEM_NAME);
	return 0;
}

static void __exit fm_exit(void)
{
	misc_deregister(&fm_dev);
	iounmap(kva);
}

module_init(fm_init);
module_exit(fm_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Cached mapping of the FPGA accelerator DDR3 window with cache maintenance ioctls");
