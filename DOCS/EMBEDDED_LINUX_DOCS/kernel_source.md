# Linux kernel source

## Programming language

- The kernel is written mostly in C with small Assembly parts for CPU/machine init and critical routines. No C++ is used (see https://lkml.org/lkml/2004/1/20/20).
- Rust support is being introduced (example: `drivers/net/phy/ax88796b_rust.rs`).
- The kernel is traditionally built with GCC (many GCC extensions are used — see https://gcc.gnu.org/onlinedocs/gcc-10.2.0/gcc/C-Extensions.html); some architectures can also be built with Clang/LLVM (https://clangbuiltlinux.github.io/).

## No user C library

- The kernel is standalone during early boot and cannot use user-space libraries; it provides its own equivalents (e.g., `printk()`, `kmalloc()`, `memset()`).

## Portability & constraints

- Code outside `arch/` is intended to be portable; the kernel supplies abstractions for endianness, I/O memory access, memory barriers, and DMA operations.
- Do not use floating point in kernel code (some targets lack an FPU).

## Memory and runtime constraints

- Kernel space has strict constraints: limited stack size (typically 4–8 KB), no automatic stack growth, no swapping of kernel memory (except tmpfs which lives in page cache), and no built-in recovery from illegal memory accesses (you get oops/panic output). Avoid recursion.

## Licensing (GPLv2)

- The Linux kernel is licensed under GNU GPL version 2. This permits use, study, modification, and redistribution, but redistributed kernel binaries (modified or not) must be provided under GPLv2 with source.
- Enforcement has been successful in court (see https://en.wikipedia.org/wiki/Gpl-violations.org#Notable_victories).
- The obligation to provide source applies when the device is distributed to customers.

## Proprietary drivers and vendor kernels

- Distributing a binary kernel containing statically linked proprietary drivers is not permitted. Out-of-tree proprietary modules remain a legal gray area and are generally discouraged (https://www.kernel.org/doc/html/latest/process/kernel-driver-statement.html).
- Many vendors ship their own kernel forks focused on hardware support; these may diverge from mainline and are usually suitable for early development but not ideal for long-term product maintenance.

## Practical notes and risks

- Some vendors (e.g., NVIDIA) use wrappers, firmware, or user-space workarounds to avoid GPL obligations; see commentary from projects like Bootlin (https://bootlin.com).

## Benefits of GPL and mainlining

- GPL drivers enable code reuse, easier distribution, and legal certainty.
- Getting a driver merged upstream brings community review, ongoing maintenance, and reduced long-term support burden.
