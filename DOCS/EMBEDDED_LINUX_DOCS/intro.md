## Origin

- The Linux kernel began in 1991 as a hobby project by Linus Torvalds and rapidly grew into a large, active community; today ~2,000+ contributors help each kernel release.

## Kernel roles

- Manage hardware resources (CPU, memory, I/O) and mediate concurrent access.
- Expose portable, architecture-agnostic APIs for user-space programs and libraries.
- Example: a single network interface can be multiplexed among multiple applications.

## System calls

- System calls are the stable kernel–user interface (≈ 400 calls).
- They provide file/device ops, networking, IPC, process management, memory mapping, timers, threads, and sync primitives.
- User programs generally call the wrapped C library functions rather than invoking syscalls directly.

## Pseudo filesystems

- The kernel exposes runtime information via pseudo (virtual) filesystems created on the fly.
  - `proc` (usually `/proc`) — OS and process information (memory, processes, params).
  - `sysfs` (usually `/sys`) — a device tree view populated by kernel frameworks and drivers.

## Versioning and releases

- Before 2003: large, infrequent stabilization releases (e.g., 2.0, 2.2, 2.4).
- Since 2003: official releases roughly every 10 weeks, with development managed via merge windows and short release-candidate cycles.
  - Example ranges: 2.6 (2003–2011), 3.x (2011–2015), 4.x (2015–2018), 5.x (2019–2022), 6.x (2022–).
- Features are added progressively to avoid major incompatible branches.

## Development model

- New release: two-week merge window for new features, then ~8 weekly release candidates (RCs) before final release.

## Stabilization and stable branches

- Fixes are merged to Linus’s mainline first; stable maintainers backport fixes that address real bugs/security issues into stable branches (e.g., 6.6.y, 6.1.y).
- Stable branches allow users to receive security and critical fixes within the same major line without API/behavior changes.
  - Support duration: stable branches are maintained for months; some are designated LTS (years).

## Long-Term Support (LTS)

- The last release of each year is typically designated LTS and may be supported (bug/security fixes) for up to 6 years.
- More details: https://www.kernel.org/category/releases.html
- Commercial vendors can offer extended support (e.g., Wind River Linux — up to 15 years; Ubuntu Core — up to 10 years).
- “If you are not using a supported distribution kernel, or a stable/longterm kernel, you have an insecure kernel.” — Greg Kroah-Hartman, 2019
- The Civil Infrastructure Platform (CIP) aims to provide much longer (≥10 years) support for selected LTS versions (e.g., 4.4, 4.19, 5.10, 6.1): https://wiki.linuxfoundation.org/civilinfrastructureplatform/start

## Vendor kernels and sub-communities

- Many chip vendors publish their own kernel sources focused on hardware support; these may differ significantly from mainline and are often suitable for early proof-of-concepts but not recommended for long-term products.
- Kernel sub-communities (architectures, driver subsystems, filesystems, memory management, scheduling, etc.) maintain specialized trees for cutting-edge development; these are typically not intended for production use.

