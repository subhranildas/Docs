## User space device drivers

 - The kernel provides some mechanisms to access hardware from userspace:
    - SPI devices with spidev, spi/spidev
    - USB devices with libusb, https://libusb.info/
    - I2C devices with i2cdev, i2c/dev-interface
    - GPIOs with libgpiod, https://libgpiod.readthedocs.io
    - Memory-mapped devices with UIO, including interrupt handling,
    driver-api/uio-howto

 - These solutions can only be used if:
    - There is no need to leverage an existing kernel subsystem such as the networking
    stack or filesystems.
    - There is no need for the kernel to act as a “multiplexer” for the device: only one
    application accesses the device.

 - Certain classes of devices like printers and scanners do not have any kernel
 support, they have always been handled in user space for historical reasons.
 Otherwise this is not how the system should be architectured. Kernel drivers
 should always be preferred!

 - Advantages
    - No need for kernel coding skills.
    - Drivers can be written in any language, even Perl!
    - Drivers can be kept proprietary.
    - Driver code can be killed and debugged. Cannot crash the kernel.
    - Can use floating-point computation.
    - Potentially higher performance, especially for memory-mapped devices, thanks to the
    avoidance of system calls.

 - Drawbacks
    - The kernel has no longer access to the device.
    - None of the standard applications will be able to use it.
    - Cannot use any hardware abstraction or software helpers from the kernel
    - Need to adapt applications when changing the hardware.
    - Less straightforward to handle interrupts: increased latency.
