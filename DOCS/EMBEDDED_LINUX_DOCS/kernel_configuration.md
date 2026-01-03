## Kernel Configuration

 - The kernel contains thousands of device drivers, filesystem drivers, network
protocols and other configurable items.
 - Thousands of options are available, that are used to selectively compile parts of
 the kernel source code.
 - The kernel configuration is the process of defining the set of options with which
 you want your kernel to be compiled
 - The set of options depends on the following
    - The target architecture and on your hardware (for device drivers, etc.)
    - The capabilities you would like to give to your kernel (network capabilities,
    filesystems, real-time, etc.). Such generic options are available in all architectures.

## Kernel configuration and build system

 - The kernel configuration and build system is based on multiple Makefiles
 - One only interacts with the main Makefile, present at the top directory of the
 kernel source tree
 - Interaction takes place
    - using the make tool, which parses the Makefile
    - through various targets, defining which action should be done (configuration,
    compilation, installation, etc.).
    - Run make help to see all available targets.
 - Example
    - cd linux/
    - make <target>

## Specifying the target architecture

First, specify the architecture for the kernel to build

 - Set ARCH to the name of a directory under arch/:
 ARCH=arm or ARCH=arm64 or ARCH=riscv, etc
 - By default, the kernel build system assumes that the kernel is configured and built
 for the host architecture (x86 in our case, native kernel compiling)
 - The kernel build system will use this setting to:
    - Use the configuration options for the target architecture.
    - Compile the kernel with source code and headers for the target architecture.

## Choosing a compiler

The compiler invoked by the kernel Makefile is $(CROSS_COMPILE)gcc

 - Specifying the compiler is already needed at configuration time, as some kernel
 configuration options depend on the capabilities of the compiler.
 - When compiling natively
    - Leave CROSS_COMPILE undefined and the kernel will be natively compiled for the host
    architecture using gcc.
 - When using a cross-compiler
    - Specify the prefix of your cross-compiler executable, for example for arm-linux-gnueabi-gcc:
    CROSS_COMPILE=arm-linux-gnueabi-

 - Set LLVM to 1 to compile your kernel with Clang.
 - See our LLVM tools for the Linux kernel presentation.


## Specifying ARCH and CROSS_COMPILE

There are actually two ways of defining ARCH and CROSS_COMPILE:
 - Pass ARCH and CROSS_COMPILE on the make command line:
    - make ARCH=arm CROSS_COMPILE=arm-linux- ...
    - Drawback: it is easy to forget to pass these variables when you run any make
    - command, causing your build and configuration to be screwed up.
 - Define ARCH and CROSS_COMPILE as environment variables:
    - export ARCH=arm
    - export CROSS_COMPILE=arm-linux-
    - Drawback: it only works inside the current shell or terminal. You could put these
    - settings in a file that you source every time you start working on the project, see
    - also the https://direnv.net/ project.


## Initial configuration

Diﬀicult to find which kernel configuration will work with your hardware and root
filesystem. Start with one that works!

 - Desktop or server case:
    - Advisable to start with the configuration of your running kernel:
    cp /boot/config-`uname -r` .config

 - Embedded platform case:
    - Default configurations stored in-tree as minimal configuration files (only listing
    settings that are different with the defaults) in arch/<arch>/configs/
    - make help will list the available configurations for your platform
    - To load a default configuration file, just run make foo_defconfig (will erase your
    current .config!)

- On ARM 32-bit, there is usually one default configuration per CPU family
- On ARM 64-bit, there is only one big default configuration to customize

## Create your own default configuration
 - Use a tool such as make menuconfig to make changes to the configuration
 - Saving your changes will overwrite your .config (not tracked by Git)
 - When happy with it, create your own default configuration file:
    - Create a minimal configuration (non-default settings) file:
    make savedefconfig
    - Save this default configuration in the right directory:
    mv defconfig arch/<arch>/configs/myown_defconfig
    - Add this file to Git.
 - This way, you can share a reference configuration inside the kernel sources and
 other developers can now get the same .config as you by running
 make myown_defconfig
 - When you use an embedded build system (Buildroot, OpenEmbedded) use its
 specific commands. E.g. make linux-menuconfig and
 make linux-update-defconfig in Buildroot.

## Create your own default configuration

 - Use a tool such as make menuconfig to make changes to the configuration
 - Saving your changes will overwrite your .config (not tracked by Git)
 - When happy with it, create your own default configuration file:
    - Create a minimal configuration (non-default settings) file:
    make savedefconfig
    - Save this default configuration in the right directory:
    mv defconfig arch/<arch>/configs/myown_defconfig
    - Add this file to Git.
 - This way, you can share a reference configuration inside the kernel sources and
 other developers can now get the same .config as you by running
 make myown_defconfig
 - When you use an embedded build system (Buildroot, OpenEmbedded) use its
 specific commands. E.g. make linux-menuconfig and
 make linux-update-defconfig in Buildroot.

## Built-in or module?

 - The kernel image is a single file, resulting from the linking of all object files that
 correspond to features enabled in the configuration
    - This is the file that gets loaded in memory by the bootloader
    - All built-in features are therefore available as soon as the kernel starts, at a time
    where no filesystem exists
 - Some features (device drivers, filesystems, etc.) can however be compiled as
 modules
    - These are plugins that can be loaded/unloaded dynamically to add/remove features
    to the kernel
    - Each module is stored as a separate file in the filesystem, and therefore access
    to a filesystem is mandatory to use modules
    - This is not possible in the early boot procedure of the kernel, because no filesystem
    is available

## Kernel option types

There are different types of options, defined in Kconfig files:

 - bool options, they are either
    - true (to include the feature in the kernel) or
    - false (to exclude the feature from the kernel)
 - tristate options, they are either
    - true (to include the feature in the kernel image) or
    - module (to include the feature as a kernel module) or
    - false (to exclude the feature)
 - int options, to specify integer values
 - hex options, to specify hexadecimal values
    - Example: CONFIG_PAGE_OFFSET=0xC0000000
 - string options, to specify string values
    - Example: CONFIG_LOCALVERSION=-no-network
    - Useful to distinguish between two kernels built from different options

## Kernel option dependencies

Enabling a network driver requires the network stack to be enabled, therefore
configuration symbols have two ways to express dependencies:

 - depends on dependency:
```md
    config B
        depends on A
```
    - B is not visible until A is
    enabled
    - Works well for dependency
    chains

 - select dependency:
```md
    config A
        select B
```
    - When A is enabled, B is enabled too (and
    cannot be disabled manually)
    - Should preferably not select symbols with
    depends on dependencies
    - Used to declare hardware features or select
    libraries

```md
config SPI_ATH79
    tristate "Atheros AR71XX/AR724X/AR913X SPI controller driver"
    depends on ATH79 || COMPILE_TEST
    select SPI_BITBANG
    help
        This enables support for the SPI controller present on the
        Atheros AR71XX/AR724X/AR913X SoCs.
```

