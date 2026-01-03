## Git packages installation and configuration

```bash
sudo apt install git gitk git-email
```

After installing git on a new machine, the first thing to do is to let git know about your name and e-mail
address

```bash
git config --global user.name 'My Name'
git config --global user.email me@mydomain.net
```

It can also be particularly useful to display line numbers when using the git grep command. This can be
enabled by default with the following configuration

```bash
git config --global grep.lineNumber true
```

## Get the kernel Sources

To begin working with the Linux kernel sources, we need to clone its reference git tree, the one managed by
Linus Torvalds.

However, this requires downloading more than 2.8 GB of data. If you are running this command from home,
or if you have very fast access to the Internet at work.

You can also do it directly by connecting to https://git.kernel.org

```bash
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux
cd linux
```


## Accessing stable releases

The Linux kernel repository from Linus Torvalds contains all the main releases of Linux, but not the stable
versions.
They are maintained by a separate team, and hosted in a separate repository.

You can access this separate repository as another remote to be able to use the stable releases.


```bash
git remote add stable https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux
git fetch stable
```


Choose a particular stable version
Let’s work with a particular stable version of the Linux kernel.


```bash
git branch -a
```

The remote branch we are interested in is remotes/stable/linux-6.7.y.

First, execute the following command to check which version you currently have


```bash
make kernelversion
```

You can also open the Makefile and look at the beginning of it to check this information.

Now, let’s create a local branch starting from that remote branch

```bash
git checkout -b 6.7.bootlin stable/linux-6.7.y
```

Check the version again using the make kernelversion command to make sure you now have a 6.7.y version.



## Use a kernel source indexing tool

Now that we know how to do things in a manual way, let’s use more automated tools.

Try Elixir at https://elixir.bootlin.com and choose the Linux version closest to yours.


## Getting familiar with the board (BeaglePlay)

Take some time to read about the board features and connectors

https://docs.beagleboard.org/latest/boards/beagleplay/01-introduction.html

### Download technical documentation

The first document to download is the datasheet for the TI AM62x SoC family, available on
https://www.ti.com/lit/gpn/am625

This document will give us details about pin assignments.

Secondly, download the Technical Reference Manual (TRM) for the TI AM62x SoC family, available on
https://www.ti.com/lit/pdf/spruiv7

Download the schematics for the BeaglePlay board:
https://openbeagle.org/beagleplay/beagleplay/-/blob/main/BeaglePlay_sch.pdf

Setting up serial communication with the board
The Beagle Play serial connector is a 3-pin header located right next to the board’s USB-C port. Using your
special USB to Serial adapter provided by your instructor, connect the ground wire (blue) to the pin labeled
”G”, the TX wire (red) to the pin labeled ”RX” and the RX wire (green) to the pin labeled ”TX” 3 .

You always should make sure that you connect the TX pin of the cable to the RX pin of the board, and vice
versa, whichever board and cables you use.

Once the USB to Serial connector is plugged in, a new serial port should appear: /dev/ttyUSB0
You can also see this device appear by looking at the output of dmesg.

To communicate with the board through the serial port, install a serial communication program, such as
picocom


```bash
sudo apt install picocom
```

If you run ls -l /dev/ttyUSB0, you can also see that only root and users belonging to the dialout group
have read and write access to this file. Therefore, you need to add your user to the dialout group:
sudo adduser $USER dialout

Important: for the group change to be effective, you have to completely log out from your session and log
in again (no need to reboot). A workaround is to run newgrp dialout, but it is not global. You have to run
it in each terminal.

Now, you can run the following command to start serial communication on /dev/ttyUSB0, with
a baudrate of 115200. If you wish to exit picocom, press [Ctrl][a] followed by [Ctrl][x].

```bash
picocom -b 115200 /dev/ttyUSB0,
```

There should be nothing on the serial line so far, as the board is not powered up yet.
Remove any SD card from the Beagle Play.

It is now time to power up your board by plugging in the USB-C cable supplied by your instructor to your
PC.

See what messages you get on the serial line. You should see U-boot start.
















