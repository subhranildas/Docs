## Difference between Serial Communication protocols like I2C, SPI and UART

## What happens when you boot

https://opensource.com/article/17/2/linux-boot-and-startup
https://leetcode.com/discuss/post/124638/what-happens-in-the-background-from-the-f4k7h/

## UART/I2C/SPI (compare, pull-ups)

https://www.totalphase.com/blog/2021/12/i2c-vs-spi-vs-uart-introduction-and-comparison-similarities-differences/

## What are Software/hardware break points, JTAG

### What is a Breakpoint?

A breakpoint is a debugging mechanism that tells the processor:
“Pause program execution here so I can inspect what’s going on.”
It allows us to halt the CPU, inspect registers, variables, memory, and then resume or step through your code.

There are two types of breakpoints:

#### Software Breakpoints

- How it works:

  - The debugger replaces an instruction in the program’s code (in Flash or RAM) with a special “break” instruction (e.g., BKPT in ARM Cortex-M).
  - When the CPU reaches that instruction, it triggers a debug exception, halting execution.
  - The debugger then takes control and lets you inspect state.

- Characteristics:

  - Location : Inside the code memory (Flash or RAM)
  - Mechanism Replaces actual instruction with a breakpoint opcode
  - Number Usually unlimited (software inserted)
  - Speed Slightly slower (since code is modified)
  - Limitation Cannot be used in read-only memory if Flash modification isn’t allowed

- Example (ARM Cortex-M):

  - When you set a breakpoint in STM32CubeIDE, it writes a BKPT 0xAB instruction at that address.

- Use Case:
  - Common for debugging application code.
  - Not ideal for Flash-based code on-the-fly, since writing to Flash is slow.

#### Hardware Breakpoints

- How it works:

  - Hardware breakpoints are implemented inside the CPU’s debug hardware.
  - The debugger programs the Flash Patch and Breakpoint (FPB) unit in ARM Cortex-M cores.
  - The FPB compares the current instruction address with the stored breakpoint address — if it matches, the CPU halts.

- Characteristics:

  - Location Stored in dedicated debug hardware
  - Mechanism Address comparison — no code modification
  - Number Limited (usually 2–8 hardware slots in Cortex-M)
  - Speed Very fast, no code modification
  - Works in Flash
  - Works in RAM

- Use Case:
  - Debugging Flash-based code.
  - Essential when debugging code in ROM, where software breakpoints can’t be inserted.

#### Watchpoints

- A watchpoint (or data breakpoint) halts execution when a specific memory address is read or written — great for tracking variable corruption.
- Implemented by the DWT (Data Watchpoint and Trace) unit.
- Example: Halt when variable x is written to.

### JTAG (Joint Test Action Group)

JTAG is a hardware debugging and testing interface standardized as IEEE 1149.1.

- It allows:
  - Programming Flash
  - Setting breakpoints
  - Reading/writing memory
  - Stepping through code
  - Boundary scan (testing hardware pin connections)

#### How it works

JTAG defines a serial test access port (TAP) with 4–5 signals:

- TCK : Test Clock Clock for JTAG interface
- TMS : Test Mode Select Controls TAP state machine
- TDI : Test Data In Serial input to target
- TDO : Test Data Out Serial output from target
- TRST : (optional) Test Reset Resets TAP controller

A JTAG debug probe (like ST-Link, J-Link, or OpenOCD) connects these lines between the PC and the MCU.

#### JTAG vs. SWD (Serial Wire Debug)

Most STM32s and ARM Cortex-M MCUs use SWD, a 2-wire subset of JTAG.

| Feature      | JTAG                | SWD                    |
| ------------ | ------------------- | ---------------------- |
| **Pins**     | 4–5                 | 2                      |
| **Speed**    | Moderate            | Faster (less overhead) |
| **Use case** | Testing + Debugging | Debugging only         |
| **Standard** | IEEE 1149.1         | ARM-specific           |

So, in STM32CubeIDE, when we connect using ST-Link, you’re actually using SWD, not full JTAG — but they serve the same purpose:
Controlling the CPU, reading/writing memory, and handling breakpoints.

## Which endianness is: A) x86 families. B) ARM families. C) internet protocols. D) other processors? One of these is kind of a trick question.

Endianness defines how multi-byte data (like 32-bit integers) are stored or transmitted:

| Type              | Description                                                 | Example for 0x12345678 |
| ----------------- | ----------------------------------------------------------- | ---------------------- |
| **Little-endian** | Least significant byte stored first (lowest memory address) | `78 56 34 12`          |
| **Big-endian**    | Most significant byte stored first                          | `12 34 56 78`          |

- x86 Families : Little-endian

  - All Intel and AMD x86 and x86-64 processors use little-endian byte order.
  - Example: Writing 0x12345678 to memory stores 78 56 34 12.
  - This convention dates back to the original Intel 8086.

- ARM Families : Bi-endian (configurable, but usually little-endian)

  - ARM processors can support both little-endian and big-endian modes (bi-endian).
  - Most modern ARM systems (e.g., STM32, Raspberry Pi, Android devices) run in little-endian mode by default.
  - Big-endian is rarely used except in some network or DSP applications.

- Internet Protocols : Big-endian — also known as network byte order.
  - All Internet standards (TCP/IP, UDP, IP headers, DNS, etc.) define multi-byte values as big-endian.
  - This ensures consistent communication between different architectures.

For example:

```c
htons(), htonl(), ntohs(), ntohl()
```

These C functions convert between host byte order and network byte order.

## Explain how interrupts work. What are some things that you should never do in an interrupt function ?

An interrupt is a signal that temporarily stops the normal execution flow of the CPU to handle an urgent event.

It allows the processor to respond immediately to hardware or software events (like a timer overflow, UART RX, GPIO edge, etc.) — without constantly polling.

### Interrupt Flow

- Main program runs normally

  - CPU executes instructions sequentially (main loop or tasks).

- Event occurs

  - A hardware peripheral (e.g., Timer, UART, GPIO) or software trigger raises an interrupt request (IRQ) to the CPU.

- CPU checks interrupt priority

  - If interrupts are enabled and the new one has higher priority than the current execution, the CPU pauses the current task.

- Context saving

  - The CPU automatically pushes key registers (like PC, PSR, LR) onto the stack.
  - This preserves the exact state of the interrupted program.

- Interrupt Service Routine (ISR) executes

  - The CPU jumps to the Interrupt Vector Table, finds the correct ISR handler, and runs it.

- ISR finishes

  - The CPU restores the previous context from the stack.
  - Execution resumes exactly where it was interrupted.

### Thing to not do in an Interrupt

| Rule                              | Description                | Recommended Action        |
| --------------------------------- | -------------------------- | ------------------------- |
| **Keep ISRs short**               | Don’t block or delay       | Use flags or queues       |
| **Avoid non-reentrant functions** | `printf`, `malloc`, etc.   | Use ISR-safe alternatives |
| **Don’t block RTOS calls**        | Avoid task delays or waits | Use `FromISR()` versions  |
| **Protect shared data**           | Avoid race conditions      | Use `volatile` or atomics |
| **Always clear interrupt flags**  | Prevent re-trigger loops   | Clear before exiting      |
| **No dynamic allocation**         | Avoid heap in ISR          | Use static buffers        |

## Where does the interrupt table reside in the memory map for various processor families?

The Interrupt Vector Table is a table of addresses (pointers) that tell the CPU where to jump when a specific interrupt or exception occurs.

Each entry in the table corresponds to one interrupt source:

Exception 0 → Reset Handler
Exception 1 → NMI Handler
Exception 2 → HardFault Handler

When an interrupt occurs, the CPU fetches the corresponding ISR address from this table.

| Architecture       | Default IVT Address            | Relocatable?    | Register Used | Notes                             |
| ------------------ | ------------------------------ | --------------- | ------------- | --------------------------------- |
| **ARM Cortex-M**   | `0x0000_0000`                  | ✅ Yes          | **VTOR**      | Can move to RAM for dynamic ISR   |
| **ARM Cortex-A/R** | `0x0000_0000` or `0xFFFF_0000` | ✅ Yes          | **VBAR**      | OS often uses high vectors        |
| **x86**            | `0x0000` (Real mode)           | ✅ Yes          | **IDTR**      | Modern OS relocates IDT           |
| **AVR**            | `0x0000`                       | ⚠️ Partially    | **Fuse bits** | Bootloader section possible       |
| **MSP430**         | End of Flash                   | ❌ Fixed        | —             | Reset vector is last word         |
| **PIC**            | `0x0000`                       | ⚠️ Limited      | —             | dsPIC/PIC24 allow remap           |
| **RISC-V**         | Impl.-defined (e.g., `0x100`)  | ✅ Yes          | **mtvec**     | Base + offset mode                |
| **PowerPC**        | `0x0000_0100`                  | ⚠️ Configurable | **IVPR**      | Sometimes mirrored at high memory |
| **MIPS**           | `0x8000_0180`                  | ✅ Yes          | **EBase**     | Relocatable exception base        |

## In which direction does the stack grow in various processor families?

| Direction                             | Description                                                                         |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| **Downward (toward lower addresses)** | Stack pointer (SP) decreases on push, increases on pop. Most common in modern CPUs. |
| **Upward (toward higher addresses)**  | Stack pointer increases on push, decreases on pop. Less common.                     |

| Processor Family                       | Stack Growth Direction                          | Notes                                                                                                         |
| -------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **x86 / x86-64**                       | Downward (toward lower addresses)               | `push` decrements SP, `pop` increments SP. Used in both real and protected mode.                              |
| **ARM Cortex-M / Cortex-A / Cortex-R** | Downward                                        | SP decrements on push; stack grows toward lower addresses. Default full-descending stack.                     |
| **AVR (8-bit)**                        | Downward                                        | Stack grows from high memory (RAM top) toward lower addresses.                                                |
| **MSP430**                             | Downward                                        | Stack pointer initialized to top of RAM; grows downward.                                                      |
| **PIC (8-bit)**                        | Upward (sometimes downward depending on family) | Classic PIC8: hardware stack grows upward (small hardware stack). PIC24/dsPIC: software stack grows downward. |
| **RISC-V**                             | Downward                                        | SP decrements on push; standard convention.                                                                   |
| **PowerPC**                            | Downward                                        | Full-descending stack convention (SP points to top of stack).                                                 |
| **MIPS**                               | Downward                                        | Stack grows toward lower addresses; SP decrements on push.                                                    |

## List some ARM cores. For embedded use, which cores were most commonly used in the past? now?

### ARM Core Families Overview

| ARM Family              | Typical Use                               | Key Features                                                                                   |
| ----------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Cortex-M                | Microcontrollers (MCUs)                   | Low power, deterministic interrupt latency, real-time control, Thumb/Thumb-2 only              |
| Cortex-R                | Real-time / safety-critical               | Hard real-time performance, tightly coupled memories, used in automotive & storage controllers |
| Cortex-A                | Applications processors                   | MMU, Linux/Android capable, high performance, often in smartphones and SBCs                    |
| Neoverse                | Datacenter & networking                   | Multicore scalable architecture for servers                                                    |
| Legacy ARM (pre-Cortex) | Classic embedded CPUs (ARM7, ARM9, ARM11) | Simpler pipelines, used before Cortex era                                                      |

### Common ARM Cores by Family

| Family     | Examples                                                                | Notes                                                              |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Cortex-M   | M0, M0+, M3, M4, M7, M23, M33, M55                                      | M4/M7 add DSP/FPU; M33 adds TrustZone; M55 supports Helium (SIMD)  |
| Cortex-R   | R4, R5, R7, R8, R52                                                     | Used in hard-real-time, e.g., automotive ECUs, SSD controllers     |
| Cortex-A   | A5, A7, A8, A9, A15, A17, A35, A53, A57, A72, A73, A75, A76, A78, A510+ | Application processors with MMU; run full OS like Linux or Android |
| Legacy ARM | ARM7TDMI, ARM9E, ARM11                                                  | Common before 2010; simpler pipeline and Thumb/ARM ISA mix         |

## Explain processor pipelines, and the pro/cons of shorter or longer pipelines.

A processor pipeline is like an assembly line for instructions. Instead of executing one instruction start-to-finish before starting the next, the CPU splits instruction execution into stages. Each stage does part of the work, and multiple instructions can be in different stages simultaneously.

| Concept             | Short Pipeline | Long Pipeline        |
| ------------------- | -------------- | -------------------- |
| Stages              | Few            | Many                 |
| Clock Frequency     | Lower          | Higher               |
| Instruction Latency | Low            | High                 |
| Throughput          | Moderate       | High (if no hazards) |
| Branch Penalty      | Small          | Large                |
| Complexity          | Simple         | Complex              |

### Hazards in Pipelines

- Data hazards: next instruction depends on the result of a previous one

  - Example: ADD R1,R2,R3 followed by SUB R4,R1,R5

- Control hazards: caused by branches or jumps

  - Long pipelines have bigger branch penalties

- Structural hazards:
  - Two instructions need the same hardware resource at the same time

### Optimizations for Longer Pipelines

- Branch prediction – guess branch outcome to reduce stalls
- Out-of-order execution – execute independent instructions early
- Speculative execution – pre-execute possible instruction paths
- Hazard forwarding / bypassing – pass results directly to dependent instructions

### Cortex-M3 Pipeline

- Processor type: ARMv7-M, 32-bit MCU
- Pipeline depth: 3 stages
- Instruction set: Thumb-2 (16-bit and 32-bit mixed instructions)

Target: Low-power, real-time embedded applications

#### Pipeline Stages (3-Stage)

The Cortex-M3 pipeline is relatively short — only 3 stages:

#### Stage Description

- Fetch (F) : Fetch instruction from Flash or memory. Cortex-M3 has prefetch and instruction buffer to reduce fetch stalls.
- Decode (D) : Decode instruction, generate control signals, and read registers from the register file.
- Execute (E) : ALU operations, address calculation, or branch evaluation. Also handles memory access and write-back.

#### Key difference from classic 5-stage ARM pipeline:

- Cortex-M3 merges memory access and write-back into the Execute stage.
- Optimized for low-latency, deterministic execution, suitable for embedded/real-time systems.

#### How Instructions Flow

- Fetch: Instruction fetched from memory or prefetch buffer.
- Decode: Instruction is decoded, operands read from registers.
- Execute: ALU executes operation or memory access; result written back to registers.
- Maximum throughput: 1 instruction per cycle in ideal conditions (pipelined)
- Branches are handled in the decode stage — low penalty due to short pipeline

#### Pipeline Features

- Thumb-2 support: Mix of 16-bit and 32-bit instructions for code density
- Single-cycle ALU operations: Many arithmetic and logic operations complete in 1 cycle
- Load/store forwarding: Reduces pipeline stalls for dependent instructions
- Low branch penalty: Usually 3 cycles or less, much lower than longer pipelines

## Explain fixed-point math. How do you convert a number into a fixed-point, and back again? Have you ever written any C functions or algorithms that used fixed-point math? Why did you?

https://www.youtube.com/watch?v=YXKDjVcCWyE

## What hardware debugging protocols are used to communicate with ARM microcontrollers?

| Protocol                           | Pins          | Speed                 | Use Case              | Notes                                                                                                                                       |
| ---------------------------------- | ------------- | --------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **JTAG (Joint Test Action Group)** | 4–5           | Moderate              | Testing + debugging   | Standard IEEE 1149.1 protocol; supports boundary scan, flash programming, and full debug control. Requires more pins.                       |
| **SWD (Serial Wire Debug)**        | 2             | Faster, less overhead | Debugging only        | ARM-specific 2-pin protocol (SWDIO + SWCLK); supports breakpoints, watchpoints, and single-step debugging. Less pin usage compared to JTAG. |
| **SWV (Serial Wire Viewer)**       | Uses SWD pins | High-speed trace      | Real-time data trace  | Often paired with SWD for performance monitoring, printf-style debug, or profiling. Requires Cortex-M with ITM support.                     |
| **SWO (Serial Wire Output)**       | 1             | High-speed trace      | Event & data output   | Part of SWV; outputs trace or debug events via a single pin.                                                                                |
| **cJTAG / ARM CoreSight**          | Varies        | High-speed trace      | Complex SoC debugging | Provides advanced tracing, performance counters, and cross-core debugging in multicore MCUs.                                                |

## What are the basic concepts of what happens before main() is called in C?

Before main() is called in a C program, there’s an important sequence of runtime initialization steps handled by the C runtime (CRT). This is crucial for embedded systems, operating systems, and bare-metal programming.

### Sequence Before main()

Startup/Reset

- On system reset, the processor fetches the reset vector (usually from Flash/ROM).
- Execution jumps to a startup routine (e.g., \_start in GCC or Reset_Handler in embedded systems).

Stack Pointer Initialization

- The stack pointer (SP) is initialized to the top of RAM.
- Local variables and function calls rely on the stack, so this must happen first.

Data Segment Initialization

- .data section (initialized global/static variables) is copied from Flash/ROM to RAM.

Example:

```c
int x = 42; // .data section
```

The CRT ensures x is correctly set in RAM before main().

BSS Segment Zeroing

- .bss section (uninitialized global/static variables) is zeroed in RAM.

Example:

```c
int y; // .bss section
```

This guarantees variables start at 0 before any code runs.

Runtime/Library Initialization

- On hosted systems (Linux/Windows), CRT may:
- Initialize standard I/O (stdin, stdout, stderr)
- Set up heap memory for dynamic allocation (malloc)
- Call constructor functions for global C++ objects (**attribute**((constructor)))

Call to main()

- After all initialization is complete, the startup routine calls:
  - int ret = main(argc, argv); is called
  - argc and argv are passed only in hosted environments (OS). In bare-metal embedded systems, main() often has no arguments.

Program Termination

- When main() returns:

  - On hosted systems, exit() is called to clean up.
  - On embedded systems, typically execution loops indefinitely or resets.

## Describe each of the following? SRAM, Pseudo-SRAM, DRAM, ROM, PROM, EPROM, EEPROM, MRAM, FRAM, ...

### SRAM (Static RAM)

- Full name: Static Random-Access Memory

- Characteristics:

  - Stores data using flip-flops (latches).
  - Does not need refresh as long as power is supplied.
  - Fast access time.

- Usage: CPU caches, small buffers in microcontrollers.
- Pros: Very fast, simple interface.
- Cons: Expensive, larger cell size, volatile.

### Pseudo-SRAM (PSRAM)

- Characteristics:

  - Appears as SRAM to the CPU but internally uses DRAM cells.
  - Requires refresh internally, but handled automatically.
  - Offers high density at lower cost compared to SRAM.

- Usage: Embedded graphics buffers, low-cost SRAM replacement.
- Pros: High density, cheaper than SRAM.
- Cons: Slightly slower than true SRAM, still volatile.

### DRAM (Dynamic RAM)

- Characteristics:

  - Stores data as charge in capacitors.
  - Requires periodic refresh to retain data.
  - High density and low cost per bit.

- Usage: Main memory in PCs, embedded systems with high RAM needs.
- Pros: High capacity, cost-effective.
- Cons: Slower than SRAM, needs refresh circuitry, volatile.

### ROM (Read-Only Memory)

- Characteristics:

  - Non-volatile memory; data is pre-programmed.
  - Cannot be modified during normal operation.

- Usage: Firmware storage, microcontroller boot code.
- Pros: Non-volatile, reliable.
- Cons: Cannot be changed (except in development).

### PROM (Programmable ROM)

- Characteristics:

  - Can be programmed once after manufacturing.
  - Uses fuse or antifuse technology.

- Usage: Custom firmware in small volumes.
- Pros: Non-volatile, customizable post-production.
- Cons: One-time programmable.

### EPROM (Erasable Programmable ROM)

- Characteristics:

  - Can be erased with UV light and reprogrammed.
  - Transparent quartz window exposes cells to UV for erasing.

- Usage: Development firmware, small-scale embedded devices.
- Pros: Reprogrammable, non-volatile.
- Cons: Erase requires UV lamp, slow.

### EEPROM (Electrically Erasable Programmable ROM)

- Characteristics:

  - Can be erased and rewritten electrically (byte-wise or page-wise).
  - Non-volatile.

- Usage: Configuration storage, calibration data in microcontrollers.
- Pros: Reprogrammable in-circuit, non-volatile.
- Cons: Limited write cycles (~10⁵–10⁶), slower than SRAM.

### MRAM (Magnetoresistive RAM)

- Characteristics:

  - Stores data using magnetic states of cells instead of charge.
  - Non-volatile, can be fast like SRAM.

- Usage: Embedded non-volatile memory, industrial, automotive.
- Pros: High endurance, fast, non-volatile.
- Cons: More expensive, limited capacity compared to DRAM.

### FRAM (Ferroelectric RAM)

- Characteristics:

- Uses ferroelectric capacitor polarization to store data.
- Non-volatile, fast, and low power.

- Usage: Embedded control systems, metering, wear-leveling data storage.
- Pros: Fast, very high endurance (~10¹² writes), non-volatile.
- Cons: Cost higher than EEPROM, lower density than DRAM.

## What is "wait state"?

A wait state is a CPU idle cycle (a delay) inserted automatically by hardware when the processor tries to access a slow memory or peripheral that cannot respond fast enough to the CPU clock speed.

## What are some common logic voltages?

| **Logic Family / Standard**           | **Nominal Supply Voltage (Vcc)** | **Logic HIGH (V<sub>IH</sub>)** | **Logic LOW (V<sub>IL</sub>)** | **Notes / Typical Use**                 |
| ------------------------------------- | -------------------------------- | ------------------------------- | ------------------------------ | --------------------------------------- |
| **TTL (Transistor-Transistor Logic)** | 5.0 V                            | ≥ 2.0 V                         | ≤ 0.8 V                        | Classic 5 V logic (e.g., 74LS series)   |
| **CMOS (5 V)**                        | 5.0 V                            | ≥ 3.5 V (≈ 0.7 × Vcc)           | ≤ 1.5 V (≈ 0.3 × Vcc)          | Legacy 4000-series CMOS                 |
| **LVCMOS33**                          | 3.3 V                            | ≥ 2.0 V                         | ≤ 0.8 V                        | Common for MCUs, FPGAs, SD cards        |
| **LVCMOS18**                          | 1.8 V                            | ≥ 1.2 V                         | ≤ 0.45 V                       | Used in low-power ICs and sensors       |
| **LVCMOS15**                          | 1.5 V                            | ≥ 1.05 V                        | ≤ 0.45 V                       | Mobile SoCs, DDR2 I/O                   |
| **LVCMOS12**                          | 1.2 V                            | ≥ 0.84 V                        | ≤ 0.36 V                       | DDR3 interfaces, ultra-low-power        |
| **LVCMOS10**                          | 1.0 V                            | ≥ 0.7 V                         | ≤ 0.3 V                        | DDR4, modern FPGAs                      |
| **LVTTL**                             | 3.3 V                            | ≥ 2.0 V                         | ≤ 0.8 V                        | TTL-compatible 3.3 V logic              |
| **Open-drain / Open-collector**       | Depends on pull-up voltage       | –                               | –                              | Shared bus (e.g., I²C, interrupt lines) |

## What are some common logic families ?

| **Logic Family**                          | **Technology**            | **Typical Voltage (Vcc)** | **Key Characteristics**                          | **Notes / Examples**                            |
| ----------------------------------------- | ------------------------- | ------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| **RTL (Resistor-Transistor Logic)**       | Bipolar                   | 5 V                       | Simple, very slow, high power                    | Early logic (1960s), obsolete                   |
| **DTL (Diode-Transistor Logic)**          | Bipolar                   | 5 V                       | Faster than RTL                                  | Predecessor to TTL                              |
| **TTL (Transistor-Transistor Logic)**     | Bipolar                   | 5 V                       | Moderate speed, moderate power                   | 74xx series; robust, noisy but tolerant         |
| **LS-TTL (Low Power Schottky TTL)**       | Bipolar + Schottky diodes | 5 V                       | Faster and lower power than standard TTL         | 74LSxx series                                   |
| **ALS-TTL (Advanced Low Power Schottky)** | Bipolar                   | 5 V                       | Even lower power, higher fan-out                 | 74ALSxx series                                  |
| **HC (High-speed CMOS)**                  | CMOS                      | 2–6 V (typically 5 V)     | High speed, low power                            | 74HCxx series                                   |
| **HCT (High-speed CMOS TTL-compatible)**  | CMOS                      | 5 V                       | TTL-level compatible inputs                      | 74HCTxx series — used in mixed TTL/CMOS systems |
| **AC (Advanced CMOS)**                    | CMOS                      | 2–6 V                     | Very fast, low power                             | 74ACxx series                                   |
| **LVCMOS (Low Voltage CMOS)**             | CMOS                      | 1.8–3.3 V                 | High speed, low power, modern MCUs               | Common in FPGAs, SoCs, STM32                    |
| **LVTTL (Low Voltage TTL)**               | Bipolar/CMOS              | 3.3 V                     | TTL thresholds, lower Vcc                        | Used in 3.3 V systems compatible with older TTL |
| **ECL (Emitter-Coupled Logic)**           | Bipolar                   | –5.2 V                    | Extremely fast, high power                       | Used in high-speed systems (GHz range)          |
| **PECL / LVPECL (Positive ECL)**          | Bipolar                   | +5 V / +3.3 V             | Differential signaling, very fast                | Used in clock, RF, and high-speed comms         |
| **CML (Current Mode Logic)**              | Bipolar                   | 3.3 V or less             | High-speed differential                          | Common in SerDes, gigabit interfaces            |
| **BiCMOS**                                | Bipolar + CMOS hybrid     | 3–5 V                     | Combines speed of bipolar with low power of CMOS | Used in high-performance mixed-signal ICs       |

## What is a CPLD? an FPGA? Describe why they might be used in an embedded system?

### What is a CPLD (Complex Programmable Logic Device)?

A CPLD is a type of programmable logic chip that can implement custom digital logic — but is simpler and smaller than an FPGA.

#### Key Characteristics:

| Feature           | Description                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| **Structure**     | Built from a small number of logic blocks (macrocells) connected by a predictable interconnect. |
| **Configuration** | Non-volatile — logic remains programmed even after power-off (Flash or EEPROM-based).           |
| **Speed**         | Deterministic and fast — good for glue logic or control applications.                           |
| **Capacity**      | Small — typically a few hundred to a few thousand logic gates.                                  |

#### Typical Uses:

- Replacing glue logic (AND, OR, NAND gates, decoders)
- Bus decoding or address mapping
- Simple state machines
- Custom peripheral logic (chip selects, wait-state generators)
- Bridging between different logic families or interfaces

#### Example Devices:

- Xilinx XC9500 series
- Altera (Intel) MAX 7000 series
- Lattice ispMACH series

### What is an FPGA (Field Programmable Gate Array)?

An FPGA is a much larger, more flexible programmable logic device, capable of implementing complex digital systems, even entire processors or signal-processing pipelines.

#### Key Characteristics

| Feature            | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| **Structure**      | Array of configurable logic blocks (CLBs) + programmable interconnect + I/O blocks. |
| **Configuration**  | Usually volatile (SRAM-based) — must be reloaded from Flash or CPU on every boot.   |
| **Capacity**       | Very large — thousands to millions of logic gates.                                  |
| **Speed**          | Very high, parallel execution (hardware concurrency).                               |
| **Reconfigurable** | Can be reprogrammed on the fly (partial reconfiguration possible).                  |

#### Typical Uses:

- High-speed digital signal processing (DSP)
- Custom hardware accelerators (e.g., AI/ML preprocessing)
- High-speed communication interfaces (PCIe, Ethernet, DDR)
- Soft-core CPUs (e.g., MicroBlaze, Nios II)

#### Real-time control systems

- Prototyping ASIC designs

#### Example Devices:

- Xilinx Artix, Spartan, Kintex, Virtex
- Intel (Altera) Cyclone, Arria, Stratix
- Lattice iCE40, ECP5

### CPLD vs FPGA

| Feature                  | **CPLD**                          | **FPGA**                                               |
| ------------------------ | --------------------------------- | ------------------------------------------------------ |
| **Logic Capacity**       | Low (hundreds of gates)           | Very high (millions of gates)                          |
| **Configuration Memory** | Non-volatile (Flash/EEPROM)       | Usually volatile (SRAM)                                |
| **Startup Time**         | Instant (no configuration load)   | Requires configuration (milliseconds)                  |
| **Power Consumption**    | Low                               | Moderate to high                                       |
| **Speed / Determinism**  | Very predictable timing           | More complex routing, can vary                         |
| **Cost**                 | Low                               | Medium to high                                         |
| **Best For**             | Simple logic, control, glue logic | Complex signal processing, SoCs, high-speed interfaces |

### Why use them in embedded systems

| Use Case                                          | CPLD                       | FPGA                                              |
| ------------------------------------------------- | -------------------------- | ------------------------------------------------- |
| **Custom peripheral interface**                   | ✅ Excellent (fixed logic) | ✅ Works well, but overkill for small logic       |
| **Timing-critical I/O control**                   | ✅ Deterministic timing    | ✅ Higher precision via parallel logic            |
| **Offloading CPU tasks**                          | ⚠️ Limited                 | ✅ Very effective for parallel workloads          |
| **Prototyping or reconfigurable logic**           | ✅ Reprogrammable          | ✅ Dynamically reconfigurable                     |
| **Bridging between buses (e.g., SPI ↔ Parallel)** | ✅ Simple and reliable     | ✅ Good for high-speed or complex bridges         |
| **Signal processing (filters, FFT, etc.)**        | ❌ Not suitable            | ✅ Ideal — can process multiple samples per clock |
| **Boot-time readiness (instant on)**              | ✅ Non-volatile            | ⚠️ Needs configuration ROM or MCU boot assist     |

### Example Embedded Use Cases

| Application                              | Why CPLD/FPGA is Used                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| **Motor control system**                 | FPGA handles high-speed PWM generation and feedback, MCU runs control loop. |
| **Custom memory interface**              | CPLD generates chip selects, ready/busy, and address decoding.              |
| **IoT gateway or industrial controller** | CPLD for deterministic I/O timing, MCU for networking stack.                |
| **Edge AI device**                       | FPGA accelerates neural network inference while MCU handles data handling.  |
| **High-speed data acquisition**          | FPGA performs parallel sampling and pre-processing before DMA to CPU.       |

## List some types of connectors found on test equipment.

| **Category**            | **Typical Connectors**                |
| ----------------------- | ------------------------------------- |
| **RF / Signal**         | BNC, SMA, N-Type, TNC, Triax          |
| **Power / DC**          | Banana, Binding Post, SHV, MHV        |
| **Digital / Control**   | USB, Ethernet, GPIB, RS-232 (DB-9/25) |
| **Optical**             | SC, LC, ST                            |
| **Video / Display**     | HDMI, DisplayPort                     |
| **Modular / Backplane** | PXI, PCIe, VXI                        |

## What is AC? What is DC? Describe the voltage in the wall outlet? Describe the voltage in USB 1.x and 2.x cables ?

### What is AC (Alternating Current)?

| Feature              | Description                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Definition**       | Current that **changes direction periodically**.                                                                                        |
| **Voltage waveform** | Typically **sinusoidal (sine wave)** — voltage alternates between positive and negative values.                                         |
| **Used in**          | Power transmission, wall outlets, motors, household appliances.                                                                         |
| **Reason for use**   | AC is **easier to transmit over long distances** and can be **transformed** (via transformers) to higher or lower voltages efficiently. |

### What is DC (Direct Current)?

| Feature              | Description                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| **Definition**       | Current flows in **one constant direction**.                                             |
| **Voltage waveform** | Constant over time (flat line on an oscilloscope).                                       |
| **Used in**          | Batteries, USB devices, microcontrollers, LED circuits.                                  |
| **Reason for use**   | DC is ideal for **low-voltage electronics** and devices that need a steady power supply. |

### Voltage in USB 1.x and 2.x

| USB Version       | Nominal Voltage | Max Current                        | Power Output | Type of Current |
| ----------------- | --------------- | ---------------------------------- | ------------ | --------------- |
| **USB 1.0 / 1.1** | **5.0 V DC**    | 0.5 A                              | 2.5 W        | DC              |
| **USB 2.0**       | **5.0 V DC**    | 0.5 A (standard), 1.5 A (charging) | 2.5–7.5 W    | DC              |

## What is RS232? RS432? RS485? MIDI? What do these have in common?

| Standard   | Type   | Signal Method | Devices          | Max Distance | Typical Speed | Example Use     |
| ---------- | ------ | ------------- | ---------------- | ------------ | ------------- | --------------- |
| **RS-232** | Serial | Single-ended  | 1-to-1           | ~15 m        | 115 kbps      | PC COM port     |
| **RS-422** | Serial | Differential  | 1-to-1           | 1200 m       | 10 Mbps       | Encoders        |
| **RS-485** | Serial | Differential  | Multi-drop       | 1200 m       | 10 Mbps       | Modbus, sensors |
| **MIDI**   | Serial | Current loop  | 1-to-1 (usually) | 15 m         | 31.25 kbps    | Musical gear    |

## What is ISO9001? What is a simple summary of it's concepts?

ISO 9001 is an international standard for Quality Management Systems (QMS), published by the International Organization for Standardization (ISO).

It defines a framework that organizations can follow to ensure their products and services consistently meet customer and regulatory requirements, while continuously improving their processes.

ISO 9001 is about “Say what you do, do what you say, and continuously improve.”

## What is A/D? D/A? OpAmp? Comparator Other Components Here? Describe each. What/when might each be used?

| Device          | Converts                     | Output Type             | Example Use             |
| --------------- | ---------------------------- | ----------------------- | ----------------------- |
| **ADC (A/D)**   | Analog → Digital             | Digital value           | Read temperature sensor |
| **DAC (D/A)**   | Digital → Analog             | Voltage/current         | Generate audio signal   |
| **Op-Amp**      | Analog → Amplified Analog    | Analog voltage          | Amplify sensor output   |
| **Comparator**  | Analog → Digital (threshold) | Logic signal            | Detect voltage limit    |
| **Voltage Ref** | —                            | Stable analog reference | ADC/DAC accuracy        |
| **MUX**         | Many analog → One            | Selected analog line    | Switch sensor inputs    |

## Have you ever used any encryption algorithms? Did you write your own from scratch or use a library (which one)? Describe which type of algorithms you used and in what situations you used them?

- AES-128 (Advanced Encryption Standard),

  - Symmetric: The same key is used for encryption and decryption.
  - Block cipher: Operates on fixed-size blocks of data (128 bits for AES).

- SHA-256 (Secure Hash Algorithm)
  - Takes an input (message) of arbitrary length and produces a fixed-length output

### mbedTLS library

## What is a CRC algorithm? Why would you use it? What are some CRC algorithms? What issues do you need to worry about when using CRC algorithms that might cause problems? Have you ever written a CRC algorithm from scratch?

### What is a CRC?

CRC stands for Cyclic Redundancy Check.
It is a hash-like algorithm used to detect errors in digital data during storage or transmission.

CRC is not encryption — it’s purely for error detection.

It treats the data as a polynomial over a finite field (GF(2)) and computes a remainder after division by a predefined generator polynomial.

The remainder is called the CRC checksum.

Think of CRC as a “fingerprint” for a data block. If the data changes (even a single bit), the CRC is very likely to change.

### Why use CRC

| Purpose                     | Example                             |
| --------------------------- | ----------------------------------- |
| Detect transmission errors  | UART, SPI, CAN bus, Ethernet frames |
| Detect storage corruption   | Flash memory, SD cards              |
| Ensure firmware integrity   | Before booting or executing code    |
| Lightweight error detection | Faster than cryptographic hashes    |

### Common CRC Algorithm

| CRC Variant   | Width  | Typical Use               | Polynomial             |
| ------------- | ------ | ------------------------- | ---------------------- |
| CRC-8         | 8-bit  | Small embedded frames     | 0x07 (x⁸ + x² + x + 1) |
| CRC-16-CCITT  | 16-bit | Modems, Bluetooth, XMODEM | 0x1021                 |
| CRC-16-MODBUS | 16-bit | Industrial protocols      | 0x8005                 |
| CRC-32        | 32-bit | Ethernet, ZIP, PNG        | 0x04C11DB7             |
| CRC-64        | 64-bit | Storage systems           | 0x42F0E1EBA9EA3693     |

### Pitfalls

| Issue                       | Explanation / Risk                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| **Endianness**              | Sending CRC LSB/MSB incorrectly can cause false failures                                      |
| **Polynomial mismatch**     | Using a different generator polynomial than the receiver breaks verification                  |
| **Bit-order mismatch**      | Some protocols reverse bits before CRC calculation                                            |
| **Padding / framing**       | Including or excluding headers incorrectly can produce wrong CRC                              |
| **Limited error detection** | CRC detects **accidental errors**, but not malicious tampering (not cryptographically secure) |

## What is the "escape character" for "Epson ESC/P"? Where is this used?

It signals the start of a printer command sequence. Anything following the ESC character is interpreted as a command, not normal text.

## Have you ever written a RAM test from scratch? What are some issues you need to test?

| Fault Type                  | Description                                    |
| --------------------------- | ---------------------------------------------- |
| **Stuck-at faults**         | Bit stuck at 0 or 1                            |
| **Coupling faults**         | Writing one bit affects a neighboring bit      |
| **Address decoding faults** | Wrong address maps to same location as another |
| **Retention faults**        | Memory loses data over time                    |
| **Read/write disturb**      | Repeated writes affect nearby memory           |

### Simple C example Walking 1

```c
int ram_test_walking1(uint32_t *start, uint32_t *end) {
    for (uint32_t *addr = start; addr < end; addr++) {
        for (uint32_t bit = 0; bit < 32; bit++) {
            uint32_t pattern = 1U << bit;
            *addr = pattern;
            if (*addr != pattern) return 0; // fail
        }
    }
    return 1; // pass
}


```

## What issues concerns software when you WRITE a value to EEPROM memory? FLASH memory?

### EEPROM

| Concern                    | Explanation                                                                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Write endurance / wear** | Typical EEPROM cells can handle **10^5 – 10^6 writes**. Repeated writes to the same address may wear it out. Software should **spread writes** (wear leveling) or **limit write frequency**. |
| **Write time / blocking**  | EEPROM writes are slower (ms range). Firmware should **avoid blocking critical tasks** during writes; use non-blocking or asynchronous mechanisms if possible.                               |
| **Atomicity / power loss** | If power is lost during a write, the data may be corrupted. Use **checksums, flags, or redundant storage** to ensure data integrity.                                                         |
| **Addressing / alignment** | Some EEPROM devices require **aligned writes** or have **page sizes**; software must respect these boundaries.                                                                               |
| **Read-after-write**       | Often a write must be **followed by a read** to verify success.                                                                                                                              |

### Flash Memory

| Concern                              | Explanation                                                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Erase-before-write**               | FLASH cannot overwrite bits from 0 → 1 directly. Must **erase entire page or sector** before writing. Software must manage **erase cycles** carefully. |
| **Write/erase endurance**            | FLASH cells have **10^4 – 10^6 write/erase cycles**. Critical data should use **wear leveling** or move to EEPROM if frequent updates needed.          |
| **Block size / alignment**           | Erase operations occur on **sectors/pages**, not single bytes. Software must handle **buffering and alignment**.                                       |
| **Write/erase time**                 | FLASH writes/erases are slower (ms to tens of ms). Avoid blocking critical real-time tasks.                                                            |
| **Power loss / corruption**          | Interrupting a write/erase can **corrupt the entire sector**. Use **dual-buffering, shadow copies, or checksums** to protect data.                     |
| **Volatile vs non-volatile caching** | Some MCUs cache writes in RAM before committing to FLASH; ensure **cache flush** if needed.                                                            |

## Conceptually, what do you need to do after reconfiguring a digital PLL? What if the digital PLL sources the clock for your microcontroller (and other concerns)?

- Conceptual Steps After Reconfiguring a PLL

  - When you change the PLL configuration (e.g., multiply/divide ratio, source, or bypass):

- Wait for Lock

  - The PLL output is unstable immediately after reconfiguration.
  - Must wait until the PLL locks to the new frequency. Most MCUs have a PLL lock status bit in a control register.

- Update Clock Dividers / Prescalers

  - If peripherals or core clocks depend on the PLL, ensure that prescalers/dividers are updated to maintain timing requirements.

- Check Peripheral Timing Constraints

  - Some peripherals (UART, SPI, ADC, timers) depend on exact frequencies. After PLL reconfiguration, recalculate any timing registers.

- Switch System Clock Safely

  - Only switch the MCU core/system clock to the new PLL output after the PLL is stable.

- Handle Glitches / Propagation Delays

  - Some MCUs allow bypassing the PLL temporarily to a safe clock (e.g., internal RC oscillator) while PLL locks.
  - This avoids glitches that could crash the MCU or peripherals.

## What is "duff's device"? Have you ever used it?

Duff’s Device is a clever C idiom combining switch-case and loop unrolling to optimize repeated operations, such as copying buffers to memory-mapped I/O.

## What is dual-port RAM? Why would it be useful in some embedded systems? What concerns do you need to worry about when using it? Have you ever used it? How?

Dual-port RAM allows two devices or processors to access the same memory concurrently, which is useful for high-speed MCU ↔ FPGA communication or DMA-based buffering in embedded systems. When using it, you need to handle possible write collisions, maintain data consistency, and respect timing constraints. I have used dual-port RAM in a project where an MCU wrote sensor data while an FPGA read it simultaneously for real-time processing, using a handshake line to indicate valid data and avoid conflicts.”

## What are virtual and physical addresses? What is a MMU?

### Physical vs Vertual Addresses

| Term                 | Definition                                                                                        | Notes / Example                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Physical Address** | The actual location in **RAM or hardware memory**.                                                | E.g., 0x2000_0000 may be the physical location of a variable in DRAM.              |
| **Virtual Address**  | The **logical address** used by a program. It is translated to a physical address by the **MMU**. | E.g., a program may access 0x0040_0000, but the MMU maps it to 0x2000_0000 in RAM. |

A physical address is the actual memory location in hardware, while a virtual address is the logical address used by software. The MMU (Memory Management Unit) is a hardware unit that translates virtual addresses to physical addresses, provides memory protection, and supports features like paging and caching. On embedded systems with an MMU, programs mostly use virtual addresses, and the MMU ensures they map correctly to RAM or peripherals

## Describe different types of Cache. When do you need to flush the cache, when to invalidate cache?

Caches are small, fast memory units that store recently used data or instructions to reduce CPU access time to slower main memory (RAM).

| **Cache Type**          | **Description**                                                                           | **Notes / Usage**                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **L1 Cache**            | Closest to CPU core; usually split into **Instruction (I-Cache)** and **Data (D-Cache)**. | Fastest and smallest (16–128 KB).                                 |
| **L2 Cache**            | Next level; often unified (stores both instructions and data).                            | Larger (128 KB–1 MB), slower than L1.                             |
| **L3 Cache**            | Shared among CPU cores in multicore systems.                                              | Large (1–32 MB), improves inter-core data sharing.                |
| **Unified Cache**       | Combines instructions and data into one pool.                                             | Simplifies hardware, slightly less performance than split caches. |
| **Write-through Cache** | Writes go both to cache and main memory immediately.                                      | Slower writes but always consistent with RAM.                     |
| **Write-back Cache**    | Writes stay in cache and only go to main memory when that line is evicted.                | Faster writes, but needs software management (flush).             |

### Cache Maintenance Operations

#### Flush (Clean) Cache

- Definition: Write any modified (dirty) cache lines back to main memory.

- When to do it:

  - When the CPU modifies data that peripherals (DMA, GPU, etc.) will read.
  - Before giving memory ownership to another bus master.

- Why: Ensures that main memory has the latest data from the CPU.

- Example (pseudocode):

  - flush_cache(buffer, size); // Write updated cache data to RAM

#### Invalidate Cache

- Definition: Mark cache lines as invalid so that subsequent reads fetch data from main memory.

- When to do it:

  - When peripherals or DMA have written data to memory that the CPU will read.
  - After receiving data from another processor or device.

- Why: Prevents the CPU from using stale data from cache.

- Example (pseudocode):

  - invalidate_cache(buffer, size); // Discard old cached data

#### Flush + Invalidate

- Definition: Write dirty lines to memory and mark them invalid.

- When to do it:

  - When both CPU and peripherals might modify shared data.
  - To ensure complete consistency after DMA operations.

## What is SIMD?

Definition

SIMD stands for Single Instruction, Multiple Data, a form of parallel processing where one instruction operates on multiple pieces of data simultaneously.

How It Works

Traditional (scalar) processors:
C[i] = A[i] + B[i]; → executes one addition per instruction.

SIMD processor:
Performs multiple additions at once — for example, four 32-bit additions using a single 128-bit instruction.

## What is a Mailbox register?

A Mailbox register is a hardware communication mechanism used to exchange data or messages between two or more processors, cores, or subsystems — for example, between a CPU and a microcontroller, or a host processor and a co-processor / peripheral (like a GPU, DSP, or FPGA).

It acts like a shared message slot in hardware memory — one side writes a message, the other side reads it.

## What is a Cacheline?

- A cache line (also called a cache block) is the smallest unit of data that can be transferred between main memory (RAM) and the CPU cache.

- It represents a fixed-size chunk of contiguous memory that the cache stores, rather than individual bytes or words.

- When a CPU accesses a memory address that isn’t in the cache (a cache miss), it doesn’t fetch just that byte or word — it fetches an entire cache line.

- Larger cache lines exploit spatial locality.

- Invalidate a cache line before DMA reads memory (to avoid stale data).
- Flush a cache line after DMA writes data (to ensure it reaches RAM).

## What is WCET and where does it matter?

WCET stands for Worst-Case Execution Time —
it is the maximum time a piece of code (function, task, or interrupt) can take to complete execution on a specific processor, under the worst possible conditions.

### How WCET is determined

| **Method**            | **Description**                                           | **Pros / Cons**                                                  |
| --------------------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| **Static analysis**   | Analyze compiled code and instruction timing using models | ✅ Deterministic, no runtime needed<br>❌ Conservative estimates |
| **Measurement-based** | Run tests under stress conditions and measure execution   | ✅ Practical<br>❌ May miss corner cases                         |
| **Hybrid approach**   | Combine static and measurement methods                    | ✅ More realistic balance                                        |

## What is Lockstep execution?

Lockstep execution is a technique where two or more processors (or cores) execute the same instructions simultaneously in parallel and continuously compare their results to detect errors. If the outputs diverge, the system detects a fault immediately.

This is commonly used in redundant systems where safety and reliability are critical.

### Example Scenario

- Automotive ECU running engine control software:
- Two lockstep cores calculate fuel injection timing simultaneously.
- If one core fails due to a hardware fault, the mismatch is detected.
- Safety mechanism prevents sending wrong signals to the engine.

### Benefits

- High reliability in safety-critical applications (ISO 26262 for automotive, DO-178C for aerospace).
- Detects single-event upsets (SEUs) and transient hardware faults.
- Can be combined with ECC memory and watchdogs for even higher fault coverage.

## What does static and what does const expressions mean?

| Keyword           | Affects    | Lifetime         | Scope    | Mutability | Common Use                   |
| ----------------- | ---------- | ---------------- | -------- | ---------- | ---------------------------- |
| `static` (local)  | Storage    | Entire program   | Function | Mutable    | Preserve value between calls |
| `static` (global) | Linkage    | Entire program   | File     | Mutable    | Hide from other files        |
| `const`           | Mutability | Same as variable | Same     | Immutable  | Prevent accidental writes    |
| `static const`    | Both       | Entire program   | File     | Immutable  | File-local constants         |

## What kind of variables would you store on stack and why?

The stack is a region of memory used for:

- Function call management (return addresses)
- Local (automatic) variables
- Function parameters
- Temporary storage

It operates on a LIFO (Last In, First Out) principle — when a function is called, its local context (called a stack frame) is pushed onto the stack; when it returns, it is popped off.

## Types of interrupt in linux

| Category               | Trigger         | Runs In          | Context           | Can Sleep? | Example                |
| ---------------------- | --------------- | ---------------- | ----------------- | ---------- | ---------------------- |
| **Hardware Interrupt** | External signal | Kernel           | Interrupt         | ❌         | Keyboard, network card |
| **Software Interrupt** | Syscall, fault  | Kernel           | Process/Interrupt | ⚠️ Depends | `int 0x80`, page fault |
| **SoftIRQ**            | Deferred task   | Kernel           | Interrupt         | ❌         | Networking             |
| **Tasklet**            | Deferred task   | Kernel           | Interrupt         | ❌         | Driver cleanup         |
| **Workqueue**          | Deferred task   | Kernel           | Process           | ✅         | Low-priority work      |
| **NMI**                | Hardware fault  | CPU special mode | Interrupt         | ❌         | Watchdog, ECC error    |
| **IPI**                | Other CPU       | Kernel           | Interrupt         | ❌         | SMP synchronization    |

```c
  ┌─────────────────────────────┐
  │      Hardware Device        │
  │ (e.g., Timer, Network, ADC) │
  └─────────────┬───────────────┘
                │ IRQ Line / Signal
                ▼
  ┌─────────────────────────────┐
  │ Interrupt Controller (PIC / │
  │      APIC / GIC)            │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │   CPU acknowledges IRQ      │
  │ Disables interrupt line     │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │      Top-Half ISR           │
  │  (Immediate, fast handler)  │
  │  Cannot sleep; minimal work │
  └─────────────┬───────────────┘
                │
                ▼
      ┌───────────────────┐
      │ Schedule Bottom   │
      │ Half / Deferred   │
      └─────────┬─────────┘
                │
        ┌───────┴────────┐
        │ Deferred Work  │
        │  Options:      │
        │  - SoftIRQ     │
        │  - Tasklet     │
        │  - Workqueue   │
        └───────┬────────┘
                │
        ┌───────┴────────┐
        │ Process Context│
        │  Can sleep     │
        │  Perform heavy │
        │  tasks         │
        └────────────────┘

```

## Explain what happen in firmware update, how the microcontroller loads new firmware ?

A firmware update replaces or modifies the program stored in non-volatile memory (Flash) on a microcontroller (MCU).

Firmware updates may be delivered in any of the following ways:

- Over a serial link (UART, USB)
- Over-the-air (OTA) via Wi-Fi, BLE, or CAN
- From external storage (SD card, EEPROM, etc.)

The new firmware is written into the MCU’s Flash memory, which stores the executable code run after reset.

### Steps

#### Step 1: Enter Bootloader Mode

- MCU starts executing from reset vector (often in bootloader Flash).

- Bootloader checks:

  - Is there a firmware update request ?
  - Is a valid application already present ?
  - If update requested → continue.
  - Else → jump to application.

#### Step 2: Receive the New Firmware Image

- Bootloader receives firmware data via communication interface:

  - UART, USB, CAN, SPI, I²C, Ethernet, BLE, or OTA.
  - Data often sent in packets (with CRC or checksum).

- Common protocols:

  - XMODEM, YMODEM, SREC, DFU, custom packet protocol.

#### Step 3: Erase Flash Memory

- Before writing, the target Flash sectors must be erased:

  - Erase → Write → Verify

- Flash can only change bits from 1 → 0.
- To reset bits back to 1, an erase operation is required.

#### Step 4: Write New Firmware to Flash

- Bootloader writes firmware data page-by-page.
- Each page (e.g., 1 KB or 2 KB) is programmed using MCU’s Flash programming API.

- Typical sequence:

  - HAL_FLASH_Unlock();
  - HAL_FLASH_Program(...);
  - HAL_FLASH_Lock();

or direct register access in low-level drivers.

#### Step 5: Verify the Written Data

- After writing, bootloader performs integrity verification:

  - CRC, Checksum, or digital signature (for secure update)

- Compare against expected value from update image
- If verification fails → erase partial image and retry / revert.

#### Step 6: Jump to the New Application

- If the update passes:

  - Bootloader sets the vector table offset (VTOR) to new app start address.
  - Loads Main Stack Pointer (MSP) from address APP_START.
  - Jumps to the Reset Handler of new firmware.

## Explain compilation chain

Compilation (Build) Chain for Cortex-M

When you write firmware in C/C++, the toolchain converts your source code into a binary executable that the Cortex-M CPU can run.
The major steps are:

```c
Source Code (.c / .h)
     ↓
Preprocessing
     ↓
Compilation
     ↓
Assembly
     ↓
Linking
     ↓
Binary / Hex file
     ↓
Programming into Flash
```

Each stage is handled by a specific tool in the ARM GCC toolchain (arm-none-eabi) or vendor toolchains like Keil IAR EWARM etc.

### Preprocessing

- Tool: arm-none-eabi-gcc -E

- The C preprocessor runs first.
  - Expands all #include, #define, and conditional compilation (#ifdef, etc.).
  - Removes comments and outputs a single expanded .i file.

Example:

```bash
arm-none-eabi-gcc -E main.c -o main.i
```

### Compilation (C → Assembly)

- Tool: arm-none-eabi-gcc -S

  - The compiler converts high-level C code into ARM assembly instructions (for Cortex-M instruction set).
  - Produces .s file (assembly).

Example:

```bash
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -S main.i -o main.s
```

Common Flags:

- -mcpu=cortex-m4 → Target CPU
- -mthumb → Generate Thumb-2 instructions (Cortex-M uses Thumb only)
- -O0, -O1, -O2, -O3 → Optimization level

### Assembly (Assembly → Object Code)

- Tool: arm-none-eabi-as
  - The assembler translates assembly into machine code.
  - Output: .o (object file)
  - Each .c file becomes a separate .o file.

Example:

```bash
arm-none-eabi-as main.s -o main.o
```

### Linking

- Tool: arm-none-eabi-ld (or via GCC)

  - Combines all object files (main.o, startup.o, drivers.o) and libraries (libc, libm, etc.) into one ELF executable.

  - Uses a linker script (.ld) to define memory layout:

- Flash memory regions

- RAM

- Stack, heap, and section addresses

Example:

```bash
arm-none-eabi-gcc main.o startup.o -Tstm32f4xx.ld -o firmware.elf
```

Example .ld (Linker Script) Snippet

```c
MEMORY
{
FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
.text : { _(.text_) } > FLASH
.data : { _(.data_) } > RAM AT > FLASH
.bss : { _(.bss_) } > RAM
}
```

This tells the linker where to place each code/data section in memory.

### Object Copy (ELF → BIN / HEX)

- Tool: arm-none-eabi-objcopy

  - Converts the ELF file (which contains debug symbols) into a binary or Intel HEX file for flashing.

Examples:

```bash
arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
arm-none-eabi-objcopy -O ihex firmware.elf firmware.hex
```

### Flashing to Microcontroller

- Tool: Depends on hardware

  - ST-Link Utility, OpenOCD, J-Link, PyOCD, etc.

  - Writes firmware.bin or .hex into MCU Flash via SWD/JTAG.

Example:

```bash
st-flash write firmware.bin 0x08000000
```

### Execution (Runtime)

When the MCU resets:

- PC loads from address 0x08000004 (reset vector).
- SP (Stack Pointer) initialized from 0x08000000.
- CPU jumps to Reset_Handler (defined in startup.s).

Reset_Handler:

- Copies .data from Flash → RAM.
- Zeros .bss.
- Calls main().

## How to make sure data integrity in a data transfer ?

- Checksums – Simple arithmetic sum of data bytes; detects basic errors.
- CRC (Cyclic Redundancy Check) – Polynomial-based error detection for stronger reliability.
- Parity Bit – Adds 1 bit per byte/word to detect single-bit errors.
- Hashing (e.g., MD5, SHA-256) – Ensures integrity of large data blocks or files.
- Sequence Numbers – Detects missing, duplicated, or out-of-order packets.
- ACK/NACK Protocols – Receiver confirms correct data reception.
- ECC (Error Correction Codes) – Detects and corrects bit errors (used in memory, storage).
- Redundant Transmission – Re-send data multiple times for comparison.
- Digital Signatures – Combines hashing and encryption for integrity + authenticity.
- Timeouts & Retransmission Logic – Detects lost or corrupted data links.
- Encryption with Integrity Check (e.g., AES-GCM) – Provides confidentiality + built-in integrity.
- Versioning / Timestamps – Ensures latest valid data is used.

## Explain Structure Padding in embedded C.

Structure Padding in embedded C is the automatic insertion of unused bytes by the compiler to align structure members according to the target architecture’s alignment rules (typically word-aligned, e.g., 4 bytes).

### Why it happens

Most processors access data faster when it’s aligned to word boundaries.

The compiler adds padding between members or at the end of the structure to maintain this alignment.

#### Example

```c
struct Example {
    char a;   // 1 byte
    int  b;   // 4 bytes
};
// Size may be 8 bytes (3 bytes padding after 'a')
```

#### To control padding

Use #pragma pack(n) or **attribute**((packed)) to reduce or remove padding (with performance trade-offs).

## How to sign firmware during update (code signing)

To ensure the authenticity (from trusted source) and integrity (not tampered) of firmware before the microcontroller installs it.

### Steps

- Generate Key Pair

  - Developer creates a private key (kept secret) and a public key (embedded in device).

- Sign Firmware

  - Compute a hash (e.g., SHA-256) of the firmware image.
  - Encrypt the hash using the private key → produces a digital signature.

- Attach Signature

  - Signature (and optional certificate) is appended to the firmware image.

- Verify on Device

  - Device computes its own hash of the received firmware.

- Decrypts the attached signature using the public key.

  - If both hashes match → firmware is authentic and untampered.

- Install Firmware

  - Only after successful verification. Otherwise, reject or rollback.

## How fast UART receiver need to sample ?

A UART receiver typically needs to sample the incoming data line at 16× the baud rate.

### Reason

- UART is asynchronous (no clock line).
- Receiver must detect bit timing by oversampling the line.
- 16× oversampling allows it to:

  - Detect the start bit edge precisely.
  - Sample each bit at its midpoint for best accuracy.
  - Tolerate small clock mismatches (±2–3%).

UART receiver should sample at least 16 times the baud rate for reliable data reception.

## Discuss a CPU profiler

A CPU profiler is a tool that measures how much CPU time is spent in different parts of a program.

Helps identify performance bottlenecks.

Shows which functions consume the most processing resources.

- Linux / Desktop: gprof, perf, valgrind, oprofile
- Embedded / MCU: Segger SystemView, ARM Streamline, SWD-based Trace + ETM

## Cache coherency, virtual memory, paging, MMU, PCIe

### Cache Coherency

- Definition: Ensures that multiple CPU caches see a consistent view of memory.
- Problem: When multiple cores or DMA devices modify the same memory, caches may become stale.
- Solution: Coherency protocols like MESI or explicit cache flush/invalidate operations.

### Virtual Memory

- Definition: Abstraction of physical memory; each process sees its own continuous address space.

- Benefits:

  - Process isolation
  - Memory protection
  - Simplified memory allocation

### Paging

- Definition: Virtual memory is divided into fixed-size pages (commonly 4 KB).

- Mechanism: CPU + MMU maps virtual pages → physical frames.
- Advantages: Efficient memory usage, swap support, avoids fragmentation.

### MMU (Memory Management Unit)

- Definition: Hardware block that translates virtual addresses → physical addresses.

- Functions:
  - Address translation
  - Access permission checks (read/write/execute)
  - Cache attributes for memory regions

### PCIe (Peripheral Component Interconnect Express)

- Definition: High-speed serial interface for connecting peripherals (e.g., GPUs, NICs).

- Key Points:

  - Point-to-point lanes, full-duplex
  - Supports DMA, memory-mapped I/O
  - Scalable bandwidth: x1, x4, x8, x16 lanes

## Code testing methods (how will you test this code)

## What embedded RTOS have you used? Have you ever written your own from scratch?

## Have you ever implemented from scratch any functions from the C Standard Library (that ships with most compilers)? Created your own because functions in C library didn't support something you needed?

## Describe how to multiply two 256-bit numbers using any 32-bit processor without FPU or special instructions. Two or more methods?

## Cache coherency, virtual memory, paging, MMU, PCIe

## What is TCP/IP

## What is "zero copy" or "zero buffer" concept?
