## Cortex-M Registers Overview

The ARM Cortex-M core provides a set of **registers** used for general-purpose computation, stack management, and program control.

### General-Purpose Registers

- **R0–R12**: 13 general-purpose registers.
  - Used to store temporary data, variables, or function parameters.
  - Often manipulated directly in assembly.

Example:

```armasm
MOV R0, #5    ; Load constant 5 into R0
ADD R1, R0, #3 ; R1 = R0 + 3
```

### Special Registers

- R13 (SP – Stack Pointer)
  Points to the current top of the stack.

Two versions exist:

MSP (Main Stack Pointer): Default after reset.
PSP (Process Stack Pointer): Used by threads in an RTOS.

- R14 (LR – Link Register)
  Holds the return address when a function/subroutine is called (BL instruction).
  Used to return back with BX LR.

- R15 (PC – Program Counter)
  Always points to the current instruction being executed. Incremented automatically as instructions run.

### xPSR (Program Status Register)

Contains flags and system status bits:

N (Negative), Z (Zero), C (Carry), V (Overflow) – condition flags.

T bit: Indicates Thumb mode (Cortex-M always runs in Thumb).

### Register Summary Table

- R0–R12 General-purpose data storage
- R13 Stack Pointer (SP: MSP/PSP)
- R14 Link Register (LR, return address)
- R15 Program Counter (PC)
- xPSR Program Status Register

## Vector Table

The vector table is a lookup table stored at the beginning of memory (usually at address 0x00000000 after reset). It tells the CPU the following:

- Where the initial stack pointer is.
- Where to find the addresses of exception and interrupt handlers.

### Structure of the Vector Table

- Entry 0: Initial value of the Main Stack Pointer (MSP).
- Entry 1: Address of the Reset Handler (first code executed after reset).
- Entry 2+: Addresses of other exception/interrupt handlers.

### Example (Cortex-M3 typical table):

Address Offset Vector Description
0x00 Initial MSP Initial Stack Pointer
0x04 Reset Reset Handler
0x08 NMI Non-Maskable Interrupt
0x0C HardFault Hard Fault Handler
0x10 MemManage Memory Management Fault
0x14 BusFault Bus Fault Handler
0x18 UsageFault Usage Fault Handler
... ... Other interrupts (e.g., SysTick, peripherals)

### Example Vector Table in Assembly

```armasm
.section .isr*vector, "a", %progbits
.word _estack               ; Initial Stack Pointer
.word Reset_Handler         ; Reset Handler
.word NMI_Handler           ; NMI
.word HardFault_Handler     ; Hard Fault
.word MemManage_Handler     ; Memory Management
.word BusFault_Handler      ; Bus Fault
.word UsageFault_Handler    ; Usage Fault
```

### Example Vector Table in C (using weak aliases)

```c
__attribute__((section(".isr_vector")))
void (* const g*pfnVectors[])(void) = {
(void (*)(void))(&_estack), // Initial stack pointer
Reset_Handler,              // Reset
NMI_Handler,                // NMI
HardFault_Handler,          // Hard Fault
MemManage_Handler,          // MemManage
BusFault_Handler,           // BusFault
UsageFault_Handler,         // UsageFault
...
};
```

## Conclusion

For Cortex-M assembly programming, mastering registers and the vector table is non-negotiable. Registers are the core of computation, while the vector table defines where your program starts and how it responds to interrupts.
This foundation is critical for writing bare-metal embedded applications, bootloaders, and even operating system kernels.
