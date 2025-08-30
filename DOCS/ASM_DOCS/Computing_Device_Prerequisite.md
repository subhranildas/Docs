## How Computers Work

Before diving into **assembly programming**, it’s important to understand the basics of how a computer (or computing device) works.
Assembly language interacts directly with the hardware, so having a mental model of what’s happening inside a CPU helps make sense of assembly instructions.

## What is a Computer?

A **computer** is an electronic device that processes information.
At its core, a computer takes **input**, performs **processing**, stores or retrieves **data**, and provides **output**.

General flow:

Input → Processing (CPU) → Output
↕
Storage (Memory)

Unlike humans, a computer follows precise instructions. These instructions are expressed in **machine code** (binary 0s and 1s)
which is what assembly language translates into.

---

## The Major Components of a Computer

- Central Processing Unit (CPU)
  The **brain of the computer** that executes instructions.  
  It is composed of:

  - **Control Unit (CU)**: Directs data flow, fetches instructions from memory, decodes them, and orchestrates execution.
  - **Arithmetic Logic Unit (ALU)**: Performs basic operations (addition, subtraction, comparisons, logical AND/OR).
  - **Registers**: Small, high-speed storage inside the CPU used for immediate calculations and temporary data.

- Memory

  - **RAM (Random Access Memory)**: Temporary, fast memory where instructions and data are stored while the program runs.
  - **Cache**: Extremely fast, small memory inside or close to the CPU to speed up execution.
  - **Registers**: Even faster, but very small (measured in bytes).

- Storage

  Long-term memory where programs and data reside:

  - **Hard Drive (HDD)** or **Solid-State Drive (SSD)**.  
    When you open a program, the CPU copies it from storage → RAM → registers.

- Input/Output Devices

  Ways of interacting with the computer:

  - **Input**: Keyboard, mouse, sensors, network.
  - **Output**: Monitor, speakers, motors, network.

---

## The Instruction Execution Cycle

Every computer follows the **fetch-decode-execute** cycle:

1. **Fetch**: Get the instruction from memory.
2. **Decode**: Figure out what the instruction means.
3. **Execute**: Perform the operation (math, memory access, jump, etc.).
4. **Store**: Save the result in a register or memory.

Example (simplified):

- Instruction in memory: `ADD R1, R2, R3`
- CPU does: `R1 = R2 + R3`

This is the fundamental loop that runs billions of times per second.

---

## How Assembly Fits In

- **Machine Code**: Binary instructions (`10110000 01100001`) — what the CPU truly understands.
- **Assembly Language**: Human-readable mnemonics (`MOV AL, 61h`).
- **High-Level Languages**: Abstracted instructions (`x = x + y;`).

When writing assembly, you are **talking directly to the CPU** in its own language.

---

## RISC vs CISC in Execution

- **RISC (Reduced Instruction Set Computer)**:

  - Few, simple instructions (fast execution).
  - Example: ARM, RISC-V.

- **CISC (Complex Instruction Set Computer)**:
  - Many, complex instructions (can do more per instruction).
  - Example: Intel x86.

## Conclusion

A computer is essentially a machine that executes instructions step by step. The CPU, memory, and I/O devices work together in a tightly coordinated dance.
Understanding this foundation is critical because when you move into assembly programming, you’ll be:

Assembly programming involves Managing registers, memory, and instructions directly.
Controlling exactly how the CPU processes information, Thinking in terms of the CPU’s fetch-decode-execute cycle etc..

Assembly is the closest one can get to the hardware and the above operations, therefore it becomes essential how a computer works on a high level.
